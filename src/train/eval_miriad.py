#!/usr/bin/env python3
"""
MIRIAD External Validation — Single-turn (identical prompt to Stage 1b training).

Evaluates the model on the MIRIAD dataset using the exact same singleturn prompt
format as d00ctrl_1b training. Supports both baseline (visit 1) and follow-up
(visit 2+) using model's own prior predictions as longitudinal context.

Usage:
    python -m src.train.eval_miriad \
        --projector_path /path/to/projector.pt \
        --lora_path /path/to/lora_adapter
"""

import os
import sys
import json
import re
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.amp import autocast
from transformers import AutoTokenizer
from peft import PeftModel

from src.model.stage1_model import LoV3DStage1Model

IMAGE_SENTINEL = "<IMAGE>"
NUM_VISUAL_TOKENS = 512

REGIONS = ['hippocampus', 'entorhinal', 'fusiform', 'midtemporal', 'ventricles']


def parse_json_from_response(text):
    """Extract JSON from model response."""
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(\{.*\})',
    ]
    for p in patterns:
        m = re.search(p, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


# ─────────────────────────────────────────────────────────────
# Prompt builders — IDENTICAL to singleturn training prompts
# ─────────────────────────────────────────────────────────────

def build_baseline_prompt(sample):
    """Build baseline singleturn prompt — identical to training format."""
    age = f"{float(sample['mr_age']):.1f} years" if sample.get('mr_age') else 'Not available'
    sex = 'Male' if sample['gender'] == 'M' else 'Female'
    education = 'Not available'
    apoe = 'Unknown'
    mmse = f"{int(float(sample['mmse']))}/30" if sample.get('mmse') else 'Not available'
    cdrsb = f"{float(sample['cdr_sum']):.1f}" if sample.get('cdr_sum') else 'Not available'

    parts = [IMAGE_SENTINEL, '']
    parts.append('## Patient Demographics')
    parts.append(f'- **Age**: {age}')
    parts.append(f'- **Sex**: {sex}')
    parts.append(f'- **Education**: {education}')
    parts.append('')
    parts.append('## Genetic Risk')
    parts.append(f'- **APOE ε4 alleles**: {apoe}')
    parts.append('')
    parts.append('## Current Cognitive Assessment')
    parts.append(f'- **MMSE**: {mmse}')
    parts.append(f'- **CDR Sum of Boxes**: {cdrsb}')
    parts.append('')
    parts.append('---')
    parts.append('')
    parts.append('**Instructions**: Examine the 3D MRI and integrate all clinical information above.')
    parts.append('')
    parts.append('**Analyze step by step:**')
    parts.append('1. Describe key findings observed in the MRI scan')
    parts.append('2. Connect imaging findings with available cognitive scores')
    parts.append('3. Based on all evidence, determine anatomical status and diagnosis')
    parts.append('')
    parts.append('First, respond with a structured JSON:')
    parts.append('')
    parts.append('```json')
    parts.append('{')
    parts.append('  "reasoning": {')
    parts.append('    "imaging_observations": "<describe key findings from the MRI>",')
    parts.append('    "clinical_integration": "<connect imaging findings with cognitive scores>",')
    parts.append('    "longitudinal_synthesis": "First visit — no prior data for comparison.",')
    parts.append('    "regions_mentioned": ["<list of regions referenced>"],')
    parts.append('    "progression_cited": false')
    parts.append('  },')
    parts.append('  "anatomical_assessment": {')
    parts.append('    "hippocampus": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
    parts.append('    "entorhinal": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
    parts.append('    "fusiform": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
    parts.append('    "midtemporal": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
    parts.append('    "ventricles": {"label": "<normal|mild_enlargement|severe_enlargement>", "confidence": 0.0}')
    parts.append('  },')
    parts.append('  "longitudinal_comparison": null,')
    parts.append('  "diagnosis": {"label": "<CN|MCI|Dementia>", "confidence": 0.0}')
    parts.append('}')
    parts.append('```')
    parts.append('')
    parts.append('After the JSON block, write [Diagnostic Summary] followed by a brief '
                 'clinical report (3–5 sentences) summarizing key findings, changes '
                 'from prior, and diagnostic impression.')

    return '\n'.join(parts)


def build_followup_prompt(sample, prior_labels, prior_dx, prior_mmse, prior_cdrsb, interval_months):
    """Build follow-up singleturn prompt — identical to training format."""
    age = f"{float(sample['mr_age']):.1f} years" if sample.get('mr_age') else 'Not available'
    sex = 'Male' if sample['gender'] == 'M' else 'Female'
    education = 'Not available'
    apoe = 'Unknown'
    mmse = f"{int(float(sample['mmse']))}/30" if sample.get('mmse') else 'Not available'
    cdrsb = f"{float(sample['cdr_sum']):.1f}" if sample.get('cdr_sum') else 'Not available'

    prior_mmse_str = f"{int(float(prior_mmse))}/30" if prior_mmse else 'Not available'
    prior_cdrsb_str = f"{float(prior_cdrsb):.1f}" if prior_cdrsb else 'Not available'

    parts = [IMAGE_SENTINEL, '']
    parts.append('## Patient Demographics')
    parts.append(f'- **Age**: {age}')
    parts.append(f'- **Sex**: {sex}')
    parts.append(f'- **Education**: {education}')
    parts.append('')
    parts.append('## Genetic Risk')
    parts.append(f'- **APOE ε4 alleles**: {apoe}')
    parts.append('')
    parts.append('## Current Cognitive Assessment')
    parts.append(f'- **MMSE**: {mmse}')
    parts.append(f'- **CDR Sum of Boxes**: {cdrsb}')
    parts.append('')
    parts.append('## Longitudinal History')
    parts.append(f'- **Prior Diagnosis**: {prior_dx}')
    parts.append(f'- **Prior MMSE**: {prior_mmse_str}')
    parts.append(f'- **Prior CDR-SB**: {prior_cdrsb_str}')
    parts.append(f'- **Interval**: {interval_months:.1f} months since last visit')
    parts.append('')
    parts.append('## Prior Anatomical Status (from previous MRI assessment)')
    for region in REGIONS:
        parts.append(f'- **{region}**: {prior_labels.get(region, "normal")}')
    parts.append('')
    parts.append('## Clinical Constraints (Irreversibility Rules)')
    parts.append('Neurodegenerative changes are irreversible:')
    parts.append('- If prior diagnosis was Dementia, current diagnosis cannot revert to MCI or CN.')
    parts.append('- If a region was severe_atrophy, it cannot improve to mild_atrophy or normal.')
    parts.append('- If a region was mild_atrophy, it cannot revert to normal.')
    parts.append('- If ventricles were severe_enlargement, they cannot shrink to mild_enlargement or normal.')
    parts.append('- If ventricles were mild_enlargement, they cannot revert to normal.')
    parts.append('')
    parts.append('---')
    parts.append('')
    parts.append('**Instructions**: Examine the 3D MRI and integrate all clinical information above, '
                 'including the prior visit data. Check your assessment against the irreversibility '
                 'constraints and provide a longitudinal comparison.')
    parts.append('')
    parts.append('**Analyze step by step:**')
    parts.append('1. Describe key findings observed in the MRI scan')
    parts.append('2. Connect imaging findings with available cognitive scores')
    parts.append('3. Determine anatomical status and diagnosis')
    parts.append('4. Check assessment against prior labels for irreversibility violations')
    parts.append('5. Compare with prior assessment and determine longitudinal changes')
    parts.append('')
    parts.append('First, respond with a structured JSON:')
    parts.append('')
    parts.append('```json')
    parts.append('{')
    parts.append('  "reasoning": {')
    parts.append('    "imaging_observations": "<describe key findings from the MRI>",')
    parts.append('    "clinical_integration": "<connect imaging findings with cognitive scores>",')
    parts.append('    "longitudinal_synthesis": "<summarize changes from prior study>",')
    parts.append('    "regions_mentioned": ["<list of regions referenced>"],')
    parts.append('    "progression_cited": false')
    parts.append('  },')
    parts.append('  "anatomical_assessment": {')
    parts.append('    "hippocampus": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
    parts.append('    "entorhinal": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
    parts.append('    "fusiform": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
    parts.append('    "midtemporal": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
    parts.append('    "ventricles": {"label": "<normal|mild_enlargement|severe_enlargement>", "confidence": 0.0}')
    parts.append('  },')
    parts.append('  "longitudinal_comparison": {')
    parts.append('    "<region>": {"direction": "<stable|progressive_atrophy|progressive_enlargement>", '
                 '"changed": false, "crossed_threshold": false},')
    parts.append('    "diagnosis_change": {"prior_diagnosis": "<prior_dx>", "changed": false}')
    parts.append('  },')
    parts.append('  "diagnosis": {"label": "<CN|MCI|Dementia>", "confidence": 0.0}')
    parts.append('}')
    parts.append('```')
    parts.append('')
    parts.append('After the JSON block, write [Diagnostic Summary] followed by a brief '
                 'clinical report (3–5 sentences) summarizing key findings, changes '
                 'from prior, and diagnostic impression.')

    return '\n'.join(parts)


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_miriad_data(csv_path, resize_dir):
    """Load miriad_clinical.csv, group by subject, sort by visit."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    subject_visits = defaultdict(list)
    for row in all_rows:
        npy_path = os.path.join(
            resize_dir, row['subject_id'],
            f"{row['subject_id']}_{row['visit_date']}_T1w_final.npy"
        )
        if not os.path.exists(npy_path):
            continue

        subject_visits[row['subject_id']].append({
            'subject_id': row['subject_id'],
            'group': row['group'],
            'gender': row['gender'],
            'base_age': row['base_age'],
            'visit': int(row['visit']),
            'visit_date': row['visit_date'],
            'mr_age': row['mr_age'] or '',
            'mmse': row['mmse'] or '',
            'cdr_sum': row['cdr_sum'] or '',
            'dx': row['dx'],
            'gt_binary': 'AD' if row['group'] == 'AD' else 'HC',
            'npy_path': npy_path,
        })

    # Sort each subject by visit number
    for sid in subject_visits:
        subject_visits[sid].sort(key=lambda x: x['visit'])

    return dict(subject_visits)


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def tokenize_and_generate(model, volume, prompt_text, system_msg, tokenizer,
                          special_ids, device, max_new_tokens=900):
    """Tokenize prompt and generate — identical to train_ablation.py singleturn eval."""
    im_start_id, im_end_id, vision_start_id, vision_end_id, image_pad_id = special_ids

    system_tokens = (
        [im_start_id]
        + tokenizer.encode('system\n' + system_msg, add_special_tokens=False)
        + [im_end_id]
        + tokenizer.encode('\n', add_special_tokens=False)
    )
    user_start = [im_start_id] + tokenizer.encode('user\n', add_special_tokens=False)

    before_img, after_img = prompt_text.split(IMAGE_SENTINEL, 1)
    before_tokens = tokenizer.encode(before_img.rstrip(), add_special_tokens=False) if before_img.strip() else []
    vision_tokens = [vision_start_id] + [image_pad_id] * NUM_VISUAL_TOKENS + [vision_end_id]
    after_tokens = tokenizer.encode(after_img, add_special_tokens=False)

    user_end = [im_end_id] + tokenizer.encode('\n', add_special_tokens=False)
    assistant_start = [im_start_id] + tokenizer.encode('assistant\n', add_special_tokens=False)

    prompt_tokens = (system_tokens + user_start + before_tokens
                     + vision_tokens + after_tokens + user_end + assistant_start)

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    image_token_mask = (input_ids == image_pad_id)

    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            output_ids = model.generate(
                volume, input_ids, attention_mask, image_token_mask,
                max_new_tokens=max_new_tokens
            )

    gen_text = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)
    return gen_text


def extract_dx(parsed):
    """Extract diagnosis label from parsed JSON."""
    if parsed is None:
        return None
    dx_info = parsed.get('diagnosis', {})
    if isinstance(dx_info, dict):
        return dx_info.get('label')
    elif isinstance(dx_info, str):
        return dx_info
    return None


def extract_labels(parsed):
    """Extract anatomical labels from parsed JSON."""
    if parsed is None:
        return {r: 'normal' for r in REGIONS}
    assessment = parsed.get('anatomical_assessment', {})
    labels = {}
    for r in REGIONS:
        pred = assessment.get(r, {})
        if isinstance(pred, dict):
            labels[r] = pred.get('label', 'normal')
        else:
            labels[r] = 'normal'
    return labels


def dx_to_binary(pred_dx):
    """Map model prediction to AD/HC binary."""
    if not pred_dx:
        return 'unknown'
    pred_dx = pred_dx.strip()
    return 'AD' if pred_dx in ['Dementia', 'MCI'] else 'HC'


# ─────────────────────────────────────────────────────────────
# Main eval
# ─────────────────────────────────────────────────────────────

def run_eval(model, subject_visits, tokenizer, system_msg, special_ids, device):
    """Run singleturn eval on all MIRIAD subjects."""
    total_scans = sum(len(v) for v in subject_visits.values())
    n_subjects = len(subject_visits)
    print(f"\n=== MIRIAD Singleturn Eval: {total_scans} scans, {n_subjects} subjects ===\n")

    dx_correct = dx_total = json_valid = 0
    dx_per_class = {'CN': [0, 0], 'Dementia': [0, 0]}
    subject_preds = {}
    per_sample_results = []
    eval_start = time.time()

    scan_idx = 0
    for sid, visits in sorted(subject_visits.items()):
        prior_labels = None
        prior_dx = None
        prior_mmse = None
        prior_cdrsb = None
        prior_age = None

        for visit in visits:
            volume = torch.from_numpy(
                np.load(visit['npy_path']).astype(np.float32)
            ).unsqueeze(0).unsqueeze(0).to(device)

            # Build prompt — baseline for first visit, follow-up for rest
            if prior_labels is None:
                prompt_text = build_baseline_prompt(visit)
                is_baseline = True
            else:
                current_age = float(visit['mr_age']) if visit['mr_age'] else 0
                interval = (current_age - prior_age) * 12 if prior_age and current_age else 12.0
                prompt_text = build_followup_prompt(
                    visit, prior_labels, prior_dx, prior_mmse, prior_cdrsb, interval)
                is_baseline = False

            gen_text = tokenize_and_generate(
                model, volume, prompt_text, system_msg, tokenizer, special_ids, device)
            parsed = parse_json_from_response(gen_text)

            pred_dx = extract_dx(parsed)
            pred_binary = dx_to_binary(pred_dx)
            gt_binary = visit['gt_binary']
            pred_labels = extract_labels(parsed)

            if parsed:
                json_valid += 1
            dx_total += 1
            is_correct = (pred_binary == gt_binary)
            if is_correct:
                dx_correct += 1

            cls_key = 'Dementia' if gt_binary == 'AD' else 'CN'
            dx_per_class[cls_key][1] += 1
            if is_correct:
                dx_per_class[cls_key][0] += 1

            if sid not in subject_preds:
                subject_preds[sid] = []
            subject_preds[sid].append({
                'pred_binary': pred_binary, 'gt_binary': gt_binary})

            per_sample_results.append({
                'subject_id': sid, 'visit': visit['visit'],
                'group': visit['group'], 'gender': visit['gender'],
                'mmse': visit['mmse'], 'cdr_sum': visit['cdr_sum'],
                'mr_age': visit['mr_age'],
                'gt_binary': gt_binary, 'pred_dx': pred_dx, 'pred_binary': pred_binary,
                'correct': is_correct, 'json_valid': parsed is not None,
                'is_baseline': is_baseline,
                'response': gen_text,
            })

            # Update prior info for next visit — use GT diagnosis (not prediction)
            prior_labels = pred_labels
            prior_dx = visit['dx']
            prior_mmse = visit['mmse']
            prior_cdrsb = visit['cdr_sum']
            prior_age = float(visit['mr_age']) if visit['mr_age'] else prior_age

            scan_idx += 1
            if scan_idx % 20 == 0 or scan_idx == total_scans:
                elapsed = time.time() - eval_start
                eta = elapsed / scan_idx * (total_scans - scan_idx) / 60
                acc = dx_correct / dx_total * 100
                print(f"  [{scan_idx}/{total_scans}] "
                      f"JSON:{json_valid}/{scan_idx} "
                      f"DX:{dx_correct}/{dx_total} ({acc:.1f}%) "
                      f"ETA:{eta:.1f}min")

    # ── Per-subject majority voting ──
    subj_correct = subj_total = 0
    subj_ad_correct = subj_ad_total = 0
    subj_hc_correct = subj_hc_total = 0
    for sid, preds in subject_preds.items():
        gt = preds[0]['gt_binary']
        ad_votes = sum(1 for p in preds if p['pred_binary'] == 'AD')
        hc_votes = sum(1 for p in preds if p['pred_binary'] == 'HC')
        majority = 'AD' if ad_votes >= hc_votes else 'HC'
        subj_total += 1
        if majority == gt:
            subj_correct += 1
        if gt == 'AD':
            subj_ad_total += 1
            if majority == 'AD':
                subj_ad_correct += 1
        else:
            subj_hc_total += 1
            if majority == 'HC':
                subj_hc_correct += 1

    # ── Breakdowns ──
    baseline_results = [r for r in per_sample_results if r['is_baseline']]
    followup_results = [r for r in per_sample_results if not r['is_baseline']]
    with_mmse = [r for r in per_sample_results if r['mmse']]
    no_mmse = [r for r in per_sample_results if not r['mmse']]

    baseline_acc = sum(1 for r in baseline_results if r['correct']) / len(baseline_results) * 100 if baseline_results else 0
    followup_acc = sum(1 for r in followup_results if r['correct']) / len(followup_results) * 100 if followup_results else 0
    mmse_acc = sum(1 for r in with_mmse if r['correct']) / len(with_mmse) * 100 if with_mmse else 0
    no_mmse_acc = sum(1 for r in no_mmse if r['correct']) / len(no_mmse) * 100 if no_mmse else 0

    elapsed_total = time.time() - eval_start

    # ── Print ──
    print(f"\n{'='*60}")
    print(f"MIRIAD Singleturn Eval Results")
    print(f"{'='*60}")
    print(f"\n--- Per-Scan Accuracy ---")
    print(f"  Total scans: {dx_total}")
    print(f"  JSON valid:  {json_valid}/{dx_total} ({json_valid/dx_total*100:.1f}%)")
    print(f"  DX accuracy: {dx_correct}/{dx_total} ({dx_correct/dx_total*100:.1f}%)")
    for cls in ['CN', 'Dementia']:
        c, t = dx_per_class[cls]
        if t > 0:
            print(f"    {cls:>10}: {c}/{t} ({c/t*100:.1f}%)")
    print(f"\n--- Baseline vs Follow-up ---")
    print(f"  Baseline:  {len(baseline_results)} scans, acc={baseline_acc:.1f}%")
    print(f"  Follow-up: {len(followup_results)} scans, acc={followup_acc:.1f}%")
    print(f"\n--- MMSE Breakdown ---")
    print(f"  With MMSE:    {len(with_mmse)} scans, acc={mmse_acc:.1f}%")
    print(f"  Without MMSE: {len(no_mmse)} scans, acc={no_mmse_acc:.1f}%")
    print(f"\n--- Per-Subject Majority Vote ---")
    print(f"  Total: {subj_correct}/{subj_total} ({subj_correct/subj_total*100:.1f}%)")
    if subj_ad_total > 0:
        print(f"    AD (sens): {subj_ad_correct}/{subj_ad_total} ({subj_ad_correct/subj_ad_total*100:.1f}%)")
    if subj_hc_total > 0:
        print(f"    HC (spec): {subj_hc_correct}/{subj_hc_total} ({subj_hc_correct/subj_hc_total*100:.1f}%)")
    print(f"\n  Time: {elapsed_total/60:.1f} min")

    return {
        'per_scan': {
            'total': dx_total, 'json_valid': json_valid,
            'dx_correct': dx_correct, 'dx_accuracy': dx_correct / dx_total if dx_total else 0,
            'per_class': {k: {'correct': v[0], 'total': v[1]} for k, v in dx_per_class.items()},
        },
        'baseline_vs_followup': {
            'baseline': {'count': len(baseline_results), 'accuracy': baseline_acc},
            'followup': {'count': len(followup_results), 'accuracy': followup_acc},
        },
        'mmse_breakdown': {
            'with_mmse': {'count': len(with_mmse), 'accuracy': mmse_acc},
            'without_mmse': {'count': len(no_mmse), 'accuracy': no_mmse_acc},
        },
        'per_subject': {
            'total': subj_total, 'correct': subj_correct,
            'accuracy': subj_correct / subj_total if subj_total else 0,
            'ad_sensitivity': subj_ad_correct / subj_ad_total if subj_ad_total else 0,
            'hc_specificity': subj_hc_correct / subj_hc_total if subj_hc_total else 0,
        },
        'per_sample_results': per_sample_results,
        'elapsed_min': elapsed_total / 60,
    }


def main():
    parser = argparse.ArgumentParser(description='MIRIAD External Validation (Singleturn)')
    parser.add_argument('--encoder_path', type=str, required=True,
                        help='Path to Stage 0 encoder checkpoint (encoder_layer3.pt)')
    parser.add_argument('--llm_path', type=str, required=True,
                        help='Path to base LLM (e.g., Qwen2.5-14B)')
    parser.add_argument('--projector_path', type=str, required=True)
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--resize_dir', type=str, required=True,
                        help='Path to MIRIAD resized MRI directory')
    parser.add_argument('--clinical_csv', type=str, required=True,
                        help='Path to MIRIAD clinical CSV (miriad_clinical.csv)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--shard_id', type=int, default=None,
                        help='Shard index (0-based). If set, only process a subset of subjects.')
    parser.add_argument('--num_shards', type=int, default=None,
                        help='Total number of shards.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda')

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load data ──
    subject_visits = load_miriad_data(args.clinical_csv, args.resize_dir)
    total = sum(len(v) for v in subject_visits.values())
    n_ad = sum(1 for vs in subject_visits.values() if vs[0]['gt_binary'] == 'AD')
    n_hc = sum(1 for vs in subject_visits.values() if vs[0]['gt_binary'] == 'HC')
    print(f"\nLoaded {total} scans, {len(subject_visits)} subjects (AD={n_ad}, HC={n_hc})")

    # ── Sharding (split by subjects) ──
    if args.shard_id is not None and args.num_shards is not None:
        all_sids = sorted(subject_visits.keys())
        shard_sids = [s for i, s in enumerate(all_sids) if i % args.num_shards == args.shard_id]
        subject_visits = {s: subject_visits[s] for s in shard_sids}
        shard_scans = sum(len(v) for v in subject_visits.values())
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(shard_sids)} subjects, {shard_scans} scans")

    # ── Load Model ──
    print("\n=== Loading Model ===")
    model = LoV3DStage1Model(args.encoder_path, args.llm_path)

    proj_weights = torch.load(args.projector_path, map_location='cpu', weights_only=True)
    model.projector.load_state_dict(proj_weights)
    print(f"Loaded projector from {args.projector_path}")

    model.llm = PeftModel.from_pretrained(model.llm, args.lora_path)
    model.llm = model.llm.merge_and_unload()
    print(f"Loaded and merged LoRA from {args.lora_path}")

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)

    system_msg = (
        "You are a neuroradiologist assessing a patient's brain MRI for signs of "
        "neurodegeneration. Based on the clinical information below and the attached "
        "3D T1-weighted MRI, provide your assessment."
    )
    special_ids = (
        tokenizer.convert_tokens_to_ids('<|im_start|>'),
        tokenizer.convert_tokens_to_ids('<|im_end|>'),
        tokenizer.convert_tokens_to_ids('<|vision_start|>'),
        tokenizer.convert_tokens_to_ids('<|vision_end|>'),
        tokenizer.convert_tokens_to_ids('<|image_pad|>'),
    )

    # ── Run Eval ──
    results = run_eval(model, subject_visits, tokenizer, system_msg, special_ids, device)

    # ── Save Results ──
    default_dir = 'results/miriad_eval'
    if args.shard_id is not None:
        default_dir = f'results/miriad_eval/shard_{args.shard_id}'
    out_dir = args.output_dir or default_dir
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, 'miriad_eval_results.json')
    results['config'] = vars(args)
    results['gpu_peak_gb'] = torch.cuda.max_memory_allocated() / 1e9

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"GPU peak: {results['gpu_peak_gb']:.2f} GB")


if __name__ == '__main__':
    main()
