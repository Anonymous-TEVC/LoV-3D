"""
Stage 3: SFT Training.

Initializes from Stage 2 DPO checkpoint. Fine-tunes projector + LoRA.

Usage:
    python src/train/train_stage3.py \
        --train_json data_splits/stage1_train.json \
        --val_json data_splits/stage1_val.json \
        --test_json data_splits/stage1_test.json \
        --encoder_path /mnt/.../encoder_layer3.pt \
        --llm_path /mnt/.../qwen2_5_14B \
        --projector_path /mnt/.../stage2/best/projector.pt \
        --lora_path /mnt/.../stage2/best/lora_adapter \
        --checkpoint_dir /mnt/.../stage3
"""

import os
import sys
import json
import re
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

from peft import get_peft_model, LoraConfig, PeftModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.stage3_dataset import Stage3Dataset
from src.data.stage1_dataset import Stage1Dataset, collate_fn, IMAGE_SENTINEL, NUM_VISUAL_TOKENS
from src.model.stage1_model import LoV3DStage1Model


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine decay with linear warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_json_from_response(text):
    """Extract JSON block from model response."""
    m = re.search(r'```json\s*(.*?)```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def evaluate_sft(model, val_loader, device, max_batches=None):
    """Evaluate SFT loss on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            with autocast('cuda', dtype=torch.bfloat16):
                loss = model(
                    volume=batch['volume'].to(device),
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device),
                    image_token_mask=batch['image_token_mask'].to(device),
                )
            n_tokens = (batch['labels'] != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))
    model.train()
    return avg_loss, perplexity


def _extract_prior_info(prompt_text):
    """Extract prior DX and prior region labels from prompt text."""
    prior_dx = None
    for line in prompt_text.split('\n'):
        if 'Previous Diagnosis' in line or 'Prior Diagnosis' in line or 'Prior Visit Diagnosis' in line:
            for dx in ['CN', 'MCI', 'Dementia']:
                if dx in line:
                    prior_dx = dx
                    break
            if prior_dx:
                break

    prior_regions = {}
    region_prompt_map = {
        'hippocampus': 'Hippocampus',
        'entorhinal': 'Entorhinal Cortex',
        'fusiform': 'Fusiform Gyrus',
        'midtemporal': 'Middle Temporal Gyrus',
        'ventricles': 'Lateral Ventricles',
    }
    for json_key, prompt_name in region_prompt_map.items():
        # New table format: | region_json | label |
        m = re.search(rf'\| {re.escape(json_key)} \| (\w+) \|', prompt_text)
        if not m:
            # Fallback for old bullet format: **Display Name**: label
            m = re.search(rf'\*\*{re.escape(prompt_name)}\*\*:\s*(\w+)', prompt_text)
        if m:
            prior_regions[json_key] = m.group(1)

    return prior_dx, prior_regions


def evaluate_generation(model, tokenizer, test_samples, dataset, device,
                        num_eval=100, max_new_tokens=700):
    """Comprehensive generation-based evaluation with all metrics."""
    model.eval()
    gc_was_enabled = getattr(model.llm, 'is_gradient_checkpointing', False)
    if gc_was_enabled:
        model.llm.gradient_checkpointing_disable()

    REGIONS_MAP = {
        'Hippocampus': 'hippocampus', 'Entorhinal': 'entorhinal',
        'Fusiform': 'fusiform', 'MidTemp': 'midtemporal', 'Ventricles': 'ventricles'
    }
    REGION_KEYS = list(REGIONS_MAP.values())

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(test_samples))[:num_eval]
    samples = [test_samples[i] for i in indices]

    # Counters
    json_valid = 0
    has_impression = 0
    dx_correct = 0
    dx_total = 0
    dx_per_class = {c: {'correct': 0, 'total': 0} for c in ['CN', 'MCI', 'Dementia']}
    region_correct = 0
    region_total = 0
    region_per_name = {r: {'correct': 0, 'total': 0} for r in REGION_KEYS}
    # DX changed vs unchanged
    dx_changed_correct = 0
    dx_changed_total = 0
    dx_unchanged_correct = 0
    dx_unchanged_total = 0
    # Region changed vs unchanged
    region_changed_correct = 0
    region_changed_total = 0
    region_unchanged_correct = 0
    region_unchanged_total = 0
    longi_dir_correct = 0
    longi_dir_total = 0
    longi_dir_per_region = {r: {'correct': 0, 'total': 0} for r in REGION_KEYS}
    longi_thresh_correct = 0
    longi_thresh_total = 0
    longi_plausible_correct = 0
    longi_plausible_total = 0

    system_msg = (
        "You are a neuroradiologist assessing a patient's brain MRI for signs of "
        "neurodegeneration. Based on the clinical information below and the attached "
        "3D T1-weighted MRI, provide your assessment."
    )
    im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    image_pad_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')

    sample_id_to_idx = {s['sample_id']: j for j, s in enumerate(dataset.samples)}

    eval_start_time = time.time()
    for i, sample in enumerate(samples):
        sample_id = sample['sample_id']
        dataset_idx = sample_id_to_idx.get(sample_id)
        if dataset_idx is None:
            continue

        data_item = dataset[dataset_idx]
        volume = data_item['volume'].unsqueeze(0).to(device)

        prompt_text = sample['prompt']
        before_img, after_img = prompt_text.split(IMAGE_SENTINEL, 1)

        system_tokens = (
            [im_start_id]
            + tokenizer.encode('system\n' + system_msg, add_special_tokens=False)
            + [im_end_id]
            + tokenizer.encode('\n', add_special_tokens=False)
        )
        user_start = [im_start_id] + tokenizer.encode('user\n', add_special_tokens=False)
        before_tokens = tokenizer.encode(before_img.rstrip(), add_special_tokens=False) if before_img.strip() else []
        vision_tokens = [vision_start_id] + [image_pad_id] * NUM_VISUAL_TOKENS + [vision_end_id]
        after_tokens = tokenizer.encode(after_img, add_special_tokens=False)
        user_end = [im_end_id] + tokenizer.encode('\n', add_special_tokens=False)
        assistant_start = [im_start_id] + tokenizer.encode('assistant\n', add_special_tokens=False)

        prompt_tokens = system_tokens + user_start + before_tokens + vision_tokens + after_tokens + user_end + assistant_start
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        image_token_mask = (input_ids == image_pad_id)

        with torch.no_grad():
            with autocast('cuda', dtype=torch.bfloat16):
                output_ids = model.generate(
                    volume, input_ids, attention_mask, image_token_mask,
                    max_new_tokens=max_new_tokens
                )

        generated_text = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)
        parsed_json = parse_json_from_response(generated_text)

        gt_json_match = re.search(r'```json\s*(.*?)```', sample['response'], re.DOTALL)
        gt_json = json.loads(gt_json_match.group(1)) if gt_json_match else None
        gt_dx = sample.get('gt_diagnosis', gt_json['diagnosis']['label'] if gt_json else None)

        # Extract prior info for changed/unchanged tracking
        prior_dx, prior_regions = _extract_prior_info(prompt_text)

        if parsed_json is not None:
            json_valid += 1

            pred_dx = parsed_json.get('diagnosis', {}).get('label')
            if pred_dx and gt_dx:
                dx_total += 1
                dx_per_class[gt_dx]['total'] += 1
                if pred_dx == gt_dx:
                    dx_correct += 1
                    dx_per_class[gt_dx]['correct'] += 1
                # DX changed vs unchanged
                if prior_dx:
                    if prior_dx != gt_dx:
                        dx_changed_total += 1
                        if pred_dx == gt_dx:
                            dx_changed_correct += 1
                    else:
                        dx_unchanged_total += 1
                        if pred_dx == gt_dx:
                            dx_unchanged_correct += 1

            if 'anatomical_assessment' in parsed_json and gt_json:
                for csv_region, json_key in REGIONS_MAP.items():
                    pred_label = parsed_json.get('anatomical_assessment', {}).get(json_key, {}).get('label')
                    gt_label = gt_json.get('anatomical_assessment', {}).get(json_key, {}).get('label')
                    if pred_label and gt_label:
                        region_total += 1
                        region_per_name[json_key]['total'] += 1
                        if pred_label == gt_label:
                            region_correct += 1
                            region_per_name[json_key]['correct'] += 1
                        # Region changed vs unchanged
                        prior_r = prior_regions.get(json_key)
                        if prior_r:
                            if prior_r != gt_label:
                                region_changed_total += 1
                                if pred_label == gt_label:
                                    region_changed_correct += 1
                            else:
                                region_unchanged_total += 1
                                if pred_label == gt_label:
                                    region_unchanged_correct += 1

            pred_longi = parsed_json.get('longitudinal_comparison')
            gt_longi = gt_json.get('longitudinal_comparison') if gt_json else None
            if pred_longi and gt_longi:
                for json_key in REGION_KEYS:
                    pred_r = pred_longi.get(json_key, {})
                    gt_r = gt_longi.get(json_key, {})
                    if isinstance(pred_r, dict) and isinstance(gt_r, dict):
                        if 'direction' in pred_r and 'direction' in gt_r:
                            longi_dir_total += 1
                            longi_dir_per_region[json_key]['total'] += 1
                            if pred_r['direction'] == gt_r['direction']:
                                longi_dir_correct += 1
                                longi_dir_per_region[json_key]['correct'] += 1
                        if 'crossed_threshold' in pred_r and 'crossed_threshold' in gt_r:
                            longi_thresh_total += 1
                            if pred_r['crossed_threshold'] == gt_r['crossed_threshold']:
                                longi_thresh_correct += 1
                pred_cc = pred_longi.get('consistency_check', {})
                gt_cc = gt_longi.get('consistency_check', {})
                if isinstance(pred_cc, dict) and isinstance(gt_cc, dict):
                    if 'is_biologically_plausible' in pred_cc and 'is_biologically_plausible' in gt_cc:
                        longi_plausible_total += 1
                        if pred_cc['is_biologically_plausible'] == gt_cc['is_biologically_plausible']:
                            longi_plausible_correct += 1

        if re.search(r'\[Impression\]', generated_text):
            has_impression += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - eval_start_time
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate if rate > 0 else 0
            dir_str = f"Dir:{longi_dir_correct}/{longi_dir_total}" if longi_dir_total > 0 else ""
            dxc_str = f"DX-chg:{dx_changed_correct}/{dx_changed_total}" if dx_changed_total > 0 else ""
            print(f"    Gen eval [{i+1}/{len(samples)}] "
                  f"JSON:{json_valid}/{i+1} DX:{dx_correct}/{dx_total} "
                  f"Reg:{region_correct}/{region_total} {dir_str} {dxc_str} "
                  f"Rate:{rate:.2f}s/sample ETA:{eta/60:.1f}min", flush=True)

    total = len(samples)
    safe_div = lambda a, b: 100 * a / b if b > 0 else 0.0

    results = {
        'num_eval': total,
        'json_valid': safe_div(json_valid, total),
        'impression': safe_div(has_impression, total),
        'dx_accuracy': safe_div(dx_correct, dx_total),
        'dx_correct': dx_correct,
        'dx_total': dx_total,
        'dx_per_class': {c: {'accuracy': safe_div(v['correct'], v['total']),
                              'correct': v['correct'], 'total': v['total']}
                          for c, v in dx_per_class.items()},
        'dx_changed_accuracy': safe_div(dx_changed_correct, dx_changed_total),
        'dx_changed_correct': dx_changed_correct,
        'dx_changed_total': dx_changed_total,
        'dx_unchanged_accuracy': safe_div(dx_unchanged_correct, dx_unchanged_total),
        'dx_unchanged_correct': dx_unchanged_correct,
        'dx_unchanged_total': dx_unchanged_total,
        'region_accuracy': safe_div(region_correct, region_total),
        'region_correct': region_correct,
        'region_total': region_total,
        'region_per_name': {r: {'accuracy': safe_div(v['correct'], v['total']),
                                 'correct': v['correct'], 'total': v['total']}
                             for r, v in region_per_name.items()},
        'region_changed_accuracy': safe_div(region_changed_correct, region_changed_total),
        'region_changed_correct': region_changed_correct,
        'region_changed_total': region_changed_total,
        'region_unchanged_accuracy': safe_div(region_unchanged_correct, region_unchanged_total),
        'region_unchanged_correct': region_unchanged_correct,
        'region_unchanged_total': region_unchanged_total,
        'longi_direction_accuracy': safe_div(longi_dir_correct, longi_dir_total),
        'longi_direction_correct': longi_dir_correct,
        'longi_direction_total': longi_dir_total,
        'longi_direction_per_region': {r: {'accuracy': safe_div(v['correct'], v['total']),
                                            'correct': v['correct'], 'total': v['total']}
                                        for r, v in longi_dir_per_region.items()},
        'longi_threshold_accuracy': safe_div(longi_thresh_correct, longi_thresh_total),
        'longi_threshold_correct': longi_thresh_correct,
        'longi_threshold_total': longi_thresh_total,
        'longi_plausible_accuracy': safe_div(longi_plausible_correct, longi_plausible_total),
        'longi_plausible_correct': longi_plausible_correct,
        'longi_plausible_total': longi_plausible_total,
    }

    if gc_was_enabled:
        model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()
    return results


def print_gen_metrics(m, prefix=""):
    """Print comprehensive generation eval metrics."""
    print(f"{prefix}JSON valid:      {m['json_valid']:.1f}%")
    imp = m.get('impression', m.get('report_complete', 0.0))
    print(f"{prefix}Report/Impr:     {imp:.1f}%")
    print(f"{prefix}DX accuracy:     {m['dx_accuracy']:.1f}% ({m['dx_correct']}/{m['dx_total']})")
    print(f"{prefix}  DX-changed:   {m.get('dx_changed_accuracy', 0):.1f}% ({m.get('dx_changed_correct', 0)}/{m.get('dx_changed_total', 0)})")
    print(f"{prefix}  DX-unchanged: {m.get('dx_unchanged_accuracy', 0):.1f}% ({m.get('dx_unchanged_correct', 0)}/{m.get('dx_unchanged_total', 0)})")
    dx_chg_gap = abs(m.get('dx_changed_accuracy', 0) - m.get('dx_unchanged_accuracy', 0))
    print(f"{prefix}  DX gap:       {dx_chg_gap:.1f}pp")
    if 'dx_per_class' in m:
        for c in ['CN', 'MCI', 'Dementia']:
            d = m['dx_per_class'].get(c, {})
            if d.get('total', 0) > 0:
                print(f"{prefix}  {c}: {d['accuracy']:.1f}% ({d['correct']}/{d['total']})")
    print(f"{prefix}Region accuracy: {m['region_accuracy']:.1f}% ({m['region_correct']}/{m['region_total']})")
    print(f"{prefix}  Reg-changed:   {m.get('region_changed_accuracy', 0):.1f}% ({m.get('region_changed_correct', 0)}/{m.get('region_changed_total', 0)})")
    print(f"{prefix}  Reg-unchanged: {m.get('region_unchanged_accuracy', 0):.1f}% ({m.get('region_unchanged_correct', 0)}/{m.get('region_unchanged_total', 0)})")
    reg_chg_gap = abs(m.get('region_changed_accuracy', 0) - m.get('region_unchanged_accuracy', 0))
    print(f"{prefix}  Reg gap:       {reg_chg_gap:.1f}pp")
    if 'region_per_name' in m:
        for r in ['hippocampus', 'entorhinal', 'fusiform', 'midtemporal', 'ventricles']:
            d = m['region_per_name'].get(r, {})
            if d.get('total', 0) > 0:
                print(f"{prefix}  {r}: {d['accuracy']:.1f}% ({d['correct']}/{d['total']})")
    if m.get('longi_direction_total', 0) > 0:
        print(f"{prefix}Longi direction:  {m['longi_direction_accuracy']:.1f}% ({m['longi_direction_correct']}/{m['longi_direction_total']})")
        if 'longi_direction_per_region' in m:
            for r in ['hippocampus', 'entorhinal', 'fusiform', 'midtemporal', 'ventricles']:
                d = m['longi_direction_per_region'].get(r, {})
                if d.get('total', 0) > 0:
                    print(f"{prefix}  {r}: {d['accuracy']:.1f}% ({d['correct']}/{d['total']})")
    if m.get('longi_threshold_total', 0) > 0:
        print(f"{prefix}Crossed threshold: {m['longi_threshold_accuracy']:.1f}% ({m['longi_threshold_correct']}/{m['longi_threshold_total']})")
    if m.get('longi_plausible_total', 0) > 0:
        print(f"{prefix}Plausibility:    {m['longi_plausible_accuracy']:.1f}% ({m['longi_plausible_correct']}/{m['longi_plausible_total']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=str, default='data_splits/stage1_train.json')
    parser.add_argument('--val_json', type=str, default='data_splits/stage1_val.json')
    parser.add_argument('--test_json', type=str, default='data_splits/stage1_test.json')
    parser.add_argument('--encoder_path', type=str, required=True,
                        help='Path to Stage 0 encoder checkpoint (encoder_layer3.pt)')
    parser.add_argument('--llm_path', type=str, required=True,
                        help='Path to base LLM (e.g. Qwen-2.5-14B)')
    parser.add_argument('--projector_path', type=str, required=True,
                        help='Path to Stage 2 projector checkpoint (projector.pt)')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='Path to Stage 2 LoRA adapter directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Output directory for Stage 3 checkpoints')
    # Training config
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--lr_projector', type=float, default=2e-5,
                        help='Lower LR for projector (already well-tuned)')
    parser.add_argument('--lr_lora', type=float, default=5e-5,
                        help='Lower LR for LoRA (already DPO-tuned)')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--gen_eval_samples', type=int, default=100)
    parser.add_argument('--skip_baseline_eval', action='store_true',
                        help='Skip pre-training baseline generation eval')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {total_memory / 1e9:.1f} GB")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n=== Loading Model ===")
    model = LoV3DStage1Model(args.encoder_path, args.llm_path)

    # Load Stage 2 projector
    proj_state = torch.load(args.projector_path, map_location='cpu', weights_only=True)
    model.projector.load_state_dict(proj_state)
    print(f"Loaded Stage 2 projector from {args.projector_path}")

    # Load Stage 2 LoRA, merge, apply fresh LoRA for Stage 3
    print(f"Loading Stage 2 LoRA from {args.lora_path}")
    model.llm = PeftModel.from_pretrained(model.llm, args.lora_path)
    model.llm = model.llm.merge_and_unload()
    print("  Stage 2 LoRA merged into base")

    print(f"Applying fresh LoRA for Stage 3 (r={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.print_trainable_parameters()

    model.to(device)

    # Gradient checkpointing
    model.llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()

    params = model.get_trainable_params()
    print(f"\nTrainable parameters: {params['total']:,} "
          f"(projector: {params['projector']:,}, llm_lora: {params['llm']:,})")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)

    # Load training dataset
    print("\n=== Loading Stage 3 Dataset ===")
    train_dataset = Stage3Dataset(
        args.train_json, args.llm_path,
        augment=True, max_seq_len=args.max_seq_len,
    )
    print(f"Training samples: {len(train_dataset)}")

    # Validation set
    val_dataset = Stage1Dataset(
        args.val_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )

    # Test set for generation eval
    with open(args.test_json) as f:
        test_samples = json.load(f)
    test_dataset = Stage1Dataset(
        args.test_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len
    )
    print(f"Val: {len(val_dataset)}, Test: {len(test_samples)}, "
          f"Gen eval subset: {args.gen_eval_samples}")

    # Optimizer
    projector_params = [p for p in model.projector.parameters() if p.requires_grad]
    lora_params = [p for p in model.llm.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': projector_params, 'lr': args.lr_projector},
        {'params': lora_params, 'lr': args.lr_lora},
    ], weight_decay=args.weight_decay)

    # Scheduler (computed over all epochs)
    steps_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n=== Training Configuration ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batch size: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"LR projector: {args.lr_projector}, LR LoRA: {args.lr_lora}")

    # Baseline generation eval
    if args.skip_baseline_eval:
        print("\n=== Skipping Pre-training Baseline Eval (--skip_baseline_eval) ===")
    else:
        print("\n=== Pre-training Generation Eval (baseline) ===")
        gen_metrics = evaluate_generation(
            model, tokenizer, test_samples, test_dataset, device,
            num_eval=args.gen_eval_samples,
        )
        print_gen_metrics(gen_metrics, prefix="  ")

    # Training loop
    print("\n=== Starting Stage 3 Training ===\n")
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Create dataloader (reshuffle each epoch)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
        )

        model.train()
        model.encoder.eval()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_steps = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            with autocast('cuda', dtype=torch.bfloat16):
                loss = model(
                    volume=batch['volume'].to(device),
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device),
                    image_token_mask=batch['image_token_mask'].to(device),
                )
                loss = loss / args.grad_accum

            loss.backward()

            n_tokens = (batch['labels'] != -100).sum().item()
            epoch_loss += loss.item() * args.grad_accum * n_tokens
            epoch_tokens += n_tokens
            epoch_steps += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_interval == 0:
                    avg_loss = epoch_loss / max(epoch_tokens, 1)
                    ppl = math.exp(min(avg_loss, 20))
                    lr_proj = optimizer.param_groups[0]['lr']
                    lr_lora = optimizer.param_groups[1]['lr']
                    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  Step {global_step}/{total_steps} | "
                          f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                          f"LR proj: {lr_proj:.2e} lora: {lr_lora:.2e} | "
                          f"GPU: {gpu_mem:.1f}GB")

                if global_step % args.eval_interval == 0:
                    val_loss, val_ppl = evaluate_sft(model, val_loader, device)
                    print(f"  >> Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

                    gen_metrics = evaluate_generation(
                        model, tokenizer, test_samples, test_dataset, device,
                        num_eval=args.gen_eval_samples,
                    )
                    print(f"  >> Gen Eval ({gen_metrics['num_eval']} samples):")
                    print_gen_metrics(gen_metrics, prefix="     ")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_dir = ckpt_dir / 'best'
                        save_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(model.projector.state_dict(), save_dir / 'projector.pt')
                        model.llm.save_pretrained(str(save_dir / 'lora_adapter'))
                        torch.save({
                            'epoch': epoch, 'global_step': global_step,
                            'val_loss': val_loss,
                            'gen_metrics': gen_metrics,
                        }, save_dir / 'training_state.pt')
                        print(f"  >> New best! Saved to {save_dir} (val_loss={val_loss:.4f})")

        # Flush remaining accumulated gradients
        if (batch_idx + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # Epoch end
        avg_loss = epoch_loss / max(epoch_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs} complete:")
        print(f"  Train Loss: {avg_loss:.4f} | Train PPL: {ppl:.2f}")

        val_loss, val_ppl = evaluate_sft(model, val_loader, device)
        print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

        gen_metrics = evaluate_generation(
            model, tokenizer, test_samples, test_dataset, device,
            num_eval=args.gen_eval_samples,
        )
        print(f"  Gen Eval ({gen_metrics['num_eval']} samples):")
        print_gen_metrics(gen_metrics, prefix="    ")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = ckpt_dir / 'best'
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.projector.state_dict(), save_dir / 'projector.pt')
            model.llm.save_pretrained(str(save_dir / 'lora_adapter'))
            torch.save({
                'epoch': epoch, 'global_step': global_step,
                'val_loss': val_loss,
                'gen_metrics': gen_metrics,
            }, save_dir / 'training_state.pt')
            print(f"  >> New best! Saved to {save_dir} (val_loss={val_loss:.4f})")

        epoch_dir = ckpt_dir / f'epoch_{epoch+1}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.projector.state_dict(), epoch_dir / 'projector.pt')
        model.llm.save_pretrained(str(epoch_dir / 'lora_adapter'))
        torch.save({
            'epoch': epoch, 'global_step': global_step,
            'val_loss': val_loss,
            'gen_metrics': gen_metrics,
        }, epoch_dir / 'training_state.pt')
        print(f"  Saved epoch checkpoint to {epoch_dir}")
        print(f"{'='*60}\n")

    print(f"\n=== Training Complete ===")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Total steps: {global_step}")


if __name__ == '__main__':
    main()
