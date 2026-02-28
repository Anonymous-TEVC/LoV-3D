#!/usr/bin/env python3
"""Zero-shot evaluation of M3D-LaMed-Phi-3-4B on LoV-3D singleturn test set.

Simplified text-based prompts (no JSON) for fair zero-shot comparison.
Outputs: diagnosis (CN/MCI/Dementia), brain region status, diagnostic summary.
Metrics: DX accuracy/F1/kappa, region accuracy/F1, BLEU/ROUGE-L.
"""

import json, re, math, time, os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter

# ============================================================================
# Metric helpers
# ============================================================================

def _tokenize_text(text):
    return re.findall(r'\w+', text.lower())


def _compute_bleu(reference_tokens, hypothesis_tokens, max_n=4):
    scores = {}
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens) - n + 1))
        hyp_ngrams = Counter(tuple(hypothesis_tokens[i:i+n]) for i in range(len(hypothesis_tokens) - n + 1))
        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = max(sum(hyp_ngrams.values()), 1)
        scores[f'bleu_{n}'] = clipped / total
    bp = min(1.0, math.exp(1 - len(reference_tokens) / max(len(hypothesis_tokens), 1)))
    for k in scores:
        scores[k] *= bp
    return scores


def _compute_rouge_l(reference_tokens, hypothesis_tokens):
    m, n = len(reference_tokens), len(hypothesis_tokens)
    if m == 0 or n == 0:
        return 0.0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if reference_tokens[i-1] == hypothesis_tokens[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    lcs_len = prev[n]
    prec = lcs_len / n
    rec = lcs_len / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _cohen_weighted_kappa(y_true, y_pred, labels):
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(y_true)
    if k == 0:
        return 0.0
    cm = [[0]*n for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt]][label_to_idx[yp]] += 1
    w = [[abs(i - j) / max(n - 1, 1) for j in range(n)] for i in range(n)]
    row_sums = [sum(cm[i]) for i in range(n)]
    col_sums = [sum(cm[i][j] for i in range(n)) for j in range(n)]
    e = [[row_sums[i] * col_sums[j] / max(k, 1) for j in range(n)] for i in range(n)]
    num = sum(w[i][j] * cm[i][j] for i in range(n) for j in range(n))
    den = sum(w[i][j] * e[i][j] for i in range(n) for j in range(n))
    if den == 0:
        return 1.0
    return round(1 - num / den, 4)


def _precision_recall_f1(tp, fp, fn):
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return round(prec * 100, 1), round(rec * 100, 1), round(f1 * 100, 1)


# ============================================================================
# Prompt construction
# ============================================================================

def build_simplified_prompt_full(original_prompt):
    """Build a simplified prompt keeping full clinical info (for RadFM / longer-context models).

    Replaces the JSON output instructions with simplified text-based instructions.
    Keeps all clinical info (demographics, cognitive scores, genetic risk,
    longitudinal history, irreversibility rules) identical.
    """
    split_marker = re.search(r'\n---\s*\n+\*\*Instructions\*\*', original_prompt)
    if split_marker:
        clinical_info = original_prompt[:split_marker.start()]
    else:
        split_marker = re.search(r'\*\*Instructions\*\*', original_prompt)
        if split_marker:
            clinical_info = original_prompt[:split_marker.start()]
        else:
            clinical_info = original_prompt

    new_instructions = """

---

**Task**: Based on the 3D brain MRI and the clinical information above, provide the following:

**Diagnosis**: What is the most likely clinical diagnosis? Answer with exactly one of: CN (Cognitively Normal), MCI (Mild Cognitive Impairment), or Dementia.

**Brain Region Assessment**: Classify each brain region's structural status. For each region, state one label:
- Hippocampus: [normal / mild_atrophy / severe_atrophy]
- Entorhinal: [normal / mild_atrophy / severe_atrophy]
- Fusiform: [normal / mild_atrophy / severe_atrophy]
- Midtemporal: [normal / mild_atrophy / severe_atrophy]
- Ventricles: [normal / mild_enlargement / severe_enlargement]

[Diagnostic Summary]
Write 3-5 sentences summarizing key MRI findings, clinical correlation, and diagnostic impression."""

    return clinical_info + new_instructions


def build_short_prompt(original_prompt):
    """Build a concise prompt for M3D-LaMed (max ~200 text tokens to stay within 512 total).

    M3D was trained on short VQA prompts (max 512 tokens including 256 image tokens).
    We extract key clinical info and use a compact task description.
    """
    # Extract key clinical info from original prompt
    age_m = re.search(r'\*\*Age\*\*:\s*([\d.]+)', original_prompt)
    sex_m = re.search(r'\*\*Sex\*\*:\s*(\w+)', original_prompt)
    mmse_m = re.search(r'\*\*MMSE\*\*:\s*([\d.]+)', original_prompt)
    cdr_m = re.search(r'\*\*CDR Sum of Boxes\*\*:\s*([\d.]+)', original_prompt)
    apoe_m = re.search(r'\*\*APOE.*?\*\*:\s*(\d)', original_prompt)

    # Build compact clinical context
    info_parts = []
    if age_m:
        info_parts.append(f"Age {age_m.group(1)}y")
    if sex_m:
        info_parts.append(sex_m.group(1))
    if mmse_m:
        info_parts.append(f"MMSE {mmse_m.group(1)}/30")
    if cdr_m:
        info_parts.append(f"CDR-SB {cdr_m.group(1)}")
    if apoe_m:
        info_parts.append(f"APOE4={apoe_m.group(1)}")
    clinical_line = ", ".join(info_parts) if info_parts else ""

    prompt = f"""Patient: {clinical_line}.
Based on this brain MRI, provide:
1. Diagnosis: CN (Cognitively Normal), MCI (Mild Cognitive Impairment), or Dementia.
2. Brain regions: For Hippocampus, Entorhinal, Fusiform, Midtemporal, Ventricles, state normal, mild_atrophy, or severe_atrophy (mild_enlargement/severe_enlargement for Ventricles).
3. Diagnostic Summary: Write 3-5 sentences summarizing findings and diagnosis."""

    return prompt


# ============================================================================
# Free-text parsing
# ============================================================================

DX_KEYWORDS = {
    'CN': [r'\bCN\b', r'cognitively\s+normal', r'no\s+cognitive\s+impairment',
           r'normal\s+cognition', r'cognitively\s+intact'],
    'MCI': [r'\bMCI\b', r'mild\s+cognitive\s+impairment'],
    'Dementia': [r'\bdementia\b', r'\bAD\b', r"alzheimer", r'\bAlzheimer',
                 r'major\s+neurocognitive\s+disorder'],
}

REGION_NAMES = {
    'hippocampus': [r'hippocampus', r'hippocampal'],
    'entorhinal': [r'entorhinal'],
    'fusiform': [r'fusiform'],
    'midtemporal': [r'midtemporal', r'middle\s+temporal'],
    'ventricles': [r'ventricle', r'ventricular'],
}

REGION_LABEL_KEYWORDS = {
    'severe_atrophy': [r'severe[_\s]?atrophy', r'severe\s+volume\s+loss',
                       r'marked\s+atrophy', r'significant\s+atrophy'],
    'mild_atrophy': [r'mild[_\s]?atrophy', r'mild\s+volume\s+loss',
                     r'slight\s+atrophy', r'minimal\s+atrophy', r'moderate\s+atrophy'],
    'severe_enlargement': [r'severe[_\s]?enlargement', r'marked\s+enlargement',
                           r'significant\s+enlargement'],
    'mild_enlargement': [r'mild[_\s]?enlargement', r'slight\s+enlargement',
                         r'minimal\s+enlargement', r'moderate\s+enlargement'],
    'normal': [r'\bnormal\b', r'within\s+normal\s+limits', r'unremarkable',
               r'no\s+atrophy', r'no\s+enlargement', r'preserved'],
}


def parse_diagnosis(text):
    """Extract diagnosis from free-text output. Returns CN, MCI, or Dementia."""
    text_lower = text.lower()
    # First try: look near "diagnosis" keyword
    dx_section = ''
    m = re.search(r'(?:diagnosis|diagnostic\s+impression)[:\s]*(.{0,200})', text_lower)
    if m:
        dx_section = m.group(1)

    # Score each class
    scores = {}
    for dx, patterns in DX_KEYWORDS.items():
        score = 0
        for pat in patterns:
            # Higher weight for matches near "diagnosis" keyword
            if dx_section and re.search(pat, dx_section, re.IGNORECASE):
                score += 3
            if re.search(pat, text_lower, re.IGNORECASE):
                score += 1
        scores[dx] = score

    if max(scores.values()) == 0:
        return None
    return max(scores, key=scores.get)


def parse_regions(text):
    """Extract region assessments from free-text output."""
    text_lower = text.lower()
    results = {}

    for region, region_patterns in REGION_NAMES.items():
        # Find text near the region name
        for rpat in region_patterns:
            m = re.search(rpat + r'[:\s,]*(.{0,100})', text_lower)
            if m:
                context = m.group(0) + m.group(1)

                # Check labels (order matters: check severe before mild)
                if region == 'ventricles':
                    label_order = ['severe_enlargement', 'mild_enlargement', 'normal']
                else:
                    label_order = ['severe_atrophy', 'mild_atrophy', 'normal']

                found = False
                for label in label_order:
                    for lpat in REGION_LABEL_KEYWORDS[label]:
                        if re.search(lpat, context, re.IGNORECASE):
                            results[region] = label
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

    return results


def extract_summary(text):
    """Extract diagnostic summary from output."""
    # Try [Diagnostic Summary] marker first
    m = re.search(r'\[Diagnostic Summary\]\s*(.+)', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try "Diagnostic Summary" without brackets
    m = re.search(r'Diagnostic Summary[:\s]*\n(.+)', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try "Summary" section
    m = re.search(r'(?:^|\n)\s*(?:Summary|Clinical Summary)[:\s]*\n(.+)', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback: use the last paragraph (after region assessment)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if paragraphs:
        # Use last paragraph that is >= 50 chars (likely a summary)
        for p in reversed(paragraphs):
            if len(p) >= 50:
                return p
    return text.strip()


# ============================================================================
# MRI preprocessing for M3D-LaMed
# ============================================================================

def load_and_resize_mri_m3d(npy_path, target_shape=(32, 256, 256)):
    """Load 128^3 npy, normalize to [0,1], resize to (1, D, H, W)."""
    vol = np.load(npy_path)  # (128, 128, 128), z-score [-3, 3]

    # Normalize to [0, 1]
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin > 1e-8:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = np.zeros_like(vol)

    vol_t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    vol_t = F.interpolate(vol_t, size=target_shape, mode='trilinear', align_corners=False)
    return vol_t.squeeze(0)  # (1, 32, 256, 256)


# ============================================================================
# Main evaluation
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to M3D-LaMed model weights')
    parser.add_argument('--test_json', type=str,
                        default='data_splits/singleturn_test.json')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for eval results')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--proj_out_num', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}, "
              f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Load model ----
    print("Loading M3D-LaMed model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=512,
        padding_side="right",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded: {type(model).__name__}")

    # ---- Load test data ----
    print(f"Loading test data from {args.test_json}")
    with open(args.test_json) as f:
        test_samples = json.load(f)
    print(f"  {len(test_samples)} test samples")

    # ---- Constants ----
    IMAGE_SENTINEL = "<IMAGE>"
    REGIONS_MAP = {
        'Hippocampus': 'hippocampus', 'Entorhinal': 'entorhinal',
        'Fusiform': 'fusiform', 'MidTemp': 'midtemporal', 'Ventricles': 'ventricles'
    }
    REGION_KEYS = list(REGIONS_MAP.values())
    DX_CLASSES = ['CN', 'MCI', 'Dementia']
    REGION_LABELS = ['normal', 'mild_atrophy', 'severe_atrophy']
    REGION_LABELS_VENT = ['normal', 'mild_enlargement', 'severe_enlargement']

    image_tokens = "<im_patch>" * args.proj_out_num

    # ---- Metric counters ----
    dx_correct = dx_total = 0
    dx_per_class = {c: {'correct': 0, 'total': 0} for c in DX_CLASSES}
    baseline_dx_correct = baseline_dx_total = 0
    followup_dx_correct = followup_dx_total = 0
    dx_y_true, dx_y_pred = [], []

    region_correct = region_total = 0
    region_per_name = {r: {'correct': 0, 'total': 0} for r in REGION_KEYS}
    region_y_true_all, region_y_pred_all = [], []
    region_y_per_name = {r: {'y_true': [], 'y_pred': []} for r in REGION_KEYS}

    bleu_scores = {f'bleu_{n}': [] for n in range(1, 5)}
    rouge_l_scores = []

    dx_parsed_count = 0
    region_parsed_count = 0
    summary_count = 0

    per_sample_results = []
    eval_start = time.time()

    # ---- Inference loop ----
    for i, sample in enumerate(test_samples):
        sample_id = sample['sample_id']
        mri_path = sample['mri_path']
        is_baseline = sample.get('is_baseline', True)
        prompt_text = sample['prompt']
        gt_labels = sample.get('gt_labels', {})
        gt_dx = sample.get('gt_diagnosis')

        # Load and resize MRI
        try:
            image_tensor = load_and_resize_mri_m3d(mri_path)
            image_tensor = image_tensor.unsqueeze(0).to(device=device, dtype=dtype)
        except Exception as e:
            print(f"  [{i+1}] SKIP {sample_id}: failed to load MRI: {e}")
            continue

        # Build SHORT prompt for M3D (fits within 512 tokens)
        short_prompt = build_short_prompt(prompt_text)

        # Prepend image tokens
        input_text = image_tokens + " " + short_prompt

        # Tokenize (max_length=512 matches M3D training)
        input_ids = tokenizer(
            input_text, max_length=512, truncation=True, return_tensors="pt"
        )['input_ids'].to(device)

        # Generate
        # Note: with inputs_embeds path, generate() returns ONLY new tokens
        # Use min_new_tokens to force output (model may otherwise produce EOS immediately)
        with torch.no_grad():
            output_ids = model.generate(
                images=image_tensor,
                inputs=input_ids,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=50,
                do_sample=False,
            )
        generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Debug: print first 5 samples
        if i < 5:
            print(f"\n  === Sample {i+1}: {sample_id} (gt_dx={gt_dx}) ===")
            print(f"  Prompt text ({len(short_prompt)} chars): {short_prompt[:200]}...")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  output_ids shape: {output_ids.shape}")
            if output_ids.shape[1] > 0:
                print(f"  output_ids first 20: {output_ids[0][:20].tolist()}")
            raw_text = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
            print(f"  Raw decode ({len(raw_text)} chars): {raw_text[:500]}")
            print(f"  Generated ({len(generated_text)} chars): {generated_text[:500]}")
            print(f"  ===")

        # Parse free-text output
        pred_dx = parse_diagnosis(generated_text)
        pred_regions = parse_regions(generated_text)
        pred_summary = extract_summary(generated_text)

        sample_record = {
            'sample_id': sample_id,
            'is_baseline': is_baseline,
            'generated_text': generated_text,
            'pred_diagnosis': pred_dx,
            'pred_regions': pred_regions,
            'pred_summary': pred_summary,
        }

        # ==== Diagnosis scoring ====
        if pred_dx is not None:
            dx_parsed_count += 1
        if pred_dx and gt_dx:
            dx_total += 1
            dx_y_true.append(gt_dx)
            dx_y_pred.append(pred_dx)
            dx_per_class[gt_dx]['total'] += 1

            is_correct = (pred_dx == gt_dx)
            if is_correct:
                dx_correct += 1
                dx_per_class[gt_dx]['correct'] += 1

            if is_baseline:
                baseline_dx_total += 1
                if is_correct: baseline_dx_correct += 1
            else:
                followup_dx_total += 1
                if is_correct: followup_dx_correct += 1

            sample_record['dx_correct'] = is_correct

        # ==== Region scoring ====
        if pred_regions:
            region_parsed_count += 1
        for csv_region, json_key in REGIONS_MAP.items():
            gt_label = gt_labels.get(csv_region)
            pred_label = pred_regions.get(json_key)
            if pred_label and gt_label:
                # Normalize gt_label format
                gt_label_norm = gt_label.lower().replace(' ', '_')
                pred_label_norm = pred_label.lower().replace(' ', '_')

                region_total += 1
                region_per_name[json_key]['total'] += 1
                region_y_true_all.append(gt_label_norm)
                region_y_pred_all.append(pred_label_norm)
                region_y_per_name[json_key]['y_true'].append(gt_label_norm)
                region_y_per_name[json_key]['y_pred'].append(pred_label_norm)

                r_correct = (pred_label_norm == gt_label_norm)
                if r_correct:
                    region_correct += 1
                    region_per_name[json_key]['correct'] += 1

        # ==== NLG metrics (summary) ====
        gt_response = sample.get('response', '')
        gt_summary_match = re.search(r'\[Diagnostic Summary\]\s*(.+)', gt_response, re.DOTALL)
        if gt_summary_match and pred_summary and len(pred_summary) > 10:
            summary_count += 1
            gt_tokens = _tokenize_text(gt_summary_match.group(1).strip())
            pred_tokens = _tokenize_text(pred_summary)
            if gt_tokens and pred_tokens:
                bleu = _compute_bleu(gt_tokens, pred_tokens)
                for k, v in bleu.items():
                    bleu_scores[k].append(v)
                rouge_l_scores.append(_compute_rouge_l(gt_tokens, pred_tokens))

        per_sample_results.append(sample_record)

        if (i + 1) % 20 == 0 or (i + 1) == len(test_samples):
            elapsed = time.time() - eval_start
            rate = elapsed / (i + 1)
            eta = rate * (len(test_samples) - i - 1)
            print(f"  [{i+1}/{len(test_samples)}] "
                  f"DX:{dx_correct}/{dx_total} "
                  f"Reg:{region_correct}/{region_total} "
                  f"Sum:{summary_count} "
                  f"ETA:{eta/60:.1f}min", flush=True)

    # =====================================================================
    # Compute aggregate metrics
    # =====================================================================
    total = len(test_samples)
    safe_div = lambda a, b: round(100 * a / b, 1) if b > 0 else 0.0

    # DX per-class P/R/F1
    dx_prf = {}
    for c in DX_CLASSES:
        tp = sum(1 for yt, yp in zip(dx_y_true, dx_y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(dx_y_true, dx_y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(dx_y_true, dx_y_pred) if yt == c and yp != c)
        p, r, f = _precision_recall_f1(tp, fp, fn)
        tn = sum(1 for yt, yp in zip(dx_y_true, dx_y_pred) if yt != c and yp != c)
        spec = safe_div(tn, tn + fp)
        dx_prf[c] = {'precision': p, 'recall': r, 'f1': f,
                      'sensitivity': r, 'specificity': spec}

    macro_f1 = round(sum(dx_prf[c]['f1'] for c in DX_CLASSES) / 3, 1) if dx_total > 0 else 0.0
    weighted_f1_num = sum(dx_prf[c]['f1'] * dx_per_class[c]['total'] for c in DX_CLASSES)
    weighted_f1 = round(weighted_f1_num / max(dx_total, 1), 1)

    dx_cm = {gt: {pred: 0 for pred in DX_CLASSES} for gt in DX_CLASSES}
    for yt, yp in zip(dx_y_true, dx_y_pred):
        if yt in dx_cm and yp in DX_CLASSES:
            dx_cm[yt][yp] += 1

    dx_kappa = _cohen_weighted_kappa(dx_y_true, dx_y_pred, DX_CLASSES)

    # Region per-name F1 and kappa
    region_kappa_per_name = {}
    region_f1_per_name = {}
    for rk in REGION_KEYS:
        yt_list = region_y_per_name[rk]['y_true']
        yp_list = region_y_per_name[rk]['y_pred']
        labels = REGION_LABELS_VENT if rk == 'ventricles' else REGION_LABELS
        region_kappa_per_name[rk] = _cohen_weighted_kappa(yt_list, yp_list, labels)
        f1_sum = 0
        n_labels = 0
        for lab in labels:
            tp = sum(1 for y, p in zip(yt_list, yp_list) if y == lab and p == lab)
            fp = sum(1 for y, p in zip(yt_list, yp_list) if y != lab and p == lab)
            fn = sum(1 for y, p in zip(yt_list, yp_list) if y == lab and p != lab)
            _, _, f1 = _precision_recall_f1(tp, fp, fn)
            if sum(1 for y in yt_list if y == lab) > 0:
                f1_sum += f1
                n_labels += 1
        region_f1_per_name[rk] = round(f1_sum / max(n_labels, 1), 1)

    region_macro_f1 = round(sum(region_f1_per_name.values()) / max(len(REGION_KEYS), 1), 1)

    avg_bleu = {k: round(sum(v) / max(len(v), 1), 4) for k, v in bleu_scores.items()}
    avg_rouge_l = round(sum(rouge_l_scores) / max(len(rouge_l_scores), 1), 4)

    # =====================================================================
    # Print report
    # =====================================================================
    elapsed_min = (time.time() - eval_start) / 60
    print(f"\n{'='*70}")
    print(f"  M3D-LaMed ZERO-SHOT EVAL RESULTS (simplified prompts)")
    print(f"  Samples: {total}  |  Time: {elapsed_min:.1f} min")
    print(f"{'='*70}")

    print(f"\n--- A. Parsing Rates ---")
    print(f"  DX parsed:     {safe_div(dx_parsed_count, total)}% ({dx_parsed_count}/{total})")
    print(f"  Regions parsed: {safe_div(region_parsed_count, total)}% ({region_parsed_count}/{total})")
    print(f"  Summary found:  {safe_div(summary_count, total)}% ({summary_count}/{total})")

    print(f"\n--- B. Diagnosis Classification ---")
    print(f"  Overall Accuracy: {safe_div(dx_correct, dx_total)}% ({dx_correct}/{dx_total})")
    print(f"  Baseline DX:      {safe_div(baseline_dx_correct, baseline_dx_total)}%")
    print(f"  Follow-up DX:     {safe_div(followup_dx_correct, followup_dx_total)}%")
    print(f"  Macro-F1:         {macro_f1}%")
    print(f"  Weighted-F1:      {weighted_f1}%")
    print(f"  Cohen's Kappa:    {dx_kappa}")
    print(f"  Per-class:")
    for c in DX_CLASSES:
        v = dx_per_class[c]
        prf = dx_prf[c]
        print(f"    {c:10s}: Acc={safe_div(v['correct'], v['total']):5.1f}% "
              f"P={prf['precision']:.1f} R={prf['recall']:.1f} F1={prf['f1']:.1f} "
              f"Sens={prf['sensitivity']:.1f} Spec={prf['specificity']:.1f}")
    print(f"  Confusion Matrix (rows=GT, cols=Pred):")
    print(f"    {'':10s} {'CN':>6s} {'MCI':>6s} {'Dem':>6s}")
    for gt_c in DX_CLASSES:
        row = [str(dx_cm[gt_c][pc]) for pc in DX_CLASSES]
        print(f"    {gt_c:10s} {row[0]:>6s} {row[1]:>6s} {row[2]:>6s}")

    print(f"\n--- C. Region Assessment ---")
    print(f"  Overall Accuracy: {safe_div(region_correct, region_total)}% ({region_correct}/{region_total})")
    print(f"  Macro Region F1:  {region_macro_f1}%")
    print(f"  Per-region:")
    for r in REGION_KEYS:
        v = region_per_name[r]
        print(f"    {r:14s}: Acc={safe_div(v['correct'], v['total']):5.1f}%  "
              f"F1={region_f1_per_name[r]:.1f}%  Kappa={region_kappa_per_name[r]}")

    print(f"\n--- D. Diagnostic Summary (NLG) ---")
    for k, v in avg_bleu.items():
        print(f"  {k.upper()}: {v:.4f}")
    print(f"  ROUGE-L: {avg_rouge_l:.4f}")
    print(f"  (computed on {len(rouge_l_scores)} samples with both GT and pred summaries)")

    print(f"\n{'='*70}")
    print(f"  SUMMARY: DX {safe_div(dx_correct, dx_total)}% (F1={macro_f1}% k={dx_kappa}) | "
          f"Reg {safe_div(region_correct, region_total)}% (F1={region_macro_f1}%) | "
          f"ROUGE-L {avg_rouge_l:.3f}")
    print(f"{'='*70}")

    # =====================================================================
    # Save results
    # =====================================================================
    metrics = {
        'model': 'M3D-LaMed-Phi-3-4B',
        'eval_type': 'zero_shot_singleturn_simplified',
        'num_eval': total,
        'parsing': {
            'dx_parsed_rate': safe_div(dx_parsed_count, total),
            'region_parsed_rate': safe_div(region_parsed_count, total),
            'summary_found_rate': safe_div(summary_count, total),
        },
        'dx_accuracy': safe_div(dx_correct, dx_total),
        'dx_correct': dx_correct, 'dx_total': dx_total,
        'dx_macro_f1': macro_f1,
        'dx_weighted_f1': weighted_f1,
        'dx_cohen_kappa': dx_kappa,
        'dx_per_class': {c: {
            'accuracy': safe_div(v['correct'], v['total']),
            'correct': v['correct'], 'total': v['total'],
            **dx_prf[c],
        } for c, v in dx_per_class.items()},
        'dx_confusion_matrix': dx_cm,
        'baseline_dx': safe_div(baseline_dx_correct, baseline_dx_total),
        'followup_dx': safe_div(followup_dx_correct, followup_dx_total),
        'region_accuracy': safe_div(region_correct, region_total),
        'region_correct': region_correct, 'region_total': region_total,
        'region_macro_f1': region_macro_f1,
        'region_per_name': {r: {
            'accuracy': safe_div(v['correct'], v['total']),
            'correct': v['correct'], 'total': v['total'],
            'f1': region_f1_per_name[r],
            'kappa': region_kappa_per_name[r],
        } for r, v in region_per_name.items()},
        'nlg': {
            **avg_bleu,
            'rouge_l': avg_rouge_l,
            'num_samples_with_summary': len(rouge_l_scores),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'full_eval_singleturn.json'
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved eval to {out_path}")

    per_sample_path = output_dir / 'per_sample_results.json'
    with open(per_sample_path, 'w') as f:
        json.dump(per_sample_results, f, indent=2, ensure_ascii=False)
    print(f"Saved per-sample results to {per_sample_path}")


if __name__ == '__main__':
    main()
