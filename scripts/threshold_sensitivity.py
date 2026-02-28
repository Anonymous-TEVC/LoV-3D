#!/usr/bin/env python3
"""
Threshold Sensitivity Analysis.

Holds model predictions fixed (from DPO eval JSON), re-derives GT labels
under 5 threshold configurations, and computes region accuracy for each.
"""

import json
import csv
import re
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_JSON = None
DATASET_CSV = "data_splits/dataset_with_labels.csv"

REGIONS_ATROPHY = ['hippocampus', 'entorhinal', 'fusiform', 'midtemporal']
REGION_EXPANSION = ['ventricles']
ALL_REGIONS = REGIONS_ATROPHY + REGION_EXPANSION

# CSV column name -> JSON region name
CSV_TO_JSON = {
    'Hippocampus': 'hippocampus',
    'Entorhinal': 'entorhinal',
    'Fusiform': 'fusiform',
    'MidTemp': 'midtemporal',
    'Ventricles': 'ventricles',
}

# 5 threshold configurations: (mild_cutoff, severe_cutoff) for atrophy
# For ventricles (expansion), thresholds are mirrored (positive)
CONFIGS = {
    'A (-0.3/-1.0)': (-0.3, -1.0),
    'B (-0.3/-1.5)': (-0.3, -1.5),
    'C (-0.5/-1.5) [default]': (-0.5, -1.5),
    'D (-0.7/-1.5)': (-0.7, -1.5),
    'E (-0.7/-2.0)': (-0.7, -2.0),
}


def classify_zscore(z, mild_thresh, severe_thresh, is_expansion=False):
    """Classify Z-score into severity label."""
    if is_expansion:
        # Ventricles: positive Z = enlargement
        if z >= -severe_thresh:  # e.g., z >= 1.5
            return 'severe'
        elif z >= -mild_thresh:  # e.g., z >= 0.5
            return 'mild'
        else:
            return 'normal'
    else:
        # Atrophy regions: negative Z = atrophy
        if z <= severe_thresh:  # e.g., z <= -1.5
            return 'severe'
        elif z <= mild_thresh:  # e.g., z <= -0.5
            return 'mild'
        else:
            return 'normal'


def normalize_label(label):
    """Normalize predicted labels to normal/mild/severe."""
    if label is None:
        return None
    label = label.lower().strip()
    if 'severe' in label:
        return 'severe'
    elif 'mild' in label:
        return 'mild'
    elif 'normal' in label:
        return 'normal'
    return None


def extract_predictions(generated_text):
    """Extract region predictions from generated text."""
    # Find JSON block in generated text
    try:
        # Try to find JSON between ```json and ```
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if not json_match:
            return {}
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return {}

    preds = {}
    anat = parsed.get('anatomical_assessment', {})
    for region in ALL_REGIONS:
        region_data = anat.get(region, {})
        if isinstance(region_data, dict):
            label = region_data.get('label', None)
            preds[region] = normalize_label(label)
        elif isinstance(region_data, str):
            preds[region] = normalize_label(region_data)
    return preds


def main():
    global EVAL_JSON, DATASET_CSV
    parser = argparse.ArgumentParser(description='Threshold sensitivity analysis')
    parser.add_argument('--eval_json', type=str, required=True,
                        help='Path to full_eval_singleturn.json from DPO eval')
    parser.add_argument('--dataset_csv', type=str, default=DATASET_CSV,
                        help='Path to dataset_with_labels.csv')
    args = parser.parse_args()
    EVAL_JSON = args.eval_json
    DATASET_CSV = args.dataset_csv

    # Load eval JSON
    print(f"Loading eval JSON: {EVAL_JSON}")
    with open(EVAL_JSON) as f:
        eval_data = json.load(f)
    samples = eval_data['samples']
    print(f"  {len(samples)} samples")

    # Load CSV with Z-scores
    print(f"Loading dataset CSV: {DATASET_CSV}")
    zscore_map = {}  # sample_id -> {region: z_score}
    with open(DATASET_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row.get('subject_id', '')
            visit = row.get('visit_date', '')
            if not subj or not visit:
                continue
            sid = f"{subj}_{visit}"
            zscores = {}
            for csv_col, json_name in CSV_TO_JSON.items():
                zcol = f"{csv_col}_zscore"
                if zcol in row and row[zcol]:
                    try:
                        zscores[json_name] = float(row[zcol])
                    except ValueError:
                        pass
            if zscores:
                zscore_map[sid] = zscores
    print(f"  {len(zscore_map)} samples with Z-scores")

    # Extract model predictions (fixed across all configs)
    print("\nExtracting model predictions from generated text...")
    pred_map = {}  # sample_id -> {region: label}
    n_parsed = 0
    for s in samples:
        sid = s['sample_id']
        preds = extract_predictions(s.get('generated_text', ''))
        if preds:
            pred_map[sid] = preds
            n_parsed += 1
    print(f"  Parsed predictions for {n_parsed}/{len(samples)} samples")

    # Find common samples (have both predictions and Z-scores)
    common_ids = set(pred_map.keys()) & set(zscore_map.keys())
    print(f"  Common samples (pred + Z-score): {len(common_ids)}")

    # Run threshold sensitivity
    print(f"\n{'='*70}")
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print(f"{'='*70}")

    results = {}
    for config_name, (mild_t, severe_t) in CONFIGS.items():
        correct = 0
        total = 0
        per_severity = {'normal': [0, 0], 'mild': [0, 0], 'severe': [0, 0]}

        for sid in common_ids:
            zscores = zscore_map[sid]
            preds = pred_map[sid]

            for region in ALL_REGIONS:
                if region not in zscores or region not in preds:
                    continue
                if preds[region] is None:
                    continue

                z = zscores[region]
                is_exp = region in REGION_EXPANSION
                gt_label = classify_zscore(z, mild_t, severe_t, is_expansion=is_exp)
                pred_label = preds[region]

                total += 1
                per_severity[gt_label][1] += 1
                if pred_label == gt_label:
                    correct += 1
                    per_severity[gt_label][0] += 1

        acc = 100 * correct / total if total > 0 else 0
        results[config_name] = {
            'accuracy': acc,
            'correct': correct,
            'total': total,
            'per_severity': per_severity,
        }

        # Severity distribution
        sv_dist = {k: v[1] for k, v in per_severity.items()}
        sv_acc = {k: (100*v[0]/v[1] if v[1] > 0 else 0) for k, v in per_severity.items()}

        is_default = 'default' in config_name
        marker = ' <<<' if is_default else ''
        print(f"\n  {config_name}{marker}")
        print(f"    Region Acc: {acc:.1f}% ({correct}/{total})")
        print(f"    Distribution: N={sv_dist['normal']} M={sv_dist['mild']} S={sv_dist['severe']}")
        print(f"    Per-severity: N={sv_acc['normal']:.1f}% M={sv_acc['mild']:.1f}% S={sv_acc['severe']:.1f}%")

    # Summary
    accs = [r['accuracy'] for r in results.values()]
    best = max(accs)
    worst = min(accs)
    best_name = [k for k, v in results.items() if v['accuracy'] == best][0]

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Range: {worst:.1f}% -- {best:.1f}%  (span: {best-worst:.1f} pp)")
    print(f"  Best config: {best_name} ({best:.1f}%)")
    for config_name, r in results.items():
        gap = best - r['accuracy']
        print(f"    {config_name}: {r['accuracy']:.1f}% (gap from best: {gap:.1f} pp)")

    # Severe class distribution across configs
    print(f"\n  Severe class proportion:")
    for config_name, r in results.items():
        sv = r['per_severity']['severe'][1]
        tot = r['total']
        pct = 100*sv/tot if tot > 0 else 0
        print(f"    {config_name}: {pct:.1f}% ({sv}/{tot})")


if __name__ == '__main__':
    main()
