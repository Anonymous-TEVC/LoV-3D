"""
Re-score DPO candidates with upgraded verifier.
Applies GT fallback when best candidate is too weak or margin too small.
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.verifier import (
    compute_composite_score, parse_json_from_response, REGIONS
)

# ─── Config ────────────────────────────────────────────────────────────────
QUALITY_THRESHOLD = 0.3   # best candidate must exceed this
MARGIN_THRESHOLD = 0.05   # chosen-rejected margin must exceed this
SHARD_DIR = Path('data_splits')
STAGE1_DATA = SHARD_DIR / 'stage1_train.json'
NUM_SHARDS = 4

# Region name mapping: CSV → JSON
CSV_TO_JSON = {
    'Hippocampus': 'hippocampus',
    'Entorhinal': 'entorhinal',
    'Fusiform': 'fusiform',
    'MidTemp': 'midtemporal',
    'Ventricles': 'ventricles',
}


def remap_labels(csv_dict):
    """Remap CSV region names to JSON region names."""
    return {CSV_TO_JSON.get(k, k): v for k, v in csv_dict.items()}


def merge_parts(p1_json, p2_json):
    """Merge Part 1 + Part 2 JSON, folding longitudinal_reasoning into reasoning."""
    merged = dict(p1_json) if p1_json else {}
    if p2_json:
        merged['longitudinal_comparison'] = p2_json.get('longitudinal_comparison')
        lr = p2_json.get('longitudinal_reasoning', {})
        if lr and 'reasoning' in merged:
            merged['reasoning']['longitudinal_synthesis'] = lr.get(
                'longitudinal_synthesis', '')
            if lr.get('progression_cited') is not None:
                merged['reasoning']['progression_cited'] = lr['progression_cited']
            if lr.get('regions_mentioned'):
                existing = set(merged['reasoning'].get('regions_mentioned', []))
                merged['reasoning']['regions_mentioned'] = list(
                    existing | set(lr['regions_mentioned']))
    return merged


def extract_prior_from_gt(gt_json):
    """Extract prior_labels and prior_dx from GT JSON longitudinal_comparison."""
    prior_labels = {}
    prior_dx = None
    lc = gt_json.get('longitudinal_comparison', {})
    if not lc:
        return None, None
    for r in REGIONS:
        r_data = lc.get(r, {})
        if isinstance(r_data, dict) and 'prior_label' in r_data:
            prior_labels[r] = r_data['prior_label']
    dx_chg = lc.get('diagnosis_change', {})
    if isinstance(dx_chg, dict):
        prior_dx = dx_chg.get('prior_diagnosis')
    return prior_labels if prior_labels else None, prior_dx


def score_candidate(merged_json, gt_z_scores, gt_labels, gt_dx,
                    prior_labels, has_longitudinal, prior_dx):
    """Score a single candidate with the upgraded verifier."""
    return compute_composite_score(
        merged_json, gt_z_scores, gt_labels, gt_dx,
        prior_labels=prior_labels,
        has_longitudinal=has_longitudinal,
        prior_dx=prior_dx,
    )


def main():
    # Load stage1 training data for GT info
    print("Loading stage1 training data...")
    with open(STAGE1_DATA) as f:
        stage1_data = json.load(f)

    # Build lookup: sample_id → GT info
    gt_lookup = {}
    for s in stage1_data:
        sid = s['sample_id']
        gt_json = s.get('gt_json', {})
        if isinstance(gt_json, str):
            gt_json = parse_json_from_response(gt_json)
            if gt_json is None:
                gt_json = {}

        prior_labels, prior_dx = extract_prior_from_gt(gt_json)

        gt_lookup[sid] = {
            'gt_z_scores': remap_labels(s['gt_z_scores']),
            'gt_labels': remap_labels(s['gt_labels']),
            'gt_dx': s['gt_diagnosis'],
            'is_baseline': s['is_baseline'],
            'prior_labels': prior_labels,
            'prior_dx': prior_dx,
            'gt_response': s.get('response', ''),
            'gt_response_part2': s.get('response_part2', ''),
        }

    print(f"  GT lookup: {len(gt_lookup)} samples")

    # Stats
    stats = defaultdict(int)
    all_chosen_scores = []
    all_rejected_scores = []
    all_margins = []
    gt_fallback_reasons = defaultdict(int)

    # Process all shards
    for shard_idx in range(NUM_SHARDS):
        shard_path = SHARD_DIR / f'dpo_candidates_shard_{shard_idx}.json'
        print(f"\nProcessing {shard_path.name}...")
        with open(shard_path) as f:
            shard_data = json.load(f)

        for sample in shard_data:
            sid = sample['sample_id']
            stats['total'] += 1

            if sample.get('skipped'):
                stats['skipped'] += 1
                continue

            gt_info = gt_lookup.get(sid)
            if not gt_info:
                stats['no_gt'] += 1
                continue

            is_fu = not gt_info['is_baseline']

            # Parse chosen
            c_p1 = parse_json_from_response(sample['chosen_response'])
            c_p2 = parse_json_from_response(
                sample.get('chosen_response_part2', '')) if is_fu else None
            merged_c = merge_parts(c_p1, c_p2)

            # Parse rejected
            r_p1 = parse_json_from_response(sample['rejected_response'])
            r_p2 = parse_json_from_response(
                sample.get('rejected_response_part2', '')) if is_fu else None
            merged_r = merge_parts(r_p1, r_p2)

            # Score both
            res_c = score_candidate(
                merged_c, gt_info['gt_z_scores'], gt_info['gt_labels'],
                gt_info['gt_dx'], gt_info['prior_labels'], is_fu,
                gt_info['prior_dx'])
            res_r = score_candidate(
                merged_r, gt_info['gt_z_scores'], gt_info['gt_labels'],
                gt_info['gt_dx'], gt_info['prior_labels'], is_fu,
                gt_info['prior_dx'])

            chosen_score = res_c['composite']
            rejected_score = res_r['composite']
            margin = chosen_score - rejected_score

            all_chosen_scores.append(chosen_score)
            all_rejected_scores.append(rejected_score)
            all_margins.append(margin)

            # Check GT fallback conditions
            needs_gt = False
            if chosen_score < QUALITY_THRESHOLD:
                needs_gt = True
                gt_fallback_reasons['low_quality'] += 1
            if margin < MARGIN_THRESHOLD:
                needs_gt = True
                gt_fallback_reasons['low_margin'] += 1

            if needs_gt:
                stats['gt_fallback'] += 1
            else:
                stats['valid_pairs'] += 1

            # Check if margin is negative (chosen worse than rejected after re-score)
            if margin < 0:
                stats['flipped'] += 1

    # ─── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DPO Re-scoring Results")
    print("=" * 60)
    print(f"Total samples:     {stats['total']}")
    print(f"Skipped:           {stats['skipped']}")
    print(f"No GT match:       {stats['no_gt']}")
    print(f"Scored:            {len(all_chosen_scores)}")
    print()

    n = len(all_chosen_scores)
    if n > 0:
        avg_c = sum(all_chosen_scores) / n
        avg_r = sum(all_rejected_scores) / n
        avg_m = sum(all_margins) / n

        print(f"Chosen scores:     mean={avg_c:.4f}, "
              f"min={min(all_chosen_scores):.4f}, "
              f"max={max(all_chosen_scores):.4f}")
        print(f"Rejected scores:   mean={avg_r:.4f}, "
              f"min={min(all_rejected_scores):.4f}, "
              f"max={max(all_rejected_scores):.4f}")
        print(f"Margins:           mean={avg_m:.4f}, "
              f"min={min(all_margins):.4f}, "
              f"max={max(all_margins):.4f}")
        print()

        print(f"Valid pairs:       {stats['valid_pairs']} "
              f"({100*stats['valid_pairs']/n:.1f}%)")
        print(f"GT fallback:       {stats['gt_fallback']} "
              f"({100*stats['gt_fallback']/n:.1f}%)")
        print(f"  - low quality:   {gt_fallback_reasons['low_quality']} "
              f"(chosen < {QUALITY_THRESHOLD})")
        print(f"  - low margin:    {gt_fallback_reasons['low_margin']} "
              f"(margin < {MARGIN_THRESHOLD})")
        print(f"Flipped (c < r):   {stats['flipped']} "
              f"({100*stats['flipped']/n:.1f}%)")

        # Distribution
        print()
        print("Chosen score distribution:")
        brackets = [(-999, -0.5), (-0.5, 0), (0, 0.3), (0.3, 0.5),
                     (0.5, 0.7), (0.7, 0.9), (0.9, 999)]
        labels = ["< -0.5", "-0.5~0", "0~0.3", "0.3~0.5",
                  "0.5~0.7", "0.7~0.9", "> 0.9"]
        for (lo, hi), label in zip(brackets, labels):
            cnt = sum(1 for s in all_chosen_scores if lo <= s < hi)
            bar = "#" * (cnt * 40 // n)
            print(f"  {label:>8s}: {cnt:4d} ({100*cnt/n:5.1f}%) {bar}")

        print()
        print("Margin distribution:")
        brackets_m = [(-999, -0.1), (-0.1, 0), (0, 0.05), (0.05, 0.1),
                       (0.1, 0.2), (0.2, 0.4), (0.4, 999)]
        labels_m = ["< -0.1", "-0.1~0", "0~0.05", "0.05~0.1",
                    "0.1~0.2", "0.2~0.4", "> 0.4"]
        for (lo, hi), label in zip(brackets_m, labels_m):
            cnt = sum(1 for m in all_margins if lo <= m < hi)
            bar = "#" * (cnt * 40 // n)
            print(f"  {label:>8s}: {cnt:4d} ({100*cnt/n:5.1f}%) {bar}")


if __name__ == '__main__':
    main()
