#!/usr/bin/env python3
"""
Verifier Weight Sensitivity Analysis (§4.3 / sec:verifier_sensitivity).

Re-scores all 3993×4 DPO candidates under 5 weight configurations:
  0. Default   : anat=0.25, longi=0.20, dx=0.25, reason=0.15, summary=0.15
                  region weights: hip=1.2, ent=1.1, fus=1.0, mid=1.0, ven=1.0
  1. Equal     : all components = 0.20
  2. Anat-heavy: anat=0.40, longi=0.15, dx=0.20, reason=0.15, summary=0.10
  3. DX-heavy  : anat=0.20, longi=0.15, dx=0.40, reason=0.15, summary=0.10
  4. Equal-reg : same component weights as default, all region weights = 1.0

Then compares:
  - % of chosen-rejected pairs identical to default
  - Kendall's τ between candidate ranking vectors

Output: prints results and fills paper placeholders.

Usage:
    python scripts/verifier_sensitivity.py
"""

import os
import sys
import json
import re
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.verifier import (
    score_region, score_longitudinal, score_diagnosis,
    score_reasoning_consistency, score_summary,
    parse_json_from_response, extract_summary_text,
    REGIONS, REGION_WEIGHTS, DX_ADJACENT,
    classify_zscore,
)


# ─── Weight Configurations ────────────────────────────────────────────────

CONFIGS = {
    'default': {
        'component_weights_longi': {
            'anatomical': 0.25, 'longitudinal': 0.20, 'diagnosis': 0.25,
            'reasoning': 0.15, 'summary': 0.15,
        },
        'component_weights_base': {
            'anatomical': 0.35, 'diagnosis': 0.35,
            'reasoning': 0.15, 'summary': 0.15,
        },
        'region_weights': dict(REGION_WEIGHTS),  # hip=1.2, ent=1.1, rest=1.0
    },
    'equal_comp': {
        'component_weights_longi': {
            'anatomical': 0.20, 'longitudinal': 0.20, 'diagnosis': 0.20,
            'reasoning': 0.20, 'summary': 0.20,
        },
        'component_weights_base': {
            'anatomical': 0.25, 'diagnosis': 0.25,
            'reasoning': 0.25, 'summary': 0.25,
        },
        'region_weights': dict(REGION_WEIGHTS),
    },
    'anat_heavy': {
        'component_weights_longi': {
            'anatomical': 0.40, 'longitudinal': 0.15, 'diagnosis': 0.20,
            'reasoning': 0.15, 'summary': 0.10,
        },
        'component_weights_base': {
            'anatomical': 0.50, 'diagnosis': 0.25,
            'reasoning': 0.15, 'summary': 0.10,
        },
        'region_weights': dict(REGION_WEIGHTS),
    },
    'dx_heavy': {
        'component_weights_longi': {
            'anatomical': 0.20, 'longitudinal': 0.15, 'diagnosis': 0.40,
            'reasoning': 0.15, 'summary': 0.10,
        },
        'component_weights_base': {
            'anatomical': 0.25, 'diagnosis': 0.50,
            'reasoning': 0.15, 'summary': 0.10,
        },
        'region_weights': dict(REGION_WEIGHTS),
    },
    'equal_region': {
        'component_weights_longi': {
            'anatomical': 0.25, 'longitudinal': 0.20, 'diagnosis': 0.25,
            'reasoning': 0.15, 'summary': 0.15,
        },
        'component_weights_base': {
            'anatomical': 0.35, 'diagnosis': 0.35,
            'reasoning': 0.15, 'summary': 0.15,
        },
        'region_weights': {r: 1.0 for r in REGIONS},
    },
}


# ─── Key Mapping ──────────────────────────────────────────────────────────

LABEL_KEY_MAP = {
    'Hippocampus': 'hippocampus', 'Entorhinal': 'entorhinal',
    'Fusiform': 'fusiform', 'MidTemp': 'midtemporal', 'Ventricles': 'ventricles',
}


def normalize_keys(d):
    return {LABEL_KEY_MAP.get(k, k.lower()): v for k, v in d.items()}


def parse_prior_from_prompt(prompt):
    prior_labels = {}
    prior_dx = None
    m = re.search(r'\*\*Prior Diagnosis\*\*:\s*(\w+)', prompt)
    if m:
        prior_dx = m.group(1)
    section = re.search(
        r'## Prior Anatomical Status.*?\n(.*?)(?=\n##|\n---|\n\*\*)',
        prompt, re.DOTALL,
    )
    if section:
        for line in section.group(1).strip().split('\n'):
            m2 = re.match(r'- \*\*(\w+)\*\*:\s*(\S+)', line)
            if m2:
                region = m2.group(1).lower()
                label = m2.group(2).strip()
                if region in LABEL_KEY_MAP.values():
                    prior_labels[region] = label
    return (prior_labels or None), prior_dx


# ─── Score one candidate with a specific weight config ────────────────────

def score_candidate(parsed_json, gt_z_scores, gt_labels, gt_dx,
                    prior_labels, has_longitudinal, prior_dx,
                    response_text, gt_summary, config):
    """Compute composite score with a specific weight configuration."""
    if parsed_json is None:
        return -2.0

    rw = config['region_weights']
    assessment = parsed_json.get('anatomical_assessment', {})

    # Anatomical score with config's region weights
    region_scores = {}
    for region in REGIONS:
        pred = assessment.get(region, {})
        pred_label = pred.get('label', 'normal')
        pred_conf = pred.get('confidence', 0.5)
        z = gt_z_scores.get(region, 0.0)
        region_scores[region] = score_region(pred_label, pred_conf, z, region)

    weighted_sum = sum(region_scores[r] * rw[r] for r in REGIONS)
    weight_total = sum(rw.values())
    anatomical_avg = weighted_sum / weight_total

    # Longitudinal
    pred_longi = parsed_json.get('longitudinal_comparison')
    longi_score = 0.0
    if has_longitudinal and pred_longi:
        longi_score = score_longitudinal(
            pred_longi, gt_z_scores, prior_labels, gt_labels)

    # Diagnosis
    pred_dx = parsed_json.get('diagnosis', {}).get('label', 'MCI')
    dx_score = score_diagnosis(pred_dx, gt_dx)

    # Reasoning
    reasoning = parsed_json.get('reasoning', {})
    reasoning_score = score_reasoning_consistency(
        reasoning, assessment, pred_longi, has_longitudinal=has_longitudinal)

    # Summary
    summary_score = score_summary(response_text, parsed_json, gt_summary) if response_text else 0.0

    # Weighted composite with config's component weights
    if has_longitudinal:
        cw = config['component_weights_longi']
        raw = (cw['anatomical'] * anatomical_avg +
               cw['longitudinal'] * longi_score +
               cw['diagnosis'] * dx_score +
               cw['reasoning'] * reasoning_score +
               cw['summary'] * summary_score)
    else:
        cw = config['component_weights_base']
        raw = (cw['anatomical'] * anatomical_avg +
               cw['diagnosis'] * dx_score +
               cw['reasoning'] * reasoning_score +
               cw['summary'] * summary_score)

    # Clinical weighting
    if (pred_dx, gt_dx) not in DX_ADJACENT and pred_dx != gt_dx:
        clinical_weight = 2.0
    elif pred_dx != gt_dx:
        clinical_weight = 1.5
    else:
        clinical_weight = 1.0

    return raw * clinical_weight


# ─── Kendall's τ ──────────────────────────────────────────────────────────

def kendall_tau(scores_a, scores_b):
    """Kendall's τ between two lists of 4 candidate scores (ranking agreement)."""
    n = len(scores_a)
    concordant = 0
    discordant = 0
    for i, j in combinations(range(n), 2):
        diff_a = scores_a[i] - scores_a[j]
        diff_b = scores_b[i] - scores_b[j]
        product = diff_a * diff_b
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
        # ties: ignore
    denom = concordant + discordant
    if denom == 0:
        return 1.0  # all tied → same ranking
    return (concordant - discordant) / denom


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    with open('data_splits/dpo_candidates_all.json') as f:
        candidates_data = json.load(f)
    with open('data_splits/singleturn_train.json') as f:
        train_data = json.load(f)
    gt_response_map = {s['sample_id']: s['response'] for s in train_data}

    print(f"Samples: {len(candidates_data)}")
    print(f"Configs: {list(CONFIGS.keys())}")

    # Score all candidates under each config
    all_scores = {name: [] for name in CONFIGS}  # name → list of [4 scores per sample]

    t0 = time.time()
    for i, sample in enumerate(candidates_data):
        sid = sample['sample_id']
        gt_labels = normalize_keys(sample['gt_labels'])
        gt_z_scores = normalize_keys(sample['gt_z_scores'])
        gt_dx = sample['gt_diagnosis']
        is_baseline = sample['is_baseline']
        has_longitudinal = not is_baseline

        prior_labels, prior_dx = None, None
        if has_longitudinal:
            prior_labels, prior_dx = parse_prior_from_prompt(sample['prompt'])

        gt_response = gt_response_map.get(sid, '')
        gt_summary = extract_summary_text(gt_response) if gt_response else None

        # Parse all 4 candidates once
        parsed_jsons = []
        response_texts = []
        for cand in sample['candidates']:
            response_texts.append(cand['response'])
            parsed_jsons.append(parse_json_from_response(cand['response']))

        # Score under each config
        for config_name, config in CONFIGS.items():
            scores = []
            for j in range(4):
                s = score_candidate(
                    parsed_jsons[j], gt_z_scores, gt_labels, gt_dx,
                    prior_labels, has_longitudinal, prior_dx,
                    response_texts[j], gt_summary, config,
                )
                scores.append(s)
            all_scores[config_name].append(scores)

        if (i + 1) % 1000 == 0:
            print(f"  Scored {i+1}/{len(candidates_data)} ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"Scoring complete in {elapsed:.1f}s\n")

    # ─── Analysis ─────────────────────────────────────────────────────
    default_scores = all_scores['default']
    n_samples = len(default_scores)

    print("=" * 70)
    print("VERIFIER WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 70)

    # For each alternative config, compare to default
    for config_name in ['equal_comp', 'anat_heavy', 'dx_heavy', 'equal_region']:
        alt_scores = all_scores[config_name]

        # 1. Pair overlap: same winner AND same loser
        same_pairs = 0
        same_winner = 0
        taus = []

        for i in range(n_samples):
            def_s = default_scores[i]
            alt_s = alt_scores[i]

            def_winner = int(np.argmax(def_s))
            def_loser = int(np.argmin(def_s))
            alt_winner = int(np.argmax(alt_s))
            alt_loser = int(np.argmin(alt_s))

            if def_winner == alt_winner and def_loser == alt_loser:
                same_pairs += 1
            if def_winner == alt_winner:
                same_winner += 1

            tau = kendall_tau(def_s, alt_s)
            taus.append(tau)

        pair_pct = 100 * same_pairs / n_samples
        winner_pct = 100 * same_winner / n_samples
        mean_tau = np.mean(taus)
        median_tau = np.median(taus)

        print(f"\n--- {config_name} vs default ---")
        print(f"  Same chosen-rejected pair: {same_pairs}/{n_samples} ({pair_pct:.1f}%)")
        print(f"  Same winner only:          {same_winner}/{n_samples} ({winner_pct:.1f}%)")
        print(f"  Kendall's τ:               mean={mean_tau:.3f}, median={median_tau:.3f}")

    # ─── Summary across all 4 alternatives ─────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*70}")

    all_pair_pcts = []
    all_taus = []
    for config_name in ['equal_comp', 'anat_heavy', 'dx_heavy', 'equal_region']:
        alt_scores = all_scores[config_name]
        same = 0
        taus = []
        for i in range(n_samples):
            def_s = default_scores[i]
            alt_s = alt_scores[i]
            if (int(np.argmax(def_s)) == int(np.argmax(alt_s)) and
                    int(np.argmin(def_s)) == int(np.argmin(alt_s))):
                same += 1
            taus.append(kendall_tau(def_s, alt_s))
        all_pair_pcts.append(100 * same / n_samples)
        all_taus.extend(taus)

    min_pair = min(all_pair_pcts)
    mean_pair = np.mean(all_pair_pcts)
    min_tau = min(np.mean(t) for t in [
        [kendall_tau(default_scores[i], all_scores[c][i]) for i in range(n_samples)]
        for c in ['equal_comp', 'anat_heavy', 'dx_heavy', 'equal_region']
    ])
    overall_tau = np.mean(all_taus)

    print(f"  Pair overlap range: {min_pair:.1f}% – {max(all_pair_pcts):.1f}%")
    print(f"  Pair overlap mean:  {mean_pair:.1f}%")
    print(f"  Kendall's τ range:  {min_tau:.3f} – overall mean {overall_tau:.3f}")
    print(f"\n  → Paper values: XX.X% pairs identical = {mean_pair:.1f}%")
    print(f"  → Paper values: Kendall's τ exceeds = {min_tau:.3f}")


if __name__ == '__main__':
    main()
