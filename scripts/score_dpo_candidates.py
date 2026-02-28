#!/usr/bin/env python3
"""
Score DPO candidates with verifier and build preference pairs.

Phase 1: Score all 3993×4 candidates → scored_candidates.json
Phase 2: Analyze score distribution → print statistics
Phase 3: Build preference pairs with GT replacement → dpo_pairs.json

Usage:
    # Phase 1+2: Score and analyze (CPU only, ~10-20 min)
    python scripts/score_dpo_candidates.py --phase score

    # Phase 3: Build pairs with thresholds (after reviewing stats)
    python scripts/score_dpo_candidates.py --phase build \
        --min_winner_score 0.3 \
        --margin_winner_score 0.5 \
        --margin 0.15
"""

import os
import sys
import json
import re
import argparse
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.verifier import (
    compute_composite_score, parse_json_from_response, extract_summary_text
)


# ─── Key Mapping ──────────────────────────────────────────────────────────

LABEL_KEY_MAP = {
    'Hippocampus': 'hippocampus',
    'Entorhinal': 'entorhinal',
    'Fusiform': 'fusiform',
    'MidTemp': 'midtemporal',
    'Ventricles': 'ventricles',
}


def normalize_keys(d):
    """Map capitalized FreeSurfer keys to lowercase verifier keys."""
    return {LABEL_KEY_MAP.get(k, k.lower()): v for k, v in d.items()}


# ─── Parse Prior Info from Prompt ─────────────────────────────────────────

def parse_prior_from_prompt(prompt):
    """Extract prior_labels and prior_dx from the prompt text.

    Returns: (prior_labels: dict or None, prior_dx: str or None)
    """
    prior_labels = {}
    prior_dx = None

    # Extract prior diagnosis
    m = re.search(r'\*\*Prior Diagnosis\*\*:\s*(\w+)', prompt)
    if m:
        prior_dx = m.group(1)

    # Extract prior anatomical status
    section = re.search(
        r'## Prior Anatomical Status.*?\n(.*?)(?=\n##|\n---|\n\*\*)',
        prompt, re.DOTALL
    )
    if section:
        for line in section.group(1).strip().split('\n'):
            m = re.match(r'- \*\*(\w+)\*\*:\s*(\S+)', line)
            if m:
                region = m.group(1).lower()
                label = m.group(2).strip()
                if region == 'midtemporal':
                    prior_labels['midtemporal'] = label
                elif region in LABEL_KEY_MAP.values():
                    prior_labels[region] = label

    if not prior_labels:
        return None, prior_dx
    return prior_labels, prior_dx


# ─── Extract GT Summary ──────────────────────────────────────────────────

def extract_gt_summary(gt_response):
    """Extract the [Diagnostic Summary] text from the GT response."""
    return extract_summary_text(gt_response)


# ─── Phase 1: Score All Candidates ───────────────────────────────────────

def score_all_candidates(candidates_path, train_path, output_path):
    """Score all DPO candidates and save results."""
    print("Loading data...")
    with open(candidates_path) as f:
        candidates_data = json.load(f)
    print(f"  Candidates: {len(candidates_data)} samples")

    # Build GT response lookup from training data
    with open(train_path) as f:
        train_data = json.load(f)
    gt_response_map = {s['sample_id']: s['response'] for s in train_data}
    print(f"  Training data: {len(train_data)} samples (GT responses)")

    scored = []
    n_json_fail = 0
    t0 = time.time()

    for i, sample in enumerate(candidates_data):
        sid = sample['sample_id']
        is_baseline = sample['is_baseline']

        # Normalize keys
        gt_labels = normalize_keys(sample['gt_labels'])
        gt_z_scores = normalize_keys(sample['gt_z_scores'])
        gt_dx = sample['gt_diagnosis']

        # Parse prior info from prompt (for follow-up)
        prior_labels, prior_dx = None, None
        has_longitudinal = not is_baseline
        if has_longitudinal:
            prior_labels, prior_dx = parse_prior_from_prompt(sample['prompt'])

        # Get GT summary
        gt_response = gt_response_map.get(sid, '')
        gt_summary = extract_gt_summary(gt_response) if gt_response else None

        # Score each candidate
        candidate_scores = []
        for j, cand in enumerate(sample['candidates']):
            response_text = cand['response']
            parsed = parse_json_from_response(response_text)
            if parsed is None:
                n_json_fail += 1

            result = compute_composite_score(
                parsed_json=parsed,
                gt_z_scores=gt_z_scores,
                gt_labels=gt_labels,
                gt_dx=gt_dx,
                prior_labels=prior_labels,
                has_longitudinal=has_longitudinal,
                prior_dx=prior_dx,
                response_text=response_text,
                gt_summary=gt_summary,
            )
            candidate_scores.append({
                'index': j,
                'composite': result['composite'],
                'anatomical': result['anatomical'],
                'longitudinal': result['longitudinal'],
                'diagnosis': result['diagnosis'],
                'reasoning': result['reasoning'],
                'summary': result['summary'],
                'clinical_weight': result['clinical_weight'],
            })

        scored.append({
            'sample_id': sid,
            'is_baseline': is_baseline,
            'gt_diagnosis': gt_dx,
            'has_prior': prior_labels is not None,
            'candidate_scores': candidate_scores,
        })

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Scored {i+1}/{len(candidates_data)} "
                  f"({elapsed:.1f}s, {(i+1)/elapsed:.0f} samples/s)")

    elapsed = time.time() - t0
    print(f"\nScoring complete: {len(scored)} samples in {elapsed:.1f}s")
    print(f"JSON parse failures: {n_json_fail}/{len(scored)*4} "
          f"({100*n_json_fail/(len(scored)*4):.1f}%)")

    with open(output_path, 'w') as f:
        json.dump(scored, f, indent=2)
    print(f"Saved scored candidates to {output_path}")

    return scored


# ─── Phase 2: Analyze Score Distribution ─────────────────────────────────

def analyze_distribution(scored):
    """Print detailed score distribution statistics."""
    all_composites = []
    winner_scores = []
    loser_scores = []
    margins = []  # max - min per sample
    dx_correct_counts = []

    for sample in scored:
        scores = [c['composite'] for c in sample['candidate_scores']]
        all_composites.extend(scores)

        best = max(scores)
        worst = min(scores)
        winner_scores.append(best)
        loser_scores.append(worst)
        margins.append(best - worst)

        n_dx_correct = sum(1 for c in sample['candidate_scores']
                          if c['diagnosis'] == 1.0)
        dx_correct_counts.append(n_dx_correct)

    import numpy as np
    all_composites = np.array(all_composites)
    winner_scores = np.array(winner_scores)
    loser_scores = np.array(loser_scores)
    margins = np.array(margins)

    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    print(f"\n--- All Candidates (N={len(all_composites)}) ---")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  P{pct}: {np.percentile(all_composites, pct):.4f}")
    print(f"  Mean: {all_composites.mean():.4f}, Std: {all_composites.std():.4f}")
    print(f"  Min: {all_composites.min():.4f}, Max: {all_composites.max():.4f}")

    print(f"\n--- Winner Scores (N={len(winner_scores)}) ---")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  P{pct}: {np.percentile(winner_scores, pct):.4f}")
    print(f"  Mean: {winner_scores.mean():.4f}, Std: {winner_scores.std():.4f}")

    print(f"\n--- Loser Scores (N={len(loser_scores)}) ---")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  P{pct}: {np.percentile(loser_scores, pct):.4f}")
    print(f"  Mean: {loser_scores.mean():.4f}, Std: {loser_scores.std():.4f}")

    print(f"\n--- Margins (max - min per sample) ---")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  P{pct}: {np.percentile(margins, pct):.4f}")
    print(f"  Mean: {margins.mean():.4f}, Std: {margins.std():.4f}")
    print(f"  Margin == 0 (all same): {(margins == 0).sum()}")
    print(f"  Margin < 0.05: {(margins < 0.05).sum()}")
    print(f"  Margin < 0.10: {(margins < 0.10).sum()}")
    print(f"  Margin < 0.15: {(margins < 0.15).sum()}")
    print(f"  Margin < 0.20: {(margins < 0.20).sum()}")

    print(f"\n--- DX Accuracy per sample ---")
    dx_counts = Counter(dx_correct_counts)
    for k in sorted(dx_counts.keys()):
        print(f"  {k}/4 correct: {dx_counts[k]} samples "
              f"({100*dx_counts[k]/len(scored):.1f}%)")

    # Threshold analysis
    print(f"\n--- GT Replacement Analysis ---")
    for thr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        n_below = (winner_scores < thr).sum()
        print(f"  Winner < {thr:.1f}: {n_below} ({100*n_below/len(winner_scores):.1f}%)")

    for thr in [0.3, 0.4, 0.5]:
        for margin_thr in [0.05, 0.10, 0.15, 0.20]:
            n_replace = ((winner_scores < thr) & (margins < margin_thr)).sum()
            print(f"  Winner < {thr:.1f} AND margin < {margin_thr:.2f}: "
                  f"{n_replace} ({100*n_replace/len(winner_scores):.1f}%)")

    # Baseline vs follow-up breakdown
    base_winners = [w for s, w in zip(scored, winner_scores) if s['is_baseline']]
    fu_winners = [w for s, w in zip(scored, winner_scores) if not s['is_baseline']]
    print(f"\n--- Baseline vs Follow-up Winners ---")
    base_winners = np.array(base_winners)
    fu_winners = np.array(fu_winners)
    print(f"  Baseline (N={len(base_winners)}): "
          f"mean={base_winners.mean():.4f}, median={np.median(base_winners):.4f}")
    print(f"  Follow-up (N={len(fu_winners)}): "
          f"mean={fu_winners.mean():.4f}, median={np.median(fu_winners):.4f}")


# ─── Phase 3: Build Preference Pairs ─────────────────────────────────────

def build_preference_pairs(scored_path, candidates_path, train_path,
                           output_path, min_winner_score, margin_winner_score,
                           margin_threshold):
    """Build DPO preference pairs with GT replacement strategy."""
    print(f"\nBuilding preference pairs...")
    print(f"  min_winner_score = {min_winner_score}")
    print(f"  margin_winner_score = {margin_winner_score}")
    print(f"  margin_threshold = {margin_threshold}")

    with open(scored_path) as f:
        scored = json.load(f)
    with open(candidates_path) as f:
        candidates_data = json.load(f)
    with open(train_path) as f:
        train_data = json.load(f)

    # Lookup maps
    cand_map = {s['sample_id']: s for s in candidates_data}
    gt_map = {s['sample_id']: s for s in train_data}

    pairs = []
    stats = Counter()

    for sample in scored:
        sid = sample['sample_id']
        cand_sample = cand_map[sid]
        gt_sample = gt_map.get(sid)

        scores = sample['candidate_scores']
        composites = [c['composite'] for c in scores]

        best_idx = max(range(4), key=lambda i: composites[i])
        worst_idx = min(range(4), key=lambda i: composites[i])
        best_score = composites[best_idx]
        worst_score = composites[worst_idx]
        score_margin = best_score - worst_score

        # GT replacement logic
        use_gt_winner = False
        if best_score < min_winner_score:
            # Rule 1: winner too weak → use GT
            use_gt_winner = True
            stats['gt_replace_low_score'] += 1
        elif (best_score < margin_winner_score and
              score_margin < margin_threshold):
            # Rule 2: winner not strong enough + all candidates similar → use GT
            use_gt_winner = True
            stats['gt_replace_low_margin'] += 1
        else:
            stats['model_winner'] += 1

        # Build chosen (winner) response
        if use_gt_winner and gt_sample:
            chosen_response = gt_sample['response']
            chosen_source = 'gt'
        else:
            chosen_response = cand_sample['candidates'][best_idx]['response']
            chosen_source = 'model'

        # Rejected (loser) response — always worst model candidate
        # But if worst == best (same idx after GT replacement), pick second worst
        if worst_idx == best_idx and not use_gt_winner:
            # Edge case: all identical scores, pick any different pair
            other_indices = [i for i in range(4) if i != best_idx]
            worst_idx = min(other_indices, key=lambda i: composites[i])
        rejected_response = cand_sample['candidates'][worst_idx]['response']

        pair = {
            'sample_id': sid,
            'mri_path': cand_sample['mri_path'],
            'is_baseline': cand_sample['is_baseline'],
            'prompt': cand_sample['prompt'],
            'chosen': chosen_response,
            'rejected': rejected_response,
            'chosen_source': chosen_source,
            'chosen_score': best_score if not use_gt_winner else None,
            'rejected_score': worst_score,
            'score_margin': score_margin,
            'winner_idx': best_idx,
            'loser_idx': worst_idx,
        }
        pairs.append(pair)

    print(f"\n--- Pair Statistics ---")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Model winner: {stats['model_winner']} "
          f"({100*stats['model_winner']/len(pairs):.1f}%)")
    print(f"  GT replace (low score): {stats['gt_replace_low_score']} "
          f"({100*stats['gt_replace_low_score']/len(pairs):.1f}%)")
    print(f"  GT replace (low margin): {stats['gt_replace_low_margin']} "
          f"({100*stats['gt_replace_low_margin']/len(pairs):.1f}%)")

    # Save
    with open(output_path, 'w') as f:
        json.dump(pairs, f, indent=2)
    print(f"\nSaved {len(pairs)} preference pairs to {output_path}")

    return pairs


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['score', 'build', 'all'],
                        default='all')
    parser.add_argument('--candidates', default='data_splits/dpo_candidates_all.json')
    parser.add_argument('--train', default='data_splits/singleturn_train.json')
    parser.add_argument('--scored_output', default='data_splits/scored_candidates.json')
    parser.add_argument('--pairs_output', default='data_splits/dpo_pairs.json')
    # GT replacement thresholds
    parser.add_argument('--min_winner_score', type=float, default=0.3,
                        help='Below this → always use GT as winner')
    parser.add_argument('--margin_winner_score', type=float, default=0.5,
                        help='Below this + small margin → use GT')
    parser.add_argument('--margin', type=float, default=0.15,
                        help='If margin < this + winner < margin_winner_score → use GT')
    args = parser.parse_args()

    if args.phase in ('score', 'all'):
        scored = score_all_candidates(
            args.candidates, args.train, args.scored_output)
        analyze_distribution(scored)

    if args.phase in ('build', 'all'):
        if args.phase == 'build':
            # Load pre-computed scores
            with open(args.scored_output) as f:
                scored = json.load(f)
            analyze_distribution(scored)

        build_preference_pairs(
            args.scored_output, args.candidates, args.train,
            args.pairs_output,
            args.min_winner_score, args.margin_winner_score, args.margin)


if __name__ == '__main__':
    main()
