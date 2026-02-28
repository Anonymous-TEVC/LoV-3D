"""
Merge DPO candidate shards into a single file.

Usage:
    python scripts/merge_dpo_shards.py --shard_dir data_splits/dpo_candidates_v2 --num_shards 12
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Merge DPO candidate shards')
    parser.add_argument('--shard_dir', type=str, default='data_splits/dpo_candidates_v2')
    parser.add_argument('--num_shards', type=int, default=12)
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: <shard_dir>/dpo_candidates_all.json)')
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    output = Path(args.output) if args.output else shard_dir / 'dpo_candidates_all.json'

    all_results = []
    missing = []
    for i in range(args.num_shards):
        shard_path = shard_dir / f'shard_{i}.json'
        if not shard_path.exists():
            missing.append(i)
            print(f"  [WARN] Missing shard {i}: {shard_path}")
            continue
        with open(shard_path) as f:
            data = json.load(f)
        print(f"  Shard {i:2d}: {len(data):4d} samples")
        all_results.extend(data)

    if missing:
        print(f"\n  WARNING: {len(missing)} shards missing: {missing}")

    # Sort by sample_id for consistency
    all_results.sort(key=lambda x: x['sample_id'])

    # Stats
    n = len(all_results)
    scores = [r['all_scores'] for r in all_results]
    best_scores = [r['best_score'] for r in all_results]
    margins = [r['margin'] for r in all_results]

    import numpy as np
    print(f"\n=== Merged Stats ===")
    print(f"  Total samples: {n}")
    print(f"  Best score:  mean={np.mean(best_scores):.3f}, std={np.std(best_scores):.3f}")
    print(f"  Margin:      mean={np.mean(margins):.3f}, std={np.std(margins):.3f}")
    print(f"  Margin > 0:  {sum(1 for m in margins if m > 0)} ({sum(1 for m in margins if m > 0)/n*100:.1f}%)")

    # Per-score stats
    all_flat = [s for ss in scores for s in ss]
    print(f"  All scores:  mean={np.mean(all_flat):.3f}, std={np.std(all_flat):.3f}")
    print(f"  Score < 0:   {sum(1 for s in all_flat if s < 0)} (parse failures)")

    with open(output, 'w') as f:
        json.dump(all_results, f, indent=1)
    print(f"\n  Saved to {output}")
    print(f"  File size: {output.stat().st_size / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
