"""
Merge DPO candidate chunks into a single file.

Chunks are named chunk_XXXX.json (from dynamic worker generation).
No scoring fields — just samples with K=4 candidate responses.

Usage:
    python scripts/merge_dpo_chunks.py --chunk_dir data_splits/dpo_candidates_singleturn
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Merge DPO candidate chunks')
    parser.add_argument('--chunk_dir', type=str, default='data_splits/dpo_candidates_singleturn')
    parser.add_argument('--output', type=str, default='data_splits/dpo_candidates_all.json')
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    chunk_files = sorted(chunk_dir.glob('chunk_*.json'))

    if not chunk_files:
        print("ERROR: No chunk files found!")
        return

    all_results = []
    for cf in chunk_files:
        try:
            with open(cf) as f:
                data = json.load(f)
            print(f"  {cf.name}: {len(data)} samples")
            all_results.extend(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [WARN] {cf.name}: corrupted ({e})")

    # Deduplicate by sample_id (in case of any overlaps)
    seen = set()
    unique = []
    for r in all_results:
        sid = r['sample_id']
        if sid not in seen:
            seen.add(sid)
            unique.append(r)
        else:
            print(f"  [DEDUP] Duplicate: {sid}")
    all_results = unique

    # Sort by sample_id
    all_results.sort(key=lambda x: x['sample_id'])

    print(f"\n=== Merged Stats ===")
    print(f"  Total samples: {len(all_results)}")
    print(f"  Chunks merged: {len(chunk_files)}")

    # Check candidate counts
    k_counts = [len(r['candidates']) for r in all_results]
    print(f"  Candidates per sample: {min(k_counts)}-{max(k_counts)}")

    output = Path(args.output)
    with open(output, 'w') as f:
        json.dump(all_results, f, indent=1)
    print(f"\n  Saved to {output}")
    print(f"  File size: {output.stat().st_size / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
