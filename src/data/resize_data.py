#!/usr/bin/env python3
"""
Resize preprocessed MRI volumes to target resolution and save as .npy.

Takes the output of the preprocessing pipeline (Step 6: *_T1w_final.nii.gz)
and resamples to the target spatial resolution for a specific encoder.

This step is intentionally SEPARATE from the preprocessing pipeline because
different encoders require different input resolutions:
  - ResNet-50 (Med3D): 128×128×128
  - ViT-based:         96×96×96 or 64×64×64
  - SwinUNETR:         128×128×128

Output format: float32 .npy, shape (D, H, W) — NO channel dimension.
Channel dim is added by the dataset loader at runtime.

Usage:
  # From preprocessed NIfTI directory
  python src/data/resize_data.py \\
      --input_dir /path/to/06_normalized \\
      --output_dir /path/to/resize \\
      --target_shape 128 128 128

  # From a manifest CSV
  python src/data/resize_data.py \\
      --manifest /path/to/manifest.csv \\
      --output_dir /path/to/resize \\
      --target_shape 128 128 128

  # Skip existing files
  python src/data/resize_data.py \\
      --input_dir /path/to/06_normalized \\
      --output_dir /path/to/resize \\
      --target_shape 128 128 128 \\
      --skip_existing --workers 16
"""

import os
import re
import csv
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def resize_single(args):
    """Resize a single NIfTI to target shape and save as .npy.

    Args:
        args: tuple of (input_path, output_path, target_shape)

    Returns:
        (subject_id, visit_date, status, error_or_stats)
    """
    input_path, output_path, target_shape, subject_id, visit_date = args

    try:
        import nibabel as nib

        img = nib.load(input_path)
        data = img.get_fdata().astype(np.float32)

        # Handle 4D (rare, but defensive)
        if data.ndim == 4:
            data = data[..., 0]

        current_shape = data.shape

        # Resize if needed
        if current_shape != target_shape:
            scale_factors = [t / s for s, t in zip(current_shape, target_shape)]
            data = ndimage.zoom(data, scale_factors, order=1)  # trilinear

        # Verify shape
        assert data.shape == target_shape, \
            f"Shape mismatch: got {data.shape}, expected {target_shape}"

        # Save as float32 .npy (NO channel dimension)
        data = data.astype(np.float32)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, data)

        stats = {
            'original_shape': list(current_shape),
            'target_shape': list(target_shape),
            'range': [float(data.min()), float(data.max())],
            'mean': float(data.mean()),
            'file_size_mb': os.path.getsize(output_path) / 1e6,
        }

        return subject_id, visit_date, 'success', stats

    except Exception as e:
        return subject_id, visit_date, 'error', str(e)


def infer_subject_visit(filename):
    """Infer subject_id and visit_date from filename."""
    stem = Path(filename).stem
    if stem.endswith('.nii'):
        stem = stem[:-4]

    # ADNI: 002_S_0295_2011-06-02_T1w_final
    m = re.match(r'(\d{3}_S_\d+)_(\d{4}-\d{2}-\d{2})', stem)
    if m:
        return m.group(1), m.group(2)

    # MIRIAD: miriad_188_AD_M_v01_s1_T1w_final
    m = re.match(r'(miriad_\d+_\w+_\w)_(v\d+_s\d+)', stem)
    if m:
        return m.group(1), m.group(2)

    # AIBL: AIBL_1345_2012-09-24_T1w_final
    m = re.match(r'(AIBL_\d+)_(\d{4}-\d{2}-\d{2})', stem)
    if m:
        return m.group(1), m.group(2)

    return stem, 'unknown'


def main():
    parser = argparse.ArgumentParser(
        description='Resize MRI volumes to target resolution (.npy)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_dir', type=str,
                             help='Directory with preprocessed NIfTI files '
                                  '(*_T1w_final.nii.gz)')
    input_group.add_argument('--manifest', type=str,
                             help='CSV with columns: subject_id, visit_date, '
                                  'nifti_path')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for .npy files')
    parser.add_argument('--target_shape', type=int, nargs=3,
                        default=[128, 128, 128],
                        help='Target shape D H W (default: 128 128 128)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip files that already exist')
    args = parser.parse_args()

    target_shape = tuple(args.target_shape)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []

    if args.manifest:
        with open(args.manifest) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row['subject_id']
                vdate = row['visit_date']
                npath = row.get('nifti_path') or row.get('input_path')
                if npath and os.path.exists(npath):
                    out_path = str(output_dir / sid / f"{sid}_{vdate}_T1w_final.npy")
                    tasks.append((npath, out_path, target_shape, sid, vdate))
    else:
        input_dir = Path(args.input_dir)
        nii_files = sorted(
            list(input_dir.rglob('*_T1w_final.nii.gz'))
            + list(input_dir.rglob('*_T1w_final.nii'))
        )
        for f in nii_files:
            sid, vdate = infer_subject_visit(f.name)
            out_path = str(output_dir / sid / f"{sid}_{vdate}_T1w_final.npy")
            tasks.append((str(f), out_path, target_shape, sid, vdate))

    if args.skip_existing:
        before = len(tasks)
        tasks = [t for t in tasks if not os.path.exists(t[1])]
        skipped = before - len(tasks)
        if skipped:
            print(f"Skipping {skipped} existing files")

    print(f"Resizing {len(tasks)} volumes to {target_shape}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}")

    if not tasks:
        print("Nothing to process.")
        return

    success, fail = 0, 0
    all_stats = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(resize_single, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Resizing"):
            sid, vdate, status, result = future.result()
            if status == 'success':
                success += 1
                all_stats.append({'subject_id': sid, 'visit_date': vdate, **result})
            else:
                fail += 1
                print(f"\n  FAILED {sid}_{vdate}: {result}")

    print(f"\nDone! success={success}, fail={fail}")

    # Verify a random sample
    if all_stats:
        sample = all_stats[0]
        print(f"\nSample verification ({sample['subject_id']}):")
        print(f"  shape={sample['target_shape']}, "
              f"range=[{sample['range'][0]:.3f}, {sample['range'][1]:.3f}], "
              f"mean={sample['mean']:.4f}, size={sample['file_size_mb']:.1f}MB")

    # Save log
    log_path = str(output_dir / 'resize_log.json')
    with open(log_path, 'w') as f:
        json.dump({
            'target_shape': list(target_shape),
            'success': success,
            'fail': fail,
            'samples': all_stats[:10],  # first 10 for reference
        }, f, indent=2)
    print(f"Log saved to {log_path}")


if __name__ == '__main__':
    main()
