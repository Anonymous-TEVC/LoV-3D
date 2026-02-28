#!/usr/bin/env python3
"""Build miriad.csv from MIRIAD dataset NIfTI files.

Parses MIRIAD naming convention: miriad_PPP_GR_G_VV_MR_S.nii
  PPP = Subject ID (188-257)
  GR  = Group (AD/HC)
  G   = Gender (M/F)
  VV  = Visit number (01-10)
  S   = Scan number (1 or 2)

Output CSV columns:
  subject_id  — e.g., miriad_237_AD_M (used by pipeline as subject folder)
  group       — AD or HC
  gender      — M or F
  visit       — 01-10
  scan        — 1 or 2
  visit_date  — v{VV}_s{S} (synthetic, used by pipeline for output naming)
  input_path  — full path to .nii file
"""

import os
import re
import csv
import argparse
from pathlib import Path

MIRIAD_ROOT = None
OUTPUT_CSV = "./miriad.csv"

def main():
    global MIRIAD_ROOT, OUTPUT_CSV
    parser = argparse.ArgumentParser(description='Build miriad.csv from MIRIAD NIfTI files')
    parser.add_argument('--miriad_root', type=str, required=True,
                        help='Path to MIRIAD dataset root directory')
    parser.add_argument('--output_csv', type=str, default=OUTPUT_CSV)
    args = parser.parse_args()
    MIRIAD_ROOT = args.miriad_root
    OUTPUT_CSV = args.output_csv

    rows = []
    pattern = re.compile(
        r'miriad_(\d+)_(AD|HC)_(M|F)_(\d+)_MR_(\d+)\.nii$'
    )

    miriad = Path(MIRIAD_ROOT)
    nii_files = sorted(miriad.rglob("*.nii"))
    # Filter out .nii.gz
    nii_files = [f for f in nii_files if not str(f).endswith('.nii.gz')]

    for nii_path in nii_files:
        m = pattern.match(nii_path.name)
        if not m:
            print(f"WARNING: cannot parse {nii_path.name}, skipping")
            continue

        pid, group, gender, visit, scan = m.groups()
        subject_id = f"miriad_{pid}_{group}_{gender}"
        visit_date = f"v{visit}_s{scan}"

        rows.append({
            'subject_id': subject_id,
            'group': group,
            'gender': gender,
            'visit': visit,
            'scan': scan,
            'visit_date': visit_date,
            'input_path': str(nii_path),
        })

    # Sort by subject_id, visit, scan
    rows.sort(key=lambda r: (r['subject_id'], r['visit'], r['scan']))

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'subject_id', 'group', 'gender', 'visit', 'scan',
            'visit_date', 'input_path',
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Stats
    subjects = set(r['subject_id'] for r in rows)
    ad = sum(1 for r in rows if r['group'] == 'AD')
    hc = sum(1 for r in rows if r['group'] == 'HC')
    print(f"MIRIAD CSV: {OUTPUT_CSV}")
    print(f"  Total scans: {len(rows)}")
    print(f"  Subjects: {len(subjects)}")
    print(f"  AD scans: {ad}, HC scans: {hc}")
    print(f"  Visits: {sorted(set(r['visit'] for r in rows))}")
    print(f"  Scan 1: {sum(1 for r in rows if r['scan'] == '1')}, "
          f"Scan 2: {sum(1 for r in rows if r['scan'] == '2')}")

if __name__ == '__main__':
    main()
