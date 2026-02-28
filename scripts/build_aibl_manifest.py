#!/usr/bin/env python3
"""
Build AIBL preprocessing manifest: match clinical CSV data with MRI DICOM directories.

Output: aibl_manifest.csv with columns:
  subject_id, visit_date, input_path, DX, MMSE, CDR, PTGENDER, APOE, sequence, series_id

Usage:
    python scripts/build_aibl_manifest.py
"""

import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── Default Paths (override via CLI) ──
CLINICAL_DIR = './aibl_19Sep2019/Data_extract_3.3.0'
MRI_ROOT = None
OUTPUT_CSV = './aibl_manifest.csv'
OUTPUT_JSON = './data_splits/aibl_metadata.json'

# Preferred sequence priority (lower = better)
SEQ_PRIORITY = {
    'MPRAGE_ADNI_confirmed': 1,
    'MPRAGE': 2,
    'MPRAGE_SAG_ISO_p2': 3,
    'MPRAGE_SAG_ISO_p2_ND': 4,
}


def load_clinical_data():
    """Load all clinical CSV files, indexed by (RID, VISCODE)."""
    # Demographics (baseline only)
    demog = {}
    with open(os.path.join(CLINICAL_DIR, 'aibl_ptdemog_01-Jun-2018.csv')) as f:
        for row in csv.DictReader(f):
            rid = row['RID'].strip()
            demog[rid] = {
                'PTGENDER': 'M' if row.get('PTGENDER', '').strip() == '1' else 'F',
                'PTDOB': row.get('PTDOB', '').strip(),
            }

    # APOE (baseline only)
    apoe = {}
    with open(os.path.join(CLINICAL_DIR, 'aibl_apoeres_01-Jun-2018.csv')) as f:
        for row in csv.DictReader(f):
            rid = row['RID'].strip()
            g1 = row.get('APGEN1', '').strip()
            g2 = row.get('APGEN2', '').strip()
            if g1 and g2 and g1 != '-4' and g2 != '-4':
                apoe[rid] = f"e{g1}/e{g2}"

    # MMSE (per visit)
    mmse = {}
    with open(os.path.join(CLINICAL_DIR, 'aibl_mmse_01-Jun-2018.csv')) as f:
        for row in csv.DictReader(f):
            rid = row['RID'].strip()
            viscode = row.get('VISCODE', '').strip()
            score = row.get('MMSCORE', '').strip()
            if score and score != '-4':
                mmse[(rid, viscode)] = score

    # CDR (per visit)
    cdr = {}
    with open(os.path.join(CLINICAL_DIR, 'aibl_cdr_01-Jun-2018.csv')) as f:
        for row in csv.DictReader(f):
            rid = row['RID'].strip()
            viscode = row.get('VISCODE', '').strip()
            score = row.get('CDGLOBAL', '').strip()
            if score and score != '-4':
                cdr[(rid, viscode)] = score

    # Diagnosis (per visit)
    dx = {}
    with open(os.path.join(CLINICAL_DIR, 'aibl_pdxconv_01-Jun-2018.csv')) as f:
        for row in csv.DictReader(f):
            rid = row['RID'].strip()
            viscode = row.get('VISCODE', '').strip()
            dxcurr = row.get('DXCURREN', '').strip()
            if dxcurr == '1':
                dx[(rid, viscode)] = 'CN'
            elif dxcurr == '2':
                dx[(rid, viscode)] = 'MCI'
            elif dxcurr == '3':
                dx[(rid, viscode)] = 'Dementia'

    return demog, apoe, mmse, cdr, dx


def scan_mri_dirs():
    """Scan AIBL MRI root and return list of (rid, sequence, session_date, dcm_dir, n_slices)."""
    scans = []
    mri_root = Path(MRI_ROOT)

    for subj_dir in sorted(mri_root.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name == 'data_preprocessing':
            continue
        rid = subj_dir.name

        for seq_dir in sorted(subj_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            seq_name = seq_dir.name
            # Skip repeat scans
            if 'REPEAT' in seq_name.upper() and seq_name not in SEQ_PRIORITY:
                continue

            for session_dir in sorted(seq_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                # Session dir name is like: 2008-10-10_10_32_18.0
                session_name = session_dir.name
                try:
                    visit_date = session_name.split('_')[0]  # e.g., '2008-10-10'
                    # Validate date format
                    datetime.strptime(visit_date, '%Y-%m-%d')
                except (ValueError, IndexError):
                    continue

                # Find DICOM series directory inside session
                for series_dir in session_dir.iterdir():
                    if not series_dir.is_dir():
                        continue
                    # Count DICOM files
                    dcm_files = [f for f in series_dir.iterdir()
                                 if f.name.endswith('.dcm')]
                    if len(dcm_files) >= 100:  # Quality filter
                        scans.append({
                            'rid': rid,
                            'sequence': seq_name,
                            'visit_date': visit_date,
                            'session_name': session_name,
                            'dcm_dir': str(series_dir),
                            'series_id': series_dir.name,
                            'n_slices': len(dcm_files),
                        })

    return scans


def date_to_viscode(rid, visit_date, all_dates):
    """Map visit date to AIBL visit code (bl, m18, m36, m54) based on ordering."""
    dates = sorted(all_dates[rid])
    idx = dates.index(visit_date)
    viscodes = ['bl', 'm18', 'm36', 'm54', 'm72', 'm90', 'm108']
    return viscodes[idx] if idx < len(viscodes) else f'm{idx * 18}'


def main():
    global CLINICAL_DIR, MRI_ROOT, OUTPUT_CSV, OUTPUT_JSON
    parser = argparse.ArgumentParser(description='Build AIBL preprocessing manifest')
    parser.add_argument('--clinical_dir', type=str, default=CLINICAL_DIR,
                        help='Path to AIBL clinical CSV directory')
    parser.add_argument('--mri_root', type=str, required=True,
                        help='Path to AIBL MRI root directory')
    parser.add_argument('--output_csv', type=str, default=OUTPUT_CSV)
    parser.add_argument('--output_json', type=str, default=OUTPUT_JSON)
    args = parser.parse_args()
    CLINICAL_DIR = args.clinical_dir
    MRI_ROOT = args.mri_root
    OUTPUT_CSV = args.output_csv
    OUTPUT_JSON = args.output_json

    print("=== Building AIBL Manifest ===\n")

    # Load clinical data
    print("Loading clinical data...")
    demog, apoe, mmse, cdr, dx = load_clinical_data()
    print(f"  Demographics: {len(demog)} subjects")
    print(f"  APOE: {len(apoe)} subjects")
    print(f"  MMSE: {len(mmse)} visit records")
    print(f"  CDR: {len(cdr)} visit records")
    print(f"  DX: {len(dx)} visit records")

    # Scan MRI directories
    print("\nScanning MRI directories...")
    scans = scan_mri_dirs()
    print(f"  Found {len(scans)} DICOM series (≥100 slices)")

    # Group by (rid, visit_date), pick best sequence per visit
    visit_scans = defaultdict(list)
    for s in scans:
        visit_scans[(s['rid'], s['visit_date'])].append(s)

    # Select best scan per visit (by sequence priority, then n_slices)
    best_scans = []
    for (rid, vdate), candidates in sorted(visit_scans.items()):
        candidates.sort(key=lambda x: (
            SEQ_PRIORITY.get(x['sequence'], 99),
            -x['n_slices']
        ))
        best_scans.append(candidates[0])

    print(f"  Best scans (1 per visit): {len(best_scans)}")

    # Collect all dates per subject for viscode mapping
    all_dates = defaultdict(set)
    for s in best_scans:
        all_dates[s['rid']].add(s['visit_date'])

    # Match with clinical data
    matched = []
    unmatched_dx = 0
    for s in best_scans:
        rid = s['rid']
        vdate = s['visit_date']
        viscode = date_to_viscode(rid, vdate, all_dates)

        d = demog.get(rid, {})
        record = {
            'subject_id': f"AIBL_{rid}",
            'visit_date': vdate,
            'input_path': s['dcm_dir'],
            'sequence': s['sequence'],
            'series_id': s['series_id'],
            'n_slices': s['n_slices'],
            'viscode': viscode,
            'DX': dx.get((rid, viscode), ''),
            'MMSE': mmse.get((rid, viscode), ''),
            'CDR': cdr.get((rid, viscode), ''),
            'PTGENDER': d.get('PTGENDER', ''),
            'PTDOB': d.get('PTDOB', ''),
            'APOE': apoe.get(rid, ''),
        }
        if record['DX']:
            matched.append(record)
        else:
            unmatched_dx += 1

    print(f"\n  Matched with DX: {len(matched)}")
    print(f"  No DX (skipped): {unmatched_dx}")

    # Stats
    dx_counts = defaultdict(int)
    for r in matched:
        dx_counts[r['DX']] += 1
    subjects = set(r['subject_id'] for r in matched)
    print(f"\n  Unique subjects: {len(subjects)}")
    print(f"  DX distribution: {dict(dx_counts)}")

    mmse_count = sum(1 for r in matched if r['MMSE'])
    cdr_count = sum(1 for r in matched if r['CDR'])
    apoe_count = sum(1 for r in matched if r['APOE'])
    print(f"  With MMSE: {mmse_count}/{len(matched)}")
    print(f"  With CDR: {cdr_count}/{len(matched)}")
    print(f"  With APOE: {apoe_count}/{len(matched)}")

    # Write manifest CSV (for preprocessing)
    fields = ['subject_id', 'visit_date', 'input_path', 'sequence', 'series_id',
              'n_slices', 'viscode', 'DX', 'MMSE', 'CDR', 'PTGENDER', 'PTDOB', 'APOE']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in matched:
            w.writerow(r)
    print(f"\nSaved manifest: {OUTPUT_CSV} ({len(matched)} rows)")

    # Write metadata JSON (for eval)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(matched, f, indent=2)
    print(f"Saved metadata: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
