"""Build Stage 0 data JSON from ADNIMERGE.csv + dataset_with_labels.csv + subject_splits.json.

Uses ADNIMERGE baseline FreeSurfer volumes (Hippocampus_bl, etc.) normalized by ICV_bl.
Matches each subject to its earliest available .npy scan file.
"""

import csv
import json
import os
import glob
import argparse


REGIONS = ['Hippocampus', 'Entorhinal', 'Fusiform', 'MidTemp', 'Ventricles']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adnimerge_path', type=str, required=True,
                        help='Path to ADNIMERGE.csv')
    parser.add_argument('--splits_path', type=str, required=True,
                        help='Path to subject_splits.json')
    parser.add_argument('--npy_dir', type=str, required=True,
                        help='Directory containing resized .npy files')
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    # Load subject splits
    with open(args.splits_path) as f:
        splits = json.load(f)
    train_subs = set(splits['train'])
    val_subs = set(splits['val'])
    test_subs = set(splits['test'])

    # Read ADNIMERGE: extract baseline FS volumes from _bl columns
    with open(args.adnimerge_path) as f:
        reader = csv.DictReader(f)
        am_rows = list(reader)

    # One entry per subject: baseline ICV-normalized volumes
    subject_vols = {}
    for row in am_rows:
        sid = row.get('PTID', '')
        if sid in subject_vols:
            continue

        # Check all 5 baseline FS volumes + ICV
        ok = True
        raw_vols = {}
        for r in REGIONS:
            v = row.get(f'{r}_bl', '')
            if v == '':
                ok = False
                break
            fv = float(v)
            if fv <= 0:
                ok = False
                break
            raw_vols[r] = fv
        if not ok:
            continue

        icv = row.get('ICV_bl', '')
        if icv == '' or float(icv) <= 0:
            continue
        icv_val = float(icv)

        # Normalize by ICV
        volumes = {r: raw_vols[r] / icv_val for r in REGIONS}
        subject_vols[sid] = volumes

    print(f"Subjects with baseline FS volumes: {len(subject_vols)}")

    # Match each subject to earliest available .npy file
    data = {'train': [], 'val': [], 'test': []}

    for sid, volumes in subject_vols.items():
        subdir = os.path.join(args.npy_dir, sid)
        if not os.path.isdir(subdir):
            continue
        npys = sorted(glob.glob(os.path.join(subdir, '*_T1w_final.npy')))
        if not npys:
            continue

        # Use earliest scan
        npy_path = npys[0]
        # Extract visit date from filename: {sid}_{date}_T1w_final.npy
        basename = os.path.basename(npy_path)
        visit_date = basename.replace(f'{sid}_', '').replace('_T1w_final.npy', '')

        sample = {
            'subject_id': sid,
            'visit_date': visit_date,
            'npy_path': npy_path,
            'volumes': volumes,
        }

        if sid in train_subs:
            data['train'].append(sample)
        elif sid in val_subs:
            data['val'].append(sample)
        elif sid in test_subs:
            data['test'].append(sample)

    print(f"Train: {len(data['train'])}, Val: {len(data['val'])}, Test: {len(data['test'])}")

    with open(args.output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {args.output_path}")


if __name__ == '__main__':
    main()
