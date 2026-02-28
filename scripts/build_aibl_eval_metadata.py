#!/usr/bin/env python3
"""
Build AIBL evaluation metadata: each preprocessed MRI matched with
clinical data (MMSE, CDR, DX, APOE, demographics) and prior visit info
for longitudinal analysis.

Output: data_splits/aibl_eval.json (list of dicts, same format as singleturn_test.json)
        data_splits/aibl_eval_stats.txt (summary statistics)

Usage:
    python scripts/build_aibl_eval_metadata.py \
        --clinical_dir ./aibl_19Sep2019/Data_extract_3.3.0 \
        --preproc_dir /path/to/AIBL/data_preprocessing/06_normalized \
        --manifest data_splits/aibl_metadata.json
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

DX_MAP = {1: "CN", 2: "MCI", 3: "Dementia"}
VISCODE_ORDER = {"bl": 0, "m18": 1, "m36": 2, "m54": 3, "m72": 4, "m90": 5, "m108": 6}


def main():
    parser = argparse.ArgumentParser(description='Build AIBL evaluation metadata')
    parser.add_argument('--clinical_dir', type=str, required=True,
                        help='Path to AIBL clinical CSV directory (Data_extract_3.3.0)')
    parser.add_argument('--preproc_dir', type=str, required=True,
                        help='Path to AIBL preprocessed MRI directory (06_normalized)')
    parser.add_argument('--manifest', type=str, default='data_splits/aibl_metadata.json',
                        help='Path to AIBL metadata manifest JSON')
    parser.add_argument('--out_json', type=str, default='data_splits/aibl_eval.json')
    parser.add_argument('--out_stats', type=str, default='data_splits/aibl_eval_stats.txt')
    args = parser.parse_args()

    AIBL_CLINICAL = Path(args.clinical_dir)
    PREPROC_DIR = Path(args.preproc_dir)
    MANIFEST = Path(args.manifest)
    OUT_JSON = Path(args.out_json)
    OUT_STATS = Path(args.out_stats)

    # ── Load clinical data ──
    mmse = pd.read_csv(AIBL_CLINICAL / "aibl_mmse_01-Jun-2018.csv")
    cdr = pd.read_csv(AIBL_CLINICAL / "aibl_cdr_01-Jun-2018.csv")
    pdx = pd.read_csv(AIBL_CLINICAL / "aibl_pdxconv_01-Jun-2018.csv")
    demog = pd.read_csv(AIBL_CLINICAL / "aibl_ptdemog_01-Jun-2018.csv")
    apoe_df = pd.read_csv(AIBL_CLINICAL / "aibl_apoeres_01-Jun-2018.csv")

    # ── Load manifest ──
    with open(MANIFEST) as f:
        manifest = json.load(f)

    # ── Build clinical lookup: (RID, VISCODE) → {MMSE, CDR, DX} ──
    clinical = {}
    for _, row in pdx.iterrows():
        rid = int(row["RID"])
        vis = row["VISCODE"]
        dx_code = row["DXCURREN"]
        dx_label = DX_MAP.get(dx_code, None)
        if (rid, vis) not in clinical:
            clinical[(rid, vis)] = {}
        clinical[(rid, vis)]["DX"] = dx_label

    for _, row in mmse.iterrows():
        rid = int(row["RID"])
        vis = row["VISCODE"]
        score = row["MMSCORE"]
        if (rid, vis) not in clinical:
            clinical[(rid, vis)] = {}
        clinical[(rid, vis)]["MMSE"] = int(score) if score > 0 else None

    for _, row in cdr.iterrows():
        rid = int(row["RID"])
        vis = row["VISCODE"]
        cdg = row["CDGLOBAL"]
        if (rid, vis) not in clinical:
            clinical[(rid, vis)] = {}
        clinical[(rid, vis)]["CDR_GLOBAL"] = float(cdg) if cdg >= 0 else None

    # ── Demographics: RID → {gender, dob} ──
    demo_map = {}
    for _, row in demog.iterrows():
        rid = int(row["RID"])
        gender = "Male" if row["PTGENDER"] == 1 else "Female"
        dob_year = str(row["PTDOB"]).replace("/", "")
        demo_map[rid] = {"gender": gender, "birth_year": int(dob_year) if dob_year.isdigit() else None}

    # ── APOE: RID → genotype string ──
    apoe_map = {}
    for _, row in apoe_df.iterrows():
        rid = int(row["RID"])
        g1 = int(row["APGEN1"]) if row["APGEN1"] > 0 else None
        g2 = int(row["APGEN2"]) if row["APGEN2"] > 0 else None
        if g1 and g2:
            apoe_map[rid] = f"e{g1}/e{g2}"
            # Count e4 alleles
            apoe_map[rid + 100000] = sum(1 for g in [g1, g2] if g == 4)

    # ── Find all preprocessed files ──
    preproc_files = {}
    for f in PREPROC_DIR.rglob("*_T1w_final.nii.gz"):
        name = f.stem.replace("_T1w_final.nii", "")
        parts = name.split("_")
        subj_id = f"AIBL_{parts[1]}"
        visit_date = parts[2]
        preproc_files[(subj_id, visit_date)] = str(f)

    print(f"Preprocessed files found: {len(preproc_files)}")

    # ── Build per-subject visit history (sorted by date) ──
    subject_visits = defaultdict(list)
    for entry in manifest:
        sid = entry["subject_id"]
        vdate = entry["visit_date"]
        viscode = entry["viscode"]
        rid = int(sid.replace("AIBL_", ""))
        subject_visits[sid].append({
            "visit_date": vdate,
            "viscode": viscode,
            "rid": rid,
        })

    for sid in subject_visits:
        subject_visits[sid].sort(key=lambda x: x["visit_date"])

    # ── Build eval records ──
    records = []
    skipped = 0

    for entry in manifest:
        sid = entry["subject_id"]
        vdate = entry["visit_date"]
        viscode = entry["viscode"]
        rid = int(sid.replace("AIBL_", ""))

        key = (sid, vdate)
        if key not in preproc_files:
            skipped += 1
            continue

        mri_path = preproc_files[key]

        clin = clinical.get((rid, viscode), {})
        dx = clin.get("DX")
        mmse_score = clin.get("MMSE")
        cdr_global = clin.get("CDR_GLOBAL")

        demo = demo_map.get(rid, {})
        gender = demo.get("gender")
        birth_year = demo.get("birth_year")

        age = None
        if birth_year:
            try:
                scan_year = int(vdate.split("-")[0])
                age = scan_year - birth_year
            except:
                pass

        apoe_geno = apoe_map.get(rid)
        apoe_e4_count = apoe_map.get(rid + 100000, None)

        visits = subject_visits[sid]
        visit_idx = next((i for i, v in enumerate(visits) if v["visit_date"] == vdate), -1)
        is_baseline = (visit_idx == 0)

        prior_dx = None
        prior_mmse = None
        prior_cdr = None
        prior_date = None
        if visit_idx > 0:
            prev = visits[visit_idx - 1]
            prior_date = prev["visit_date"]
            prev_clin = clinical.get((prev["rid"], prev["viscode"]), {})
            prior_dx = prev_clin.get("DX")
            prior_mmse = prev_clin.get("MMSE")
            prior_cdr = prev_clin.get("CDR_GLOBAL")

        record = {
            "sample_id": f"{sid}_{vdate}",
            "subject_id": sid,
            "visit_date": vdate,
            "viscode": viscode,
            "mri_path": mri_path,
            "is_baseline": is_baseline,
            "visit_number": visit_idx + 1,
            "total_visits": len(visits),
            "age": age,
            "sex": gender,
            "apoe_genotype": apoe_geno,
            "apoe_e4_count": apoe_e4_count,
            "DX": dx,
            "MMSE": mmse_score,
            "CDR_GLOBAL": cdr_global,
            "prior_visit_date": prior_date,
            "prior_DX": prior_dx,
            "prior_MMSE": prior_mmse,
            "prior_CDR_GLOBAL": prior_cdr,
            "dx_changed": (prior_dx != dx) if (prior_dx and dx) else None,
        }
        records.append(record)

    records.sort(key=lambda x: (x["subject_id"], x["visit_date"]))

    # ── Save ──
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} records to {OUT_JSON}")
    print(f"Skipped {skipped} (no preprocessed file)")

    # ── Statistics ──
    stats_lines = []
    stats_lines.append(f"=== AIBL Evaluation Dataset Statistics ===")
    stats_lines.append(f"Total records: {len(records)}")
    stats_lines.append(f"Subjects: {len(set(r['subject_id'] for r in records))}")
    stats_lines.append(f"Baseline scans: {sum(1 for r in records if r['is_baseline'])}")
    stats_lines.append(f"Follow-up scans: {sum(1 for r in records if not r['is_baseline'])}")
    stats_lines.append("")

    dx_counts = defaultdict(int)
    for r in records:
        dx_counts[r["DX"] or "Unknown"] += 1
    stats_lines.append("DX distribution:")
    for dx, cnt in sorted(dx_counts.items()):
        stats_lines.append(f"  {dx}: {cnt} ({100*cnt/len(records):.1f}%)")
    stats_lines.append("")

    mmse_avail = sum(1 for r in records if r["MMSE"] is not None)
    stats_lines.append(f"MMSE available: {mmse_avail}/{len(records)} ({100*mmse_avail/len(records):.1f}%)")

    cdr_avail = sum(1 for r in records if r["CDR_GLOBAL"] is not None)
    stats_lines.append(f"CDR available: {cdr_avail}/{len(records)} ({100*cdr_avail/len(records):.1f}%)")

    apoe_avail = sum(1 for r in records if r["apoe_genotype"] is not None)
    stats_lines.append(f"APOE available: {apoe_avail}/{len(records)} ({100*apoe_avail/len(records):.1f}%)")
    stats_lines.append("")

    dx_changed = sum(1 for r in records if r["dx_changed"] is True)
    followups_with_prior = sum(1 for r in records if r["prior_DX"] is not None)
    stats_lines.append(f"Follow-ups with prior DX: {followups_with_prior}")
    stats_lines.append(f"DX conversions: {dx_changed}")
    stats_lines.append("")

    visit_counts = Counter(r["total_visits"] for r in records if r["is_baseline"])
    stats_lines.append("Visits per subject:")
    for nv, cnt in sorted(visit_counts.items()):
        stats_lines.append(f"  {nv} visits: {cnt} subjects")

    stats_text = "\n".join(stats_lines)
    with open(OUT_STATS, "w") as f:
        f.write(stats_text)
    print(f"\n{stats_text}")


if __name__ == "__main__":
    main()
