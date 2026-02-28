#!/usr/bin/env python3
"""
Build AIBL test JSON (same format as singleturn_test.json) for eval.
Maps each preprocessed+resized MRI to clinical data and builds prompts.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

PROJ = Path(".")
RESIZE_DIR = None
EVAL_JSON = PROJ / "data_splits/aibl_eval.json"
OUT_JSON = PROJ / "data_splits/aibl_test.json"

IMAGE_SENTINEL = "<IMAGE>"

# ── Prompt templates ──

BASELINE_INSTRUCTIONS = """---

**Instructions**: Examine the 3D MRI and integrate all clinical information above.

**Analyze step by step:**
1. Describe key findings observed in the MRI scan
2. Connect imaging findings with available cognitive scores
3. Based on all evidence, determine anatomical status and diagnosis

First, respond with a structured JSON:

```json
{
  "reasoning": {
    "imaging_observations": "<describe key findings from the MRI>",
    "clinical_integration": "<connect imaging findings with cognitive scores>",
    "longitudinal_synthesis": "First visit — no prior data for comparison.",
    "regions_mentioned": ["<list of regions referenced>"],
    "progression_cited": false
  },
  "anatomical_assessment": {
    "hippocampus": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},
    "entorhinal": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},
    "fusiform": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},
    "midtemporal": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},
    "ventricles": {"label": "<normal|mild_enlargement|severe_enlargement>", "confidence": 0.0}
  },
  "longitudinal_comparison": null,
  "diagnosis": {"label": "<CN|MCI|Dementia>", "confidence": 0.0}
}
```

After the JSON block, write [Diagnostic Summary] followed by a brief clinical report (3–5 sentences) summarizing key findings."""

FOLLOWUP_INSTRUCTIONS = """---

**Instructions**: Examine the 3D MRI and integrate all clinical information above, including the prior visit data. Check your assessment against the irreversibility constraints and provide a longitudinal comparison.

**Analyze step by step:**
1. Describe key findings observed in the MRI scan
2. Connect imaging findings with available cognitive scores
3. Determine anatomical status and diagnosis
4. Check assessment against prior labels for irreversibility compliance
5. Compare with prior visit to characterize interval changes

First, respond with a structured JSON:

```json
{
  "reasoning": {
    "imaging_observations": "<describe key findings from the MRI>",
    "clinical_integration": "<connect imaging findings with cognitive scores>",
    "longitudinal_synthesis": "<compare with prior visit findings>",
    "regions_mentioned": ["<list of regions referenced>"],
    "progression_cited": true
  },
  "anatomical_assessment": {
    "hippocampus": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},
    "entorhinal": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},
    "fusiform": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},
    "midtemporal": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},
    "ventricles": {"label": "<normal|mild_enlargement|severe_enlargement>", "confidence": 0.0}
  },
  "longitudinal_comparison": {
    "hippocampus": "<stable|progressed|improved>",
    "entorhinal": "<stable|progressed|improved>",
    "fusiform": "<stable|progressed|improved>",
    "midtemporal": "<stable|progressed|improved>",
    "ventricles": "<stable|progressed|improved>"
  },
  "diagnosis": {"label": "<CN|MCI|Dementia>", "confidence": 0.0}
}
```

After the JSON block, write [Diagnostic Summary] followed by a brief clinical report (3–5 sentences) summarizing key findings and interval changes."""


def build_prompt(record):
    """Build a prompt for an AIBL sample in the same format as ADNI."""
    parts = [IMAGE_SENTINEL, "\n"]

    # Demographics
    parts.append("\n## Patient Demographics\n")
    if record["age"]:
        parts.append(f"- **Age**: {record['age']:.1f} years\n")
    else:
        parts.append("- **Age**: Not available\n")
    parts.append(f"- **Sex**: {record['sex'] or 'Not available'}\n")
    parts.append("- **Education**: Not available\n")

    # APOE
    parts.append("\n## Genetic Risk\n")
    if record["apoe_e4_count"] is not None:
        parts.append(f"- **APOE ε4 alleles**: {record['apoe_e4_count']}\n")
    else:
        parts.append("- **APOE ε4 alleles**: Not available\n")

    # Current cognitive
    parts.append("\n## Current Cognitive Assessment\n")
    if record["MMSE"] is not None:
        parts.append(f"- **MMSE**: {record['MMSE']}/30\n")
    else:
        parts.append("- **MMSE**: Not available\n")
    if record["CDR_GLOBAL"] is not None:
        parts.append(f"- **CDR Sum of Boxes**: {record['CDR_GLOBAL']}\n")
    else:
        parts.append("- **CDR Sum of Boxes**: Not available\n")

    is_baseline = record["is_baseline"]

    if not is_baseline and record.get("prior_DX"):
        # Longitudinal info
        parts.append("\n## Longitudinal History\n")
        parts.append(f"- **Prior Diagnosis**: {record['prior_DX']}\n")
        if record["prior_MMSE"] is not None:
            parts.append(f"- **Prior MMSE**: {record['prior_MMSE']}/30\n")
        else:
            parts.append("- **Prior MMSE**: Not available\n")
        if record["prior_CDR_GLOBAL"] is not None:
            parts.append(f"- **Prior CDR-SB**: {record['prior_CDR_GLOBAL']}\n")
        else:
            parts.append("- **Prior CDR-SB**: Not available\n")

        # Interval
        try:
            d1 = datetime.strptime(record["prior_visit_date"], "%Y-%m-%d")
            d2 = datetime.strptime(record["visit_date"], "%Y-%m-%d")
            months = (d2 - d1).days / 30.44
            parts.append(f"- **Interval**: {months:.1f} months since last visit\n")
        except:
            parts.append("- **Interval**: Not available\n")

        # No prior anatomical status for AIBL (no FreeSurfer)
        parts.append("\n## Prior Anatomical Status (from previous MRI assessment)\n")
        parts.append("- Not available (external dataset, no prior volumetric assessment)\n")

        # Irreversibility rules (DX only)
        parts.append("\n## Clinical Constraints (Irreversibility Rules)\n")
        parts.append("Neurodegenerative changes are irreversible:\n")
        parts.append("- If prior diagnosis was Dementia, current diagnosis cannot revert to MCI or CN.\n")
        parts.append("- If a region was severe_atrophy, it cannot improve to mild_atrophy or normal.\n")
        parts.append("- If a region was mild_atrophy, it cannot revert to normal.\n")
        parts.append("- If ventricles were severe_enlargement, they cannot shrink to mild_enlargement or normal.\n")
        parts.append("- If ventricles were mild_enlargement, they cannot revert to normal.\n")

        parts.append(FOLLOWUP_INSTRUCTIONS)
    else:
        parts.append(BASELINE_INSTRUCTIONS)

    return "".join(parts)


def main():
    global RESIZE_DIR, EVAL_JSON, OUT_JSON
    parser = argparse.ArgumentParser(description='Build AIBL test JSON for eval')
    parser.add_argument('--resize_dir', type=str, required=True,
                        help='Path to AIBL resized MRI directory')
    parser.add_argument('--eval_json', type=str, default=str(EVAL_JSON))
    parser.add_argument('--out_json', type=str, default=str(OUT_JSON))
    args = parser.parse_args()
    RESIZE_DIR = Path(args.resize_dir)
    EVAL_JSON = Path(args.eval_json)
    OUT_JSON = Path(args.out_json)

    with open(EVAL_JSON) as f:
        records = json.load(f)

    test_samples = []
    skipped = 0

    for r in records:
        sid = r["subject_id"]
        vdate = r["visit_date"]

        # Check resized npy exists
        npy_path = RESIZE_DIR / sid / f"{sid}_{vdate}_T1w_final.npy"
        if not npy_path.exists():
            skipped += 1
            continue

        prompt = build_prompt(r)

        sample = {
            "sample_id": r["sample_id"],
            "subject_id": sid,
            "visit_date": vdate,
            "mri_path": str(npy_path),
            "is_baseline": r["is_baseline"],
            "prompt": prompt,
            "gt_diagnosis": r["DX"],
            "gt_json": None,  # no FreeSurfer region ground truth for AIBL
            "response": "```json\n{}\n```\n\n[Diagnostic Summary]\nPlaceholder.",  # dummy for Stage1Dataset
        }
        test_samples.append(sample)

    with open(OUT_JSON, "w") as f:
        json.dump(test_samples, f, indent=2)

    print(f"Built {len(test_samples)} AIBL test samples -> {OUT_JSON}")
    print(f"Skipped {skipped} (no resized npy)")

    # Stats
    from collections import Counter
    dx_dist = Counter(s["gt_diagnosis"] for s in test_samples)
    bl_count = sum(1 for s in test_samples if s["is_baseline"])
    print(f"  Baseline: {bl_count}, Follow-up: {len(test_samples) - bl_count}")
    print(f"  DX: {dict(dx_dist)}")


if __name__ == "__main__":
    main()
