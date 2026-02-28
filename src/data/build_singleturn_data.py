#!/usr/bin/env python3
"""
Build Single-Turn SFT data — ablation baseline.

All information (demographics, cognition, prior history, FreeSurfer volumes)
is given in ONE user turn. The model predicts the structured JSON assessment
AND writes the three-section report in ONE assistant turn.

vs Multi-turn:
  - Multi-turn splits info across 2-3 turns (blind read → reconciliation → report)
  - Single-turn gives everything at once → model predicts everything at once

Prompt:  Image + demographics + genetics + cognition + [prior info] + FreeSurfer + instructions
Response: JSON prediction + [Comparison] + [Findings] + [Impression]

Usage:
    python src/data/build_singleturn_data.py
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from build_stage1_data import (
    DATA_CSV, SPLITS_JSON, NORMATIVE_JSON, RESIZE_DIR,
    REGIONS_CSV, REGIONS_JSON, REGION_CSV_TO_JSON, REGION_JSON_TO_CSV,
    REGION_DISPLAY, LABEL_DESCRIPTION, DIRECTION_DESCRIPTION,
    ATROPHY_LABELS, EXPANSION_LABELS,
    IMAGE_SENTINEL, SEED,
    _get_direction,
    _build_imaging_observations, _build_clinical_integration,
    _build_longitudinal_synthesis,
    _detect_conflicts, _build_reconciliation_reasoning,
)

OUT_DIR = Path('data_splits')


# ──────────────────────────────────────────────────────────────────────
# Diagnostic Summary builder (replaces 3-section report for single-turn)
# ──────────────────────────────────────────────────────────────────────

def _build_diagnostic_summary(labels, dx, row, prior_row, prior_labels, prior_dx, interval, longi, rng):
    """Build [Diagnostic Summary]: 3-5 sentence brief clinical report."""
    parts = []

    # Sentence 1: Overall structural characterization
    abnormal = []
    for r_csv in REGIONS_CSV:
        label = labels.get(r_csv)
        if label and label != 'normal':
            display = REGION_DISPLAY[REGION_CSV_TO_JSON[r_csv]]
            abnormal.append((display, label))

    if not abnormal:
        parts.append("No significant parenchymal volume loss for age.")
    else:
        severe_list = [d for d, l in abnormal if 'severe' in l]
        mild_list = [d for d, l in abnormal if 'mild' in l]
        if severe_list and mild_list:
            parts.append(
                f"Findings are consistent with severe volume loss in the "
                f"{' and '.join(severe_list)} and mild volume loss in the "
                f"{' and '.join(mild_list)}."
            )
        elif severe_list:
            parts.append(f"Findings are consistent with severe volume loss in the {' and '.join(severe_list)}.")
        else:
            parts.append(f"Findings are consistent with mild volume loss in the {' and '.join(mild_list)}.")

    # Sentence 2: Longitudinal trajectory (follow-up only)
    if prior_labels is not None and longi is not None:
        changed_regions = []
        for r_csv, r_json in zip(REGIONS_CSV, REGIONS_JSON):
            if r_json in longi and isinstance(longi[r_json], dict):
                if longi[r_json].get('crossed_threshold'):
                    changed_regions.append(REGION_DISPLAY[r_json])
        if changed_regions:
            regions_str = ' and '.join(changed_regions)
            parts.append(rng.choice([
                f"Interval progression is noted in the {regions_str}, suggestive of progressive neurodegeneration.",
                f"Progressive changes in the {regions_str} compared to prior, consistent with ongoing neurodegeneration.",
            ]))
        else:
            parts.append(rng.choice([
                "Imaging appearance remains stable compared to prior.",
                "No significant interval change, suggesting stable disease.",
            ]))

    # Sentence 3: Clinical correlation
    cog_parts = []
    if pd.notna(row.get('MMSE')):
        cog_parts.append(f"MMSE of {int(row['MMSE'])}")
    if pd.notna(row.get('CDRSB')):
        cog_parts.append(f"CDR-SB of {row['CDRSB']:.1f}")

    if cog_parts:
        cog_str = ' and '.join(cog_parts)
        if dx == 'CN':
            if abnormal:
                parts.append(f"{cog_str} indicate preserved cognitive function despite structural changes.")
            else:
                parts.append(f"{cog_str} are consistent with the unremarkable imaging findings.")
        elif dx == 'MCI':
            parts.append(rng.choice([
                f"The pattern of atrophy is suggestive of early neurodegenerative change, correlating with {cog_str}.",
                f"Findings are suggestive of early neurodegenerative change, consistent with {cog_str}.",
            ]))
        else:
            apoe = row.get('APOE4')
            if pd.notna(apoe) and int(apoe) > 0:
                parts.append(rng.choice([
                    f"Overall pattern is suggestive of neurodegenerative dementia, likely Alzheimer etiology given APOE ε4 positivity, correlating with {cog_str}.",
                    f"Structural pattern is consistent with Alzheimer-type neurodegeneration in the setting of APOE ε4 carrier status and {cog_str}.",
                ]))
            else:
                parts.append(rng.choice([
                    f"Overall pattern is suggestive of neurodegenerative dementia, correlating with {cog_str}.",
                    f"Structural findings are consistent with advanced neurodegenerative disease, consistent with {cog_str}.",
                ]))

    return '[Diagnostic Summary]\n' + ' '.join(parts)


# ──────────────────────────────────────────────────────────────────────
# Single-turn prompt builder
# ──────────────────────────────────────────────────────────────────────

def build_singleturn_prompt(row, prior_row):
    """Build a single-turn prompt: all info in one user turn.

    Model is asked to PREDICT the JSON (DX, regions, longitudinal comparison)
    and write the clinical report. No GT labels are leaked into the prompt.

    Args:
        row:       current visit row
        prior_row: prior visit row, or None for baselines
    """
    is_baseline = prior_row is None
    parts = [IMAGE_SENTINEL, '']

    # ── Demographics ──
    age = f"{row['AGE']:.1f}" if pd.notna(row['AGE']) else 'Not available'
    sex = row['PTGENDER'] if pd.notna(row['PTGENDER']) else 'Not available'
    edu = str(int(row['PTEDUCAT'])) if pd.notna(row['PTEDUCAT']) else 'Not available'
    parts.append('## Patient Demographics')
    parts.append(f'- **Age**: {age} years')
    parts.append(f'- **Sex**: {sex}')
    parts.append(f'- **Education**: {edu} years')

    # ── Genetic Risk ──
    apoe = str(int(row['APOE4'])) if pd.notna(row['APOE4']) else 'Unknown'
    parts.append('')
    parts.append('## Genetic Risk')
    parts.append(f'- **APOE ε4 alleles**: {apoe}')

    # ── Current Cognitive Assessment ──
    mmse = f"{int(row['MMSE'])}/30" if pd.notna(row['MMSE']) else 'Not available'
    cdrsb = f"{row['CDRSB']:.1f}" if pd.notna(row['CDRSB']) else 'Not available'
    parts.append('')
    parts.append('## Current Cognitive Assessment')
    parts.append(f'- **MMSE**: {mmse}')
    parts.append(f'- **CDR Sum of Boxes**: {cdrsb}')

    # ── Prior visit info (follow-up only) ──
    if not is_baseline:
        prior_dx = prior_row['DX'] if pd.notna(prior_row.get('DX')) else 'Not available'
        prior_mmse = f"{int(prior_row['MMSE'])}/30" if pd.notna(prior_row.get('MMSE')) else 'Not available'
        prior_cdrsb = f"{prior_row['CDRSB']:.1f}" if pd.notna(prior_row.get('CDRSB')) else 'Not available'
        interval = f"{row['months_since_prior']:.1f}" if pd.notna(row.get('months_since_prior')) else 'Not available'

        parts.append('')
        parts.append('## Longitudinal History')
        parts.append(f'- **Prior Diagnosis**: {prior_dx}')
        parts.append(f'- **Prior MMSE**: {prior_mmse}')
        parts.append(f'- **Prior CDR-SB**: {prior_cdrsb}')
        parts.append(f'- **Interval**: {interval} months since last visit')

        # Prior Anatomical Status
        parts.append('')
        parts.append('## Prior Anatomical Status (from previous MRI assessment)')
        for r_csv in REGIONS_CSV:
            r_json = REGION_CSV_TO_JSON[r_csv]
            col = f'{r_csv}_label'
            prior_label = prior_row.get(col)
            if pd.notna(prior_label):
                parts.append(f'- **{r_json}**: {prior_label}')
            else:
                parts.append(f'- **{r_json}**: not_available')

        # Clinical Constraints
        parts.append('')
        parts.append('## Clinical Constraints (Irreversibility Rules)')
        parts.append('Neurodegenerative changes are irreversible:')
        parts.append('- If prior diagnosis was Dementia, current diagnosis cannot revert to MCI or CN.')
        parts.append('- If a region was severe_atrophy, it cannot improve to mild_atrophy or normal.')
        parts.append('- If a region was mild_atrophy, it cannot revert to normal.')
        parts.append('- If ventricles were severe_enlargement, they cannot shrink to mild_enlargement or normal.')
        parts.append('- If ventricles were mild_enlargement, they cannot revert to normal.')

    # ── Instructions ──
    parts.append('')
    parts.append('---')
    parts.append('')

    if is_baseline:
        parts.append('**Instructions**: Examine the 3D MRI and integrate all clinical information above.')
        parts.append('')
        parts.append('**Analyze step by step:**')
        parts.append('1. Describe key findings observed in the MRI scan')
        parts.append('2. Connect imaging findings with available cognitive scores')
        parts.append('3. Based on all evidence, determine anatomical status and diagnosis')
        parts.append('')
        parts.append('First, respond with a structured JSON:')
        parts.append('')
        parts.append('```json')
        parts.append('{')
        parts.append('  "reasoning": {')
        parts.append('    "imaging_observations": "<describe key findings from the MRI>",')
        parts.append('    "clinical_integration": "<connect imaging findings with cognitive scores>",')
        parts.append('    "longitudinal_synthesis": "First visit — no prior data for comparison.",')
        parts.append('    "regions_mentioned": ["<list of regions referenced>"],')
        parts.append('    "progression_cited": false')
        parts.append('  },')
        parts.append('  "anatomical_assessment": {')
        parts.append('    "hippocampus": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
        parts.append('    "entorhinal": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
        parts.append('    "fusiform": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
        parts.append('    "midtemporal": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
        parts.append('    "ventricles": {"label": "<normal|mild_enlargement|severe_enlargement>", "confidence": 0.0}')
        parts.append('  },')
        parts.append('  "longitudinal_comparison": null,')
        parts.append('  "diagnosis": {"label": "<CN|MCI|Dementia>", "confidence": 0.0}')
        parts.append('}')
        parts.append('```')
    else:
        parts.append('**Instructions**: Examine the 3D MRI and integrate all clinical information above, '
                     'including the prior visit data. Check your assessment against the irreversibility '
                     'constraints and provide a longitudinal comparison.')
        parts.append('')
        parts.append('**Analyze step by step:**')
        parts.append('1. Describe key findings observed in the MRI scan')
        parts.append('2. Connect imaging findings with available cognitive scores')
        parts.append('3. Determine anatomical status and diagnosis')
        parts.append('4. Check assessment against prior labels for irreversibility violations')
        parts.append('5. Compare with prior assessment and determine longitudinal changes')
        parts.append('')
        parts.append('First, respond with a structured JSON:')
        parts.append('')
        parts.append('```json')
        parts.append('{')
        parts.append('  "reasoning": {')
        parts.append('    "imaging_observations": "<describe key findings from the MRI>",')
        parts.append('    "clinical_integration": "<connect imaging findings with cognitive scores>",')
        parts.append('    "longitudinal_synthesis": "<summarize changes from prior study>",')
        parts.append('    "regions_mentioned": ["<list of regions referenced>"],')
        parts.append('    "progression_cited": false')
        parts.append('  },')
        parts.append('  "anatomical_assessment": {')
        parts.append('    "hippocampus": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
        parts.append('    "entorhinal": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
        parts.append('    "fusiform": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
        parts.append('    "midtemporal": {"label": "<normal|mild_atrophy|severe_atrophy>", "confidence": 0.0},')
        parts.append('    "ventricles": {"label": "<normal|mild_enlargement|severe_enlargement>", "confidence": 0.0}')
        parts.append('  },')
        parts.append('  "longitudinal_comparison": {')
        parts.append('    "<region>": {"direction": "<stable|progressive_atrophy|progressive_enlargement>", "changed": false, "crossed_threshold": false},')
        parts.append('    "diagnosis_change": {"prior_diagnosis": "<DX>", "changed": false}')
        parts.append('  },')
        parts.append('  "diagnosis": {"label": "<CN|MCI|Dementia>", "confidence": 0.0}')
        parts.append('}')
        parts.append('```')

    parts.append('')
    parts.append(
        'After the JSON block, write [Diagnostic Summary] followed by a brief '
        'clinical report (3\u20135 sentences) summarizing key findings, changes '
        'from prior, and diagnostic impression.'
    )

    return '\n'.join(parts)


# ──────────────────────────────────────────────────────────────────────
# Single-turn response builder
# ──────────────────────────────────────────────────────────────────────

def build_singleturn_response(row, prior_row, rng):
    """Build the single-turn GT response: JSON + report in one block.

    Returns:
        (response_text, gt_json)
    """
    labels = {r: row.get(f'{r}_label') for r in REGIONS_CSV}
    confidences = {r: row.get(f'{r}_confidence') for r in REGIONS_CSV}
    dx = row['DX']
    is_baseline = prior_row is None

    # ── Build reasoning text ──
    imaging_obs = _build_imaging_observations(labels, rng)
    clinical_int = _build_clinical_integration(
        labels, dx, row.get('MMSE'), row.get('CDRSB'), rng
    )

    # ── anatomical_assessment ──
    anat = {}
    for r_csv, r_json in zip(REGIONS_CSV, REGIONS_JSON):
        anat[r_json] = {
            'label': labels[r_csv],
            'confidence': round(confidences[r_csv], 2) if pd.notna(confidences[r_csv]) else 0.5,
        }

    if is_baseline:
        # ══════ BASELINE ══════
        mentioned = set()
        for r_csv, r_json in zip(REGIONS_CSV, REGIONS_JSON):
            if labels[r_csv] != 'normal':
                mentioned.add(r_json)

        response_json = {
            'reasoning': {
                'imaging_observations': imaging_obs,
                'clinical_integration': clinical_int,
                'longitudinal_synthesis': "First visit — no prior data for comparison.",
                'regions_mentioned': sorted(mentioned),
                'progression_cited': False,
            },
            'anatomical_assessment': anat,
            'longitudinal_comparison': None,
            'diagnosis': {'label': dx, 'confidence': 0.9},
        }

        report = _build_diagnostic_summary(labels, dx, row, None, None, None, None, None, rng)
        gt_json = response_json.copy()
        final_labels = labels
        final_dx = dx

    else:
        # ══════ FOLLOW-UP ══════
        prior_labels = {r: prior_row.get(f'{r}_label') for r in REGIONS_CSV}
        prior_dx = prior_row['DX'] if pd.notna(prior_row.get('DX')) else None

        # Conflict detection & monotonicity enforcement
        has_conflict, conflicts, final_labels, final_dx = _detect_conflicts(
            labels, dx, prior_labels, prior_dx
        )

        # Use final_verified labels for anatomical_assessment
        final_anat = {}
        for r_csv, r_json in zip(REGIONS_CSV, REGIONS_JSON):
            final_anat[r_json] = {
                'label': final_labels[r_csv],
                'confidence': round(confidences[r_csv], 2) if pd.notna(confidences[r_csv]) else 0.5,
            }

        # Longitudinal comparison
        longi = {}
        for r_csv, r_json in zip(REGIONS_CSV, REGIONS_JSON):
            direction, crossed, changed, _ = _get_direction(
                prior_labels[r_csv], final_labels[r_csv], r_csv
            )
            longi[r_json] = {
                'direction': direction,
                'changed': changed,
                'crossed_threshold': crossed,
            }
        longi['diagnosis_change'] = {
            'prior_diagnosis': prior_dx,
            'changed': (prior_dx != final_dx) if prior_dx else False,
        }

        # Regions mentioned & progression
        mentioned = set()
        progression_cited = False
        for r_csv, r_json in zip(REGIONS_CSV, REGIONS_JSON):
            if final_labels[r_csv] != 'normal':
                mentioned.add(r_json)
            direction, crossed, _, _ = _get_direction(
                prior_labels[r_csv], final_labels[r_csv], r_csv
            )
            if crossed:
                mentioned.add(r_json)
                progression_cited = True

        interval = row.get('months_since_prior')
        longi_synth = _build_longitudinal_synthesis(
            prior_labels, final_labels, interval, rng
        )

        response_json = {
            'reasoning': {
                'imaging_observations': imaging_obs,
                'clinical_integration': clinical_int,
                'longitudinal_synthesis': longi_synth,
                'regions_mentioned': sorted(mentioned),
                'progression_cited': progression_cited,
            },
            'anatomical_assessment': final_anat,
            'longitudinal_comparison': longi,
            'diagnosis': {'label': final_dx, 'confidence': 0.9},
        }

        report = _build_diagnostic_summary(
            final_labels, final_dx, row, prior_row,
            prior_labels, prior_dx, interval, longi, rng
        )

        gt_json = response_json.copy()

    # ── Assemble: JSON block + report ──
    json_block = f"```json\n{json.dumps(response_json, indent=2)}\n```"
    response_text = json_block + '\n\n' + report

    return response_text, gt_json


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)

    print("Loading data...")
    df = pd.read_csv(DATA_CSV)
    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    print(f"  Total rows: {len(df)}")
    print(f"  Subjects: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    required_cols = [f'{r}_label' for r in REGIONS_CSV]
    mask = df[required_cols].notna().all(axis=1)
    df_fs = df[mask].copy()
    print(f"  With all 5 FS labels: {len(df_fs)}")

    df_fs = df_fs.sort_values(['subject_id', 'visit_date'])

    prior_map = {}
    for subj, group in df_fs.groupby('subject_id'):
        visits = group.sort_values('visit_date')
        prev_row = None
        for _, row in visits.iterrows():
            key = (row['subject_id'], row['visit_date'])
            if row.get('is_baseline', 1) == 1 or prev_row is None:
                prior_map[key] = None
            else:
                has_prior_labels = all(
                    pd.notna(prev_row.get(f'{r}_label'))
                    for r in REGIONS_CSV
                )
                prior_map[key] = prev_row if has_prior_labels else None
            prev_row = row

    split_sets = {
        'train': set(splits['train']),
        'val': set(splits['val']),
        'test': set(splits['test']),
    }

    results = {'train': [], 'val': [], 'test': []}

    for _, row in df_fs.iterrows():
        subj = row['subject_id']
        visit = row['visit_date']

        split_name = None
        for s, subjects in split_sets.items():
            if subj in subjects:
                split_name = s
                break
        if split_name is None:
            continue

        mri_path = RESIZE_DIR / subj / f"{subj}_{visit}_T1w_final.npy"
        if not mri_path.exists():
            continue

        key = (subj, visit)
        prior_row = prior_map.get(key)
        is_baseline = row.get('is_baseline', 1) == 1
        if not is_baseline and prior_row is None:
            is_baseline = True

        effective_prior = prior_row if not is_baseline else None

        # Build single-turn prompt and response
        prompt = build_singleturn_prompt(row, effective_prior)
        response, gt_json = build_singleturn_response(row, effective_prior, rng)

        # GT metadata
        gt_labels = {r: row[f'{r}_label'] for r in REGIONS_CSV}
        gt_z_scores = {r: round(row[f'{r}_zscore'], 4)
                       for r in REGIONS_CSV if pd.notna(row.get(f'{r}_zscore'))}

        # Change metadata
        has_dx_change = False
        n_region_changes = 0
        if not is_baseline and prior_row is not None:
            if pd.notna(prior_row.get('DX')) and prior_row['DX'] != row['DX']:
                has_dx_change = True
            for r in REGIONS_CSV:
                prior_l = prior_row.get(f'{r}_label')
                if prior_l and prior_l != row.get(f'{r}_label'):
                    n_region_changes += 1

        sample = {
            'sample_id': f"{subj}_{visit}",
            'subject_id': subj,
            'visit_date': visit,
            'mri_path': str(mri_path),
            'is_baseline': is_baseline,
            'prompt': prompt,
            'response': response,
            'gt_json': gt_json,
            'gt_labels': gt_labels,
            'gt_diagnosis': row['DX'],
            'gt_z_scores': gt_z_scores,
            'has_dx_change': has_dx_change,
            'n_region_changes': n_region_changes,
        }

        results[split_name].append(sample)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, samples in results.items():
        out_path = OUT_DIR / f'singleturn_{split_name}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        n_bl = sum(1 for s in samples if s['is_baseline'])
        n_fu = sum(1 for s in samples if not s['is_baseline'])
        print(f"  {split_name}: {len(samples)} samples ({n_bl} baseline, {n_fu} follow-up) → {out_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
