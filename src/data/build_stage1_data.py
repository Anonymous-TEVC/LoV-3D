#!/usr/bin/env python3
"""
Build Stage 1 SFT data — Single-turn format.

All samples (baseline and follow-up) use a single turn:
  Turn 1: Image + demographics + prior info (if follow-up) → JSON assessment
  Turn 2: Clinical report ([Comparison] + [Findings] + [Impression])

Outputs stage1_{train,val,test}.json with:
  - prompt:          Turn 1 prompt (image + demographics + instructions)
  - response:        Turn 1 response JSON
  - prompt_report:   Report turn prompt (shared template)
  - response_report: Report turn response ([Comparison]+[Findings]+[Impression])
  - gt_json:         Combined GT for evaluation (all fields in one dict)

Usage:
    python src/data/build_stage1_data.py
"""

import json
import random
import re
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DATA_CSV = Path('data_splits/dataset_with_labels.csv')
SPLITS_JSON = Path('data_splits/subject_splits.json')
NORMATIVE_JSON = Path('data_splits/normative_model.json')
RESIZE_DIR = None  # Must be set: path to resized MRI volumes (128^3 .npy files)
OUT_DIR = Path('data_splits')

REGIONS_CSV = ['Hippocampus', 'Entorhinal', 'Fusiform', 'MidTemp', 'Ventricles']
REGIONS_JSON = ['hippocampus', 'entorhinal', 'fusiform', 'midtemporal', 'ventricles']
REGION_CSV_TO_JSON = dict(zip(REGIONS_CSV, REGIONS_JSON))
REGION_JSON_TO_CSV = dict(zip(REGIONS_JSON, REGIONS_CSV))

REGION_DISPLAY = {
    'hippocampus': 'hippocampus',
    'entorhinal': 'entorhinal cortex',
    'fusiform': 'fusiform gyrus',
    'midtemporal': 'middle temporal gyrus',
    'ventricles': 'lateral ventricles',
}

ATROPHY_LABELS = ['normal', 'mild_atrophy', 'severe_atrophy']
EXPANSION_LABELS = ['normal', 'mild_enlargement', 'severe_enlargement']

LABEL_DESCRIPTION = {
    'normal': 'within normal limits for age',
    'mild_atrophy': 'mild volume loss',
    'severe_atrophy': 'severe volume loss',
    'mild_enlargement': 'mildly dilated',
    'severe_enlargement': 'markedly dilated',
}

DIRECTION_DESCRIPTION = {
    'stable': 'stable',
    'progressive_atrophy': 'interval volume loss',
    'progressive_enlargement': 'interval enlargement',
}

IMAGE_SENTINEL = '<IMAGE>'
SEED = 42


# ──────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────

def build_prompt_report(row, prior_row):
    """Build the report-turn prompt with FreeSurfer volumetric data.

    Args:
        row: current visit row (must have FreeSurfer columns)
        prior_row: prior visit row, or None for baselines
    """
    parts = []

    # ── FreeSurfer volumetric data (bullet-point format) ──
    parts.append('## FreeSurfer Volumetric Measurements')
    parts.append('(Z-score: standard deviations from age-matched normative mean; '
                 'negative = smaller than expected)')
    parts.append('')
    for r_csv, r_json in zip(REGIONS_CSV, REGIONS_JSON):
        display = REGION_DISPLAY[r_json]
        display_cap = display[0].upper() + display[1:]
        vol = f"{row[r_csv]:.0f}" if pd.notna(row.get(r_csv)) else 'N/A'
        zscore = f"{row[f'{r_csv}_zscore']:.2f}" if pd.notna(row.get(f'{r_csv}_zscore')) else 'N/A'
        if prior_row is not None:
            prior_vol = f"{prior_row[r_csv]:.0f}" if pd.notna(prior_row.get(r_csv)) else 'N/A'
            prior_zscore = f"{prior_row[f'{r_csv}_zscore']:.2f}" if pd.notna(prior_row.get(f'{r_csv}_zscore')) else 'N/A'
            parts.append(f'- {display_cap}: {vol} mm³ (z-score = {zscore}); '
                         f'prior: {prior_vol} mm³ (z-score = {prior_zscore})')
        else:
            parts.append(f'- {display_cap}: {vol} mm³ (z-score = {zscore})')

    icv = f"{row['ICV']:.0f}" if pd.notna(row.get('ICV')) else 'N/A'
    parts.append(f'- ICV: {icv} mm³')

    # ── Report instructions ──
    parts.append('')
    parts.append(
        'Based on your assessment above and the volumetric data, write a '
        'structured clinical report with three sections:'
    )
    parts.append('')
    parts.append('[Comparison]')
    parts.append(
        'Establish the temporal baseline. For a first visit, state no prior '
        'available. For a follow-up, reference the prior examination and '
        'interval only.'
    )
    parts.append('Do NOT include any current observations or diagnoses.')
    parts.append('')
    parts.append('[Findings]')
    parts.append(
        'Describe the imaging findings for each assessed region based on your '
        'assessment above. Incorporate the volumetric measurements and z-scores '
        'where relevant. Report any interval changes if prior data is available.'
    )
    parts.append(
        'Report only objective imaging findings. Do NOT include clinical '
        'diagnoses or disease labels.'
    )
    parts.append('')
    parts.append('[Impression]')
    parts.append(
        'Synthesize the findings to characterize the overall pattern and, if '
        'applicable, the disease trajectory. Use descriptive language (e.g., '
        '"consistent with," "suggestive of"). Note correlation or discordance '
        'with clinical information where relevant.'
    )
    parts.append('Do NOT introduce observations not described in [Findings].')

    return '\n'.join(parts)


def build_prompt(row, prior_row):
    """Build the single-turn prompt.

    Returns:
        (prompt, None, prompt_report):
            prompt:        Image + demographics + genetics + cognition + instructions
            None:          (reserved, always None)
            prompt_report: Report turn prompt (shared template)
    """
    is_baseline = row.get('is_baseline', 1) == 1
    has_prior_fs = prior_row is not None

    # ── Clinical info ──
    parts = [IMAGE_SENTINEL, '']

    # Demographics
    age = f"{row['AGE']:.1f}" if pd.notna(row['AGE']) else 'Not available'
    sex = row['PTGENDER'] if pd.notna(row['PTGENDER']) else 'Not available'
    edu = str(int(row['PTEDUCAT'])) if pd.notna(row['PTEDUCAT']) else 'Not available'
    parts.append('## Patient Demographics')
    parts.append(f'- **Age**: {age} years')
    parts.append(f'- **Sex**: {sex}')
    parts.append(f'- **Education**: {edu} years')

    # Genetic Risk
    apoe = str(int(row['APOE4'])) if pd.notna(row['APOE4']) else 'Unknown'
    parts.append('')
    parts.append('## Genetic Risk')
    parts.append(f'- **APOE ε4 alleles**: {apoe}')

    # Current Cognitive Assessment
    mmse = f"{int(row['MMSE'])}/30" if pd.notna(row['MMSE']) else 'Not available'
    cdrsb = f"{row['CDRSB']:.1f}" if pd.notna(row['CDRSB']) else 'Not available'
    parts.append('')
    parts.append('## Current Cognitive Assessment')
    parts.append(f'- **MMSE**: {mmse}')
    parts.append(f'- **CDR Sum of Boxes**: {cdrsb}')

    # ── Instructions ──
    parts.append('')
    parts.append('---')
    parts.append('')

    parts.append('**Instructions**: Examine the 3D MRI and integrate all clinical information above.')
    parts.append('')
    parts.append('**Analyze step by step:**')
    parts.append('1. Describe key findings observed in the MRI scan')
    parts.append('2. Connect imaging findings with available cognitive scores')
    parts.append('3. Based on all evidence, determine anatomical status and diagnosis')
    parts.append('')
    parts.append('Respond with a structured JSON:')
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

    return '\n'.join(parts), None, build_prompt_report(row, prior_row if (not is_baseline and has_prior_fs) else None)


# ──────────────────────────────────────────────────────────────────────
# Response builder
# ──────────────────────────────────────────────────────────────────────

def _get_direction(prior_label, current_label, region):
    """Determine longitudinal direction between prior and current labels.

    Returns:
        (direction, crossed_threshold, changed, prior_label)
    """
    if prior_label is None or current_label is None:
        return 'stable', False, False, prior_label

    if region == 'Ventricles':
        order = EXPANSION_LABELS
    else:
        order = ATROPHY_LABELS

    if prior_label not in order or current_label not in order:
        return 'stable', False, False, prior_label

    pi, ci = order.index(prior_label), order.index(current_label)
    changed = (pi != ci)
    if ci > pi:
        direction = 'progressive_enlargement' if region == 'Ventricles' else 'progressive_atrophy'
        return direction, True, changed, prior_label
    elif ci < pi:
        return 'stable', False, changed, prior_label
    else:
        return 'stable', False, False, prior_label


def _build_imaging_observations(labels, rng):
    """Build imaging_observations string from region labels."""
    abnormal = []
    normal_regions = []

    for r_json in REGIONS_JSON:
        r_csv = REGION_JSON_TO_CSV[r_json]
        label = labels.get(r_csv)
        if label is None:
            continue
        display = REGION_DISPLAY[r_json]
        if label == 'normal':
            normal_regions.append(display)
        else:
            desc = LABEL_DESCRIPTION[label]
            abnormal.append((display, desc, label))

    parts = []

    severe = [(d, desc) for d, desc, l in abnormal if 'severe' in l]
    mild = [(d, desc) for d, desc, l in abnormal if 'mild' in l]

    if severe:
        regions_str = ' and '.join(r for r, _ in severe)
        desc = severe[0][1]
        templates = [
            f"{desc.capitalize()} involving the {regions_str}.",
            f"The {regions_str} {'shows' if len(severe) == 1 else 'show'} {desc}.",
            f"There is {desc} in the {regions_str}.",
        ]
        parts.append(rng.choice(templates))

    if mild:
        regions_str = ' and '.join(r for r, _ in mild)
        desc = mild[0][1]
        templates = [
            f"{'Additional' if severe else ''} {desc} is noted in the {regions_str}.".replace('  ', ' ').strip(),
            f"The {regions_str} {'shows' if len(mild) == 1 else 'show'} {desc}.",
            f"There is {desc} in the {regions_str}.",
        ]
        parts.append(rng.choice(templates))

    if normal_regions:
        if 'lateral ventricles' in normal_regions:
            parts.append("Ventricular caliber is within normal limits for age.")
            normal_regions.remove('lateral ventricles')
        if normal_regions:
            regions_str = ', '.join(normal_regions)
            parts.append(f"The {regions_str} {'appears' if len(normal_regions) == 1 else 'appear'} normal.")

    if not parts:
        parts.append("No significant structural abnormality identified.")

    return ' '.join(parts)


def _build_clinical_integration(labels, dx, mmse, cdrsb, rng):
    """Build clinical_integration string connecting imaging with cognition."""
    n_abnormal = sum(1 for r in REGIONS_CSV if labels.get(r) not in (None, 'normal'))
    has_severe = any('severe' in str(labels.get(r, '')) for r in REGIONS_CSV)

    cog_parts = []
    cog_desc = None
    if pd.notna(mmse):
        mmse_val = int(mmse)
        if mmse_val >= 27:
            cog_desc = 'preserved cognitive function'
        elif mmse_val >= 24:
            cog_desc = 'mild cognitive impairment'
        else:
            cog_desc = 'significant cognitive decline'
        cog_parts.append(f"MMSE of {mmse_val}")
    if pd.notna(cdrsb):
        cdrsb_val = cdrsb
        if cog_desc is None:
            if cdrsb_val <= 0.5:
                cog_desc = 'preserved cognitive function'
            elif cdrsb_val <= 4.0:
                cog_desc = 'mild cognitive impairment'
            else:
                cog_desc = 'significant cognitive decline'
        cog_parts.append(f"CDR-SB {cdrsb_val:.1f}")

    if not cog_parts:
        return "Clinical cognitive data not available for correlation."

    cog_str = ' and '.join(cog_parts)

    if dx == 'CN':
        if n_abnormal == 0:
            templates = [
                f"{cog_str} indicate {cog_desc}, consistent with normal structural findings.",
                f"Structural findings are unremarkable, in keeping with {cog_str} suggesting {cog_desc}.",
            ]
        else:
            templates = [
                f"Despite mild structural changes, {cog_str} indicate {cog_desc}.",
                f"{cog_str} suggest {cog_desc}, though mild structural changes are noted.",
            ]
    elif dx == 'MCI':
        templates = [
            f"{cog_str} are consistent with the observed structural changes.",
            f"The pattern of regional atrophy correlates with {cog_str}.",
            f"Structural findings support the clinical picture indicated by {cog_str}.",
        ]
    else:
        if has_severe:
            templates = [
                f"{cog_str} are consistent with the degree of structural change.",
                f"The severity of structural atrophy is concordant with {cog_str}.",
            ]
        else:
            templates = [
                f"{cog_str} suggest cognitive impairment beyond what imaging alone demonstrates.",
                f"Clinical measures ({cog_str}) indicate impairment, with supporting structural changes.",
            ]

    return rng.choice(templates)


def _build_longitudinal_synthesis(prior_labels, current_labels, interval, rng):
    """Build longitudinal_synthesis string from prior→current comparison."""
    if prior_labels is None:
        return "First visit — no prior data for comparison."

    changed = []
    stable = []
    for r_csv in REGIONS_CSV:
        prior_l = prior_labels.get(r_csv)
        current_l = current_labels.get(r_csv)
        if prior_l is None or current_l is None:
            continue
        direction, crossed, _, _ = _get_direction(prior_l, current_l, r_csv)
        r_json = REGION_CSV_TO_JSON[r_csv]
        display = REGION_DISPLAY[r_json]
        if crossed:
            changed.append((display, prior_l, current_l))
        else:
            stable.append(display)

    interval_str = f"{interval:.0f}" if pd.notna(interval) else 'unknown'

    if not changed:
        templates = [
            f"Compared to the prior study {interval_str} months ago, no significant interval change is identified.",
            f"No significant progression compared to the prior examination {interval_str} months ago.",
            f"Stable appearance compared to prior study {interval_str} months ago.",
        ]
        return rng.choice(templates)

    change_parts = []
    for display, prior_l, current_l in changed:
        from_desc = LABEL_DESCRIPTION.get(prior_l, prior_l)
        to_desc = LABEL_DESCRIPTION.get(current_l, current_l)
        change_parts.append(f"{display} ({from_desc} to {to_desc})")

    change_str = ' and '.join(change_parts)

    templates = [
        f"Compared to the prior study {interval_str} months ago, interval progression in the {change_str}, with severity threshold crossing.",
        f"There is interval worsening in the {change_str} since the prior examination {interval_str} months ago.",
    ]
    result = rng.choice(templates)

    if stable:
        stable_str = ', '.join(stable)
        result += f" Remaining regions ({stable_str}) are stable."

    return result


def _build_report(labels, dx, row, prior_row, prior_labels, prior_dx, interval, longi, rng):
    """Build the three-section clinical report GT: [Comparison] + [Findings] + [Impression].

    Args:
        labels: dict of {region_csv: label} for current visit (final verified for follow-up)
        dx: current diagnosis (final verified for follow-up)
        row: current visit row (with FreeSurfer volumes and z-scores)
        prior_row: prior visit row (with FreeSurfer volumes), or None
        prior_labels: dict of {region_csv: label} for prior visit, or None
        prior_dx: prior diagnosis string, or None
        interval: months since prior, or None
        longi: longitudinal_comparison dict, or None
        rng: random.Random instance
    """
    # ── [Comparison] ──
    if prior_labels is None:
        comparison = "No prior examination available for comparison."
    else:
        interval_str = f"{interval:.0f}" if pd.notna(interval) else 'unknown'
        prior_dx_str = prior_dx if prior_dx else 'not available'
        comparison = (
            f"Comparison is made with MRI brain from {interval_str} months prior, "
            f"diagnosed as {prior_dx_str} at that time."
        )

    # ── [Findings] ──
    findings_parts = []

    # Current region status with volumetric data
    for r_csv, r_json in zip(REGIONS_CSV, REGIONS_JSON):
        label = labels.get(r_csv)
        if label is None:
            continue
        display = REGION_DISPLAY[r_json]
        desc = LABEL_DESCRIPTION[label]

        # Capitalize region display name for sentence start
        display_cap = display[0].upper() + display[1:]

        # Build quantitative parenthetical
        quant_parts = []
        vol = row.get(r_csv)
        if pd.notna(vol):
            quant_parts.append(f"volume {vol:.0f} mm³")
        zscore = row.get(f'{r_csv}_zscore')
        if pd.notna(zscore):
            quant_parts.append(f"z-score = {zscore:.2f}")
        quant_str = f" ({', '.join(quant_parts)})" if quant_parts else ""

        if label == 'normal':
            sentence = f"{display_cap}: within normal limits for age{quant_str}."
        else:
            sentence = f"{display_cap}: {desc}{quant_str}."

        # Append interval change for follow-up
        if longi is not None and r_json in longi and isinstance(longi[r_json], dict):
            direction = longi[r_json].get('direction', 'stable')
            dir_desc = DIRECTION_DESCRIPTION.get(direction, 'stable')
            # Add prior volume for context if available
            prior_quant = ""
            if prior_row is not None:
                prior_vol = prior_row.get(r_csv)
                if pd.notna(prior_vol):
                    prior_quant = f" (prior {prior_vol:.0f} mm³)"
            if direction == 'stable':
                sentence = sentence[:-1] + f"; {dir_desc}."
            else:
                sentence = sentence[:-1] + f"; {dir_desc}{prior_quant} compared to prior."

        findings_parts.append(sentence)

    findings = ' '.join(findings_parts)

    # ── [Impression] ──
    imp_parts = []

    # Overall structural characterization
    abnormal = []
    for r_csv in REGIONS_CSV:
        label = labels.get(r_csv)
        if label and label != 'normal':
            display = REGION_DISPLAY[REGION_CSV_TO_JSON[r_csv]]
            abnormal.append((display, label))

    if not abnormal:
        imp_parts.append("No significant parenchymal volume loss for age.")
    else:
        severe_list = [d for d, l in abnormal if 'severe' in l]
        mild_list = [d for d, l in abnormal if 'mild' in l]
        if severe_list and mild_list:
            severe_str = ' and '.join(severe_list)
            mild_str = ' and '.join(mild_list)
            imp_parts.append(
                f"Findings are consistent with severe volume loss in the {severe_str} "
                f"and mild volume loss in the {mild_str}."
            )
        elif severe_list:
            regions_str = ' and '.join(severe_list)
            imp_parts.append(f"Findings are consistent with severe volume loss in the {regions_str}.")
        else:
            regions_str = ' and '.join(mild_list)
            imp_parts.append(f"Findings are consistent with mild volume loss in the {regions_str}.")

    # Longitudinal trajectory (follow-up only)
    if prior_labels is not None and longi is not None:
        changed_regions = []
        for r_csv, r_json in zip(REGIONS_CSV, REGIONS_JSON):
            if r_json in longi and isinstance(longi[r_json], dict):
                if longi[r_json].get('crossed_threshold'):
                    changed_regions.append(REGION_DISPLAY[r_json])
        if changed_regions:
            regions_str = ' and '.join(changed_regions)
            templates = [
                f"Interval progression is noted in the {regions_str}, suggestive of progressive neurodegeneration.",
                f"Progressive changes in the {regions_str} compared to prior, consistent with ongoing neurodegeneration.",
            ]
            imp_parts.append(rng.choice(templates))
        else:
            templates = [
                "Imaging appearance remains stable compared to prior.",
                "No significant interval change, suggesting stable disease.",
            ]
            imp_parts.append(rng.choice(templates))

    # Clinical correlation
    cog_parts = []
    if pd.notna(row.get('MMSE')):
        cog_parts.append(f"MMSE of {int(row['MMSE'])}")
    if pd.notna(row.get('CDRSB')):
        cog_parts.append(f"CDR-SB of {row['CDRSB']:.1f}")

    if cog_parts:
        cog_str = ' and '.join(cog_parts)
        if dx == 'CN':
            if abnormal:
                imp_parts.append(f"{cog_str} indicate preserved cognitive function despite structural changes.")
            else:
                imp_parts.append(f"{cog_str} are consistent with the unremarkable imaging findings.")
        elif dx == 'MCI':
            templates = [
                f"The pattern of atrophy is suggestive of early neurodegenerative change, correlating with {cog_str}.",
                f"Findings are suggestive of early neurodegenerative change, consistent with {cog_str}.",
            ]
            imp_parts.append(rng.choice(templates))
        else:
            apoe = row.get('APOE4')
            if pd.notna(apoe) and int(apoe) > 0:
                templates = [
                    f"Overall pattern is suggestive of neurodegenerative dementia, likely Alzheimer etiology given APOE ε4 positivity, correlating with {cog_str}.",
                    f"Structural pattern is consistent with Alzheimer-type neurodegeneration in the setting of APOE ε4 carrier status and {cog_str}.",
                ]
            else:
                templates = [
                    f"Overall pattern is suggestive of neurodegenerative dementia, correlating with {cog_str}.",
                    f"Structural findings are consistent with advanced neurodegenerative disease, consistent with {cog_str}.",
                ]
            imp_parts.append(rng.choice(templates))

    impression = ' '.join(imp_parts)

    # Assemble full report
    report = f"[Comparison]\n{comparison}\n\n[Findings]\n{findings}\n\n[Impression]\n{impression}"
    return report


SEVERITY = {
    'normal': 0, 'mild_atrophy': 1, 'severe_atrophy': 2,
    'mild_enlargement': 1, 'severe_enlargement': 2,
}
DX_SEVERITY = {'CN': 0, 'MCI': 1, 'Dementia': 2}


def _detect_conflicts(current_labels, current_dx, prior_labels, prior_dx):
    """Detect irreversibility conflicts between Part 1 blind read and prior status.

    Returns:
        (has_conflict, conflicts_list, final_labels, final_dx)
        where final_labels/final_dx enforce monotonicity (severity can only stay or increase).
    """
    conflicts = []
    final_labels = {}

    # Region conflicts
    for r_csv, r_json in zip(REGIONS_CSV, REGIONS_JSON):
        curr_l = current_labels.get(r_csv)
        prior_l = prior_labels.get(r_csv) if prior_labels else None
        if curr_l is None or prior_l is None:
            final_labels[r_csv] = curr_l
            continue

        curr_sev = SEVERITY.get(curr_l, 0)
        prior_sev = SEVERITY.get(prior_l, 0)

        if curr_sev < prior_sev:
            # Current is less severe than prior → irreversibility violation
            if r_csv == 'Ventricles':
                rule = f"Ventricular enlargement cannot reverse; {prior_l} cannot improve to {curr_l}"
            else:
                rule = f"Atrophy cannot reverse; {prior_l} cannot improve to {curr_l}"
            conflicts.append({
                'field': r_json,
                'blind_read_value': curr_l,
                'prior_value': prior_l,
                'rule_violated': rule,
            })
            final_labels[r_csv] = prior_l  # enforce prior severity
        else:
            final_labels[r_csv] = curr_l  # current is equal or worse → keep

    # Diagnosis conflict (only Dementia is irreversible)
    final_dx = current_dx
    if prior_dx == 'Dementia' and current_dx in ('CN', 'MCI'):
        conflicts.append({
            'field': 'diagnosis',
            'blind_read_value': current_dx,
            'prior_value': 'Dementia',
            'rule_violated': 'AD diagnosis is irreversible; Dementia cannot revert to ' + current_dx,
        })
        final_dx = 'Dementia'

    return len(conflicts) > 0, conflicts, final_labels, final_dx


def _build_reconciliation_reasoning(has_conflict, conflicts, rng):
    """Build reconciliation_reasoning text for training GT."""
    if not has_conflict:
        templates = [
            "Part 1 assessment is consistent with prior clinical status. All current severity levels are equal to or greater than prior levels. No corrections required.",
            "No irreversibility conflicts detected. The blind read is consistent with the patient's clinical history. No corrections needed.",
            "Current assessment is concordant with prior status. No reversals detected; no corrections applied.",
        ]
        return rng.choice(templates)

    parts = []
    region_conflicts = [c for c in conflicts if c['field'] != 'diagnosis']
    dx_conflict = [c for c in conflicts if c['field'] == 'diagnosis']

    if dx_conflict:
        c = dx_conflict[0]
        parts.append(
            f"The blind read suggests {c['blind_read_value']}, but the patient was previously "
            f"diagnosed with Dementia. Since Alzheimer's disease is irreversible, the diagnosis "
            f"is maintained at Dementia."
        )

    for c in region_conflicts:
        display = REGION_DISPLAY.get(c['field'], c['field'])
        parts.append(
            f"The blind read assessed the {display} as {c['blind_read_value']}, but the prior "
            f"record indicates {c['prior_value']}. Since neurodegeneration is irreversible, "
            f"maintaining prior severity level."
        )

    return ' '.join(parts)


def build_response(row, prior_row, rng):
    """Build the single-turn GT response.

    Returns:
        (response, None, response_report, gt_json):
            response:        Response text (JSON block)
            None:            (reserved, always None)
            response_report: Report turn response ([Comparison]+[Findings]+[Impression])
            gt_json:         Combined GT dict for evaluation
    """
    labels = {}
    confidences = {}
    for r in REGIONS_CSV:
        labels[r] = row.get(f'{r}_label')
        confidences[r] = row.get(f'{r}_confidence')

    is_baseline = row.get('is_baseline', 1) == 1
    has_prior_fs = prior_row is not None and not is_baseline

    # ── anatomical_assessment ──
    anat = {}
    for r_csv, r_json in zip(REGIONS_CSV, REGIONS_JSON):
        anat[r_json] = {
            'label': labels[r_csv],
            'confidence': round(confidences[r_csv], 2) if pd.notna(confidences[r_csv]) else 0.5,
        }

    dx = row['DX']
    dx_conf = 0.9

    # ── Build reasoning text fields ──
    imaging_obs = _build_imaging_observations(labels, rng)
    clinical_int = _build_clinical_integration(
        labels, dx, row.get('MMSE'), row.get('CDRSB'), rng
    )

    # All samples (baseline and follow-up) use the same single-turn format
    mentioned = set()
    for r_csv, r_json in zip(REGIONS_CSV, REGIONS_JSON):
        if labels[r_csv] != 'normal':
            mentioned.add(r_json)

    reasoning = {
        'imaging_observations': imaging_obs,
        'clinical_integration': clinical_int,
        'longitudinal_synthesis': "First visit — no prior data for comparison.",
        'regions_mentioned': sorted(mentioned),
        'progression_cited': False,
    }

    response_json = {
        'reasoning': reasoning,
        'anatomical_assessment': anat,
        'longitudinal_comparison': None,
        'diagnosis': {'label': dx, 'confidence': dx_conf},
    }

    response_text = f"```json\n{json.dumps(response_json, indent=2)}\n```"

    # ── Report turn response ──
    response_report = _build_report(labels, dx, row, None, None, None, None, None, rng)

    gt_json = response_json.copy()

    return response_text, None, response_report, gt_json


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    global RESIZE_DIR
    parser = argparse.ArgumentParser(description='Build Stage 1 SFT data')
    parser.add_argument('--resize_dir', type=str, required=True,
                        help='Path to resized MRI volumes (128^3 .npy files)')
    args = parser.parse_args()
    RESIZE_DIR = Path(args.resize_dir)

    rng = random.Random(SEED)

    print("Loading data...")
    df = pd.read_csv(DATA_CSV)
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    with open(NORMATIVE_JSON) as f:
        normative = json.load(f)

    print(f"  Total rows: {len(df)}")
    print(f"  Subjects: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    required_cols = [f'{r}_label' for r in REGIONS_CSV]
    mask = df[required_cols].notna().all(axis=1)
    df_fs = df[mask].copy()
    print(f"  With all 5 FS labels: {len(df_fs)}")

    df_fs = df_fs.sort_values(['subject_id', 'visit_date'])

    row_index = {}
    for _, row in df_fs.iterrows():
        row_index[(row['subject_id'], row['visit_date'])] = row

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

        # Build singleturn prompt and response
        prompt_part1, _, prompt_report = build_prompt(row, prior_row)
        response_part1, _, response_report, gt_json = build_response(row, prior_row, rng)

        # GT metadata
        gt_labels = {r: row[f'{r}_label'] for r in REGIONS_CSV}
        gt_z_scores = {r: round(row[f'{r}_zscore'], 4)
                       for r in REGIONS_CSV if pd.notna(row.get(f'{r}_zscore'))}
        gt_zones = {r: row[f'{r}_zone']
                    for r in REGIONS_CSV if pd.notna(row.get(f'{r}_zone'))}
        gt_confidences = {r: round(row[f'{r}_confidence'], 2)
                         for r in REGIONS_CSV if pd.notna(row.get(f'{r}_confidence'))}

        # Change metadata for loss weighting
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
            # Singleturn data
            'prompt': prompt_part1,
            'response': response_part1,
            'prompt_report': prompt_report,
            'response_report': response_report,
            # GT for evaluation
            'gt_json': gt_json,
            'gt_labels': gt_labels,
            'gt_diagnosis': row['DX'],
            'gt_z_scores': gt_z_scores,
            'gt_zones': gt_zones,
            'gt_confidences': gt_confidences,
            'has_dx_change': has_dx_change,
            'n_region_changes': n_region_changes,
        }

        results[split_name].append(sample)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, samples in results.items():
        out_path = OUT_DIR / f'stage1_{split_name}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        n_bl = sum(1 for s in samples if s['is_baseline'])
        n_fu = sum(1 for s in samples if not s['is_baseline'])
        n_dx_chg = sum(1 for s in samples if s.get('has_dx_change'))
        n_reg_chg = sum(1 for s in samples if s.get('n_region_changes', 0) > 0)
        print(f"  {split_name}: {len(samples)} samples ({n_bl} baseline, {n_fu} follow-up, "
              f"{n_dx_chg} DX-changed, {n_reg_chg} region-changed) → {out_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
