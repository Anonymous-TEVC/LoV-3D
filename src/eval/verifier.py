"""
Verifier: Scores model outputs against FreeSurfer ground truth.

Implements IO Spec §5:
  - §5.1: Per-region anatomical score (with clinical region weighting)
  - §5.2: Longitudinal consistency score
  - §5.3: Diagnosis score
  - §5.4: Reasoning consistency score (enhanced: 5 sub-checks)
  - §5.5: Composite score with clinical weighting
  - §5.6: Diagnostic summary score (JSON-summary consistency, DX consistency,
          ROUGE-L vs GT, hallucination detection)

Used in Stage 2 (DPO) to score K=4 candidates and build preference pairs.
"""

import json
import re


# ─── Constants ─────────────────────────────────────────────────────────────

REGIONS = ['hippocampus', 'entorhinal', 'fusiform', 'midtemporal', 'ventricles']

REGION_DIRECTION = {
    'hippocampus': 'atrophy', 'entorhinal': 'atrophy',
    'fusiform': 'atrophy', 'midtemporal': 'atrophy',
    'ventricles': 'expansion',
}

# Clinical importance weighting — hippocampus and entorhinal are
# the most important regions for AD diagnosis
REGION_WEIGHTS = {
    'hippocampus': 1.2,
    'entorhinal': 1.1,
    'fusiform': 1.0,
    'midtemporal': 1.0,
    'ventricles': 1.0,
}

# Z-score thresholds per IO Spec §2.2
THRESHOLDS = {
    'atrophy': {'mild': -0.5, 'severe': -1.5},
    'expansion': {'mild': 0.5, 'severe': 1.5},
}

ATROPHY_LABELS = ['normal', 'mild_atrophy', 'severe_atrophy']
EXPANSION_LABELS = ['normal', 'mild_enlargement', 'severe_enlargement']

ATROPHY_ORDER = {'severe_atrophy': 0, 'mild_atrophy': 1, 'normal': 2}
EXPANSION_ORDER = {'severe_enlargement': 0, 'mild_enlargement': 1, 'normal': 2}

# Diagnosis adjacency
DX_ADJACENT = {
    ('CN', 'MCI'), ('MCI', 'CN'),
    ('MCI', 'Dementia'), ('Dementia', 'MCI'),
}

# Keywords for imaging observations consistency check
ATROPHY_KEYWORDS = ['atroph', 'reduc', 'loss', 'shrink', 'thin', 'decreased']
ENLARGEMENT_KEYWORDS = ['enlarg', 'expan', 'dilat', 'increased', 'widened']
NORMAL_KEYWORDS = ['normal', 'preserved', 'unremarkable', 'within normal',
                   'no significant']

# Display names used in GT summaries (build_singleturn_data.py)
REGION_DISPLAY_NAMES = {
    'hippocampus': 'hippocampus',
    'entorhinal': 'entorhinal cortex',
    'fusiform': 'fusiform gyrus',
    'midtemporal': 'middle temporal gyrus',
    'ventricles': 'lateral ventricles',
}

# Diagnosis display mapping
DX_KEYWORDS = {
    'CN': ['cognitively normal', 'preserved cognitive', 'unremarkable'],
    'MCI': ['mild cognitive', 'early neurodegenerative', 'mci'],
    'Dementia': ['dementia', 'alzheimer', 'advanced neurodegenerative'],
}


# ─── ROUGE-L Helper ──────────────────────────────────────────────────────

def _lcs_length(x, y):
    """Compute length of longest common subsequence of two sequences."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_f1(reference, hypothesis, beta=1.2):
    """Compute ROUGE-L F1 score between two strings (word-level).

    Returns: float in [0, 1]
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words or not hyp_words:
        return 0.0
    lcs = _lcs_length(ref_words, hyp_words)
    precision = lcs / len(hyp_words)
    recall = lcs / len(ref_words)
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (recall + beta_sq * precision)


# ─── Classification (§2.1-2.3) ────────────────────────────────────────────

def classify_zscore(z, region):
    """Classify Z-score into (label, zone_type, suggested_confidence)."""
    direction = REGION_DIRECTION[region]
    tol = 0.25

    if direction == 'atrophy':
        z_mild, z_severe = -0.5, -1.5
        if z > z_mild:
            label = 'normal'
            dist = z - z_mild
        elif z > z_severe:
            label = 'mild_atrophy'
            dist = min(z_mild - z, z - z_severe)
        else:
            label = 'severe_atrophy'
            dist = z_severe - z
    else:
        z_mild, z_severe = 0.5, 1.5
        if z < z_mild:
            label = 'normal'
            dist = z_mild - z
        elif z < z_severe:
            label = 'mild_enlargement'
            dist = min(z - z_mild, z_severe - z)
        else:
            label = 'severe_enlargement'
            dist = z - z_severe

    zone_type = 'core' if dist > tol else 'tolerance'

    if zone_type == 'tolerance':
        confidence = 0.55 + 0.60 * dist
    else:
        confidence = min(0.95, 0.75 + 0.10 * (dist - tol))

    return label, zone_type, round(confidence, 2)


def is_adjacent_label(label1, label2):
    """Check if two labels are adjacent (1 step apart)."""
    direction = None
    for d in ['atrophy', 'expansion']:
        labels = ATROPHY_LABELS if d == 'atrophy' else EXPANSION_LABELS
        if label1 in labels and label2 in labels:
            direction = d
            break
    if direction is None:
        return False

    order = ATROPHY_ORDER if direction == 'atrophy' else EXPANSION_ORDER
    return abs(order.get(label1, -1) - order.get(label2, -1)) == 1


def is_impossible_reversal(current_label, prior_label, region):
    """True if current is >=2 levels better than prior."""
    direction = REGION_DIRECTION[region]
    order = ATROPHY_ORDER if direction == 'atrophy' else EXPANSION_ORDER
    improvement = order.get(current_label, 0) - order.get(prior_label, 0)
    return improvement >= 2


def compute_gt_direction(current_label, prior_label, region):
    """Compute ground-truth direction from prior to current."""
    direction = REGION_DIRECTION[region]
    if direction == 'atrophy':
        order = ATROPHY_ORDER
        progressive = 'progressive_atrophy'
    else:
        order = EXPANSION_ORDER
        progressive = 'progressive_enlargement'

    cur_ord = order.get(current_label, 1)
    pri_ord = order.get(prior_label, 1)

    if cur_ord < pri_ord:
        return progressive
    return 'stable'


# ─── Reasoning Helpers ────────────────────────────────────────────────────

def _score_imaging_observations(obs_text, anatomical_assessment):
    """Check keyword consistency between imaging observations and predicted labels.

    For each region mentioned in the text, checks if atrophy/enlargement/normal
    keywords match the predicted anatomical label.

    Returns: float in [0, 1]
    """
    obs_lower = obs_text.lower()
    consistent = 0
    contradictions = 0
    total = 0

    for region in REGIONS:
        pred = anatomical_assessment.get(region, {})
        if not isinstance(pred, dict):
            continue
        label = pred.get('label', 'normal')

        if region not in obs_lower:
            continue
        total += 1

        # Context window around region mention
        idx = obs_lower.find(region)
        context = obs_lower[max(0, idx - 50):idx + len(region) + 80]

        has_atrophy = any(kw in context for kw in ATROPHY_KEYWORDS)
        has_enlarge = any(kw in context for kw in ENLARGEMENT_KEYWORDS)
        has_normal = any(kw in context for kw in NORMAL_KEYWORDS)

        if 'atrophy' in label:
            if has_atrophy:
                consistent += 1
            elif has_normal or has_enlarge:
                contradictions += 1
        elif 'enlargement' in label:
            if has_enlarge:
                consistent += 1
            elif has_normal or has_atrophy:
                contradictions += 1
        else:  # normal
            if has_normal or (not has_atrophy and not has_enlarge):
                consistent += 1
            elif has_atrophy or has_enlarge:
                contradictions += 1

    if total == 0:
        return 0.5
    return max(0.0, min(1.0, (consistent - contradictions) / total))


def _score_longitudinal_synthesis(synth_text, longitudinal_comparison):
    """Check if longitudinal synthesis text matches structured comparison data.

    Returns: float in [0, 1]
    """
    if not synth_text or not longitudinal_comparison:
        return 0.5

    text_lower = synth_text.lower()
    score = 0.5

    progressive_regions = []
    for r in REGIONS:
        r_data = longitudinal_comparison.get(r, {})
        if isinstance(r_data, dict):
            if r_data.get('direction', 'stable') != 'stable':
                progressive_regions.append(r)

    if progressive_regions:
        mentioned = sum(1 for r in progressive_regions if r in text_lower)
        score += 0.3 * (mentioned / len(progressive_regions))
    else:
        if any(kw in text_lower for kw in ['stable', 'unchanged', 'no interval']):
            score += 0.3
        elif any(kw in text_lower for kw in ['progress', 'worsen']):
            score -= 0.2

    return max(0.0, min(1.0, score))


def _check_internal_consistency(reasoning, anatomical_assessment,
                                longitudinal_comparison):
    """Check for internal contradictions between reasoning and structured output.

    Returns: float in [0, 1]
    """
    score = 1.0

    mentioned = set(reasoning.get('regions_mentioned', []))

    # Penalty: mentioning regions not in vocabulary
    invalid_regions = mentioned - set(REGIONS)
    if invalid_regions:
        score -= 0.3 * min(1.0, len(invalid_regions) / 2)

    # Penalty: severity mismatch between observations text and labels
    obs_text = str(reasoning.get('imaging_observations', '')).lower()
    for region in REGIONS:
        pred = anatomical_assessment.get(region, {})
        if not isinstance(pred, dict):
            continue
        label = pred.get('label', 'normal')
        if region not in obs_text:
            continue

        idx = obs_text.find(region)
        context = obs_text[max(0, idx - 30):idx + len(region) + 50]
        if 'severe' in context and label == 'normal':
            score -= 0.15
        elif ('normal' in context or 'preserved' in context) and 'severe' in label:
            score -= 0.15

    return max(0.0, score)


# ─── Scoring Functions (§5) ───────────────────────────────────────────────

def score_region(predicted_label, predicted_confidence, gt_z_score, region):
    """
    Score a single region prediction (§5.1).

    Returns: float in [-1.0, 1.0]
    """
    gt_label, zone_type, _ = classify_zscore(gt_z_score, region)

    if zone_type == 'core':
        if predicted_label == gt_label:
            if predicted_confidence >= 0.70:
                return 1.0
            else:
                return 0.5
        elif is_adjacent_label(predicted_label, gt_label):
            return -0.5
        else:
            return -1.0
    else:  # tolerance
        if predicted_label == gt_label:
            return 1.0
        elif is_adjacent_label(predicted_label, gt_label):
            if predicted_confidence < 0.80:
                return 0.7
            else:
                return 0.3
        else:
            return -1.0


def score_longitudinal(pred_longi, gt_z_scores, prior_labels, gt_labels):
    """
    Score longitudinal comparison (§5.2).

    Args:
        pred_longi: predicted longitudinal_comparison dict
        gt_z_scores: dict of region -> current Z-score
        prior_labels: dict of region -> prior label string
        gt_labels: dict of region -> current GT label string

    Returns: float score
    """
    if pred_longi is None or prior_labels is None:
        return 0.0

    total_score = 0.0
    n_regions = 0

    for region in REGIONS:
        pred_r = pred_longi.get(region, {})
        prior_label = prior_labels.get(region)
        current_label = gt_labels.get(region)

        if not prior_label or not current_label:
            continue

        n_regions += 1
        region_score = 0.0

        gt_direction = compute_gt_direction(current_label, prior_label, region)
        gt_crossed = (current_label != prior_label) and gt_direction.startswith('progressive')

        # Direction scoring
        pred_direction = pred_r.get('direction', 'stable')
        if pred_direction == gt_direction:
            region_score += 0.4
        else:
            region_score -= 0.4

        # Threshold crossing
        pred_crossed = pred_r.get('crossed_threshold', False)
        if pred_crossed == gt_crossed:
            region_score += 0.3
        else:
            region_score -= 0.3

        total_score += region_score

    # Consistency check scoring
    pred_cc = pred_longi.get('consistency_check', {})
    for region in REGIONS:
        prior_label = prior_labels.get(region)
        current_label = gt_labels.get(region)
        if not prior_label or not current_label:
            continue

        reversal = is_impossible_reversal(current_label, prior_label, region)
        if reversal:
            if not pred_cc.get('is_biologically_plausible', True):
                total_score += 0.3
            else:
                total_score -= 0.5
        else:
            if pred_cc.get('is_biologically_plausible', True):
                total_score += 0.1

    return total_score / max(n_regions, 1)


def score_diagnosis(predicted_dx, gt_dx):
    """Score diagnosis prediction (§5.3)."""
    if predicted_dx == gt_dx:
        return 1.0
    elif (predicted_dx, gt_dx) in DX_ADJACENT:
        return -0.5
    else:
        return -1.0


def score_reasoning_consistency(reasoning, anatomical_assessment,
                                longitudinal_comparison,
                                has_longitudinal=False):
    """Score reasoning consistency (§5.4) — enhanced with 5 sub-checks.

    Sub-checks:
      A. Region coverage (0.20): abnormal regions mentioned in regions_mentioned
      B. Imaging observations (0.25): keyword consistency with predicted labels
      C. Longitudinal citation (0.25): progression_cited + synthesis accuracy
      D. Clinical integration (0.15): presence of clinical reasoning text
      E. Internal consistency (0.15): no contradictions between fields
    """
    if not reasoning:
        return -0.5

    score = 0.0

    # ── A. Region coverage (0.20) ──
    abnormal_regions = [
        r for r, v in anatomical_assessment.items()
        if isinstance(v, dict) and v.get('label') not in ('normal',)
    ]
    mentioned = set(reasoning.get('regions_mentioned', []))

    if abnormal_regions:
        coverage = len(mentioned & set(abnormal_regions)) / len(abnormal_regions)
        score += 0.20 * coverage
    else:
        if len(mentioned) <= 1:
            score += 0.20
        else:
            score += 0.10

    # ── B. Imaging observations consistency (0.25) ──
    obs_text = reasoning.get('imaging_observations', '')
    if isinstance(obs_text, str) and obs_text.strip():
        obs_score = _score_imaging_observations(obs_text, anatomical_assessment)
        score += 0.25 * obs_score
    else:
        score += 0.05  # missing observations — mild penalty

    # ── C. Longitudinal citation (0.25) ──
    if has_longitudinal and longitudinal_comparison is not None:
        any_crossed = any(
            v.get('crossed_threshold', False)
            for k, v in longitudinal_comparison.items()
            if k not in ('consistency_check', 'diagnosis_change')
            and isinstance(v, dict)
        )
        any_progressive = any(
            v.get('direction', 'stable') != 'stable'
            for k, v in longitudinal_comparison.items()
            if k not in ('consistency_check', 'diagnosis_change')
            and isinstance(v, dict)
        )

        if any_crossed:
            if reasoning.get('progression_cited', False):
                score += 0.15
            else:
                score -= 0.15
        elif any_progressive:
            if reasoning.get('progression_cited', False):
                score += 0.10
            else:
                score += 0.05
        else:  # all stable
            if not reasoning.get('progression_cited', False):
                score += 0.10
            else:
                score -= 0.05

        # Longitudinal synthesis text
        synth_text = reasoning.get('longitudinal_synthesis', '')
        if isinstance(synth_text, str) and synth_text.strip():
            synth_score = _score_longitudinal_synthesis(
                synth_text, longitudinal_comparison)
            score += 0.10 * synth_score
    else:
        # Baseline
        if reasoning.get('progression_cited', False):
            score -= 0.15  # hallucination
        else:
            score += 0.15

        synth_text = reasoning.get('longitudinal_synthesis', '')
        if isinstance(synth_text, str) and len(synth_text.strip()) > 50:
            score -= 0.10  # should not have detailed synthesis for baseline
        else:
            score += 0.10

    # ── D. Clinical integration (0.15) ──
    clin_text = reasoning.get('clinical_integration', '')
    if isinstance(clin_text, str) and clin_text.strip():
        score += 0.10
    else:
        score += 0.05

    # ── E. Internal consistency (0.15) ──
    consistency = _check_internal_consistency(
        reasoning, anatomical_assessment, longitudinal_comparison)
    score += 0.15 * consistency

    return score


def score_change_detection(pred_longi, gt_labels, prior_labels, gt_dx,
                           prior_dx=None):
    """Score accuracy of predicted 'changed' flags (§5.6).

    Checks per-region changed flags and diagnosis_change against ground truth.

    Returns: float score normalized by number of checks
    """
    if pred_longi is None or prior_labels is None:
        return 0.0

    score = 0.0
    total_checks = 0

    # Per-region changed flag
    for region in REGIONS:
        pred_r = pred_longi.get(region, {})
        if not isinstance(pred_r, dict) or 'changed' not in pred_r:
            continue
        gt_changed = (gt_labels.get(region) != prior_labels.get(region))
        total_checks += 1
        if pred_r['changed'] == gt_changed:
            score += 1.0
        else:
            score -= 0.5

    # Diagnosis change
    pred_dx_chg = pred_longi.get('diagnosis_change', {})
    if isinstance(pred_dx_chg, dict) and 'changed' in pred_dx_chg and prior_dx:
        total_checks += 1
        gt_dx_changed = (gt_dx != prior_dx)
        if pred_dx_chg['changed'] == gt_dx_changed:
            score += 1.0
        else:
            score -= 0.5

    return score / max(total_checks, 1)


def extract_summary_text(response_text):
    """Extract [Diagnostic Summary] text from full model response."""
    # Try explicit header
    for header in ['[Diagnostic Summary]', '[Impression]']:
        idx = response_text.find(header)
        if idx >= 0:
            return response_text[idx + len(header):].strip()
    # Fallback: text after the JSON code block
    m = re.search(r'```\s*\n(.*)', response_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ''


def score_summary(response_text, parsed_json, gt_summary=None):
    """Score the diagnostic summary section (§5.7).

    Four sub-checks (weights within this component):
      A. JSON-Summary consistency (0.30): atrophy/severity keywords match JSON labels
      B. DX consistency (0.25): summary conclusion matches JSON diagnosis
      C. ROUGE-L (0.10): similarity with GT summary (if available)
      D. Hallucination detection (0.35): penalize fabricated regions/data

    Args:
        response_text: full model response string (JSON + summary)
        parsed_json: parsed JSON dict from the response
        gt_summary: ground-truth summary string (or None)

    Returns: float score in [-0.5, 1.0]
    """
    summary = extract_summary_text(response_text)
    if not summary or parsed_json is None:
        return -0.5

    summary_lower = summary.lower()
    score = 0.0
    assessment = parsed_json.get('anatomical_assessment', {})

    # ── A. JSON-Summary consistency (0.30) ──
    consistent = 0
    contradictions = 0
    total = 0
    for region in REGIONS:
        pred = assessment.get(region, {})
        if not isinstance(pred, dict):
            continue
        label = pred.get('label', 'normal')

        # Check all display name variants for this region
        display = REGION_DISPLAY_NAMES.get(region, region)
        found_in_summary = (region in summary_lower or display in summary_lower)
        if not found_in_summary:
            continue
        total += 1

        # Get context around mention
        idx = summary_lower.find(display)
        if idx < 0:
            idx = summary_lower.find(region)
        context = summary_lower[max(0, idx - 60):idx + len(display) + 80]

        has_severe = 'severe' in context
        has_mild = 'mild' in context
        has_normal_kw = any(kw in context for kw in NORMAL_KEYWORDS)
        has_loss = any(kw in context for kw in ATROPHY_KEYWORDS)
        has_enlarge = any(kw in context for kw in ENLARGEMENT_KEYWORDS)

        if 'severe' in label:
            if has_severe or has_loss:
                consistent += 1
            elif has_normal_kw or has_mild:
                contradictions += 1
        elif 'mild' in label:
            if has_mild or has_loss:
                consistent += 1
            elif has_severe:
                contradictions += 1
            elif has_normal_kw:
                contradictions += 1
        elif 'enlargement' in label:
            if has_enlarge:
                consistent += 1
            elif has_loss or has_normal_kw:
                contradictions += 1
        else:  # normal
            if has_normal_kw or (not has_loss and not has_enlarge and not has_severe):
                consistent += 1
            elif has_loss or has_enlarge or has_severe:
                contradictions += 1

    if total > 0:
        json_summary_score = max(0.0, (consistent - contradictions) / total)
    else:
        json_summary_score = 0.3  # no regions mentioned — mild penalty
    score += 0.30 * json_summary_score

    # ── B. DX consistency (0.25) ──
    pred_dx = parsed_json.get('diagnosis', {}).get('label', '')
    dx_mentioned = False
    for dx_label, keywords in DX_KEYWORDS.items():
        if any(kw in summary_lower for kw in keywords):
            if dx_label == pred_dx:
                dx_mentioned = True
                score += 0.25
                break
            else:
                dx_mentioned = True
                score -= 0.15  # contradicts own JSON
                break
    if not dx_mentioned:
        score += 0.10  # neutral — no explicit DX claim in summary

    # ── C. ROUGE-L with GT summary (0.10) ──
    if gt_summary and gt_summary.strip():
        gt_text = gt_summary
        # Strip header if present
        for header in ['[Diagnostic Summary]', '[Impression]']:
            if gt_text.startswith(header):
                gt_text = gt_text[len(header):].strip()
        rl = rouge_l_f1(gt_text, summary)
        score += 0.10 * rl
    else:
        score += 0.04  # no GT — give neutral baseline

    # ── D. Hallucination detection (0.35) ──
    halluc_penalty = 0.0

    # D1: Fabricated regions (not in the 5 canonical regions)
    fake_regions = ['temporal pole', 'parietal', 'occipital', 'frontal',
                    'caudate', 'putamen', 'thalamus', 'amygdala', 'cingulate',
                    'precuneus', 'cerebellum', 'insul']
    for fake in fake_regions:
        if fake in summary_lower:
            halluc_penalty += 0.10

    # D2: Mentions region as abnormal but JSON says normal
    for region in REGIONS:
        pred = assessment.get(region, {})
        if not isinstance(pred, dict):
            continue
        label = pred.get('label', 'normal')
        display = REGION_DISPLAY_NAMES.get(region, region)
        idx = summary_lower.find(display)
        if idx < 0:
            idx = summary_lower.find(region)
        if idx < 0:
            continue
        context = summary_lower[max(0, idx - 40):idx + len(display) + 60]
        if label == 'normal' and any(kw in context for kw in
                                      ['loss', 'atroph', 'severe', 'enlarg']):
            halluc_penalty += 0.10

    halluc_score = max(0.0, 1.0 - halluc_penalty)
    score += 0.35 * halluc_score

    return max(-0.5, min(1.0, score))


def compute_composite_score(parsed_json, gt_z_scores, gt_labels, gt_dx,
                            prior_labels=None, has_longitudinal=False,
                            prior_dx=None, response_text=None,
                            gt_summary=None):
    """
    Compute composite Verifier score for a candidate response (§5.5).

    Args:
        parsed_json: parsed model output JSON
        gt_z_scores: dict of region -> Z-score (current visit ground truth)
        gt_labels: dict of region -> label string (current visit)
        gt_dx: ground truth diagnosis string
        prior_labels: dict of region -> prior label string (or None for baseline)
        has_longitudinal: whether this is a follow-up sample
        prior_dx: prior visit diagnosis string (or None)
        response_text: full model response string (for summary scoring)
        gt_summary: ground-truth diagnostic summary string (or None)

    Returns:
        dict with composite score and component breakdowns
    """
    if parsed_json is None:
        return {
            'composite': -2.0,
            'anatomical': -1.0,
            'longitudinal': 0.0,
            'diagnosis': -1.0,
            'reasoning': 0.0,
            'summary': -0.5,
            'clinical_weight': 1.0,
        }

    # Anatomical scores (with region weighting)
    region_scores = {}
    assessment = parsed_json.get('anatomical_assessment', {})
    for region in REGIONS:
        pred = assessment.get(region, {})
        pred_label = pred.get('label', 'normal')
        pred_conf = pred.get('confidence', 0.5)
        z = gt_z_scores.get(region, 0.0)
        region_scores[region] = score_region(pred_label, pred_conf, z, region)

    weighted_sum = sum(region_scores[r] * REGION_WEIGHTS[r] for r in REGIONS)
    weight_total = sum(REGION_WEIGHTS.values())
    anatomical_avg = weighted_sum / weight_total

    # Longitudinal score
    pred_longi = parsed_json.get('longitudinal_comparison')
    if has_longitudinal and pred_longi:
        longi_score = score_longitudinal(
            pred_longi, gt_z_scores, prior_labels, gt_labels)
    else:
        longi_score = 0.0

    # Diagnosis score
    pred_dx = parsed_json.get('diagnosis', {}).get('label', 'MCI')
    dx_score = score_diagnosis(pred_dx, gt_dx)

    # Reasoning consistency (enhanced)
    reasoning = parsed_json.get('reasoning', {})
    reasoning_score = score_reasoning_consistency(
        reasoning, assessment, pred_longi, has_longitudinal=has_longitudinal)

    # Summary score (§5.6)
    summary_score = 0.0
    if response_text:
        summary_score = score_summary(response_text, parsed_json, gt_summary)

    # Weighted composite (summary = 0.15)
    if has_longitudinal:
        raw_score = (0.25 * anatomical_avg +
                     0.20 * longi_score +
                     0.25 * dx_score +
                     0.15 * reasoning_score +
                     0.15 * summary_score)
    else:
        raw_score = (0.35 * anatomical_avg +
                     0.35 * dx_score +
                     0.15 * reasoning_score +
                     0.15 * summary_score)

    # Clinical weighting
    if (pred_dx, gt_dx) not in DX_ADJACENT and pred_dx != gt_dx:
        clinical_weight = 2.0  # Critical miss (CN<->Dementia)
    elif pred_dx != gt_dx:
        clinical_weight = 1.5  # Adjacent miss
    else:
        clinical_weight = 1.0

    composite = raw_score * clinical_weight

    return {
        'composite': composite,
        'anatomical': anatomical_avg,
        'longitudinal': longi_score,
        'diagnosis': dx_score,
        'reasoning': reasoning_score,
        'summary': summary_score,
        'clinical_weight': clinical_weight,
        'region_scores': region_scores,
    }


def parse_json_from_response(text):
    """Extract JSON block from model response."""
    m = re.search(r'```json\s*(.*?)```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None
