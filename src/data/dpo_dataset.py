"""
DPO Preference Pair Dataset for Stage 2.

Loads DPO candidate shards with:
  - Flip correction: swap chosen/rejected if chosen_score < rejected_score
  - GT fallback: replace chosen with GT when quality/margin too low
  - Singleturn tokenization

Each sample returns tokenized chosen + rejected + shared MRI volume.
"""

import json
import re
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.verifier import compute_composite_score, parse_json_from_response, REGIONS

IMAGE_SENTINEL = "<IMAGE>"
NUM_VISUAL_TOKENS = 512

# Region name mapping: CSV → JSON
CSV_TO_JSON = {
    'Hippocampus': 'hippocampus', 'Entorhinal': 'entorhinal',
    'Fusiform': 'fusiform', 'MidTemp': 'midtemporal', 'Ventricles': 'ventricles',
}


def _remap_labels(csv_dict):
    return {CSV_TO_JSON.get(k, k): v for k, v in csv_dict.items()}


def _score_response(response_text, gt_info):
    """Score a single response with the upgraded verifier."""
    parsed = parse_json_from_response(response_text) if response_text else None
    if not parsed:
        return -1.0
    res = compute_composite_score(
        parsed, gt_info['gt_z_scores'], gt_info['gt_labels'], gt_info['gt_dx'],
        prior_labels=gt_info.get('prior_labels'),
        has_longitudinal=False,
        prior_dx=gt_info.get('prior_dx'),
    )
    return res['composite']


class DPODataset(Dataset):
    def __init__(self, dpo_candidates_path, gt_json_path, tokenizer_path,
                 quality_threshold=0.6, margin_threshold=0.05,
                 margin_quality_threshold=0.8,
                 max_seq_len=2048, **kwargs):
        """
        Args:
            dpo_candidates_path: path to dpo_candidates_all.json (K=4 per sample)
            gt_json_path: path to stage1_train.json for GT fallback
            tokenizer_path: path to Qwen-2.5-14B
            quality_threshold: if best_score < this → GT fallback (strategy 1)
            margin_threshold: margin threshold for strategy 2
            margin_quality_threshold: if margin < margin_threshold AND
                best_score < this → GT fallback (strategy 2)
            max_seq_len: max token sequence length
        """
        self.max_seq_len = max_seq_len

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.image_pad_token_id = self.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        self.vision_start_id = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.vision_end_id = self.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        self.im_start_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')

        # Load GT data for fallback
        print(f"Loading GT data from {gt_json_path}...")
        with open(gt_json_path) as f:
            gt_data = json.load(f)
        gt_by_id = {s['sample_id']: s for s in gt_data}

        # Load candidates (K=4 per sample, pre-scored)
        print(f"Loading DPO candidates from {dpo_candidates_path}...")
        with open(dpo_candidates_path) as f:
            raw_samples = json.load(f)

        # Build preference pairs from candidates
        self.samples = []
        stats = {'total': 0, 'skipped_no_gt': 0, 'gt_fallback': 0, 'valid': 0}

        for s in raw_samples:
            stats['total'] += 1
            sid = s['sample_id']
            gt = gt_by_id.get(sid)
            if not gt:
                stats['skipped_no_gt'] += 1
                continue

            candidates = s['candidates']
            scores = [c['score'] for c in candidates]
            best_idx = int(np.argmax(scores))
            worst_idx = int(np.argmin(scores))
            chosen = candidates[best_idx]
            rejected = candidates[worst_idx]
            margin = scores[best_idx] - scores[worst_idx]

            is_gt_fallback = False
            # Strategy 1: best score too low → GT fallback
            # Strategy 2: small margin AND best score below secondary threshold → GT fallback
            if (scores[best_idx] < quality_threshold or
                    (margin < margin_threshold and scores[best_idx] < margin_quality_threshold)):
                chosen_resp = gt['response']
                is_gt_fallback = True
                stats['gt_fallback'] += 1
            else:
                chosen_resp = chosen['response']

            sample = {
                'sample_id': sid,
                'mri_path': s['mri_path'],
                'is_baseline': s.get('is_baseline', False),
                'prompt': s['prompt'],
                'chosen_response': chosen_resp,
                'rejected_response': rejected['response'],
                'is_gt_fallback': is_gt_fallback,
                'margin': margin,
            }
            self.samples.append(sample)
            stats['valid'] += 1

        # Build sample_id → index lookup for fast access
        self.id_to_idx = {s['sample_id']: i for i, s in enumerate(self.samples)}

        print(f"DPO Dataset: {stats['total']} total, {stats['skipped_no_gt']} no_gt, "
              f"{stats['gt_fallback']} GT fallback, {stats['valid']} valid pairs")
        print(f"  quality_threshold={quality_threshold}, margin_threshold={margin_threshold}, "
              f"margin_quality_threshold={margin_quality_threshold}")

    def _build_turn1_prompt_tokens(self, prompt_text):
        """Build tokens for the first user turn (with image). Same as Stage1Dataset."""
        assert IMAGE_SENTINEL in prompt_text, f"Prompt missing {IMAGE_SENTINEL}"
        before_img, after_img = prompt_text.split(IMAGE_SENTINEL, 1)

        system_msg = (
            "You are a neuroradiologist assessing a patient's brain MRI for signs of "
            "neurodegeneration. Based on the clinical information below and the attached "
            "3D T1-weighted MRI, provide your assessment."
        )

        system_tokens = (
            [self.im_start_id]
            + self.tokenizer.encode('system\n' + system_msg, add_special_tokens=False)
            + [self.im_end_id]
            + self.tokenizer.encode('\n', add_special_tokens=False)
        )
        user_start = (
            [self.im_start_id]
            + self.tokenizer.encode('user\n', add_special_tokens=False)
        )
        before_tokens = (
            self.tokenizer.encode(before_img.rstrip(), add_special_tokens=False)
            if before_img.strip() else []
        )
        vision_tokens = (
            [self.vision_start_id]
            + [self.image_pad_token_id] * NUM_VISUAL_TOKENS
            + [self.vision_end_id]
        )
        after_tokens = self.tokenizer.encode(after_img, add_special_tokens=False)
        user_end = [self.im_end_id] + self.tokenizer.encode('\n', add_special_tokens=False)
        assistant_start = (
            [self.im_start_id]
            + self.tokenizer.encode('assistant\n', add_special_tokens=False)
        )

        return (system_tokens + user_start + before_tokens + vision_tokens
                + after_tokens + user_end + assistant_start)

    def _tokenize_response(self, prompt_text, response_text):
        """Tokenize prompt + response (singleturn)."""
        prompt1_tokens = self._build_turn1_prompt_tokens(prompt_text)
        response1_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)

        response_end = [self.im_end_id, self.tokenizer.eos_token_id]
        all_tokens = prompt1_tokens + response1_tokens + response_end

        if len(all_tokens) > self.max_seq_len:
            max_response = self.max_seq_len - len(prompt1_tokens) - len(response_end)
            if max_response > 0:
                response1_tokens = response1_tokens[:max_response]
                all_tokens = prompt1_tokens + response1_tokens + response_end
            else:
                all_tokens = all_tokens[:self.max_seq_len]

        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        labels = torch.full((len(all_tokens),), -100, dtype=torch.long)
        r1_start = len(prompt1_tokens)
        labels[r1_start:] = input_ids[r1_start:]

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        image_token_mask = (input_ids == self.image_pad_token_id)

        return input_ids, attention_mask, labels, image_token_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load MRI volume
        volume = np.load(s['mri_path']).astype(np.float32)
        volume = torch.from_numpy(volume[np.newaxis])

        # Tokenize chosen
        c_ids, c_mask, c_labels, c_img_mask = self._tokenize_response(
            s['prompt'], s['chosen_response'],
        )

        # Tokenize rejected
        r_ids, r_mask, r_labels, r_img_mask = self._tokenize_response(
            s['prompt'], s['rejected_response'],
        )

        return {
            'volume': volume,
            'chosen_input_ids': c_ids,
            'chosen_attention_mask': c_mask,
            'chosen_labels': c_labels,
            'chosen_image_token_mask': c_img_mask,
            'rejected_input_ids': r_ids,
            'rejected_attention_mask': r_mask,
            'rejected_labels': r_labels,
            'rejected_image_token_mask': r_img_mask,
            'sample_id': s['sample_id'],
            'is_gt_fallback': s['is_gt_fallback'],
            'margin': s['margin'],
        }


def _pad_sequence(tensors, pad_value):
    """Right-pad a list of 1-D tensors to the same length."""
    max_len = max(t.shape[0] for t in tensors)
    padded = []
    for t in tensors:
        pad_len = max_len - t.shape[0]
        if pad_len > 0:
            padded.append(torch.cat([t, torch.full((pad_len,), pad_value, dtype=t.dtype)]))
        else:
            padded.append(t)
    return torch.stack(padded)


def dpo_collate_fn(batch):
    """Collate DPO batch with independent padding for chosen/rejected."""
    volumes = torch.stack([b['volume'] for b in batch])

    # Pad chosen sequences
    chosen_input_ids = _pad_sequence([b['chosen_input_ids'] for b in batch], 0)
    chosen_labels = _pad_sequence([b['chosen_labels'] for b in batch], -100)
    chosen_attention_mask = _pad_sequence([b['chosen_attention_mask'] for b in batch], 0)
    chosen_image_token_mask = _pad_sequence(
        [b['chosen_image_token_mask'] for b in batch], False)

    # Pad rejected sequences
    rejected_input_ids = _pad_sequence([b['rejected_input_ids'] for b in batch], 0)
    rejected_labels = _pad_sequence([b['rejected_labels'] for b in batch], -100)
    rejected_attention_mask = _pad_sequence([b['rejected_attention_mask'] for b in batch], 0)
    rejected_image_token_mask = _pad_sequence(
        [b['rejected_image_token_mask'] for b in batch], False)

    return {
        'volume': volumes,
        'chosen_input_ids': chosen_input_ids,
        'chosen_attention_mask': chosen_attention_mask,
        'chosen_labels': chosen_labels,
        'chosen_image_token_mask': chosen_image_token_mask,
        'rejected_input_ids': rejected_input_ids,
        'rejected_attention_mask': rejected_attention_mask,
        'rejected_labels': rejected_labels,
        'rejected_image_token_mask': rejected_image_token_mask,
        'sample_ids': [b['sample_id'] for b in batch],
        'is_gt_fallback': [b['is_gt_fallback'] for b in batch],
        'margins': [b['margin'] for b in batch],
    }
