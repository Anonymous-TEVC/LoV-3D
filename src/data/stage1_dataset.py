"""
Stage 1 Dataset: Loads MRI volumes + tokenized prompts/responses for VLM training.

Single-turn format:
  2 turns (JSON assessment + clinical report)
  Labels mask prompt tokens so generation can't see future info.

Each sample returns:
  - volume: (1, 128, 128, 128) float32 tensor
  - input_ids: (seq_len,) long tensor
  - attention_mask: (seq_len,) long tensor
  - labels: (seq_len,) long tensor (-100 for prompt tokens)
  - image_token_mask: (seq_len,) bool tensor (True at <|image_pad|> positions)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from monai import transforms as T
from transformers import AutoTokenizer


IMAGE_SENTINEL = "<IMAGE>"
NUM_VISUAL_TOKENS = 512


class Stage1Dataset(Dataset):
    def __init__(self, json_path, tokenizer_path, augment=False, max_seq_len=2048):
        with open(json_path) as f:
            self.samples = json.load(f)

        self.max_seq_len = max_seq_len
        self.augment = augment

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

        if augment:
            self.transform = T.Compose([
                T.RandFlip(spatial_axis=0, prob=0.5),
                T.RandAffine(
                    rotate_range=[np.radians(5)] * 3,
                    prob=0.5,
                    padding_mode='zeros',
                ),
                T.RandShiftIntensity(offsets=0.05, prob=0.5),
            ])
        else:
            self.transform = None

    def _build_turn1_prompt_tokens(self, prompt_text):
        """Build tokens for the first user turn (with image)."""
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
        before_tokens = self.tokenizer.encode(before_img.rstrip(), add_special_tokens=False) if before_img.strip() else []
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

        prompt_tokens = (
            system_tokens + user_start + before_tokens + vision_tokens
            + after_tokens + user_end + assistant_start
        )
        return prompt_tokens

    def _build_text_turn_tokens(self, prompt_text, is_user=True):
        """Build tokens for a text-only user or assistant turn (no image)."""
        role = 'user' if is_user else 'assistant'
        tokens = (
            [self.im_start_id]
            + self.tokenizer.encode(role + '\n' + prompt_text, add_special_tokens=False)
            + [self.im_end_id]
            + self.tokenizer.encode('\n', add_special_tokens=False)
        )
        return tokens

    def _tokenize_sample(self, prompt_text, response_text,
                         prompt_report=None, response_report=None):
        """
        Tokenize prompt + response (singleturn format).

        Format (2 turns):
            [SYS] [USER1: prompt] [ASST1: json] [USER_R: report_prompt] [ASST_R: report] [EOS]
            Labels: [-100 for prompts] [json tokens] [-100 for report_prompt] [report tokens]
        """
        prompt1_tokens = self._build_turn1_prompt_tokens(prompt_text)
        response1_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)

        # Build assistant turn boundary tokens
        asst_end = [self.im_end_id] + self.tokenizer.encode('\n', add_special_tokens=False)
        asst_start = (
            [self.im_start_id]
            + self.tokenizer.encode('assistant\n', add_special_tokens=False)
        )
        final_end = [self.im_end_id, self.tokenizer.eos_token_id]

        # Collect all segments: list of (tokens, is_response) tuples
        segments = [(prompt1_tokens, False), (response1_tokens, True)]

        if prompt_report is not None and response_report is not None:
            # Report turn
            user_r_tokens = self._build_text_turn_tokens(prompt_report, is_user=True)
            response_r_tokens = self.tokenizer.encode(response_report, add_special_tokens=False)
            interlude_r = asst_end + user_r_tokens + asst_start
            segments.append((interlude_r, False))
            segments.append((response_r_tokens, True))

        # Assemble all tokens
        all_tokens = []
        for seg_tokens, _ in segments:
            all_tokens.extend(seg_tokens)
        all_tokens.extend(final_end)

        # Truncate if needed: try to truncate last response first
        if len(all_tokens) > self.max_seq_len:
            # Find the last response segment and truncate it
            total = sum(len(s) for s, _ in segments) + len(final_end)
            overflow = total - self.max_seq_len

            # Walk segments in reverse, truncating response segments
            for i in range(len(segments) - 1, -1, -1):
                seg_tokens, is_resp = segments[i]
                if is_resp and overflow > 0:
                    max_keep = len(seg_tokens) - overflow
                    if max_keep >= 50:
                        segments[i] = (seg_tokens[:max_keep], True)
                        overflow = 0
                        break
                    else:
                        # Drop this response and its preceding interlude
                        overflow -= len(seg_tokens)
                        segments[i] = ([], True)
                        if i > 0:
                            prev_tokens, prev_is_resp = segments[i - 1]
                            if not prev_is_resp and i > 1:
                                overflow -= len(prev_tokens)
                                segments[i - 1] = ([], False)

            all_tokens = []
            for seg_tokens, _ in segments:
                all_tokens.extend(seg_tokens)
            all_tokens.extend(final_end)
            all_tokens = all_tokens[:self.max_seq_len]

        # Build labels: mask prompt segments, keep response segments
        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        labels = torch.full((len(all_tokens),), -100, dtype=torch.long)

        pos = 0
        for seg_tokens, is_resp in segments:
            seg_len = len(seg_tokens)
            if is_resp and seg_len > 0:
                end = min(pos + seg_len, len(all_tokens))
                labels[pos:end] = input_ids[pos:end]
            pos += seg_len

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        image_token_mask = (input_ids == self.image_pad_token_id)

        return input_ids, attention_mask, labels, image_token_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        volume = np.load(sample['mri_path']).astype(np.float32)
        volume = volume[np.newaxis]
        volume = torch.from_numpy(volume)

        if self.transform is not None:
            volume = self.transform(volume)

        input_ids, attention_mask, labels, image_token_mask = self._tokenize_sample(
            sample['prompt'], sample['response'],
            sample.get('prompt_report'), sample.get('response_report'),
        )

        return {
            'volume': volume,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'image_token_mask': image_token_mask,
            'sample_id': sample['sample_id'],
        }


def collate_fn(batch):
    """Collate function with right-padding for variable-length sequences."""
    max_len = max(b['input_ids'].shape[0] for b in batch)

    padded_input_ids = []
    padded_labels = []
    padded_attention_mask = []
    padded_image_mask = []
    volumes = []
    sample_ids = []

    for b in batch:
        seq_len = b['input_ids'].shape[0]
        pad_len = max_len - seq_len

        if pad_len > 0:
            padded_input_ids.append(
                torch.cat([b['input_ids'], torch.zeros(pad_len, dtype=torch.long)])
            )
            padded_labels.append(
                torch.cat([b['labels'], torch.full((pad_len,), -100, dtype=torch.long)])
            )
            padded_attention_mask.append(
                torch.cat([b['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
            )
            padded_image_mask.append(
                torch.cat([b['image_token_mask'], torch.zeros(pad_len, dtype=torch.bool)])
            )
        else:
            padded_input_ids.append(b['input_ids'])
            padded_labels.append(b['labels'])
            padded_attention_mask.append(b['attention_mask'])
            padded_image_mask.append(b['image_token_mask'])

        volumes.append(b['volume'])
        sample_ids.append(b['sample_id'])

    result = {
        'volume': torch.stack(volumes),
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'labels': torch.stack(padded_labels),
        'image_token_mask': torch.stack(padded_image_mask),
        'sample_ids': sample_ids,
    }

    if 'has_dx_change' in batch[0]:
        result['has_dx_change'] = torch.tensor(
            [b.get('has_dx_change', 0) for b in batch], dtype=torch.long)
        result['n_region_changes'] = torch.tensor(
            [b.get('n_region_changes', 0) for b in batch], dtype=torch.long)

    return result
