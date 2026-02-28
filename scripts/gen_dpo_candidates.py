"""
Generate DPO preference pair candidates (K=4) for Stage 2 — Singleturn.

For each training sample:
  1. Generate K=4 responses with temperature sampling
  2. Save ALL 4 candidates (responses only, NO scoring)

Scoring is done separately after the verifier is finalized.

Prompt construction is IDENTICAL to evaluate_generation_singleturn /
run_full_eval_singleturn in train_ablation.py.

Supports sharding: --shard_id X --num_shards N
  or explicit range: --start_idx X --end_idx Y

Usage:
    python scripts/gen_dpo_candidates.py \
        --shard_id 0 --start_idx 0 --end_idx 333 \
        --projector_path /path/to/projector.pt \
        --lora_path /path/to/lora_adapter
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from peft import PeftModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.stage1_dataset import Stage1Dataset, IMAGE_SENTINEL, NUM_VISUAL_TOKENS
from src.model.stage1_model import LoV3DStage1Model


def build_prompt_tokens(sample, tokenizer):
    """Build prompt token IDs for generation.

    IDENTICAL to the prompt construction in:
      - evaluate_generation_singleturn() (train_ablation.py:116-133)
      - run_full_eval_singleturn()       (train_ablation.py:1083-1100)
    """
    im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    image_pad_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')

    system_msg = (
        "You are a neuroradiologist assessing a patient's brain MRI for signs of "
        "neurodegeneration. Based on the clinical information below and the attached "
        "3D T1-weighted MRI, provide your assessment."
    )

    prompt_text = sample['prompt']
    before_img, after_img = prompt_text.split(IMAGE_SENTINEL, 1)

    system_tokens = (
        [im_start_id]
        + tokenizer.encode('system\n' + system_msg, add_special_tokens=False)
        + [im_end_id]
        + tokenizer.encode('\n', add_special_tokens=False)
    )
    user_start = [im_start_id] + tokenizer.encode('user\n', add_special_tokens=False)
    before_tokens = tokenizer.encode(before_img.rstrip(), add_special_tokens=False) if before_img.strip() else []
    vision_tokens = [vision_start_id] + [image_pad_id] * NUM_VISUAL_TOKENS + [vision_end_id]
    after_tokens = tokenizer.encode(after_img, add_special_tokens=False)
    user_end = [im_end_id] + tokenizer.encode('\n', add_special_tokens=False)
    assistant_start = [im_start_id] + tokenizer.encode('assistant\n', add_special_tokens=False)

    prompt_tokens = (system_tokens + user_start + before_tokens
                     + vision_tokens + after_tokens + user_end + assistant_start)
    return prompt_tokens, image_pad_id


def generate_singleturn(model, tokenizer, volume, prompt_tokens,
                        image_pad_id, temperature, device):
    """Generate a single response with temperature sampling.

    max_new_tokens=900: same as run_full_eval_singleturn (JSON + Diagnostic Summary).
    """
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    image_token_mask = (input_ids == image_pad_id)

    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            output_ids = model.generate(
                volume, input_ids, attention_mask, image_token_mask,
                max_new_tokens=900, temperature=temperature,
            )
    text = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)
    return text


def main():
    parser = argparse.ArgumentParser(description='Generate DPO candidates (K=4, singleturn, no scoring)')
    parser.add_argument('--train_json', type=str, default='data_splits/singleturn_train.json')
    parser.add_argument('--encoder_path', type=str, required=True,
                        help='Path to Stage 0 encoder checkpoint (encoder_layer3.pt)')
    parser.add_argument('--llm_path', type=str, required=True,
                        help='Path to base LLM (e.g., Qwen2.5-14B)')
    parser.add_argument('--projector_path', type=str, required=True,
                        help='Path to Stage 1b best projector.pt')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='Path to Stage 1b best lora_adapter/')
    parser.add_argument('--output_dir', type=str, default='data_splits/dpo_candidates_singleturn')
    parser.add_argument('--K', type=int, default=4, help='Number of candidates per sample')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--shard_id', type=int, required=True)
    parser.add_argument('--start_idx', type=int, default=None, help='Start sample index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=None, help='End sample index (exclusive)')
    parser.add_argument('--index_file', type=str, default=None, help='JSON file with list of global indices')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_seq_len', type=int, default=2560)
    args = parser.parse_args()

    # Seed — different per shard for diverse candidates
    base_seed = args.seed + args.shard_id * 1000
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Shard {args.shard_id} | Device: {device}")
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {total_memory / 1e9:.1f} GB")

    # Load model
    print("\n=== Loading Model ===")
    model = LoV3DStage1Model(args.encoder_path, args.llm_path)

    proj_state = torch.load(args.projector_path, map_location='cpu', weights_only=True)
    model.projector.load_state_dict(proj_state)
    print(f"Loaded projector from {args.projector_path}")

    model.llm = PeftModel.from_pretrained(model.llm, args.lora_path)
    print(f"Loaded LoRA from {args.lora_path}")

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)

    # Load training data
    with open(args.train_json) as f:
        all_samples = json.load(f)
    print(f"Total training samples: {len(all_samples)}")

    # Build dataset for MRI loading
    dataset = Stage1Dataset(
        args.train_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len,
    )
    sample_id_to_idx = {s['sample_id']: j for j, s in enumerate(dataset.samples)}

    # Shard assignment: by index file or by range
    if args.index_file:
        with open(args.index_file) as f:
            shard_indices = json.load(f)
        print(f"Shard {args.shard_id}: {len(shard_indices)} samples from {args.index_file}\n")
    else:
        shard_indices = list(range(args.start_idx, min(args.end_idx, len(all_samples))))
        print(f"Shard {args.shard_id}: {len(shard_indices)} samples [{args.start_idx}:{args.end_idx})\n")

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{args.shard_id}.json"

    # Resume support: skip already generated samples
    results = []
    skip_count = 0
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        skip_count = len(results)
        print(f"Resuming: {skip_count} samples already done, continuing from index {skip_count}\n")

    start_time = time.time()

    for si, global_idx in enumerate(shard_indices):
        # Skip already completed
        if si < skip_count:
            continue

        sample = all_samples[global_idx]
        sample_id = sample['sample_id']

        # Get MRI volume
        dataset_idx = sample_id_to_idx.get(sample_id)
        if dataset_idx is None:
            print(f"  [WARN] {sample_id} not found in dataset, skipping")
            continue
        data_item = dataset[dataset_idx]
        volume = data_item['volume'].unsqueeze(0).to(device)

        # Build prompt tokens (identical to train_ablation.py singleturn eval)
        prompt_tokens, image_pad_id = build_prompt_tokens(sample, tokenizer)

        # Generate K candidates
        candidates = []
        for k in range(args.K):
            # Different seed per candidate for diversity
            cand_seed = base_seed + global_idx * 100 + k
            torch.manual_seed(cand_seed)
            torch.cuda.manual_seed(cand_seed)

            response_text = generate_singleturn(
                model, tokenizer, volume, prompt_tokens,
                image_pad_id, temperature=args.temperature, device=device,
            )
            candidates.append({'response': response_text})

        result = {
            'sample_id': sample_id,
            'mri_path': sample['mri_path'],
            'is_baseline': sample.get('is_baseline', True),
            'prompt': sample['prompt'],
            'gt_json': sample.get('gt_json'),
            'gt_diagnosis': sample.get('gt_diagnosis'),
            'gt_labels': sample.get('gt_labels'),
            'gt_z_scores': sample.get('gt_z_scores'),
            'candidates': candidates,
        }
        results.append(result)

        # Progress logging
        done = si - skip_count + 1
        elapsed = time.time() - start_time
        rate = elapsed / done
        remaining = len(shard_indices) - si - 1
        eta = rate * remaining
        if done % 10 == 0 or done == 1:
            print(f"  [{si+1}/{len(shard_indices)}] {sample_id} | "
                  f"K={args.K} responses generated | "
                  f"{rate:.1f}s/sample ETA={eta/60:.1f}min")

        # Checkpoint save after every sample
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=1)

    # Final save
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=1)

    elapsed = time.time() - start_time
    done = len(results) - skip_count
    print(f"\n=== Shard {args.shard_id} Complete ===")
    print(f"  Samples: {len(results)} total ({done} new)")
    print(f"  Saved to {out_path}")
    if done > 0:
        print(f"  Time: {elapsed/60:.1f} min ({elapsed/done:.1f}s/sample)")
    print(f"  GPU Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == '__main__':
    main()
