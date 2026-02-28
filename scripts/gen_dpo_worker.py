"""
Dynamic DPO candidate generation worker.

Each worker loops through chunks and claims them via atomic mkdir.
Multiple workers can run in parallel — fast GPUs process more chunks.

Workflow:
  1. Load model once
  2. Loop: try to claim next unclaimed chunk via mkdir(lock_dir/chunk_XX)
  3. Process claimed chunk (generate K=4 responses per sample)
  4. Save results to chunk_XX.json
  5. Repeat until all chunks done

Usage:
    python scripts/gen_dpo_worker.py --worker_id 0 --chunk_size 50
"""

import sys
import os
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
    """Build prompt token IDs — identical to train_ablation.py singleturn eval."""
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
    """Generate one response with temperature sampling. max_new_tokens=900."""
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    image_token_mask = (input_ids == image_pad_id)

    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            output_ids = model.generate(
                volume, input_ids, attention_mask, image_token_mask,
                max_new_tokens=900, temperature=temperature,
            )
    return tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)


def try_claim_chunk(lock_dir, chunk_id):
    """Atomically claim a chunk via mkdir. Returns True if claimed."""
    lock_path = lock_dir / f"chunk_{chunk_id:04d}"
    try:
        lock_path.mkdir(parents=False, exist_ok=False)
        return True
    except FileExistsError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Dynamic DPO candidate worker')
    parser.add_argument('--worker_id', type=int, required=True)
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
    parser.add_argument('--chunk_size', type=int, default=50, help='Samples per chunk')
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_seq_len', type=int, default=2560)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Worker {args.worker_id} | Device: {device}")
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {total_memory / 1e9:.1f} GB")

    # Load model (one-time cost)
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
    total = len(all_samples)
    print(f"Total training samples: {total}")

    # Build dataset for MRI loading
    dataset = Stage1Dataset(
        args.train_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len,
    )
    sample_id_to_idx = {s['sample_id']: j for j, s in enumerate(dataset.samples)}

    # Chunk setup
    num_chunks = (total + args.chunk_size - 1) // args.chunk_size
    out_dir = Path(args.output_dir)
    lock_dir = out_dir / "locks"
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nChunk config: {num_chunks} chunks of {args.chunk_size} samples")
    print(f"Output: {out_dir}/chunk_XXXX.json")
    print(f"Locks:  {lock_dir}/chunk_XXXX/")

    base_seed = args.seed
    total_processed = 0
    worker_start = time.time()

    # Dynamic work loop: claim and process chunks
    chunk_order = list(range(num_chunks))
    # Stagger starting point to reduce contention
    start_offset = args.worker_id % num_chunks
    chunk_order = chunk_order[start_offset:] + chunk_order[:start_offset]

    for chunk_id in chunk_order:
        # Try to claim this chunk
        if not try_claim_chunk(lock_dir, chunk_id):
            continue  # Already claimed by another worker

        # Claimed! Process this chunk
        start_idx = chunk_id * args.chunk_size
        end_idx = min(start_idx + args.chunk_size, total)
        chunk_out = out_dir / f"chunk_{chunk_id:04d}.json"

        # Check if output already exists (from a previous run)
        if chunk_out.exists():
            try:
                with open(chunk_out) as f:
                    existing = json.load(f)
                if len(existing) == (end_idx - start_idx):
                    print(f"  Chunk {chunk_id}: already complete ({len(existing)} samples), skipping")
                    total_processed += len(existing)
                    continue
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupt file, redo

        print(f"\n=== Worker {args.worker_id}: Chunk {chunk_id} [{start_idx}:{end_idx}) ===")

        results = []
        chunk_start = time.time()

        for si in range(start_idx, end_idx):
            sample = all_samples[si]
            sample_id = sample['sample_id']

            dataset_idx = sample_id_to_idx.get(sample_id)
            if dataset_idx is None:
                print(f"  [WARN] {sample_id} not found, skipping")
                continue
            data_item = dataset[dataset_idx]
            volume = data_item['volume'].unsqueeze(0).to(device)

            prompt_tokens, image_pad_id = build_prompt_tokens(sample, tokenizer)

            # Generate K candidates
            candidates = []
            for k in range(args.K):
                cand_seed = base_seed + si * 100 + k
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

            # Per-sample progress
            done_in_chunk = len(results)
            elapsed_chunk = time.time() - chunk_start
            rate = elapsed_chunk / done_in_chunk
            remaining = (end_idx - start_idx) - done_in_chunk
            eta_chunk = rate * remaining
            if done_in_chunk % 5 == 0 or done_in_chunk == 1:
                print(f"  [{done_in_chunk}/{end_idx - start_idx}] {sample_id} | "
                      f"{rate:.1f}s/sample ETA_chunk={eta_chunk/60:.1f}min")

            # Save after every sample (crash-safe)
            with open(chunk_out, 'w') as f:
                json.dump(results, f, indent=1)

        total_processed += len(results)
        elapsed_chunk = time.time() - chunk_start
        print(f"  Chunk {chunk_id} done: {len(results)} samples in {elapsed_chunk/60:.1f}min")

    # Summary
    elapsed_total = time.time() - worker_start
    print(f"\n=== Worker {args.worker_id} Finished ===")
    print(f"  Total processed: {total_processed} samples")
    print(f"  Wall time: {elapsed_total/60:.1f} min")
    if total_processed > 0:
        print(f"  Avg: {elapsed_total/total_processed:.1f}s/sample")
    print(f"  GPU Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == '__main__':
    main()
