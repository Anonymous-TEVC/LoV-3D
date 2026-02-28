#!/usr/bin/env python3
"""
Standalone full eval (35+ metrics) for Stage 1b checkpoint.

Usage:
    python scripts/eval_1b_full.py \
        --projector_path /path/to/projector.pt \
        --lora_path /path/to/lora_adapter \
        --output_dir results/1b_d00ctrl_eval
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.stage1_model import LoV3DStage1Model
from src.data.stage1_dataset import Stage1Dataset
from src.train.train_ablation import run_full_eval_singleturn


def main():
    parser = argparse.ArgumentParser(description='Full eval (35+ metrics) for Stage 1b')
    parser.add_argument('--encoder_path', type=str, required=True,
                        help='Path to Stage 0 encoder checkpoint (encoder_layer3.pt)')
    parser.add_argument('--llm_path', type=str, required=True,
                        help='Path to base LLM (e.g., Qwen2.5-14B)')
    parser.add_argument('--projector_path', type=str, required=True)
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--test_json', type=str,
                        default='data_splits/singleturn_test.json')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda')

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load Model ──
    print("\n=== Loading Model ===")
    model = LoV3DStage1Model(args.encoder_path, args.llm_path)

    proj_weights = torch.load(args.projector_path, map_location='cpu', weights_only=True)
    model.projector.load_state_dict(proj_weights)
    print(f"Loaded projector from {args.projector_path}")

    model.llm = PeftModel.from_pretrained(model.llm, args.lora_path)
    model.llm = model.llm.merge_and_unload()
    print(f"Loaded and merged LoRA from {args.lora_path}")

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)

    # ── Load Test Data ──
    with open(args.test_json) as f:
        test_samples = json.load(f)
    test_dataset = Stage1Dataset(
        args.test_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len
    )
    print(f"\nTest samples: {len(test_samples)}")

    # ── Run Full Eval ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running full eval (35+ metrics) on {len(test_samples)} test samples")
    print(f"Checkpoint: {args.projector_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    run_full_eval_singleturn(
        model, tokenizer, test_samples, test_dataset, device,
        output_dir, output_name='full_eval_singleturn.json'
    )

    print(f"\nGPU Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("Done.")


if __name__ == '__main__':
    main()
