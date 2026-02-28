#!/usr/bin/env python3
"""
Stage 2 DPO — Eval Only.

Loads a Stage 2 DPO checkpoint (projector + LoRA adapter) and runs the
full 479-sample test evaluation, producing the same detailed metrics as
train_stage2.py final eval.

Usage:
    python -m src.train.eval_stage2 \
        --checkpoint_dir /mnt/.../stage2/b01_a10_pt/epoch_1 \
        --experiment_name b01_a10_pt_e1
"""

import os, sys, json, argparse
import torch
import numpy as np
from pathlib import Path
from peft import PeftModel, LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoTokenizer

from src.model.stage1_model import LoV3DStage1Model
from src.data.stage1_dataset import Stage1Dataset
from src.train.train_ablation import run_full_eval_singleturn


def main():
    parser = argparse.ArgumentParser(description='Stage 2 DPO — Eval Only')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory with projector.pt + lora_adapter/')
    parser.add_argument('--experiment_name', type=str, required=True)

    # Model paths
    parser.add_argument('--encoder_path', type=str, required=True,
                        help='Path to Stage 0 encoder checkpoint (encoder_layer3.pt)')
    parser.add_argument('--llm_path', type=str, required=True,
                        help='Path to base LLM (e.g., Qwen2.5-14B)')
    parser.add_argument('--s1b_projector', type=str, required=True,
                        help='Stage 1b best projector.pt')
    parser.add_argument('--s1b_lora', type=str, required=True,
                        help='Stage 1b best lora_adapter/')

    # LoRA config (must match training)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)

    # Data
    parser.add_argument('--test_json', type=str, default='data_splits/stage1_test.json')
    parser.add_argument('--max_seq_len', type=int, default=2048)

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda')
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    ckpt_dir = Path(args.checkpoint_dir)
    print(f"\n{'='*60}")
    print(f"Stage 2 Eval: {args.experiment_name}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"{'='*60}")

    # ─── Load Model (same flow as train_stage2.py) ─────────────────
    print("\n=== Loading Model ===")
    model = LoV3DStage1Model(args.encoder_path, args.llm_path)

    # Load Stage 1b projector
    proj_s1b = torch.load(args.s1b_projector, map_location='cpu', weights_only=True)
    model.projector.load_state_dict(proj_s1b)
    print(f"Loaded Stage 1b projector")

    # Load Stage 1b LoRA → merge into base
    model.llm = PeftModel.from_pretrained(model.llm, args.s1b_lora)
    model.llm = model.llm.merge_and_unload()
    print("Stage 1b LoRA merged into base")

    # Apply DPO LoRA structure (same config as training)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,  # no dropout at eval
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora_config)

    # Load DPO checkpoint weights
    proj_dpo = torch.load(ckpt_dir / 'projector.pt', map_location='cpu', weights_only=True)
    model.projector.load_state_dict(proj_dpo)
    print(f"Loaded DPO projector from {ckpt_dir / 'projector.pt'}")

    adapter_path = ckpt_dir / 'lora_adapter'
    safetensors_file = adapter_path / 'adapter_model.safetensors'
    bin_file = adapter_path / 'adapter_model.bin'
    if safetensors_file.exists():
        from safetensors.torch import load_file
        adapter_weights = load_file(str(safetensors_file))
    elif bin_file.exists():
        adapter_weights = torch.load(bin_file, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")
    set_peft_model_state_dict(model.llm, adapter_weights)
    print(f"Loaded DPO LoRA from {adapter_path}")

    model.to(device)
    model.eval()

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)

    # ─── Load Test Data ────────────────────────────────────────────
    with open(args.test_json) as f:
        test_samples = json.load(f)
    test_dataset = Stage1Dataset(
        args.test_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len,
    )
    print(f"Test samples: {len(test_samples)}")

    # ─── Run Full Eval ─────────────────────────────────────────────
    print(f"\n=== Full Eval ({len(test_samples)} samples) ===\n")
    run_full_eval_singleturn(model, tokenizer, test_samples, test_dataset, device, ckpt_dir)

    print(f"\n=== Done ===")
    print(f"Experiment: {args.experiment_name}")
    print(f"GPU Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == '__main__':
    main()
