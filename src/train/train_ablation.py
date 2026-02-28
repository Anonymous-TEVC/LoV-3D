"""
Ablation study: Combined Stage 1a → 1b training.

Runs both phases in a single SLURM job:
  Phase 1 (Stage 1a): Projector alignment — encoder frozen, LLM frozen
  Phase 2 (Stage 1b): Projector + LoRA SFT

Does NOT modify original train_stage1a.py or train_stage1b.py.
Imports evaluation functions from train_stage3.py (DX-changed, Region-changed tracking).

Single-turn generation: all samples produce a single JSON + report response.

Usage:
    python src/train/train_ablation.py --checkpoint_dir /mnt/.../ablation/baseline
"""

import os
import re
import sys
import json
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast

from peft import get_peft_model, LoraConfig, PeftModel, set_peft_model_state_dict
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.stage3_dataset import Stage3Dataset
from src.data.stage1_dataset import (
    Stage1Dataset, collate_fn, IMAGE_SENTINEL, NUM_VISUAL_TOKENS,
)
from src.model.stage1_model import LoV3DStage1Model
from src.train.train_stage3 import (
    evaluate_generation, print_gen_metrics, evaluate_sft,
    get_cosine_schedule_with_warmup, parse_json_from_response,
    _extract_prior_info,
)


def evaluate_generation_singleturn(model, tokenizer, test_samples, dataset, device,
                                    num_eval=100):
    """Single-turn generation eval for epoch-level monitoring.

    All samples produce a single JSON response (max_new_tokens=400).
    GT comes from sample['gt_json'] directly.
    """
    model.eval()
    gc_was_enabled = getattr(model.llm, 'is_gradient_checkpointing', False)
    if gc_was_enabled:
        model.llm.gradient_checkpointing_disable()

    REGIONS_MAP = {
        'Hippocampus': 'hippocampus', 'Entorhinal': 'entorhinal',
        'Fusiform': 'fusiform', 'MidTemp': 'midtemporal', 'Ventricles': 'ventricles'
    }
    REGION_KEYS = list(REGIONS_MAP.values())

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(test_samples))[:num_eval]
    samples = [test_samples[i] for i in indices]

    # Counters
    json_valid = 0
    dx_correct = dx_total = 0
    dx_per_class = {c: {'correct': 0, 'total': 0} for c in ['CN', 'MCI', 'Dementia']}
    region_correct = region_total = 0
    region_per_name = {r: {'correct': 0, 'total': 0} for r in REGION_KEYS}
    dx_changed_correct = dx_changed_total = 0
    dx_unchanged_correct = dx_unchanged_total = 0
    region_changed_correct = region_changed_total = 0
    region_unchanged_correct = region_unchanged_total = 0
    longi_dir_correct = longi_dir_total = 0
    longi_dir_per_region = {r: {'correct': 0, 'total': 0} for r in REGION_KEYS}

    system_msg = (
        "You are a neuroradiologist assessing a patient's brain MRI for signs of "
        "neurodegeneration. Based on the clinical information below and the attached "
        "3D T1-weighted MRI, provide your assessment."
    )
    im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    image_pad_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')

    sample_id_to_idx = {s['sample_id']: j for j, s in enumerate(dataset.samples)}

    eval_start_time = time.time()
    for i, sample in enumerate(samples):
        sample_id = sample['sample_id']
        dataset_idx = sample_id_to_idx.get(sample_id)
        if dataset_idx is None:
            continue

        data_item = dataset[dataset_idx]
        volume = data_item['volume'].unsqueeze(0).to(device)

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
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        image_token_mask = (input_ids == image_pad_id)

        # Single generation
        with torch.no_grad():
            with autocast('cuda', dtype=torch.bfloat16):
                output_ids = model.generate(
                    volume, input_ids, attention_mask, image_token_mask,
                    max_new_tokens=400
                )
        generated_text = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)
        parsed_json = parse_json_from_response(generated_text)

        # GT from sample
        gt_json = sample.get('gt_json')
        gt_dx = sample.get('gt_diagnosis', gt_json['diagnosis']['label'] if gt_json else None)

        # Extract prior info for changed/unchanged tracking
        prior_dx, prior_regions = _extract_prior_info(prompt_text)

        if parsed_json is not None:
            json_valid += 1

            pred_dx = parsed_json.get('diagnosis', {}).get('label')
            if pred_dx and gt_dx:
                dx_total += 1
                dx_per_class[gt_dx]['total'] += 1
                if pred_dx == gt_dx:
                    dx_correct += 1
                    dx_per_class[gt_dx]['correct'] += 1
                # DX changed vs unchanged
                if prior_dx:
                    if prior_dx != gt_dx:
                        dx_changed_total += 1
                        if pred_dx == gt_dx:
                            dx_changed_correct += 1
                    else:
                        dx_unchanged_total += 1
                        if pred_dx == gt_dx:
                            dx_unchanged_correct += 1

            if 'anatomical_assessment' in parsed_json and gt_json:
                for csv_region, json_key in REGIONS_MAP.items():
                    pred_label = parsed_json.get('anatomical_assessment', {}).get(json_key, {}).get('label')
                    gt_label = gt_json.get('anatomical_assessment', {}).get(json_key, {}).get('label')
                    if pred_label and gt_label:
                        region_total += 1
                        region_per_name[json_key]['total'] += 1
                        if pred_label == gt_label:
                            region_correct += 1
                            region_per_name[json_key]['correct'] += 1
                        # Region changed vs unchanged
                        prior_r = prior_regions.get(json_key)
                        if prior_r:
                            if prior_r != gt_label:
                                region_changed_total += 1
                                if pred_label == gt_label:
                                    region_changed_correct += 1
                            else:
                                region_unchanged_total += 1
                                if pred_label == gt_label:
                                    region_unchanged_correct += 1

            # Longitudinal direction (singleturn: all in one JSON)
            pred_longi = parsed_json.get('longitudinal_comparison')
            gt_longi = gt_json.get('longitudinal_comparison') if gt_json else None
            if pred_longi and gt_longi:
                for json_key in REGION_KEYS:
                    pred_r = pred_longi.get(json_key, {})
                    gt_r = gt_longi.get(json_key, {})
                    if isinstance(pred_r, dict) and isinstance(gt_r, dict):
                        if 'direction' in pred_r and 'direction' in gt_r:
                            longi_dir_total += 1
                            longi_dir_per_region[json_key]['total'] += 1
                            if pred_r['direction'] == gt_r['direction']:
                                longi_dir_correct += 1
                                longi_dir_per_region[json_key]['correct'] += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - eval_start_time
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate if rate > 0 else 0
            dir_str = f"Dir:{longi_dir_correct}/{longi_dir_total}" if longi_dir_total > 0 else ""
            dxc_str = f"DX-chg:{dx_changed_correct}/{dx_changed_total}" if dx_changed_total > 0 else ""
            print(f"    Gen eval [{i+1}/{len(samples)}] "
                  f"JSON:{json_valid}/{i+1} DX:{dx_correct}/{dx_total} "
                  f"Reg:{region_correct}/{region_total} {dir_str} {dxc_str} "
                  f"Rate:{rate:.2f}s/sample ETA:{eta/60:.1f}min", flush=True)

    total = len(samples)
    safe_div = lambda a, b: 100 * a / b if b > 0 else 0.0

    results = {
        'num_eval': total,
        'json_valid': safe_div(json_valid, total),
        'dx_accuracy': safe_div(dx_correct, dx_total),
        'dx_correct': dx_correct,
        'dx_total': dx_total,
        'dx_per_class': {c: {'accuracy': safe_div(v['correct'], v['total']),
                              'correct': v['correct'], 'total': v['total']}
                          for c, v in dx_per_class.items()},
        'dx_changed_accuracy': safe_div(dx_changed_correct, dx_changed_total),
        'dx_changed_correct': dx_changed_correct,
        'dx_changed_total': dx_changed_total,
        'dx_unchanged_accuracy': safe_div(dx_unchanged_correct, dx_unchanged_total),
        'dx_unchanged_correct': dx_unchanged_correct,
        'dx_unchanged_total': dx_unchanged_total,
        'region_accuracy': safe_div(region_correct, region_total),
        'region_correct': region_correct,
        'region_total': region_total,
        'region_per_name': {r: {'accuracy': safe_div(v['correct'], v['total']),
                                 'correct': v['correct'], 'total': v['total']}
                             for r, v in region_per_name.items()},
        'region_changed_accuracy': safe_div(region_changed_correct, region_changed_total),
        'region_changed_correct': region_changed_correct,
        'region_changed_total': region_changed_total,
        'region_unchanged_accuracy': safe_div(region_unchanged_correct, region_unchanged_total),
        'region_unchanged_correct': region_unchanged_correct,
        'region_unchanged_total': region_unchanged_total,
        'longi_direction_accuracy': safe_div(longi_dir_correct, longi_dir_total),
        'longi_direction_correct': longi_dir_correct,
        'longi_direction_total': longi_dir_total,
        'longi_direction_per_region': {r: {'accuracy': safe_div(v['correct'], v['total']),
                                            'correct': v['correct'], 'total': v['total']}
                                        for r, v in longi_dir_per_region.items()},
    }

    if gc_was_enabled:
        model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description='Ablation: Stage 1a+1b training')

    # === Paths ===
    parser.add_argument('--train_json', type=str, default='data_splits/stage1_train.json')
    parser.add_argument('--val_json', type=str, default='data_splits/stage1_val.json')
    parser.add_argument('--test_json', type=str, default='data_splits/stage1_test.json')
    parser.add_argument('--encoder_path', type=str, required=True,
                        help='Path to Stage 0 encoder checkpoint (encoder_layer3.pt)')
    parser.add_argument('--llm_path', type=str, required=True,
                        help='Path to base LLM (e.g., Qwen2.5-14B)')
    parser.add_argument('--checkpoint_dir', type=str, required=True)

    # === Stage 1a config ===
    parser.add_argument('--epochs_1a', type=int, default=3)
    parser.add_argument('--lr_1a', type=float, default=1e-3)
    parser.add_argument('--skip_1a', action='store_true',
                        help='Skip Stage 1a, load projector from --projector_path')
    parser.add_argument('--projector_path', type=str, default=None,
                        help='Pre-trained projector to load (skips Phase 1)')

    # === Stage 1b config ===
    parser.add_argument('--epochs_1b', type=int, default=5)
    parser.add_argument('--lr_projector', type=float, default=1e-4)
    parser.add_argument('--lr_lora', type=float, default=2e-4)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # === Change detection loss weighting ===
    parser.add_argument('--dx_change_weight', type=float, default=0.0,
                        help='Bonus loss weight for DX-changed samples (0=off, suggested 2.0)')
    parser.add_argument('--region_change_weight', type=float, default=0.0,
                        help='Bonus loss weight per region change (0=off, suggested 0.5)')

    # === Common training config ===
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--gen_eval_samples', type=int, default=100)
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Limit epoch-end val eval to N samples (None=full val set)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--baseline_eval_path', type=str, default=None,
                        help='Path to pre-computed baseline_eval.json (skip redundant baseline gen eval)')
    parser.add_argument('--baseline_only', action='store_true',
                        help='Run only pre-training baseline eval, save results, then exit')
    parser.add_argument('--skip_1b', action='store_true',
                        help='Skip Stage 1b. After Stage 1a, run full test eval and exit.')
    parser.add_argument('--skip_baseline_eval', action='store_true',
                        help='Skip pre-training baseline generation eval (save ~1-2h)')
    parser.add_argument('--skip_epoch_gen_eval', action='store_true',
                        help='Skip per-epoch generation eval (only run full eval after training)')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Resume 1b training from this epoch (load epoch_N checkpoint)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {total_memory / 1e9:.1f} GB")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = vars(args).copy()
    with open(ckpt_dir / 'ablation_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Print experiment summary
    print(f"\n{'='*60}")
    print(f"Ablation Experiment: {ckpt_dir.name}")
    mode_parts = []
    if args.dx_change_weight > 0 or args.region_change_weight > 0:
        mode_parts.append(f"chg_wt(dx={args.dx_change_weight}, reg={args.region_change_weight})")
    if not mode_parts:
        mode_parts.append("baseline")
    print(f"Mode: {', '.join(mode_parts)}")
    print(f"Phase 1: {args.epochs_1a} epochs (LR={args.lr_1a})")
    print(f"Phase 2: {args.epochs_1b} epochs (LR proj={args.lr_projector}, lora={args.lr_lora})")
    print(f"{'='*60}")

    # =====================================================================
    # Phase 1: Stage 1a — Projector Alignment
    # =====================================================================
    print(f"\n{'='*60}")
    print("Phase 1: Stage 1a — Projector Alignment")
    print(f"  Encoder: frozen, LLM: frozen, Projector: trainable")
    print(f"{'='*60}")

    # Build model with Stage 0 encoder
    model = LoV3DStage1Model(args.encoder_path, args.llm_path)

    # Freeze encoder and LLM
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()
    for p in model.llm.parameters():
        p.requires_grad = False

    model.to(device)

    # Gradient checkpointing (needed for backward even with frozen LLM)
    model.llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if args.skip_1a and args.projector_path:
        # Skip Phase 1, load pre-trained projector
        print(f"\nSkipping Phase 1, loading projector from {args.projector_path}")
        proj_state = torch.load(args.projector_path, map_location='cpu', weights_only=True)
        model.projector.load_state_dict(proj_state)
    else:
        # Train projector
        train_dataset_1a = Stage3Dataset(
            args.train_json, args.llm_path,
            augment=True, max_seq_len=args.max_seq_len,
        )
        val_dataset_1a = Stage1Dataset(
            args.val_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len
        )
        train_loader_1a = DataLoader(
            train_dataset_1a, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
        )
        val_loader_1a = DataLoader(
            val_dataset_1a, batch_size=args.batch_size, shuffle=False,
            num_workers=2, collate_fn=collate_fn,
        )

        print(f"Training samples: {len(train_dataset_1a)}, Val: {len(val_dataset_1a)}")

        optimizer_1a = torch.optim.AdamW(
            model.projector.parameters(), lr=args.lr_1a, weight_decay=args.weight_decay
        )
        steps_per_epoch_1a = math.ceil(len(train_loader_1a) / args.grad_accum)
        total_steps_1a = steps_per_epoch_1a * args.epochs_1a
        warmup_steps_1a = int(0.03 * total_steps_1a)
        scheduler_1a = get_cosine_schedule_with_warmup(optimizer_1a, warmup_steps_1a, total_steps_1a)

        print(f"Steps/epoch: {steps_per_epoch_1a}, Total: {total_steps_1a}, Warmup: {warmup_steps_1a}")

        best_val_loss_1a = float('inf')
        global_step_1a = 0

        for epoch in range(args.epochs_1a):
            model.train()
            model.encoder.eval()
            epoch_loss = 0.0
            epoch_tokens = 0
            optimizer_1a.zero_grad()

            for batch_idx, batch in enumerate(train_loader_1a):
                with autocast('cuda', dtype=torch.bfloat16):
                    loss = model(
                        volume=batch['volume'].to(device),
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device),
                        image_token_mask=batch['image_token_mask'].to(device),
                    )
                    loss_scaled = loss / args.grad_accum

                loss_scaled.backward()
                n_tokens = (batch['labels'] != -100).sum().item()
                epoch_loss += loss.item() * n_tokens
                epoch_tokens += n_tokens

                if (batch_idx + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.projector.parameters(), args.grad_clip)
                    optimizer_1a.step()
                    scheduler_1a.step()
                    optimizer_1a.zero_grad()
                    global_step_1a += 1

                    if global_step_1a % args.log_interval == 0:
                        avg_loss = epoch_loss / max(epoch_tokens, 1)
                        lr = optimizer_1a.param_groups[0]['lr']
                        print(f"  [1a] Epoch {epoch+1}/{args.epochs_1a} | "
                              f"Step {global_step_1a}/{total_steps_1a} | "
                              f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")

            # Flush remaining accumulated gradients
            if (batch_idx + 1) % args.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(model.projector.parameters(), args.grad_clip)
                optimizer_1a.step()
                scheduler_1a.step()
                optimizer_1a.zero_grad()
                global_step_1a += 1

            # Epoch eval
            avg_loss = epoch_loss / max(epoch_tokens, 1)
            val_loss, val_ppl = evaluate_sft(model, val_loader_1a, device,
                                             max_batches=args.max_val_samples)
            val_n = args.max_val_samples or len(val_loader_1a)
            print(f"\n[1a] Epoch {epoch+1}/{args.epochs_1a}: "
                  f"Train={avg_loss:.4f} Val={val_loss:.4f} PPL={val_ppl:.2f} "
                  f"(val on {val_n} samples)")

            if val_loss < best_val_loss_1a:
                best_val_loss_1a = val_loss
                save_dir_1a = ckpt_dir / 'stage1a_best'
                save_dir_1a.mkdir(parents=True, exist_ok=True)
                torch.save(model.projector.state_dict(), save_dir_1a / 'projector.pt')
                print(f"  >> New best! Saved to {save_dir_1a} (val_loss={val_loss:.4f})")

        print(f"\n[1a] Complete. Best val loss: {best_val_loss_1a:.4f}")

        # Load best projector for Phase 2
        best_proj = torch.load(ckpt_dir / 'stage1a_best' / 'projector.pt',
                               map_location='cpu', weights_only=True)
        model.projector.load_state_dict(best_proj)
        print(f"Loaded best Stage 1a projector for Phase 2")

        # Clean up 1a data
        del train_dataset_1a, train_loader_1a
        torch.cuda.empty_cache()

    # =====================================================================
    # --skip_1b: Run full test eval after Stage 1a, then exit
    # =====================================================================
    if args.skip_1b:
        print(f"\n{'='*60}")
        print("--skip_1b: Running full test eval on Stage 1a projector (no LoRA)")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
        with open(args.test_json) as f:
            test_samples = json.load(f)
        test_dataset = Stage1Dataset(
            args.test_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len
        )
        print(f"  Test samples: {len(test_samples)}")

        run_full_eval_singleturn(model, tokenizer, test_samples, test_dataset, device, ckpt_dir)

        print(f"\n=== Stage 1a eval complete (--skip_1b). ===")
        print(f"GPU Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        return

    # =====================================================================
    # Phase 2: Stage 1b — SFT with Perturbation
    # =====================================================================
    print(f"\n{'='*60}")
    print("Phase 2: Stage 1b — SFT with Prior Perturbation")
    print(f"  Encoder: frozen, Projector: trainable, LLM LoRA: trainable")
    print(f"{'='*60}")

    # Disable gradient checkpointing for LoRA setup
    model.llm.gradient_checkpointing_disable()

    # Apply fresh LoRA to LLM
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.print_trainable_parameters()

    # Re-enable gradient checkpointing
    model.llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Resume from epoch checkpoint if requested
    start_epoch = 0
    resume_global_step = 0
    if args.resume_epoch > 0:
        resume_dir = ckpt_dir / f'epoch_{args.resume_epoch}'
        assert resume_dir.exists(), f"Resume dir not found: {resume_dir}"
        proj_state = torch.load(resume_dir / 'projector.pt', map_location='cpu', weights_only=True)
        model.projector.load_state_dict(proj_state)
        lora_dir = resume_dir / 'lora_adapter'
        if lora_dir.exists():
            adapter_path = lora_dir / 'adapter_model.safetensors'
            if adapter_path.exists():
                from safetensors.torch import load_file
                adapter_weights = load_file(str(adapter_path))
            else:
                adapter_weights = torch.load(
                    str(lora_dir / 'adapter_model.bin'),
                    map_location='cpu', weights_only=True
                )
            set_peft_model_state_dict(model.llm, adapter_weights)
        state = torch.load(resume_dir / 'training_state.pt', map_location='cpu', weights_only=True)
        resume_global_step = state.get('global_step', 0)
        start_epoch = args.resume_epoch
        print(f"\n  Resumed from epoch {args.resume_epoch} (global_step={resume_global_step})")

    params = model.get_trainable_params()
    print(f"Trainable: {params['total']:,} (proj: {params['projector']:,}, lora: {params['llm']:,})")

    # Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)

    train_dataset_1b = Stage3Dataset(
        args.train_json, args.llm_path,
        augment=True, max_seq_len=args.max_seq_len,
    )

    val_dataset = Stage1Dataset(
        args.val_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )

    with open(args.test_json) as f:
        test_samples = json.load(f)
    test_dataset = Stage1Dataset(
        args.test_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len
    )
    print(f"Train: {len(train_dataset_1b)}, Val: {len(val_dataset)}, "
          f"Test: {len(test_samples)}, Gen eval: {args.gen_eval_samples}")

    # Optimizer (projector + LoRA)
    projector_params = [p for p in model.projector.parameters() if p.requires_grad]
    lora_params = [p for p in model.llm.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': projector_params, 'lr': args.lr_projector},
        {'params': lora_params, 'lr': args.lr_lora},
    ], weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_dataset_1b) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs_1b
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\nSteps/epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}")

    # Pre-training baseline eval (skip if pre-computed baseline is provided)
    if args.skip_baseline_eval:
        print("\n=== Skipping pre-training baseline eval (--skip_baseline_eval) ===")
    elif args.baseline_eval_path and Path(args.baseline_eval_path).exists():
        print(f"\n=== Loading Pre-computed Baseline from {args.baseline_eval_path} ===")
        with open(args.baseline_eval_path) as f:
            baseline_metrics = json.load(f)
        print_gen_metrics(baseline_metrics, prefix="  ")
        # Copy to this experiment's dir
        with open(ckpt_dir / 'baseline_eval.json', 'w') as f:
            json.dump(baseline_metrics, f, indent=2)
    else:
        print("\n=== Pre-training Baseline Generation Eval ===")
        baseline_metrics = evaluate_generation_singleturn(
            model, tokenizer, test_samples, test_dataset, device,
            num_eval=args.gen_eval_samples,
        )
        print_gen_metrics(baseline_metrics, prefix="  ")
        with open(ckpt_dir / 'baseline_eval.json', 'w') as f:
            json.dump(baseline_metrics, f, indent=2)

    if args.baseline_only:
        print(f"\n=== Baseline-only mode: done. Saved to {ckpt_dir / 'baseline_eval.json'} ===")
        return

    # Training loop
    print(f"\n=== Starting Phase 2 Training (epoch {start_epoch+1}→{args.epochs_1b}) ===\n")
    best_val_loss = float('inf')
    global_step = resume_global_step

    for epoch in range(start_epoch, args.epochs_1b):
        print(f"\n[1b] Epoch {epoch+1}/{args.epochs_1b}")

        train_loader = DataLoader(
            train_dataset_1b, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
        )

        model.train()
        model.encoder.eval()
        epoch_loss = 0.0
        epoch_tokens = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            with autocast('cuda', dtype=torch.bfloat16):
                loss = model(
                    volume=batch['volume'].to(device),
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device),
                    image_token_mask=batch['image_token_mask'].to(device),
                )

                # Per-sample change detection loss weighting
                if 'has_dx_change' in batch and (args.dx_change_weight > 0 or args.region_change_weight > 0):
                    has_dx = batch['has_dx_change'].float().to(device)
                    n_reg = batch['n_region_changes'].float().to(device)
                    weight = 1.0 + args.dx_change_weight * has_dx + args.region_change_weight * n_reg
                    loss = loss * weight.mean()

                loss_scaled = loss / args.grad_accum

            loss_scaled.backward()
            n_tokens = (batch['labels'] != -100).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_interval == 0:
                    avg_loss = epoch_loss / max(epoch_tokens, 1)
                    ppl = math.exp(min(avg_loss, 20))
                    lr_p = optimizer.param_groups[0]['lr']
                    lr_l = optimizer.param_groups[1]['lr']
                    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  [1b] Step {global_step}/{total_steps} | "
                          f"Loss: {avg_loss:.4f} PPL: {ppl:.2f} | "
                          f"LR proj: {lr_p:.2e} lora: {lr_l:.2e} | "
                          f"GPU: {gpu_mem:.1f}GB")

        # Flush remaining accumulated gradients
        if (batch_idx + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # Epoch end evaluation
        avg_loss = epoch_loss / max(epoch_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        val_loss, val_ppl = evaluate_sft(model, val_loader, device,
                                         max_batches=args.max_val_samples)
        val_n = args.max_val_samples or len(val_loader)

        print(f"\n{'='*60}")
        print(f"[1b] Epoch {epoch+1}/{args.epochs_1b} complete:")
        print(f"  Train Loss: {avg_loss:.4f} PPL: {ppl:.2f}")
        print(f"  Val Loss: {val_loss:.4f} PPL: {val_ppl:.2f} (val on {val_n} samples)")

        # Generation eval (skip if --skip_epoch_gen_eval)
        gen_metrics = None
        if not args.skip_epoch_gen_eval:
            gen_metrics = evaluate_generation_singleturn(
                model, tokenizer, test_samples, test_dataset, device,
                num_eval=args.gen_eval_samples,
            )
            print(f"  Gen Eval ({gen_metrics['num_eval']} samples):")
            print_gen_metrics(gen_metrics, prefix="    ")

        # Save best (by val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = ckpt_dir / 'best'
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.projector.state_dict(), save_dir / 'projector.pt')
            model.llm.save_pretrained(str(save_dir / 'lora_adapter'))
            torch.save({
                'epoch': epoch, 'global_step': global_step,
                'val_loss': val_loss,
            }, save_dir / 'training_state.pt')
            print(f"  >> New best! Saved to {save_dir} (val_loss={val_loss:.4f})")

        # Epoch checkpoint
        epoch_dir = ckpt_dir / f'epoch_{epoch+1}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.projector.state_dict(), epoch_dir / 'projector.pt')
        model.llm.save_pretrained(str(epoch_dir / 'lora_adapter'))
        torch.save({
            'epoch': epoch, 'global_step': global_step,
            'val_loss': val_loss,
        }, epoch_dir / 'training_state.pt')
        print(f"  Saved checkpoint to {epoch_dir}")
        print(f"{'='*60}")

    print(f"\n=== Ablation Training Complete ===")
    print(f"Best val loss (Phase 2): {best_val_loss:.4f}")
    print(f"GPU Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # =====================================================================
    # Full Test Set Evaluation (Stage 2 Baseline)
    # =====================================================================
    print(f"\n{'='*60}")
    print("Full Test Set Evaluation — Stage 2 Baseline")
    print(f"  Loading best checkpoint from {ckpt_dir / 'best'}")
    print(f"  Test samples: {len(test_samples)}")
    print(f"{'='*60}")

    # Reload best checkpoint
    best_dir = ckpt_dir / 'best'
    best_proj = torch.load(best_dir / 'projector.pt', map_location='cpu', weights_only=True)
    model.projector.load_state_dict(best_proj)
    best_lora_dir = best_dir / 'lora_adapter'
    if best_lora_dir.exists():
        # Load best LoRA weights into current PeftModel (swaps adapter weights only,
        # base LLM weights untouched — avoids merge_and_unload contamination)
        adapter_path = best_lora_dir / 'adapter_model.safetensors'
        if adapter_path.exists():
            from safetensors.torch import load_file
            adapter_weights = load_file(str(adapter_path))
        else:
            adapter_weights = torch.load(
                str(best_lora_dir / 'adapter_model.bin'),
                map_location='cpu', weights_only=True
            )
        set_peft_model_state_dict(model.llm, adapter_weights)
    print(f"  Best checkpoint loaded")

    run_full_eval_singleturn(model, tokenizer, test_samples, test_dataset, device, ckpt_dir)

    print(f"\n=== All Done ===")
    print(f"GPU Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")



# ---------------------------------------------------------------------------
# NLG helpers (no external deps)
# ---------------------------------------------------------------------------

def _tokenize_text(text):
    """Simple whitespace + punctuation tokenizer for NLG metrics."""
    return re.findall(r'\w+', text.lower())


def _compute_bleu(reference_tokens, hypothesis_tokens, max_n=4):
    """Compute BLEU-1..max_n with brevity penalty."""
    from collections import Counter
    scores = {}
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens) - n + 1))
        hyp_ngrams = Counter(tuple(hypothesis_tokens[i:i+n]) for i in range(len(hypothesis_tokens) - n + 1))
        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = max(sum(hyp_ngrams.values()), 1)
        scores[f'bleu_{n}'] = clipped / total
    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(reference_tokens) / max(len(hypothesis_tokens), 1)))
    for k in scores:
        scores[k] *= bp
    return scores


def _compute_rouge_l(reference_tokens, hypothesis_tokens):
    """Compute ROUGE-L F1 via longest common subsequence."""
    m, n = len(reference_tokens), len(hypothesis_tokens)
    if m == 0 or n == 0:
        return 0.0
    # LCS length via DP (space-optimized)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if reference_tokens[i-1] == hypothesis_tokens[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    lcs_len = prev[n]
    prec = lcs_len / n
    rec = lcs_len / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _compute_ece(confidences, accuracies, n_bins=10):
    """Expected Calibration Error."""
    if not confidences:
        return 0.0
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    total = len(confidences)
    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        indices = [i for i, c in enumerate(confidences) if lo <= c < hi or (b == n_bins - 1 and c == hi)]
        if not indices:
            continue
        bin_acc = sum(accuracies[i] for i in indices) / len(indices)
        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        ece += len(indices) / total * abs(bin_acc - bin_conf)
    return round(ece, 4)


def _cohen_weighted_kappa(y_true, y_pred, labels):
    """Weighted (linear) Cohen's Kappa for ordinal classes."""
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(y_true)
    if k == 0:
        return 0.0
    # Confusion matrix
    cm = [[0]*n for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt]][label_to_idx[yp]] += 1
    # Weight matrix (linear)
    w = [[abs(i - j) / max(n - 1, 1) for j in range(n)] for i in range(n)]
    # Expected matrix
    row_sums = [sum(cm[i]) for i in range(n)]
    col_sums = [sum(cm[i][j] for i in range(n)) for j in range(n)]
    e = [[row_sums[i] * col_sums[j] / max(k, 1) for j in range(n)] for i in range(n)]
    num = sum(w[i][j] * cm[i][j] for i in range(n) for j in range(n))
    den = sum(w[i][j] * e[i][j] for i in range(n) for j in range(n))
    if den == 0:
        return 1.0
    return round(1 - num / den, 4)


def _precision_recall_f1(tp, fp, fn):
    """Compute precision, recall, F1 from counts."""
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return round(prec * 100, 1), round(rec * 100, 1), round(f1 * 100, 1)


def run_full_eval_singleturn(model, tokenizer, test_samples, test_dataset, device,
                             ckpt_dir, output_name='full_eval_singleturn.json'):
    """Comprehensive single-turn evaluation with 35+ metrics.

    Metrics: classification (DX/region accuracy, F1, kappa, confusion matrix),
    hallucination detection, confidence calibration, reasoning quality,
    NLG (BLEU/ROUGE-L on Diagnostic Summary), longitudinal analysis.
    """
    model.eval()
    gc_was_enabled = getattr(model.llm, 'is_gradient_checkpointing', False)
    if gc_was_enabled:
        model.llm.gradient_checkpointing_disable()

    REGIONS_MAP = {
        'Hippocampus': 'hippocampus', 'Entorhinal': 'entorhinal',
        'Fusiform': 'fusiform', 'MidTemp': 'midtemporal', 'Ventricles': 'ventricles'
    }
    REGION_KEYS = list(REGIONS_MAP.values())
    DX_CLASSES = ['CN', 'MCI', 'Dementia']
    REGION_LABELS = ['normal', 'mild_atrophy', 'severe_atrophy']
    REGION_LABELS_VENT = ['normal', 'mild_enlargement', 'severe_enlargement']

    system_msg = (
        "You are a neuroradiologist assessing a patient's brain MRI for signs of "
        "neurodegeneration. Based on the clinical information below and the attached "
        "3D T1-weighted MRI, provide your assessment."
    )
    im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    image_pad_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')

    sample_id_to_idx = {s['sample_id']: j for j, s in enumerate(test_dataset.samples)}

    # ---- Counters: Format ----
    json_valid = 0
    has_report = 0

    # ---- Counters: DX ----
    dx_correct = dx_total = 0
    dx_per_class = {c: {'correct': 0, 'total': 0} for c in DX_CLASSES}
    baseline_dx_correct = baseline_dx_total = 0
    followup_dx_correct = followup_dx_total = 0
    dx_y_true, dx_y_pred = [], []       # for F1 / kappa / confusion matrix
    dx_conf_correct, dx_conf_wrong = [], []  # confidence calibration
    dx_conf_all, dx_acc_all = [], []     # ECE
    dx_adjacent_errors = 0               # CN↔MCI
    dx_critical_errors = 0               # CN↔Dementia

    # ---- Counters: Region ----
    region_correct = region_total = 0
    region_per_name = {r: {'correct': 0, 'total': 0} for r in REGION_KEYS}
    region_y_true_all, region_y_pred_all = [], []
    region_y_per_name = {r: {'y_true': [], 'y_pred': []} for r in REGION_KEYS}
    # Per-severity
    severity_correct = {'normal': 0, 'abnormal_mild': 0, 'abnormal_severe': 0}
    severity_total = {'normal': 0, 'abnormal_mild': 0, 'abnormal_severe': 0}
    # False abnormal / false severe
    false_abnormal = false_abnormal_total = 0    # GT=normal, pred≠normal
    false_severe = false_severe_total = 0        # GT∈{normal,mild}, pred=severe
    # Region confidence
    region_conf_all, region_acc_all = [], []

    # ---- Counters: Hallucination ----
    cognitive_halluc = cognitive_halluc_possible = 0
    prior_dx_halluc = prior_dx_halluc_possible = 0
    longi_halluc = longi_halluc_possible = 0

    # ---- Counters: Reasoning ----
    reasoning_complete = 0
    imaging_obs_present = 0
    clinical_int_present = 0
    longi_syn_present = 0
    reasoning_total = 0
    # Region mention P/R/F1
    region_mention_tp = region_mention_fp = region_mention_fn = 0
    progression_cited_correct = progression_cited_total = 0

    # ---- Counters: NLG ----
    bleu_scores = {f'bleu_{n}': [] for n in range(1, 5)}
    rouge_l_scores = []

    # ---- Counters: Longitudinal ----
    longi_dir_correct = longi_dir_total = 0
    longi_dir_per_region = {r: {'correct': 0, 'total': 0} for r in REGION_KEYS}
    dx_changed_detect_tp = dx_changed_detect_fn = 0
    dx_changed_detect_fp = dx_changed_detect_tn = 0
    region_changed_tp = region_changed_fn = 0
    region_changed_fp = region_changed_tn = 0

    per_sample_results = []
    eval_start = time.time()

    for i, sample in enumerate(test_samples):
        sample_id = sample['sample_id']
        dataset_idx = sample_id_to_idx.get(sample_id)
        if dataset_idx is None:
            continue

        data_item = test_dataset[dataset_idx]
        volume = data_item['volume'].unsqueeze(0).to(device)

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
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        image_token_mask = (input_ids == image_pad_id)

        is_baseline = sample.get('is_baseline', True)

        # Single generation: JSON + report
        with torch.no_grad():
            with autocast('cuda', dtype=torch.bfloat16):
                output_ids = model.generate(
                    volume, input_ids, attention_mask, image_token_mask,
                    max_new_tokens=900
                )
        generated_text = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)

        # Parse JSON from output
        parsed_json = parse_json_from_response(generated_text)

        # Report marker
        has_all_markers = bool(re.search(r'\[Diagnostic Summary\]', generated_text))
        if has_all_markers:
            has_report += 1

        gt_json = sample.get('gt_json')
        gt_dx = sample.get('gt_diagnosis')

        sample_record = {
            'sample_id': sample_id,
            'is_baseline': is_baseline,
            'generated_text': generated_text,
        }

        # ==== Diagnosis ====
        pred_dx = None
        pred_dx_conf = None
        if parsed_json is not None:
            json_valid += 1
            pred_dx = parsed_json.get('diagnosis', {}).get('label')
            pred_dx_conf = parsed_json.get('diagnosis', {}).get('confidence')
            sample_record['pred_diagnosis'] = pred_dx
            sample_record['dx_correct'] = (pred_dx == gt_dx)

            if pred_dx and gt_dx:
                dx_total += 1
                dx_y_true.append(gt_dx)
                dx_y_pred.append(pred_dx)
                dx_per_class[gt_dx]['total'] += 1

                is_correct = (pred_dx == gt_dx)
                if is_correct:
                    dx_correct += 1
                    dx_per_class[gt_dx]['correct'] += 1

                # Confidence calibration
                if pred_dx_conf is not None:
                    dx_conf_all.append(float(pred_dx_conf))
                    dx_acc_all.append(1 if is_correct else 0)
                    if is_correct:
                        dx_conf_correct.append(float(pred_dx_conf))
                    else:
                        dx_conf_wrong.append(float(pred_dx_conf))

                # Error type
                if not is_correct:
                    pair = frozenset([gt_dx, pred_dx])
                    if pair == frozenset(['CN', 'Dementia']):
                        dx_critical_errors += 1
                    elif pair == frozenset(['CN', 'MCI']) or pair == frozenset(['MCI', 'Dementia']):
                        dx_adjacent_errors += 1

                if is_baseline:
                    baseline_dx_total += 1
                    if is_correct:
                        baseline_dx_correct += 1
                else:
                    followup_dx_total += 1
                    if is_correct:
                        followup_dx_correct += 1

            # ==== Region scoring ====
            if 'anatomical_assessment' in parsed_json and gt_json:
                for csv_region, json_key in REGIONS_MAP.items():
                    pred_r = parsed_json.get('anatomical_assessment', {}).get(json_key, {})
                    gt_r = gt_json.get('anatomical_assessment', {}).get(json_key, {})
                    pred_label = pred_r.get('label') if isinstance(pred_r, dict) else None
                    gt_label = gt_r.get('label') if isinstance(gt_r, dict) else None
                    pred_conf = pred_r.get('confidence') if isinstance(pred_r, dict) else None

                    if pred_label and gt_label:
                        region_total += 1
                        region_per_name[json_key]['total'] += 1
                        region_y_true_all.append(gt_label)
                        region_y_pred_all.append(pred_label)
                        region_y_per_name[json_key]['y_true'].append(gt_label)
                        region_y_per_name[json_key]['y_pred'].append(pred_label)

                        r_correct = (pred_label == gt_label)
                        if r_correct:
                            region_correct += 1
                            region_per_name[json_key]['correct'] += 1

                        # Confidence calibration
                        if pred_conf is not None:
                            region_conf_all.append(float(pred_conf))
                            region_acc_all.append(1 if r_correct else 0)

                        # Per-severity accuracy
                        is_normal = (gt_label == 'normal')
                        is_mild = gt_label in ('mild_atrophy', 'mild_enlargement')
                        is_severe = gt_label in ('severe_atrophy', 'severe_enlargement')
                        if is_normal:
                            severity_total['normal'] += 1
                            if r_correct:
                                severity_correct['normal'] += 1
                        elif is_mild:
                            severity_total['abnormal_mild'] += 1
                            if r_correct:
                                severity_correct['abnormal_mild'] += 1
                        elif is_severe:
                            severity_total['abnormal_severe'] += 1
                            if r_correct:
                                severity_correct['abnormal_severe'] += 1

                        # False abnormal: GT=normal, pred≠normal
                        if is_normal:
                            false_abnormal_total += 1
                            if pred_label != 'normal':
                                false_abnormal += 1

                        # False severe: GT∈{normal,mild}, pred=severe
                        pred_is_severe = pred_label in ('severe_atrophy', 'severe_enlargement')
                        if not is_severe:
                            false_severe_total += 1
                            if pred_is_severe:
                                false_severe += 1

            # ==== Longitudinal comparison (follow-ups only) ====
            gt_longi = gt_json.get('longitudinal_comparison') if gt_json else None
            pred_longi = parsed_json.get('longitudinal_comparison') if parsed_json else None

            if not is_baseline and gt_longi and pred_longi and isinstance(pred_longi, dict):
                # Direction accuracy per region
                for rk in REGION_KEYS:
                    gt_dir_info = gt_longi.get(rk)
                    pred_dir_info = pred_longi.get(rk)
                    if isinstance(gt_dir_info, dict) and isinstance(pred_dir_info, dict):
                        gt_dir = gt_dir_info.get('direction')
                        pred_dir = pred_dir_info.get('direction')
                        if gt_dir and pred_dir:
                            longi_dir_total += 1
                            longi_dir_per_region[rk]['total'] += 1
                            if gt_dir == pred_dir:
                                longi_dir_correct += 1
                                longi_dir_per_region[rk]['correct'] += 1

                        # Region changed detection
                        gt_changed = gt_dir_info.get('changed')
                        pred_changed = pred_dir_info.get('changed')
                        if gt_changed is not None and pred_changed is not None:
                            if gt_changed and pred_changed:
                                region_changed_tp += 1
                            elif gt_changed and not pred_changed:
                                region_changed_fn += 1
                            elif not gt_changed and pred_changed:
                                region_changed_fp += 1
                            else:
                                region_changed_tn += 1

                # DX changed detection
                gt_dx_change = gt_longi.get('diagnosis_change', {})
                pred_dx_change = pred_longi.get('diagnosis_change', {})
                if isinstance(gt_dx_change, dict) and isinstance(pred_dx_change, dict):
                    gt_dxc = gt_dx_change.get('changed')
                    pred_dxc = pred_dx_change.get('changed')
                    if gt_dxc is not None and pred_dxc is not None:
                        if gt_dxc and pred_dxc:
                            dx_changed_detect_tp += 1
                        elif gt_dxc and not pred_dxc:
                            dx_changed_detect_fn += 1
                        elif not gt_dxc and pred_dxc:
                            dx_changed_detect_fp += 1
                        else:
                            dx_changed_detect_tn += 1

            # ==== Reasoning quality ====
            pred_reasoning = parsed_json.get('reasoning', {}) if parsed_json else {}
            gt_reasoning = gt_json.get('reasoning', {}) if gt_json else {}

            if pred_reasoning:
                reasoning_total += 1
                io = pred_reasoning.get('imaging_observations', '')
                ci = pred_reasoning.get('clinical_integration', '')
                ls = pred_reasoning.get('longitudinal_synthesis', '')
                if io and len(str(io)) > 5:
                    imaging_obs_present += 1
                if ci and len(str(ci)) > 5:
                    clinical_int_present += 1
                if ls and len(str(ls)) > 5:
                    longi_syn_present += 1
                if (io and len(str(io)) > 5 and ci and len(str(ci)) > 5
                        and ls and len(str(ls)) > 5):
                    reasoning_complete += 1

                # Region mention P/R/F1
                pred_regions = set(pred_reasoning.get('regions_mentioned', []))
                # GT abnormal regions = regions with non-normal labels
                gt_abnormal = set()
                if gt_json and 'anatomical_assessment' in gt_json:
                    for rk in REGION_KEYS:
                        gt_lab = gt_json['anatomical_assessment'].get(rk, {}).get('label', 'normal')
                        if gt_lab != 'normal':
                            gt_abnormal.add(rk)
                if gt_abnormal or pred_regions:
                    region_mention_tp += len(pred_regions & gt_abnormal)
                    region_mention_fp += len(pred_regions - gt_abnormal)
                    region_mention_fn += len(gt_abnormal - pred_regions)

                # Progression cited accuracy (follow-ups only)
                if not is_baseline:
                    gt_prog = gt_reasoning.get('progression_cited')
                    pred_prog = pred_reasoning.get('progression_cited')
                    if gt_prog is not None and pred_prog is not None:
                        progression_cited_total += 1
                        if gt_prog == pred_prog:
                            progression_cited_correct += 1

            # ==== Hallucination detection ====
            # 1. Cognitive score hallucination: prompt says "Not available" but model mentions specific score
            mmse_unavail = 'Not available' in prompt_text.split('MMSE')[1][:40] if 'MMSE' in prompt_text else False
            cdrsb_unavail = 'Not available' in prompt_text.split('CDR')[1][:60] if 'CDR' in prompt_text else False
            if mmse_unavail or cdrsb_unavail:
                cognitive_halluc_possible += 1
                # Check if model fabricated a score value
                gen_lower = generated_text.lower()
                fabricated = False
                if mmse_unavail and re.search(r'mmse\s*(of|=|:|\s)\s*\d+', gen_lower):
                    fabricated = True
                if cdrsb_unavail and re.search(r'cdr[\s-]*(sb|sum)?\s*(of|=|:|\s)\s*\d+', gen_lower):
                    fabricated = True
                if fabricated:
                    cognitive_halluc += 1
                sample_record['cognitive_halluc'] = fabricated

            # 2. Prior diagnosis hallucination: baseline but model claims prior
            if is_baseline:
                prior_dx_halluc_possible += 1
                has_prior_claim = bool(re.search(
                    r'(prior|previous)\s+(diagnosis|visit|study|scan)',
                    generated_text, re.IGNORECASE
                ))
                # Exclude template phrases like "First visit — no prior data"
                if has_prior_claim and not re.search(r'(no prior|first visit|no previous)', generated_text, re.IGNORECASE):
                    prior_dx_halluc += 1
                    sample_record['prior_dx_halluc'] = True
                else:
                    sample_record['prior_dx_halluc'] = False

            # 3. Longitudinal fabrication on baseline
            if is_baseline:
                longi_halluc_possible += 1
                # Check if model fabricated longitudinal comparison
                pred_longi_bl = parsed_json.get('longitudinal_comparison') if parsed_json else None
                if pred_longi_bl is not None and pred_longi_bl != {} and pred_longi_bl is not False:
                    # Model should output null for baseline
                    if isinstance(pred_longi_bl, dict) and any(
                        pred_longi_bl.get(rk, {}).get('direction') not in (None, 'stable', '')
                        for rk in REGION_KEYS if isinstance(pred_longi_bl.get(rk), dict)
                    ):
                        longi_halluc += 1
                        sample_record['longi_halluc'] = True
                    else:
                        sample_record['longi_halluc'] = False
                else:
                    sample_record['longi_halluc'] = False

        else:
            sample_record['pred_diagnosis'] = None
            sample_record['dx_correct'] = False

        # ==== NLG metrics on Diagnostic Summary ====
        gt_response = sample.get('response', '')
        gt_summary_match = re.search(r'\[Diagnostic Summary\]\s*(.+)', gt_response, re.DOTALL)
        pred_summary_match = re.search(r'\[Diagnostic Summary\]\s*(.+)', generated_text, re.DOTALL)
        if gt_summary_match and pred_summary_match:
            gt_tokens = _tokenize_text(gt_summary_match.group(1).strip())
            pred_tokens = _tokenize_text(pred_summary_match.group(1).strip())
            if gt_tokens and pred_tokens:
                bleu = _compute_bleu(gt_tokens, pred_tokens)
                for k, v in bleu.items():
                    bleu_scores[k].append(v)
                rouge_l_scores.append(_compute_rouge_l(gt_tokens, pred_tokens))

        sample_record['has_report'] = has_all_markers
        per_sample_results.append(sample_record)

        if (i + 1) % 20 == 0 or (i + 1) == len(test_samples):
            elapsed = time.time() - eval_start
            rate = elapsed / (i + 1)
            eta = rate * (len(test_samples) - i - 1)
            print(f"  Single-turn eval [{i+1}/{len(test_samples)}] "
                  f"JSON:{json_valid}/{i+1} DX:{dx_correct}/{dx_total} "
                  f"Reg:{region_correct}/{region_total} "
                  f"Report:{has_report}/{i+1} "
                  f"ETA:{eta/60:.1f}min", flush=True)

    # =====================================================================
    # Compute aggregate metrics
    # =====================================================================
    total = len(test_samples)
    safe_div = lambda a, b: round(100 * a / b, 1) if b > 0 else 0.0
    safe_div4 = lambda a, b: round(a / b, 4) if b > 0 else 0.0

    # --- DX: P/R/F1 per class, macro, weighted ---
    dx_prf = {}
    for c in DX_CLASSES:
        tp = sum(1 for yt, yp in zip(dx_y_true, dx_y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(dx_y_true, dx_y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(dx_y_true, dx_y_pred) if yt == c and yp != c)
        p, r, f = _precision_recall_f1(tp, fp, fn)
        # Sensitivity = recall, Specificity = TN / (TN + FP)
        tn = sum(1 for yt, yp in zip(dx_y_true, dx_y_pred) if yt != c and yp != c)
        spec = safe_div(tn, tn + fp)
        dx_prf[c] = {'precision': p, 'recall': r, 'f1': f,
                      'sensitivity': r, 'specificity': spec}

    macro_f1 = round(sum(dx_prf[c]['f1'] for c in DX_CLASSES) / 3, 1)
    weighted_f1_num = sum(dx_prf[c]['f1'] * dx_per_class[c]['total'] for c in DX_CLASSES)
    weighted_f1 = round(weighted_f1_num / max(dx_total, 1), 1)

    # DX confusion matrix
    dx_cm = {gt: {pred: 0 for pred in DX_CLASSES} for gt in DX_CLASSES}
    for yt, yp in zip(dx_y_true, dx_y_pred):
        if yt in dx_cm and yp in DX_CLASSES:
            dx_cm[yt][yp] += 1

    # Cohen's weighted kappa (ordinal: CN < MCI < Dementia)
    dx_kappa = _cohen_weighted_kappa(dx_y_true, dx_y_pred, DX_CLASSES)

    # --- Region: weighted kappa per region, F1 per region ---
    region_kappa_per_name = {}
    region_f1_per_name = {}
    for rk in REGION_KEYS:
        yt_list = region_y_per_name[rk]['y_true']
        yp_list = region_y_per_name[rk]['y_pred']
        labels = REGION_LABELS_VENT if rk == 'ventricles' else REGION_LABELS
        region_kappa_per_name[rk] = _cohen_weighted_kappa(yt_list, yp_list, labels)
        # Macro F1 across severity levels for this region
        f1_sum = 0
        n_labels = 0
        for lab in labels:
            tp = sum(1 for y, p in zip(yt_list, yp_list) if y == lab and p == lab)
            fp = sum(1 for y, p in zip(yt_list, yp_list) if y != lab and p == lab)
            fn = sum(1 for y, p in zip(yt_list, yp_list) if y == lab and p != lab)
            _, _, f1 = _precision_recall_f1(tp, fp, fn)
            if sum(1 for y in yt_list if y == lab) > 0:
                f1_sum += f1
                n_labels += 1
        region_f1_per_name[rk] = round(f1_sum / max(n_labels, 1), 1)

    region_macro_f1 = round(sum(region_f1_per_name.values()) / max(len(REGION_KEYS), 1), 1)

    # --- Hallucination summary ---
    overall_halluc_samples = set()
    for sr in per_sample_results:
        if sr.get('cognitive_halluc') or sr.get('prior_dx_halluc') or sr.get('longi_halluc'):
            overall_halluc_samples.add(sr['sample_id'])
    # Add false abnormal as sample-level
    # (false_abnormal is region-level, so compute sample-level separately)
    halluc_any_count = len(overall_halluc_samples)

    # --- Confidence calibration ---
    dx_ece = _compute_ece(dx_conf_all, dx_acc_all)
    mean_conf_correct = round(sum(dx_conf_correct) / max(len(dx_conf_correct), 1), 3)
    mean_conf_wrong = round(sum(dx_conf_wrong) / max(len(dx_conf_wrong), 1), 3)
    overconf_count = sum(1 for c, a in zip(dx_conf_all, dx_acc_all) if c > 0.8 and a == 0)
    overconf_rate = safe_div(overconf_count, len(dx_conf_all))
    region_ece = _compute_ece(region_conf_all, region_acc_all)

    # --- NLG averages ---
    avg_bleu = {k: round(sum(v) / max(len(v), 1), 4) for k, v in bleu_scores.items()}
    avg_rouge_l = round(sum(rouge_l_scores) / max(len(rouge_l_scores), 1), 4)

    # --- Region mention P/R/F1 ---
    rm_prec, rm_rec, rm_f1 = _precision_recall_f1(
        region_mention_tp, region_mention_fp, region_mention_fn)

    # --- Longitudinal change detection P/R/F1 ---
    rc_prec, rc_rec, rc_f1 = _precision_recall_f1(
        region_changed_tp, region_changed_fp, region_changed_fn)
    dxc_prec, dxc_rec, dxc_f1 = _precision_recall_f1(
        dx_changed_detect_tp, dx_changed_detect_fp, dx_changed_detect_fn)

    # =====================================================================
    # Print comprehensive report
    # =====================================================================
    elapsed_min = (time.time() - eval_start) / 60
    print(f"\n{'='*70}")
    print(f"  SINGLE-TURN COMPREHENSIVE EVAL RESULTS")
    print(f"  Experiment: {ckpt_dir.name}")
    print(f"  Samples: {total}  |  Time: {elapsed_min:.1f} min")
    print(f"{'='*70}")

    print(f"\n--- A. Format Quality ---")
    print(f"  JSON valid:   {safe_div(json_valid, total)}% ({json_valid}/{total})")
    print(f"  Report:       {safe_div(has_report, total)}% ({has_report}/{total})")

    print(f"\n--- B. Diagnosis Classification ---")
    print(f"  Overall Accuracy: {safe_div(dx_correct, dx_total)}% ({dx_correct}/{dx_total})")
    print(f"  Baseline DX:      {safe_div(baseline_dx_correct, baseline_dx_total)}%")
    print(f"  Follow-up DX:     {safe_div(followup_dx_correct, followup_dx_total)}%")
    print(f"  Macro-F1:         {macro_f1}%")
    print(f"  Weighted-F1:      {weighted_f1}%")
    print(f"  Cohen's Kappa:    {dx_kappa}")
    dx_errors = dx_total - dx_correct
    print(f"  Adjacent errors:  {dx_adjacent_errors}/{dx_errors if dx_errors else 0} "
          f"({safe_div(dx_adjacent_errors, max(dx_errors, 1))}%)")
    print(f"  Critical errors:  {dx_critical_errors}/{dx_errors if dx_errors else 0} "
          f"({safe_div(dx_critical_errors, max(dx_errors, 1))}%)")
    print(f"  Per-class:")
    for c in DX_CLASSES:
        v = dx_per_class[c]
        prf = dx_prf[c]
        print(f"    {c:10s}: Acc={safe_div(v['correct'], v['total']):5.1f}% "
              f"P={prf['precision']:.1f} R={prf['recall']:.1f} F1={prf['f1']:.1f} "
              f"Sens={prf['sensitivity']:.1f} Spec={prf['specificity']:.1f}")
    print(f"  Confusion Matrix (rows=GT, cols=Pred):")
    print(f"    {'':10s} {'CN':>6s} {'MCI':>6s} {'Dem':>6s}")
    for gt_c in DX_CLASSES:
        row = [str(dx_cm[gt_c][pc]) for pc in DX_CLASSES]
        print(f"    {gt_c:10s} {row[0]:>6s} {row[1]:>6s} {row[2]:>6s}")

    print(f"\n--- C. Region Assessment ---")
    print(f"  Overall Accuracy: {safe_div(region_correct, region_total)}% ({region_correct}/{region_total})")
    print(f"  Macro Region F1:  {region_macro_f1}%")
    print(f"  Per-region:")
    for r in REGION_KEYS:
        v = region_per_name[r]
        print(f"    {r:14s}: Acc={safe_div(v['correct'], v['total']):5.1f}%  "
              f"F1={region_f1_per_name[r]:.1f}%  Kappa={region_kappa_per_name[r]}")
    print(f"  Per-severity accuracy:")
    for sev in ['normal', 'abnormal_mild', 'abnormal_severe']:
        print(f"    {sev:16s}: {safe_div(severity_correct[sev], severity_total[sev])}% "
              f"({severity_correct[sev]}/{severity_total[sev]})")
    print(f"  False Abnormal Rate: {safe_div(false_abnormal, false_abnormal_total)}% "
          f"({false_abnormal}/{false_abnormal_total})")
    print(f"  False Severe Rate:   {safe_div(false_severe, false_severe_total)}% "
          f"({false_severe}/{false_severe_total})")

    print(f"\n--- D. Hallucination Detection ---")
    print(f"  Cognitive score halluc:  {safe_div(cognitive_halluc, cognitive_halluc_possible)}% "
          f"({cognitive_halluc}/{cognitive_halluc_possible})")
    print(f"  Prior DX halluc (BL):    {safe_div(prior_dx_halluc, prior_dx_halluc_possible)}% "
          f"({prior_dx_halluc}/{prior_dx_halluc_possible})")
    print(f"  Longitudinal halluc (BL):{safe_div(longi_halluc, longi_halluc_possible)}% "
          f"({longi_halluc}/{longi_halluc_possible})")
    print(f"  False Abnormal Rate:     {safe_div(false_abnormal, false_abnormal_total)}%")
    print(f"  Any hallucination:       {safe_div(halluc_any_count, total)}% "
          f"({halluc_any_count}/{total})")

    print(f"\n--- E. Confidence Calibration ---")
    print(f"  DX ECE:              {dx_ece}")
    print(f"  Region ECE:          {region_ece}")
    print(f"  Mean conf (correct): {mean_conf_correct}")
    print(f"  Mean conf (wrong):   {mean_conf_wrong}")
    print(f"  Overconfidence rate: {overconf_rate}% "
          f"({overconf_count}/{len(dx_conf_all)})")

    print(f"\n--- F. Reasoning Quality ---")
    print(f"  Completeness:        {safe_div(reasoning_complete, reasoning_total)}% "
          f"({reasoning_complete}/{reasoning_total})")
    print(f"  imaging_observations:{safe_div(imaging_obs_present, reasoning_total)}%")
    print(f"  clinical_integration:{safe_div(clinical_int_present, reasoning_total)}%")
    print(f"  longi_synthesis:     {safe_div(longi_syn_present, reasoning_total)}%")
    print(f"  Region mention P/R/F1: {rm_prec:.1f}/{rm_rec:.1f}/{rm_f1:.1f}")
    print(f"  Progression cited acc: {safe_div(progression_cited_correct, progression_cited_total)}%")

    print(f"\n--- G. NLG (Diagnostic Summary) ---")
    for k, v in avg_bleu.items():
        print(f"  {k.upper()}: {v:.4f}")
    print(f"  ROUGE-L: {avg_rouge_l:.4f}")
    print(f"  (computed on {len(rouge_l_scores)} samples with both GT and pred summaries)")

    print(f"\n--- H. Longitudinal Analysis (follow-ups) ---")
    print(f"  Direction accuracy:  {safe_div(longi_dir_correct, longi_dir_total)}% "
          f"({longi_dir_correct}/{longi_dir_total})")
    print(f"  Per-region direction:")
    for r in REGION_KEYS:
        v = longi_dir_per_region[r]
        print(f"    {r:14s}: {safe_div(v['correct'], v['total']):5.1f}% ({v['correct']}/{v['total']})")
    print(f"  DX-change detection:     P={dxc_prec:.1f} R={dxc_rec:.1f} F1={dxc_f1:.1f}")
    print(f"  Region-change detection: P={rc_prec:.1f} R={rc_rec:.1f} F1={rc_f1:.1f}")

    print(f"\n{'='*70}")
    print(f"  SUMMARY: JSON {safe_div(json_valid, total)}% | "
          f"DX {safe_div(dx_correct, dx_total)}% (F1={macro_f1}% κ={dx_kappa}) | "
          f"Reg {safe_div(region_correct, region_total)}% (F1={region_macro_f1}%) | "
          f"Halluc {safe_div(halluc_any_count, total)}% | "
          f"ROUGE-L {avg_rouge_l:.3f}")
    print(f"{'='*70}")

    # =====================================================================
    # Build metrics dict
    # =====================================================================
    metrics = {
        'num_eval': total,
        # Format
        'json_valid': safe_div(json_valid, total),
        'report_complete': safe_div(has_report, total),
        # DX classification
        'dx_accuracy': safe_div(dx_correct, dx_total),
        'dx_correct': dx_correct, 'dx_total': dx_total,
        'dx_macro_f1': macro_f1,
        'dx_weighted_f1': weighted_f1,
        'dx_cohen_kappa': dx_kappa,
        'dx_adjacent_errors': dx_adjacent_errors,
        'dx_critical_errors': dx_critical_errors,
        'dx_per_class': {c: {
            'accuracy': safe_div(v['correct'], v['total']),
            'correct': v['correct'], 'total': v['total'],
            **dx_prf[c],
        } for c, v in dx_per_class.items()},
        'dx_confusion_matrix': dx_cm,
        'baseline_dx': safe_div(baseline_dx_correct, baseline_dx_total),
        'followup_dx': safe_div(followup_dx_correct, followup_dx_total),
        # Region
        'region_accuracy': safe_div(region_correct, region_total),
        'region_correct': region_correct, 'region_total': region_total,
        'region_macro_f1': region_macro_f1,
        'region_per_name': {r: {
            'accuracy': safe_div(v['correct'], v['total']),
            'correct': v['correct'], 'total': v['total'],
            'f1': region_f1_per_name[r],
            'kappa': region_kappa_per_name[r],
        } for r, v in region_per_name.items()},
        'region_severity': {sev: {
            'accuracy': safe_div(severity_correct[sev], severity_total[sev]),
            'correct': severity_correct[sev], 'total': severity_total[sev],
        } for sev in severity_total},
        'false_abnormal_rate': safe_div(false_abnormal, false_abnormal_total),
        'false_severe_rate': safe_div(false_severe, false_severe_total),
        # Hallucination
        'hallucination': {
            'cognitive_score': {
                'rate': safe_div(cognitive_halluc, cognitive_halluc_possible),
                'count': cognitive_halluc, 'possible': cognitive_halluc_possible,
            },
            'prior_dx_baseline': {
                'rate': safe_div(prior_dx_halluc, prior_dx_halluc_possible),
                'count': prior_dx_halluc, 'possible': prior_dx_halluc_possible,
            },
            'longi_baseline': {
                'rate': safe_div(longi_halluc, longi_halluc_possible),
                'count': longi_halluc, 'possible': longi_halluc_possible,
            },
            'false_abnormal_rate': safe_div(false_abnormal, false_abnormal_total),
            'overall_rate': safe_div(halluc_any_count, total),
            'overall_count': halluc_any_count,
        },
        # Calibration
        'calibration': {
            'dx_ece': dx_ece,
            'region_ece': region_ece,
            'mean_conf_correct': mean_conf_correct,
            'mean_conf_wrong': mean_conf_wrong,
            'overconfidence_rate': overconf_rate,
        },
        # Reasoning
        'reasoning': {
            'completeness': safe_div(reasoning_complete, reasoning_total),
            'imaging_obs': safe_div(imaging_obs_present, reasoning_total),
            'clinical_int': safe_div(clinical_int_present, reasoning_total),
            'longi_syn': safe_div(longi_syn_present, reasoning_total),
            'region_mention_precision': rm_prec,
            'region_mention_recall': rm_rec,
            'region_mention_f1': rm_f1,
            'progression_cited_acc': safe_div(progression_cited_correct, progression_cited_total),
        },
        # NLG
        'nlg': {**avg_bleu, 'rouge_l': avg_rouge_l,
                'num_samples': len(rouge_l_scores)},
        # Longitudinal
        'longitudinal': {
            'direction_accuracy': safe_div(longi_dir_correct, longi_dir_total),
            'direction_per_region': {r: {
                'accuracy': safe_div(v['correct'], v['total']),
                'correct': v['correct'], 'total': v['total'],
            } for r, v in longi_dir_per_region.items()},
            'dx_change_detection': {
                'precision': dxc_prec, 'recall': dxc_rec, 'f1': dxc_f1,
                'tp': dx_changed_detect_tp, 'fp': dx_changed_detect_fp,
                'fn': dx_changed_detect_fn, 'tn': dx_changed_detect_tn,
            },
            'region_change_detection': {
                'precision': rc_prec, 'recall': rc_rec, 'f1': rc_f1,
                'tp': region_changed_tp, 'fp': region_changed_fp,
                'fn': region_changed_fn, 'tn': region_changed_tn,
            },
        },
    }

    output = {
        'experiment': str(ckpt_dir.name),
        'eval_samples': total,
        'metrics': metrics,
        'samples': per_sample_results,
    }
    output_path = ckpt_dir / output_name
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved single-turn eval to {output_path}")
    print(f"  ({len(per_sample_results)} per-sample records)")

    if gc_was_enabled:
        model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )


if __name__ == '__main__':
    main()
