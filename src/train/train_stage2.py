"""
Stage 2: DPO Training with SFT Regularization.

Initializes from Stage 1b checkpoint. Trains on preference pairs (chosen/rejected)
from DPO candidate shards, with optional GT fallback for weak candidates.

Loss: L = L_DPO + alpha * L_SFT(GT)

Memory-efficient: single model copy, reference via adapter disable/enable.

Usage:
    python src/train/train_stage2.py \
        --projector_path /mnt/.../stage1b/best/projector.pt \
        --lora_path /mnt/.../stage1b/best/lora_adapter \
        --checkpoint_dir /mnt/.../stage2/q06_a01_m005 \
        --experiment_name q06_a01_m005 \
        --quality_threshold 0.6 --sft_alpha 0.1 --margin_threshold 0.05
"""

import os
import re
import sys
import copy
import json
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

from peft import get_peft_model, LoraConfig, PeftModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dpo_dataset import DPODataset, dpo_collate_fn
from src.data.stage1_dataset import Stage1Dataset, collate_fn, IMAGE_SENTINEL, NUM_VISUAL_TOKENS
from src.model.stage1_model import LoV3DStage1Model
from src.train.train_ablation import evaluate_generation_singleturn, run_full_eval_singleturn
from src.train.train_stage3 import (
    print_gen_metrics, evaluate_sft, get_cosine_schedule_with_warmup,
    parse_json_from_response,
)


# ---------------------------------------------------------------------------
# Core DPO functions
# ---------------------------------------------------------------------------

def compute_log_probs(model, volume, input_ids, attention_mask, labels,
                      image_token_mask, ref_projector=None,
                      length_normalize=False):
    """Compute per-sample log-probabilities.

    Args:
        model: LoV3DStage1Model with PEFT-wrapped LLM
        volume: (B, 1, 128, 128, 128)
        input_ids, attention_mask, labels, image_token_mask: from collate
        ref_projector: if provided, use instead of model.projector
        length_normalize: if True, return per-token average log-probs (SimPO style)
                          instead of summed log-probs. Requires larger beta (~2-5).

    Returns:
        log_probs: (B,) summed (or averaged if length_normalize) log-prob per sample
        avg_nll: (B,) average NLL per token (always per-token, for PPL)
    """
    B = volume.shape[0]

    # Visual features (always frozen encoder)
    with torch.no_grad():
        visual_features = model.encoder(volume)  # (B, 512, 1024)

    # Project with policy or reference projector
    projector = ref_projector if ref_projector is not None else model.projector
    proj_dtype = next(projector.parameters()).dtype
    visual_embeds = projector(visual_features.to(proj_dtype)).to(torch.bfloat16)

    # Text embeddings + image token replacement
    embed_fn = model.llm.get_input_embeddings()
    text_embeds = embed_fn(input_ids).clone()

    for i in range(B):
        mask_positions = image_token_mask[i].nonzero(as_tuple=True)[0]
        text_embeds[i, mask_positions] = visual_embeds[i]

    # Forward through LLM
    outputs = model.llm(
        inputs_embeds=text_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # (B, seq_len, vocab_size)

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    # Log-softmax and gather target token log-probs
    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    target_log_probs = log_probs_all.gather(
        dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    # Mask non-target positions
    target_mask = (shift_labels != -100).float()
    target_log_probs = target_log_probs * target_mask

    # Log-probs per sample
    summed_log_probs = target_log_probs.sum(dim=-1)
    n_tokens = target_mask.sum(dim=-1).clamp(min=1)

    if length_normalize:
        log_probs = summed_log_probs / n_tokens   # per-token average
    else:
        log_probs = summed_log_probs               # summed (standard DPO)

    # Average NLL per token (always per-token, for PPL & NLL anchor)
    avg_nll = -summed_log_probs / n_tokens

    return log_probs, avg_nll


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1,
             label_smoothing=0.0, margin_weights=None):
    """DPO loss with optional label smoothing (cDPO) and margin weighting.

    Args:
        label_smoothing: cDPO label noise (0.0 = standard DPO, 0.1 = 10% flip tolerance)
        margin_weights: (B,) per-sample weights from Verifier score margins.
                        Higher margin → cleaner signal → more weight.

    Returns:
        loss: scalar
        chosen_rewards: (B,)
        rejected_rewards: (B,)
    """
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    dpo_logits = beta * (policy_logratios - ref_logratios)

    # cDPO: label smoothing for noisy preferences
    if label_smoothing > 0:
        per_sample_loss = (
            -(1 - label_smoothing) * F.logsigmoid(dpo_logits)
            - label_smoothing * F.logsigmoid(-dpo_logits)
        )
    else:
        per_sample_loss = -F.logsigmoid(dpo_logits)

    # Margin weighting: higher Verifier margin → more weight
    if margin_weights is not None:
        loss = (margin_weights * per_sample_loss).sum() / margin_weights.sum()
    else:
        loss = per_sample_loss.mean()

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return loss, chosen_rewards, rejected_rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Stage 2: DPO + SFT regularization')

    # Paths
    parser.add_argument('--dpo_candidates', type=str, default='data_splits/dpo_candidates_all.json')
    parser.add_argument('--gt_json', type=str, default='data_splits/stage1_train.json')
    parser.add_argument('--val_json', type=str, default='data_splits/stage1_val.json')
    parser.add_argument('--test_json', type=str, default='data_splits/stage1_test.json')
    parser.add_argument('--encoder_path', type=str, required=True,
                        help='Path to Stage 0 encoder checkpoint (encoder_layer3.pt)')
    parser.add_argument('--llm_path', type=str, required=True,
                        help='Path to base LLM (e.g., Qwen2.5-14B)')
    parser.add_argument('--projector_path', type=str, required=True,
                        help='Stage 1b best projector.pt')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='Stage 1b best lora_adapter/')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)

    # DPO config
    parser.add_argument('--quality_threshold', type=float, default=0.6)
    parser.add_argument('--margin_threshold', type=float, default=0.05)
    parser.add_argument('--margin_quality_threshold', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO temperature parameter')
    parser.add_argument('--sft_alpha', type=float, default=0.1,
                        help='SFT regularization weight: L = L_DPO + alpha * L_SFT')
    parser.add_argument('--nll_alpha', type=float, default=0.0,
                        help='Chosen NLL weight to prevent likelihood displacement: '
                             'L += nll_alpha * NLL(chosen). Recommended: 0.5-1.0')
    parser.add_argument('--length_normalize', action='store_true', default=False,
                        help='Use per-token average log-probs instead of summed (SimPO style). '
                             'Requires larger beta (~2-5 instead of 0.1-0.3).')
    parser.add_argument('--margin_weighting', action='store_true', default=False,
                        help='Weight DPO loss by Verifier score margin per sample.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='cDPO label smoothing for noisy preferences (0.0-0.1).')

    # Training config
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr_projector', type=float, default=1e-5)
    parser.add_argument('--lr_lora', type=float, default=5e-5)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N samples (not steps)')
    parser.add_argument('--gen_eval_samples', type=int, default=20,
                        help='Samples for epoch-end quick eval')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {total_memory / 1e9:.1f} GB")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Stage 2 DPO: {args.experiment_name}")
    print(f"  quality_threshold={args.quality_threshold}, "
          f"margin_threshold={args.margin_threshold}")
    print(f"  sft_alpha={args.sft_alpha}, beta={args.beta}")
    print(f"{'='*60}")

    # ─── Load Model ─────────────────────────────────────────────────
    print("\n=== Loading Model ===")
    model = LoV3DStage1Model(args.encoder_path, args.llm_path)

    # Load Stage 1b projector
    proj_state = torch.load(args.projector_path, map_location='cpu', weights_only=True)
    model.projector.load_state_dict(proj_state)
    print(f"Loaded Stage 1b projector from {args.projector_path}")

    # Load Stage 1b LoRA, merge into base
    print(f"Loading Stage 1b LoRA from {args.lora_path}")
    model.llm = PeftModel.from_pretrained(model.llm, args.lora_path)
    model.llm = model.llm.merge_and_unload()
    print("  Stage 1b LoRA merged into base weights")

    # Save frozen reference projector
    ref_projector = copy.deepcopy(model.projector)
    ref_projector.requires_grad_(False)
    ref_projector.eval()
    print("  Reference projector saved (frozen copy)")

    # Apply fresh DPO LoRA
    print(f"Applying DPO LoRA (r={args.lora_rank}, alpha={args.lora_alpha})...")
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

    model.to(device)
    ref_projector.to(device)

    # Gradient checkpointing
    model.llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()

    params = model.get_trainable_params()
    print(f"\nTrainable: {params['total']:,} "
          f"(projector: {params['projector']:,}, llm_lora: {params['llm']:,})")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)

    # ─── Load Datasets ──────────────────────────────────────────────
    print("\n=== Loading DPO Dataset ===")
    train_dataset = DPODataset(
        dpo_candidates_path=args.dpo_candidates,
        gt_json_path=args.gt_json,
        tokenizer_path=args.llm_path,
        quality_threshold=args.quality_threshold,
        margin_threshold=args.margin_threshold,
        margin_quality_threshold=args.margin_quality_threshold,
        max_seq_len=args.max_seq_len,
    )
    print(f"DPO training samples: {len(train_dataset)}")

    # GT SFT dataset (original Stage 1 training data for regularization)
    print("Loading GT SFT dataset for regularization...")
    sft_dataset = Stage1Dataset(
        args.gt_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len,
    )
    print(f"GT SFT samples: {len(sft_dataset)}")

    # Validation set for epoch-end quick eval
    with open(args.val_json) as f:
        val_samples = json.load(f)
    val_dataset = Stage1Dataset(
        args.val_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len,
    )
    print(f"Val: {len(val_samples)} (epoch eval: {args.gen_eval_samples})")

    # Test set for final full eval
    with open(args.test_json) as f:
        test_samples = json.load(f)
    test_dataset = Stage1Dataset(
        args.test_json, args.llm_path, augment=False, max_seq_len=args.max_seq_len,
    )
    print(f"Test: {len(test_samples)} (final full eval)")

    # ─── Optimizer & Scheduler ──────────────────────────────────────
    projector_params = [p for p in model.projector.parameters() if p.requires_grad]
    lora_params = [p for p in model.llm.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': projector_params, 'lr': args.lr_projector},
        {'params': lora_params, 'lr': args.lr_lora},
    ], weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n=== Training Configuration ===")
    print(f"DPO samples: {len(train_dataset)}, GT SFT samples: {len(sft_dataset)}")
    print(f"Batch size: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"LR projector: {args.lr_projector}, LR LoRA: {args.lr_lora}")
    print(f"DPO beta={args.beta}, SFT alpha={args.sft_alpha} (on GT data), NLL alpha={args.nll_alpha}")
    if args.length_normalize:
        print(f"  Length normalization: ON (per-token avg log-probs)")
    if args.margin_weighting:
        print(f"  Margin weighting: ON")
    if args.label_smoothing > 0:
        print(f"  Label smoothing (cDPO): {args.label_smoothing}")

    # ─── Training Loop ──────────────────────────────────────────────
    print(f"\n=== Starting Stage 2 DPO Training ===\n")
    global_step = 0
    best_dx_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=dpo_collate_fn,
            pin_memory=True, drop_last=True,
        )

        # GT SFT dataloader (cycles independently alongside DPO)
        sft_loader = DataLoader(
            sft_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=2, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
        )
        sft_iter = iter(sft_loader)

        model.train()
        model.encoder.eval()
        ref_projector.eval()
        optimizer.zero_grad()

        # Logging accumulators (reset every log_interval samples)
        w_dpo = 0.0
        w_sft = 0.0
        w_total = 0.0
        w_chosen_reward = 0.0
        w_rejected_reward = 0.0
        w_reward_correct = 0
        w_chosen_nll = 0.0
        w_json_valid = 0
        w_count = 0

        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            volume = batch['volume'].to(device)
            B = volume.shape[0]

            # ── Reference forward (no grad, adapter OFF) ──
            _ln = args.length_normalize
            model.llm.disable_adapter_layers()
            with torch.no_grad():
                with autocast('cuda', dtype=torch.bfloat16):
                    ref_c_logps, _ = compute_log_probs(
                        model, volume,
                        batch['chosen_input_ids'].to(device),
                        batch['chosen_attention_mask'].to(device),
                        batch['chosen_labels'].to(device),
                        batch['chosen_image_token_mask'].to(device),
                        ref_projector=ref_projector,
                        length_normalize=_ln,
                    )
                    ref_r_logps, _ = compute_log_probs(
                        model, volume,
                        batch['rejected_input_ids'].to(device),
                        batch['rejected_attention_mask'].to(device),
                        batch['rejected_labels'].to(device),
                        batch['rejected_image_token_mask'].to(device),
                        ref_projector=ref_projector,
                        length_normalize=_ln,
                    )
            model.llm.enable_adapter_layers()

            # ── Policy forward (with grad, adapter ON) ──
            with autocast('cuda', dtype=torch.bfloat16):
                pol_c_logps, chosen_nll = compute_log_probs(
                    model, volume,
                    batch['chosen_input_ids'].to(device),
                    batch['chosen_attention_mask'].to(device),
                    batch['chosen_labels'].to(device),
                    batch['chosen_image_token_mask'].to(device),
                    length_normalize=_ln,
                )
                pol_r_logps, _ = compute_log_probs(
                    model, volume,
                    batch['rejected_input_ids'].to(device),
                    batch['rejected_attention_mask'].to(device),
                    batch['rejected_labels'].to(device),
                    batch['rejected_image_token_mask'].to(device),
                    length_normalize=_ln,
                )

                # Margin weights (if enabled)
                _mw = None
                if args.margin_weighting:
                    _mw = torch.tensor(batch['margins'], device=device,
                                       dtype=torch.float32).clamp(min=0.01)

                # DPO loss
                loss_dpo, c_rewards, r_rewards = dpo_loss(
                    pol_c_logps, pol_r_logps,
                    ref_c_logps, ref_r_logps,
                    beta=args.beta,
                    label_smoothing=args.label_smoothing,
                    margin_weights=_mw,
                )

                # SFT regularization on original GT data (not DPO chosen!)
                try:
                    sft_batch = next(sft_iter)
                except StopIteration:
                    sft_iter = iter(sft_loader)
                    sft_batch = next(sft_iter)
                loss_sft = model(
                    volume=sft_batch['volume'].to(device),
                    input_ids=sft_batch['input_ids'].to(device),
                    attention_mask=sft_batch['attention_mask'].to(device),
                    labels=sft_batch['labels'].to(device),
                    image_token_mask=sft_batch['image_token_mask'].to(device),
                )

                # Chosen NLL: prevents likelihood displacement
                # (policy moving away from both chosen & rejected)
                loss_nll = chosen_nll.mean() if args.nll_alpha > 0 else 0.0

                loss_total = (loss_dpo
                              + args.sft_alpha * loss_sft
                              + args.nll_alpha * loss_nll)
                (loss_total / args.grad_accum).backward()

            # ── Accumulate logging metrics ──
            with torch.no_grad():
                w_dpo += loss_dpo.item() * B
                w_sft += loss_sft.item() * B
                w_total += loss_total.item() * B
                w_chosen_reward += c_rewards.sum().item()
                w_rejected_reward += r_rewards.sum().item()
                w_reward_correct += (c_rewards > r_rewards).sum().item()
                w_chosen_nll += chosen_nll.sum().item()
                w_count += B

                # JSON validity check on chosen responses
                for sid in batch['sample_ids']:
                    idx = train_dataset.id_to_idx.get(sid)
                    if idx is not None:
                        resp = train_dataset.samples[idx]['chosen_response']
                        w_json_valid += 1 if parse_json_from_response(resp) is not None else 0

            # ── Optimizer step ──
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # ── Log every N samples ──
            if w_count >= args.log_interval and w_count > 0:
                avg_dpo = w_dpo / w_count
                avg_sft = w_sft / w_count
                avg_tot = w_total / w_count
                avg_c_rew = w_chosen_reward / w_count
                avg_r_rew = w_rejected_reward / w_count
                rew_margin = avg_c_rew - avg_r_rew
                rew_acc = w_reward_correct / w_count
                avg_ppl = math.exp(min(w_chosen_nll / w_count, 20))
                json_rate = w_json_valid / w_count
                lr_p = optimizer.param_groups[0]['lr']
                lr_l = optimizer.param_groups[1]['lr']

                elapsed = time.time() - epoch_start
                samples_done = batch_idx + 1
                rate = samples_done / elapsed if elapsed > 0 else 0
                eta = (len(train_loader) - samples_done) / rate if rate > 0 else 0

                print(f"  [{args.experiment_name}] Step {global_step}/{total_steps} "
                      f"({samples_done}/{len(train_loader)}) | "
                      f"DPO={avg_dpo:.4f} SFT={avg_sft:.4f} Total={avg_tot:.4f} | "
                      f"R_w={avg_c_rew:.3f} R_l={avg_r_rew:.3f} Margin={rew_margin:.3f} | "
                      f"PPL={avg_ppl:.2f} JSON={json_rate:.0%} RewAcc={rew_acc:.0%} | "
                      f"LR={lr_l:.2e} ETA={eta/60:.1f}min", flush=True)

                # Reset accumulators
                w_dpo = w_sft = w_total = 0.0
                w_chosen_reward = w_rejected_reward = w_chosen_nll = 0.0
                w_reward_correct = w_json_valid = w_count = 0

        # ─── Flush remaining accumulated gradients ──────────────────
        if (batch_idx + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # ─── Epoch End ──────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs} complete ({epoch_time/60:.1f} min)")

        # Save checkpoint directly (skip val eval — go straight to full test eval)
        best_dir = ckpt_dir / 'best'
        best_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.projector.state_dict(), best_dir / 'projector.pt')
        model.llm.save_pretrained(str(best_dir / 'lora_adapter'))
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'args': vars(args),
        }, best_dir / 'training_state.pt')
        print(f"  Saved checkpoint to {best_dir}")
        print(f"{'='*60}\n")

    # ─── Final Full Eval ────────────────────────────────────────────
    print(f"\n=== Final Full Eval (all {len(test_samples)} test samples) ===")

    # Load best checkpoint for final eval
    best_dir = ckpt_dir / 'best'
    if best_dir.exists():
        best_proj = torch.load(best_dir / 'projector.pt', map_location='cpu',
                               weights_only=True)
        model.projector.load_state_dict(best_proj)
        from peft import set_peft_model_state_dict
        adapter_path = best_dir / 'lora_adapter'
        if (adapter_path / 'adapter_model.safetensors').exists():
            from safetensors.torch import load_file
            adapter_weights = load_file(str(adapter_path / 'adapter_model.safetensors'))
        else:
            adapter_weights = torch.load(
                adapter_path / 'adapter_model.bin', map_location='cpu',
                weights_only=True,
            )
        set_peft_model_state_dict(model.llm, adapter_weights)
        print(f"  Loaded best checkpoint from {best_dir}")

    run_full_eval_singleturn(model, tokenizer, test_samples, test_dataset, device, ckpt_dir)

    print(f"\n=== All Done ===")
    print(f"Experiment: {args.experiment_name}")
    print(f"Best DX accuracy: {best_dx_acc:.1f}%")
    print(f"GPU Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == '__main__':
    main()
