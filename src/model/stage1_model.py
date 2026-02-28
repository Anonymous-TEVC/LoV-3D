"""
Stage 1 Model: Frozen ResNet50 Encoder + Visual Projector + Qwen-2.5-14B (LoRA).

Forward flow:
  1. MRI → Encoder → 16³×1024 → AdaptiveAvgPool3d(8) → 8³×1024 = 512 tokens
  2. Projector: Linear(1024→5120) → GELU → Linear(5120→5120)
  3. Replace <|image_pad|> positions in LLM embeddings with visual tokens
  4. Forward through Qwen-2.5-14B → next-token prediction loss
"""

import torch
import torch.nn as nn
from monai.networks.nets import resnet50
from transformers import AutoModelForCausalLM


class FrozenEncoder(nn.Module):
    """Frozen ResNet50 encoder (conv1→layer3) with spatial pooling to 8³."""

    def __init__(self, checkpoint_path):
        super().__init__()

        # Build MONAI ResNet50 backbone
        backbone = resnet50(
            pretrained=False, spatial_dims=3,
            n_input_channels=1, num_classes=1,
        )

        # Extract layers conv1→layer3
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act = backbone.act
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # Spatial pooling: 16³ → 8³ (for 128³ input)
        self.spatial_pool = nn.AdaptiveAvgPool3d(output_size=8)

        # Load Stage 0 encoder weights
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # Only spatial_pool should be unexpected (it has no parameters)
        if missing:
            print(f"  WARNING: Missing keys in encoder: {missing[:5]}...")
        if unexpected:
            print(f"  NOTE: Unexpected keys (ignored): {unexpected[:5]}...")
        print(f"  Loaded encoder: {len(state_dict)} keys from {checkpoint_path}")

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode=True):
        """Override train to always stay in eval mode (frozen BN)."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (B, 1, 128, 128, 128) float32
        Returns:
            features: (B, 512, 1024)
        """
        x = self.conv1(x)       # (B, 64, 128, 128, 128)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)     # (B, 64, 64, 64, 64)
        x = self.layer1(x)      # (B, 256, 64, 64, 64)
        x = self.layer2(x)      # (B, 512, 32, 32, 32)
        x = self.layer3(x)      # (B, 1024, 16, 16, 16)
        x = self.spatial_pool(x) # (B, 1024, 8, 8, 8)

        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)  # (B, 512, 1024)
        return x


class VisualProjector(nn.Module):
    """2-layer MLP projector: 1024 → 5120 → 5120 (LLaVA-1.5 style)."""

    def __init__(self, in_dim=1024, out_dim=5120):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 512, 1024)
        Returns:
            (B, 512, 5120)
        """
        return self.proj(x)


class LoV3DStage1Model(nn.Module):
    """Full Stage 1 model: frozen encoder + projector + LLM."""

    def __init__(self, encoder_path, llm_path, image_pad_token_id=151655):
        super().__init__()

        print("Initializing LoV3D Stage 1 Model...")

        # Frozen encoder
        print("  Loading frozen encoder...")
        self.encoder = FrozenEncoder(encoder_path)

        # Trainable projector
        print("  Initializing visual projector...")
        self.projector = VisualProjector(in_dim=1024, out_dim=5120)

        # LLM
        print("  Loading Qwen-2.5-14B...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print(f"  LLM loaded: {sum(p.numel() for p in self.llm.parameters()) / 1e9:.1f}B params")

        self.image_pad_token_id = image_pad_token_id

    def get_trainable_params(self):
        """Return count of trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        projector = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        llm = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        return {'total': total, 'projector': projector, 'llm': llm}

    def forward(self, volume, input_ids, attention_mask, labels, image_token_mask):
        """
        Args:
            volume: (B, 1, 128, 128, 128) float32
            input_ids: (B, seq_len) long
            attention_mask: (B, seq_len) long
            labels: (B, seq_len) long, -100 for non-target tokens
            image_token_mask: (B, seq_len) bool, True at <|image_pad|> positions
        Returns:
            loss: scalar tensor
        """
        B = volume.shape[0]

        # Step 1: Extract visual features (frozen, no grad)
        with torch.no_grad():
            visual_features = self.encoder(volume)  # (B, 512, 1024) float32

        # Step 2: Project to LLM dimension
        proj_dtype = next(self.projector.parameters()).dtype
        visual_embeds = self.projector(visual_features.to(proj_dtype))  # (B, 512, 5120)
        visual_embeds = visual_embeds.to(torch.bfloat16)

        # Step 3: Get text embeddings from LLM (clone to allow in-place replacement)
        embed_fn = self.llm.get_input_embeddings()
        text_embeds = embed_fn(input_ids).clone()  # (B, seq_len, 5120) bf16

        # Step 4: Replace <|image_pad|> positions with visual embeddings
        for i in range(B):
            mask_positions = image_token_mask[i].nonzero(as_tuple=True)[0]
            n_img = mask_positions.shape[0]
            assert n_img == 512, f"Expected 512 image tokens, got {n_img}"
            text_embeds[i, mask_positions] = visual_embeds[i]

        # Step 5: Forward through LLM with inputs_embeds
        outputs = self.llm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        return outputs.loss

    @torch.no_grad()
    def generate(self, volume, input_ids, attention_mask, image_token_mask,
                 max_new_tokens=600, temperature=0.0):
        """
        Generate response for evaluation.

        Args:
            volume: (1, 1, 128, 128, 128)
            input_ids: (1, prompt_len) — prompt tokens only (no response)
            attention_mask: (1, prompt_len)
            image_token_mask: (1, prompt_len)
            max_new_tokens: max tokens to generate
            temperature: 0.0 for greedy decoding
        Returns:
            generated_ids: (1, prompt_len + new_tokens)
        """
        B = volume.shape[0]
        assert B == 1, "Generation only supports batch size 1"

        # Extract visual features
        visual_features = self.encoder(volume)
        proj_dtype = next(self.projector.parameters()).dtype
        visual_embeds = self.projector(visual_features.to(proj_dtype)).to(torch.bfloat16)

        # Get text embeddings and replace image tokens
        embed_fn = self.llm.get_input_embeddings()
        text_embeds = embed_fn(input_ids).clone()
        mask_positions = image_token_mask[0].nonzero(as_tuple=True)[0]
        text_embeds[0, mask_positions] = visual_embeds[0]

        # Generate
        gen_kwargs = {
            'inputs_embeds': text_embeds,
            'attention_mask': attention_mask,
            'max_new_tokens': max_new_tokens,
            'use_cache': True,
            'pad_token_id': self.llm.config.eos_token_id,
        }
        if temperature == 0.0:
            gen_kwargs['do_sample'] = False
        else:
            gen_kwargs['do_sample'] = True
            gen_kwargs['temperature'] = temperature

        outputs = self.llm.generate(**gen_kwargs)
        return outputs
