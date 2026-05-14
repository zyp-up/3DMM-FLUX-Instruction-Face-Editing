# -*- coding: utf-8 -*-
"""Conditional DECA Encoder used for Stage1 pretraining.

The image backbone produces one DECA feature vector, which attends to Qwen3
prompt features before the DECA-style regression head predicts expression and
jaw parameters.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from decalib.models import resnet as deca_resnet


STAGE1_EXTERNAL_STATE_PREFIXES = ("text_encoder.",)


def stage1_checkpoint_state_dict(model: nn.Module, exclude_external: bool = True) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Return Stage1 weights for checkpointing.

    Qwen text_encoder is frozen and recoverable from cfg.model.text_encoder_path,
    so new checkpoints store only the fine-tuned Stage1 parts by default.
    """
    state = model.state_dict()
    excluded = list(STAGE1_EXTERNAL_STATE_PREFIXES) if exclude_external else []
    if not excluded:
        return state, excluded
    return {
        k: v for k, v in state.items()
        if not any(k.startswith(prefix) for prefix in excluded)
    }, excluded


def load_stage1_checkpoint_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    allowed_missing_prefixes: Sequence[str] = STAGE1_EXTERNAL_STATE_PREFIXES,
) -> Tuple[List[str], List[str]]:
    """Load Stage1 weights while only tolerating intentionally external prefixes."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if not any(k.startswith(p) for p in allowed_missing_prefixes)]
    if real_missing or unexpected:
        raise RuntimeError(
            "Stage1 checkpoint mismatch: "
            f"missing(non-external)={real_missing[:20]} unexpected={unexpected[:20]}"
        )
    return list(missing), list(unexpected)


def _load_deca_resnet50_weights(model: nn.Module, deca_model_tar: str):
    """Load E_flame.encoder.* weights from the official DECA checkpoint."""
    ckpt = torch.load(deca_model_tar, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    enc_state = {}
    prefix = "E_flame.encoder."
    for k, v in ckpt.items():
        if k.startswith(prefix):
            enc_state[k[len(prefix):]] = v
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    return missing, unexpected


class ConditionalDECAEncoder(nn.Module):
    """Trainable image-text encoder that predicts (psi, jaw)."""

    PSI_DIM = 50
    JAW_DIM = 3

    def __init__(
        self,
        text_encoder_path: str,
        deca_model_tar: Optional[str] = None,
        feature_dim: int = 2048,
        text_hidden_size: int = 7680,  # FLUX.2 Klein: concat Qwen3 layers 9/18/27, 3 * 2560
        text_encoder_out_layers: Tuple[int, ...] = (9, 18, 27),
        num_attn_heads: int = 8,
        reg_hidden: int = 1024,
        psi_dim: Optional[int] = None,
        jaw_dim: Optional[int] = None,
        freeze_backbone: bool = False,
        freeze_text_encoder: bool = True,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        tokenizer_path: Optional[str] = None,
        load_text_encoder: bool = True,
    ):
        super().__init__()
        # Allow instance-level overrides for ablations.
        if psi_dim is not None:
            self.PSI_DIM = psi_dim
        if jaw_dim is not None:
            self.JAW_DIM = jaw_dim
        self.feature_dim = feature_dim
        self.reg_hidden = reg_hidden
        self.freeze_text_encoder = freeze_text_encoder
        self.text_encoder_dtype = text_encoder_dtype
        self.text_encoder_out_layers = tuple(text_encoder_out_layers)
        self.load_text_encoder = load_text_encoder
        self.text_encoder = None
        self.tokenizer = None

        # ---- 1) Image backbone initialized from DECA when available ----
        self.encoder = deca_resnet.load_ResNet50Model()  # output: (B, 2048)
        if deca_model_tar is not None and os.path.isfile(deca_model_tar):
            missing, unexpected = _load_deca_resnet50_weights(self.encoder, deca_model_tar)
            print(f"[ConditionalDECAEncoder] loaded DECA resnet50 weights "
                  f"from {deca_model_tar}, missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            print(f"[ConditionalDECAEncoder] WARNING: deca_model_tar not provided or not found, "
                  f"ResNet50 starts from random init.")

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

        # ---- 2) Qwen3 text encoder, usually frozen ----
        # Stage2 can reuse pipe.text_encoder and pass external text_hidden_states.
        if load_text_encoder:
            from transformers import AutoModel, AutoTokenizer
            # FLUX stores tokenizer files in a sibling tokenizer/ directory.
            resolved_tok_path = tokenizer_path
            if resolved_tok_path is None:
                has_tok_here = any(
                    os.path.isfile(os.path.join(text_encoder_path, f))
                    for f in ("tokenizer.json", "vocab.json", "tokenizer_config.json")
                )
                if has_tok_here:
                    resolved_tok_path = text_encoder_path
                else:
                    sibling = os.path.join(os.path.dirname(text_encoder_path.rstrip("/")), "tokenizer")
                    if os.path.isdir(sibling):
                        resolved_tok_path = sibling
                    else:
                        resolved_tok_path = text_encoder_path
            print(f"[ConditionalDECAEncoder] loading Qwen3 text encoder from {text_encoder_path} ...")
            print(f"[ConditionalDECAEncoder] loading Qwen3 tokenizer from {resolved_tok_path} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_tok_path)
            # Catch silent fallback to the wrong tokenizer.
            if self.tokenizer.vocab_size < 10000:
                raise RuntimeError(
                    f"Tokenizer vocab_size={self.tokenizer.vocab_size} does not look like Qwen3. "
                    f"Check that tokenizer_path={resolved_tok_path} contains tokenizer.json / vocab.json."
                )
            self.text_encoder = AutoModel.from_pretrained(
                text_encoder_path, torch_dtype=text_encoder_dtype,
            )
            if freeze_text_encoder:
                for p in self.text_encoder.parameters():
                    p.requires_grad_(False)
                self.text_encoder.eval()

        # ---- 3) Text-to-KV projection ----
        self.text_kv_proj = nn.Linear(text_hidden_size, feature_dim)

        # ---- 4) Cross-attention: Q=image_feat (1 token), KV=text_kv ----
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_attn_heads,
            batch_first=True,
        )
        self.attn_norm_q = nn.LayerNorm(feature_dim)
        self.attn_norm_kv = nn.LayerNorm(feature_dim)
        self.post_attn_norm = nn.LayerNorm(feature_dim)

        # ---- 5) Regression head aligned with DECA ResnetEncoder.layers ----
        #     (feature_dim) -> (reg_hidden) -> (PSI_DIM + JAW_DIM)
        self.outsize = self.PSI_DIM + self.JAW_DIM
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, reg_hidden),
            nn.ReLU(),
            nn.Linear(reg_hidden, self.outsize),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def trainable_parameters(self):
        """Return parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text exactly like FLUX.2 Klein: concat Qwen3 hidden layers 9/18/27."""
        if self.text_encoder is None:
            raise RuntimeError("ConditionalDECAEncoder was built with load_text_encoder=False; pass text_hidden_states directly.")
        ctx = torch.no_grad() if self.freeze_text_encoder else torch.enable_grad()
        with ctx:
            out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden = torch.stack([out.hidden_states[i] for i in self.text_encoder_out_layers], dim=1)
        b, n_layers, seq_len, hidden_dim = hidden.shape
        return hidden.permute(0, 2, 1, 3).reshape(b, seq_len, n_layers * hidden_dim)

    def tokenize(self, prompts, max_length: int = 64, device=None):
        """Tokenize prompts for standalone Stage1 use."""
        if self.tokenizer is None:
            raise RuntimeError("ConditionalDECAEncoder was built with load_text_encoder=False; use the external FLUX tokenizer.")
        enc = self.tokenizer(
            prompts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        ref_image: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ref_image:            (B, 3, 224, 224)
            input_ids:            optional (B, L), used for internal Qwen3 encoding
            attention_mask:       optional (B, L), used for internal Qwen3 encoding
            text_hidden_states:   optional cached prompt features, (B, L, 7680)

        Returns:
            psi_pred: (B, 50)
            jaw_pred: (B, 3)
        """
        # 1) Image features
        image_feat = self.encoder(ref_image)           # (B, 2048)

        # 2) Text features, preferring cached hidden states
        if text_hidden_states is None:
            assert input_ids is not None and attention_mask is not None, \
                "must provide either text_hidden_states or (input_ids, attention_mask)"
            text_hidden_states = self.encode_text(input_ids, attention_mask)
        # Match the backbone dtype.
        text_hidden_states = text_hidden_states.to(image_feat.dtype)

        text_kv = self.text_kv_proj(text_hidden_states)  # (B, L, 2048)

        # 3) Cross-attention
        q = self.attn_norm_q(image_feat).unsqueeze(1)       # (B, 1, 2048)
        kv = self.attn_norm_kv(text_kv)                     # (B, L, 2048)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)        # True means padding.
        attn_out, _ = self.cross_attn(
            query=q, key=kv, value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = image_feat + attn_out.squeeze(1)            # Residual fusion.
        fused = self.post_attn_norm(fused)

        # 4) Regression head
        params = self.layers(fused)                          # (B, PSI_DIM + JAW_DIM)
        psi_pred = params[:, : self.PSI_DIM]                 # (B, PSI_DIM)
        jaw_pred = params[:, self.PSI_DIM : self.PSI_DIM + self.JAW_DIM]  # (B, JAW_DIM)
        return psi_pred, jaw_pred

    # ------------------------------------------------------------------
    # Stage2 entry point: consume external text_hidden_states from pipe.text_encoder.
    # ------------------------------------------------------------------
    def predict_from_text_hidden(
        self,
        ref_image: torch.Tensor,
        text_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Named wrapper around forward(..., text_hidden_states=...) for Stage2.

        Args:
            ref_image: (B, 3, 224, 224)
            text_hidden_states: (B, L, text_hidden_size) -- FLUX.2 prompt_embeds, concat Qwen3 layers 9/18/27
            attention_mask:     (B, L), used as the cross-attention key padding mask
        """
        return self.forward(
            ref_image=ref_image,
            input_ids=None,
            attention_mask=attention_mask,
            text_hidden_states=text_hidden_states,
        )

    # ------------------------------------------------------------------
    # Stage2 memory optimization: release internal Qwen3 after switching to
    # predict_from_text_hidden(...).
    # ------------------------------------------------------------------
    def drop_text_encoder(self):
        if getattr(self, "text_encoder", None) is not None:
            del self.text_encoder
            self.text_encoder = None
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Keep frozen modules in eval mode.
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text_encoder and getattr(self, "text_encoder", None) is not None:
            self.text_encoder.eval()
        return self
