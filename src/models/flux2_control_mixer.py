# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Flux2ControlMixer(nn.Module):
    """CMM for FLUX.2: maps 9-channel target/reference controls to tokens."""
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        joint_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.joint_dim = joint_dim

        self.tgt_stem = nn.Sequential(
            ConvGNAct(9, hidden_dim // 4, stride=1),
            ConvGNAct(hidden_dim // 4, hidden_dim // 2, stride=2),
            ConvGNAct(hidden_dim // 2, hidden_dim, stride=2),
        )
        self.ref_stem = nn.Sequential(
            ConvGNAct(9, hidden_dim // 4, stride=1),
            ConvGNAct(hidden_dim // 4, hidden_dim // 2, stride=2),
            ConvGNAct(hidden_dim // 2, hidden_dim, stride=2),
        )

        self.q_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, joint_dim)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def _to_tokens(self, x: torch.Tensor, token_hw) -> torch.Tensor:
        if token_hw is None:
            raise ValueError("token_hw is required: CMM tokens must align with current FLUX latent token grid")
        x = nn.functional.adaptive_avg_pool2d(x, token_hw)
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2).contiguous()

    def forward(self, tgt_control: torch.Tensor, ref_control: torch.Tensor, token_hw) -> torch.Tensor:
        tgt_feat = self.tgt_stem(tgt_control)
        ref_feat = self.ref_stem(ref_control)

        q = self.q_norm(self._to_tokens(tgt_feat, token_hw=token_hw))
        kv = self.kv_norm(self._to_tokens(ref_feat, token_hw=token_hw))

        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        fused = q + self.gate * attn_out
        fused = fused + self.ff(fused)

        return self.out_proj(fused)
