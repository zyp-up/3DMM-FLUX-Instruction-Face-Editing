# -*- coding: utf-8 -*-
"""
Conditional DECA Encoder (Stage1 预训练用)

结构:
    ResNet50 (DECA 预训练初始化) → image_feat (B, feature_dim)
        注: 原版 decalib.models.resnet.load_ResNet50Model 输出已经是
            AdaptiveAvgPool2d + flatten 后的单向量 (B, 2048), 不是 feature map.
    Qwen3 Text Encoder (冻结)   → text_hidden (B, L, text_hidden_size)
                                    → Linear → text_kv (B, L, feature_dim)
    一次 Cross-Attn: Q=image_feat.unsqueeze(1) (B, 1, feature_dim)
                     KV=text_kv (B, L, feature_dim) → fused (B, feature_dim)
        (与原版回归头 "吃单向量" 的设计天然对齐, 所以 Q 保持 1 token)
    原版 DECA 回归头: feature_dim → reg_hidden → (PSI_DIM + JAW_DIM) -> split → psi (B, PSI_DIM), jaw (B, JAW_DIM)

说明:
  - forward 接口: 模型内置 Qwen3, 调用时只需传 ref_image + input_ids + attention_mask.
  - Qwen3 参数全程冻结 + bf16 推理; 训练时 set_requires_grad 只覆盖可训练部分.
  - 回归头结构严格对齐 decalib.models.encoders.ResnetEncoder.layers (Linear+ReLU+Linear).
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
    """从 DECA 官方 deca_model.tar 里抽出 E_flame.encoder.* 的权重装到我们的 ResNet50 上.

    DECA 的 state_dict 里 encoder 部分的 key 前缀是 'E_flame.encoder.',
    我们把这一层前缀去掉后 load 到 resnet50 (deca_resnet.load_ResNet50Model() 返回) 即可.
    """
    ckpt = torch.load(deca_model_tar, map_location="cpu")
    # DECA 的 checkpoint 直接就是 state_dict (key 带 'E_flame.xxx')
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
    """可训练的 Conditional DECA Encoder. 接收图像 + 文本, 预测 (psi, jaw)."""

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
        # 允许实例级覆盖类默认值, 但一般使用 PSI_DIM=50, JAW_DIM=3
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

        # ---- 1) 图像 backbone: ResNet50, DECA 预训练初始化 ----
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

        # ---- 2) 文本编码器: Qwen3 (冻结, bf16) ----
        # Stage2 训练/推理会复用 FLUX pipeline 的 pipe.text_encoder，
        # 因此可设置 load_text_encoder=False，仅保留吃外部 text_hidden_states 的能力。
        if load_text_encoder:
            from transformers import AutoModel, AutoTokenizer
            # FLUX 目录布局: text_encoder/ 里只有模型权重, tokenizer 在隻壁 tokenizer/
            # 如果没显式指定 tokenizer_path, 就自动查一下兄弟目录
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
                        resolved_tok_path = text_encoder_path  # 既然没找到, 也只能给 AutoTokenizer 自己报错
            print(f"[ConditionalDECAEncoder] loading Qwen3 text encoder from {text_encoder_path} ...")
            print(f"[ConditionalDECAEncoder] loading Qwen3 tokenizer from {resolved_tok_path} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_tok_path)
            # sanity check: 正常 Qwen3 tokenizer vocab ~151936, fallback 到 GPT-2 会是 50257
            if self.tokenizer.vocab_size < 10000:
                raise RuntimeError(
                    f"Tokenizer vocab_size={self.tokenizer.vocab_size} 看起来不是 Qwen3 ("
                    f"期望 >=150000). 很可能 AutoTokenizer 静默回退到了 GPT-2. "
                    f"检查 tokenizer_path={resolved_tok_path} 是否含 tokenizer.json / vocab.json."
                )
            self.text_encoder = AutoModel.from_pretrained(
                text_encoder_path, torch_dtype=text_encoder_dtype,
            )
            if freeze_text_encoder:
                for p in self.text_encoder.parameters():
                    p.requires_grad_(False)
                self.text_encoder.eval()

        # ---- 3) 文本 → KV 投影 ----
        self.text_kv_proj = nn.Linear(text_hidden_size, feature_dim)

        # ---- 4) 一次 Cross-Attn: Q=image_feat (1 token), KV=text_kv ----
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_attn_heads,
            batch_first=True,
        )
        self.attn_norm_q = nn.LayerNorm(feature_dim)
        self.attn_norm_kv = nn.LayerNorm(feature_dim)
        self.post_attn_norm = nn.LayerNorm(feature_dim)

        # ---- 5) 回归头: 严格对齐 decalib.models.encoders.ResnetEncoder.layers ----
        #     (feature_dim) -> (reg_hidden) -> (PSI_DIM + JAW_DIM)
        self.outsize = self.PSI_DIM + self.JAW_DIM
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, reg_hidden),
            nn.ReLU(),
            nn.Linear(reg_hidden, self.outsize),
        )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def trainable_parameters(self):
        """返回需要优化的参数(便于外部构造 optimizer param_groups)."""
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
        """便捷 tokenize 方法. 训练时一般用不到 (我们在 Dataset 里已经 tokenize 过)."""
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
    # 前向
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
            input_ids:            (B, L) 可选 -- 若提供就现场跑 Qwen3
            attention_mask:       (B, L) 可选 -- 若提供就现场跑 Qwen3
            text_hidden_states:   (B, L, 7680) 可选 -- 若提供就直接用(已缓存)

        Returns:
            psi_pred: (B, 50)
            jaw_pred: (B, 3)
        """
        # 1) 图像特征
        image_feat = self.encoder(ref_image)           # (B, 2048)

        # 2) 文本特征 (缓存优先; 否则现场编码)
        if text_hidden_states is None:
            assert input_ids is not None and attention_mask is not None, \
                "must provide either text_hidden_states or (input_ids, attention_mask)"
            text_hidden_states = self.encode_text(input_ids, attention_mask)
        # 统一到主 dtype
        text_hidden_states = text_hidden_states.to(image_feat.dtype)

        text_kv = self.text_kv_proj(text_hidden_states)  # (B, L, 2048)

        # 3) 一次 Cross-Attn
        q = self.attn_norm_q(image_feat).unsqueeze(1)       # (B, 1, 2048)
        kv = self.attn_norm_kv(text_kv)                     # (B, L, 2048)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)        # True 表示 padding
        attn_out, _ = self.cross_attn(
            query=q, key=kv, value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = image_feat + attn_out.squeeze(1)            # (B, 2048) 残差
        fused = self.post_attn_norm(fused)

        # 4) 回归头
        params = self.layers(fused)                          # (B, PSI_DIM + JAW_DIM)
        psi_pred = params[:, : self.PSI_DIM]                 # (B, PSI_DIM)
        jaw_pred = params[:, self.PSI_DIM : self.PSI_DIM + self.JAW_DIM]  # (B, JAW_DIM)
        return psi_pred, jaw_pred

    # ------------------------------------------------------------------
    # Stage2 专用入口: 直接吃外部 text_hidden_states (例如来自 pipe.text_encoder)
    # 这样 Stage1 / Stage2 可以共用同一份 Qwen3 输出, 避免语义漂移。
    # ------------------------------------------------------------------
    def predict_from_text_hidden(
        self,
        ref_image: torch.Tensor,
        text_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """等价于 forward(ref_image, text_hidden_states=...), 显式命名, Stage2 用。

        Args:
            ref_image: (B, 3, 224, 224)
            text_hidden_states: (B, L, text_hidden_size) -- FLUX.2 prompt_embeds, concat Qwen3 layers 9/18/27
            attention_mask:     (B, L) -- 对应 text 的 mask, 用作 cross-attn 的 key_padding_mask
        """
        return self.forward(
            ref_image=ref_image,
            input_ids=None,
            attention_mask=attention_mask,
            text_hidden_states=text_hidden_states,
        )

    # ------------------------------------------------------------------
    # Stage2 优化: 释放内部 Qwen3, 之后仅通过 predict_from_text_hidden(...) 调用
    # 原因: Stage2 里 pipe.text_encoder 与本模块的 self.text_encoder 是同一份
    # Qwen3 权重(bit-exact), 冗余显存 ~5GB, 释放后由外部统一前向。
    # ------------------------------------------------------------------
    def drop_text_encoder(self):
        if getattr(self, "text_encoder", None) is not None:
            del self.text_encoder
            self.text_encoder = None
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 训练/评估模式: 保持冻结模块始终 eval
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text_encoder and getattr(self, "text_encoder", None) is not None:
            self.text_encoder.eval()
        return self