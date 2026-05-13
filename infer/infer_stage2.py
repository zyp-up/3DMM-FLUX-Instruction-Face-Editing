# -*- coding: utf-8 -*-
"""
Stage2 推理脚本:
输入一张参考图 + 文本指令，输出:
1) Stage1 预测得到的目标控制图 D_T (rendered/normal/albedo)
2) 参考控制图 D_R (rendered/normal/albedo)
3) FLUX.2 生成图像 (默认已开启 Reference Control Guidance / RCG)
4) 中间 tensor 便于调试 (control_9ch / tokens / latent)

用法 (以下 4 条示例命令自包含、可独立复现, 不依赖任何默认值兜底):
# 1）默认用法: 启用 RCG, λ 从 yaml 读取 (默认 3.0)
python infer/infer_stage2.py \
  --config configs/infer_stage2.yaml \
  --ref 原图.png \
  --prompt "make her burst into laughter" \
  --stage2_ckpt ./checkpoints/stage2/stage2-20260508-232246/best-step-3480.pt \
  --output_dir ./output_stage2

# 2）命令行 sweep RCG 系数 λ (说明: λ 接受任意 float, 以下为推荐起点)
for LAM in 0.0 1.0 3.0 5.0 7.0; do
  python infer/infer_stage2.py \
    --config configs/infer_stage2.yaml \
    --ref 原图.png \
    --prompt "make her burst into laughter" \
    --stage2_ckpt ./checkpoints/stage2/stage2-20260507-161431/best-step-1920.pt \
    --rcg_lambda ${LAM} \
    --output_dir ./output_stage2/sweep_lambda_${LAM}
done

# 3）关掉 RCG 做 A/B 对照 (仍使用 ref_latents + CMM(D_T, D_R), 只是不做两路外推)
python infer/infer_stage2.py \
  --config configs/infer_stage2.yaml \
  --ref 原图.png \
  --prompt "make her burst into laughter" \
  --stage2_ckpt ./checkpoints/stage2/stage2-20260507-161431/best-step-1920.pt \
  --rcg_enabled false \
  --output_dir ./output_stage2/no_rcg

# 4）临时覆盖 yaml 任意字段 (例: 一次性改分辨率/步数/RCG系数)
python infer/infer_stage2.py \
  --config configs/infer_stage2.yaml \
  --ref 原图.png \
  --prompt "smile" \
  --stage2_ckpt ./checkpoints/stage2/stage2-20260507-161431/best-step-1920.pt \
  --opts sampling.num_inference_steps=28 sampling.height=512 sampling.width=512 \
         sampling.rcg.lambda=2.5
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.utils import save_image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_NUMPY_LEGACY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}
for _name, _target in _NUMPY_LEGACY_ALIASES.items():
    if not hasattr(np, _name):
        setattr(np, _name, _target)

from diffusers import DiffusionPipeline
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import datasets as deca_dataset

from src.models.conditional_deca_encoder import ConditionalDECAEncoder, load_stage1_checkpoint_state_dict
from src.models.flux2_control_mixer import Flux2ControlMixer

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _coerce_scalar(v: str):
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if v.lower() in {"null", "none", "~"}:
        return None
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v


def apply_opts(cfg: dict, opts):
    if not opts:
        return cfg
    for kv in opts:
        if "=" not in kv:
            raise ValueError(f"--opts 项必须形如 key.path=value, 收到: {kv!r}")
        k, v = kv.split("=", 1)
        cur = cfg
        parts = k.strip().split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = _coerce_scalar(v.strip())
    return cfg


def configure_deca_cfg(deca_data_dir: str):
    deca_cfg.pretrained_modelpath = os.path.join(deca_data_dir, "deca_model.tar")
    m = deca_cfg.model
    m.topology_path = os.path.join(deca_data_dir, "head_template.obj")
    m.dense_template_path = os.path.join(deca_data_dir, "texture_data_256.npy")
    m.fixed_displacement_path = os.path.join(deca_data_dir, "fixed_displacement_256.npy")
    m.flame_model_path = os.path.join(deca_data_dir, "generic_model.pkl")
    m.flame_lmk_embedding_path = os.path.join(deca_data_dir, "landmark_embedding.npy")
    m.face_mask_path = os.path.join(deca_data_dir, "uv_face_mask.png")
    m.face_eye_mask_path = os.path.join(deca_data_dir, "uv_face_eye_mask.png")
    m.mean_tex_path = os.path.join(deca_data_dir, "mean_texture.jpg")
    m.tex_path = os.path.join(deca_data_dir, "FLAME_texture.npz")
    m.tex_type = "FLAME"
    m.use_tex = True
    deca_cfg.rasterizer_type = "pytorch3d"


def load_deca(device: str, deca_data_dir: str):
    configure_deca_cfg(deca_data_dir)
    deca = DECA(config=deca_cfg, device=device)
    deca.eval()
    return deca


def freeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def build_stage1_from_ckpt(
    stage1_ckpt_path: str,
    device: str,
    stage2_ckpt_path: str = None,
):
    """Build ConditionalDECAEncoder:
      - structure cfg  ← stage1 ckpt ['cfg']['model']  (只用于实例化)
      - weights        ← stage2 ckpt ['stage1_model']  (端到端微调后的版本, 首选)
                         fallback → stage1 ckpt ['model']  (只有旧 ckpt 或调试用)
    这样可以保证推理时使用的 conditional encoder 与 CMM 是同一次训练联合优化的版本。
    """
    ckpt = torch.load(stage1_ckpt_path, map_location="cpu")
    if "cfg" not in ckpt or "model" not in ckpt["cfg"]:
        raise ValueError("Stage1 checkpoint missing cfg.model")
    cfg_model = dict(ckpt["cfg"]["model"])
    cfg_model["text_encoder_dtype"] = DTYPE_MAP[cfg_model.get("text_encoder_dtype", "bf16")]
    cfg_model["text_encoder_path"] = resolve_path(cfg_model["text_encoder_path"])
    cfg_model["load_text_encoder"] = False

    deca_model_tar = cfg_model.get("deca_model_tar")
    if deca_model_tar is not None:
        deca_model_tar = resolve_path(deca_model_tar)
    fallback = str((PROJECT_ROOT / "data" / "deca_model.tar").resolve())
    if deca_model_tar is None or (not os.path.isfile(deca_model_tar)):
        deca_model_tar = fallback
    cfg_model["deca_model_tar"] = deca_model_tar

    model = ConditionalDECAEncoder(**cfg_model).to(device)

    state_dict = None
    weight_src = None
    allowed_missing = ["text_encoder."]
    if stage2_ckpt_path is not None and os.path.isfile(stage2_ckpt_path):
        s2 = torch.load(stage2_ckpt_path, map_location="cpu")
        if isinstance(s2, dict) and "stage1_model" in s2:
            state_dict = s2["stage1_model"]
            weight_src = f"stage2_ckpt:{Path(stage2_ckpt_path).name}['stage1_model']"
    if state_dict is None:
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        allowed_missing = ckpt.get("excluded_state_dict_prefixes", ["text_encoder."]) if isinstance(ckpt, dict) else ["text_encoder."]
        weight_src = f"stage1_ckpt:{Path(stage1_ckpt_path).name}['model']  (FALLBACK)"
        print("[Stage1][WARN] stage2 ckpt 里没找到 'stage1_model' key, 回退到 stage1 老权重; "
              "推理结果与端到端训练的 CMM 可能不完全对齐。")

    missing, unexpected = load_stage1_checkpoint_state_dict(model, state_dict, allowed_missing)
    print(f"[Stage1] weights loaded from {weight_src}: "
          f"missing_external={len(missing)}, unexpected={len(unexpected)}")
    print("[Stage1] built without internal Qwen; reusing pipe.text_encoder for text features.")
    model.eval()
    return model


def build_control_mixer(cfg: dict, stage2_ckpt_path: str, device: str, dtype: torch.dtype = torch.float32):
    m = cfg["model"]
    mixer = Flux2ControlMixer(
        hidden_dim=m["control_mixer_hidden_dim"],
        num_heads=m["control_mixer_heads"],
        joint_dim=m["control_joint_dim"],
        dropout=m["control_dropout"],
    )
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("control_mixer", ckpt)
    missing, unexpected = mixer.load_state_dict(state, strict=False)
    print(f"[Stage2] control_mixer loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    # CMM 必须与 pipe (FLUX2 transformer) 同 dtype, 否则 context_embedder 的 F.linear 会 dtype mismatch
    mixer.to(device=device, dtype=dtype)
    mixer.eval()
    return mixer


def _pack_latents_fallback(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
    latents = latents.view(b, (h // 2) * (w // 2), c * 4)
    return latents


def _unpack_latents_fallback(tokens: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
    b, n, cc = tokens.shape
    c = cc // 4
    x = tokens.view(b, token_h, token_w, c, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(b, c, token_h * 2, token_w * 2)


def _prepare_img_ids_fallback(batch_size: int, token_h: int, token_w: int, device, dtype, t_coord: float = 0.0):
    # 对齐官方 flux2.sampling.prc_img: x_ids = cartesian_prod(t, h, w, l)。
    ys = torch.arange(token_h, device=device, dtype=dtype)
    xs = torch.arange(token_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    ids = torch.stack(
        [torch.full_like(yy, float(t_coord)), yy, xx, torch.zeros_like(yy)],
        dim=-1,
    )
    ids = ids.view(1, token_h * token_w, 4)
    return ids.repeat(batch_size, 1, 1)


def _compute_flux_mu(scheduler, image_seq_len: int, base_seq_len: int = 256, max_seq_len: int = 4096,
                     base_shift: float = 0.5, max_shift: float = 1.15) -> float:
    """Compute the dynamic shift (mu) for FLUX scheduler based on image sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def load_ref_native_image(path: str, height: int, width: int, device: str) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def encode_image_to_flux_tokens(pipe, image: torch.Tensor, sample: bool = False, t_coord: float = 0.0):
    posterior = pipe.vae.encode(image)
    if hasattr(posterior, "latent_dist"):
        latents = posterior.latent_dist.sample() if sample else posterior.latent_dist.mode()
    elif isinstance(posterior, (tuple, list)):
        latents = posterior[0]
    else:
        latents = posterior.latents
    packed = _pack_latents_fallback(latents)
    if hasattr(pipe.vae, "bn") and hasattr(pipe.vae.bn, "running_mean"):
        eps = getattr(pipe.vae.config, "batch_norm_eps", 1e-4)
        num_features = pipe.vae.bn.running_mean.numel()
        if num_features == packed.shape[-1]:
            mean = pipe.vae.bn.running_mean.view(1, 1, -1).to(packed.device, packed.dtype)
            std = torch.sqrt(pipe.vae.bn.running_var.view(1, 1, -1).to(packed.device, packed.dtype) + eps)
            packed = (packed - mean) / std
        elif num_features == latents.shape[1]:
            mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype) + eps)
            latents = (latents - mean) / std
            packed = _pack_latents_fallback(latents)
    token_h = latents.shape[2] // 2
    token_w = latents.shape[3] // 2
    img_ids = _prepare_img_ids_fallback(image.shape[0], token_h, token_w, image.device, packed.dtype, t_coord=t_coord)
    return packed, img_ids, (token_h, token_w)


@torch.no_grad()
def encode_flux_prompts(pipe, prompts, max_length: int, device: str, dtype: torch.dtype):
    """与训练端 encode_flux_prompts 简直一致:
    - prompt_embeds_flux (B, L, 7680): FLUX.2 concat Qwen3 layers 9/18/27, shared by transformer and Stage1
    - text_attn_mask     (B, L)      : Stage1 cross-attn 用
    - txt_ids            (B, L, 4)   : FLUX rope ids
    """
    prompt_embeds_flux, _ = pipe.encode_prompt(
        prompt=list(prompts),
        device=device,
        max_sequence_length=max_length,
    )
    prompt_embeds_flux = prompt_embeds_flux.to(dtype)
    txt_ids = torch.zeros(
        prompt_embeds_flux.shape[0], prompt_embeds_flux.shape[1], 4, dtype=dtype, device=device,
    )
    txt_ids[..., 3] = torch.arange(prompt_embeds_flux.shape[1], device=device, dtype=dtype)

    texts = [
        pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]
    tokenized = pipe.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    text_attn_mask = tokenized["attention_mask"].to(device)
    return prompt_embeds_flux, prompt_embeds_flux, text_attn_mask, txt_ids


def clone_code_dict(code: dict):
    out = {}
    for k, v in code.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def tensor01(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu().clamp(0.0, 1.0)


@torch.no_grad()
def run_infer(args, cfg):
    device = args.device
    flux_dtype = DTYPE_MAP[cfg["flux"].get("torch_dtype", "bf16")]

    pipe = DiffusionPipeline.from_pretrained(
        resolve_path(cfg["flux"]["model_path"]),
        torch_dtype=flux_dtype,
    ).to(device)
    freeze_module(pipe.text_encoder)
    freeze_module(pipe.transformer)
    freeze_module(pipe.vae)

    deca = load_deca(device=device, deca_data_dir=resolve_path(cfg["deca"]["data_dir"]))

    stage1_model = build_stage1_from_ckpt(
        resolve_path(args.stage1_ckpt),
        device=device,
        stage2_ckpt_path=resolve_path(args.stage2_ckpt),
    )
    mixer = build_control_mixer(cfg, resolve_path(args.stage2_ckpt), device=device, dtype=flux_dtype)

    deca_test_size = int(cfg["deca"].get("test_size", args.deca_test_size))
    dataset = deca_dataset.TestData([resolve_path(args.ref)], iscrop=True, size=deca_test_size)
    sample = dataset[0]

    ref_crop = sample["image"].unsqueeze(0).to(device)

    # ① 统一的文本编码: FLUX.2 prompt_embeds (7680) 同时供 transformer 和 Stage1 使用
    max_text_len = int(cfg.get("text", {}).get("max_sequence_length", 512))
    prompt_embeds, text_hidden_for_stage1, text_attn_mask, txt_ids = encode_flux_prompts(
        pipe, [args.prompt], max_text_len, device, flux_dtype,
    )

    # ② DECA encode ref
    code_ref = deca.encode(ref_crop)

    # ③ Stage1 Conditional Encoder: 走 predict_from_text_hidden, 与 Stage1 训练时 bit-exact 对齐
    psi_pred, jaw_pred = stage1_model.predict_from_text_hidden(
        ref_image=ref_crop,
        text_hidden_states=text_hidden_for_stage1,
        attention_mask=text_attn_mask,
    )

    code_tgt = clone_code_dict(code_ref)
    code_tgt["exp"] = psi_pred.to(code_ref["exp"].dtype)
    code_tgt["pose"] = code_ref["pose"].clone()
    code_tgt["pose"][:, 3:6] = jaw_pred.to(code_ref["pose"].dtype)

    op_ref, _ = deca.decode(code_ref)
    op_tgt, _ = deca.decode(code_tgt)

    use_alpha_mask = bool(cfg.get("data", {}).get("use_alpha_mask", True))
    if use_alpha_mask and "alpha_images" in op_ref:
        alpha_ref = op_ref["alpha_images"].float()
        ref_rendered = op_ref["rendered_images"].float() * alpha_ref
        ref_normal = op_ref["normal_images"].float() * alpha_ref
        ref_albedo = op_ref["albedo_images"].float() * alpha_ref
    else:
        ref_rendered = op_ref["rendered_images"]
        ref_normal = op_ref["normal_images"]
        ref_albedo = op_ref["albedo_images"]
    if use_alpha_mask and "alpha_images" in op_tgt:
        alpha_tgt = op_tgt["alpha_images"].float()
        tgt_rendered = op_tgt["rendered_images"].float() * alpha_tgt
        tgt_normal = op_tgt["normal_images"].float() * alpha_tgt
        tgt_albedo = op_tgt["albedo_images"].float() * alpha_tgt
    else:
        tgt_rendered = op_tgt["rendered_images"]
        tgt_normal = op_tgt["normal_images"]
        tgt_albedo = op_tgt["albedo_images"]

    ref_control = torch.cat([ref_rendered, ref_normal, ref_albedo], dim=1)
    tgt_control = torch.cat([tgt_rendered, tgt_normal, tgt_albedo], dim=1)

    infer_size = int(args.height)
    ref_control_r = torch.nn.functional.interpolate(
        ref_control, size=(infer_size, infer_size), mode="bilinear", align_corners=False
    )
    tgt_control_r = torch.nn.functional.interpolate(
        tgt_control, size=(infer_size, infer_size), mode="bilinear", align_corners=False
    )

    # ---- RCG (Reference Control Guidance) 对齐 ControlFace 官方 pipeline_point.py ----
    # 官方公式:
    #   eps_ref = eps_theta(x_t, t, e_text, CMM(D_R, D_R))   # 两路都喂 D_R → "不变化" 预测
    #   eps_tgt = eps_theta(x_t, t, e_text, CMM(D_T, D_R))   # 目标表情预测
    #   eps = eps_ref + λ * (eps_tgt - eps_ref)
    rcg_cfg = cfg.get("sampling", {}).get("rcg", {}) or {}
    rcg_enabled = bool(rcg_cfg.get("enabled", False))
    rcg_lambda = float(rcg_cfg.get("lambda", 1.0))
    batch_length = 2 if rcg_enabled else 1

    # 文本特征在前面已经 encode 过一次 (prompt_embeds / txt_ids), 这里直接用。
    encoder_hidden_states = prompt_embeds.to(flux_dtype)
    txt_ids_full = txt_ids

    h = int(args.height)
    w = int(args.width)
    latent_h = h // 8
    latent_w = w // 8
    # 与训练端 encode_image_to_flux_tokens 一致:
    #   token_h = latents.shape[2] // 2 = H/8/2 = H/16  (不区分 no_pack/pack 分支)
    #   in_channels (DiT) = vae_latent_channels * 4 (因为 2x2 pack)
    in_channels = int(pipe.transformer.config.in_channels)
    vae_latent_channels = int(getattr(pipe.vae.config, "latent_channels", in_channels // 4))
    token_h = latent_h // 2
    token_w = latent_w // 2
    seq = token_h * token_w
    print(f"[FLUX] in_channels={in_channels}, vae_latent_channels={vae_latent_channels}, "
          f"token_grid=({token_h},{token_w}), latent_grid=({latent_h},{latent_w})")

    latents = torch.randn(
        1, seq, in_channels, dtype=flux_dtype, device=device, generator=torch.Generator(device).manual_seed(args.seed)
    )

    ref_native = load_ref_native_image(resolve_path(args.ref), h, w, device).to(flux_dtype)
    ref_latents, ref_img_ids, ref_token_hw = encode_image_to_flux_tokens(
        pipe, ref_native * 2.0 - 1.0, sample=False, t_coord=10.0
    )
    if ref_token_hw != (token_h, token_w):
        raise ValueError(f"ref latent token grid {ref_token_hw} != sampling token grid {(token_h, token_w)}")

    img_ids = _prepare_img_ids_fallback(1, token_h, token_w, device, latents.dtype, t_coord=0.0)

    # CMM 已经 to(flux_dtype), 9ch 控制图 (fp32 DECA 输出) 也要对齐, 否则 Conv2d 会 dtype mismatch。
    # control tokens concat 到 FLUX hidden_states/noise token 侧, 维度必须等于 in_channels=128。
    control_tokens_tgt = mixer(tgt_control_r.to(flux_dtype), ref_control_r.to(flux_dtype), token_hw=(token_h, token_w))
    if rcg_enabled:
        control_tokens_ref = mixer(ref_control_r.to(flux_dtype), ref_control_r.to(flux_dtype), token_hw=(token_h, token_w))
    else:
        control_tokens_ref = None

    control_ids_tgt = _prepare_img_ids_fallback(1, token_h, token_w, device, img_ids.dtype, t_coord=20.0)
    img_ids_tgt = torch.cat([img_ids, ref_img_ids, control_ids_tgt], dim=1)
    if rcg_enabled:
        control_ids_ref = _prepare_img_ids_fallback(1, token_h, token_w, device, img_ids.dtype, t_coord=20.0)
        img_ids_ref = torch.cat([img_ids, ref_img_ids, control_ids_ref], dim=1)
        img_ids_full = torch.cat([img_ids_ref, img_ids_tgt], dim=0)
        encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
        txt_ids_full = txt_ids_full.repeat(2, 1, 1)
    else:
        img_ids_full = img_ids_tgt
    scheduler = pipe.scheduler

    # FLUX.2 scheduler 开启 use_dynamic_shifting=True, 需要按当前 image_seq_len 计算 mu
    # 并以 sigmas=linspace(1.0, 1/N, N) + mu 的方式调 set_timesteps (官方 FluxPipeline 做法)。
    image_seq_len = seq  # token_h * token_w
    sched_cfg = getattr(scheduler, "config", None)
    base_seq_len = int(getattr(sched_cfg, "base_image_seq_len", 256)) if sched_cfg else 256
    max_seq_len = int(getattr(sched_cfg, "max_image_seq_len", 4096)) if sched_cfg else 4096
    base_shift = float(getattr(sched_cfg, "base_shift", 0.5)) if sched_cfg else 0.5
    max_shift = float(getattr(sched_cfg, "max_shift", 1.15)) if sched_cfg else 1.15

    # 优先用 diffusers 自带 calculate_shift, 不可用则回退手写公式
    mu = None
    try:
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift as _calc_shift
        mu = float(_calc_shift(image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift))
    except Exception:
        mu = _compute_flux_mu(scheduler, image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift)

    sigmas = np.linspace(1.0, 1.0 / args.num_inference_steps, args.num_inference_steps)
    try:
        scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
    except TypeError:
        # 更老版本的 diffusers scheduler 不接受 mu kwarg
        scheduler.set_timesteps(args.num_inference_steps, device=device)

    if rcg_enabled:
        print(f"[RCG] enabled: lambda={rcg_lambda} (FLUX.2-klein 非蒸馏模型, transformer guidance 固定为 None)")
    else:
        print("[RCG] disabled: 单路 image-editing 条件 (latents + ref_latents + CMM(D_T, D_R))")

    # 训练端 (train_stage2.py L426) 使用 t = torch.rand(...) ∈ [0, 1] 作为 timestep 送入 transformer；
    # 而 FLUX 的 FlowMatchEulerDiscreteScheduler.timesteps 范围是 [0, num_train_timesteps] (通常 1000)。
    # 若直接喂原始 t 给 transformer, time embedding 量级错位 1000 倍, 产生的 flow 是随机方向, 解码出来是纯噪声。
    # 这里把 t 归一化到 [0, 1] 再送入 transformer, 与训练语义严格对齐; scheduler.step 仍用原始 t。
    num_train_timesteps = float(getattr(scheduler.config, "num_train_timesteps", 1000))
    for t in scheduler.timesteps:
        t_norm = float(t) / num_train_timesteps
        t_in = torch.tensor([t_norm], device=device, dtype=latents.dtype).expand(batch_length)
        if rcg_enabled:
            latent_ref = torch.cat([latents, ref_latents, control_tokens_ref], dim=1)
            latent_tgt = torch.cat([latents, ref_latents, control_tokens_tgt], dim=1)
            latent_model_input = torch.cat([latent_ref, latent_tgt], dim=0)
        else:
            latent_model_input = torch.cat([latents, ref_latents, control_tokens_tgt], dim=1)
        # FLUX.2-klein-base 非蒸馏 (transformer.config.guidance_embeds=False), 必须传 guidance=None;
        # 原代码传 torch.full((B,), 4.0) 会破坏 timestep embedding, 输出乱码 (诊断 #6 验证)
        pred = pipe.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=encoder_hidden_states,
            timestep=t_in,
            img_ids=img_ids_full,
            txt_ids=txt_ids_full,
            guidance=None,
            return_dict=True,
        ).sample
        pred = pred[:, :latents.shape[1], :]
        if rcg_enabled:
            # 官方: noise_pred_ref, noise_pred_all = noise_pred.chunk(2)
            noise_pred_ref, noise_pred_tgt = pred.chunk(2, dim=0)
            pred = noise_pred_ref + rcg_lambda * (noise_pred_tgt - noise_pred_ref)
        latents = scheduler.step(pred, t, latents, return_dict=True).prev_sample

    # =========== VAE 反归一化 + unpack: 严格镜像 train_stage2.encode_image_to_flux_tokens =========
    # 训练端两种分支 (以 num_features 为判据):
    #   (A) num_features == packed.last_dim (= in_channels)  → 在 packed (B, seq, 128) 上做
    #       正向: packed = (packed - mean) / std
    #       逆向: packed = packed * std + mean  → unpack → latents
    #   (B) num_features == latents.dim1 (= vae_latent_channels)  → 在 latents (B, 32, H, W) 上做
    #       正向: latents = (latents - mean) / std → pack → packed
    #       逆向: unpack packed → latents → latents = latents * std + mean
    packed = latents  # (1, seq, in_channels)
    lat_4d = None
    if hasattr(pipe.vae, "bn") and hasattr(pipe.vae.bn, "running_mean"):
        eps = getattr(pipe.vae.config, "batch_norm_eps", 1e-4)
        num_features = pipe.vae.bn.running_mean.numel()
        if num_features == packed.shape[-1]:
            # 分支 (A): 在 packed last-dim 上反归一化, 再 unpack
            mean = pipe.vae.bn.running_mean.view(1, 1, -1).to(packed.device, packed.dtype)
            std = torch.sqrt(pipe.vae.bn.running_var.view(1, 1, -1).to(packed.device, packed.dtype) + eps)
            packed = packed * std + mean
            lat_4d = _unpack_latents_fallback(packed, token_h, token_w)
        elif num_features == vae_latent_channels:
            # 分支 (B): 先 unpack 到 (B, 32, H, W), 再在 dim=1 上反归一化
            lat_4d = _unpack_latents_fallback(packed, token_h, token_w)
            mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat_4d.device, lat_4d.dtype)
            std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1).to(lat_4d.device, lat_4d.dtype) + eps)
            lat_4d = lat_4d * std + mean
        else:
            print(f"[VAE-denorm][WARN] bn.num_features={num_features} 不匹配 "
                  f"packed.last_dim={packed.shape[-1]} 也不匹配 vae_latent_channels={vae_latent_channels}, "
                  "跳过反归一化 (可能导致解码图偏色/偏亮)")
    if lat_4d is None:
        # 没有 bn 或者 num_features 不匹配 → 直接 unpack 不做反归一化
        lat_4d = _unpack_latents_fallback(packed, token_h, token_w)

    image = pipe.vae.decode(lat_4d, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)

    return {
        "generated": tensor01(image),
        "ref_rendered": tensor01(ref_rendered),
        "ref_normal": tensor01(ref_normal),
        "ref_albedo": tensor01(ref_albedo),
        "tgt_rendered": tensor01(tgt_rendered),
        "tgt_normal": tensor01(tgt_normal),
        "tgt_albedo": tensor01(tgt_albedo),
        "ref_control_9ch": ref_control.detach().cpu(),
        "tgt_control_9ch": tgt_control.detach().cpu(),
        "control_tokens": control_tokens_tgt.detach().cpu(),
        "rcg_enabled": rcg_enabled,
        "rcg_lambda": rcg_lambda,
        "psi_pred": psi_pred.detach().cpu(),
        "jaw_pred": jaw_pred.detach().cpu(),
    }


def save_outputs(result: dict, ref_path: str, output_dir: str, prompt: str):
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(ref_path).stem

    save_image(result["generated"], os.path.join(output_dir, f"{stem}_stage2_generated.png"))

    save_image(result["ref_rendered"], os.path.join(output_dir, f"{stem}_ref_rendered.png"))
    save_image(result["ref_normal"], os.path.join(output_dir, f"{stem}_ref_normal.png"))
    save_image(result["ref_albedo"], os.path.join(output_dir, f"{stem}_ref_albedo.png"))

    save_image(result["tgt_rendered"], os.path.join(output_dir, f"{stem}_tgt_rendered.png"))
    save_image(result["tgt_normal"], os.path.join(output_dir, f"{stem}_tgt_normal.png"))
    save_image(result["tgt_albedo"], os.path.join(output_dir, f"{stem}_tgt_albedo.png"))

    torch.save({"control": result["ref_control_9ch"]}, os.path.join(output_dir, f"{stem}_ref_control_9ch.pt"))
    torch.save({"control": result["tgt_control_9ch"]}, os.path.join(output_dir, f"{stem}_tgt_control_9ch.pt"))
    torch.save({"control_tokens": result["control_tokens"]}, os.path.join(output_dir, f"{stem}_control_tokens.pt"))
    torch.save(
        {"psi_pred": result["psi_pred"], "jaw_pred": result["jaw_pred"], "prompt": prompt},
        os.path.join(output_dir, f"{stem}_stage2_prediction.pt"),
    )

    summary = {
        "prompt": prompt,
        "psi_pred_shape": list(result["psi_pred"].shape),
        "jaw_pred_shape": list(result["jaw_pred"].shape),
        "jaw_pred": result["jaw_pred"].view(-1).tolist(),
    }
    with open(os.path.join(output_dir, f"{stem}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str((PROJECT_ROOT / "configs" / "infer_stage2.yaml").resolve()),
        help="推理 YAML, 默认 configs/infer_stage2.yaml",
    )
    parser.add_argument(
        "--opts",
        nargs="*",
        default=None,
        help="dot-path 覆盖 YAML, 例如 sampling.num_inference_steps=28 sampling.rcg.lambda=3.5",
    )
    # ---- per-call 必填 ----
    parser.add_argument("--ref", type=str, required=True, help="参考人脸图像路径")
    parser.add_argument("--prompt", type=str, required=True, help="文本指令")
    # ---- 这些都可选: 不给就从 YAML 读 ----
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--stage2_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--deca_test_size", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    # ---- RCG 快捷覆盖 (等价于 --opts sampling.rcg.xxx=yyy) ----
    parser.add_argument("--rcg_lambda", type=float, default=None,
                        help="Reference Control Guidance 系数λ; 覆盖 yaml sampling.rcg.lambda")
    parser.add_argument("--rcg_enabled", type=lambda s: str(s).lower() in {"1", "true", "yes"},
                        default=None, help="是否启用 RCG; 覆盖 yaml sampling.rcg.enabled")
    return parser.parse_args()


def _fill_from_cfg(args, cfg):
    """CLI 未给的字段, 从 YAML 里兜底。"""
    sampling = cfg.get("sampling", {})
    output = cfg.get("output", {})
    deca = cfg.get("deca", {})
    stage1 = cfg.get("stage1", {})
    stage2 = cfg.get("stage2", {})

    if args.stage1_ckpt is None:
        args.stage1_ckpt = stage1.get("ckpt_path")
    if args.stage2_ckpt is None:
        args.stage2_ckpt = stage2.get("ckpt_path")
    if args.output_dir is None:
        args.output_dir = output.get("dir", "./output_stage2")
    if args.deca_test_size is None:
        args.deca_test_size = int(deca.get("test_size", 256))
    if args.height is None:
        args.height = int(sampling.get("height", 1024))
    if args.width is None:
        args.width = int(sampling.get("width", 1024))
    if args.num_inference_steps is None:
        args.num_inference_steps = int(sampling.get("num_inference_steps", 28))
    if args.seed is None:
        args.seed = int(sampling.get("seed", 42))

    # ---- RCG CLI 覆盖 ----
    rcg_cfg = sampling.setdefault("rcg", {})
    if args.rcg_lambda is not None:
        rcg_cfg["lambda"] = float(args.rcg_lambda)
    if args.rcg_enabled is not None:
        rcg_cfg["enabled"] = bool(args.rcg_enabled)
    return args


def main():
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    cfg = apply_opts(cfg, args.opts)
    args = _fill_from_cfg(args, cfg)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    stage1_ckpt = resolve_path(args.stage1_ckpt) if args.stage1_ckpt else None
    stage2_ckpt = resolve_path(args.stage2_ckpt) if args.stage2_ckpt else None
    ref_path = resolve_path(args.ref)
    output_dir = resolve_path(args.output_dir)

    if not stage1_ckpt or not os.path.isfile(stage1_ckpt):
        raise FileNotFoundError(f"stage1 checkpoint not found: {stage1_ckpt}")
    if not stage2_ckpt or not os.path.isfile(stage2_ckpt):
        raise FileNotFoundError(f"stage2 checkpoint not found: {stage2_ckpt}")
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"ref image not found: {ref_path}")

    args.stage1_ckpt = stage1_ckpt
    args.stage2_ckpt = stage2_ckpt
    args.ref = ref_path

    print(f"[Init] config       = {args.config}")
    print(f"[Init] stage1_ckpt  = {stage1_ckpt}")
    print(f"[Init] stage2_ckpt  = {stage2_ckpt}")
    print(f"[Init] ref          = {ref_path}")
    print(f"[Init] prompt       = {args.prompt}")
    print(f"[Init] H x W        = {args.height} x {args.width}")
    print(f"[Init] steps        = {args.num_inference_steps}, seed = {args.seed}")
    _rcg = cfg.get("sampling", {}).get("rcg", {}) or {}
    print(f"[Init] RCG          = enabled={bool(_rcg.get('enabled', False))}, lambda={_rcg.get('lambda', 1.0)}")

    result = run_infer(args, cfg)
    save_outputs(result, ref_path=ref_path, output_dir=output_dir, prompt=args.prompt)

    print(f"\n[Done] outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()