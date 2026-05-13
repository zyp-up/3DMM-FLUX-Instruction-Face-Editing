# -*- coding: utf-8 -*-
import argparse
import math
import os
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset
from tqdm.auto import tqdm


def ensure_numpy_legacy_aliases() -> None:
    legacy_aliases = {
        'bool': bool,
        'int': int,
        'float': float,
        'complex': complex,
        'object': object,
        'unicode': str,
        'str': str,
    }
    np_dict = vars(np)
    for name, value in legacy_aliases.items():
        if name not in np_dict:
            setattr(np, name, value)

try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML required: pip install pyyaml") from e

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if Path.cwd().resolve() != _PROJECT_ROOT:
    os.chdir(_PROJECT_ROOT)

from diffusers import DiffusionPipeline
ensure_numpy_legacy_aliases()
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg

from src.datasets.stage2_dataset import Stage2Dataset, build_stage2_collate_fn
from src.losses.stage2_loss import Stage2Loss, Stage2LossWeights
from src.models.conditional_deca_encoder import ConditionalDECAEncoder, load_stage1_checkpoint_state_dict
from src.models.flux2_control_mixer import Flux2ControlMixer


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def set_seed(seed: int, rank: int = 0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _set_nested(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split('.')
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = yaml.safe_load(value) if isinstance(value, str) else value


def load_config(path: str, cli_opts: Optional[List[str]] = None) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    for item in cli_opts or []:
        k, v = item.split('=', 1)
        _set_nested(cfg, k.strip(), v.strip())
    return cfg


def ddp_setup() -> Tuple[int, int, int, torch.device]:
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device('cuda', local_rank)


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return t
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def split_train_val(dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    n_val = int(round(n * val_ratio))
    return Subset(dataset, idx[n_val:]), Subset(dataset, idx[:n_val])


class BucketBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int, shuffle: bool, drop_last: bool, rank: int = 0, world_size: int = 1, seed: int = 42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.buckets = self._collect_buckets()

    def _base_sample(self, idx: int) -> Dict[str, Any]:
        if isinstance(self.dataset, Subset):
            base_idx = self.dataset.indices[idx]
            return self.dataset.dataset.samples[base_idx]
        return self.dataset.samples[idx]

    def _collect_buckets(self) -> Dict[str, List[int]]:
        buckets: Dict[str, List[int]] = {}
        for i in range(len(self.dataset)):
            key = self._base_sample(i).get("bucket", "unknown")
            buckets.setdefault(key, []).append(i)
        return buckets

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> List[List[int]]:
        # 各 rank 在每个 bucket 内必须拿到相同数量的样本, 否则 batch 数不齐,
        # DDP 训练/验证里的 all_reduce 会出现 SeqNum 错位 → NCCL watchdog 超时。
        # 做法: 每个 bucket 先 shuffle, 再 trim 到 world_size 的整数倍 (丢弃尾部不足部分),
        # 再按 rank 切片, 这样所有 rank 的 idxs 长度严格相同。
        # 注意: 这里 "丢弃" 只发生在 bucket 尾部不能被 world_size 整除的小段, 量级可忽。
        rng = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []
        for idxs in self.buckets.values():
            idxs = list(idxs)
            if self.shuffle:
                rng.shuffle(idxs)
            usable = (len(idxs) // self.world_size) * self.world_size
            idxs = idxs[:usable]
            idxs = idxs[self.rank::self.world_size]
            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start:start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


def resolve_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((_PROJECT_ROOT / p).resolve())


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def enable_gradient_checkpointing(module: torch.nn.Module) -> bool:
    if hasattr(module, "enable_gradient_checkpointing"):
        module.enable_gradient_checkpointing()
        module.train()
        return True
    if hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable()
        module.train()
        return True

    enabled = False
    for submodule in module.modules():
        if hasattr(submodule, "gradient_checkpointing"):
            submodule.gradient_checkpointing = True
            enabled = True
    if enabled:
        module.train()
    return enabled


def build_stage1_from_ckpt(stage1_ckpt_path: str) -> ConditionalDECAEncoder:
    ckpt = torch.load(stage1_ckpt_path, map_location="cpu")
    if "cfg" not in ckpt or "model" not in ckpt["cfg"]:
        raise ValueError("Stage1 checkpoint missing cfg.model")
    cfg_model = dict(ckpt["cfg"]["model"])

    text_dtype = DTYPE_MAP[cfg_model.get("text_encoder_dtype", "bf16")]
    cfg_model["text_encoder_dtype"] = text_dtype

    deca_model_tar = cfg_model.get("deca_model_tar")
    if deca_model_tar is not None:
        deca_model_tar = resolve_path(deca_model_tar)
    fallback_deca_model_tar = resolve_path("./data/deca_model.tar")
    if deca_model_tar is None or (not os.path.isfile(deca_model_tar)):
        deca_model_tar = fallback_deca_model_tar
    cfg_model["deca_model_tar"] = deca_model_tar

    cfg_model["text_encoder_path"] = resolve_path(cfg_model["text_encoder_path"])
    cfg_model["load_text_encoder"] = False

    model = ConditionalDECAEncoder(**cfg_model)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    allowed_missing = ckpt.get("excluded_state_dict_prefixes", ["text_encoder."]) if isinstance(ckpt, dict) else ["text_encoder."]
    load_stage1_checkpoint_state_dict(model, state_dict, allowed_missing)
    if int(os.environ.get("RANK", 0)) == 0:
        print("[Stage1] built without internal Qwen; reusing pipe.text_encoder for text features.")
    return model


def load_flux_pipeline(model_path: str, torch_dtype: torch.dtype, device: str):
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    pipe.to(device)
    freeze_module(pipe.text_encoder)
    freeze_module(pipe.transformer)
    freeze_module(pipe.vae)
    return pipe


@torch.no_grad()
def encode_flux_prompts(pipe, prompts, max_length: int, device: str, dtype: torch.dtype):
    """Encode prompts with FLUX.2 Klein official text path.

    prompt_embeds_flux is concat(Qwen3 hidden layers 9/18/27), shape (B, L, 7680).
    Stage1 consumes the same tensor, so Stage1/Stage2 text conditioning is aligned.
    """
    prompt_embeds_flux, _ = pipe.encode_prompt(
        prompt=list(prompts),
        device=device,
        max_sequence_length=max_length,
    )
    prompt_embeds_flux = prompt_embeds_flux.to(dtype)
    txt_ids = torch.zeros(
        prompt_embeds_flux.shape[0],
        prompt_embeds_flux.shape[1],
        4,
        dtype=dtype,
        device=device,
    )
    txt_ids[..., 3] = torch.arange(prompt_embeds_flux.shape[1], device=device, dtype=dtype)

    texts = [pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ) for p in prompts]
    tokenized = pipe.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    text_attention_mask = tokenized["attention_mask"].to(device)
    return prompt_embeds_flux, prompt_embeds_flux, text_attention_mask, txt_ids


def _pack_latents_fallback(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"latents H/W must be even, got {(h, w)}")
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
    latents = latents.view(b, (h // 2) * (w // 2), c * 4)
    return latents


def _prepare_img_ids_fallback(batch_size: int, token_h: int, token_w: int, device, dtype, t_coord: float = 0.0):
    # 对齐官方 flux2.sampling.prc_img: x_ids = cartesian_prod(t, h, w, l)。
    ys = torch.arange(token_h, device=device, dtype=dtype)
    xs = torch.arange(token_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    ids = torch.stack(
        [
            torch.full_like(yy, float(t_coord)),
            yy,
            xx,
            torch.zeros_like(yy),
        ],
        dim=-1,
    ).view(1, token_h * token_w, 4)
    return ids.repeat(batch_size, 1, 1)


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
    img_ids = _prepare_img_ids_fallback(
        batch_size=image.shape[0],
        token_h=token_h,
        token_w=token_w,
        device=image.device,
        dtype=packed.dtype,
        t_coord=t_coord,
    )
    return packed, img_ids, (token_h, token_w)


def code_to_device(code: Dict[str, torch.Tensor], device: str):
    out = {}
    for k, v in code.items():
        out[k] = v.to(device, non_blocking=True)
    return out


def clone_code_dict(code: Dict[str, torch.Tensor]):
    out = {}
    for k, v in code.items():
        out[k] = v.clone()
    return out


def compose_pred_code(code_ref: Dict[str, torch.Tensor], psi_pred: torch.Tensor, jaw_pred: torch.Tensor):
    code_tgt = clone_code_dict(code_ref)
    code_tgt["exp"] = psi_pred.to(code_ref["exp"].dtype)
    pose = code_ref["pose"].clone()
    pose[:, 3:6] = jaw_pred.to(code_ref["pose"].dtype)
    code_tgt["pose"] = pose
    return code_tgt


def decode_control_9ch(
    deca: DECA,
    code: Dict[str, torch.Tensor],
    ref_image: torch.Tensor,
    use_alpha_mask: bool = True,
) -> torch.Tensor:
    # deca.decode 要求 codedict 里带 'images' (内部要 batch_size = images.shape[0],
    # 以及 light/render 时用于尺寸/贴图参考)。
    # 我们的 code 来自离线 .pt, 不包含 images, 这里把 dataset 的
    # ref_image (B,3,224,224) 补上。不修改原 code dict (避免污染下游 compose)。
    code_fp32 = {
        k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in code.items()
    }
    code_fp32["images"] = ref_image.float()
    with torch.cuda.amp.autocast(enabled=False):
        opdict, _ = deca.decode(code_fp32)
        rendered = opdict["rendered_images"].float()
        normal = opdict["normal_images"].float()
        albedo = opdict["albedo_images"].float()
        if use_alpha_mask and "alpha_images" in opdict:
            # alpha_images: (B,1,H,W) in [0,1]; 1=face mesh foreground, 0=background
            alpha = opdict["alpha_images"].float()
            rendered = rendered * alpha
            normal = normal * alpha
            albedo = albedo * alpha
        return torch.cat([rendered, normal, albedo], dim=1)


def build_loss(cfg_loss: Dict[str, Any]) -> Stage2Loss:
    return Stage2Loss(Stage2LossWeights(**cfg_loss))


def forward_stage2_batch(batch, stage1_model, control_mixer, deca, pipe, loss_fn, cfg, device, flux_dtype):
    ref_image = batch['ref_image'].to(device, non_blocking=True)
    ref_native_image = batch['ref_native_image'].to(device, non_blocking=True)
    tgt_image = batch['tgt_image'].to(device, non_blocking=True)
    psi_tgt = batch['psi_tgt'].to(device, non_blocking=True)
    jaw_tgt = batch['jaw_tgt'].to(device, non_blocking=True)
    prompts = batch['prompts']
    ref_code = code_to_device(batch['ref_params'], device)

    # ① 统一的文本编码: 一次 Qwen3 forward, 两路输出
    #    - prompt_embeds(7680): FLUX 原版路径, 送入 transformer
    #    - text_hidden_for_stage1(7680): 同一份 FLUX prompt_embeds, 送入 stage1 encoder
    prompt_embeds, text_hidden_for_stage1, text_attn_mask, txt_ids = encode_flux_prompts(
        pipe=pipe,
        prompts=prompts,
        max_length=cfg['data']['max_text_length'],
        device=device,
        dtype=flux_dtype,
    )

    # ② Stage1 Conditional Encoder: 直接吿外部 text_last_hidden, 不再走自己的 Qwen3
    psi_pred, jaw_pred = unwrap_model(stage1_model).predict_from_text_hidden(
        ref_image=ref_image,
        text_hidden_states=text_hidden_for_stage1,
        attention_mask=text_attn_mask,
    )

    # ③ Flow Matching 输入使用 target image 原生 bucket 分辨率；ref image 作为 visual condition 不加噪
    tgt_image_norm = tgt_image * 2.0 - 1.0
    ref_native_norm = ref_native_image * 2.0 - 1.0
    clean_latents, img_ids, token_hw = encode_image_to_flux_tokens(pipe, tgt_image_norm, sample=False, t_coord=0.0)
    ref_latents, ref_img_ids, ref_token_hw = encode_image_to_flux_tokens(pipe, ref_native_norm, sample=False, t_coord=10.0)
    if ref_token_hw != token_hw:
        raise ValueError(f"ref latent token grid {ref_token_hw} != target token grid {token_hw}")
    t = torch.rand(clean_latents.shape[0], device=device, dtype=clean_latents.dtype)
    noise = torch.randn_like(clean_latents)
    noisy_latents = (1.0 - t.view(-1, 1, 1)) * clean_latents + t.view(-1, 1, 1) * noise
    flow_target = noise - clean_latents

    # ④ DECA 渲染 ref/tgt 控制图
    # ref_control 恒定不需要梯度；tgt_control 默认也切断 DECA 渲染图，避免
    # FLUX flow loss 通过 renderer 反传到 Stage1 造成巨量显存占用。
    # Stage1 仍由 aux loss 监督；如需端到端 flow->Stage1，可把
    # stage2.detach_deca_control 设为 false。
    detach_deca_control = bool(cfg.get('stage2', {}).get('detach_deca_control', True))
    use_alpha_mask = bool(cfg.get('data', {}).get('use_alpha_mask', True))
    with torch.no_grad():
        ref_control = decode_control_9ch(deca, ref_code, ref_image, use_alpha_mask=use_alpha_mask)
    if detach_deca_control:
        with torch.no_grad():
            pred_code = compose_pred_code(ref_code, psi_pred.detach(), jaw_pred.detach())
            tgt_control = decode_control_9ch(deca, pred_code, ref_image, use_alpha_mask=use_alpha_mask)
    else:
        pred_code = compose_pred_code(ref_code, psi_pred, jaw_pred)
        tgt_control = decode_control_9ch(deca, pred_code, ref_image, use_alpha_mask=use_alpha_mask)
    native_hw = tgt_image.shape[-2:]
    ref_control = F.interpolate(ref_control, size=native_hw, mode='bilinear', align_corners=False)
    tgt_control = F.interpolate(tgt_control, size=native_hw, mode='bilinear', align_corners=False)

    # ⑤ CMM 投射成 control tokens, 拼到 hidden_states/noise token 侧
    control_tokens = control_mixer(tgt_control.to(flux_dtype), ref_control.to(flux_dtype), token_hw=token_hw)
    control_ids = _prepare_img_ids_fallback(
        control_tokens.shape[0], token_hw[0], token_hw[1], device, img_ids.dtype, t_coord=20.0
    )
    model_input = torch.cat([noisy_latents, ref_latents, control_tokens], dim=1)
    model_img_ids = torch.cat([img_ids, ref_img_ids, control_ids], dim=1)
    # FLUX.2-klein-base 是非蒸馏模型 (transformer.config.guidance_embeds=False), 必须传 guidance=None;
    # 传 distilled guidance 张量会破坏 timestep embedding, 训出来的模型推理就是乱码 (诊断 #6 验证)
    flow_pred_full = pipe.transformer(hidden_states=model_input, encoder_hidden_states=prompt_embeds, timestep=t, img_ids=model_img_ids, txt_ids=txt_ids, guidance=None, return_dict=True).sample
    flow_pred = flow_pred_full[:, :clean_latents.shape[1], :]
    return loss_fn(flow_pred=flow_pred, flow_tgt=flow_target, psi_pred=psi_pred, jaw_pred=jaw_pred, psi_tgt=psi_tgt, jaw_tgt=jaw_tgt)


@torch.no_grad()
def run_validation(stage1_model, control_mixer, deca, pipe, loss_fn, val_loader, cfg, device, flux_dtype, amp_dtype, use_amp):
    stage1_model.eval(); control_mixer.eval()
    agg = {'flow': 0.0, 'aux_total': 0.0, 'psi_mse': 0.0, 'jaw_mse': 0.0, 'reg_psi': 0.0, 'reg_jaw_roll': 0.0, 'reg_jaw_close': 0.0, 'total': 0.0}
    n_batches = 0

    # — 跨 rank 对齐验证集 batch 数 —
    # BucketBatchSampler 按 bucket 按 rank 切片, 各 rank 的 len(val_loader) 可能不一样。
    # 如果不对齐, 后面的 reduce_mean(...) 会在不同 rank 上被调用不同次数,
    # 导致 NCCL ALLREDUCE SeqNum 不一致 → watchdog 600s 超时。
    # 这里先全局取 min, 只跑最小 batch 数, 多余样本丢弃 (对验证指标影响可忽)。
    local_n = len(val_loader)
    if dist.is_initialized():
        n_t = torch.tensor([local_n], device=device, dtype=torch.long)
        dist.all_reduce(n_t, op=dist.ReduceOp.MIN)
        global_min_n = int(n_t.item())
    else:
        global_min_n = local_n

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= global_min_n:
            break
        autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else torch.cuda.amp.autocast(enabled=False)
        with autocast_ctx:
            _, d = forward_stage2_batch(batch, stage1_model, control_mixer, deca, pipe, loss_fn, cfg, device, flux_dtype)
        for k in agg:
            agg[k] += float(reduce_mean(d[k]).item())
        n_batches += 1
    for k in agg:
        agg[k] /= max(n_batches, 1)
    stage1_model.train(); control_mixer.train()
    return agg


def save_ckpt(path: str, stage1_model, control_mixer, cfg: Dict[str, Any], step: int, epoch: int, val_metrics=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'stage1_model': unwrap_model(stage1_model).state_dict(),
            'control_mixer': unwrap_model(control_mixer).state_dict(),
            'step': step,
            'epoch': epoch,
            'val': val_metrics,
            'cfg': cfg,
        },
        path,
    )
    return path


def save_best_ckpt(out_dir: str, stage1_model, control_mixer, cfg: Dict[str, Any], step: int, epoch: int, val_metrics: Dict[str, float]):
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(out_dir):
        if fn.startswith('best-step-') and fn.endswith('.pt'):
            try:
                os.remove(os.path.join(out_dir, fn))
            except OSError:
                pass
    return save_ckpt(os.path.join(out_dir, f'best-step-{step}.pt'), stage1_model, control_mixer, cfg, step, epoch, val_metrics)


def _rotate_step_ckpts(out_dir: str, limit: Optional[int]):
    if not limit or limit <= 0 or not os.path.isdir(out_dir):
        return
    files = []
    for fn in os.listdir(out_dir):
        if fn.startswith('step-') and fn.endswith('.pt'):
            try:
                s = int(fn[len('step-'):-3])
            except ValueError:
                continue
            files.append((s, os.path.join(out_dir, fn)))
    if len(files) <= limit:
        return
    files.sort(key=lambda x: x[0])
    for _, path in files[: len(files) - limit]:
        try:
            os.remove(path)
        except OSError:
            pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--opts', nargs='*', default=None,
                        help='override yaml via dot-path, e.g. train.lr_stage1=1e-5 train.batch_size=1')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(resolve_path(args.config), args.opts)
    rank, local_rank, world_size, device = ddp_setup()
    set_seed(cfg['seed'], rank)
    flux_dtype = DTYPE_MAP[cfg["flux"].get("torch_dtype", "bf16")]
    amp_dtype = DTYPE_MAP[cfg["train"].get("amp_dtype", "bf16")]
    use_amp = bool(cfg["train"].get("use_amp", True))

    run_name = cfg['logging'].get('wandb_run_name') or f"stage2-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join(resolve_path(cfg['ckpt']['out_dir']), run_name)
    os.makedirs(out_dir, exist_ok=True)
    cfg['ckpt']['out_dir'] = out_dir
    if is_main(rank):
        print(f"[ckpt] run_name={run_name}")
        print(f"[ckpt] out_dir={out_dir}")

    wandb = None
    if is_main(rank) and cfg['logging'].get('wandb_mode', 'online') != 'disabled':
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            api_key = cfg['logging'].get('wandb_api_key') or os.environ.get('WANDB_API_KEY')
            if api_key:
                try:
                    wandb.login(key=api_key, relogin=False, verify=False)
                except Exception as e:
                    print(f"[WARN] wandb.login failed: {e}")
            safe_cfg = {k: v for k, v in cfg.items() if k != 'logging'}
            safe_logging = {k: v for k, v in cfg['logging'].items() if k != 'wandb_api_key'}
            safe_cfg['logging'] = safe_logging
            wandb.init(
                project=cfg['logging']['wandb_project'],
                name=run_name,
                entity=cfg['logging'].get('wandb_entity'),
                mode=cfg['logging'].get('wandb_mode', 'online'),
                config=safe_cfg,
            )
        except ImportError:
            print('[WARN] wandb not installed; skipping wandb logging.')
            wandb = None

    dataset = Stage2Dataset(
        sources=cfg['data']['sources'],
        deca_image_size=cfg['data']['deca_image_size'],
        crop_scale=cfg['data']['crop_scale'],
        prompt_key_en=cfg['data']['prompt_key_en'],
        prompt_key_cn=cfg['data']['prompt_key_cn'],
        lang_prob_en=cfg['data']['lang_prob_en'],
        use_fan=cfg['data']['use_fan'],
        pre_size=cfg['data']['pre_size'],
        verbose=is_main(rank),
    )
    train_set, val_set = split_train_val(dataset, cfg['train'].get('val_ratio', 0.0), cfg['train'].get('val_split_seed', 42))
    if is_main(rank):
        print(f"[data] train={len(train_set)}, val={len(val_set)}")
    collate = build_stage2_collate_fn()
    train_sampler = BucketBatchSampler(train_set, cfg['train']['batch_size'], shuffle=True, drop_last=True, rank=rank, world_size=world_size, seed=cfg['train'].get('val_split_seed', 42))
    val_sampler = BucketBatchSampler(val_set, cfg['train']['batch_size'], shuffle=False, drop_last=False, rank=rank, world_size=world_size, seed=cfg['train'].get('val_split_seed', 42))
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'],
        persistent_workers=cfg['data']['persistent_workers'],
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=0,           # val 不启 worker: 避免与 train_loader 同时 fork 子进程
        pin_memory=False,        # 同上, pin_memory 会拉起额外的 CUDA context
        persistent_workers=False,
        collate_fn=collate,
    )

    pipe = load_flux_pipeline(
        model_path=resolve_path(cfg["flux"]["model_path"]),
        torch_dtype=flux_dtype,
        device=device,
    )
    if cfg['train'].get('gradient_checkpointing', False):
        enabled = enable_gradient_checkpointing(pipe.transformer)
        if is_main(rank):
            print(f"[memory] transformer gradient_checkpointing={'enabled' if enabled else 'unavailable'}")
    deca = load_deca(device=device, deca_data_dir=resolve_path(cfg["deca"]["data_dir"]))

    stage1_model = build_stage1_from_ckpt(
        resolve_path(cfg['stage1']['ckpt_path']),
    )
    stage1_model = stage1_model.to(device)
    stage1_model.train()
    # 注: pipe.transformer 参数已冻结，但需要对输入侧 control_tokens 求梯度，
    #     因此 transformer backward 仍会保存激活；上面的 gradient checkpointing
    #     能显著降低这部分显存，代价是训练速度变慢。

    control_mixer = Flux2ControlMixer(
        hidden_dim=cfg["model"]["control_mixer_hidden_dim"],
        num_heads=cfg["model"]["control_mixer_heads"],
        joint_dim=cfg["model"]["control_joint_dim"],
        dropout=cfg["model"]["control_dropout"],
    ).to(device)
    control_mixer.train()

    if world_size > 1:
        stage1_model = DDP(stage1_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        control_mixer = DDP(control_mixer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    loss_fn = build_loss(cfg['loss']).to(device)

    trainable_stage1 = [p for p in stage1_model.parameters() if p.requires_grad]
    trainable_cmm = [p for p in control_mixer.parameters() if p.requires_grad]
    trainable = trainable_stage1 + trainable_cmm
    if is_main(rank):
        n_tr = sum(p.numel() for p in trainable)
        n_all = sum(p.numel() for p in list(stage1_model.parameters()) + list(control_mixer.parameters()))
        print(f"[params] trainable={n_tr/1e6:.2f}M / total={n_all/1e6:.2f}M ({100*n_tr/max(n_all,1):.2f}%)")

    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_stage1, "lr": cfg['train']['lr_stage1']},
            {"params": trainable_cmm, "lr": cfg['train']['lr_cmm']},
        ],
        weight_decay=cfg['train']['weight_decay'],
        betas=tuple(cfg['train']['betas']),
        eps=cfg['train']['eps'],
    )

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * cfg['train']['epochs'] if cfg['train'].get('max_steps') is None else int(cfg['train']['max_steps'])
    warmup_steps = int(total_steps * cfg['train'].get('warmup_ratio', 0.0))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    best_val = [float('inf')]
    global_step = 0
    max_steps = cfg['train'].get('max_steps', None)
    best_mode = cfg['ckpt']['best_mode']
    best_key = cfg['ckpt']['best_metric']
    train_start_time = time.time() if 'time' in globals() else __import__('time').time()
    last_eval_step = [-1]
    last_save_step = [-1]

    def _fmt_hms(sec: float) -> str:
        sec = max(int(sec), 0)
        h, r = divmod(sec, 3600)
        m, s = divmod(r, 60)
        if h > 0:
            return f'{h}h {m}m {s}s'
        if m > 0:
            return f'{m}m {s}s'
        return f'{s}s'

    def run_val_and_maybe_save_best(cur_epoch: int, tag: str) -> None:
        if last_eval_step[0] == global_step:
            return
        val_metrics = run_validation(stage1_model, control_mixer, deca, pipe, loss_fn, val_loader, cfg, device, flux_dtype, amp_dtype, use_amp)
        last_eval_step[0] = global_step
        if not is_main(rank):
            return
        log_d = {f'val/{k}': v for k, v in val_metrics.items()}
        if wandb is not None:
            wandb.log(log_d, step=global_step)
        tqdm.write(f"[val][{tag}] epoch={cur_epoch + 1} step={global_step} " + ' '.join(f'{k}={v:.4f}' for k, v in val_metrics.items()))
        cur = val_metrics[best_key.split('/')[-1]]
        is_better = (cur < best_val[0]) if best_mode == 'min' else (cur > best_val[0])
        if cfg['ckpt']['save_best_only'] and is_better:
            best_val[0] = cur
            best_path = save_best_ckpt(cfg['ckpt']['out_dir'], stage1_model, control_mixer, cfg, global_step, cur_epoch, val_metrics)
            if wandb is not None:
                wandb.summary['best_' + best_key] = best_val[0]
            tqdm.write(f"[ckpt] saved new best {best_key}={best_val[0]:.4f} -> {best_path}")

    def maybe_save_step_ckpt(cur_epoch: int, force: bool = False) -> None:
        save_every_steps = cfg['ckpt'].get('save_every_steps') or 0
        if not force and (save_every_steps <= 0 or global_step % save_every_steps != 0):
            return
        if not is_main(rank) or last_save_step[0] == global_step:
            return
        path = save_ckpt(os.path.join(cfg['ckpt']['out_dir'], f'step-{global_step}.pt'), stage1_model, control_mixer, cfg, global_step, cur_epoch)
        last_save_step[0] = global_step
        _rotate_step_ckpts(cfg['ckpt']['out_dir'], cfg['ckpt'].get('save_total_limit'))
        if cfg['ckpt'].get('save_last', False):
            save_ckpt(os.path.join(cfg['ckpt']['out_dir'], 'last.pt'), stage1_model, control_mixer, cfg, global_step, cur_epoch)
        tqdm.write(f'[ckpt] saved {path}')

    pbar = tqdm(total=total_steps, desc='Train', dynamic_ncols=True, disable=not is_main(rank))

    for epoch in range(cfg['train']['epochs']):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        stage1_model.train(); control_mixer.train()
        epoch_lang_counts = {'en': 0, 'cn': 0}
        t0 = __import__('time').time()
        for batch in train_loader:
            for lg in batch['langs']:
                epoch_lang_counts[lg] = epoch_lang_counts.get(lg, 0) + 1
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else torch.cuda.amp.autocast(enabled=False)
            with autocast_ctx:
                total_loss, loss_dict = forward_stage2_batch(batch, stage1_model, control_mixer, deca, pipe, loss_fn, cfg, device, flux_dtype)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg['train']['grad_clip'])
            optimizer.step()
            scheduler.step()

            if is_main(rank):
                pbar.update(1)

            if is_main(rank) and (global_step % cfg['train']['log_every'] == 0):
                lr_now = optimizer.param_groups[0]['lr']
                log_d = {f'train/{k}': float(v.item()) for k, v in loss_dict.items()}
                log_d.update({'train/lr': lr_now, 'train/grad_norm': float(grad_norm), 'train/epoch': epoch, 'train/step': global_step})
                if wandb is not None:
                    wandb.log(log_d, step=global_step)
                done_steps = global_step + 1
                elapsed = __import__('time').time() - train_start_time
                speed = done_steps / max(elapsed, 1e-6)
                remaining = (total_steps - done_steps) / max(speed, 1e-6)
                epoch_frac = done_steps / max(steps_per_epoch, 1)
                mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                stat = {'loss': round(log_d['train/total'], 8), 'grad_norm': round(float(grad_norm), 4), 'learning_rate': lr_now, 'flow': round(log_d['train/flow'], 8), 'aux_total': round(log_d['train/aux_total'], 8), 'psi_mse': round(log_d['train/psi_mse'], 8), 'jaw_mse': round(log_d['train/jaw_mse'], 8), 'epoch': round(epoch_frac, 2), 'global_step/max_steps': f'{done_steps}/{total_steps}', 'percentage': f'{done_steps / max(total_steps, 1) * 100:.2f}%', 'elapsed_time': _fmt_hms(elapsed), 'remaining_time': _fmt_hms(remaining), 'memory(GiB)': round(mem_gib, 2), 'train_speed(iter/s)': round(speed, 6)}
                tqdm.write(str(stat))

            global_step += 1
            maybe_save_step_ckpt(epoch)
            val_every_steps = cfg['train'].get('val_every_steps') or 0
            if val_every_steps > 0 and global_step % val_every_steps == 0:
                run_val_and_maybe_save_best(epoch, tag='step')
            if max_steps is not None and global_step >= int(max_steps):
                break

        dt = __import__('time').time() - t0
        if is_main(rank):
            tot = sum(epoch_lang_counts.values())
            if wandb is not None:
                wandb.log({'epoch/time_s': dt, 'epoch/lang_en': epoch_lang_counts.get('en', 0), 'epoch/lang_cn': epoch_lang_counts.get('cn', 0), 'epoch/lang_ratio_en': epoch_lang_counts.get('en', 0) / max(tot, 1)}, step=global_step)
            tqdm.write(f"[epoch {epoch} done] time={dt:.1f}s lang en/cn={epoch_lang_counts['en']}/{epoch_lang_counts['cn']}")
            save_every_epoch = cfg['ckpt'].get('save_every_epoch') or 0
            if save_every_epoch > 0 and (epoch + 1) % save_every_epoch == 0:
                save_ckpt(os.path.join(cfg['ckpt']['out_dir'], f'epoch-{epoch + 1}.pt'), stage1_model, control_mixer, cfg, global_step, epoch)
                if cfg['ckpt'].get('save_last', False):
                    save_ckpt(os.path.join(cfg['ckpt']['out_dir'], 'last.pt'), stage1_model, control_mixer, cfg, global_step, epoch)
                tqdm.write(f"[ckpt] saved epoch-{epoch + 1}.pt")

        val_every_epoch = cfg['train'].get('val_every_epoch') or 0
        if val_every_epoch > 0 and (epoch + 1) % val_every_epoch == 0:
            run_val_and_maybe_save_best(epoch, tag='epoch')
        if max_steps is not None and global_step >= int(max_steps):
            break

    final_epoch = cfg['train']['epochs'] - 1
    if global_step > 0 and (cfg['ckpt'].get('save_every_steps') or 0) > 0:
        maybe_save_step_ckpt(final_epoch, force=True)
    if global_step > 0 and (((cfg['train'].get('val_every_steps') or 0) > 0) or ((cfg['train'].get('val_every_epoch') or 0) > 0)):
        run_val_and_maybe_save_best(final_epoch, tag='final')
    if is_main(rank) and cfg['ckpt'].get('save_last', False) and last_save_step[0] != global_step:
        save_ckpt(os.path.join(cfg['ckpt']['out_dir'], 'last.pt'), stage1_model, control_mixer, cfg, global_step, final_epoch)
        print('[ckpt] saved final last.pt')
    if is_main(rank):
        print(f"[Done] checkpoints saved to: {out_dir}")
    if is_main(rank) and wandb is not None:
        wandb.finish()
    ddp_cleanup()


if __name__ == "__main__":
    main()