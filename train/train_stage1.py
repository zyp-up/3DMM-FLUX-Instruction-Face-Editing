# -*- coding: utf-8 -*-
"""Stage1 training entrypoint for supervised Conditional DECA Encoder pretraining.

Usage from the project root:
    torchrun --standalone --nproc_per_node=8 train/train_stage1.py \
        --config configs/stage1.yaml

    # Override config fields from CLI:
    torchrun ... train/train_stage1.py --config configs/stage1.yaml \
        --opts train.lr=5e-5 data.batch_size=16 train.epochs=10

The script is YAML-driven, supports torchrun DDP, and keeps a single best
checkpoint by validation metric.
"""

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

# Keep imports and config-relative paths stable when launched from train/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if Path.cwd().resolve() != _PROJECT_ROOT:
    os.chdir(_PROJECT_ROOT)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML required: pip install pyyaml") from e

from src.datasets.stage1_dataset import Stage1Dataset, build_collate_fn
from src.losses.stage1_loss import Stage1Loss, Stage1LossWeights
from src.models.conditional_deca_encoder import ConditionalDECAEncoder, stage1_checkpoint_state_dict


# ===========================================================================
# Config helpers: YAML loading and dot-path CLI overrides.
# ===========================================================================
def _set_nested(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    try:
        value = yaml.safe_load(value) if isinstance(value, str) else value
    except Exception:
        pass
    cur[keys[-1]] = value


def load_config(config_path: str, cli_opts: Optional[List[str]] = None) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if cli_opts:
        for item in cli_opts:
            if "=" not in item:
                raise ValueError(f"--opts expect key=value, got: {item}")
            k, v = item.split("=", 1)
            _set_nested(cfg, k.strip(), v.strip())
    return cfg


# ===========================================================================
# DDP helpers
# ===========================================================================
def ddp_setup() -> Tuple[int, int, int, torch.device]:
    """Initialize DDP from torchrun env and return rank/device metadata."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, local_rank, world_size, device


def is_main(rank: int) -> bool:
    return rank == 0


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across DDP ranks."""
    if not dist.is_initialized():
        return t
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t


# ===========================================================================
# Random seed
# ===========================================================================
def set_seed(seed: int, rank: int = 0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ===========================================================================
# Train/validation split with a fixed seed.
# ===========================================================================
def split_train_val(
    dataset: Stage1Dataset, val_ratio: float, seed: int,
) -> Tuple[Subset, Subset]:
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    n_val = int(round(n * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ===========================================================================
# Scheduler: linear warmup + cosine decay.
# ===========================================================================
def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ===========================================================================
# Model construction
# ===========================================================================
DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def build_model(cfg_model: Dict[str, Any]) -> ConditionalDECAEncoder:
    kwargs = dict(cfg_model)
    kwargs["text_encoder_dtype"] = DTYPE_MAP[kwargs.get("text_encoder_dtype", "bf16")]
    return ConditionalDECAEncoder(**kwargs)


def build_loss(cfg_loss: Dict[str, Any]) -> Stage1Loss:
    w = Stage1LossWeights(**cfg_loss)
    return Stage1Loss(w)


# ===========================================================================
# Validation pass
# ===========================================================================
@torch.no_grad()
def run_validation(
    model: nn.Module,
    loss_fn: Stage1Loss,
    val_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    agg = {"psi_mse": 0.0, "jaw_mse": 0.0, "reg_psi": 0.0, "reg_jaw_roll": 0.0, "total": 0.0}
    n_batches = 0
    for batch in val_loader:
        ref_image = batch["ref_image"].to(device, non_blocking=True)
        psi_tgt = batch["psi_tgt"].to(device, non_blocking=True)
        jaw_tgt = batch["jaw_tgt"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp \
            else torch.cuda.amp.autocast(enabled=False)
        with autocast_ctx:
            psi_pred, jaw_pred = model(
                ref_image=ref_image, input_ids=input_ids, attention_mask=attention_mask,
            )
            psi_pred = psi_pred.float()
            jaw_pred = jaw_pred.float()
            _, d = loss_fn(psi_pred, jaw_pred, psi_tgt, jaw_tgt)
        for k in agg:
            agg[k] += float(reduce_mean(d[k]).item())
        n_batches += 1

    for k in agg:
        agg[k] /= max(n_batches, 1)
    model.train()
    return agg


# ===========================================================================
# Checkpoint IO
# ===========================================================================
def save_ckpt(
    out_dir: str, name: str, model: nn.Module, step: int, epoch: int,
    val_metrics: Optional[Dict[str, float]], cfg: Dict[str, Any],
) -> str:
    """Save a checkpoint and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    to_save = model.module if isinstance(model, DDP) else model
    state_dict, excluded_prefixes = stage1_checkpoint_state_dict(
        to_save,
        exclude_external=cfg.get("model", {}).get("freeze_text_encoder", True),
    )
    path = os.path.join(out_dir, name)
    torch.save({
        "model": state_dict,
        "excluded_state_dict_prefixes": excluded_prefixes,
        "step": step,
        "epoch": epoch,
        "val": val_metrics,
        "cfg": cfg,
    }, path)
    return path


def save_best_ckpt(
    out_dir: str, model: nn.Module, step: int, epoch: int,
    val_metrics: Dict[str, float], cfg: Dict[str, Any],
) -> str:
    """Keep only the current best-step-{step}.pt checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(out_dir):
        if fn.startswith("best-step-") and fn.endswith(".pt"):
            try:
                os.remove(os.path.join(out_dir, fn))
            except OSError:
                pass
    best_name = f"best-step-{step}.pt"
    return save_ckpt(out_dir, best_name, model, step, epoch, val_metrics, cfg)


def _rotate_step_ckpts(out_dir: str, limit: Optional[int]):
    """Keep the newest limit step checkpoints."""
    if not limit or limit <= 0:
        return
    if not os.path.isdir(out_dir):
        return
    files = []
    for fn in os.listdir(out_dir):
        if fn.startswith("step-") and fn.endswith(".pt"):
            try:
                s = int(fn[len("step-"):-len(".pt")])
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


# ===========================================================================
# Main training loop
# ===========================================================================
def train(cfg: Dict[str, Any]):
    rank, local_rank, world_size, device = ddp_setup()
    set_seed(cfg["seed"], rank)

    run_name = cfg["logging"].get("wandb_run_name") or \
               f"stage1-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    cfg["ckpt"]["out_dir"] = os.path.join(cfg["ckpt"]["out_dir"], run_name)
    if is_main(rank):
        print(f"[ckpt] run_name={run_name}")
        print(f"[ckpt] out_dir={cfg['ckpt']['out_dir']}")

    # --- wandb on main process ---
    wandb = None
    if is_main(rank) and cfg["logging"].get("wandb_mode", "online") != "disabled":
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            # Prefer explicit key, then fall back to env/netrc.
            api_key = cfg["logging"].get("wandb_api_key") or os.environ.get("WANDB_API_KEY")
            if api_key:
                masked = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 16 else "***"
                print(f"[wandb] login with api_key={masked}")
                try:
                    wandb.login(key=api_key, relogin=False, verify=False)
                except Exception as e:
                    print(f"[WARN] wandb.login failed: {e}; fallback to env/netrc")
            # Do not persist API keys in wandb config.
            safe_cfg = {k: v for k, v in cfg.items() if k != "logging"}
            safe_logging = {k: v for k, v in cfg["logging"].items() if k != "wandb_api_key"}
            safe_cfg["logging"] = safe_logging
            wandb.init(
                project=cfg["logging"]["wandb_project"],
                name=run_name,
                entity=cfg["logging"].get("wandb_entity"),
                mode=cfg["logging"].get("wandb_mode", "online"),
                config=safe_cfg,
            )
        except ImportError:
            print("[WARN] wandb not installed; skipping wandb logging.")
            wandb = None

    # --- Model / loss ---
    if is_main(rank):
        print("[build] constructing model ...")
    model = build_model(cfg["model"]).to(device)
    loss_fn = build_loss(cfg["loss"]).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=True)   # Frozen Qwen3 has no-grad parameters.

    # --- Data ---
    base_ds = Stage1Dataset(
        sources=cfg["data"]["sources"],
        image_size=cfg["data"]["image_size"],
        crop_scale=cfg["data"]["crop_scale"],
        prompt_key_en=cfg["data"]["prompt_key_en"],
        prompt_key_cn=cfg["data"]["prompt_key_cn"],
        lang_prob_en=cfg["data"]["lang_prob_en"],
        use_fan=cfg["data"]["use_fan"],
        pre_size=cfg["data"].get("pre_size", 256),
        verbose=is_main(rank),
    )
    train_set, val_set = split_train_val(
        base_ds, cfg["train"]["val_ratio"], cfg["train"]["val_split_seed"],
    )
    if is_main(rank):
        print(f"[data] train={len(train_set)}, val={len(val_set)}")

    tokenizer = model.module.tokenizer if isinstance(model, DDP) else model.tokenizer
    collate = build_collate_fn(tokenizer, max_length=cfg["data"]["max_text_length"])

    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=True,
    ) if world_size > 1 else None
    val_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=cfg["data"]["num_workers"],
        collate_fn=collate, pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=cfg["data"]["persistent_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
        sampler=val_sampler, num_workers=cfg["data"]["num_workers"],
        collate_fn=collate, pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=cfg["data"]["persistent_workers"],
    )

    # --- Optimizer / scheduler ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    if is_main(rank):
        n_tr = sum(p.numel() for p in trainable)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[params] trainable={n_tr/1e6:.2f}M / total={n_all/1e6:.2f}M "
              f"({100*n_tr/n_all:.2f}%)")

    optimizer = torch.optim.AdamW(
        trainable, lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"]["betas"]), eps=cfg["train"]["eps"],
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = int(total_steps * cfg["train"]["warmup_ratio"])
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    amp_dtype = DTYPE_MAP[cfg["train"]["amp_dtype"]]
    use_amp = cfg["train"]["use_amp"]

    # --- Main loop ---
    best_val = [float("inf")]
    global_step = 0
    best_mode = cfg["ckpt"]["best_mode"]
    best_key = cfg["ckpt"]["best_metric"]
    train_start_time = time.time()
    last_eval_step = [-1]
    last_save_step = [-1]

    def _fmt_hms(sec: float) -> str:
        sec = max(int(sec), 0)
        h, r = divmod(sec, 3600)
        m, s = divmod(r, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        if m > 0:
            return f"{m}m {s}s"
        return f"{s}s"

    def run_val_and_maybe_save_best(cur_epoch: int, tag: str) -> None:
        """Run validation once per step and save a new best checkpoint if needed."""
        if last_eval_step[0] == global_step:
            return
        val_metrics = run_validation(
            model, loss_fn, val_loader, device, amp_dtype, use_amp,
        )
        last_eval_step[0] = global_step
        if not is_main(rank):
            return
        log_d = {f"val/{k}": v for k, v in val_metrics.items()}
        if wandb is not None:
            wandb.log(log_d, step=global_step)
        tqdm.write(f"[val][{tag}] epoch={cur_epoch + 1} step={global_step} " + " ".join(
            f"{k}={v:.4f}" for k, v in val_metrics.items()))

        cur = val_metrics[best_key.split("/")[-1]]
        is_better = (cur < best_val[0]) if best_mode == "min" else (cur > best_val[0])
        if cfg["ckpt"]["save_best_only"] and is_better:
            best_val[0] = cur
            best_path = save_best_ckpt(
                cfg["ckpt"]["out_dir"], model, global_step, cur_epoch,
                val_metrics, cfg,
            )
            if wandb is not None:
                wandb.summary["best_" + best_key] = best_val[0]
            tqdm.write(f"[ckpt] saved new best {best_key}={best_val[0]:.4f} -> {best_path}")

    def maybe_save_step_ckpt(cur_epoch: int, force: bool = False) -> None:
        """Save periodic step checkpoints; force treats the final step as eligible."""
        save_every_steps = cfg["ckpt"].get("save_every_steps") or 0
        if not force and (save_every_steps <= 0 or global_step % save_every_steps != 0):
            return
        if not is_main(rank) or last_save_step[0] == global_step:
            return
        path = save_ckpt(
            cfg["ckpt"]["out_dir"], f"step-{global_step}.pt",
            model, global_step, cur_epoch, None, cfg,
        )
        last_save_step[0] = global_step
        _rotate_step_ckpts(
            cfg["ckpt"]["out_dir"],
            cfg["ckpt"].get("save_total_limit"),
        )
        if cfg["ckpt"].get("save_last", False):
            save_ckpt(cfg["ckpt"]["out_dir"], "last.pt",
                      model, global_step, cur_epoch, None, cfg)
        tqdm.write(f"[ckpt] saved {path}")

    pbar = tqdm(
        total=total_steps, desc="Train", dynamic_ncols=True,
        disable=not is_main(rank),
    )

    for epoch in range(cfg["train"]["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_lang_counts = {"en": 0, "cn": 0}
        t0 = time.time()

        for step_in_epoch, batch in enumerate(train_loader):
            ref_image = batch["ref_image"].to(device, non_blocking=True)
            psi_tgt = batch["psi_tgt"].to(device, non_blocking=True)
            jaw_tgt = batch["jaw_tgt"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            for lg in batch["langs"]:
                epoch_lang_counts[lg] = epoch_lang_counts.get(lg, 0) + 1

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp \
                else torch.cuda.amp.autocast(enabled=False)
            with autocast_ctx:
                psi_pred, jaw_pred = model(
                    ref_image=ref_image, input_ids=input_ids, attention_mask=attention_mask,
                )
                psi_pred = psi_pred.float()
                jaw_pred = jaw_pred.float()
                total_loss, loss_dict = loss_fn(psi_pred, jaw_pred, psi_tgt, jaw_tgt)

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=cfg["train"]["grad_clip"],
            )
            optimizer.step()
            scheduler.step()

            # --- Logging ---
            if is_main(rank):
                pbar.update(1)

            if is_main(rank) and (global_step % cfg["train"]["log_every"] == 0):
                lr_now = optimizer.param_groups[0]["lr"]
                log_d = {f"train/{k}": float(v.item()) for k, v in loss_dict.items()}
                log_d.update({
                    "train/lr": lr_now,
                    "train/grad_norm": float(grad_norm),
                    "train/epoch": epoch,
                    "train/step": global_step,
                })
                if wandb is not None:
                    wandb.log(log_d, step=global_step)

                done_steps = global_step + 1
                elapsed = time.time() - train_start_time
                speed = done_steps / max(elapsed, 1e-6)
                remaining = (total_steps - done_steps) / max(speed, 1e-6)
                epoch_frac = done_steps / max(steps_per_epoch, 1)
                if torch.cuda.is_available():
                    mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
                else:
                    mem_gib = 0.0

                stat = {
                    "loss": round(log_d["train/total"], 8),
                    "grad_norm": round(float(grad_norm), 4),
                    "learning_rate": lr_now,
                    "psi_mse": round(log_d["train/psi_mse"], 8),
                    "jaw_mse": round(log_d["train/jaw_mse"], 8),
                    "epoch": round(epoch_frac, 2),
                    "global_step/max_steps": f"{done_steps}/{total_steps}",
                    "percentage": f"{done_steps / max(total_steps, 1) * 100:.2f}%",
                    "elapsed_time": _fmt_hms(elapsed),
                    "remaining_time": _fmt_hms(remaining),
                    "memory(GiB)": round(mem_gib, 2),
                    "train_speed(iter/s)": round(speed, 6),
                }
                tqdm.write(str(stat))

            global_step += 1

            # --- Step checkpoints / validation ---
            maybe_save_step_ckpt(epoch)

            val_every_steps = cfg["train"].get("val_every_steps") or 0
            if val_every_steps > 0 and global_step % val_every_steps == 0:
                run_val_and_maybe_save_best(epoch, tag="step")

        # --- Epoch-end logging / validation ---
        dt = time.time() - t0
        if is_main(rank):
            tot = sum(epoch_lang_counts.values())
            if wandb is not None:
                wandb.log({
                    "epoch/time_s": dt,
                    "epoch/lang_en": epoch_lang_counts.get("en", 0),
                    "epoch/lang_cn": epoch_lang_counts.get("cn", 0),
                    "epoch/lang_ratio_en": epoch_lang_counts.get("en", 0) / max(tot, 1),
                }, step=global_step)
            tqdm.write(f"[epoch {epoch} done] time={dt:.1f}s "
                       f"lang en/cn={epoch_lang_counts['en']}/{epoch_lang_counts['cn']}")

            # --- Epoch checkpoints ---
            save_every_epoch = cfg["ckpt"].get("save_every_epoch") or 0
            if save_every_epoch > 0 and (epoch + 1) % save_every_epoch == 0:
                save_ckpt(
                    cfg["ckpt"]["out_dir"], f"epoch-{epoch + 1}.pt",
                    model, global_step, epoch, None, cfg,
                )
                if cfg["ckpt"].get("save_last", False):
                    save_ckpt(cfg["ckpt"]["out_dir"], "last.pt",
                              model, global_step, epoch, None, cfg)
                tqdm.write(f"[ckpt] saved epoch-{epoch + 1}.pt")

        val_every_epoch = cfg["train"].get("val_every_epoch") or 0
        if val_every_epoch > 0 and (epoch + 1) % val_every_epoch == 0:
            run_val_and_maybe_save_best(epoch, tag="epoch")

    # --- Final save/validation guard ---
    final_epoch = cfg["train"]["epochs"] - 1
    if global_step > 0 and (cfg["ckpt"].get("save_every_steps") or 0) > 0:
        maybe_save_step_ckpt(final_epoch, force=True)
    if global_step > 0 and (
        (cfg["train"].get("val_every_steps") or 0) > 0 or
        (cfg["train"].get("val_every_epoch") or 0) > 0
    ):
        run_val_and_maybe_save_best(final_epoch, tag="final")

    if is_main(rank) and cfg["ckpt"].get("save_last", False) and last_save_step[0] != global_step:
        save_ckpt(cfg["ckpt"]["out_dir"], "last.pt",
                  model, global_step, final_epoch, None, cfg)
        print("[ckpt] saved final last.pt")

    if is_main(rank) and wandb is not None:
        wandb.finish()
    ddp_cleanup()


# ===========================================================================
# Entrypoint
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--opts", nargs="*", default=None,
                   help="override yaml via dot-path, e.g. train.lr=5e-5 data.batch_size=16")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    train(cfg)


if __name__ == "__main__":
    main()
