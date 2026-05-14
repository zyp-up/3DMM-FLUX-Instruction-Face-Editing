"""Stage1 inference script.

Given a reference face and a text instruction, it saves reference/target DECA
control maps plus 9-channel tensors for CMM smoke tests.

Example:
python infer/infer_stage1.py \
    --ref ./reference.png \
    --prompt "make her burst into laughter" \
    --output_dir ./output_stage1 \
    --ckpt ./checkpoints/stage1/best-step-2320.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.utils import save_image

# Disable cuDNN for environments where small GPU convolutions trigger
# CUDNN_STATUS_NOT_INITIALIZED; GPU execution still uses non-cuDNN kernels.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# chumpy expects aliases removed in newer numpy versions.
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

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import datasets as deca_dataset
from src.models.conditional_deca_encoder import ConditionalDECAEncoder, load_stage1_checkpoint_state_dict


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def resolve_path(path_str: str) -> str:
    if path_str is None:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def load_yaml(path: str):
    with open(path, "r") as f:
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


def build_stage1_model(cfg_model: dict, ckpt_path: str, device: str):
    model_kwargs = dict(cfg_model)

    model_kwargs["text_encoder_path"] = resolve_path(model_kwargs["text_encoder_path"])
    if model_kwargs.get("tokenizer_path") is not None:
        model_kwargs["tokenizer_path"] = resolve_path(model_kwargs["tokenizer_path"])

    deca_model_tar = resolve_path(model_kwargs.get("deca_model_tar"))
    fallback_deca_model_tar = str((PROJECT_ROOT / "data" / "deca_model.tar").resolve())
    if deca_model_tar is None or (not os.path.isfile(deca_model_tar)):
        if os.path.isfile(fallback_deca_model_tar):
            deca_model_tar = fallback_deca_model_tar
    model_kwargs["deca_model_tar"] = deca_model_tar

    model_kwargs["text_encoder_dtype"] = DTYPE_MAP[model_kwargs.get("text_encoder_dtype", "bf16")]

    model = ConditionalDECAEncoder(**model_kwargs).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    allowed_missing = ckpt.get("excluded_state_dict_prefixes", ["text_encoder."]) if isinstance(ckpt, dict) else ["text_encoder."]
    missing, unexpected = load_stage1_checkpoint_state_dict(model, state_dict, allowed_missing)
    print(f"[Stage1] checkpoint loaded from: {ckpt_path}")
    print(f"[Stage1] load_state_dict missing_external={len(missing)}, unexpected={len(unexpected)}")

    model.eval()
    return model, ckpt


def clone_code_dict(code_dict: dict):
    out = {}
    for k, v in code_dict.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def tensor01(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu().clamp(0.0, 1.0)


@torch.no_grad()
def run_stage1_infer(
    deca,
    stage1_model,
    ref_path: str,
    prompt: str,
    device: str,
    max_text_length: int = 64,
    deca_test_size: int = 256,
    use_alpha_mask: bool = True,
):
    dataset = deca_dataset.TestData([ref_path], iscrop=True, size=deca_test_size)
    sample = dataset[0]

    ref_crop = sample["image"].unsqueeze(0).to(device)                 # (1,3,224,224) for DECA/Stage1
    original_image = sample["original_image"].unsqueeze(0).to(device)  # used by decode(render_orig=True)
    tform = sample["tform"].unsqueeze(0)
    tform = torch.inverse(tform).transpose(1, 2).to(device)

    tokenized = stage1_model.tokenize(
        [prompt],
        max_length=max_text_length,
        device=device,
    )

    code_ref = deca.encode(ref_crop)
    code_ref["tform"] = tform

    psi_pred, jaw_pred = stage1_model(
        ref_image=ref_crop,
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
    )

    psi_pred = psi_pred.to(code_ref["exp"].dtype)
    jaw_pred = jaw_pred.to(code_ref["pose"].dtype)

    code_tgt = clone_code_dict(code_ref)
    code_tgt["exp"] = psi_pred
    code_tgt["pose"] = code_ref["pose"].clone()
    code_tgt["pose"][:, 3:6] = jaw_pred

    opdict_ref, _ = deca.decode(
        code_ref,
        render_orig=True,
        original_image=original_image,
        tform=code_ref["tform"],
    )
    opdict_tgt, _ = deca.decode(
        code_tgt,
        render_orig=True,
        original_image=original_image,
        tform=code_tgt["tform"],
    )

    if use_alpha_mask and "alpha_images" in opdict_ref:
        alpha_ref = opdict_ref["alpha_images"].float()
        ref_rendered = opdict_ref["rendered_images"].float() * alpha_ref
        ref_normal = opdict_ref["normal_images"].float() * alpha_ref
        ref_albedo = opdict_ref["albedo_images"].float() * alpha_ref
    else:
        ref_rendered = opdict_ref["rendered_images"]
        ref_normal = opdict_ref["normal_images"]
        ref_albedo = opdict_ref["albedo_images"]

    if use_alpha_mask and "alpha_images" in opdict_tgt:
        alpha_tgt = opdict_tgt["alpha_images"].float()
        tgt_rendered = opdict_tgt["rendered_images"].float() * alpha_tgt
        tgt_normal = opdict_tgt["normal_images"].float() * alpha_tgt
        tgt_albedo = opdict_tgt["albedo_images"].float() * alpha_tgt
    else:
        tgt_rendered = opdict_tgt["rendered_images"]
        tgt_normal = opdict_tgt["normal_images"]
        tgt_albedo = opdict_tgt["albedo_images"]

    ref_control_9ch = torch.cat([ref_rendered, ref_normal, ref_albedo], dim=1)  # (1,9,H,W)
    tgt_control_9ch = torch.cat([tgt_rendered, tgt_normal, tgt_albedo], dim=1)  # (1,9,H,W)

    return {
        "sample": sample,
        "prompt": prompt,
        "psi_pred": psi_pred.detach().cpu(),
        "jaw_pred": jaw_pred.detach().cpu(),
        "code_ref": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in code_ref.items()},
        "code_tgt": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in code_tgt.items()},
        "ref_rendered": tensor01(ref_rendered),
        "ref_normal": tensor01(ref_normal),
        "ref_albedo": tensor01(ref_albedo),
        "tgt_rendered": tensor01(tgt_rendered),
        "tgt_normal": tensor01(tgt_normal),
        "tgt_albedo": tensor01(tgt_albedo),
        "ref_control_9ch": ref_control_9ch.detach().cpu(),
        "tgt_control_9ch": tgt_control_9ch.detach().cpu(),
        "original_image": tensor01(original_image),
        "ref_crop": tensor01(ref_crop),
    }


def save_outputs(result: dict, ref_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(ref_path).stem

    save_image(result["original_image"], os.path.join(output_dir, f"{stem}_original_image.png"))
    save_image(result["ref_crop"], os.path.join(output_dir, f"{stem}_ref_crop_224.png"))

    save_image(result["ref_rendered"], os.path.join(output_dir, f"{stem}_ref_rendered.png"))
    save_image(result["ref_normal"], os.path.join(output_dir, f"{stem}_ref_normal.png"))
    save_image(result["ref_albedo"], os.path.join(output_dir, f"{stem}_ref_albedo.png"))

    save_image(result["tgt_rendered"], os.path.join(output_dir, f"{stem}_tgt_rendered.png"))
    save_image(result["tgt_normal"], os.path.join(output_dir, f"{stem}_tgt_normal.png"))
    save_image(result["tgt_albedo"], os.path.join(output_dir, f"{stem}_tgt_albedo.png"))

    # Save CMM-ready controls for inspection.
    torch.save(
        {
            "control": result["ref_control_9ch"],
            "channel_order": ["rendered_rgb", "normal_rgb", "albedo_rgb"],
        },
        os.path.join(output_dir, f"{stem}_ref_control_9ch.pt"),
    )
    torch.save(
        {
            "control": result["tgt_control_9ch"],
            "channel_order": ["rendered_rgb", "normal_rgb", "albedo_rgb"],
        },
        os.path.join(output_dir, f"{stem}_tgt_control_9ch.pt"),
    )

    torch.save(
        {
            "psi_pred": result["psi_pred"],
            "jaw_pred": result["jaw_pred"],
            "prompt": result["prompt"],
            "code_ref": result["code_ref"],
            "code_tgt": result["code_tgt"],
        },
        os.path.join(output_dir, f"{stem}_stage1_prediction.pt"),
    )

    summary = {
        "prompt": result["prompt"],
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
        default=str((PROJECT_ROOT / "configs" / "stage1.yaml").resolve()),
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str((PROJECT_ROOT / "checkpoints" / "stage1" / "best-step-2400.pt").resolve()),
    )
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str((PROJECT_ROOT / "output_stage1").resolve()),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--deca_data_dir",
        type=str,
        default=str((PROJECT_ROOT / "DECA" / "data").resolve()),
    )
    parser.add_argument("--deca_test_size", type=int, default=256)
    parser.add_argument(
        "--no_alpha_mask",
        action="store_true",
        help="disable DECA alpha masking and keep original control-map backgrounds",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_yaml(resolve_path(args.config))
    ckpt_path = resolve_path(args.ckpt)
    ref_path = resolve_path(args.ref)
    output_dir = resolve_path(args.output_dir)
    deca_data_dir = resolve_path(args.deca_data_dir)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"ref image not found: {ref_path}")
    if not os.path.isdir(deca_data_dir):
        raise FileNotFoundError(f"DECA data dir not found: {deca_data_dir}")

    device = args.device
    print(f"[Init] device = {device}")
    print(f"[Init] cuDNN enabled = {torch.backends.cudnn.enabled}")
    print(f"[Init] config = {args.config}")
    print(f"[Init] ckpt   = {ckpt_path}")
    print(f"[Init] ref    = {ref_path}")
    print(f"[Init] prompt = {args.prompt}")

    stage1_model, ckpt = build_stage1_model(cfg["model"], ckpt_path, device)
    deca = load_deca(device=device, deca_data_dir=deca_data_dir)

    result = run_stage1_infer(
        deca=deca,
        stage1_model=stage1_model,
        ref_path=ref_path,
        prompt=args.prompt,
        device=device,
        max_text_length=cfg["data"].get("max_text_length", 64),
        deca_test_size=args.deca_test_size,
        use_alpha_mask=(not args.no_alpha_mask),
    )
    save_outputs(result, ref_path=ref_path, output_dir=output_dir)

    print("\n[Done] saved files:")
    print(f"  - {output_dir}")
    print("  - *_ref_rendered.png / *_ref_normal.png / *_ref_albedo.png")
    print("  - *_tgt_rendered.png / *_tgt_normal.png / *_tgt_albedo.png")
    print("  - *_ref_control_9ch.pt / *_tgt_control_9ch.pt")
    print("  - *_stage1_prediction.pt / *_summary.json")

    if isinstance(ckpt, dict):
        print(f"[Checkpoint] step={ckpt.get('step')} epoch={ckpt.get('epoch')} val={ckpt.get('val')}")
    print(f"[Prediction] jaw_pred = {result['jaw_pred'].view(-1).tolist()}")


if __name__ == "__main__":
    main()
