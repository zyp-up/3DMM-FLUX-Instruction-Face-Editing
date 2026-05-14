# -*- coding: utf-8 -*-
"""Stage1 dataset for supervised Conditional DECA Encoder training.

Each sample uses image_a as the reference image, maps image_b to its offline
DECA parameter file, and samples one prompt from the configured language fields.
FAN cropping follows the same protocol as scripts/extract_deca_params.py.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from skimage.io import imread
from skimage.transform import estimate_transform, resize, warp
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# FAN cropper, lazily initialized once per DataLoader worker.
# ---------------------------------------------------------------------------
_WORKER_FAN = None  # Per-worker singleton.


def _get_worker_fan():
    global _WORKER_FAN
    if _WORKER_FAN is None:
        import face_alignment
        # face_alignment renamed _2D to TWO_D in newer releases.
        lm_enum = face_alignment.LandmarksType
        if hasattr(lm_enum, "TWO_D"):
            lm_type = lm_enum.TWO_D
        elif hasattr(lm_enum, "_2D"):
            lm_type = lm_enum._2D
        else:
            raise RuntimeError(
                f"face_alignment.LandmarksType has neither TWO_D nor _2D: {list(lm_enum)}"
            )
        _WORKER_FAN = face_alignment.FaceAlignment(
            lm_type,
            flip_input=False,
            device="cpu",  # Avoid CUDA re-init inside forked DataLoader workers.
        )
    return _WORKER_FAN


def _fan_run(image_np: np.ndarray):
    """Return (bbox_or_[0], bbox_type), matching decalib FAN.run.

    image: 0-255, uint8, rgb, [h, w, 3]
    """
    fan = _get_worker_fan()
    out = fan.get_landmarks(image_np)
    if out is None:
        return [0], "kpt68"
    kpt = out[0].squeeze()
    left = float(np.min(kpt[:, 0]))
    right = float(np.max(kpt[:, 0]))
    top = float(np.min(kpt[:, 1]))
    bottom = float(np.max(kpt[:, 1]))
    return [left, top, right, bottom], "kpt68"


def _bbox2point(left, right, top, bottom, bbox_type):
    """Mirror DECA TestData.bbox2point."""
    if bbox_type == "kpt68":
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([
            right - (right - left) / 2.0,
            bottom - (bottom - top) / 2.0,
        ])
    elif bbox_type == "bbox":
        old_size = (right - left + bottom - top) / 2
        center = np.array([
            right - (right - left) / 2.0,
            bottom - (bottom - top) / 2.0 + old_size * 0.12,
        ])
    else:
        raise NotImplementedError(bbox_type)
    return old_size, center


def _deca_preprocess(
    image_path: str,
    detector_run,
    pre_size: int = 256,
    crop_scale: float = 1.25,
    crop_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Apply DECA TestData-style crop preprocessing.

    Returns:
        dst_image: (3, crop_size, crop_size) float32 in [0, 1], CHW.
        tform_params: (3, 3) similarity transform used by Stage2 reprojection.
        fan_detected: whether FAN detected a face.
    """
    im = imread(image_path)
    if pre_size is not None:
        im = (resize(im, (pre_size, pre_size), anti_aliasing=True) * 255.).astype(np.uint8)
    image = np.array(im)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    h, w, _ = image.shape

    bbox, bbox_type = detector_run(image)
    fan_detected = len(bbox) >= 4
    if not fan_detected:
        # DECA fallback: treat the full image as a bbox.
        left, right, top, bottom = 0, h - 1, 0, w - 1
        bbox_type = "bbox"
    else:
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]

    old_size, center = _bbox2point(left, right, top, bottom, bbox_type)
    sz = int(old_size * crop_scale)
    src_pts = np.array([
        [center[0] - sz / 2, center[1] - sz / 2],
        [center[0] - sz / 2, center[1] + sz / 2],
        [center[0] + sz / 2, center[1] - sz / 2],
    ])
    DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
    tform = estimate_transform("similarity", src_pts, DST_PTS)

    image_norm = image / 255.0
    dst_image = warp(image_norm, tform.inverse, output_shape=(crop_size, crop_size))
    dst_image = dst_image.transpose(2, 0, 1).astype(np.float32)
    return dst_image, tform.params.astype(np.float32), fan_detected


# ---------------------------------------------------------------------------
# Stage1Dataset
# ---------------------------------------------------------------------------
class Stage1Dataset(Dataset):
    """Supervised Stage1 dataset."""

    def __init__(
        self,
        sources: List[Dict[str, str]],
        image_size: int = 224,
        crop_scale: float = 1.25,
        prompt_key_en: str = "instruction_en",
        prompt_key_cn: str = "instruction_cn",
        lang_prob_en: float = 0.5,
        use_fan: bool = True,
        pre_size: int = 256,
        verbose: bool = True,
    ):
        """
        Args:
            sources: dicts with {jsonl, src_root, params_root}; multiple sources are merged.
            image_size: output crop size, normally 224 for DECA.
            crop_scale: FAN bbox expansion scale.
            prompt_key_en / prompt_key_cn: prompt fields in JSONL.
            lang_prob_en: probability of sampling the English prompt field.
            use_fan: false skips FAN and uses full-image bbox.
            pre_size: pre-resize size aligned with extract_deca_params.py.
        """
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.prompt_key_en = prompt_key_en
        self.prompt_key_cn = prompt_key_cn
        self.lang_prob_en = float(lang_prob_en)
        self.use_fan = use_fan
        self.pre_size = pre_size

        self.samples: List[Dict[str, Any]] = []
        for src in sources:
            jsonl_path = src["jsonl"]
            src_root = src["src_root"].rstrip("/")
            params_root = src["params_root"].rstrip("/")
            n_add = 0
            with open(jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    self.samples.append({
                        "item": item,
                        "src_root": src_root,
                        "params_root": params_root,
                    })
                    n_add += 1
            if verbose:
                print(f"[Stage1Dataset] loaded {n_add} samples from {jsonl_path}")
        if verbose:
            print(f"[Stage1Dataset] total samplesx: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # Path mapping
    # ------------------------------------------------------------------
    def _image_to_param_path(self, image_path: str, src_root: str, params_root: str) -> str:
        """Map an image path under src_root to its .pt path under params_root."""
        rel = os.path.relpath(image_path, src_root)
        rel_noext = os.path.splitext(rel)[0]
        return os.path.join(params_root, rel_noext + ".pt")

    # ------------------------------------------------------------------
    # Prompt sampling
    # ------------------------------------------------------------------
    def _pick_from_field(self, item: Dict[str, Any], key: str) -> Optional[str]:
        val = item.get(key, None)
        if val is None:
            return None
        if isinstance(val, list):
            if len(val) == 0:
                return None
            return random.choice(val)
        return str(val)

    def _pick_prompt(self, item: Dict[str, Any], idx: int) -> Tuple[str, str]:
        """Sample a prompt language, falling back to the other field if needed."""
        del idx
        if random.random() < self.lang_prob_en:
            primary_key, fallback_key, lang = self.prompt_key_en, self.prompt_key_cn, "en"
        else:
            primary_key, fallback_key, lang = self.prompt_key_cn, self.prompt_key_en, "cn"
        prompt = self._pick_from_field(item, primary_key)
        if prompt is None:
            prompt = self._pick_from_field(item, fallback_key)
            lang = "en" if lang == "cn" else "cn"
        if prompt is None:
            raise KeyError(
                f"Sample has no prompt in either '{self.prompt_key_en}' or "
                f"'{self.prompt_key_cn}' (pair_id={item.get('pair_id')})"
            )
        return prompt, lang

    # ------------------------------------------------------------------
    # Main sample path
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]
        item = rec["item"]
        src_root = rec["src_root"]
        params_root = rec["params_root"]

        image_a_path = item["image_a_path"]
        image_b_path = item["image_b_path"]

        # ---- 1) DECA-compatible reference-image preprocessing ----
        detector_run = _fan_run if self.use_fan else (lambda _img: ([0], "kpt68"))
        dst_image, tform_params, fan_detected = _deca_preprocess(
            image_path=image_a_path,
            detector_run=detector_run,
            pre_size=self.pre_size,
            crop_scale=self.crop_scale,
            crop_size=self.image_size,
        )
        ref_image = torch.from_numpy(dst_image)       # (3, 224, 224) float32 in [0, 1]
        tform = torch.from_numpy(tform_params)        # (3, 3) float32

        # ---- 2) Load target DECA parameters ----
        pt_path = self._image_to_param_path(image_b_path, src_root, params_root)
        params = torch.load(pt_path, map_location="cpu", weights_only=True)
        # Expected keys include shape, tex, exp, pose, cam, light.
        psi_tgt = params["exp"].detach().float().view(-1)                # (50,)
        pose = params["pose"].detach().float().view(-1)                  # (6,)
        jaw_tgt = pose[3:6].clone()                                      # (3,)

        # ---- 3) Sample prompt ----
        prompt, lang = self._pick_prompt(item, idx)

        return {
            "ref_image": ref_image,       # (3, 224, 224) float [0, 1]
            "tform": tform,               # (3, 3) similarity transform for Stage2.
            "psi_tgt": psi_tgt,           # (50,)
            "jaw_tgt": jaw_tgt,           # (3,)
            "prompt": prompt,             # str
            "lang": lang,                 # "en" / "cn"
            "pair_id": item.get("pair_id", ""),
            "fan_detected": fan_detected,
        }


# ---------------------------------------------------------------------------
# Collate: tokenize and pad per batch.
# ---------------------------------------------------------------------------
def build_collate_fn(tokenizer, max_length: int = 64):
    """Build the collate_fn used by the training DataLoader."""

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ref_image = torch.stack([b["ref_image"] for b in batch], dim=0)
        tform = torch.stack([b["tform"] for b in batch], dim=0)
        psi_tgt = torch.stack([b["psi_tgt"] for b in batch], dim=0)
        jaw_tgt = torch.stack([b["jaw_tgt"] for b in batch], dim=0)
        prompts = [b["prompt"] for b in batch]
        texts = []
        for prompt in prompts:
            if hasattr(tokenizer, "apply_chat_template"):
                texts.append(tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ))
            else:
                texts.append(prompt)
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "ref_image": ref_image,
            "tform": tform,
            "psi_tgt": psi_tgt,
            "jaw_tgt": jaw_tgt,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "prompts": prompts,
            "langs": [b["lang"] for b in batch],
            "pair_ids": [b["pair_id"] for b in batch],
            "fan_detected": torch.tensor(
                [b["fan_detected"] for b in batch], dtype=torch.bool
            ),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="./face_emoji/v1_pairs_with_instructions.jsonl")
    p.add_argument("--src_root",
                   default="./face_emoji/final_data_v1_bucket_postprocessed")
    p.add_argument("--params_root", default="./face_emoji/deca_params/v1")
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()

    ds = Stage1Dataset(sources=[{
        "jsonl": args.jsonl,
        "src_root": args.src_root,
        "params_root": args.params_root,
    }])
    print(f"dataset size = {len(ds)}")

    # Inspect a few samples.
    for i in range(min(args.n, len(ds))):
        sample = ds[i]
        print(f"\n[sample {i}]")
        print(f"  pair_id       = {sample['pair_id']}")
        print(f"  lang          = {sample['lang']}")
        print(f"  prompt        = {sample['prompt']}")
        print(f"  ref_image     = {tuple(sample['ref_image'].shape)} dtype={sample['ref_image'].dtype}")
        print(f"  psi_tgt       = {tuple(sample['psi_tgt'].shape)}  "
              f"mean={sample['psi_tgt'].mean().item():.3f} std={sample['psi_tgt'].std().item():.3f}")
        print(f"  jaw_tgt       = {sample['jaw_tgt'].tolist()}")
        print(f"  fan_detected  = {sample['fan_detected']}")

    # Estimate the prompt-language distribution without loading images.
    n_en, n_cn = 0, 0
    for i in range(len(ds)):
        _, lang = ds._pick_prompt(ds.samples[i]["item"], i)
        if lang == "en":
            n_en += 1
        else:
            n_cn += 1
    print(f"\n[epoch lang dist (Bernoulli p_en={ds.lang_prob_en})]"
          f" en={n_en}, cn={n_cn}, "
          f"ratio={n_en / max(n_en + n_cn, 1):.4f} / {n_cn / max(n_en + n_cn, 1):.4f}")
