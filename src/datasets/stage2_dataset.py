# -*- coding: utf-8 -*-
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage.io import imread
from torch.utils.data import Dataset

from src.datasets.stage1_dataset import _deca_preprocess, _fan_run


_PARAM_KEYS = ["shape", "tex", "exp", "pose", "cam", "light", "detail"]


class Stage2Dataset(Dataset):
    def __init__(
        self,
        sources: List[Dict[str, str]],
        deca_image_size: int = 224,
        crop_scale: float = 1.25,
        prompt_key_en: str = "instruction_en",
        prompt_key_cn: str = "instruction_cn",
        lang_prob_en: float = 0.5,
        use_fan: bool = True,
        pre_size: int = 256,
        verbose: bool = True,
    ):
        self.deca_image_size = deca_image_size
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
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    image_b_path = item["image_b_path"]
                    rel_b = os.path.relpath(image_b_path, src_root)
                    bucket = rel_b.split(os.sep)[0]
                    self.samples.append(
                        {
                            "item": item,
                            "src_root": src_root,
                            "params_root": params_root,
                            "bucket": bucket,
                        }
                    )
                    n_add += 1
            if verbose:
                print(f"[Stage2Dataset] loaded {n_add} samples from {jsonl_path}")
        if verbose:
            print(f"[Stage2Dataset] total samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def _image_to_param_path(self, image_path: str, src_root: str, params_root: str) -> str:
        rel = os.path.relpath(image_path, src_root)
        rel_noext = os.path.splitext(rel)[0]
        return os.path.join(params_root, rel_noext + ".pt")

    def _load_param_dict(self, pt_path: str) -> Dict[str, torch.Tensor]:
        params = torch.load(pt_path, map_location="cpu", weights_only=True)
        out = {}
        for k in _PARAM_KEYS:
            if k not in params:
                continue
            v = params[k]
            if not torch.is_tensor(v):
                continue
            v = v.detach().float()
            if v.ndim > 1 and v.shape[0] == 1:
                v = v.squeeze(0)
            out[k] = v
        return out

    def _pick_from_field(self, item: Dict[str, Any], key: str) -> Optional[str]:
        val = item.get(key, None)
        if val is None:
            return None
        if isinstance(val, list):
            if len(val) == 0:
                return None
            return random.choice(val)
        return str(val)

    def _load_native_image_chw(self, image_path: str) -> torch.Tensor:
        im = imread(image_path)
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        if im.ndim == 3 and im.shape[2] > 3:
            im = im[:, :, :3]
        h, w = im.shape[:2]
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(f"Stage2 native image size must be multiple of 16, got {(h, w)}: {image_path}")
        im = im.astype(np.float32) / 255.0
        return torch.from_numpy(im.transpose(2, 0, 1).copy())

    def _pick_prompt(self, item: Dict[str, Any]) -> Tuple[str, str]:
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]
        item = rec["item"]
        src_root = rec["src_root"]
        params_root = rec["params_root"]

        image_a_path = item["image_a_path"]
        image_b_path = item["image_b_path"]

        detector_run = _fan_run if self.use_fan else (lambda _img: ([0], "kpt68"))

        ref_deca_image, _, ref_fan_detected = _deca_preprocess(
            image_path=image_a_path,
            detector_run=detector_run,
            pre_size=self.pre_size,
            crop_scale=self.crop_scale,
            crop_size=self.deca_image_size,
        )
        ref_native_image = self._load_native_image_chw(image_a_path)
        tgt_train_image = self._load_native_image_chw(image_b_path)
        if ref_native_image.shape[-2:] != tgt_train_image.shape[-2:]:
            ref_native_image = F.interpolate(
                ref_native_image.unsqueeze(0),
                size=tgt_train_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        ref_params_path = self._image_to_param_path(image_a_path, src_root, params_root)
        tgt_params_path = self._image_to_param_path(image_b_path, src_root, params_root)

        ref_params = self._load_param_dict(ref_params_path)
        tgt_params = self._load_param_dict(tgt_params_path)

        prompt, lang = self._pick_prompt(item)

        return {
            "ref_image": torch.from_numpy(ref_deca_image),
            "ref_native_image": ref_native_image,
            "tgt_image": tgt_train_image,
            "ref_params": ref_params,
            "tgt_params": tgt_params,
            "psi_tgt": tgt_params["exp"].view(-1),
            "jaw_tgt": tgt_params["pose"].view(-1)[3:6].clone(),
            "prompt": prompt,
            "lang": lang,
            "pair_id": item.get("pair_id", ""),
            "ref_fan_detected": ref_fan_detected,
            "bucket": rec["bucket"],
        }


def build_stage2_collate_fn():
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ref_image = torch.stack([b["ref_image"] for b in batch], dim=0)
        tgt_shapes = {tuple(b["tgt_image"].shape[-2:]) for b in batch}
        ref_native_shapes = {tuple(b["ref_native_image"].shape[-2:]) for b in batch}
        if len(tgt_shapes) != 1:
            raise ValueError(f"Stage2 batch mixes native resolutions: {sorted(tgt_shapes)}")
        if ref_native_shapes != tgt_shapes:
            raise ValueError(f"ref_native_image shapes {sorted(ref_native_shapes)} do not match target shapes {sorted(tgt_shapes)}")
        ref_native_image = torch.stack([b["ref_native_image"] for b in batch], dim=0)
        tgt_image = torch.stack([b["tgt_image"] for b in batch], dim=0)
        psi_tgt = torch.stack([b["psi_tgt"] for b in batch], dim=0)
        jaw_tgt = torch.stack([b["jaw_tgt"] for b in batch], dim=0)

        ref_params = {}
        tgt_params = {}
        keys = batch[0]["ref_params"].keys()
        for k in keys:
            ref_params[k] = torch.stack([b["ref_params"][k] for b in batch], dim=0)
        keys = batch[0]["tgt_params"].keys()
        for k in keys:
            tgt_params[k] = torch.stack([b["tgt_params"][k] for b in batch], dim=0)

        return {
            "ref_image": ref_image,
            "ref_native_image": ref_native_image,
            "tgt_image": tgt_image,
            "ref_params": ref_params,
            "tgt_params": tgt_params,
            "psi_tgt": psi_tgt,
            "jaw_tgt": jaw_tgt,
            "prompts": [b["prompt"] for b in batch],
            "langs": [b["lang"] for b in batch],
            "pair_ids": [b["pair_id"] for b in batch],
            "buckets": [b["bucket"] for b in batch],
            "ref_fan_detected": torch.tensor([b["ref_fan_detected"] for b in batch], dtype=torch.bool),
        }

    return collate_fn