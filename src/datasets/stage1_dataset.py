# -*- coding: utf-8 -*-
"""
Stage1 Dataset: Conditional DECA Encoder 的训练数据集.

输入 jsonl 每条样本:
    {
      "pair_id": ...,
      "image_a_path": "./data/.../{bucket}/{exp_a}/{name}.JPG",  # 参考图
      "image_b_path": "./data/.../{bucket}/{exp_b}/{name}.JPG",  # 目标图
      "instruction_en": [5 条英文指令, list],
      ... (其余字段不用)
    }

Dataset 在 __getitem__ 做:
    1) 读 image_a, 在线 FAN 检测 + 仿射 crop 到 224x224  -> ref_image
    2) 根据 image_b_path 映射出 {params_root}/{bucket}/{exp_b}/{name}.pt
       读 .pt 中的 exp[:50] 作 psi_Tgt, pose[3:6] 作 jaw_Tgt
    3) 从 instruction_en / instruction_cn 各 5 条里伯努利抽样 1 条:
         lang_prob_en 概率抽英文, 1-lang_prob_en 概率抽中文 (默认 0.5).
         独立随机 -> 能让同一样本跨 epoch 看到不同语言指令, 算作文本侧数据增强.
         单 epoch 中/英文比例期望 50/50, 会有 ±√(N/4) 的波动 (N=3014 时 ±27)。

collate_fn:
    batch 内统一 tokenize 并 padding (传入的 tokenizer 是 Qwen3 tokenizer).

注:
    - 我们复用 scripts/extract_deca_params.py 的 FAN crop 协议
      (bbox 扩成正方形, 再 resize 到 224, 与 DECA 原始输入一致).
    - FAN 未检测到人脸时降级为整图 resize 224, 并在 batch 里返回 fan_detected=False.
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
# FAN + crop (沿用 extract_deca_params.py 里的 CPUFAN 思路, 在 worker 内懒加载)
# ---------------------------------------------------------------------------
_WORKER_FAN = None  # 每个 DataLoader worker 内单例


def _get_worker_fan():
    global _WORKER_FAN
    if _WORKER_FAN is None:
        import face_alignment
        # face_alignment >=1.4.0 把 _2D 重命名为 TWO_D, 这里做向下兼容
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
            device="cpu",  # 关键: 避免 DataLoader fork 后 CUDA re-init
        )
    return _WORKER_FAN


def _fan_run(image_np: np.ndarray):
    """与 decalib.datasets.detectors.FAN.run 签名一致: 返回 (bbox_or_[0], bbox_type).

    image: 0-255, uint8, rgb, [h, w, 3]
    成功: ([left, top, right, bottom], 'kpt68')
    失败: ([0], 'kpt68')
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
    """完全复制自 DECA 官方 TestData.bbox2point (见 decalib/datasets/testdata.py)."""
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
    """完全复刻 DECA 官方 TestData._process 的管线 (与 scripts/extract_deca_params.py 一致).

    Returns:
        dst_image: (3, crop_size, crop_size) float32 in [0, 1], CHW, 同官方 encode 输入.
        tform_params: (3, 3) float32, similarity transform 矩阵 (Stage2 逐像素重投影用).
        fan_detected: bool, FAN 是否成功检测到人脸.
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
        # 官方兜底: 整图当 bbox, 且 bbox_type 改为 'bbox'
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
    """Stage1 监督训练数据集."""

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
            sources: 每项字典需要含 {jsonl, src_root, params_root}. 支持多数据源合并.
            image_size: ref_image 输出尺寸, 默认 224 (DECA 原版输入).
            crop_scale: FAN bbox 扩张系数, 默认 1.25.
            prompt_key_en / prompt_key_cn: jsonl 里英文 / 中文指令字段名.
                值为 list 时随机抽 1 条; 若某条样本缺失某个语言字段, 会自动回落到另一个.
            lang_prob_en: 伯努利抽样英文的概率 (默认 0.5, 即中/英 50/50).
            use_fan: 关闭后跳过 FAN, 直接整图 bbox 走官方几何 (调试用).
            pre_size: 官方 TestData 的预 resize 边长 (默认 256, 与 extract_deca_params 对齐).
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
    # 路径映射
    # ------------------------------------------------------------------
    def _image_to_param_path(self, image_path: str, src_root: str, params_root: str) -> str:
        """把 image_b 在 src_root 下的相对路径映射到 params_root 下的 .pt."""
        rel = os.path.relpath(image_path, src_root)
        rel_noext = os.path.splitext(rel)[0]
        return os.path.join(params_root, rel_noext + ".pt")

    # ------------------------------------------------------------------
    # Prompt 抽取 (中英文交替采样)
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
        """伯努利抽样语言 (概率 lang_prob_en -> en), 并从对应字段里随机抽 1 条.
        若首选语言字段缺失/为空, 自动回落到另一个语言.

        idx 参数仅用于跟 random state 解耦 (无硬依赖), 保留是为了后续如果想做 
        seeded sampling 的时候留个接口 (当前版本不用 idx)."""
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
    # 主逻辑
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]
        item = rec["item"]
        src_root = rec["src_root"]
        params_root = rec["params_root"]

        image_a_path = item["image_a_path"]
        image_b_path = item["image_b_path"]

        # ---- 1) 完全对齐 DECA 官方 TestData._process 的预处理 ----
        #   与 scripts/extract_deca_params.py::CropDataset._process 等价.
        #   use_fan=False 时正常调 _fan_run 但结果会被当作 miss, 走整图 bbox 分支.
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

        # ---- 2) 读目标参数 ----
        pt_path = self._image_to_param_path(image_b_path, src_root, params_root)
        params = torch.load(pt_path, map_location="cpu", weights_only=True)
        # params dict 结构: shape(100), tex(50), exp(50), pose(6), cam(3), light(27), ...
        psi_tgt = params["exp"].detach().float().view(-1)                # (50,)
        pose = params["pose"].detach().float().view(-1)                  # (6,)
        jaw_tgt = pose[3:6].clone()                                      # (3,)

        # ---- 3) 抽 prompt (根据 idx 奇偶交替采样中/英文) ----
        prompt, lang = self._pick_prompt(item, idx)

        return {
            "ref_image": ref_image,       # (3, 224, 224) float [0, 1]
            "tform": tform,               # (3, 3) similarity transform, Stage2 逐像素用
            "psi_tgt": psi_tgt,           # (50,)
            "jaw_tgt": jaw_tgt,           # (3,)
            "prompt": prompt,             # str
            "lang": lang,                 # "en" / "cn"
            "pair_id": item.get("pair_id", ""),
            "fan_detected": fan_detected,
        }


# ---------------------------------------------------------------------------
# Collate: 批内 tokenize + pad
# ---------------------------------------------------------------------------
def build_collate_fn(tokenizer, max_length: int = 64):
    """返回一个 collate_fn. 训练脚本里: DataLoader(..., collate_fn=build_collate_fn(tok))"""

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
# 自测
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

    # 抽 n 条看内容
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

    # 统计一整个 epoch 中英文分布 (只跑 _pick_prompt 不读图, 很快)
    # 因为是伯努利采样, 每次跑结果不同, 约 50/50 ±√(N/4)
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
