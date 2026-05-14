"""Offline DECA parameter extraction with mirrored output paths.

=============================================================================
Single-GPU example for raf + v1
=============================================================================
python scripts/extract_deca_params.py \
    --src_root ./face_emoji/final_data_raf_bucket_postprocessed \
    --out_root ./face_emoji/deca_params/raf \
    --src_root ./face_emoji/final_data_v1_bucket_postprocessed \
    --out_root ./face_emoji/deca_params/v1 \
    --batch_size 32 --num_workers 16

=============================================================================
Multi-GPU sharding example; scripts/run_extract_multigpu.sh is preferred
for r in 0 1 2 3 4 5 6 7; do \
    CUDA_VISIBLE_DEVICES=$r \
    nohup python scripts/extract_deca_params.py \
        --src_root ./face_emoji/final_data_raf_bucket_postprocessed \
        --out_root ./face_emoji/deca_params/raf \
        --src_root ./face_emoji/final_data_v1_bucket_postprocessed \
        --out_root ./face_emoji/deca_params/v1 \
        --shard_id $r --num_shards 8 \
        --batch_size 32 --num_workers 16 \
        > logs/extract_shard_$r.log 2>&1 & \
done; wait

=============================================================================
Notes
=============================================================================
1) Each shard processes imgs[shard_id::num_shards] and skips existing .pt files.
2) FAN is initialized once per DataLoader worker.
3) DECA.encode runs by GPU batch, then saves one .pt per source image.
4) Output paths mirror the source directory:
     {src_root}/<rel>/name.JPG  =>  {out_root}/<rel>/name.pt
   Example:
     .../final_data_raf_bucket_postprocessed/544x736/angry/raf_xxx.jpg
     .../ControlFace-main/face_emoji/deca_params/raf/544x736/angry/raf_xxx.pt
   .pt content: dict(shape, tex, exp, pose, cam, light, detail, tform, src_image_path).
5) Failed samples are appended to {out_root}/_failed_shard{shard_id}.jsonl.
6) Restarts are safe because existing outputs are skipped.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from skimage.transform import estimate_transform, warp, resize
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg


CODE_FIELDS = ["shape", "tex", "exp", "pose", "cam", "light", "detail"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class CPUFAN(object):
    """FAN-compatible detector forced to CPU for DataLoader workers."""

    def __init__(self):
        import face_alignment
        lm_enum = face_alignment.LandmarksType
        if hasattr(lm_enum, "TWO_D"):
            lm_type = lm_enum.TWO_D
        elif hasattr(lm_enum, "_2D"):
            lm_type = lm_enum._2D
        else:
            raise RuntimeError(f"Unsupported face_alignment.LandmarksType: {list(lm_enum)}")
        self.model = face_alignment.FaceAlignment(
            lm_type,
            flip_input=False,
            device="cpu",
        )

    def run(self, image):
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], "kpt68"
        kpt = out[0].squeeze()
        left = np.min(kpt[:, 0]); right = np.max(kpt[:, 0])
        top = np.min(kpt[:, 1]); bottom = np.max(kpt[:, 1])
        return [left, top, right, bottom], "kpt68"


def configure_deca_cfg(deca_data_dir: str):
    """Point DECA config paths to the selected DECA data directory."""
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


def walk_images(src_root: str):
    """Return sorted image paths under src_root."""
    result = []
    for root, _, files in os.walk(src_root):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                result.append(os.path.join(root, fn))
    result.sort()
    return result


def src_to_out_path(src_path: str, src_root: str, out_root: str) -> str:
    """Map {src_root}/<rel>/name.JPG to {out_root}/<rel>/name.pt."""
    rel = os.path.relpath(src_path, src_root)
    rel_no_ext = os.path.splitext(rel)[0]
    return os.path.join(out_root, rel_no_ext + ".pt")


def _bbox2point(left, right, top, bottom, bbox_type):
    """Mirror TestData.bbox2point for kpt68 and bbox detections."""
    if bbox_type == "kpt68":
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0,
                           bottom - (bottom - top) / 2.0])
    elif bbox_type == "bbox":
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0,
                           bottom - (bottom - top) / 2.0 + old_size * 0.12])
    else:
        raise NotImplementedError(bbox_type)
    return old_size, center


class CropDataset(Dataset):
    """CPU preprocessing dataset with one FAN instance per worker."""

    def __init__(self, img_paths, crop_size=224, scale=1.25, size=256):
        self.img_paths = img_paths
        self.crop_size = crop_size
        self.scale = scale
        self.size = size
        self.face_detector = None  # lazy init in worker

    def _lazy_init_detector(self):
        if self.face_detector is None:
            # Use CPU FAN to avoid CUDA re-init inside forked workers.
            self.face_detector = CPUFAN()

    def __len__(self):
        return len(self.img_paths)

    def _process(self, image_path):
        im = imread(image_path)
        if self.size is not None:
            im = (resize(im, (self.size, self.size), anti_aliasing=True) * 255.).astype(np.uint8)
        image = np.array(im)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        h, w, _ = image.shape

        bbox, bbox_type = self.face_detector.run(image)
        if len(bbox) < 4:
            left, right, top, bottom = 0, h - 1, 0, w - 1
            bbox_type = "bbox"
        else:
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
        old_size, center = _bbox2point(left, right, top, bottom, bbox_type)
        sz = int(old_size * self.scale)
        src_pts = np.array([
            [center[0] - sz / 2, center[1] - sz / 2],
            [center[0] - sz / 2, center[1] + sz / 2],
            [center[0] + sz / 2, center[1] - sz / 2],
        ])
        DST_PTS = np.array([[0, 0], [0, self.crop_size - 1], [self.crop_size - 1, 0]])
        tform = estimate_transform("similarity", src_pts, DST_PTS)

        image_norm = image / 255.
        dst_image = warp(image_norm, tform.inverse,
                         output_shape=(self.crop_size, self.crop_size))
        dst_image = dst_image.transpose(2, 0, 1)
        return (
            torch.tensor(dst_image).float(),
            torch.tensor(tform.params).float(),
        )

    def __getitem__(self, index):
        """Return an ok/fail record consumed by _collate."""
        self._lazy_init_detector()
        path = self.img_paths[index]
        try:
            img, tform = self._process(path)
            return {"ok": True, "path": path, "image": img, "tform": tform}
        except Exception as e:
            return {"ok": False, "path": path,
                    "reason": f"{type(e).__name__}: {e}"}


def _collate(batch):
    """Stack successful samples and keep failed records."""
    ok = [b for b in batch if b is not None and b["ok"]]
    fail = [b for b in batch if b is not None and not b["ok"]]
    out = {"fail": fail}
    if ok:
        out["paths"] = [b["path"] for b in ok]
        out["images"] = torch.stack([b["image"] for b in ok], dim=0)
        out["tforms"] = torch.stack([b["tform"] for b in ok], dim=0)
    else:
        out["paths"] = []
        out["images"] = None
        out["tforms"] = None
    return out


def process_one_dataset(deca: DECA, src_root: str, out_root: str,
                        device: str, image_size: int,
                        batch_size: int, num_workers: int,
                        shard_id: int, num_shards: int):
    """Extract one mirrored (src_root, out_root) pair."""
    src_root = os.path.abspath(src_root)
    out_root = os.path.abspath(out_root)
    os.makedirs(out_root, exist_ok=True)
    failed_log_path = os.path.join(out_root, f"_failed_shard{shard_id}.jsonl")

    print(f"\n[dataset][shard {shard_id}/{num_shards}] src={src_root}")
    print(f"                              out={out_root}")

    all_imgs = walk_images(src_root)
    sharded = all_imgs[shard_id::num_shards]

    # Resume-safe skip for existing .pt files.
    todo, n_existed = [], 0
    for p in sharded:
        if os.path.exists(src_to_out_path(p, src_root, out_root)):
            n_existed += 1
        else:
            todo.append(p)
    print(f"  total={len(all_imgs)}, shard={len(sharded)}, existed={n_existed}, todo={len(todo)}")

    if not todo:
        return

    dataset = CropDataset(todo, crop_size=224, scale=1.25, size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    n_saved, n_failed = 0, 0
    with open(failed_log_path, "a") as flog:
        pbar = tqdm(total=len(todo), desc=f"{os.path.basename(src_root)}[s{shard_id}]")
        for batch in loader:
            # Persist failures before continuing the batch.
            for f in batch["fail"]:
                flog.write(json.dumps({
                    "src_image_path": f["path"],
                    "reason": f["reason"],
                }) + "\n")
                n_failed += 1
            if batch["fail"]:
                flog.flush()
                pbar.update(len(batch["fail"]))

            if batch["images"] is None:
                continue

            # Batched GPU inference.
            images = batch["images"].to(device, non_blocking=True)
            with torch.no_grad():
                codedict = deca.encode(images)

            # Save one payload per source image.
            tforms = batch["tforms"]
            paths = batch["paths"]
            B = images.shape[0]
            for i in range(B):
                payload = {}
                for k in CODE_FIELDS:
                    if k in codedict:
                        payload[k] = codedict[k][i:i+1].detach().cpu()
                payload["tform"] = tforms[i:i+1].detach().cpu()
                payload["src_image_path"] = paths[i]
                out_path = src_to_out_path(paths[i], src_root, out_root)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(payload, out_path)
                n_saved += 1
            pbar.update(B)
        pbar.close()

    print(f"  saved:   {n_saved}")
    print(f"  failed:  {n_failed}  (see {failed_log_path})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, action="append", required=True,
                        help="source image root; repeat in the same order as --out_root")
    parser.add_argument("--out_root", type=str, action="append", required=True,
                        help="output parameter root; repeat in the same order as --src_root")
    parser.add_argument(
        "--deca_data_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "DECA", "data"),
        help="DECA data directory; default: ControlFace-main/DECA/data",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256,
                        help="TestData size value aligned with sample.py")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for DECA.encode")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="DataLoader preprocessing workers")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="current shard index, [0, num_shards)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="total shard count; usually number of GPUs")
    args = parser.parse_args()

    if len(args.src_root) != len(args.out_root):
        raise ValueError(
            f"--src_root ({len(args.src_root)}) and --out_root ({len(args.out_root)}) must have the same length"
        )
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"shard_id {args.shard_id} is outside [0, {args.num_shards})")

    print(f"[cfg] DECA data dir: {args.deca_data_dir}")
    print(f"[cfg] shard_id={args.shard_id}/{args.num_shards}, "
          f"batch_size={args.batch_size}, num_workers={args.num_workers}")
    configure_deca_cfg(args.deca_data_dir)

    print("[init] loading DECA ...")
    deca = DECA(config=deca_cfg, device=args.device)
    deca.eval()

    for src_root, out_root in zip(args.src_root, args.out_root):
        process_one_dataset(
            deca=deca,
            src_root=src_root,
            out_root=out_root,
            device=args.device,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )

    print("\n=== all done ===")


if __name__ == "__main__":
    main()
