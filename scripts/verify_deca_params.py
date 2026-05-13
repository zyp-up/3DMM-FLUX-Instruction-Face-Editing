"""
DECA 参数提取完整性校验工具。

用法:
    python scripts/verify_deca_params.py \
        --src_root /.../final_data_raf_bucket_postprocessed \
        --out_root /.../deca_params/raf

支持与 extract 一样的多 (src,out) 对同时校验:
    --src_root A --out_root A_out --src_root B --out_root B_out

nohup python scripts/verify_deca_params.py \
    --src_root ./data/final_data_raf_bucket_postprocessed \
    --out_root ./deca_params/raf \
    --src_root ./data/final_data_v1_bucket_postprocessed \
    --out_root ./deca_params/v1 \
    --deep_check \
    > verify_deep.log 2>&1 &

输出:
  1) 终端汇总表: total / saved / failed / missing / orphan / dup_fail
  2) {out_root}/_missing.txt     : 漏网源图路径 (需重跑)
  3) {out_root}/_orphan.txt      : 多出的 .pt 路径 (可考虑清理)
  4) {out_root}/_dup_fail.txt    : 既成功又失败的路径 (异常,建议删掉 .pt 重跑)
  5) {out_root}/_verify_summary.json : 结构化汇总
  6) 可选 --deep_check: 逐个打开 .pt 验证字段完整 & 形状正确
"""

import os
import json
import glob
import argparse
from collections import defaultdict

import torch
from tqdm import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
REQUIRED_FIELDS = {
    "shape":  (1, 100),
    "tex":    (1, 50),
    "exp":    (1, 50),
    "pose":   (1, 6),
    "cam":    (1, 3),
    "light":  (1, 9, 3),
    "detail": (1, 128),
    "tform":  (1, 3, 3),
}


def walk_images(src_root):
    out = []
    for root, _, files in os.walk(src_root):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, fn))
    return set(out)


def walk_pts(out_root):
    """扫描 out_root 下所有 .pt (排除 failed jsonl 与 summary)."""
    out = []
    for root, _, files in os.walk(out_root):
        for fn in files:
            if fn.endswith(".pt"):
                out.append(os.path.join(root, fn))
    return set(out)


def pt_to_src(pt_path, src_root, out_root):
    """
    根据镜像目录约定, 把 out_root/<rel>/name.pt 还原成候选源图路径 (多扩展名都试一遍).
    返回实际存在的源图, 否则返回 None.
    """
    rel = os.path.relpath(pt_path, out_root)
    rel_no_ext = os.path.splitext(rel)[0]
    for ext in IMG_EXTS + tuple(e.upper() for e in IMG_EXTS):
        cand = os.path.join(src_root, rel_no_ext + ext)
        if os.path.exists(cand):
            return cand
    # 源图已经不存在,返回去掉扩展名的规范形式,便于 diff
    return os.path.join(src_root, rel_no_ext + ".<?>")


def src_to_pt(src_path, src_root, out_root):
    rel = os.path.relpath(src_path, src_root)
    rel_no_ext = os.path.splitext(rel)[0]
    return os.path.join(out_root, rel_no_ext + ".pt")


def load_failed(out_root):
    """合并 _failed.jsonl + _failed_shard*.jsonl, 返回 {src_image_path -> reason}."""
    failed = {}
    patterns = ["_failed.jsonl", "_failed_shard*.jsonl"]
    for pat in patterns:
        for fp in glob.glob(os.path.join(out_root, pat)):
            with open(fp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    sp = rec.get("src_image_path")
                    if sp:
                        failed[sp] = rec.get("reason", "unknown")
    return failed


def deep_check_pt(pt_path):
    """打开 .pt 检查必需字段与形状, 返回 (ok, reason)."""
    try:
        d = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        return False, f"load_error: {type(e).__name__}: {e}"
    for k, shape in REQUIRED_FIELDS.items():
        if k not in d:
            return False, f"missing_field:{k}"
        t = d[k]
        if not hasattr(t, "shape"):
            return False, f"not_tensor:{k}"
        if tuple(t.shape) != shape:
            return False, f"bad_shape:{k}:{tuple(t.shape)}!={shape}"
    return True, "ok"


def verify_one(src_root, out_root, deep_check=False):
    src_root = os.path.abspath(src_root)
    out_root = os.path.abspath(out_root)
    print(f"\n=== verify ===")
    print(f"  src: {src_root}")
    print(f"  out: {out_root}")

    print("  scanning source images ...")
    A = walk_images(src_root)                          # set of src paths
    print(f"    source images: {len(A)}")

    print("  scanning .pt files ...")
    B_pt = walk_pts(out_root)                          # set of pt paths
    # 把 .pt 路径映射回 src 路径, 便于和 A 取交差
    pt_to_src_map = {}
    for pt in B_pt:
        sp = pt_to_src(pt, src_root, out_root)
        pt_to_src_map[pt] = sp
    B = set(pt_to_src_map.values())
    print(f"    .pt files    : {len(B_pt)}")

    print("  loading failed logs ...")
    F_map = load_failed(out_root)
    F = set(F_map.keys())
    print(f"    failed records: {len(F)}")

    missing  = A - B - F                               # 既没 .pt 也没失败记录
    orphan   = B - A                                   # 有 .pt 但源图不在
    dup_fail = B & F                                   # 既成功又失败
    covered  = (A & B) | (A & F)

    print(f"\n  --- summary ---")
    print(f"    total source  : {len(A)}")
    print(f"    saved (A∩B)   : {len(A & B)}")
    print(f"    failed (A∩F)  : {len(A & F)}")
    print(f"    covered       : {len(covered)}  ({100.0*len(covered)/max(1,len(A)):.2f}%)")
    print(f"    MISSING       : {len(missing)}   <-- 需重跑")
    print(f"    orphan .pt    : {len(orphan)}")
    print(f"    dup_fail      : {len(dup_fail)}")

    # 写出清单
    def dump(path, items):
        with open(path, "w") as f:
            for x in sorted(items):
                f.write(x + "\n")
    dump(os.path.join(out_root, "_missing.txt"),  missing)
    dump(os.path.join(out_root, "_orphan.txt"),   orphan)
    dump(os.path.join(out_root, "_dup_fail.txt"), dup_fail)

    summary = {
        "src_root": src_root, "out_root": out_root,
        "total_source": len(A),
        "saved": len(A & B), "failed": len(A & F),
        "missing": len(missing), "orphan": len(orphan),
        "dup_fail": len(dup_fail),
        "coverage": len(covered) / max(1, len(A)),
    }

    # 深度字段校验
    if deep_check and len(B_pt) > 0:
        print("  deep checking .pt contents ...")
        bad = []
        for pt in tqdm(sorted(B_pt)):
            ok, reason = deep_check_pt(pt)
            if not ok:
                bad.append((pt, reason))
        dump_bad = os.path.join(out_root, "_bad_pt.txt")
        with open(dump_bad, "w") as f:
            for pt, r in bad:
                f.write(f"{pt}\t{r}\n")
        summary["bad_pt"] = len(bad)
        print(f"    bad_pt        : {len(bad)}  (see {dump_bad})")

    with open(os.path.join(out_root, "_verify_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", action="append", required=True)
    ap.add_argument("--out_root", action="append", required=True)
    ap.add_argument("--deep_check", action="store_true",
                    help="逐个打开 .pt 校验字段和形状, 慢但彻底")
    args = ap.parse_args()
    assert len(args.src_root) == len(args.out_root)

    all_summary = []
    for s, o in zip(args.src_root, args.out_root):
        all_summary.append(verify_one(s, o, deep_check=args.deep_check))

    print("\n=== overall ===")
    total  = sum(x["total_source"] for x in all_summary)
    saved  = sum(x["saved"] for x in all_summary)
    failed = sum(x["failed"] for x in all_summary)
    miss   = sum(x["missing"] for x in all_summary)
    print(f"  total={total}  saved={saved}  failed={failed}  missing={miss}")
    if miss == 0:
        print("  ✅ 全部源图都有对应产出或明确失败记录")
    else:
        print(f"  ❌ 有 {miss} 张图未处理, 请查看各 out_root 下的 _missing.txt")


if __name__ == "__main__":
    main()
