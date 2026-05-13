"""
生成表情配对 JSONL
扫描 final_data 中所有图片，按人物ID分组，
对同一人的所有表情做 C(n,2) 组合，输出配对 JSONL。
JSONL 中的图片路径默认跟随 --data_dir 的写法；开源配置推荐使用相对路径。

启动命令：
python3 generate_pairs_jsonl.py \
    --data_dir ./face_emoji/final_data_raf_bucket_postprocessed \
    --output ./raf_pairs.jsonl

python3 generate_pairs_jsonl.py \
    --data_dir ./face_emoji/final_data_v1_bucket_postprocessed \
    --output ./v1_pairs.jsonl

"""

import os
import json
import argparse
from itertools import combinations
from collections import defaultdict

EXPRESSIONS = ['neutral', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']


def extract_person_id(filename):
    """从文件名提取 (person_id, dataset)"""
    if filename.startswith('multi_pie_'):
        parts = filename.split('_')
        subject = parts[2]
        return f'mp_{subject}', 'multi_pie'
    elif filename.startswith('kdef_'):
        raw = filename[5:]
        person = raw[1:4]
        return f'kdef_{person}', 'kdef'
    elif filename.startswith('oulu_'):
        parts = filename.split('_')
        subject = parts[1]
        return f'oulu_{subject}', 'oulu'
    elif filename.startswith('raf_'):
        name_no_ext = os.path.splitext(filename)[0]
        return name_no_ext, 'raf'
    return None, None


def main():
    parser = argparse.ArgumentParser(description='生成表情配对 JSONL')
    parser.add_argument('--data_dir', type=str, required=True, help='final_data 目录路径，建议使用相对路径')
    parser.add_argument('--output', type=str, required=True, help='输出 JSONL 文件路径，建议使用相对路径')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    # 扫描所有图片，按人物ID分组
    person_images = defaultdict(list)

    for root, dirs, files in os.walk(data_dir):
        expr = os.path.basename(root)
        if expr not in EXPRESSIONS:
            continue
        for fname in sorted(files):
            person_id, dataset = extract_person_id(fname)
            if person_id is None:
                continue
            abs_path = os.path.join(root, fname)
            rel_path = os.path.join(args.data_dir.rstrip(os.sep), os.path.relpath(abs_path, data_dir))
            person_images[person_id].append({
                'expression': expr,
                'filename': fname,
                'abs_path': rel_path,
                'dataset': dataset,
            })

    # 统计
    print(f"总人数: {len(person_images)}")
    ds_count = defaultdict(int)
    for pid, imgs in person_images.items():
        ds_count[imgs[0]['dataset']] += 1
    for ds, cnt in sorted(ds_count.items()):
        print(f"  {ds}: {cnt} 人")

    # C(n,2) 生成配对
    pair_count = 0
    with open(args.output, 'w', encoding='utf-8') as f:
        for person_id in sorted(person_images.keys()):
            images = person_images[person_id]
            dataset = images[0]['dataset']

            for img_a, img_b in combinations(images, 2):
                pair = {
                    'pair_id': f"{person_id}_{img_a['expression']}_{img_b['expression']}",
                    'person_id': person_id,
                    'dataset': dataset,
                    'image_a_path': img_a['abs_path'],
                    'image_a_filename': img_a['filename'],
                    'expression_a': img_a['expression'],
                    'image_b_path': img_b['abs_path'],
                    'image_b_filename': img_b['filename'],
                    'expression_b': img_b['expression'],
                    'check_result': None,
                }
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                pair_count += 1

    # 输出统计
    print(f"\n总配对数: {pair_count}")
    ds_pairs = defaultdict(int)
    for pid, imgs in person_images.items():
        n = len(imgs)
        ds_pairs[imgs[0]['dataset']] += n * (n - 1) // 2
    for ds, cnt in sorted(ds_pairs.items()):
        print(f"  {ds}: {cnt} 对")
    print(f"\nJSONL 已保存: {args.output}")


if __name__ == '__main__':
    main()
