"""Generate expression-pair JSONL files.

The script groups images by identity and writes all C(n, 2) expression pairs
for each person. Output paths follow --data_dir; relative paths are preferred
for release-friendly configs.

Examples:
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
    """Extract (person_id, dataset) from a dataset-specific filename."""
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
    parser = argparse.ArgumentParser(description='Generate expression-pair JSONL')
    parser.add_argument('--data_dir', type=str, required=True, help='final_data directory, preferably relative')
    parser.add_argument('--output', type=str, required=True, help='output JSONL path, preferably relative')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    # Group images by identity.
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

    print(f"Total identities: {len(person_images)}")
    ds_count = defaultdict(int)
    for pid, imgs in person_images.items():
        ds_count[imgs[0]['dataset']] += 1
    for ds, cnt in sorted(ds_count.items()):
        print(f"  {ds}: {cnt} identities")

    # Generate C(n, 2) pairs per identity.
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

    print(f"\nTotal pairs: {pair_count}")
    ds_pairs = defaultdict(int)
    for pid, imgs in person_images.items():
        n = len(imgs)
        ds_pairs[imgs[0]['dataset']] += n * (n - 1) // 2
    for ds, cnt in sorted(ds_pairs.items()):
        print(f"  {ds}: {cnt} pairs")
    print(f"\nJSONL saved: {args.output}")


if __name__ == '__main__':
    main()
