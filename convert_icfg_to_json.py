"""
Convert ICFG-PEDES .pkl format to RaSa's expected JSON format.

Expected output format (per item):
{
    "file_path": "relative/path/to/image.jpg",
    "captions": ["caption1", "caption2", ...],
    "id": person_id_integer
}

Usage:
    python convert_icfg_to_json.py \
        --data_dir /path/to/WS_DATASETS/REID/ICFG-PEDES \
        --output_dir /path/to/WS_DATASETS/REID/ICFG-PEDES/processed_data
"""

import os
import json
import pickle
import argparse


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def convert_split(pkl_path, output_path, data_dir):
    """Convert a .pkl split file to RaSa JSON format."""
    data = load_pkl(pkl_path)
    print(f"Loaded {pkl_path}: {type(data)}")

    # Handle different possible .pkl structures
    results = []

    if isinstance(data, list):
        # List of dicts or tuples
        for item in data:
            if isinstance(item, dict):
                # Common format: {'file_path': ..., 'captions': [...], 'id': ...}
                entry = {
                    'file_path': item.get('file_path', item.get('img_path', '')),
                    'captions': item.get('captions', [item.get('caption', '')]),
                    'id': item.get('id', item.get('pid', item.get('person_id', 0))),
                }
                results.append(entry)
    elif isinstance(data, dict):
        # Could be {img_path: {captions: [...], id: ...}}
        print(f"  Keys: {list(data.keys())[:10]}")
        # Try to interpret based on structure
        if 'train' in str(pkl_path) or 'test' in str(pkl_path):
            for key, val in data.items():
                if isinstance(val, dict):
                    results.append({
                        'file_path': val.get('file_path', key),
                        'captions': val.get('captions', []),
                        'id': val.get('id', 0),
                    })

    if not results:
        # Fallback: print structure for debugging
        print(f"  Could not auto-convert. Data structure:")
        if isinstance(data, list) and len(data) > 0:
            print(f"  First item type: {type(data[0])}")
            print(f"  First item: {data[0]}")
        elif isinstance(data, dict):
            first_key = list(data.keys())[0]
            print(f"  First key: {first_key}")
            print(f"  First value: {data[first_key]}")
        return False

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} items to {output_path}")
    return True


def convert_icfg_json(json_path, output_dir):
    """
    Convert ICFG-PEDES.json (the main annotation file) to train/val/test splits.
    Common format:
    [
        {"split": "train", "file_path": "...", "captions": [...], "id": ...},
        ...
    ]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded ICFG-PEDES.json: {len(data)} items")

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        # Check if it has split info
        if 'split' in data[0]:
            splits = {}
            for item in data:
                split = item['split']
                if split not in splits:
                    splits[split] = []
                splits[split].append({
                    'file_path': item.get('file_path', item.get('img_path', '')),
                    'captions': item.get('captions', [item.get('caption', '')]),
                    'id': item.get('id', item.get('pid', 0)),
                })
        else:
            # No split field — look for id-based splitting
            # ICFG-PEDES typically: first 3102 IDs = train, next 500 = val, last 500 = test
            print("  No 'split' field found. Attempting ID-based split.")
            all_ids = sorted(set(item.get('id', item.get('pid', 0)) for item in data))
            print(f"  Total unique IDs: {len(all_ids)}")

            # Standard ICFG-PEDES split: 3102 train, 500 val, 500 test
            train_ids = set(all_ids[:3102])
            val_ids = set(all_ids[3102:3602])
            test_ids = set(all_ids[3602:])

            splits = {'train': [], 'val': [], 'test': []}
            for item in data:
                pid = item.get('id', item.get('pid', 0))
                entry = {
                    'file_path': item.get('file_path', item.get('img_path', '')),
                    'captions': item.get('captions', [item.get('caption', '')]),
                    'id': pid,
                }
                if pid in train_ids:
                    splits['train'].append(entry)
                elif pid in val_ids:
                    splits['val'].append(entry)
                else:
                    splits['test'].append(entry)

        os.makedirs(output_dir, exist_ok=True)
        for split_name, items in splits.items():
            out_path = os.path.join(output_dir, f'{split_name}.json')
            with open(out_path, 'w') as f:
                json.dump(items, f, indent=2)
            print(f"  {split_name}: {len(items)} items -> {out_path}")
        return True

    print("  Unknown format in ICFG-PEDES.json")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/mrsach_dang_quoc_minh_101/workspace/WS_DATASETS/REID/ICFG-PEDES')
    parser.add_argument('--output_dir', default='/mrsach_dang_quoc_minh_101/workspace/WS_DATASETS/REID/ICFG-PEDES/processed_data')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Try ICFG-PEDES.json first (most common)
    json_path = os.path.join(args.data_dir, 'ICFG-PEDES.json')
    if os.path.exists(json_path):
        print(f"Found ICFG-PEDES.json, converting...")
        success = convert_icfg_json(json_path, args.output_dir)
        if success:
            return

    # Try .pkl files
    pkl_dir = os.path.join(args.data_dir, 'processed_data')
    for pkl_name, json_name in [('train_save.pkl', 'train.json'), ('test_save.pkl', 'test.json')]:
        pkl_path = os.path.join(pkl_dir, pkl_name)
        if os.path.exists(pkl_path):
            output_path = os.path.join(args.output_dir, json_name)
            print(f"\nConverting {pkl_name}...")
            convert_split(pkl_path, output_path, args.data_dir)

    # Create val.json as copy of test if no separate val
    test_json = os.path.join(args.output_dir, 'test.json')
    val_json = os.path.join(args.output_dir, 'val.json')
    if os.path.exists(test_json) and not os.path.exists(val_json):
        import shutil
        shutil.copy(test_json, val_json)
        print(f"Copied test.json -> val.json (no separate val split)")


if __name__ == "__main__":
    main()
