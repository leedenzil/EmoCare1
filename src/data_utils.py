"""
Data utilities for EmoCare1 multimodal sentiment analysis.
Handles data splitting and dataset creation.
"""

import csv
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def stratified_split(data: List[Dict],
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     label_key: str = 'post_sentiment',
                     random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Perform stratified split to maintain class distribution across splits.

    Args:
        data: List of data dictionaries
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        label_key: Key in dict containing the label
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    random.seed(random_seed)

    # Group data by label
    label_groups = defaultdict(list)
    for item in data:
        label = item[label_key]
        label_groups[label].append(item)

    train_data, val_data, test_data = [], [], []

    # Split each label group proportionally
    for label, items in label_groups.items():
        # Shuffle items for this label
        random.shuffle(items)

        n = len(items)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train_data.extend(items[:train_size])
        val_data.extend(items[train_size:train_size + val_size])
        test_data.extend(items[train_size + val_size:])

    # Shuffle the final splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data


def load_csv_data(csv_path: str) -> List[Dict]:
    """Load data from CSV file into list of dictionaries."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
    return data


def save_csv_data(data: List[Dict], csv_path: str):
    """Save list of dictionaries to CSV file."""
    if not data:
        print(f"Warning: No data to save to {csv_path}")
        return

    fieldnames = list(data[0].keys())

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Saved {len(data)} samples to {csv_path}")


def print_split_statistics(train_data: List[Dict],
                           val_data: List[Dict],
                           test_data: List[Dict],
                           label_key: str = 'post_sentiment'):
    """Print statistics about the data splits."""

    def count_labels(data):
        counts = defaultdict(int)
        for item in data:
            counts[item[label_key]] += 1
        return counts

    def count_modalities(data):
        counts = {'text_only': 0, 'text_image': 0, 'text_video': 0}
        for item in data:
            post_hint = item.get('post_hint', '')
            local_media = item.get('local_media_path', '')

            if post_hint == 'text_only' or not local_media:
                counts['text_only'] += 1
            elif 'video' in post_hint or local_media.endswith('.mp4'):
                counts['text_video'] += 1
            elif 'image' in post_hint or local_media.endswith(('.jpeg', '.jpg', '.png')):
                counts['text_image'] += 1
        return counts

    print("\n" + "="*70)
    print("DATA SPLIT STATISTICS")
    print("="*70)

    total = len(train_data) + len(val_data) + len(test_data)
    print(f"\nTotal samples: {total}")
    print(f"  Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"  Val:   {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"  Test:  {len(test_data)} ({len(test_data)/total*100:.1f}%)")

    print(f"\n{'Set':<10} {'Sentiment':<15} {'Count':<8} {'Percentage'}")
    print("-"*70)

    for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        label_counts = count_labels(split_data)
        total_split = len(split_data)

        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{split_name:<10} {label:<15} {count:<8} {count/total_split*100:>6.1f}%")
        print()

    print("\nModality Distribution:")
    print("-"*70)
    for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        mod_counts = count_modalities(split_data)
        total_split = len(split_data)

        print(f"{split_name}:")
        for modality, count in mod_counts.items():
            print(f"  {modality}: {count} ({count/total_split*100:.1f}%)")

    print("="*70)


if __name__ == "__main__":
    # Split the labeled_data_1k.csv into train/val/test
    print("Loading labeled_data_1k.csv...")
    data = load_csv_data('data/labeled_data_1k.csv')
    print(f"Loaded {len(data)} samples")

    print("\nPerforming stratified split (80/10/10)...")
    train_data, val_data, test_data = stratified_split(
        data,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        label_key='post_sentiment',
        random_seed=42
    )

    # Save splits
    print("\nSaving splits...")
    save_csv_data(train_data, 'data/train_split.csv')
    save_csv_data(val_data, 'data/val_split.csv')
    save_csv_data(test_data, 'data/test_split.csv')

    # Print statistics
    print_split_statistics(train_data, val_data, test_data)

    print("\n✅ Data splitting complete!")
