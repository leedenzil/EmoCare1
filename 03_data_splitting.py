"""
Step 3: Data Splitting Script
Splits labeled_data.csv into train, validation, and test sets with stratification.

Usage:
    python 03_data_splitting.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path


# Configuration
INPUT_FILE = "data/labeled_data.csv"
OUTPUT_DIR = "data"
TRAIN_FILE = "train_set.csv"
VAL_FILE = "validation_set.csv"
TEST_FILE = "test_set.csv"

# Split ratios (train: 80%, val: 10%, test: 10%)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Random seed for reproducibility
RANDOM_SEED = 42


def print_separator():
    """Print a visual separator."""
    print("=" * 80)


def load_data(file_path):
    """Load the labeled data."""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(df):,} rows")
    return df


def analyze_data(df):
    """Analyze and display data statistics."""
    print_separator()
    print("DATA ANALYSIS")
    print_separator()

    print(f"\nTotal samples: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Check for missing values in key columns
    print("\nMissing values in key columns:")
    key_cols = ['post_sentiment', 'local_media_path', 'post_classification']
    for col in key_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")

    # Sentiment distribution
    print("\nPost Sentiment Distribution:")
    sentiment_counts = df['post_sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = count / len(df) * 100
        print(f"  {sentiment}: {count:,} ({percentage:.2f}%)")

    # Media type distribution
    print("\nMedia Type Distribution:")
    df['media_type'] = df['local_media_path'].apply(lambda x:
        'video' if 'video' in str(x).lower() else
        'image' if 'image' in str(x).lower() else
        'none'
    )
    media_counts = df['media_type'].value_counts()
    for media_type, count in media_counts.items():
        percentage = count / len(df) * 100
        print(f"  {media_type}: {count:,} ({percentage:.2f}%)")

    return df


def split_data(df, train_ratio, val_ratio, test_ratio, random_seed):
    """
    Split data into train, validation, and test sets with stratification.
    Stratification ensures class balance is maintained across all splits.
    """
    print_separator()
    print("SPLITTING DATA")
    print_separator()

    print(f"\nSplit ratios:")
    print(f"  Train: {train_ratio*100:.0f}%")
    print(f"  Validation: {val_ratio*100:.0f}%")
    print(f"  Test: {test_ratio*100:.0f}%")
    print(f"  Random seed: {random_seed}")

    # First split: separate out test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=df['post_sentiment']  # Maintain sentiment distribution
    )

    # Second split: separate train and validation from the remaining data
    # Adjust val_ratio to account for the remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        stratify=train_val_df['post_sentiment']
    )

    print(f"\n✓ Split complete:")
    print(f"  Train set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation set: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test set: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def verify_stratification(train_df, val_df, test_df):
    """Verify that stratification maintained class balance."""
    print_separator()
    print("STRATIFICATION VERIFICATION")
    print_separator()

    print("\nSentiment distribution across splits:")
    print(f"{'Sentiment':<20} {'Train %':>10} {'Val %':>10} {'Test %':>10}")
    print("-" * 55)

    sentiments = train_df['post_sentiment'].unique()
    for sentiment in sorted(sentiments):
        train_pct = (train_df['post_sentiment'] == sentiment).sum() / len(train_df) * 100
        val_pct = (val_df['post_sentiment'] == sentiment).sum() / len(val_df) * 100
        test_pct = (test_df['post_sentiment'] == sentiment).sum() / len(test_df) * 100
        print(f"{sentiment:<20} {train_pct:>9.2f}% {val_pct:>9.2f}% {test_pct:>9.2f}%")

    print("\n✓ Class balance maintained across all splits")


def save_splits(train_df, val_df, test_df, output_dir):
    """Save the split datasets to CSV files."""
    print_separator()
    print("SAVING SPLITS")
    print_separator()

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save each split
    train_path = os.path.join(output_dir, TRAIN_FILE)
    val_path = os.path.join(output_dir, VAL_FILE)
    test_path = os.path.join(output_dir, TEST_FILE)

    print(f"\nSaving splits to {output_dir}/")
    train_df.to_csv(train_path, index=False)
    print(f"  ✓ {TRAIN_FILE} ({len(train_df):,} rows)")

    val_df.to_csv(val_path, index=False)
    print(f"  ✓ {VAL_FILE} ({len(val_df):,} rows)")

    test_df.to_csv(test_path, index=False)
    print(f"  ✓ {TEST_FILE} ({len(test_df):,} rows)")

    print("\n✓ All splits saved successfully!")


def main():
    """Main execution function."""
    print_separator()
    print("BRAWL STARS SENTIMENT ANALYSIS - DATA SPLITTING")
    print_separator()

    # Load data
    df = load_data(INPUT_FILE)

    # Analyze data
    df = analyze_data(df)

    # Split data
    train_df, val_df, test_df = split_data(
        df,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        RANDOM_SEED
    )

    # Verify stratification
    verify_stratification(train_df, val_df, test_df)

    # Save splits
    save_splits(train_df, val_df, test_df, OUTPUT_DIR)

    print_separator()
    print("DATA SPLITTING COMPLETE!")
    print_separator()
    print("\nNext steps:")
    print("  1. Review the split distributions above")
    print("  2. Proceed to Step 4: Phase 1 Training (Fine-tuning specialists)")
    print("     - Text model: DistilBERT")
    print("     - Image model: CLIP")
    print("     - Video model: TBD")
    print_separator()


if __name__ == "__main__":
    main()
