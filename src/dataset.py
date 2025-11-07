"""
PyTorch Dataset for multimodal sentiment analysis.
Handles text, images, and videos with missing modality support.
"""

import os
import csv
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. Dataset class will be limited.")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: Transformers not installed.")

from preprocessing import (
    RedditTextPreprocessor,
    ImagePreprocessor,
    VideoPreprocessor,
    SentimentLabelEncoder
)


class MultimodalSentimentDataset(Dataset):
    """
    PyTorch Dataset for multimodal Reddit posts.
    Handles text-only, text+image, and text+video posts.
    """

    def __init__(self,
                 csv_path: str,
                 root_dir: str = '.',
                 tokenizer_name: str = 'distilbert-base-uncased',
                 max_length: int = 128,
                 image_size: Tuple[int, int] = (224, 224),
                 num_video_frames: int = 8,
                 augment_images: bool = False,
                 video_aggregation: str = 'concat'):
        """
        Args:
            csv_path: Path to CSV file with data
            root_dir: Root directory for media files
            tokenizer_name: HuggingFace tokenizer to use
            max_length: Maximum sequence length for text
            image_size: Target image size (H, W)
            num_video_frames: Number of frames to sample from videos
            augment_images: Whether to apply data augmentation (for training)
            video_aggregation: How to aggregate video frames ('mean', 'max', 'concat')
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Dataset class")

        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers is required for text tokenization")

        self.root_dir = Path(root_dir)
        self.augment_images = augment_images
        self.video_aggregation = video_aggregation

        # Load data from CSV
        self.data = self._load_csv(csv_path)

        # Initialize preprocessors
        self.text_preprocessor = RedditTextPreprocessor()
        self.image_preprocessor = ImagePreprocessor(target_size=image_size)
        self.video_preprocessor = VideoPreprocessor(
            num_frames=num_video_frames,
            target_size=image_size
        )

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        print(f"Loaded {len(self.data)} samples from {csv_path}")

    def _load_csv(self, csv_path: str):
        """Load data from CSV file."""
        data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - input_ids: Tokenized text (Tensor)
                - attention_mask: Attention mask for text (Tensor)
                - image: Image tensor or zero tensor if missing (Tensor)
                - video: Video tensor or zero tensor if missing (Tensor)
                - has_image: Boolean indicating if image is present
                - has_video: Boolean indicating if video is present
                - label: Sentiment label index (int)
                - post_id: Reddit post ID (str)
        """
        item = self.data[idx]

        # Process text
        text = self.text_preprocessor(item)
        text_encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Process image (if present)
        has_image = False
        image_tensor = None
        local_media = item.get('local_media_path', '')

        if local_media and local_media.endswith(('.jpeg', '.jpg', '.png')):
            image_path = self.root_dir / local_media
            image_array = self.image_preprocessor.load_and_preprocess(
                str(image_path),
                augment=self.augment_images
            )

            if image_array is not None:
                has_image = True
                # Convert to torch tensor (H, W, C) -> (C, H, W)
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
            else:
                # Use zero image if loading failed
                zero_image = self.image_preprocessor.get_zero_image()
                image_tensor = torch.from_numpy(zero_image).permute(2, 0, 1).float()
        else:
            # No image - use zero tensor
            zero_image = self.image_preprocessor.get_zero_image()
            image_tensor = torch.from_numpy(zero_image).permute(2, 0, 1).float()

        # Process video (if present)
        has_video = False
        video_tensor = None

        if local_media and local_media.endswith('.mp4'):
            video_path = self.root_dir / local_media
            frames = self.video_preprocessor.load_and_sample_frames(str(video_path))

            if frames is not None and len(frames) > 0:
                has_video = True

                # Aggregate or stack frames
                aggregated = self.video_preprocessor.aggregate_frames(
                    frames,
                    method=self.video_aggregation
                )

                if self.video_aggregation == 'concat':
                    # Shape: (num_frames, H, W, C) -> (num_frames, C, H, W)
                    video_tensor = torch.from_numpy(aggregated)
                    video_tensor = video_tensor.permute(0, 3, 1, 2).float()
                else:
                    # Shape: (H, W, C) -> (C, H, W)
                    video_tensor = torch.from_numpy(aggregated).permute(2, 0, 1).float()
            else:
                # Use zero video
                zero_video = self.video_preprocessor.get_zero_video()
                video_tensor = torch.from_numpy(zero_video).permute(2, 0, 1).float()
        else:
            # No video - use zero tensor
            zero_video = self.video_preprocessor.get_zero_video()
            video_tensor = torch.from_numpy(zero_video).permute(2, 0, 1).float()

        # Get sentiment label
        sentiment = item.get('post_sentiment', 'Neutral/Other')
        label = SentimentLabelEncoder.encode(sentiment)

        return {
            'input_ids': text_encoded['input_ids'].squeeze(0),
            'attention_mask': text_encoded['attention_mask'].squeeze(0),
            'image': image_tensor,
            'video': video_tensor,
            'has_image': has_image,
            'has_video': has_video,
            'label': label,
            'post_id': item.get('id', ''),
            'sentiment_name': sentiment,
        }


def get_class_weights(dataset: MultimodalSentimentDataset) -> torch.Tensor:
    """
    Calculate class weights for the dataset to handle class imbalance.

    Args:
        dataset: MultimodalSentimentDataset instance

    Returns:
        Tensor of class weights
    """
    from collections import Counter

    # Count labels
    labels = [item['post_sentiment'] for item in dataset.data]
    label_counts = Counter(labels)

    # Convert to class weights
    num_classes = SentimentLabelEncoder.NUM_CLASSES
    total = len(labels)
    weights = torch.zeros(num_classes)

    for label_name, count in label_counts.items():
        idx = SentimentLabelEncoder.encode(label_name)
        # Inverse frequency weighting
        weights[idx] = total / (num_classes * count)

    return weights


if __name__ == "__main__":
    # Test the dataset
    print("Testing MultimodalSentimentDataset...")
    print("="*70)

    try:
        # Create dataset
        dataset = MultimodalSentimentDataset(
            csv_path='data/train_split.csv',
            root_dir='.',
            augment_images=False
        )

        print(f"\nDataset size: {len(dataset)}")

        # Get a sample
        print("\nLoading sample 0...")
        sample = dataset[0]

        print("\nSample structure:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor of shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  {key}: {value}")

        # Calculate class weights
        print("\nCalculating class weights...")
        weights = get_class_weights(dataset)
        print("\nClass weights:")
        for idx in range(len(weights)):
            label_name = SentimentLabelEncoder.decode(idx)
            print(f"  {label_name} (idx={idx}): {weights[idx]:.3f}")

        print("\n✅ Dataset test successful!")

    except Exception as e:
        print(f"\n❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
