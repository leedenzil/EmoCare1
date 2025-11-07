"""
Preprocessing utilities for EmoCare1 multimodal sentiment analysis.
Handles Reddit text, images, and videos.
"""

import re
import os
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL/Pillow not installed. Image processing will be limited.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not installed. Video processing will be limited.")


class RedditTextPreprocessor:
    """Preprocesses Reddit post text for sentiment analysis."""

    def __init__(self):
        # Common Reddit markdown patterns
        self.markdown_patterns = [
            (r'\*\*(.*?)\*\*', r'\1'),  # **bold** -> bold
            (r'\*(.*?)\*', r'\1'),      # *italic* -> italic
            (r'~~(.*?)~~', r'\1'),      # ~~strikethrough~~ -> strikethrough
            (r'`(.*?)`', r'\1'),        # `code` -> code
        ]

        # URL pattern
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

        # Reddit-specific patterns
        self.reddit_user_pattern = r'u/[A-Za-z0-9_-]+'
        self.reddit_sub_pattern = r'r/[A-Za-z0-9_]+'

    def preprocess(self, text: str, remove_urls: bool = True,
                   keep_special_tokens: bool = True) -> str:
        """
        Preprocess Reddit text for sentiment analysis.

        Args:
            text: Raw Reddit post text (title + body)
            remove_urls: Whether to replace URLs with [URL] token
            keep_special_tokens: Whether to keep Reddit-specific info like /s (sarcasm)

        Returns:
            Preprocessed text ready for model input
        """
        if not text or not isinstance(text, str):
            return ""

        # Combine title and text if both present
        text = str(text).strip()

        # Remove markdown formatting but keep content
        for pattern, replacement in self.markdown_patterns:
            text = re.sub(pattern, replacement, text)

        # Handle URLs
        if remove_urls:
            text = re.sub(self.url_pattern, '[URL]', text)

        # Replace Reddit mentions with generic tokens
        text = re.sub(self.reddit_user_pattern, '[USER]', text)
        text = re.sub(self.reddit_sub_pattern, '[SUBREDDIT]', text)

        # Keep important markers if requested
        if keep_special_tokens:
            # Keep /s for sarcasm detection (crucial for sentiment!)
            pass  # Don't remove /s

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def combine_title_and_text(self, title: str, text: str) -> str:
        """
        Combine Reddit post title and body text.
        Title is often more important for sentiment.
        """
        title = str(title).strip() if title else ""
        text = str(text).strip() if text else ""

        if title and text:
            # Use [SEP] token for BERT-style models
            return f"{title} [SEP] {text}"
        elif title:
            return title
        elif text:
            return text
        else:
            return ""

    def __call__(self, post_data: Dict[str, Any]) -> str:
        """
        Process a Reddit post dictionary.

        Args:
            post_data: Dictionary with 'title' and 'text' keys

        Returns:
            Preprocessed combined text
        """
        title = post_data.get('title', '')
        text = post_data.get('text', '')

        # Combine title and text
        combined = self.combine_title_and_text(title, text)

        # Preprocess
        processed = self.preprocess(combined)

        return processed


class SentimentLabelEncoder:
    """Encodes sentiment labels to integer indices."""

    # Based on your data analysis
    LABEL_TO_IDX = {
        'Neutral/Other': 0,
        'Joy': 1,
        'Anger': 2,
        'Surprise': 3,
        'Sadness': 4,
    }

    IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

    NUM_CLASSES = len(LABEL_TO_IDX)

    @classmethod
    def encode(cls, label: str) -> int:
        """Convert sentiment label to index."""
        return cls.LABEL_TO_IDX.get(label, 0)  # Default to Neutral

    @classmethod
    def decode(cls, idx: int) -> str:
        """Convert index to sentiment label."""
        return cls.IDX_TO_LABEL.get(idx, 'Neutral/Other')

    @classmethod
    def get_class_weights(cls, label_counts: Dict[str, int]) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced data.
        Uses inverse frequency weighting.

        Args:
            label_counts: Dictionary mapping labels to their counts

        Returns:
            Dictionary mapping class indices to weights
        """
        total = sum(label_counts.values())
        weights = {}

        for label, count in label_counts.items():
            idx = cls.encode(label)
            # Inverse frequency weight
            weights[idx] = total / (cls.NUM_CLASSES * count)

        return weights


# Example usage and testing
if __name__ == "__main__":
    # Test text preprocessor
    preprocessor = RedditTextPreprocessor()

    test_cases = [
        {
            'title': 'Why are the new modes called **Trio** when they\'re 4 teams?',
            'text': 'I mean trio showdown is trio because instead of solo/duo, trio has teams of 3.\nBut for these modes they already have teams of 3, isn\'t it funny to call these variants Trio /s',
        },
        {
            'title': 'This is dumb',
            'text': 'Check out this post https://reddit.com/r/BrawlStars/comments/123456',
        },
        {
            'title': 'u/someuser is wrong about r/BrawlStars balance',
            'text': 'They said `mortis` is OP but actually **he\'s not**',
        },
    ]

    print("Testing RedditTextPreprocessor:")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        result = preprocessor(test)
        print(f"\nTest {i}:")
        print(f"  Title: {test['title']}")
        print(f"  Text: {test['text']}")
        print(f"  Processed: {result}")

    print("\n" + "=" * 60)
    print("\nTesting SentimentLabelEncoder:")
    print("=" * 60)

    # Test label encoding
    labels = ['Joy', 'Anger', 'Sadness', 'Neutral/Other', 'Surprise']
    for label in labels:
        idx = SentimentLabelEncoder.encode(label)
        decoded = SentimentLabelEncoder.decode(idx)
        print(f"{label} -> {idx} -> {decoded}")

    # Test class weights (based on actual distribution from your data)
    label_counts = {
        'Neutral/Other': 475,
        'Joy': 429,
        'Anger': 341,
        'Surprise': 98,
        'Sadness': 81,
    }

    weights = SentimentLabelEncoder.get_class_weights(label_counts)
    print("\nClass Weights (for handling imbalance):")
    for idx, weight in sorted(weights.items()):
        label = SentimentLabelEncoder.decode(idx)
        print(f"  {label} (idx={idx}): {weight:.3f}")


class ImagePreprocessor:
    """
    Preprocesses images for ResNet-50 and other CNN models.
    Handles game screenshots, memes, tier lists, etc.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True):
        """
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize with ImageNet stats
        """
        if not HAS_PIL:
            raise ImportError("PIL/Pillow is required for image processing")

        self.target_size = target_size
        self.normalize = normalize

        # ImageNet normalization statistics (standard for pre-trained models)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def load_and_preprocess(self, image_path: str,
                            augment: bool = False) -> Optional[np.ndarray]:
        """
        Load and preprocess an image from disk.

        Args:
            image_path: Path to image file
            augment: Whether to apply data augmentation (for training)

        Returns:
            Preprocessed image as numpy array (H, W, C) or None if loading fails
        """
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return None

        try:
            # Load image
            img = Image.open(image_path).convert('RGB')

            # Apply augmentation if requested (for training)
            if augment:
                img = self._augment(img)

            # Resize
            img = img.resize(self.target_size, Image.BILINEAR)

            # Convert to numpy array [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Normalize with ImageNet stats
            if self.normalize:
                img_array = (img_array - self.mean) / self.std

            return img_array

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def _augment(self, img: Image.Image) -> Image.Image:
        """
        Apply data augmentation for training.
        Simple augmentations that preserve sentiment content.
        """
        import random

        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random brightness adjustment (±20%)
        if random.random() > 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)

        # Random contrast adjustment (±20%)
        if random.random() > 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)

        return img

    def get_zero_image(self) -> np.ndarray:
        """Returns a zero-filled image (for missing modality)."""
        zero_img = np.zeros((*self.target_size, 3), dtype=np.float32)
        if self.normalize:
            # Apply normalization (broadcasting will handle dimensions)
            return (zero_img - self.mean) / self.std
        else:
            return zero_img


class VideoPreprocessor:
    """
    Preprocesses videos by sampling frames.
    Uses frame sampling + aggregation strategy for efficiency.
    """

    def __init__(self, num_frames: int = 8,
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True):
        """
        Args:
            num_frames: Number of frames to sample from each video
            target_size: Target frame size (height, width)
            normalize: Whether to normalize frames with ImageNet stats
        """
        if not HAS_CV2:
            raise ImportError("OpenCV (cv2) is required for video processing")

        self.num_frames = num_frames
        self.target_size = target_size
        self.normalize = normalize

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def load_and_sample_frames(self, video_path: str) -> Optional[List[np.ndarray]]:
        """
        Load video and sample frames uniformly.

        Args:
            video_path: Path to video file

        Returns:
            List of preprocessed frames or None if loading fails
        """
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            return None

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error: Cannot open video {video_path}")
                return None

            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                print(f"Error: Video has 0 frames: {video_path}")
                cap.release()
                return None

            # Determine frame indices to sample (evenly spaced)
            if total_frames <= self.num_frames:
                # Sample all frames if video is short
                frame_indices = list(range(total_frames))
            else:
                # Sample evenly throughout the video
                frame_indices = np.linspace(0, total_frames - 1,
                                           self.num_frames, dtype=int)

            # Extract frames
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if ret:
                    # Preprocess frame
                    processed_frame = self._preprocess_frame(frame)
                    frames.append(processed_frame)

            cap.release()

            if len(frames) == 0:
                print(f"Warning: No frames extracted from {video_path}")
                return None

            return frames

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single video frame.

        Args:
            frame: OpenCV frame (BGR format)

        Returns:
            Preprocessed frame (RGB, normalized)
        """
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        frame = cv2.resize(frame, self.target_size)

        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        if self.normalize:
            frame = (frame - self.mean) / self.std

        return frame

    def aggregate_frames(self, frames: List[np.ndarray],
                        method: str = 'mean') -> np.ndarray:
        """
        Aggregate multiple frames into a single representation.

        Args:
            frames: List of preprocessed frames
            method: Aggregation method ('mean', 'max', or 'concat')

        Returns:
            Aggregated frame representation
        """
        if not frames:
            return self.get_zero_video()

        frames_array = np.stack(frames)  # Shape: (num_frames, H, W, C)

        if method == 'mean':
            return np.mean(frames_array, axis=0)
        elif method == 'max':
            return np.max(frames_array, axis=0)
        elif method == 'concat':
            # Return all frames (will be processed by model individually)
            return frames_array
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def get_zero_video(self) -> np.ndarray:
        """Returns a zero-filled video representation (for missing modality)."""
        zero_frame = np.zeros((*self.target_size, 3), dtype=np.float32)
        if self.normalize:
            # Apply normalization (broadcasting will handle dimensions)
            return (zero_frame - self.mean) / self.std
        else:
            return zero_frame
