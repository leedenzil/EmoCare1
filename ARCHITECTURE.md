# EmoCare1 Multimodal Sentiment Analysis Architecture

## Overview
Three-specialist multimodal sentiment analysis for Reddit Brawl Stars posts using state-of-the-art models.

## Data Summary
- **Total Samples**: 6,250 labeled Reddit posts
- **Train**: 4,999 samples (80%)
- **Validation**: 623 samples (10%)
- **Test**: 628 samples (10%)

**Modality Distribution**:
- Text + Image: 75.5% (4,717 samples)
- Text + Video: 19.8% (1,238 samples)
- Text only: 4.7% (295 samples)

**Sentiment Classes** (5 classes):
1. Neutral/Other: 31.6%
2. Joy: 29.1%
3. Anger: 26.6%
4. Surprise: 7.3%
5. Sadness: 5.4%

---

## Architecture Design

### Phase 1: Specialist Models (Independent Training)

#### 1. Text Specialist: DistilBERT
- **Model**: `distilbert-base-uncased` (66M parameters)
- **Input**: Preprocessed Reddit text (title + body, max 128 tokens)
- **Output**: 768-dim embedding + 5-class sentiment prediction
- **Training**:
  - Learning rate: 2e-5
  - Batch size: 16
  - Epochs: 3-4 (early stopping)
  - Class-weighted loss (Sadness: 3.5x, Surprise: 2.9x)
- **Why DistilBERT**: Fast, handles Reddit slang well, proven for sentiment

#### 2. Image Specialist: CLIP-ViT-B/32 ⭐
- **Model**: `openai/clip-vit-base-patch32` (151M parameters)
- **Input**: 224x224 RGB images (memes, screenshots, tier lists)
- **Output**: 512-dim embedding + 5-class sentiment prediction
- **Training**:
  - Learning rate: 1e-4
  - Batch size: 32
  - Epochs: 5-10
  - Data augmentation: horizontal flip, brightness, contrast
  - Class-weighted loss
- **Why CLIP over ResNet-50**:
  - ✅ Trained on 400M internet images (vs ImageNet's academic dataset)
  - ✅ Understands semantic relationships in images
  - ✅ Better for memes with text overlays
  - ✅ Image embeddings already aligned with text space
  - ✅ Expected +5-8% F1 improvement
  - ✅ Perfect for Reddit content (game screenshots, memes, tier lists)

#### 3. Video Specialist: CLIP-ViT-B/32 (Frame-based)
- **Model**: Same CLIP model as images
- **Input**: 8 frames sampled uniformly from video
- **Processing**: Each frame through CLIP → aggregate embeddings
- **Aggregation**: Mean pooling of 8 frame embeddings
- **Output**: 512-dim embedding + 5-class sentiment prediction
- **Training**:
  - Learning rate: 1e-4
  - Batch size: 16
  - Epochs: 5-10
  - Class-weighted loss
- **Why Frame-based CLIP**: Videos are short gameplay clips, temporal dynamics less important than visual content

---

### Phase 2: Fusion Model (Multimodal Integration)

#### Architecture: Cross-Modal Attention Fusion

```
Input:
  - Text embedding (768-dim from DistilBERT)
  - Image embedding (512-dim from CLIP)
  - Video embedding (512-dim from CLIP)

Fusion Model:
  1. Project to common dimension (512-dim)
     - text_proj: Linear(768 → 512)
     - image_proj: Identity (already 512)
     - video_proj: Identity (already 512)

  2. Layer Normalization for each modality

  3. Cross-Modal Multi-Head Attention
     - 8 attention heads
     - Allows text to attend to visual content
     - Allows visual modalities to attend to each other

  4. Modality Dropout (Training Only)
     - Randomly zero out embeddings (25% per modality)
     - Forces robustness to missing modalities

  5. Fusion + Classification
     - Concatenate attended representations: [512*3 = 1536-dim]
     - MLP: 1536 → 512 → 256 → 5 classes
     - Dropout: 0.3
     - ReLU activations

Output: 5-class sentiment prediction
```

**Training**:
- Learning rate: 1e-4
- Batch size: 32
- Epochs: 10-15
- Modality dropout: 0.25
- Class-weighted loss
- Early stopping on validation F1

**Key Innovation**: Modality dropout ensures the model works well even when images or videos are missing (text-only posts).

---

## Implementation Details

### Preprocessing Pipeline

**Text** (`RedditTextPreprocessor`):
- Remove markdown formatting (keep content)
- Replace URLs with `[URL]` token
- Replace usernames with `[USER]`
- Replace subreddits with `[SUBREDDIT]`
- **Keep `/s` markers** (sarcasm detection)
- Combine title + text with `[SEP]` token

**Images** (`ImagePreprocessor`):
- Resize to 224x224
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Augmentation (training): horizontal flip, brightness ±20%, contrast ±20%
- Missing images: zero tensor

**Videos** (`VideoPreprocessor`):
- Sample 8 frames uniformly across video duration
- Process each frame as image (224x224, normalized)
- Missing videos: zero tensor

### Handling Missing Modalities

**Strategy**: Zero Tensors + Modality Flags
- Text-only posts: `has_image=False, has_video=False`
- Images/videos replaced with normalized zero tensors
- Fusion model trained with modality dropout → learns to handle missing data
- At inference: zero tensors are ignored by attention mechanism

### Class Imbalance Mitigation

**Class Weights** (inverse frequency):
- Neutral/Other: 0.60x
- Joy: 0.66x
- Anger: 0.84x
- Surprise: 2.91x ⭐
- Sadness: 3.52x ⭐

Applied to CrossEntropyLoss during all training phases.

---

## Expected Performance

### Baseline (Text-only DistilBERT):
- Macro F1: ~72-75%
- Accuracy: ~78-80%

### With Multimodal Fusion (CLIP + DistilBERT):
- **Target Macro F1: 80-85%**
- **Target Accuracy: 82-88%**
- Per-class F1:
  - Joy, Anger, Neutral: 85-90%
  - Surprise: 75-82%
  - Sadness: 70-78%

### CLIP Advantage Over ResNet-50:
- Expected +5-8% absolute F1 improvement on visual classes
- Better meme understanding
- Better text-in-image handling (tier lists, annotated screenshots)

---

## Training Pipeline

### Step 1: Train Text Specialist (~1-2 hours)
```bash
python src/train_text_specialist.py \
  --train_data data/train_split.csv \
  --val_data data/val_split.csv \
  --output_dir models/text_specialist \
  --epochs 4 \
  --batch_size 16 \
  --learning_rate 2e-5
```

### Step 2: Train Image Specialist (~1-2 hours)
```bash
python src/train_image_specialist.py \
  --train_data data/train_split.csv \
  --val_data data/val_split.csv \
  --output_dir models/image_specialist \
  --model_name openai/clip-vit-base-patch32 \
  --epochs 8 \
  --batch_size 32 \
  --learning_rate 1e-4
```

### Step 3: Train Video Specialist (~1-2 hours)
```bash
python src/train_video_specialist.py \
  --train_data data/train_split.csv \
  --val_data data/val_split.csv \
  --output_dir models/video_specialist \
  --model_name openai/clip-vit-base-patch32 \
  --epochs 8 \
  --batch_size 16 \
  --learning_rate 1e-4
```

### Step 4: Generate Embeddings (~30 min)
```bash
python src/generate_embeddings.py \
  --data data/train_split.csv \
  --text_model models/text_specialist \
  --image_model models/image_specialist \
  --video_model models/video_specialist \
  --output data/train_embeddings.pt

python src/generate_embeddings.py \
  --data data/val_split.csv \
  --output data/val_embeddings.pt
```

### Step 5: Train Fusion Model (~2-3 hours)
```bash
python src/train_fusion.py \
  --train_embeddings data/train_embeddings.pt \
  --val_embeddings data/val_embeddings.pt \
  --output_dir models/fusion_model \
  --epochs 15 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --modality_dropout 0.25
```

### Step 6: Evaluation
```bash
python src/evaluate.py \
  --test_data data/test_split.csv \
  --fusion_model models/fusion_model \
  --output results/
```

---

## Evaluation Metrics

### Primary Metric: Macro F1-Score
- Treats all classes equally (critical for imbalanced data)
- Target: **≥ 80%**

### Secondary Metrics:
- Weighted F1-Score
- Per-class F1, Precision, Recall
- Confusion Matrix
- Accuracy (informative but biased by class imbalance)

### Ablation Study:
- Text-only performance
- Image-only performance
- Video-only performance
- Text + Image
- Text + Video
- All modalities
- **Goal**: Quantify each modality's contribution

---

## File Structure

```
EmoCare1/
├── data/
│   ├── labeled_data_1k.csv (6,250 samples)
│   ├── train_split.csv (4,999)
│   ├── val_split.csv (623)
│   ├── test_split.csv (628)
│   ├── train_embeddings.pt
│   ├── val_embeddings.pt
│   └── media/
│       ├── images/
│       └── videos/
├── src/
│   ├── preprocessing.py (text, image, video preprocessors)
│   ├── data_utils.py (stratified splitting)
│   ├── dataset.py (PyTorch Dataset)
│   ├── train_text_specialist.py
│   ├── train_image_specialist.py (CLIP-based)
│   ├── train_video_specialist.py (CLIP-based)
│   ├── generate_embeddings.py
│   ├── train_fusion.py
│   ├── evaluate.py
│   └── inference.py
├── models/
│   ├── text_specialist/
│   ├── image_specialist/
│   ├── video_specialist/
│   └── fusion_model/
├── results/
│   ├── confusion_matrix.png
│   ├── per_class_f1.csv
│   └── ablation_study.txt
└── ARCHITECTURE.md (this file)
```

---

## Why This Architecture?

### 1. Handles Missing Modalities
- 4.7% text-only, 75.5% text+image, 19.8% text+video
- Zero tensors + modality dropout = robust to missing data

### 2. CLIP for Visual Understanding
- Better than ResNet for internet/social media images
- Understands memes, screenshots, tier lists
- Image-text alignment helps fusion

### 3. Class Imbalance Handling
- Weighted loss ensures rare classes (Sadness, Surprise) aren't ignored
- Stratified splits maintain distribution

### 4. Two-Phase Training
- Specialists learn modality-specific patterns
- Fusion learns optimal combination
- Prevents "modality collapse" (model ignoring weaker modalities)

### 5. Proven by Research
- Cross-modal attention: state-of-the-art for multimodal fusion
- Modality dropout: proven robustness technique
- CLIP: best vision encoder for internet content (2024 research)

---

## Next Steps

1. ✅ Data preprocessing utilities
2. ✅ Data splitting (stratified)
3. ✅ PyTorch Dataset class
4. ⏳ Train text specialist (DistilBERT)
5. ⏳ Train image specialist (CLIP)
6. ⏳ Train video specialist (CLIP)
7. ⏳ Generate embeddings
8. ⏳ Train fusion model
9. ⏳ Evaluate and ablation study
10. ⏳ Inference pipeline

---

**Document Version**: 1.0
**Last Updated**: November 7, 2025
**Model Choice**: CLIP-ViT-B/32 for visual modalities (chosen over ResNet-50)
