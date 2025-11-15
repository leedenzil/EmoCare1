"""
Step 4B: Image Model Training v2 (Improved) - Python Script Version

This is an improved version of the image model training with better handling of class imbalance.

Improvements in v2:
- Class weights (~5x for Sadness/Surprise) - Fixes 0% F1!
- Data augmentation (rotation, flip, color jitter)
- Label smoothing (0.1)
- Gradient clipping (1.0)
- Early stopping (patience=4)
- ReduceLROnPlateau scheduler
- Windows-optimized data loading

Expected improvements:
- Sadness: 0% → 20-30% F1
- Surprise: 0% → 18-28% F1
- Overall: 45.7% → 52-58% accuracy

Usage:
    python 04_train_image_model_v2.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
import json
import os
from pathlib import Path
import warnings
import random
from tqdm import tqdm
import argparse

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
TRAIN_DATA = "data/train_set.csv"
VAL_DATA = "data/validation_set.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results/image_model_v2"

# Model configuration
MODEL_NAME = 'openai/clip-vit-base-patch32'
BATCH_SIZE = 32  # Increased for better GPU utilization (safe for RTX 2080 Super 8GB)
EPOCHS = 12
LEARNING_RATE = 1e-5

# Training improvements
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 4
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5

# Data augmentation
AUG_ROTATION_DEGREES = 15
AUG_FLIP_PROB = 0.5
AUG_COLOR_JITTER = 0.2

# Sentiment labels
LABELS = ['Anger', 'Joy', 'Neutral/Other', 'Sadness', 'Surprise']
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# Random seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def apply_augmentation(image):
    """Apply random augmentations to increase dataset diversity."""
    # Random horizontal flip
    if random.random() < AUG_FLIP_PROB:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Random rotation
    if random.random() < 0.5:
        angle = random.uniform(-AUG_ROTATION_DEGREES, AUG_ROTATION_DEGREES)
        image = image.rotate(angle, fillcolor=(255, 255, 255))

    # Random brightness
    if random.random() < 0.5:
        factor = random.uniform(1 - AUG_COLOR_JITTER, 1 + AUG_COLOR_JITTER)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)

    # Random contrast
    if random.random() < 0.5:
        factor = random.uniform(1 - AUG_COLOR_JITTER, 1 + AUG_COLOR_JITTER)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(factor)

    # Random saturation
    if random.random() < 0.5:
        factor = random.uniform(1 - AUG_COLOR_JITTER, 1 + AUG_COLOR_JITTER)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(factor)

    return image

# ============================================================================
# DATASET
# ============================================================================

class BrawlStarsImageDataset(Dataset):
    def __init__(self, dataframe, processor, augment=False):
        self.data = dataframe.reset_index(drop=True)
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Normalize path (handle Windows backslashes)
        image_path = str(row['local_media_path']).replace('\\', '/')

        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Apply augmentation during training
            if self.augment:
                image = apply_augmentation(image)

            # Process with CLIP
            inputs = self.processor(images=image, return_tensors="pt")

            # Get label
            label = LABEL_TO_ID[row['post_sentiment']]

            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy sample
            dummy_image = Image.new('RGB', (224, 224), color='black')
            inputs = self.processor(images=dummy_image, return_tensors="pt")
            label = LABEL_TO_ID[row['post_sentiment']]
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }

# ============================================================================
# MODEL
# ============================================================================

class ImageSentimentClassifier(nn.Module):
    def __init__(self, n_classes=5):
        super(ImageSentimentClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained(MODEL_NAME)
        self.vision_embed_dim = self.clip.vision_model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.vision_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )

    def forward(self, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.pooler_output
        logits = self.classifier(image_embeds)
        return logits

    def get_embedding(self, pixel_values):
        """Extract embedding without classification head (for Phase 2)"""
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.pooler_output
        return image_embeds

# ============================================================================
# LOSS FUNCTION
# ============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for better calibration"""
    def __init__(self, weight=None, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_pred = torch.log_softmax(pred, dim=-1)

        # Apply label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Compute loss per sample
        loss = torch.sum(-true_dist * log_pred, dim=-1)

        # Apply class weights to each sample based on its target class
        if self.weight is not None:
            loss = loss * self.weight[target]

        return torch.mean(loss)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(pixel_values)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)

        optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)
        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_predictions/total_samples:.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            outputs = model(pixel_values)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1, all_preds, all_labels

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(history, best_epoch, results_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
    axes[0].axvline(best_epoch - 1, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss (v2)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='s')
    axes[1].plot(history['val_f1'], label='Validation F1', marker='^')
    axes[1].axvline(best_epoch - 1, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Training and Validation Metrics (v2)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate plot
    axes[2].plot(history['learning_rates'], marker='o', color='purple')
    axes[2].axvline(best_epoch - 1, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule (v2)')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(val_labels, val_preds, results_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(val_labels, val_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=LABELS,
        yticklabels=LABELS,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Image Model v2 (Validation Set)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_f1_comparison(metrics_v2, results_dir):
    """Plot F1 scores with v1 comparison if available"""
    f1_scores_v2 = [metrics_v2['per_class'][label]['f1_score'] for label in LABELS]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Try to load v1 metrics
    v1_metrics_path = "results/image_model/evaluation_metrics.json"
    if os.path.exists(v1_metrics_path):
        with open(v1_metrics_path, 'r') as f:
            metrics_v1 = json.load(f)

        f1_scores_v1 = [metrics_v1['per_class'][label]['f1_score'] for label in LABELS]

        x = np.arange(len(LABELS))
        width = 0.35

        ax.bar(x - width/2, f1_scores_v1, width, label='v1 (Original)', color='#95a5a6', alpha=0.7)
        ax.bar(x + width/2, f1_scores_v2, width, label='v2 (Improved)',
               color=['#e74c3c', '#f39c12', '#95a5a6', '#3498db', '#9b59b6'])

        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Per-Class F1 Scores - Image Model v1 vs v2 Comparison\n(Watch Sadness & Surprise!)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, rotation=45, ha='right')
        ax.legend()
    else:
        ax.bar(LABELS, f1_scores_v2, color=['#e74c3c', '#f39c12', '#95a5a6', '#3498db', '#9b59b6'])
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Per-Class F1 Scores - Image Model v2', fontsize=14, fontweight='bold')
        ax.set_xticklabels(LABELS, rotation=45, ha='right')

    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{results_dir}/f1_scores_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Create directories
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("IMAGE MODEL v2 TRAINING - Python Script Version")
    print("=" * 80)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Max Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")

    print(f"\n🆕 v2 Improvements:")
    print(f"  - Class weights (fixes 0% F1 on Sadness/Surprise)")
    print(f"  - Data augmentation (rotation, flip, color jitter)")
    print(f"  - Label smoothing ({LABEL_SMOOTHING})")
    print(f"  - Gradient clipping ({GRADIENT_CLIP_NORM})")
    print(f"  - Early stopping (patience={EARLY_STOP_PATIENCE})")
    print(f"  - ReduceLROnPlateau scheduler")

    # Load data
    print(f"\nLoading datasets...")
    train_df = pd.read_csv(TRAIN_DATA)
    val_df = pd.read_csv(VAL_DATA)

    # Filter for images only
    train_df = train_df[train_df['media_type'] == 'image'].reset_index(drop=True)
    val_df = val_df[val_df['media_type'] == 'image'].reset_index(drop=True)

    print(f"  Train set (images): {len(train_df):,} samples")
    print(f"  Validation set (images): {len(val_df):,} samples")

    # Compute class weights
    print(f"\nComputing class weights...")
    train_labels_numeric = train_df['post_sentiment'].map(LABEL_TO_ID).values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(LABELS)),
        y=train_labels_numeric
    )
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    train_counts = train_df['post_sentiment'].value_counts()
    print("\nClass weights:")
    for label, weight in zip(LABELS, class_weights):
        count = train_counts.get(label, 0)
        print(f"  {label:15s}: {weight:.4f} (n={count:,})")

    # Initialize processor
    print(f"\nLoading CLIP processor...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # Create datasets
    train_dataset = BrawlStarsImageDataset(train_df, processor, augment=True)
    val_dataset = BrawlStarsImageDataset(val_df, processor, augment=False)

    # Create dataloaders
    # Python scripts handle multiprocessing better than Jupyter notebooks on Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,          # 4 parallel workers (works in .py, not in notebooks!)
        pin_memory=True,
        prefetch_factor=2,      # Preload 2 batches per worker
        persistent_workers=True # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,          # 2 workers for validation
        pin_memory=True,
        persistent_workers=True
    )

    print(f"✓ Created {len(train_loader)} train batches and {len(val_loader)} validation batches")
    print(f"🚀 Optimized for .py execution:")
    print(f"   - batch_size={BATCH_SIZE}")
    print(f"   - num_workers=4 (parallel data loading)")
    print(f"   - pin_memory=True + prefetch_factor=2")
    print(f"   Expected GPU usage: 70-90%!")

    # Initialize model
    print(f"\nInitializing model...")
    model = ImageSentimentClassifier(n_classes=len(LABELS))
    model = model.to(device)
    print(f"✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss, optimizer, scheduler
    criterion = LabelSmoothingCrossEntropy(weight=class_weights_tensor, smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rates': []
    }

    best_val_f1 = 0
    best_epoch = 0
    epochs_without_improvement = 0

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_f1, _, _ = eval_model(model, val_loader, criterion, device)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['learning_rates'].append(current_lr)

        # Print metrics
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")

        # Update scheduler
        scheduler.step(val_f1)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{MODEL_DIR}/image_specialist_v2_best.pth")
            print(f"  ✓ New best model saved! (F1: {val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early stopping! No improvement for {EARLY_STOP_PATIENCE} epochs.")
            print(f"   Best F1: {best_val_f1:.4f} at epoch {best_epoch}")
            break

    # Save final model
    torch.save(model.state_dict(), f"{MODEL_DIR}/image_specialist_v2.pth")

    # Save training history
    with open(f"{RESULTS_DIR}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Load best model for evaluation
    print(f"\nLoading best model (Epoch {best_epoch}, F1: {best_val_f1:.4f})...")
    model.load_state_dict(torch.load(f"{MODEL_DIR}/image_specialist_v2_best.pth"))

    # Final evaluation
    print(f"\nEvaluating on validation set...")
    val_loss, val_acc, val_f1, val_preds, val_labels = eval_model(model, val_loader, criterion, device)

    # Generate metrics
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        val_labels, val_preds, labels=range(len(LABELS))
    )

    metrics_v2 = {
        "version": "v2",
        "improvements": [
            "Class weights (Sadness/Surprise ~5x)",
            "Data augmentation (rotation, flip, color jitter)",
            "Label smoothing (0.1)",
            "Gradient clipping (1.0)",
            "Early stopping (patience=4)",
            "ReduceLROnPlateau scheduler"
        ],
        "overall": {
            "accuracy": float(val_acc),
            "weighted_f1": float(val_f1),
            "loss": float(val_loss),
            "best_epoch": best_epoch,
            "total_epochs": epoch + 1
        },
        "per_class": {}
    }

    for idx, label in enumerate(LABELS):
        metrics_v2["per_class"][label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1_score": float(f1_per_class[idx]),
            "support": int(support[idx])
        }

    # Save metrics
    with open(f"{RESULTS_DIR}/evaluation_metrics.json", 'w') as f:
        json.dump(metrics_v2, f, indent=2)

    # Save classification report
    report = classification_report(val_labels, val_preds, target_names=LABELS, digits=4)
    with open(f"{RESULTS_DIR}/evaluation_report.txt", 'w') as f:
        f.write("IMAGE MODEL (CLIP) v2 EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("Improvements in v2:\n")
        f.write("  - Class weights for severe imbalance (Sadness/Surprise ~5x)\n")
        f.write("  - Data augmentation (rotation, flip, color jitter)\n")
        f.write("  - Label smoothing (0.1)\n")
        f.write("  - Gradient clipping (1.0)\n")
        f.write("  - Early stopping (patience=4)\n")
        f.write("  - ReduceLROnPlateau scheduler\n\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Validation Weighted F1: {val_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Generate plots
    print(f"\nGenerating visualizations...")
    plot_training_curves(history, best_epoch, RESULTS_DIR)
    plot_confusion_matrix(val_labels, val_preds, RESULTS_DIR)
    plot_f1_comparison(metrics_v2, RESULTS_DIR)

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

    print("\nFinal Performance (v2):")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Weighted F1: {val_f1:.4f}")
    print(f"  Best Epoch: {best_epoch}/{epoch + 1}")

    print("\nPer-Class F1 Scores (v2):")
    for label in LABELS:
        f1 = metrics_v2['per_class'][label]['f1_score']
        support_count = metrics_v2['per_class'][label]['support']
        print(f"  {label:15s}: {f1:.4f} (n={support_count})")

    print(f"\nFiles Generated:")
    print(f"  1. {MODEL_DIR}/image_specialist_v2_best.pth")
    print(f"  2. {RESULTS_DIR}/evaluation_metrics.json")
    print(f"  3. {RESULTS_DIR}/training_curves.png")
    print(f"  4. {RESULTS_DIR}/confusion_matrix.png")
    print(f"  5. {RESULTS_DIR}/f1_scores_comparison.png")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
