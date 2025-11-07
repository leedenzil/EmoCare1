"""
Training script for Image Specialist Model (CLIP-ViT-B/32).
Fine-tunes CLIP vision encoder for sentiment classification on images.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import CLIPModel, CLIPProcessor, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, classification_report

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import MultimodalSentimentDataset, get_class_weights
from preprocessing import SentimentLabelEncoder
from results_logger import ResultsLogger


class ImageSpecialistModel(nn.Module):
    """Image specialist model using CLIP vision encoder."""

    def __init__(self, model_name='openai/clip-vit-base-patch32', num_classes=5, dropout=0.3):
        super().__init__()
        # Load CLIP model
        self.clip = CLIPModel.from_pretrained(model_name)

        # We only need the vision encoder
        self.vision_model = self.clip.vision_model

        # Get embedding dimension (512 for CLIP-ViT-B/32)
        self.embed_dim = self.clip.config.projection_dim

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (batch_size, 3, 224, 224)

        Returns:
            logits: (batch_size, num_classes)
            embeddings: (batch_size, embed_dim) - image embeddings
        """
        # Get vision embeddings from CLIP
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        # Get pooled output (CLS token)
        pooled_output = vision_outputs.pooler_output  # (batch_size, hidden_size)

        # Project to embedding space
        embeddings = self.clip.visual_projection(pooled_output)  # (batch_size, embed_dim)

        # Classify
        pooled = self.dropout(embeddings)
        logits = self.classifier(pooled)

        return logits, embeddings


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Only use images (ignore text and video)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        has_image = batch['has_image']

        # Skip samples without images
        valid_indices = [i for i, has_img in enumerate(has_image) if has_img]
        if len(valid_indices) == 0:
            continue

        images = images[valid_indices]
        labels = labels[valid_indices]

        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            has_image = batch['has_image']

            # Skip samples without images
            valid_indices = [i for i, has_img in enumerate(has_image) if has_img]
            if len(valid_indices) == 0:
                continue

            images = images[valid_indices]
            labels = labels[valid_indices]

            logits, _ = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, f1, all_preds, all_labels


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_dataset = MultimodalSentimentDataset(
        csv_path=args.train_data,
        root_dir=args.root_dir,
        tokenizer_name='distilbert-base-uncased',  # Not used for images, but required
        max_length=128,
        augment_images=True,  # Enable augmentation for training
        normalization_type='clip'  # Use CLIP normalization for CLIP models
    )

    val_dataset = MultimodalSentimentDataset(
        csv_path=args.val_data,
        root_dir=args.root_dir,
        tokenizer_name='distilbert-base-uncased',
        max_length=128,
        augment_images=False,
        normalization_type='clip'  # Use CLIP normalization for CLIP models
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Get class weights
    class_weights = get_class_weights(train_dataset)
    class_weights = class_weights.to(device)
    print(f"\nClass weights: {class_weights}")

    # Create model
    print(f"\nInitializing model: {args.model_name}")
    model = ImageSpecialistModel(
        model_name=args.model_name,
        num_classes=SentimentLabelEncoder.NUM_CLASSES,
        dropout=args.dropout
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Initialize results logger
    logger = ResultsLogger(args.output_dir, 'image_specialist')
    logger.log_config({
        'model_name': args.model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'dropout': args.dropout,
        'patience': args.patience,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'class_weights': class_weights.cpu().tolist()
    })

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")

        # Evaluate
        val_loss, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        # Log epoch results
        logger.log_epoch(epoch + 1, train_loss, train_f1, val_loss, val_f1)

        # Print and log classification report
        if epoch % 2 == 0:  # Every 2 epochs
            print("\nClassification Report:")
            target_names = [SentimentLabelEncoder.decode(i)
                          for i in range(SentimentLabelEncoder.NUM_CLASSES)]
            report_str = classification_report(val_labels, val_preds,
                                       target_names=target_names,
                                       zero_division=0)
            print(report_str)

            # Log as dictionary for JSON
            report_dict = classification_report(val_labels, val_preds,
                                       target_names=target_names,
                                       zero_division=0,
                                       output_dict=True)
            logger.log_classification_report(epoch + 1, report_dict)

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0

            # Log best metrics
            logger.log_best_metrics(epoch + 1, val_f1, val_loss)

            # Save model
            save_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss,
            }, save_path)
            print(f"✅ Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Log final metrics and save results
    logger.log_final_metrics(
        best_val_f1=best_f1,
        total_epochs=epoch + 1,
        early_stopped=(patience_counter >= args.patience)
    )
    logger.save()

    print(f"\n{'='*60}")
    print(f"Training complete! Best Val F1: {best_f1:.4f}")
    print(f"Model saved to: {args.output_dir}/best_model.pt")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Specialist Model (CLIP)')

    # Data arguments
    parser.add_argument('--train_data', type=str, default='data/train_split.csv',
                       help='Path to training data CSV')
    parser.add_argument('--val_data', type=str, default='data/val_split.csv',
                       help='Path to validation data CSV')
    parser.add_argument('--root_dir', type=str, default='.',
                       help='Root directory for media files')
    parser.add_argument('--output_dir', type=str, default='models/image_specialist',
                       help='Output directory for model checkpoints')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch32',
                       help='HuggingFace CLIP model name')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=8,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')

    args = parser.parse_args()
    main(args)
