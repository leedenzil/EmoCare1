"""
Step 7: Automated Sentiment Prediction on New Posts

This script:
1. Scrapes new posts from r/BrawlStars
2. Downloads media (images/videos)
3. Runs through all 4 trained models (text, image, video, fusion)
4. Predicts sentiment
5. Appends to sentiment_labelled.csv for dashboard visualization

Usage:
    python 07_predict_new_posts.py --num_posts 100

For daily automation:
    python 07_run_daily_update.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel, CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import praw
import requests
import os
from pathlib import Path
from datetime import datetime
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MODEL_DIR = "models"
MEDIA_DIR = "media"
OUTPUT_CSV = "EmoCare_Visualisation/sentiment_labelled.csv"

# Model parameters
TEXT_MODEL_NAME = 'distilbert-base-uncased'
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
MAX_LENGTH = 128
NUM_FRAMES = 8

# Embedding dimensions
TEXT_EMBED_DIM = 768
IMAGE_EMBED_DIM = 512
VIDEO_EMBED_DIM = 512
TOTAL_EMBED_DIM = TEXT_EMBED_DIM + IMAGE_EMBED_DIM + VIDEO_EMBED_DIM

# Sentiment labels
LABELS = ['Anger', 'Joy', 'Neutral/Other', 'Sadness', 'Surprise']
ID_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ===== MODEL DEFINITIONS =====

class TextSentimentClassifier(nn.Module):
    def __init__(self, n_classes=5):
        super(TextSentimentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(TEXT_MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def get_embedding(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]
        return pooled_output


class ImageSentimentClassifier(nn.Module):
    def __init__(self, n_classes=5):
        super(ImageSentimentClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.vision_embed_dim = self.clip.vision_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.vision_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )

    def get_embedding(self, pixel_values):
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.pooler_output
        return image_embeds


class VideoSentimentClassifier(nn.Module):
    def __init__(self, n_classes=5, num_frames=8):
        super(VideoSentimentClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.num_frames = num_frames
        self.vision_embed_dim = self.clip.vision_model.config.hidden_size

        self.temporal_attention = nn.Sequential(
            nn.Linear(self.vision_embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.vision_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )

    def get_embedding(self, pixel_values):
        with torch.no_grad():
            batch_size, num_frames, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(batch_size * num_frames, C, H, W)

            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            frame_embeds = vision_outputs.pooler_output
            frame_embeds = frame_embeds.view(batch_size, num_frames, -1)

            attention_scores = self.temporal_attention(frame_embeds)
            attention_weights = torch.softmax(attention_scores, dim=1)
            video_embed = torch.sum(frame_embeds * attention_weights, dim=1)

        return video_embed


class FusionModel(nn.Module):
    def __init__(self, input_dim, n_classes=5):
        super(FusionModel, self).__init__()

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.fusion(x)


# ===== UTILITY FUNCTIONS =====

def download_media(url, post_id, media_type):
    """Download image or video from URL."""
    try:
        if media_type == 'image':
            media_path = f"{MEDIA_DIR}/images/{post_id}.jpg"
        else:
            media_path = f"{MEDIA_DIR}/videos/{post_id}.mp4"

        # Create directory if doesn't exist
        Path(media_path).parent.mkdir(parents=True, exist_ok=True)

        # Download
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(media_path, 'wb') as f:
                f.write(response.content)
            return media_path
        return None
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def extract_frames(video_path, num_frames=8):
    """Extract evenly spaced frames from a video."""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return None

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

        cap.release()

        if len(frames) < num_frames:
            while len(frames) < num_frames:
                frames.append(frames[-1] if frames else Image.new('RGB', (224, 224), color='black'))

        return frames[:num_frames]
    except:
        return None


def scrape_new_posts(num_posts=100):
    """Scrape new posts from r/BrawlStars."""
    print(f"Scraping {num_posts} new posts from r/BrawlStars...")

    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT')
    )

    subreddit = reddit.subreddit('BrawlStars')
    posts_data = []

    for post in subreddit.new(limit=num_posts):
        # Determine media type and URL
        media_path = None
        post_hint = post.post_hint if hasattr(post, 'post_hint') else None

        if post_hint == 'image':
            media_path = download_media(post.url, post.id, 'image')
        elif post_hint == 'hosted:video':
            video_url = post.media['reddit_video']['fallback_url'] if post.media else None
            if video_url:
                media_path = download_media(video_url, post.id, 'video')

        posts_data.append({
            'id': post.id,
            'title': post.title,
            'text': post.selftext,
            'url': post.url,
            'permalink': post.permalink,
            'score': post.score,
            'created_utc': post.created_utc,
            'post_hint': post_hint,
            'local_media_path': media_path
        })

    df = pd.DataFrame(posts_data)
    print(f"✓ Scraped {len(df)} posts")
    return df


def predict_sentiment(post, models, tokenizer, clip_processor):
    """Predict sentiment for a single post."""
    text_model, image_model, video_model, fusion_model = models

    # --- TEXT EMBEDDING ---
    title = str(post['title']) if pd.notna(post['title']) else ""
    text = str(post['text']) if pd.notna(post['text']) else ""
    combined_text = f"{title} {text}".strip()

    encoding = tokenizer(combined_text, max_length=MAX_LENGTH, padding='max_length',
                        truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    text_embed = text_model.get_embedding(input_ids, attention_mask).squeeze().cpu().numpy()

    if text_embed.shape[0] != TEXT_EMBED_DIM:
        text_embed = text_embed.flatten()[:TEXT_EMBED_DIM]
        if len(text_embed) < TEXT_EMBED_DIM:
            text_embed = np.pad(text_embed, (0, TEXT_EMBED_DIM - len(text_embed)))

    # --- IMAGE EMBEDDING ---
    media_type = 'image' if 'images' in str(post['local_media_path']) else \
                 'video' if 'videos' in str(post['local_media_path']) else 'none'

    if media_type == 'image' and pd.notna(post['local_media_path']):
        try:
            image = Image.open(post['local_media_path']).convert('RGB')
            inputs = clip_processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            image_embed = image_model.get_embedding(pixel_values).squeeze().cpu().numpy()

            if image_embed.shape[0] != IMAGE_EMBED_DIM:
                image_embed = image_embed.flatten()[:IMAGE_EMBED_DIM]
                if len(image_embed) < IMAGE_EMBED_DIM:
                    image_embed = np.pad(image_embed, (0, IMAGE_EMBED_DIM - len(image_embed)))
        except:
            image_embed = np.zeros(IMAGE_EMBED_DIM, dtype=np.float32)
    else:
        image_embed = np.zeros(IMAGE_EMBED_DIM, dtype=np.float32)

    if image_embed.shape != (IMAGE_EMBED_DIM,):
        image_embed = np.zeros(IMAGE_EMBED_DIM, dtype=np.float32)

    # --- VIDEO EMBEDDING ---
    if media_type == 'video' and pd.notna(post['local_media_path']):
        frames = extract_frames(post['local_media_path'], NUM_FRAMES)

        if frames:
            try:
                pixel_values_list = []
                for frame in frames:
                    inputs = clip_processor(images=frame, return_tensors="pt")
                    pixel_values_list.append(inputs['pixel_values'].squeeze(0))
                pixel_values = torch.stack(pixel_values_list).unsqueeze(0).to(device)
                video_embed = video_model.get_embedding(pixel_values).squeeze().cpu().numpy()

                if video_embed.shape[0] != VIDEO_EMBED_DIM:
                    video_embed = video_embed.flatten()[:VIDEO_EMBED_DIM]
                    if len(video_embed) < VIDEO_EMBED_DIM:
                        video_embed = np.pad(video_embed, (0, VIDEO_EMBED_DIM - len(video_embed)))
            except:
                video_embed = np.zeros(VIDEO_EMBED_DIM, dtype=np.float32)
        else:
            video_embed = np.zeros(VIDEO_EMBED_DIM, dtype=np.float32)
    else:
        video_embed = np.zeros(VIDEO_EMBED_DIM, dtype=np.float32)

    if video_embed.shape != (VIDEO_EMBED_DIM,):
        video_embed = np.zeros(VIDEO_EMBED_DIM, dtype=np.float32)

    # --- FUSION PREDICTION ---
    combined_embed = np.concatenate([text_embed, image_embed, video_embed])
    combined_tensor = torch.FloatTensor(combined_embed).unsqueeze(0).to(device)

    with torch.no_grad():
        output = fusion_model(combined_tensor)
        _, prediction = torch.max(output, dim=1)
        predicted_sentiment = ID_TO_LABEL[prediction.item()]

    return predicted_sentiment


def main():
    parser = argparse.ArgumentParser(description='Predict sentiment on new Reddit posts')
    parser.add_argument('--num_posts', type=int, default=100, help='Number of posts to scrape')
    args = parser.parse_args()

    print("="*80)
    print("AUTOMATED SENTIMENT PREDICTION SYSTEM")
    print("="*80)

    # Load models
    print("\nLoading trained models...")
    text_model = TextSentimentClassifier(n_classes=len(LABELS))
    text_model.load_state_dict(torch.load(f"{MODEL_DIR}/text_specialist_best.pth", map_location=device))
    text_model = text_model.to(device)
    text_model.eval()

    image_model = ImageSentimentClassifier(n_classes=len(LABELS))
    image_model.load_state_dict(torch.load(f"{MODEL_DIR}/image_specialist_best.pth", map_location=device))
    image_model = image_model.to(device)
    image_model.eval()

    video_model = VideoSentimentClassifier(n_classes=len(LABELS), num_frames=NUM_FRAMES)
    video_model.load_state_dict(torch.load(f"{MODEL_DIR}/video_specialist_best.pth", map_location=device))
    video_model = video_model.to(device)
    video_model.eval()

    fusion_model = FusionModel(input_dim=TOTAL_EMBED_DIM, n_classes=len(LABELS))
    fusion_model.load_state_dict(torch.load(f"{MODEL_DIR}/fusion_model_best.pth", map_location=device))
    fusion_model = fusion_model.to(device)
    fusion_model.eval()

    print("✓ All models loaded")

    # Load processors
    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    print("✓ Processors loaded")

    models = (text_model, image_model, video_model, fusion_model)

    # Scrape new posts
    new_posts_df = scrape_new_posts(args.num_posts)

    # Predict sentiments
    print(f"\nPredicting sentiments for {len(new_posts_df)} posts...")
    predictions = []
    for idx, post in new_posts_df.iterrows():
        try:
            sentiment = predict_sentiment(post, models, tokenizer, clip_processor)
            predictions.append(sentiment)
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(new_posts_df)} posts...")
        except Exception as e:
            print(f"  Error predicting post {post['id']}: {e}")
            predictions.append('Neutral/Other')  # Default to neutral on error

    new_posts_df['post_sentiment'] = predictions
    new_posts_df['post_classification'] = ''  # Can add classification logic if needed
    new_posts_df['sentiment_analysis'] = ''
    new_posts_df['labeling_error'] = ''

    # Load existing data and append
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        # Remove duplicates by ID
        existing_ids = set(existing_df['id'])
        new_posts_df = new_posts_df[~new_posts_df['id'].isin(existing_ids)]

        if len(new_posts_df) > 0:
            combined_df = pd.concat([existing_df, new_posts_df], ignore_index=True)
            combined_df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n✓ Added {len(new_posts_df)} new posts to {OUTPUT_CSV}")
        else:
            print(f"\n✓ No new posts to add (all already exist in dataset)")
    else:
        new_posts_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✓ Created new dataset with {len(new_posts_df)} posts at {OUTPUT_CSV}")

    # Print sentiment distribution
    print("\nSentiment Distribution:")
    print(new_posts_df['post_sentiment'].value_counts())

    print("\n" + "="*80)
    print("PREDICTION COMPLETE!")
    print("="*80)
    print(f"\nDashboard updated. Run: streamlit run EmoCare_Visualisation/Dashboard.py")
    print("="*80)


if __name__ == "__main__":
    main()
