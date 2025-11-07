#!/usr/bin/env python3
"""
Reddit Post Scraper by ID (Expandable, Interruptible, No Duplicates)

Scrapes post content from a CSV of post IDs and appends to raw_data.csv.
Bypasses PRAW's 1000 post limit by fetching posts individually by ID.

Features:
- ✅ Expandable: Appends to existing raw_data.csv
- ✅ Interruptible: Save progress every N posts (Ctrl+C safe)
- ✅ No duplicates: Skips already-scraped posts
- ✅ Media download: Downloads images/videos to media/ folder
"""

import pandas as pd
import praw
import os
import time
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# Reddit API Credentials (loaded from .env file)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# File Paths
DATA_DIR = "data"
ID_CSV = "active_reddit_post_ids.csv"  # Input: CSV with post IDs
RAW_DATA_CSV = os.path.join(DATA_DIR, 'raw_data.csv')  # Output: Append here

# Media Download Settings
MEDIA_DIR = "media"
MEDIA_IMAGES_DIR = os.path.join(MEDIA_DIR, "images")
MEDIA_VIDEOS_DIR = os.path.join(MEDIA_DIR, "videos")

# Processing Settings
BATCH_SIZE = 50  # Save progress every N posts
SLEEP_BETWEEN_POSTS = 0.1  # Seconds to wait between API calls (to avoid rate limits)

# ============================================================================
# SETUP
# ============================================================================

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MEDIA_IMAGES_DIR, exist_ok=True)
    os.makedirs(MEDIA_VIDEOS_DIR, exist_ok=True)

def initialize_reddit():
    """Initialize PRAW Reddit instance."""
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        # Test authentication
        print(f"✅ Reddit API initialized")
        print(f"   Rate limit remaining: {reddit.auth.limits}")
        return reddit
    except Exception as e:
        print(f"❌ Error initializing Reddit API: {e}")
        print("   Please check your credentials!")
        exit(1)

# ============================================================================
# MEDIA DOWNLOAD FUNCTIONS
# ============================================================================

def download_media(url, post_id):
    """
    Download media file from URL.
    
    Returns:
        str: Local path to downloaded file, or None if failed
    """
    if not url or pd.isna(url):
        return None
    
    try:
        # Determine file type and directory
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Image extensions
        if any(path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
            save_dir = MEDIA_IMAGES_DIR
            extension = path.split('.')[-1]
            local_filename = f"{post_id}.{extension}"
        
        # Video extensions
        elif any(path.endswith(ext) for ext in ['.mp4', '.mov']):
            save_dir = MEDIA_VIDEOS_DIR
            extension = path.split('.')[-1]
            local_filename = f"{post_id}.{extension}"
        
        # Reddit video (special handling)
        elif 'v.redd.it' in url:
            save_dir = MEDIA_VIDEOS_DIR
            local_filename = f"{post_id}.mp4"
        
        else:
            # Unsupported media type
            return None
        
        local_path = os.path.join(save_dir, local_filename)
        
        # Skip if already downloaded
        if os.path.exists(local_path):
            return local_path
        
        # Download file
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Save file
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return local_path
    
    except Exception as e:
        # Download failed - not critical
        return None

def get_media_info(submission):
    """
    Extract media URL and type from a Reddit submission.
    
    Returns:
        tuple: (media_url, post_hint)
    """
    post_hint = None
    media_url = None
    
    # Check post_hint first
    if hasattr(submission, 'post_hint'):
        post_hint = submission.post_hint
    
    # Image post
    if post_hint == 'image' or (hasattr(submission, 'url') and 
                                 any(submission.url.lower().endswith(ext) 
                                     for ext in ['.jpg', '.jpeg', '.png', '.gif'])):
        media_url = submission.url
        post_hint = 'image'
    
    # Video post
    elif post_hint == 'hosted:video' or (hasattr(submission, 'is_video') and submission.is_video):
        post_hint = 'hosted:video'
        if hasattr(submission, 'media') and submission.media:
            try:
                media_url = submission.media['reddit_video']['fallback_url']
            except:
                media_url = submission.url
        else:
            media_url = submission.url
    
    # Rich video (YouTube, etc.)
    elif post_hint == 'rich:video':
        media_url = submission.url
    
    # Link post
    elif post_hint == 'link':
        media_url = submission.url
    
    return media_url, post_hint

# ============================================================================
# POST SCRAPING FUNCTIONS
# ============================================================================

def scrape_post_by_id(reddit, post_id):
    """
    Scrape a single post by its ID.
    
    Args:
        reddit: PRAW Reddit instance
        post_id: Post ID (without 't3_' prefix)
    
    Returns:
        dict: Post data or None if failed
    """
    try:
        # Fetch submission
        submission = reddit.submission(id=post_id)
        
        # Get basic info
        post_data = {
            'id': submission.id,
            'title': submission.title,
            'text': submission.selftext if submission.selftext else None,
            'author': str(submission.author) if submission.author else '[deleted]',
            'score': submission.score,
            'upvote_ratio': submission.upvote_ratio,
            'num_comments': submission.num_comments,
            'created_utc': submission.created_utc,
            'url': submission.url,
            'permalink': submission.permalink,
        }
        
        # Get media info
        media_url, post_hint = get_media_info(submission)
        post_data['media_url'] = media_url
        post_data['post_hint'] = post_hint
        
        # Download media if available
        local_media_path = None
        if media_url:
            local_media_path = download_media(media_url, post_id)
        
        post_data['local_media_path'] = local_media_path
        
        return post_data
    
    except Exception as e:
        # Post might be deleted, removed, or inaccessible
        print(f"      ⚠️  Failed to scrape post {post_id}: {type(e).__name__}")
        return None

# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def load_existing_data():
    """Load existing raw_data.csv and return set of already-scraped IDs."""
    try:
        df = pd.read_csv(RAW_DATA_CSV)
        existing_ids = set(df['id'])
        print(f"✅ Loaded {len(df)} posts from {RAW_DATA_CSV}")
        print(f"   Found {len(existing_ids)} unique post IDs")
        return df, existing_ids
    except FileNotFoundError:
        print(f"📝 No existing {RAW_DATA_CSV} found. Creating new file.")
        return pd.DataFrame(), set()

def load_id_csv():
    """Load the CSV with post IDs to scrape."""
    try:
        df = pd.read_csv(ID_CSV)
        print(f"✅ Loaded {len(df)} post IDs from {ID_CSV}")
        
        # Check for required columns
        if 'id' not in df.columns and 'active_post_id' not in df.columns:
            print("❌ Error: CSV must have 'id' or 'active_post_id' column!")
            exit(1)
        
        # Extract IDs (remove 't3_' prefix if present)
        if 'id' in df.columns:
            post_ids = df['id'].tolist()
        else:
            # Remove 't3_' prefix from active_post_id
            post_ids = df['active_post_id'].str.replace('t3_', '', regex=False).tolist()
        
        print(f"   Extracted {len(post_ids)} post IDs")
        return post_ids
    
    except FileNotFoundError:
        print(f"❌ Error: {ID_CSV} not found!")
        print(f"   Please place your CSV file in the current directory.")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading {ID_CSV}: {e}")
        exit(1)

def save_progress(df_existing, new_posts):
    """Save progress by appending new posts to raw_data.csv."""
    if len(new_posts) == 0:
        return df_existing
    
    # Create DataFrame from new posts
    df_new = pd.DataFrame(new_posts)
    
    # Combine with existing data
    if len(df_existing) > 0:
        # Ensure columns match
        all_columns = list(set(df_existing.columns) | set(df_new.columns))
        df_existing = df_existing.reindex(columns=all_columns)
        df_new = df_new.reindex(columns=all_columns)
        
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    # Save to CSV
    df_combined.to_csv(RAW_DATA_CSV, index=False)
    
    return df_combined

def main():
    """Main processing function."""
    print("="*70)
    print("🔍 Reddit Post Scraper by ID")
    print("="*70)
    
    # Check credentials
    if REDDIT_CLIENT_ID == "YOUR_CLIENT_ID_HERE":
        print("\n❌ ERROR: Please set your Reddit API credentials in the script!")
        print("   Edit the REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT")
        exit(1)
    
    # Setup
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Initialize Reddit API
    print("\n🔑 Connecting to Reddit API...")
    reddit = initialize_reddit()
    
    # Load data
    print("\n📊 Loading data...")
    df_existing, existing_ids = load_existing_data()
    post_ids_to_scrape = load_id_csv()
    
    # Filter out duplicates
    new_post_ids = [pid for pid in post_ids_to_scrape if pid not in existing_ids]
    duplicates = len(post_ids_to_scrape) - len(new_post_ids)
    
    print(f"\n🔄 Post Status:")
    print(f"   Total IDs in CSV:       {len(post_ids_to_scrape)}")
    print(f"   Already scraped:        {duplicates}")
    print(f"   New posts to scrape:    {len(new_post_ids)}")
    
    if len(new_post_ids) == 0:
        print("\n✅ All posts already scraped! Nothing to do.")
        return
    
    print(f"\n⏱️  Estimated time: {len(new_post_ids) * SLEEP_BETWEEN_POSTS / 60:.1f} minutes")
    print(f"   (Processing {len(new_post_ids)} posts at ~{SLEEP_BETWEEN_POSTS}s each)")
    
    print(f"\n⏹️  Press Ctrl+C to stop anytime (progress saved every {BATCH_SIZE} posts)")
    
    # Process posts
    print("\n" + "="*70)
    print("🚀 Starting to scrape posts...")
    print("="*70 + "\n")
    
    new_posts = []
    successful = 0
    failed = 0
    
    try:
        for idx, post_id in enumerate(tqdm(new_post_ids, desc="Scraping posts"), 1):
            # Scrape post
            post_data = scrape_post_by_id(reddit, post_id)
            
            if post_data:
                new_posts.append(post_data)
                successful += 1
            else:
                failed += 1
            
            # Save progress every BATCH_SIZE posts
            if idx % BATCH_SIZE == 0:
                print(f"\n💾 Saving progress... ({successful} successful, {failed} failed)")
                df_existing = save_progress(df_existing, new_posts)
                new_posts = []  # Clear buffer
                print(f"✅ Progress saved! Total posts now: {len(df_existing)}")
            
            # Rate limit protection
            time.sleep(SLEEP_BETWEEN_POSTS)
    
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⏹️  INTERRUPTED BY USER (Ctrl+C)")
        print("="*70)
        print("Saving progress...\n")
    
    finally:
        # Save any remaining posts
        if len(new_posts) > 0:
            print(f"💾 Saving final batch...")
            df_existing = save_progress(df_existing, new_posts)
        
        # Final stats
        print("\n" + "="*70)
        print("✅ SCRAPING COMPLETE")
        print("="*70)
        
        print(f"\n📊 Results:")
        print(f"   Successfully scraped:  {successful}")
        print(f"   Failed (deleted/removed): {failed}")
        print(f"   Total in dataset:      {len(df_existing)}")
        
        print(f"\n💾 Data saved to:")
        print(f"   {RAW_DATA_CSV}")
        
        # Media stats
        if len(df_existing) > 0:
            media_count = df_existing['local_media_path'].notna().sum()
            print(f"\n📁 Media downloaded:")
            print(f"   Posts with media: {media_count}")
        
        print("\n" + "="*70)
        print(f"✅ Done! Ready for AI labeling.")
        print("="*70)

if __name__ == "__main__":
    main()
