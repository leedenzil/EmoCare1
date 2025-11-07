#!/usr/bin/env python3
"""
AI-Powered Labeling Script (Optimized with Concurrent Processing)

Features:
- 15-20x faster than sequential processing
- Interruptible (Ctrl+C to stop)
- Resumable (skips already labeled posts)
- Expandable (appends to existing dataset)
- Proper media upload failure handling

Usage:
    python ai_labeling_script.py
"""

import pandas as pd
import os
import json
import time
import asyncio
from tqdm.auto import tqdm
import google.generativeai as genai
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# API Configuration (loaded from .env file)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"  # Use the latest stable model

# File Paths
DATA_DIR = "data"
RAW_DATA_CSV = os.path.join(DATA_DIR, 'raw_data.csv')
LABELED_DATA_CSV = os.path.join(DATA_DIR, 'labeled_data_1k.csv')

# Labeling Configuration
NEW_LABEL_TARGET = 1000  # Max posts to label in this run

# Concurrent Processing Settings (Optimized for 1,500 RPM limit)
MAX_CONCURRENT_REQUESTS = 25  # Process 25 posts simultaneously
MAX_CONCURRENT_UPLOADS = 5    # Only 5 media uploads at once (prevents hanging)
BATCH_SAVE_INTERVAL = 20     # Save progress every N posts
RETRY_ATTEMPTS = 3            # Number of retries for failed requests
RETRY_DELAY = 2               # Initial delay between retries (seconds)

# ============================================================================
# LABELING PROMPT TEMPLATE
# ============================================================================

LABELING_PROMPT_TEMPLATE = """
You are an expert sentiment analyst for the game Brawl Stars. Your task is to analyze a Reddit post (which may include text, an image, and/or a video) and provide a structured JSON output.

Analyze the user's sentiment and categorize the post. The user's post content is provided first, followed by the media.

The 5 possible 'post_sentiment' values are:
1.  **Joy**: Happiness, excitement, pride (e.g., getting a new Brawler, winning a hard match, liking a new skin).
2.  **Anger**: Frustration, rage, annoyance (e.g., losing to a specific Brawler, bad teammates, game bugs, matchmaking issues).
3.  **Sadness**: Disappointment, grief (e.g., missing a shot, losing a high-stakes game, a favorite Brawler getting nerfed).
4.  **Surprise**: Shock, disbelief (e.g., a sudden clutch play, an unexpected new feature, a rare bug).
5.  **Neutral/Other**: Objective discussion, questions, news, or art that doesn't convey a strong emotion.

The 6 possible 'post_classification' values are:
1.  **Gameplay Clip**: A video or image showing a match, a specific play, or a replay.
2.  **Meme/Humor**: A meme, joke, or funny edit.
3.  **Discussion**: A text-based post asking a question or starting a conversation.
4.  **Feedback/Rant**: A post providing feedback, suggestions, or complaining about the game.
5.  **Art/Concept**: Fan art, skin concepts, or creative edits.
6.  **Achievement/Loot**: A screenshot of a new Brawler unlock, a high rank, or a Starr Drop reward.

--- EXAMPLES ---

[EXAMPLE 1]
Post Text: "This is the 5th time I've lost to an Edgar in a row. FIX YOUR GAME SUPERCELL!!"
Post Media: <image of defeat screen>
Output:
{{
  "post_classification": "Feedback/Rant",
  "post_sentiment": "Anger",
  "sentiment_analysis": "The user is clearly angry, using all-caps ('FIX YOUR GAME') and expressing frustration at repeatedly losing to a specific Brawler (Edgar). The defeat screen image confirms the loss."
}}

[EXAMPLE 2]
Post Text: "I CAN'T BELIEVE I FINALLY GOT HIM!!"
Post Media: <image of legendary Brawler unlock>
Output:
{{
  "post_classification": "Achievement/Loot",
  "post_sentiment": "Joy",
  "sentiment_analysis": "The user is excited and happy, indicated by the all-caps text and the celebratory nature of unlocking a new legendary Brawler, which is a rare event."
}}

[EXAMPLE 3]
Post Text: "Check out this insane 1v3 I pulled off with Mortis"
Post Media: <video showing a fast-paced gameplay clip where the player (Mortis) defeats three opponents>
Output:
{{
  "post_classification": "Gameplay Clip",
  "post_sentiment": "Joy",
  "sentiment_analysis": "The user is proud and excited about their 'insane 1v3' play. This is a clear expression of joy and pride in their own skill. The video clip demonstrates the achievement."
}}

--- TASK ---

Analyze the following post and provide ONLY the JSON output. Do not include '```json' or any other text outside the JSON block.

[POST CONTENT]
Title: {post_title}
Text: {post_text}

[POST MEDIA]
"""

# ============================================================================
# SETUP AND INITIALIZATION
# ============================================================================

def initialize_gemini():
    """Initialize the Gemini API client."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        print(f"✅ Gemini API configured successfully")
        print(f"   Model: {GEMINI_MODEL}")
        print(f"   Rate limits: 1,500 RPM, 4M TPM")
        return model
    except Exception as e:
        print(f"❌ Error configuring Gemini API: {e}")
        print("   Please check your API key!")
        exit(1)


# ============================================================================
# MEDIA UPLOAD FUNCTIONS
# ============================================================================

async def upload_media_async(media_path: str, post_id: str = "unknown") -> Tuple[Optional[Any], Optional[str]]:
    """
    Asynchronously upload media file to Gemini with progress tracking.
    
    Returns:
        Tuple[Optional[uploaded_file], Optional[error_message]]
        - (file, None) if upload successful
        - (None, None) if no media exists
        - (None, error) if upload failed
    """
    # No media path provided
    if pd.isna(media_path) or not os.path.exists(media_path):
        return (None, None)  # No media to upload - this is OK
    
    # Get file info
    file_size = os.path.getsize(media_path) / (1024 * 1024)  # Size in MB
    file_name = os.path.basename(media_path)
    
    # Media exists, try to upload
    try:
        # Upload with timeout
        print(f"   📤 Uploading {file_name} ({file_size:.1f}MB) for post {post_id}...", end="", flush=True)
        
        # Run upload in thread pool with timeout
        loop = asyncio.get_event_loop()
        try:
            uploaded_file = await asyncio.wait_for(
                loop.run_in_executor(None, genai.upload_file, media_path),
                timeout=60.0  # 60 second timeout for upload
            )
        except asyncio.TimeoutError:
            print(" ❌ Timeout")
            return (None, "Media upload timeout (>60s)")
        
        print(" ✓ Uploaded", end="", flush=True)
        
        # Wait for processing
        max_wait = 30
        wait_count = 0
        while uploaded_file.state.name == "PROCESSING" and wait_count < max_wait:
            if wait_count % 5 == 0:
                print(".", end="", flush=True)
            await asyncio.sleep(1)
            uploaded_file = genai.get_file(uploaded_file.name)
            wait_count += 1
        
        if uploaded_file.state.name == "FAILED":
            print(" ❌ Failed")
            return (None, "Media upload failed (processing failed)")
        
        if wait_count >= max_wait:
            print(" ❌ Timeout")
            return (None, "Media upload timeout (processing >30s)")
        
        print(" ✅")
        return (uploaded_file, None)  # Success
    
    except asyncio.TimeoutError:
        print(" ❌ Timeout")
        return (None, "Media upload timeout")
    except Exception as e:
        print(f" ❌ Error")
        return (None, f"Media upload error: {str(e)}")


# ============================================================================
# LABELING FUNCTIONS
# ============================================================================

def get_gemini_label_sync(gemini_model, post_row, post_has_media: bool, uploaded_file: Optional[Any]) -> Dict[str, Any]:
    """
    Synchronous labeling function.
    
    Args:
        gemini_model: Gemini model instance
        post_row: Post data
        post_has_media: Whether this post should have media
        uploaded_file: Uploaded media file (if any)
    """
    post_title = post_row.get('title', '')
    post_text = post_row.get('text', '')
    
    # Format prompt
    prompt = LABELING_PROMPT_TEMPLATE.format(
        post_title=post_title if pd.notna(post_title) else "",
        post_text=post_text if pd.notna(post_text) else ""
    )
    
    # Prepare media payload
    media_payload = []
    if uploaded_file:
        media_payload.append(uploaded_file)
    elif not post_has_media:
        # Post legitimately has no media
        media_payload.append("No media provided.")
    else:
        # Post SHOULD have media but we don't have it
        return {"error": "Internal error: Post has media but file not provided"}
    
    # Make API call
    try:
        full_prompt = [prompt] + media_payload
        response = gemini_model.generate_content(full_prompt)
        
        # Clean up uploaded file
        if uploaded_file and hasattr(uploaded_file, 'name'):
            try:
                genai.delete_file(uploaded_file.name)
            except:
                pass
        
        return {"result": response.text}
    
    except Exception as e:
        return {"error": f"API error: {str(e)}"}


async def get_gemini_label_async(gemini_model, post_row: pd.Series, executor, upload_semaphore, retry_count: int = 0) -> Dict[str, Any]:
    """
    Async wrapper with proper media handling and retry logic.
    """
    post_id = post_row['id']
    media_path = post_row.get('local_media_path')
    
    try:
        # Check if post has media
        post_has_media = pd.notna(media_path) and os.path.exists(media_path)
        
        # Try to upload media if it exists (with semaphore to limit concurrent uploads)
        if post_has_media:
            async with upload_semaphore:  # Limit concurrent uploads
                uploaded_file, upload_error = await upload_media_async(media_path, post_id)
            
            if upload_error:
                # CRITICAL: Media upload failed for a post that HAS media
                # We MUST reject this labeling attempt
                return {
                    'id': post_id,
                    'result': {"error": f"Media upload failed: {upload_error}"}
                }
        else:
            uploaded_file = None
        
        # Now do the actual labeling
        loop = asyncio.get_event_loop()
        label_result = await loop.run_in_executor(
            executor,
            lambda: get_gemini_label_sync(gemini_model, post_row, post_has_media, uploaded_file)
        )
        
        # Check if labeling itself failed
        if "error" in label_result:
            return {'id': post_id, 'result': label_result}
        
        return {'id': post_id, 'result': label_result["result"]}
    
    except Exception as e:
        # Retry with exponential backoff
        if retry_count < RETRY_ATTEMPTS:
            delay = RETRY_DELAY * (2 ** retry_count)
            await asyncio.sleep(delay)
            return await get_gemini_label_async(gemini_model, post_row, executor, upload_semaphore, retry_count + 1)
        
        return {
            'id': post_id,
            'result': {"error": f"Failed after {RETRY_ATTEMPTS} retries: {str(e)}"}
        }


async def process_batch_concurrent(gemini_model, batch_df, executor, semaphore, upload_semaphore):
    """Process a batch concurrently with rate limiting."""
    async def process_with_limit(row):
        async with semaphore:
            return await get_gemini_label_async(gemini_model, row, executor, upload_semaphore)
    
    tasks = [process_with_limit(row) for _, row in batch_df.iterrows()]
    return await asyncio.gather(*tasks)


# ============================================================================
# DATA MANAGEMENT FUNCTIONS
# ============================================================================

def save_incremental_progress(df_old_labeled, results_so_far, df_original):
    """Save progress incrementally."""
    if len(results_so_far) == 0:
        return df_old_labeled
    
    # Parse results
    parsed_labels = []
    for item in results_so_far:
        post_id = item['id']
        raw_json = item['result']
        
        try:
            if isinstance(raw_json, dict) and 'error' in raw_json:
                # Error case (including media upload failures)
                parsed_labels.append({
                    'id': post_id,
                    'labeling_error': raw_json['error']
                })
            elif isinstance(raw_json, str):
                # Success case
                clean_json_str = raw_json.strip().replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json_str)
                parsed_labels.append({
                    'id': post_id,
                    'post_classification': data.get('post_classification'),
                    'post_sentiment': data.get('post_sentiment'),
                    'sentiment_analysis': data.get('sentiment_analysis'),
                    'labeling_error': None
                })
            else:
                parsed_labels.append({
                    'id': post_id,
                    'labeling_error': 'Unknown response format'
                })
        except Exception as e:
            parsed_labels.append({
                'id': post_id,
                'labeling_error': f"Parse error: {str(e)}"
            })
    
    # Merge with original data
    df_labels = pd.DataFrame(parsed_labels)
    processed_ids = set(df_labels['id'])
    df_processed_original = df_original[df_original['id'].isin(processed_ids)].copy()
    df_new_labeled_final = pd.merge(df_processed_original, df_labels, on='id', how='left')
    
    # Filter out errors
    df_new_golden = df_new_labeled_final[df_new_labeled_final['labeling_error'].isna()].copy()
    df_errors = df_new_labeled_final[df_new_labeled_final['labeling_error'].notna()]
    
    # Report media upload failures specifically
    media_upload_errors = df_errors[df_errors['labeling_error'].str.contains('Media upload failed', na=False)]
    if len(media_upload_errors) > 0:
        print(f"   ⚠️  {len(media_upload_errors)} posts skipped due to media upload failures")
    
    # Combine with old data (EXPANDABLE)
    if len(df_old_labeled) > 0:
        df_combined = pd.concat([df_old_labeled, df_new_golden], ignore_index=True)
        if not df_old_labeled.empty:
            df_combined = df_combined.reindex(columns=df_old_labeled.columns)
    else:
        df_combined = df_new_golden
    
    # Save
    if len(df_combined) > 0:
        df_combined.to_csv(LABELED_DATA_CSV, index=False)
    
    return df_combined


# ============================================================================
# MAIN LABELING PROCESS
# ============================================================================

async def main_concurrent_labeling(gemini_model):
    """Main async function for concurrent labeling."""
    
    print("="*70)
    print("🤖 AI-POWERED LABELING (CONCURRENT)")
    print("="*70)
    
    # --- 1. Load Existing Data (RESUMABLE) ---
    try:
        df_old_labeled = pd.read_csv(LABELED_DATA_CSV)
        already_labeled_ids = set(df_old_labeled['id'])
        print(f"\n✅ Loaded {len(df_old_labeled)} previously labeled posts")
    except FileNotFoundError:
        df_old_labeled = pd.DataFrame()
        already_labeled_ids = set()
        print(f"\n📝 No existing labeled data. Starting from scratch.")
    
    # --- 2. Load Raw Data ---
    try:
        df_all_raw = pd.read_csv(RAW_DATA_CSV)
        print(f"✅ Loaded {len(df_all_raw)} raw posts")
    except FileNotFoundError:
        print(f"\n❌ Error: {RAW_DATA_CSV} not found!")
        print(f"   Please run the data collection script first.")
        return
    
    # --- 3. Find Unlabeled Posts (EXPANDABLE) ---
    df_to_label = df_all_raw[~df_all_raw['id'].isin(already_labeled_ids)].copy()

    # Filter to only include posts with media attached
    df_to_label = df_to_label[df_to_label['local_media_path'].notna()].copy()
    print(f"   Filtering to posts with media only...")

    df_to_label = df_to_label.head(NEW_LABEL_TARGET)
    
    if len(df_to_label) == 0:
        print("\n✅ No new posts with media to label. Dataset is up to date!")
        return

    print(f"\n📊 Found {len(df_to_label)} new posts with media to label")
    print(f"   (Text-only posts are skipped)")
    
    print(f"\n⚡ Concurrent Processing:")
    print(f"   Max concurrent:   {MAX_CONCURRENT_REQUESTS} posts")
    print(f"   Batch save:       Every {BATCH_SAVE_INTERVAL} posts")
    
    print(f"\n⏱️  Estimated time:   {len(df_to_label) / (MAX_CONCURRENT_REQUESTS * 1.2) / 60:.1f} minutes")
    print(f"   (vs sequential:   {len(df_to_label) * 3.5 / 60:.1f} minutes)")
    
    print(f"\n⏹️  Press Ctrl+C to stop anytime (progress is saved)")
    
    # --- 4. Setup Concurrent Processing ---
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    upload_semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)  # Limit concurrent uploads
    
    print(f"   ⚠️  Note: Only {MAX_CONCURRENT_UPLOADS} media files upload at once (prevents hanging)")
    
    all_results = []
    num_batches = (len(df_to_label) + BATCH_SAVE_INTERVAL - 1) // BATCH_SAVE_INTERVAL
    
    # --- 5. Process Batches (INTERRUPTIBLE) ---
    try:
        print("\n" + "="*70)
        print("🚀 Starting concurrent labeling...\n")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SAVE_INTERVAL
            end_idx = min((batch_idx + 1) * BATCH_SAVE_INTERVAL, len(df_to_label))
            batch_df = df_to_label.iloc[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} (posts {start_idx+1}-{end_idx})...")
            
            # Process batch concurrently
            batch_results = await process_batch_concurrent(gemini_model, batch_df, executor, semaphore, upload_semaphore)
            all_results.extend(batch_results)
            
            # Save after each batch
            df_combined = save_incremental_progress(df_old_labeled, all_results, df_to_label)
            
            print(f"✅ Batch {batch_idx + 1} complete. Progress saved. ({len(all_results)} total)\n")
    
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⏹️  INTERRUPTED BY USER (Ctrl+C)")
        print("="*70)
        print("Saving progress...\n")
    
    finally:
        # Cleanup
        executor.shutdown(wait=False)
        
        # --- 6. Final Processing ---
        if len(all_results) == 0:
            print("⚠️  No posts were labeled in this session.")
        else:
            print(f"\n📊 Processing {len(all_results)} results...")
            df_combined = save_incremental_progress(df_old_labeled, all_results, df_to_label)
            
            # Count results
            successes = sum(1 for r in all_results if not (isinstance(r['result'], dict) and 'error' in r['result']))
            failures = len(all_results) - successes
            
            # Count media-specific failures
            media_failures = sum(1 for r in all_results 
                               if isinstance(r['result'], dict) 
                               and 'error' in r['result']
                               and 'Media upload failed' in r['result']['error'])
            
            print(f"\n" + "="*70)
            print("✅ LABELING COMPLETE")
            print("="*70)
            
            print(f"\n💾 Dataset updated!")
            print(f"   Total labeled posts: {len(df_combined)}")
            print(f"   Saved to:            {LABELED_DATA_CSV}")
            
            print(f"\n📋 Session Statistics:")
            print(f"   ✅ Successfully labeled:    {successes}")
            print(f"   ❌ Failed (total):          {failures}")
            if media_failures > 0:
                print(f"   📷 Media upload failures:   {media_failures}")
                print(f"      (Posts NOT labeled without media context)")
            
            # Show distributions
            if 'post_sentiment' in df_combined.columns:
                print(f"\n📈 Sentiment Distribution:")
                sentiment_counts = df_combined['post_sentiment'].value_counts()
                for sentiment, count in sentiment_counts.items():
                    print(f"   {sentiment}: {count}")
            
            if 'post_classification' in df_combined.columns:
                print(f"\n📂 Classification Distribution:")
                class_counts = df_combined['post_classification'].value_counts()
                for classification, count in class_counts.items():
                    print(f"   {classification}: {count}")
            
            print("\n" + "="*70)
            print(f"✅ Done! {len(df_combined)} posts now labeled.")
            print("="*70)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the script."""
    print("\n" + "="*70)
    print("🤖 AI-Powered Labeling Script")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"   Model:            {GEMINI_MODEL}")
    print(f"   Target posts:     {NEW_LABEL_TARGET}")
    print(f"   Concurrent:       {MAX_CONCURRENT_REQUESTS}")
    print(f"   Raw data:         {RAW_DATA_CSV}")
    print(f"   Labeled data:     {LABELED_DATA_CSV}")
    
    # Check if API key is set
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("\n❌ ERROR: Please set your Gemini API key in the script!")
        print("   Edit the GEMINI_API_KEY variable at the top of this file.")
        exit(1)
    
    # Initialize Gemini
    gemini_model = initialize_gemini()
    
    # Run the async labeling process
    try:
        asyncio.run(main_concurrent_labeling(gemini_model))
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()