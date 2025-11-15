"""
Daily Automated Update Script

This script runs the sentiment prediction pipeline daily and can be scheduled using:
- Windows: Task Scheduler
- Linux/Mac: cron

Usage:
    python 07_run_daily_update.py

Schedule with cron (Linux/Mac):
    # Edit crontab
    crontab -e

    # Add line to run daily at 2 AM:
    0 2 * * * cd /path/to/EmoCare1 && /path/to/python 07_run_daily_update.py >> logs/daily_update.log 2>&1

Schedule with Task Scheduler (Windows):
    1. Open Task Scheduler
    2. Create Basic Task
    3. Trigger: Daily at 2:00 AM
    4. Action: Start a program
       Program: python
       Arguments: C:\\path\\to\\EmoCare1\\07_run_daily_update.py
       Start in: C:\\path\\to\\EmoCare1
"""

import subprocess
import os
from datetime import datetime
from pathlib import Path

# Configuration
NUM_POSTS_TO_SCRAPE = 200  # Number of new posts to process daily
LOG_DIR = "logs"


def log_message(message):
    """Print and log message."""
    timestamp = datetime.now().strftime("%Y-%m-%%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)

    # Write to log file
    Path(LOG_DIR).mkdir(exist_ok=True)
    log_file = f"{LOG_DIR}/daily_update_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')


def main():
    log_message("="*80)
    log_message("DAILY SENTIMENT UPDATE - STARTED")
    log_message("="*80)

    try:
        # Run prediction script
        log_message(f"Scraping and predicting {NUM_POSTS_TO_SCRAPE} new posts...")

        result = subprocess.run(
            ['python', '07_predict_new_posts.py', '--num_posts', str(NUM_POSTS_TO_SCRAPE)],
            capture_output=True,
            text=True,
            check=True
        )

        log_message("Prediction script output:")
        log_message(result.stdout)

        if result.returncode == 0:
            log_message("✓ Daily update completed successfully!")
        else:
            log_message(f"✗ Prediction script failed with return code {result.returncode}")
            log_message(f"Error: {result.stderr}")

    except subprocess.CalledProcessError as e:
        log_message(f"✗ Error running prediction script: {e}")
        log_message(f"Error output: {e.stderr}")
    except Exception as e:
        log_message(f"✗ Unexpected error: {e}")

    log_message("="*80)
    log_message("DAILY SENTIMENT UPDATE - FINISHED")
    log_message("="*80)


if __name__ == "__main__":
    main()
