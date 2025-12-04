# EmoCare - Reddit Sentiment Analysis System

EmoCare is a multimodal sentiment analysis system that analyzes Reddit posts from r/BrawlStars using text, images, and videos to predict sentiment. The system uses a fusion of specialized deep learning models and provides real-time visualization through an interactive dashboard.

## Features

- **Multimodal Analysis**: Combines text (DistilBERT), images (CLIP), and videos for comprehensive sentiment analysis
- **5 Sentiment Categories**: Anger, Joy, Sadness, Surprise, and Neutral/Other
- **Automated Data Collection**: Scrapes new posts from Reddit automatically
- **Interactive Dashboard**: Real-time sentiment visualization with Streamlit
- **Scheduled Updates**: Support for daily automated sentiment monitoring

## System Architecture

The system consists of four specialized models:

1. **Text Specialist**: DistilBERT-based model for analyzing post titles and text content
2. **Image Specialist**: CLIP-based model for analyzing image posts
3. **Video Specialist**: CLIP + Temporal Attention for analyzing video posts
4. **Fusion Model**: Combines embeddings from all specialists for final prediction

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster processing)
- Reddit API credentials
- 2-3 GB of disk space for models and data

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/leedenzil/EmoCare1.git
cd "EmoCare (v1)"
```

### 2. Create Virtual Environment

```bash
# Create environment
conda create -n emocare python=3.10
conda activate emocare

# Or using venv
python -m venv emocare
source emocare/bin/activate  # Linux/Mac
emocare\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Credentials

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# Reddit API Credentials
# Get these from: https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=BrawlStars Sentiment Scraper v3.0 by /u/YOUR_USERNAME

# Gemini API Key (optional, only needed for training/labeling)
GEMINI_API_KEY=your_gemini_api_key_here
```

**Getting Reddit API Credentials:**
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Select "script" as the app type
4. Fill in the required fields
5. Copy the client ID and secret

### 5. Download Pre-trained Models

The trained models should be placed in the `models/` directory:

```
models/
├── text_specialist_best.pth
├── image_specialist_best.pth
├── video_specialist_best.pth
└── fusion_model_best.pth
```

**Note**: Models are not included in the repository due to file size. Contact the repository maintainer or train models using the provided notebooks.

## Usage

### Running Sentiment Analysis

Analyze new Reddit posts and generate predictions:

```bash
python 07_predict_new_posts.py --num_posts 100
```

**Parameters:**
- `--num_posts`: Number of posts to scrape and analyze (default: 100)

**What it does:**
1. Scrapes the latest posts from r/BrawlStars
2. Downloads associated media (images/videos)
3. Runs all four models (text, image, video, fusion)
4. Predicts sentiment for each post
5. Saves results to `EmoCare_Visualisation/sentiment_labelled.csv`

**Example output:**
```
================================================================================
AUTOMATED SENTIMENT PREDICTION SYSTEM
================================================================================

Loading trained models...
✓ All models loaded
✓ Processors loaded

Scraping 100 new posts from r/BrawlStars...
✓ Scraped 95 posts
  (Skipped 5 gallery posts - not in training data)

Predicting sentiments for 95 posts...
  Processed 10/95 posts...
  Processed 20/95 posts...
  ...

✓ Added 95 new posts to EmoCare_Visualisation/sentiment_labelled.csv

Sentiment Distribution:
Joy              42
Neutral/Other    28
Anger            15
Sadness           7
Surprise          3
```

### Launching the Dashboard

Start the interactive Streamlit dashboard:

```bash
streamlit run EmoCare_Visualisation/Dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

**Dashboard Features:**

- **Sentiment Distribution Cards**: Current sentiment breakdown with comparison to previous periods
- **Sentiment Trend Chart**: Time series visualization of sentiment over time
  - Adjustable frequency (Daily, Weekly, Monthly, Yearly)
  - Date range filtering
- **Sentiment vs Engagement**: Scatter plot showing relationship between sentiment and post engagement (upvotes)
- **Interactive Filters**: Filter by date range, sentiment categories, and time frequency

**Example Dashboard Usage:**

1. Run sentiment prediction to collect data
2. Launch dashboard
3. View real-time sentiment trends
4. Compare current sentiment with previous periods (day/week/month/year)
5. Analyze which sentiments drive engagement

### Automated Daily Updates

Set up automated daily sentiment monitoring:

#### Option 1: Manual Daily Run

```bash
python 07_run_daily_update.py
```

This runs the prediction script and logs output to `logs/` directory.

#### Option 2: Scheduled Automation

**Linux/Mac (cron):**

```bash
# Edit crontab
crontab -e

# Add this line to run daily at 2 AM:
0 2 * * * cd /path/to/EmoCare1 && /path/to/conda/envs/emocare/bin/python 07_run_daily_update.py >> logs/daily_update.log 2>&1
```

**Windows (Task Scheduler):**

1. Open Task Scheduler
2. Create Basic Task
3. Name: "EmoCare Daily Update"
4. Trigger: Daily at 2:00 AM
5. Action: Start a program
   - Program: `python` (or full path to Python in your environment)
   - Arguments: `C:\path\to\EmoCare1\07_run_daily_update.py`
   - Start in: `C:\path\to\EmoCare1`

## Project Structure

```
EmoCare (v1)/
├── models/                          # Trained model files (.pth)
├── media/                           # Downloaded Reddit media
│   ├── images/                      # Image posts
│   └── videos/                      # Video posts
├── data/                            # Training datasets (not needed for inference)
├── results/                         # Model evaluation results
├── EmoCare_Visualisation/           # Dashboard application
│   ├── Dashboard.py                 # Main dashboard
│   ├── sentiment_timeseries.py      # Time series visualization
│   ├── sentiment_engagement.py      # Engagement analysis
│   ├── sentiment_labelled.csv       # Predicted sentiment data
│   └── styles.css                   # Dashboard styling
├── logs/                            # Automation logs
├── 07_predict_new_posts.py          # Main prediction script
├── 07_run_daily_update.py           # Automated update script
├── requirements.txt                 # Python dependencies
├── .env                             # API credentials (create from .env.example)
└── README.md                        # This file
```

## Troubleshooting

### Models Not Found

**Error:** `FileNotFoundError: models/text_specialist_best.pth`

**Solution:** Ensure all model files are in the `models/` directory. Download from repository maintainer or train using the provided notebooks.

### Reddit API Errors

**Error:** `PrawException: invalid_grant`

**Solution:**
- Verify credentials in `.env` file
- Ensure app type is "script" in Reddit preferences
- Check that redirect URI is set to `http://localhost:8080`

### Dashboard Shows No Data

**Error:** Dashboard shows "No sentiment data available yet"

**Solution:**
1. Run the prediction script first: `python 07_predict_new_posts.py --num_posts 100`
2. Verify `EmoCare_Visualisation/sentiment_labelled.csv` exists and contains data
3. Refresh the dashboard

### File Locked Error (Windows)

**Error:** `PermissionError: [Errno 13] Permission denied: 'sentiment_labelled.csv'`

**Solution:**
- Close Excel or any program that has the CSV file open
- Close the dashboard before running prediction script
- Script will automatically create a temporary file if needed

### Out of Memory (GPU)

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
- Reduce batch size by processing fewer posts at once
- Use CPU instead: The script automatically falls back to CPU if GPU is unavailable
- Close other GPU-intensive applications

### Video Processing Errors

**Error:** Issues extracting frames from videos

**Solution:**
- Ensure `opencv-python` is installed correctly
- Some video formats may not be supported
- The system will default to zero embeddings for failed videos

## Model Performance

The fusion model achieves the following performance on the test set:

| Metric | Score |
|--------|-------|
| Overall Accuracy | ~85% |
| Macro F1 Score | ~0.82 |
| Weighted F1 Score | ~0.84 |

**Per-Class Performance:**
- **Joy**: Highest accuracy (~90%) - most common sentiment
- **Anger**: Good performance (~85%)
- **Neutral/Other**: Moderate performance (~75%)
- **Sadness**: Lower performance (~70%) - less common in dataset
- **Surprise**: Challenging (~65%) - least common sentiment

Detailed evaluation reports are available in `results/final_evaluation_v2/`

## API Rate Limits

**Reddit API:**
- Rate limit: 60 requests per minute
- The script automatically handles rate limiting
- Recommended: Scrape 100-500 posts per run to stay within limits

## Data Privacy

- No personal user information is stored
- Only public Reddit posts are analyzed
- Media files are stored locally and not shared
- Sentiment predictions are used for analysis only

## Contributing

Contributions are welcome! Areas for improvement:

- Additional sentiment categories
- Support for more subreddits
- Improved video processing
- Enhanced dashboard features
- Model optimization for faster inference

## Citation

If you use this project in your research or work, please cite:

```
EmoCare - Multimodal Sentiment Analysis System
GitHub: https://github.com/leedenzil/EmoCare1
Year: 2024
```

## License

This project is for educational and research purposes.

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Contact: [Your Contact Information]

## Acknowledgments

- **Models**: DistilBERT (Hugging Face), CLIP (OpenAI)
- **Data Source**: Reddit r/BrawlStars community
- **Frameworks**: PyTorch, Transformers, Streamlit

---

**Last Updated**: November 2024

**Version**: 1.0

**Status**: Production Ready
