# EmoCare Sentiment Monitoring Dashboard

Real-time sentiment analysis dashboard for Brawl Stars Reddit community, powered by a trained multimodal AI model.

## 🎯 Features

- **📊 Sentiment Distribution**: Overview of current sentiment with comparison metrics
- **📈 Sentiment Trend**: Time-series visualization showing sentiment changes over time
- **💬 Engagement Analysis**: See which sentiments get the most community engagement
- **🔄 Auto-Update**: Automatically updates with new predictions from trained model
- **🎨 Interactive Filters**: Filter by date range, sentiment type, and frequency

## 🚀 Quick Start

### 1. Run the Dashboard

```bash
# From the EmoCare1 directory
streamlit run EmoCare_Visualisation/Dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 2. Update with New Predictions

#### Manual Update (Ad-hoc)

```bash
# Scrape and predict 100 new posts
python 07_predict_new_posts.py --num_posts 100
```

#### Automated Daily Update

```bash
# Run once to test
python 07_run_daily_update.py
```

**Schedule for Daily Automation:**

**Linux/Mac (using cron):**
```bash
# Edit crontab
crontab -e

# Add this line to run daily at 2 AM:
0 2 * * * cd /path/to/EmoCare1 && python 07_run_daily_update.py >> logs/daily_update.log 2>&1
```

**Windows (using Task Scheduler):**
1. Open Task Scheduler
2. Create Basic Task → Name: "EmoCare Daily Update"
3. Trigger: Daily at 2:00 AM
4. Action: Start a program
   - Program: `python`
   - Arguments: `C:\path\to\EmoCare1\07_run_daily_update.py`
   - Start in: `C:\path\to\EmoCare1`

## 📁 File Structure

```
EmoCare_Visualisation/
├── Dashboard.py                 # Main dashboard application
├── sentiment_timeseries.py      # Time series chart component
├── sentiment_engagement.py      # Engagement analysis component
├── sentiment_labelled.csv       # Labeled sentiment data (auto-updated)
├── styles.css                   # Custom dashboard styling
├── .streamlit/
│   └── config.toml             # Streamlit configuration
└── README.md                   # This file
```

## 📊 Dashboard Components

### Sentiment Distribution Cards
- Shows current percentage for each sentiment
- Compares against previous period (day/week/month/year)
- Color-coded progress bars

### Sentiment Trend Chart
- Line chart showing sentiment distribution over time
- Adjustable frequency: Daily, Weekly, Monthly, Yearly
- Interactive hover for detailed information

### Engagement by Sentiment
- Bar chart showing average Reddit score per sentiment
- Identifies which sentiments resonate most with community

## 🔧 Configuration

### Streamlit Settings

Edit `.streamlit/config.toml` to customize:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
```

### Color Palette

Colors are defined in `Dashboard.py`:
```python
color_map = {
    'Anger': '#C0392B',          # Deep red
    'Joy': '#F39C12',            # Warm amber
    'Neutral/Other': '#7F8C8D',  # Slate gray
    'Sadness': '#2980B9',        # Professional blue
    'Surprise': '#8E44AD'        # Muted purple
}
```

## 🎮 Use Cases

### 1. Track Game Updates
Monitor community sentiment before and after major updates:
- Filter to specific date ranges
- Compare sentiment distribution
- Identify positive vs negative reactions

### 2. Competitive Analysis
Track sentiment around specific events:
- New brawler releases
- Balance changes
- Seasonal events

### 3. Community Health Monitoring
- Identify increasing frustration (Anger trending up)
- Celebrate community joy (Joy spikes)
- Spot unusual patterns (Surprise surges)

## 🔄 How the System Works

```
New Reddit Posts
    ↓
[07_predict_new_posts.py]
    ├─→ Scrape posts from r/BrawlStars
    ├─→ Download media (images/videos)
    ├─→ Run through trained models:
    │    ├─ Text Model (DistilBERT)
    │    ├─ Image Model (CLIP)
    │    ├─ Video Model (CLIP + Temporal Attention)
    │    └─ Fusion Model
    ↓
Predicted Sentiment
    ↓
Append to sentiment_labelled.csv
    ↓
[Dashboard.py]
    └─→ Auto-refreshes with new data
```

## 📈 Data Format

`sentiment_labelled.csv` contains:
- `id`: Reddit post ID
- `title`: Post title
- `text`: Post content
- `score`: Reddit upvotes
- `created_utc`: Timestamp (Unix format)
- `post_sentiment`: Predicted sentiment (Anger/Joy/Neutral/Sadness/Surprise)
- `local_media_path`: Path to downloaded media

## 🎯 Best Practices

1. **Daily Updates**: Run predictions daily to keep data fresh
2. **Archive Old Data**: Periodically backup `sentiment_labelled.csv`
3. **Monitor Logs**: Check `logs/daily_update.log` for errors
4. **Disk Space**: Monitor `media/` folder size (grows with new posts)
5. **API Limits**: Respect Reddit API rate limits (max 60 requests/min)

## 🐛 Troubleshooting

### Dashboard won't load
```bash
# Check if streamlit is installed
pip install streamlit

# Try running with verbose output
streamlit run Dashboard.py --logger.level debug
```

### No data showing
- Check if `sentiment_labelled.csv` exists and has data
- Verify file path in `Dashboard.py` line 16
- Check date filters aren't excluding all data

### Prediction script fails
- Ensure `.env` file has Reddit API credentials
- Check if trained models exist in `models/` directory
- Verify CUDA/GPU availability if using GPU

### Daily scheduler not running
- **Linux**: Check cron logs: `grep CRON /var/log/syslog`
- **Windows**: Check Task Scheduler History tab
- Verify Python path is correct in scheduled task

## 📞 Support

For issues or questions:
1. Check the main project README
2. Review training notebooks (Steps 0-6)
3. Inspect log files in `logs/` directory

## 🎉 Production Deployment

For production use:
1. **Set up a server** (e.g., AWS EC2, DigitalOcean)
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Configure environment**: Copy `.env.example` to `.env`
4. **Schedule daily updates**: Use cron or systemd timer
5. **Deploy dashboard**: Use `streamlit run` or containerize with Docker
6. **Set up monitoring**: Use tools like PM2 or supervisor

---

**Built with ❤️ using Streamlit, PyTorch, and CLIP**
