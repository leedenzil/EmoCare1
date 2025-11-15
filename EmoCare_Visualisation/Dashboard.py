import streamlit as st
import pandas as pd
import plotly.express as px
import datetime as dt
import os
from sentiment_timeseries import render_sentiment_trend
from sentiment_engagement import render_sentiment_engagement

st.set_page_config(
    page_title="EmoCare1",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    """Load sentiment data, handling missing or empty CSV files."""
    # Get the directory where Dashboard.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "sentiment_labelled.csv")

    # Check if file exists
    if not os.path.exists(csv_path):
        # Create empty DataFrame with proper columns
        return pd.DataFrame(columns=[
            'id', 'title', 'text', 'url', 'permalink', 'score',
            'created_utc', 'post_hint', 'local_media_path', 'post_sentiment',
            'post_classification', 'sentiment_analysis', 'labeling_error'
        ])

    try:
        df = pd.read_csv(csv_path)
        # Handle empty CSV file
        if len(df) == 0:
            return pd.DataFrame(columns=[
                'id', 'title', 'text', 'url', 'permalink', 'score',
                'created_utc', 'post_hint', 'local_media_path', 'post_sentiment',
                'post_classification', 'sentiment_analysis', 'labeling_error'
            ])
        return df
    except pd.errors.EmptyDataError:
        # File exists but is completely empty
        return pd.DataFrame(columns=[
            'id', 'title', 'text', 'url', 'permalink', 'score',
            'created_utc', 'post_hint', 'local_media_path', 'post_sentiment',
            'post_classification', 'sentiment_analysis', 'labeling_error'
        ])

# load data
df = load_data()

# Check if we have data
if len(df) == 0:
    st.title("EmoCare1 - Sentiment Dashboard")
    st.warning("📊 No sentiment data available yet")
    st.info("""
    **Get started by collecting sentiment data:**

    Run the prediction script to scrape and analyze Reddit posts:
    ```
    python 07_predict_new_posts.py --num_posts 100
    ```

    Once data is collected, refresh this dashboard to see the visualizations.
    """)
    st.stop()

# Load CSS file (if exists)
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css("styles.css")

# Professional / muted color palette
color_map = {
    'Anger': '#C0392B',          # deep red
    'Joy': '#F39C12',            # warm amber
    'Neutral/Other': '#7F8C8D',  # slate gray
    'Sadness': '#2980B9',        # professional blue
    'Surprise': '#8E44AD'        # muted purple
}

emoji_map = {
    'Anger': '😡',
    'Joy': '😄',
    'Neutral/Other': '😐',
    'Sadness': '😢',
    'Surprise': '😱'
}

# Top title and small compare select
title_col, compare_col = st.columns([3, 1])
with title_col:
    st.markdown(
        "<h3 style='font-size:25px; font-weight:700; margin-bottom:0;'>Sentiment Distribution</h3>",
        unsafe_allow_html=True
    )

with compare_col:
    st.markdown("<div style='margin-top:-5px'></div>", unsafe_allow_html=True)
    compare_option = st.selectbox(
        "Compare Against",
        ("Day before", "Week before", "Month before", "Year before"),
        label_visibility="collapsed",
        key="compare_option"
    )

# Ensure datetime column
df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
df['date'] = df['created_utc'].dt.date

# summary cards (current vs previous)
latest_date = df['date'].max()
if compare_option == "Day before":
    compare_date = latest_date - dt.timedelta(days=1)
elif compare_option == "Week before":
    compare_date = latest_date - dt.timedelta(weeks=1)
elif compare_option == "Month before":
    compare_date = latest_date - dt.timedelta(days=30)
else:
    compare_date = latest_date - dt.timedelta(days=365)

current_df = df[df['date'] == latest_date]
previous_df = df[df['date'] == compare_date]

current_counts = current_df['post_sentiment'].value_counts(normalize=True) * 100
previous_counts = previous_df['post_sentiment'].value_counts(normalize=True) * 100

comparison = pd.DataFrame({
    'current': current_counts,
    'previous': previous_counts
}).fillna(0)

desired_order = ['Joy', 'Anger', 'Surprise', 'Sadness', 'Neutral/Other']
comparison = comparison.reindex(desired_order).dropna(how='all')
comparison['change'] = (comparison['current'] - comparison['previous']).round(1)

cols = st.columns(len(comparison))
for i, (label, row) in enumerate(comparison.iterrows()):
    pct = row['current']
    diff = row['change']
    sign = "▲" if diff > 0 else ("▼" if diff < 0 else "•")
    color_diff = "#2ecc71" if diff > 0 else ("#e74c3c" if diff < 0 else "#bdc3c7")

    col = cols[i]
    col_color = color_map.get(label, '#7f8c8d')
    emoji = emoji_map.get(label, '🔎')

    col.markdown(
        f"""
        <div class="sentiment-card">
            <div class="emoji">{emoji}</div>
            <div style="padding-top:6px;">
                <div class="sentiment-label">{label}</div>
                <div class="sentiment-value">{pct:.1f}%</div>
                <div style="font-size:16px; color:{color_diff}; margin-top:4px;">
                    {sign} {abs(diff):.1f}% since last {compare_option.lower().split()[0]}
                </div>
            </div>
            <div class="progress-bg">
                <div class="progress-fill" style="width:{pct:.1f}%; background:{col_color}; box-shadow: 0 6px 16px {col_color}22;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr>", unsafe_allow_html=True)

# --- Global filters row (above the two charts) ---
# place an empty spacer column to push filters to the right
spacer_col, start_col, end_col, freq_col, sent_col = st.columns([3, 1.2, 1.2, 0.9, 2])

min_date = df['created_utc'].dt.date.min()
max_date = df['created_utc'].dt.date.max()

with start_col:
    start_date = st.date_input(
        "Start",
        value=min_date, 
        key="start_date"
    )

with end_col:
    end_date = st.date_input(
        "End",
        value=max_date,  
        key="end_date"
    )
with sent_col:
    all_sentiments = sorted(df['post_sentiment'].dropna().unique().tolist())
    selected_sentiments = st.multiselect("Sentiments", options=all_sentiments, default=all_sentiments, key="global_sentiment_choice")

with freq_col:
    # Frequency is a visual control that only affects the Sentiment Trend chart
    # Place label visible so it lines up with Start/End/Sentiments
    freq_option = st.selectbox(
        "Frequency",
        ["Daily", "Weekly", "Monthly", "Yearly"],
        index=2,
        key="global_freq_option"
    )

# apply filters
mask = (
    (df['created_utc'].dt.date >= start_date) &
    (df['created_utc'].dt.date <= end_date) &
    (df['post_sentiment'].isin(selected_sentiments))
)
df_filtered = df.loc[mask]

# charts row
left_col, divider_col, right_col = st.columns([2.45, 0.05, 1.5])

with left_col:
    render_sentiment_trend(df_filtered, color_map, freq_option)

with divider_col:
    st.markdown(
        """
        <div style="
            border-left: 2px solid rgba(255, 255, 255, 0.2);
            height: 100%;
            margin-top: 30px;
            margin-bottom: 30px;">
        </div>
        """,
        unsafe_allow_html=True
    )

with right_col:
    render_sentiment_engagement(df_filtered, color_map)
            