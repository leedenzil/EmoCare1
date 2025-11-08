import streamlit as st
import pandas as pd
import plotly.express as px

def render_sentiment_trend(df, color_map, freq_option="Monthly"):


    title_col = st.columns([1])[0]

    with title_col:
        st.markdown(
            "<h3 style='font-size:25px; font-weight:700; margin-bottom:0;'>Sentiment Trend</h3>",
            unsafe_allow_html=True,
        )

    # ensure datetime column exists and is valid
    df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')
    df = df.dropna(subset=['created_utc'])

    # Use the dataframe as-is (already filtered by dashboard)
    trend_df = df.copy()

    trend_df = trend_df

  
    if freq_option == "Weekly":
        trend_df['period'] = trend_df['created_utc'].dt.to_period('W').apply(lambda r: r.start_time)
    elif freq_option == "Monthly":
        trend_df['period'] = trend_df['created_utc'].dt.to_period('M').apply(lambda r: r.start_time)
    elif freq_option == "Yearly":
        trend_df['period'] = trend_df['created_utc'].dt.to_period('Y').apply(lambda r: r.start_time)
    else:
        trend_df['period'] = trend_df['created_utc'].dt.to_period('D').apply(lambda r: r.start_time)


    trend_counts = (
        trend_df.groupby(['period', 'post_sentiment'])
                .size()
                .reset_index(name='count')
    )

    if trend_counts.empty:
        st.warning("No data available for the selected range.")
        return

    trend_counts['percent'] = (
        trend_counts.groupby('period')['count']
                    .transform(lambda x: x / x.sum() * 100)
    )

    if freq_option == "Daily":
        trend_counts['period_label'] = trend_counts['period'].dt.strftime('%b %d, %Y')  # e.g. "Nov 07, 2025"
    elif freq_option == "Weekly":
        trend_counts['period_label'] = trend_counts['period'].dt.strftime('Week of %b %d, %Y')  # "Week of Nov 01, 2025"
    elif freq_option == "Monthly":
        trend_counts['period_label'] = trend_counts['period'].dt.strftime('%B %Y')  # "November 2025"
    elif freq_option == "Yearly":
        trend_counts['period_label'] = trend_counts['period'].dt.strftime('%Y')  # "2025"


    fig = px.line(
        trend_counts,
        x='period',
        y='percent',
        color='post_sentiment',
        color_discrete_map=color_map,
        markers=True,
        custom_data=['period_label', 'count', 'post_sentiment']  # Include data for hover template
    )
    
    hovertemp = """
    Date: %{customdata[0]}<br>
    Distribution: %{y:.1f}%<br>
    Frequency: %{customdata[1]}
    """
    
    fig.update_traces(
        hovertemplate=hovertemp,
        marker=dict(size=8),  # Adjust marker size
        line=dict(width=2)    # Adjust line width
    )
    
    fig.update_layout(
        template='plotly_dark',
        hovermode='closest',
        legend_title_text='Sentiment',
        xaxis_title='Date',
        yaxis_title='Percentage Of Post (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, l=30, r=30, b=40), 
        font=dict(size=14)
    )

    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': False  
    })
    
    return None
