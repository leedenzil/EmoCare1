import streamlit as st
import pandas as pd
import plotly.express as px

def render_sentiment_engagement(df, color_map, start_date=None, end_date=None, selected_sentiments=None):
   
    st.markdown(
        "<h3 style='font-size:25px; font-weight:700; margin-bottom:0;'>Engagement by Sentiment</h3>",
        unsafe_allow_html=True
    )

    df = df.copy()
    if start_date and end_date:
        mask = (df['created_utc'].dt.date >= start_date) & (df['created_utc'].dt.date <= end_date)
        df = df.loc[mask]
    
    if selected_sentiments:
        df = df[df['post_sentiment'].isin(selected_sentiments)]

    df = df.copy()
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df.dropna(subset=['score', 'post_sentiment'])

    # Compute average Reddit score per sentiment
    engagement_df = (
        df.groupby('post_sentiment')['score']
          .mean()
          .reset_index()
          .sort_values('score', ascending=False)
    )

    category_order = engagement_df['post_sentiment'].tolist()

    # Create base figure (we'll suppress native x ticks and add custom annotations to avoid overlap at 100% zoom)
    fig = px.bar(
        engagement_df,
        x='post_sentiment',
        y='score',
        color='post_sentiment',
        color_discrete_map=color_map,
        text_auto='.2f',
        title='',
    )

    # Update traces with precise width and positioning
    fig.update_traces(
        textfont_size=16,
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Avg Score: %{y:.2f}<extra></extra>",
        marker=dict(line=dict(width=0)),
        width=0.6,  # Wider bars for better label alignment
        offset=0    
    )

    peak = engagement_df['score'].max()
    y_max = max(peak * 1.25, peak + 120)

    fig.update_layout(
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            title_text='Sentiment',
            yanchor="top",
            y=0.99,           
            xanchor="left",
            x=1.02,           
            font=dict(size=13)
        ),

        yaxis_title='Average Post Score',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, l=40, r=30, b=120),  
        font=dict(size=15),
        bargap=0.4,          
        uniformtext=dict(mode='hide', minsize=12),
        bargroupgap=0,     
        yaxis=dict(range=[0, y_max], tickfont=dict(size=13), titlefont=dict(size=14)),
        xaxis=dict(
            tickfont=dict(size=1),  
            title=dict(text='Sentiment', font=dict(size=14), standoff=70),
            tickangle=0,
            tickmode='array',
            ticktext=['' for _ in category_order],  # blank out native labels
            tickvals=list(range(len(category_order))),
            range=[-0.5, len(category_order)-0.5],
            constrain='domain',
            showgrid=False,
            automargin=False
        ),
        uniformtext_minsize=12,
        uniformtext_mode='hide'
    )

    for idx, lab in enumerate(category_order):
        display_lab = lab.replace('Neutral/Other', 'Neutral<br>/Other')
        fig.add_annotation(
            x=idx + 0.20,  # moderate right shift (tuned)
            y=-0.15,      # place above new axis title position
            xref='x',
            yref='paper',
            showarrow=False,
            text=f"<span style='font-size:13px'>{display_lab}</span>",
            align='center'
        )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
