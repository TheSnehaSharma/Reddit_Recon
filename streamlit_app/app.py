"""
Reddit Conversation Intelligence Dashboard
A professional social listening and conversation analytics platform

Pages:
1. Overview
2. Sentiment Analysis
3. Topic Intelligence
4. Emotion Analysis
5. Engagement Analysis
6. Predictive Analytics
7. Anomaly Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_loader import get_filter_options, load_main_data, get_previous_period_data
from metrics import (format_number, calculate_kpis, calculate_topic_velocity, 
                     detect_anomalies, calculate_engagement_percentiles, 
                     get_posting_time_heatmap_data, get_sentiment_label)
from charts import (create_volume_trend, create_sentiment_over_time, create_top_topics_chart,
                    create_sentiment_heatmap, create_emotion_distribution, create_engagement_by_topic,
                    create_time_heatmap, create_scatter_plot, create_box_plot, create_anomaly_chart)

# ================================
# PAGE CONFIGURATION
# ================================

st.set_page_config(
    page_title="Reddit Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# PROFESSIONAL STYLING
# ================================

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .section-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 500;
    }
    
    .stButton button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    
    .chart-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    .dataframe {
        font-size: 0.9rem;
    }
    
    .dataframe th {
        background-color: #34495e !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #3498db;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        font-weight: 500;
    }
    
    .kpi-change {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .kpi-change.positive {
        color: #27ae60;
    }
    
    .kpi-change.negative {
        color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)


# ================================
# SIDEBAR NAVIGATION
# ================================

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem 0;'>
        <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🎯</div>
        <div style='font-size: 1.8rem; font-weight: 700; color: white;'>REDDIT INTELLIGENCE</div>
        <div style='font-size: 0.85rem; color: rgba(255,255,255,0.8); margin-top: 0.5rem;'>Conversation Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<div style='color: white; font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem;'>NAVIGATION</div>", unsafe_allow_html=True)
    
    # Initialize page state
    if 'page' not in st.session_state:
        st.session_state.page = 'overview'
    
    # Navigation buttons
    pages = {
        'overview': '📊 Overview',
        'sentiment': '💭 Sentiment Analysis',
        'topics': '📚 Topic Intelligence',
        'emotions': '🎭 Emotion Analysis',
        'engagement': '🔥 Engagement Analysis',
        'predictive': '🤖 Predictive Analytics',
        'anomaly': '⚠️ Anomaly Detection'
    }
    
    for page_key, page_name in pages.items():
        if st.button(page_name, key=f"nav_{page_key}"):
            st.session_state.page = page_key
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("<div style='color: white; font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem;'>FILTERS</div>", unsafe_allow_html=True)
    
    # Get filter options
    filters = get_filter_options()
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=filters['max_date'] - timedelta(days=30),
            min_value=filters['min_date'],
            max_value=filters['max_date']
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=filters['max_date'],
            min_value=filters['min_date'],
            max_value=filters['max_date']
        )
    
    # Subreddit filter
    selected_subreddits = st.multiselect(
        "Subreddits",
        options=filters['subreddits'],
        default=[]
    )
    
    # Topic filter
    selected_topics = st.multiselect(
        "Topics",
        options=filters['topics'],
        default=[]
    )
    
    # Sentiment filter
    selected_sentiments = st.multiselect(
        "Sentiment",
        options=['Positive', 'Neutral', 'Negative'],
        default=[]
    )
    
    # Emotion filter
    selected_emotions = st.multiselect(
        "Emotions",
        options=filters['emotions'],
        default=[]
    )
    
    # Reset button
    if st.button("🔄 Reset Filters"):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("""
    <div style='color: rgba(255,255,255,0.7); font-size: 0.8rem; padding: 1rem 0;'>
        <strong>Data Source:</strong> Reddit Archive<br>
        <strong>Update:</strong> Daily<br>
        <strong>Cache:</strong> 1 hour
    </div>
    """, unsafe_allow_html=True)

# ================================
# LOAD DATA
# ================================

# Load current period data
df = load_main_data(
    start_date, end_date,
    selected_subreddits if selected_subreddits else None,
    selected_topics if selected_topics else None,
    selected_sentiments if selected_sentiments else None,
    selected_emotions if selected_emotions else None
)

# Load previous period data for comparisons
df_prev = get_previous_period_data(
    start_date, end_date,
    selected_subreddits if selected_subreddits else None,
    selected_topics if selected_topics else None,
    selected_sentiments if selected_sentiments else None,
    selected_emotions if selected_emotions else None
)

# Check if data is loaded
if df.empty:
    st.warning("⚠️ No data available for the selected filters. Please adjust your filters and try again.")
    st.stop()

# Calculate KPIs
kpis = calculate_kpis(df, df_prev)


# ================================
# PAGE: OVERVIEW
# ================================

if st.session_state.page == 'overview':
    st.title("📊 Overview Dashboard")
    st.markdown("*Executive summary of Reddit conversation intelligence*")
    
    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "TOTAL POSTS",
            format_number(kpis['total_posts']),
            f"{kpis['posts_change']:+.1f}%" if kpis['posts_change'] != 0 else None
        )
    
    with col2:
        sentiment_text = get_sentiment_label(kpis['avg_sentiment'])
        sentiment_color = "🟢" if sentiment_text == "Positive" else "🔴" if sentiment_text == "Negative" else "🟡"
        st.metric(
            "AVERAGE SENTIMENT",
            f"{sentiment_color} {kpis['avg_sentiment']:.2f}",
            sentiment_text
        )
    
    with col3:
        st.metric(
            "TOTAL ENGAGEMENT",
            format_number(kpis['total_engagement'])
        )
    
    with col4:
        st.metric(
            "TRENDING TOPIC",
            kpis['trending_topic'][:20] if kpis['trending_topic'] else "N/A",
            f"{kpis['topic_change']:+.0f}%" if kpis['topic_change'] != 0 else None
        )
    
    with col5:
        st.metric(
            "DOMINANT EMOTION",
            kpis['dominant_emotion'] if kpis['dominant_emotion'] else "N/A",
            f"{kpis['dominant_emotion_pct']:.0f}% of posts"
        )
    
    # Charts
    st.markdown("<div class='section-header'>Conversation Trends</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_volume = create_volume_trend(df)
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        fig_sentiment = create_sentiment_over_time(df)
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    st.markdown("<div class='section-header'>Topic Analysis</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_topics = create_top_topics_chart(df, 10)
        st.plotly_chart(fig_topics, use_container_width=True)
    
    with col2:
        st.markdown("### Key Insights")
        st.info(f"""
        **Data Overview:**
        * Analyzing {len(df):,} posts
        * From {df['created_date'].min().date()} to {df['created_date'].max().date()}
        * Covering {df['subreddit'].nunique()} subreddits
        * {df['author'].nunique():,} unique authors
        
        **Engagement:**
        * Avg score: {df['post_score'].mean():.1f}
        * Avg comments: {df['num_comments'].mean():.1f}
        * Avg engagement: {df['engagement_score'].mean():.2f}
        """)
    
    # Recent conversations table
    st.markdown("<div class='section-header'>Recent Conversations</div>", unsafe_allow_html=True)
    
    recent = df.head(100)[['created_date', 'subreddit', 'title', 'topic_label', 
                           'sentiment_label', 'emotion_label', 'post_score', 'num_comments']].copy()
    recent.columns = ['Date', 'Subreddit', 'Title', 'Topic', 'Sentiment', 'Emotion', 'Score', 'Comments']
    
    st.dataframe(
        recent,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Score": st.column_config.NumberColumn("Score", format="%d ⬆️"),
            "Comments": st.column_config.NumberColumn("Comments", format="%d 💬"),
        },
        use_container_width=True,
        height=400
    )


# ================================
# PAGE: SENTIMENT ANALYSIS
# ================================

elif st.session_state.page == 'sentiment':
    st.title("💭 Sentiment Analysis")
    st.markdown("*Understanding emotional polarity in conversations*")
    
    # Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_dist = df['sentiment_label'].value_counts().reset_index()
        sentiment_dist.columns = ['sentiment', 'count']
        sentiment_dist['percentage'] = (sentiment_dist['count'] / len(df) * 100).round(1)
        
        colors = {'Positive': '#27ae60', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
        
        fig_dist = px.bar(
            sentiment_dist.sort_values('count', ascending=False),
            x='count',
            y='sentiment',
            orientation='h',
            title='Sentiment Distribution',
            text='percentage',
            color='sentiment',
            color_discrete_map=colors
        )
        fig_dist.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_dist.update_layout(showlegend=False, height=350, template='plotly_white')
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Sentiment trend
        fig_trend = df.groupby('created_date').apply(
            lambda x: x['sentiment_score'].mean()
        ).reset_index(name='avg_sentiment')
        
        fig_sent_trend = px.line(
            fig_trend,
            x='created_date',
            y='avg_sentiment',
            title='Sentiment Trend Over Time',
            markers=True
        )
        fig_sent_trend.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
        fig_sent_trend.update_layout(height=350, template='plotly_white')
        st.plotly_chart(fig_sent_trend, use_container_width=True)
    
    # Sentiment by Topic
    st.markdown("<div class='section-header'>Sentiment by Topic</div>", unsafe_allow_html=True)
    
    fig_heatmap = create_sentiment_heatmap(df)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Sentiment vs Engagement
    st.markdown("<div class='section-header'>Sentiment Impact on Engagement</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_box = create_box_plot(df, 'sentiment_label', 'engagement_score', 
                                  'Engagement Distribution by Sentiment')
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Average metrics by sentiment
        sent_metrics = df.groupby('sentiment_label').agg({
            'post_score': 'mean',
            'num_comments': 'mean',
            'engagement_score': 'mean'
        }).round(2).reset_index()
        
        st.markdown("### Average Metrics by Sentiment")
        st.dataframe(sent_metrics, use_container_width=True, hide_index=True)
        
        # Insights
        st.success(f"""
        **Key Findings:**
        * Highest engagement: {sent_metrics.loc[sent_metrics['engagement_score'].idxmax(), 'sentiment_label']}
        * Most common: {sentiment_dist.iloc[0]['sentiment']} ({sentiment_dist.iloc[0]['percentage']:.1f}%)
        * Average sentiment score: {df['sentiment_score'].mean():.3f}
        """)

# ================================
# PAGE: TOPIC INTELLIGENCE
# ================================

elif st.session_state.page == 'topics':
    st.title("📚 Topic Intelligence")
    st.markdown("*Discovering what people are talking about*")
    
    # Topic ranking
    st.markdown("<div class='section-header'>Topic Ranking</div>", unsafe_allow_html=True)
    
    topic_stats = df.groupby('topic_label').agg({
        'post_id': 'count',
        'engagement_score': 'mean',
        'post_score': 'sum'
    }).reset_index()
    topic_stats.columns = ['Topic', 'Post Count', 'Avg Engagement', 'Total Score']
    topic_stats['Percentage'] = (topic_stats['Post Count'] / len(df) * 100).round(1)
    topic_stats = topic_stats.sort_values('Post Count', ascending=False).head(15)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_topics = create_top_topics_chart(df, 15)
        st.plotly_chart(fig_topics, use_container_width=True)
    
    with col2:
        st.markdown("### Top 10 Topics")
        st.dataframe(
            topic_stats.head(10)[['Topic', 'Post Count', 'Percentage']],
            use_container_width=True,
            hide_index=True
        )
    
    # Topic evolution
    st.markdown("<div class='section-header'>Topic Evolution</div>", unsafe_allow_html=True)
    
    # Get top 5 topics
    top_5_topics = df['topic_label'].value_counts().head(5).index.tolist()
    topic_time = df[df['topic_label'].isin(top_5_topics)].groupby(['created_date', 'topic_label']).size().reset_index(name='count')
    
    fig_evolution = px.line(
        topic_time,
        x='created_date',
        y='count',
        color='topic_label',
        title='Top 5 Topics Over Time',
        markers=True
    )
    fig_evolution.update_layout(height=400, template='plotly_white', hovermode='x unified')
    st.plotly_chart(fig_evolution, use_container_width=True)
    
    # Topic velocity
    st.markdown("<div class='section-header'>🔥 Trending Topics (Topic Velocity)</div>", unsafe_allow_html=True)
    
    if not df_prev.empty:
        velocity_df = calculate_topic_velocity(df, df_prev)
        velocity_df = velocity_df.head(15)
        
        fig_velocity = px.bar(
            velocity_df.sort_values('velocity'),
            x='velocity',
            y='topic_label',
            orientation='h',
            title='Topic Growth Rate (%)',
            color='velocity',
            color_continuous_scale='RdYlGn',
            text='velocity'
        )
        fig_velocity.update_traces(texttemplate='%{text:+.0f}%', textposition='outside')
        fig_velocity.update_layout(height=500, showlegend=False, template='plotly_white')
        st.plotly_chart(fig_velocity, use_container_width=True)
    else:
        st.info("Topic velocity requires data from a previous period for comparison.")
    
    # Topic sentiment
    st.markdown("<div class='section-header'>Topic Sentiment Comparison</div>", unsafe_allow_html=True)
    
    topic_sentiment = df.groupby(['topic_label', 'sentiment_label']).size().unstack(fill_value=0)
    topic_sentiment_pct = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0) * 100
    top_topics_sent = df['topic_label'].value_counts().head(10).index
    topic_sentiment_pct = topic_sentiment_pct.loc[top_topics_sent].round(1)
    
    fig_topic_sent = px.bar(
        topic_sentiment_pct.reset_index().melt(id_vars='topic_label'),
        x='value',
        y='topic_label',
        color='sentiment_label',
        orientation='h',
        title='Sentiment Distribution by Topic (%)',
        color_discrete_map={'Positive': '#27ae60', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'},
        barmode='stack'
    )
    fig_topic_sent.update_layout(height=400, template='plotly_white')
    st.plotly_chart(fig_topic_sent, use_container_width=True)


# ================================
# PAGE: EMOTION ANALYSIS
# ================================

elif st.session_state.page == 'emotions':
    st.title("🎭 Emotion Analysis")
    st.markdown("*Understanding emotional responses in conversations*")
    
    # Emotion distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_emotion_dist = create_emotion_distribution(df)
        st.plotly_chart(fig_emotion_dist, use_container_width=True)
    
    with col2:
        # Emotion over time (top 5)
        top_5_emotions = df['emotion_label'].value_counts().head(5).index.tolist()
        emotion_time = df[df['emotion_label'].isin(top_5_emotions)].groupby(['created_date', 'emotion_label']).size().reset_index(name='count')
        emotion_time_pct = emotion_time.groupby('created_date').apply(
            lambda x: x.assign(percentage=x['count'] / x['count'].sum() * 100)
        ).reset_index(drop=True)
        
        fig_emotion_time = px.area(
            emotion_time_pct,
            x='created_date',
            y='percentage',
            color='emotion_label',
            title='Emotion Distribution Over Time (Top 5, %)'
        )
        fig_emotion_time.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig_emotion_time, use_container_width=True)
    
    # Emotion vs Engagement
    st.markdown("<div class='section-header'>Emotion Impact on Engagement</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        emotion_engagement = df.groupby('emotion_label').agg({
            'engagement_score': 'mean',
            'post_score': 'mean',
            'num_comments': 'mean'
        }).sort_values('engagement_score', ascending=False).reset_index()
        
        fig_emotion_eng = px.bar(
            emotion_engagement,
            x='engagement_score',
            y='emotion_label',
            orientation='h',
            title='Average Engagement by Emotion',
            color='engagement_score',
            color_continuous_scale='Viridis'
        )
        fig_emotion_eng.update_layout(height=400, showlegend=False, template='plotly_white')
        st.plotly_chart(fig_emotion_eng, use_container_width=True)
    
    with col2:
        st.markdown("### Metrics by Emotion")
        st.dataframe(
            emotion_engagement.round(2),
            column_config={
                "emotion_label": "Emotion",
                "engagement_score": "Avg Engagement",
                "post_score": "Avg Score",
                "num_comments": "Avg Comments"
            },
            use_container_width=True,
            hide_index=True
        )
    
    # Emotion by Topic
    st.markdown("<div class='section-header'>Emotion by Topic</div>", unsafe_allow_html=True)
    
    # Create emotion-topic heatmap
    top_topics_list = df['topic_label'].value_counts().head(10).index
    emotion_topic = df[df['topic_label'].isin(top_topics_list)].groupby(['topic_label', 'emotion_label']).size().unstack(fill_value=0)
    emotion_topic_pct = emotion_topic.div(emotion_topic.sum(axis=1), axis=0) * 100
    
    fig_emotion_topic = go.Figure(data=go.Heatmap(
        z=emotion_topic_pct.values,
        x=emotion_topic_pct.columns,
        y=emotion_topic_pct.index,
        colorscale='Reds',
        text=emotion_topic_pct.values.round(1),
        texttemplate='%{text}%',
        textfont={"size": 9},
        colorbar=dict(title="Percentage")
    ))
    
    fig_emotion_topic.update_layout(
        title='Dominant Emotions by Topic (%)',
        xaxis_title='Emotion',
        yaxis_title='Topic',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_emotion_topic, use_container_width=True)

# ================================
# PAGE: ENGAGEMENT ANALYSIS
# ================================

elif st.session_state.page == 'engagement':
    st.title("🔥 Engagement Analysis")
    st.markdown("*Understanding what drives Reddit interaction*")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AVG POST SCORE", f"{df['post_score'].mean():.1f}")
    
    with col2:
        st.metric("AVG COMMENTS", f"{df['num_comments'].mean():.1f}")
    
    with col3:
        st.metric("AVG ENGAGEMENT", f"{df['engagement_score'].mean():.2f}")
    
    with col4:
        highest_eng_topic = df.groupby('topic_label')['engagement_score'].mean().idxmax()
        st.metric("TOP TOPIC", highest_eng_topic[:15])
    
    # Topic engagement ranking
    st.markdown("<div class='section-header'>Topic Engagement Ranking</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_topic_eng = create_engagement_by_topic(df, 10)
        st.plotly_chart(fig_topic_eng, use_container_width=True)
    
    with col2:
        topic_eng_table = df.groupby('topic_label').agg({
            'engagement_score': 'mean',
            'post_id': 'count'
        }).sort_values('engagement_score', ascending=False).head(10).round(2).reset_index()
        topic_eng_table.columns = ['Topic', 'Avg Engagement', 'Posts']
        
        st.markdown("### Top 10 by Engagement")
        st.dataframe(topic_eng_table, use_container_width=True, hide_index=True)
    
    # Best time to post
    st.markdown("<div class='section-header'>Best Time to Post</div>", unsafe_allow_html=True)
    
    pivot_data = get_posting_time_heatmap_data(df)
    fig_time_heatmap = create_time_heatmap(pivot_data, 'Engagement Score')
    st.plotly_chart(fig_time_heatmap, use_container_width=True)
    
    st.info("""
    **How to read this heatmap:**
    * Darker colors = Higher average engagement
    * Each cell shows the average engagement score for posts made during that day/hour
    * Use this to identify optimal posting times
    """)
    
    # Text length vs Engagement
    st.markdown("<div class='section-header'>Content Analysis</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_length_eng = create_scatter_plot(
            df, 
            'text_length', 
            'engagement_score',
            'Text Length vs Engagement',
            'sentiment_label'
        )
        st.plotly_chart(fig_length_eng, use_container_width=True)
    
    with col2:
        # Engagement by sentiment
        fig_sent_eng = create_box_plot(df, 'sentiment_label', 'engagement_score',
                                       'Engagement by Sentiment')
        st.plotly_chart(fig_sent_eng, use_container_width=True)
    
    # Top performing posts
    st.markdown("<div class='section-header'>Top Performing Posts</div>", unsafe_allow_html=True)
    
    top_posts = df.nlargest(20, 'engagement_score')[
        ['created_date', 'subreddit', 'title', 'topic_label', 'sentiment_label', 
         'emotion_label', 'post_score', 'num_comments', 'engagement_score']
    ].copy()
    top_posts.columns = ['Date', 'Subreddit', 'Title', 'Topic', 'Sentiment', 'Emotion', 
                         'Score', 'Comments', 'Engagement']
    top_posts['Engagement'] = top_posts['Engagement'].round(2)
    
    st.dataframe(top_posts, use_container_width=True, hide_index=True, height=400)


# ================================
# PAGE: PREDICTIVE ANALYTICS
# ================================

elif st.session_state.page == 'predictive':
    st.title("🤖 Predictive Analytics")
    st.markdown("*Machine learning insights for engagement prediction*")
    
    st.info("""
    This page demonstrates how NLP features can predict Reddit engagement.
    In a production environment, this would include trained ML models.
    """)
    
    # Prepare data for ML
    df_ml, engagement_threshold = calculate_engagement_percentiles(df)
    
    st.markdown(f"**High Engagement Threshold:** {engagement_threshold:.2f} (Top 25%)")
    
    # Feature importance (simulated)
    st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Calculate correlation with engagement
        feature_importance = pd.DataFrame({
            'Feature': ['Topic', 'Posting Hour', 'Emotion', 'Text Length', 'Sentiment Score', 
                       'Day of Week', 'Subreddit', 'Emotion Confidence', 'Topic Confidence'],
            'Importance': [0.28, 0.19, 0.15, 0.13, 0.10, 0.08, 0.04, 0.02, 0.01]
        })
        
        fig_importance = px.bar(
            feature_importance.sort_values('Importance'),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance for Engagement Prediction',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(height=400, showlegend=False, template='plotly_white')
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.markdown("### Model Performance (Simulated)")
        
        model_metrics = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'Accuracy': [0.71, 0.76, 0.81],
            'F1 Score': [0.68, 0.74, 0.79],
            'ROC-AUC': [0.73, 0.80, 0.86]
        })
        
        st.dataframe(model_metrics, use_container_width=True, hide_index=True)
        
        st.success("**Best Model:** XGBoost with 81% accuracy")
    
    # Model comparison chart
    st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)
    
    model_comp = model_metrics.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    fig_model_comp = px.bar(
        model_comp,
        x='Score',
        y='Model',
        color='Metric',
        orientation='h',
        title='Model Performance Metrics',
        barmode='group'
    )
    fig_model_comp.update_layout(height=350, template='plotly_white')
    st.plotly_chart(fig_model_comp, use_container_width=True)
    
    # Engagement prediction simulator
    st.markdown("<div class='section-header'>Engagement Prediction Simulator</div>", unsafe_allow_html=True)
    
    st.markdown("**Predict engagement for hypothetical posts:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sim_topic = st.selectbox("Topic", df['topic_label'].unique()[:10])
        sim_sentiment = st.selectbox("Sentiment", ['Positive', 'Neutral', 'Negative'])
        sim_emotion = st.selectbox("Emotion", df['emotion_label'].unique()[:7])
    
    with col2:
        sim_hour = st.slider("Posting Hour", 0, 23, 12)
        sim_day = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                               'Friday', 'Saturday', 'Sunday'])
        sim_length = st.slider("Text Length", 10, 1000, 200)
    
    with col3:
        if st.button("🎯 Predict Engagement", use_container_width=True):
            # Simulated prediction
            base_score = 0.5
            
            # Topic influence
            topic_eng = df[df['topic_label'] == sim_topic]['engagement_score'].mean()
            topic_norm = (topic_eng - df['engagement_score'].mean()) / df['engagement_score'].std()
            
            # Sentiment influence
            sent_mult = {'Positive': 1.1, 'Neutral': 1.0, 'Negative': 0.9}[sim_sentiment]
            
            # Hour influence (peak hours 18-22)
            hour_mult = 1.2 if 18 <= sim_hour <= 22 else 1.0
            
            # Calculate probability
            prob = min(0.95, max(0.05, base_score + topic_norm * 0.15 * sent_mult * hour_mult))
            
            engagement_class = "HIGH" if prob > 0.5 else "LOW"
            color = "green" if prob > 0.5 else "orange"
            
            st.markdown(f"""
            <div style='background-color: white; padding: 2rem; border-radius: 8px; 
                        border-left: 4px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h2 style='color: {color}; margin: 0;'>{engagement_class} ENGAGEMENT</h2>
                <h1 style='color: #2c3e50; margin: 0.5rem 0;'>{prob*100:.0f}%</h1>
                <p style='color: #7f8c8d; margin: 0;'>Probability of high engagement</p>
                
                <hr style='margin: 1.5rem 0;'>
                
                <h4 style='color: #2c3e50;'>Top Influencing Factors:</h4>
                <ul style='color: #555;'>
                    <li><strong>Topic:</strong> {sim_topic}</li>
                    <li><strong>Sentiment:</strong> {sim_sentiment}</li>
                    <li><strong>Posting Time:</strong> {sim_hour}:00 on {sim_day}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Engagement distribution
    st.markdown("<div class='section-header'>Engagement Distribution</div>", unsafe_allow_html=True)
    
    fig_eng_dist = px.histogram(
        df,
        x='engagement_score',
        nbins=50,
        title='Engagement Score Distribution',
        labels={'engagement_score': 'Engagement Score', 'count': 'Number of Posts'},
        color_discrete_sequence=['#3498db']
    )
    fig_eng_dist.add_vline(x=engagement_threshold, line_dash="dash", line_color="red", 
                           annotation_text=f"Top 25% Threshold ({engagement_threshold:.2f})")
    fig_eng_dist.update_layout(height=350, template='plotly_white')
    st.plotly_chart(fig_eng_dist, use_container_width=True)

# ================================
# PAGE: ANOMALY DETECTION
# ================================

elif st.session_state.page == 'anomaly':
    st.title("⚠️ Anomaly Detection")
    st.markdown("*Identifying unusual patterns in conversations*")
    
    # Daily aggregation for anomaly detection
    daily_data = df.groupby('created_date').size().reset_index(name='post_count')
    daily_data = detect_anomalies(daily_data, 'post_count', window=7, threshold=2.5)
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        anomaly_count = daily_data['is_anomaly'].sum()
        st.metric("ANOMALIES DETECTED", anomaly_count)
    
    with col2:
        if anomaly_count > 0:
            largest_spike = daily_data[daily_data['is_anomaly']]['post_count'].max()
            st.metric("LARGEST SPIKE", f"{largest_spike:.0f} posts")
        else:
            st.metric("LARGEST SPIKE", "N/A")
    
    with col3:
        # Sentiment shift
        recent_sentiment = df[df['created_date'] >= df['created_date'].max() - timedelta(days=3)]['sentiment_score'].mean()
        overall_sentiment = df['sentiment_score'].mean()
        sentiment_shift = recent_sentiment - overall_sentiment
        st.metric("SENTIMENT SHIFT", f"{sentiment_shift:+.3f}", 
                 "Recent 3 days vs average")
    
    with col4:
        alert_level = "HIGH" if anomaly_count > 3 else "MEDIUM" if anomaly_count > 0 else "LOW"
        alert_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[alert_level]
        st.metric("ALERT LEVEL", f"{alert_color} {alert_level}")
    
    # Volume anomalies
    st.markdown("<div class='section-header'>Conversation Volume Anomalies</div>", unsafe_allow_html=True)
    
    fig_anomaly = create_anomaly_chart(daily_data, 'created_date', 'post_count', 'is_anomaly')
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    if daily_data['is_anomaly'].any():
        st.warning(f"""
        **⚠️ {anomaly_count} anomalies detected in conversation volume.**
        
        Anomalies are identified using z-score analysis with a 7-day rolling window.
        Points with z-score > 2.5 are flagged as unusual.
        """)
    else:
        st.success("✅ No significant anomalies detected in conversation volume.")
    
    # Topic spike detection
    st.markdown("<div class='section-header'>Topic Spike Detection</div>", unsafe_allow_html=True)
    
    if not df_prev.empty:
        topic_velocity_df = calculate_topic_velocity(df, df_prev)
        topic_spikes = topic_velocity_df[topic_velocity_df['velocity'] > 100].sort_values('velocity', ascending=False)
        
        if not topic_spikes.empty:
            fig_spikes = px.bar(
                topic_spikes.head(10),
                x='velocity',
                y='topic_label',
                orientation='h',
                title='Topics with Significant Spikes (>100% growth)',
                color='velocity',
                color_continuous_scale='Reds',
                text='velocity'
            )
            fig_spikes.update_traces(texttemplate='%{text:+.0f}%', textposition='outside')
            fig_spikes.update_layout(height=400, showlegend=False, template='plotly_white')
            st.plotly_chart(fig_spikes, use_container_width=True)
            
            st.warning(f"🔥 **{len(topic_spikes)} topics** showing significant growth (>100%)")
        else:
            st.info("No significant topic spikes detected.")
    else:
        st.info("Topic spike detection requires data from a previous period.")
    
    # Negative sentiment spike
    st.markdown("<div class='section-header'>Negative Sentiment Analysis</div>", unsafe_allow_html=True)
    
    sentiment_daily = df.groupby('created_date').apply(
        lambda x: (x['sentiment_label'] == 'Negative').sum() / len(x) * 100
    ).reset_index(name='negative_pct')
    
    sentiment_daily['rolling_avg'] = sentiment_daily['negative_pct'].rolling(window=7, center=True).mean()
    threshold_pct = sentiment_daily['negative_pct'].mean() + sentiment_daily['negative_pct'].std()
    
    fig_negative = px.line(
        sentiment_daily,
        x='created_date',
        y='negative_pct',
        title='Negative Sentiment Over Time (%)',
        markers=True
    )
    fig_negative.add_hline(y=threshold_pct, line_dash="dash", line_color="red",
                          annotation_text=f"Alert Threshold ({threshold_pct:.1f}%)")
    fig_negative.update_layout(height=350, template='plotly_white')
    st.plotly_chart(fig_negative, use_container_width=True)
    
    # Anomalous posts table
    st.markdown("<div class='section-header'>Unusual Posts</div>", unsafe_allow_html=True)
    
    # Posts with extreme engagement
    extreme_posts = df[(df['engagement_score'] > df['engagement_score'].quantile(0.99)) |
                       (df['post_score'] > df['post_score'].quantile(0.99))].copy()
    
    if not extreme_posts.empty:
        extreme_display = extreme_posts[['created_date', 'subreddit', 'title', 'topic_label', 
                                        'sentiment_label', 'post_score', 'num_comments', 
                                        'engagement_score']].head(20)
        extreme_display.columns = ['Date', 'Subreddit', 'Title', 'Topic', 'Sentiment', 
                                  'Score', 'Comments', 'Engagement']
        
        st.dataframe(extreme_display, use_container_width=True, hide_index=True, height=400)
        st.info(f"Showing {len(extreme_display)} posts with exceptional engagement (top 1%)")
    else:
        st.info("No exceptional posts identified.")

# ================================
# FOOTER
# ================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
    <h3 style='color: #2c3e50;'>🎯 Reddit Conversation Intelligence Platform</h3>
    <p>Powered by Databricks Lakehouse • Streamlit • NLP AI Models</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        Real-time social listening and conversation analytics for strategic insights
    </p>
</div>
""", unsafe_allow_html=True)
