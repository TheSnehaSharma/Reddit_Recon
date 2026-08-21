import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from databricks import sql
import os
from datetime import datetime, timedelta
import calendar
# Optional wordcloud import - app works without it
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None
    plt = None

import numpy as np

# ================================
# PAGE CONFIGURATION
# ================================

st.set_page_config(
    page_title="Reddit Recon | Analytics Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================
# REDDIT THEME & STYLING
# ================================

REDDIT_ORANGE = "#FF4500"
REDDIT_BLUE = "#0079D3"
REDDIT_DARK = "#1A1A1B"
REDDIT_LIGHT = "#DAE0E6"

st.markdown("""
<style>
    .main {
        background-color: #DAE0E6;
    }
    
    .navbar {
        background: linear-gradient(90deg, #FF4500 0%, #FF6B35 100%);
        padding: 1rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .navbar-content {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .navbar-logo {
        font-size: 2rem;
    }
    
    .navbar-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    .section-header {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #FF4500;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    
    .section-header h2 {
        color: #1A1A1B;
        margin: 0;
        font-size: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        background-color: transparent;
        border-radius: 4px;
        color: #1A1A1B;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF4500;
        color: white;
    }
    
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #DAE0E6;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .dataframe th {
        background-color: #FF4500 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px !important;
    }
    
    .stButton button {
        background-color: #FF4500;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    .stButton button:hover {
        background-color: #FF6B35;
    }
</style>
""", unsafe_allow_html=True)

# Save for next chunk
print("Part 1/3 written")

# ================================
# DATABASE CONNECTION
# ================================

@st.cache_resource
def get_databricks_connection():
    """Establish connection to Databricks SQL Warehouse."""
    try:
        return sql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_TOKEN")
        )
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        return None

@st.cache_data(ttl=300)
def run_query(query):
    """Execute SQL query and return results as DataFrame."""
    conn = get_databricks_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            cursor.close()
            return pd.DataFrame(result, columns=columns)
        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

# ================================
# DATA FETCHING FUNCTIONS
# ================================

def get_available_dates():
    """Get available date range."""
    query = """
    SELECT 
        MIN(DATE(created_at)) as min_date,
        MAX(DATE(created_at)) as max_date
    FROM workspace.Reddit_Recon.posts_bronze
    """
    df = run_query(query)
    if not df.empty:
        return df.iloc[0]['min_date'], df.iloc[0]['max_date']
    return None, None

def get_available_subreddits():
    """Get list of subreddits."""
    query = """
    SELECT DISTINCT subreddit
    FROM workspace.Reddit_Recon.posts_bronze
    WHERE subreddit IS NOT NULL
    ORDER BY subreddit
    """
    df = run_query(query)
    return df['subreddit'].tolist() if not df.empty else []

def check_gold_layer():
    """Check if Gold layer exists."""
    query = "SHOW TABLES IN workspace.Reddit_Recon LIKE 'posts_gold'"
    df = run_query(query)
    return not df.empty

def get_kpi_metrics(start_date, end_date, subreddits):
    """Get KPI metrics."""
    sub_filter = ""
    if subreddits:
        subs = "','".join(subreddits)
        sub_filter = f"AND subreddit IN ('{subs}')"
    
    query = f"""
    SELECT 
        COUNT(*) as total_posts,
        SUM(score) as total_score,
        AVG(score) as avg_score,
        SUM(num_comments) as total_comments,
        AVG(num_comments) as avg_comments,
        COUNT(DISTINCT subreddit) as unique_subreddits,
        COUNT(DISTINCT author) as unique_authors
    FROM workspace.Reddit_Recon.posts_bronze
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {sub_filter}
    """
    return run_query(query)

def get_top_subreddits(start_date, end_date, limit=10):
    """Get top subreddits by posts and score."""
    query = f"""
    SELECT 
        subreddit,
        COUNT(*) as post_count,
        SUM(score) as total_score,
        AVG(score) as avg_score
    FROM workspace.Reddit_Recon.posts_bronze
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY subreddit
    ORDER BY total_score DESC
    LIMIT {limit}
    """
    return run_query(query)

def get_daily_trends(start_date, end_date, subreddits):
    """Get daily trends."""
    sub_filter = ""
    if subreddits:
        subs = "','".join(subreddits)
        sub_filter = f"AND subreddit IN ('{subs}')"
    
    query = f"""
    SELECT 
        DATE(created_at) as date,
        COUNT(*) as posts,
        SUM(score) as total_score,
        AVG(score) as avg_score,
        SUM(num_comments) as comments
    FROM workspace.Reddit_Recon.posts_bronze
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {sub_filter}
    GROUP BY DATE(created_at)
    ORDER BY date
    """
    return run_query(query)

def get_sentiment_data(start_date, end_date, subreddits):
    """Get sentiment analysis data."""
    sub_filter = ""
    if subreddits:
        subs = "','".join(subreddits)
        sub_filter = f"AND subreddit IN ('{subs}')"
    
    query = f"""
    SELECT 
        sentiment,
        emotion,
        topic,
        subreddit,
        COUNT(*) as count
    FROM workspace.Reddit_Recon.posts_gold
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {sub_filter}
    GROUP BY sentiment, emotion, topic, subreddit
    """
    return run_query(query)

def get_subreddit_sentiment_summary(start_date, end_date):
    """Get dominant sentiment and emotion per subreddit."""
    query = f"""
    WITH sentiment_counts AS (
        SELECT 
            subreddit,
            sentiment,
            COUNT(*) as sent_count,
            ROW_NUMBER() OVER (PARTITION BY subreddit ORDER BY COUNT(*) DESC) as sent_rank
        FROM workspace.Reddit_Recon.posts_gold
        WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY subreddit, sentiment
    ),
    emotion_counts AS (
        SELECT 
            subreddit,
            emotion,
            COUNT(*) as emo_count,
            ROW_NUMBER() OVER (PARTITION BY subreddit ORDER BY COUNT(*) DESC) as emo_rank
        FROM workspace.Reddit_Recon.posts_gold
        WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY subreddit, emotion
    )
    SELECT 
        s.subreddit,
        s.sentiment as dominant_sentiment,
        s.sent_count as sentiment_count,
        e.emotion as dominant_emotion,
        e.emo_count as emotion_count
    FROM sentiment_counts s
    JOIN emotion_counts e ON s.subreddit = e.subreddit
    WHERE s.sent_rank = 1 AND e.emo_rank = 1
    ORDER BY s.sent_count DESC
    """
    return run_query(query)

def get_text_for_wordcloud(start_date, end_date, subreddit=None):
    """Get text data for word cloud."""
    sub_filter = f"AND subreddit = '{subreddit}'" if subreddit else ""
    
    query = f"""
    SELECT title, selftext
    FROM workspace.Reddit_Recon.posts_bronze
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {sub_filter}
    LIMIT 1000
    """
    return run_query(query)

def get_raw_data(start_date, end_date, subreddits, sentiments, emotions, topics, sort_by, use_gold):
    """Get raw data with filters."""
    table = "workspace.Reddit_Recon.posts_gold" if use_gold else "workspace.Reddit_Recon.posts_bronze"
    
    filters = [f"DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'"]
    
    if subreddits:
        subs = "','".join(subreddits)
        filters.append(f"subreddit IN ('{subs}')")
    
    if use_gold:
        if sentiments:
            sents = "','".join(sentiments)
            filters.append(f"sentiment IN ('{sents}')")
        if emotions:
            emos = "','".join(emotions)
            filters.append(f"emotion IN ('{emos}')")
        if topics:
            tops = "','".join(topics)
            filters.append(f"topic IN ('{tops}')")
    
    where_clause = " AND ".join(filters)
    
    gold_cols = ", sentiment, emotion, topic" if use_gold else ""
    
    sort_mapping = {
        "Time (Newest)": "created_at DESC",
        "Time (Oldest)": "created_at ASC",
        "Score (High to Low)": "score DESC",
        "Score (Low to High)": "score ASC",
        "Comments (High to Low)": "num_comments DESC",
        "Comments (Low to High)": "num_comments ASC"
    }
    
    query = f"""
    SELECT 
        id,
        created_at,
        subreddit,
        author,
        title,
        score,
        num_comments,
        url
        {gold_cols}
    FROM {table}
    WHERE {where_clause}
    ORDER BY {sort_mapping.get(sort_by, 'created_at DESC')}
    LIMIT 500
    """
    return run_query(query)

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def create_wordcloud(text_data, title):
    """Create word cloud visualization."""
    if text_data.empty:
        return None
    
    text = " ".join(text_data['title'].fillna('') + " " + text_data['selftext'].fillna(''))
    
    if not text.strip():
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='Oranges',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig


# ================================
# MAIN APP
# ================================

def main():
    # Navbar with Reddit logo
    st.markdown("""
    <div class="navbar">
        <div class="navbar-content">
            <span class="navbar-logo">🔍</span>
            <h1 class="navbar-title">Reddit_Recon</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check connection
    if not get_databricks_connection():
        st.stop()
    
    # Check Gold layer
    gold_available = check_gold_layer()
    
    # Get date range
    min_date, max_date = get_available_dates()
    
    if not min_date or not max_date:
        st.warning("⚠️ No data available. Please run the Bronze Layer ETL pipeline.")
        st.stop()
    
    # Convert to datetime
    if isinstance(min_date, str):
        min_date = pd.to_datetime(min_date)
    if isinstance(max_date, str):
        max_date = pd.to_datetime(max_date)
    
    # Get all subreddits
    all_subreddits = get_available_subreddits()
    
    # Tabs for three sections
    tab1, tab2, tab3 = st.tabs(["📊 KPIs & Metrics", "🎭 Sentiment Analysis", "📋 Raw Data"])
    
    # ================================
    # TAB 1: KPIs & METRICS
    # ================================
    
    with tab1:
        st.markdown('<div class="section-header"><h2>Key Performance Indicators</h2></div>', unsafe_allow_html=True)
        
        # Filters
        col1, col2 = st.columns([2, 1])
        
        with col1:
            date_option = st.radio(
                "📅 Date Range:",
                ["Last 7 Days", "Last 30 Days", "Custom Range", "Specific Month", "All Time"],
                horizontal=True
            )
            
            if date_option == "Last 7 Days":
                start_date = max_date - timedelta(days=7)
                end_date = max_date
            elif date_option == "Last 30 Days":
                start_date = max_date - timedelta(days=30)
                end_date = max_date
            elif date_option == "Custom Range":
                col_a, col_b = st.columns(2)
                with col_a:
                    start_date = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
                with col_b:
                    end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
            elif date_option == "Specific Month":
                col_a, col_b = st.columns(2)
                with col_a:
                    year_options = list(range(min_date.year, max_date.year + 1))
                    selected_year = st.selectbox("Year", year_options, index=len(year_options)-1)
                with col_b:
                    selected_month = st.selectbox("Month", range(1, 13), 
                                                 format_func=lambda x: calendar.month_name[x])
                start_date = datetime(selected_year, selected_month, 1).date()
                last_day = calendar.monthrange(selected_year, selected_month)[1]
                end_date = datetime(selected_year, selected_month, last_day).date()
            else:  # All Time
                start_date = min_date
                end_date = max_date
        
        with col2:
            selected_subreddits = st.multiselect(
                "🏷️ Filter Subreddits:",
                all_subreddits,
                default=[]
            )
        
        # Fetch KPI data
        kpi_data = get_kpi_metrics(start_date, end_date, selected_subreddits)
        
        if not kpi_data.empty:
            kpi = kpi_data.iloc[0]
            
            # Display KPIs
            st.markdown("### 📈 Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📝 Total Posts", f"{int(kpi['total_posts']):,}")
                st.metric("👥 Unique Authors", f"{int(kpi['unique_authors']):,}")
            
            with col2:
                st.metric("⬆️ Total Score", f"{int(kpi['total_score']):,}")
                st.metric("📊 Avg Score", f"{int(kpi['avg_score']):,}")
            
            with col3:
                st.metric("💬 Total Comments", f"{int(kpi['total_comments']):,}")
                st.metric("📈 Avg Comments", f"{int(kpi['avg_comments']):,}")
            
            with col4:
                st.metric("🏷️ Subreddits", f"{int(kpi['unique_subreddits']):,}")
                engagement = (kpi['total_comments'] / kpi['total_posts']) if kpi['total_posts'] > 0 else 0
                st.metric("🔥 Engagement", f"{engagement:.1f}")
            
            st.markdown("---")
            
            # Daily trends
            st.markdown("### 📅 Daily Trends")
            daily_data = get_daily_trends(start_date, end_date, selected_subreddits)
            
            if not daily_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.line(daily_data, x='date', y='posts', 
                                 title='Daily Post Volume',
                                 labels={'posts': 'Number of Posts', 'date': 'Date'})
                    fig.update_traces(line_color=REDDIT_ORANGE, line_width=3)
                    fig.update_layout(hovermode='x unified', plot_bgcolor='white')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.line(daily_data, x='date', y='total_score',
                                 title='Daily Total Score',
                                 labels={'total_score': 'Total Score', 'date': 'Date'})
                    fig.update_traces(line_color=REDDIT_BLUE, line_width=3)
                    fig.update_layout(hovermode='x unified', plot_bgcolor='white')
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Top subreddits
            st.markdown("### 🏆 Top 10 Subreddits")
            top_subs = get_top_subreddits(start_date, end_date, 10)
            
            if not top_subs.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(top_subs, x='post_count', y='subreddit',
                                orientation='h',
                                title='By Post Count',
                                labels={'post_count': 'Number of Posts', 'subreddit': ''})
                    fig.update_traces(marker_color=REDDIT_ORANGE)
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, plot_bgcolor='white')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(top_subs, x='total_score', y='subreddit',
                                orientation='h',
                                title='By Total Score',
                                labels={'total_score': 'Total Score', 'subreddit': ''})
                    fig.update_traces(marker_color=REDDIT_BLUE)
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, plot_bgcolor='white')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No data available for selected filters.")

    
    # ================================
    # TAB 2: SENTIMENT ANALYSIS
    # ================================
    
    with tab2:
        if not gold_available:
            st.warning("⚠️ Gold Layer not available. Run the Gold Layer pipeline to enable sentiment analysis.")
            st.info("💡 The Gold Layer adds AI-powered sentiment, emotion, and topic analysis to your data.")
            st.stop()
        
        st.markdown('<div class="section-header"><h2>AI-Powered Sentiment Analysis</h2></div>', unsafe_allow_html=True)
        
        # Use same date filters from tab1
        sentiment_data = get_sentiment_data(start_date, end_date, selected_subreddits)
        
        if sentiment_data.empty:
            st.info("ℹ️ No sentiment data available for selected filters.")
        else:
            # Overall distributions
            st.markdown("### 📊 Overall Distribution")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Sentiment distribution
                sent_dist = sentiment_data.groupby('sentiment')['count'].sum().reset_index()
                colors_map = {'positive': '#46d160', 'negative': '#ff4757', 'neutral': '#747d8c'}
                fig = px.pie(sent_dist, values='count', names='sentiment',
                            title='Sentiment Distribution',
                            color='sentiment',
                            color_discrete_map=colors_map)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Emotion distribution
                emo_dist = sentiment_data.groupby('emotion')['count'].sum().reset_index().nlargest(7, 'count')
                fig = px.bar(emo_dist, x='emotion', y='count',
                            title='Top Emotions',
                            labels={'count': 'Count', 'emotion': 'Emotion'})
                fig.update_traces(marker_color=REDDIT_ORANGE)
                fig.update_layout(plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                # Topic distribution
                topic_dist = sentiment_data.groupby('topic')['count'].sum().reset_index().nlargest(10, 'count')
                fig = px.bar(topic_dist, x='topic', y='count',
                            title='Top Topics',
                            labels={'count': 'Count', 'topic': 'Topic'})
                fig.update_traces(marker_color=REDDIT_BLUE)
                fig.update_layout(plot_bgcolor='white', xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Subreddit-level insights
            st.markdown("### 🎯 Subreddit-Level Insights")
            
            subreddit_summary = get_subreddit_sentiment_summary(start_date, end_date)
            
            if not subreddit_summary.empty:
                # Display table
                st.markdown("#### Dominant Sentiment & Emotion by Subreddit")
                display_df = subreddit_summary[['subreddit', 'dominant_sentiment', 'dominant_emotion', 'sentiment_count', 'emotion_count']]
                display_df.columns = ['Subreddit', 'Dominant Sentiment', 'Dominant Emotion', 'Sentiment Count', 'Emotion Count']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Sentiment vs Topic/Emotion correlations
            st.markdown("### 🔗 Correlations & Relationships")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment vs Topic
                st.markdown("#### Sentiment by Topic")
                sent_topic = sentiment_data.groupby(['topic', 'sentiment'])['count'].sum().reset_index()
                sent_topic_top = sent_topic[sent_topic['topic'].isin(
                    sent_topic.groupby('topic')['count'].sum().nlargest(10).index
                )]
                
                fig = px.bar(sent_topic_top, x='topic', y='count', color='sentiment',
                            barmode='stack',
                            labels={'count': 'Count', 'topic': 'Topic'},
                            color_discrete_map=colors_map)
                fig.update_layout(plot_bgcolor='white', xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sentiment vs Emotion
                st.markdown("#### Sentiment by Emotion")
                sent_emo = sentiment_data.groupby(['emotion', 'sentiment'])['count'].sum().reset_index()
                
                fig = px.bar(sent_emo, x='emotion', y='count', color='sentiment',
                            barmode='stack',
                            labels={'count': 'Count', 'emotion': 'Emotion'},
                            color_discrete_map=colors_map)
                fig.update_layout(plot_bgcolor='white', xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Word clouds
            st.markdown("### ☁️ Word Clouds")
            
            cloud_option = st.radio("Select view:", ["Overall", "By Subreddit"], horizontal=True)
            
            if cloud_option == "Overall":
                text_data = get_text_for_wordcloud(start_date, end_date)
                if not text_data.empty:
                    fig = create_wordcloud(text_data, "Overall Reddit Word Cloud")
                    if fig:
                        st.pyplot(fig)
                else:
                    st.info("No text data available for word cloud")
            else:
                # By subreddit
                top_subs_list = get_top_subreddits(start_date, end_date, 6)
                if not top_subs_list.empty:
                    cols = st.columns(2)
                    for idx, (_, row) in enumerate(top_subs_list.iterrows()):
                        sub = row['subreddit']
                        text_data = get_text_for_wordcloud(start_date, end_date, sub)
                        if not text_data.empty:
                            fig = create_wordcloud(text_data, f"r/{sub}")
                            if fig:
                                with cols[idx % 2]:
                                    st.pyplot(fig)
    
    # ================================
    # TAB 3: RAW DATA
    # ================================
    
    with tab3:
        st.markdown('<div class="section-header"><h2>Raw Data Explorer</h2></div>', unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sort_by = st.selectbox(
                "Sort by:",
                ["Time (Newest)", "Time (Oldest)", "Score (High to Low)", 
                 "Score (Low to High)", "Comments (High to Low)", "Comments (Low to High)"]
            )
        
        with col2:
            raw_subreddits = st.multiselect(
                "Subreddits:",
                all_subreddits,
                default=[]
            )
        
        if gold_available:
            with col3:
                sentiments = st.multiselect(
                    "Sentiments:",
                    ["positive", "negative", "neutral"],
                    default=[]
                )
            
            with col4:
                emotions = st.multiselect(
                    "Emotions:",
                    ["joy", "anger", "fear", "sadness", "surprise", "disgust", "neutral"],
                    default=[]
                )
            
            # Topics filter
            topics = st.multiselect(
                "Topics:",
                ["technology", "politics", "entertainment", "sports", "science", 
                 "business", "health", "education", "gaming", "news"],
                default=[]
            )
        else:
            sentiments = []
            emotions = []
            topics = []
        
        # Fetch data
        raw_data = get_raw_data(
            start_date, end_date, raw_subreddits, sentiments, 
            emotions, topics, sort_by, gold_available
        )
        
        if not raw_data.empty:
            st.markdown(f"### Showing {len(raw_data)} posts")
            
            # Format display
            display_cols = ['created_at', 'subreddit', 'title', 'author', 'score', 'num_comments']
            if gold_available:
                display_cols += ['sentiment', 'emotion', 'topic']
            
            display_df = raw_data[display_cols].copy()
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Download
            csv = raw_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv,
                file_name=f"reddit_data_{start_date}_{end_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("ℹ️ No data matches your filters.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7c7c7c; padding: 1rem;'>
        <p><strong>Reddit Recon Analytics</strong> | Powered by Databricks & Streamlit</p>
        <p style='font-size: 0.85rem;'>Data refreshes every 5 minutes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
