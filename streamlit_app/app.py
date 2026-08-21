import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from databricks import sql
import os
from datetime import datetime, timedelta
import calendar

# ================================
# REDDIT STYLING & CONFIGURATION
# ================================

# Reddit color scheme
REDDIT_ORANGE = "#FF4500"
REDDIT_BLUE = "#5f99cf"
REDDIT_DARK = "#1c1c1c"
REDDIT_LIGHT = "#f6f7f8"
REDDIT_UPVOTE = "#FF8b60"
REDDIT_DOWNVOTE = "#9494FF"

# Page configuration
st.set_page_config(
    page_title="Reddit Recon Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Reddit-style theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --reddit-orange: #FF4500;
        --reddit-blue: #5f99cf;
        --reddit-dark: #1c1c1c;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #FF4500 0%, #FF6B35 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: #ffffff;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #FF4500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    
    /* Subreddit pill */
    .subreddit-pill {
        background: #FF4500;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    /* Score badge */
    .score-badge {
        background: linear-gradient(135deg, #FF8b60 0%, #FF4500 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(255,69,0,0.3);
    }
    
    /* Sentiment badges */
    .sentiment-positive {
        background: #46d160;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
    }
    
    .sentiment-negative {
        background: #ff4757;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
    }
    
    .sentiment-neutral {
        background: #747d8c;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {  /* Sidebar */
        background-color: #f6f7f8;
    }
    
    /* Filter section */
    .filter-section {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Data table styling */
    .dataframe {
        border: none !important;
    }
    
    .dataframe th {
        background-color: #FF4500 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px !important;
    }
    
    .dataframe td {
        padding: 10px !important;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    /* Info box */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
        st.info("📝 Make sure to set environment variables: DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
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
            st.error(f"❌ Query execution failed: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

# ================================
# DATA FETCHING FUNCTIONS
# ================================

def get_available_dates():
    """Get available date range from bronze table."""
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
    """Get list of all subreddits."""
    query = """
    SELECT DISTINCT subreddit
    FROM workspace.Reddit_Recon.posts_bronze
    WHERE subreddit IS NOT NULL
    ORDER BY subreddit
    """
    df = run_query(query)
    return df['subreddit'].tolist() if not df.empty else []

def check_gold_layer_available():
    """Check if Gold layer (enriched data) is available."""
    query = """
    SHOW TABLES IN workspace.Reddit_Recon LIKE 'posts_gold'
    """
    df = run_query(query)
    return not df.empty

def get_filtered_data(start_date, end_date, selected_subreddits, use_gold=False):
    """Get filtered Reddit posts data."""
    table = "workspace.Reddit_Recon.posts_gold" if use_gold else "workspace.Reddit_Recon.posts_bronze"
    
    subreddit_filter = ""
    if selected_subreddits:
        subreddits_str = "','".join(selected_subreddits)
        subreddit_filter = f"AND subreddit IN ('{subreddits_str}')"
    
    sentiment_cols = ", sentiment, sentiment_score" if use_gold else ""
    emotion_cols = ", emotion, emotion_score" if use_gold else ""
    topic_cols = ", topic, topic_score" if use_gold else ""
    
    query = f"""
    SELECT 
        id,
        author,
        subreddit,
        title,
        selftext,
        score,
        num_comments,
        created_at,
        DATE(created_at) as post_date,
        url,
        link_flair_text
        {sentiment_cols}
        {emotion_cols}
        {topic_cols}
    FROM {table}
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {subreddit_filter}
    ORDER BY created_at DESC
    """
    return run_query(query)

def get_daily_metrics(start_date, end_date, selected_subreddits):
    """Get daily aggregated metrics."""
    subreddit_filter = ""
    if selected_subreddits:
        subreddits_str = "','".join(selected_subreddits)
        subreddit_filter = f"AND subreddit IN ('{subreddits_str}')"
    
    query = f"""
    SELECT 
        DATE(created_at) as post_date,
        COUNT(*) as post_count,
        SUM(score) as total_score,
        AVG(score) as avg_score,
        SUM(num_comments) as total_comments,
        AVG(num_comments) as avg_comments,
        COUNT(DISTINCT subreddit) as unique_subreddits,
        COUNT(DISTINCT author) as unique_authors
    FROM workspace.Reddit_Recon.posts_bronze
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {subreddit_filter}
    GROUP BY DATE(created_at)
    ORDER BY post_date
    """
    return run_query(query)

def get_subreddit_metrics(start_date, end_date, selected_subreddits):
    """Get metrics by subreddit."""
    subreddit_filter = ""
    if selected_subreddits:
        subreddits_str = "','".join(selected_subreddits)
        subreddit_filter = f"AND subreddit IN ('{subreddits_str}')"
    
    query = f"""
    SELECT 
        subreddit,
        COUNT(*) as post_count,
        SUM(score) as total_score,
        AVG(score) as avg_score,
        SUM(num_comments) as total_comments,
        MAX(score) as max_score
    FROM workspace.Reddit_Recon.posts_bronze
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {subreddit_filter}
    GROUP BY subreddit
    ORDER BY total_score DESC
    LIMIT 20
    """
    return run_query(query)

def get_sentiment_distribution(start_date, end_date, selected_subreddits):
    """Get sentiment distribution from Gold layer."""
    subreddit_filter = ""
    if selected_subreddits:
        subreddits_str = "','".join(selected_subreddits)
        subreddit_filter = f"AND subreddit IN ('{subreddits_str}')"
    
    query = f"""
    SELECT 
        sentiment,
        COUNT(*) as count,
        AVG(sentiment_score) as avg_confidence
    FROM workspace.Reddit_Recon.posts_gold
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {subreddit_filter}
    GROUP BY sentiment
    ORDER BY count DESC
    """
    return run_query(query)

def get_emotion_distribution(start_date, end_date, selected_subreddits):
    """Get emotion distribution from Gold layer."""
    subreddit_filter = ""
    if selected_subreddits:
        subreddits_str = "','".join(selected_subreddits)
        subreddit_filter = f"AND subreddit IN ('{subreddits_str}')"
    
    query = f"""
    SELECT 
        emotion,
        COUNT(*) as count,
        AVG(emotion_score) as avg_confidence
    FROM workspace.Reddit_Recon.posts_gold
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {subreddit_filter}
    GROUP BY emotion
    ORDER BY count DESC
    """
    return run_query(query)

def get_topic_distribution(start_date, end_date, selected_subreddits):
    """Get topic distribution from Gold layer."""
    subreddit_filter = ""
    if selected_subreddits:
        subreddits_str = "','".join(selected_subreddits)
        subreddit_filter = f"AND subreddit IN ('{subreddits_str}')"
    
    query = f"""
    SELECT 
        topic,
        COUNT(*) as count,
        AVG(topic_score) as avg_confidence
    FROM workspace.Reddit_Recon.posts_gold
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {subreddit_filter}
    GROUP BY topic
    ORDER BY count DESC
    LIMIT 15
    """
    return run_query(query)

# ================================
# MAIN APP
# ================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Reddit Recon Analytics</h1>
        <p>Professional Reddit Data Analysis Dashboard | Real-time Insights & Trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check database connection
    if not get_databricks_connection():
        st.stop()
    
    # Check if Gold layer is available
    gold_available = check_gold_layer_available()
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("## 🎛️ Filters")
        
        # Date range filter
        st.markdown("### 📅 Date Range")
        min_date, max_date = get_available_dates()
        
        if min_date and max_date:
            # Convert to datetime if they're strings
            if isinstance(min_date, str):
                min_date = pd.to_datetime(min_date)
            if isinstance(max_date, str):
                max_date = pd.to_datetime(max_date)
            
            date_option = st.radio(
                "Select period:",
                ["Last 7 Days", "Last 30 Days", "Custom Range", "All Time"]
            )
            
            if date_option == "Last 7 Days":
                start_date = max_date - timedelta(days=7)
                end_date = max_date
            elif date_option == "Last 30 Days":
                start_date = max_date - timedelta(days=30)
                end_date = max_date
            elif date_option == "Custom Range":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
                with col2:
                    end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
            else:  # All Time
                start_date = min_date
                end_date = max_date
            
            # Monthly filter option
            st.markdown("### 📆 Month Filter")
            use_month_filter = st.checkbox("Filter by specific month")
            
            if use_month_filter:
                year_options = list(range(min_date.year, max_date.year + 1))
                selected_year = st.selectbox("Year", year_options, index=len(year_options)-1)
                selected_month = st.selectbox("Month", range(1, 13), 
                                             format_func=lambda x: calendar.month_name[x])
                
                # Set date range to selected month
                start_date = datetime(selected_year, selected_month, 1).date()
                last_day = calendar.monthrange(selected_year, selected_month)[1]
                end_date = datetime(selected_year, selected_month, last_day).date()
        else:
            st.warning("⚠️ No data available yet. Please run the Bronze Layer ETL pipeline first.")
            st.stop()
        
        # Subreddit filter
        st.markdown("### 🏷️ Subreddits")
        available_subreddits = get_available_subreddits()
        
        if available_subreddits:
            subreddit_option = st.radio(
                "Select subreddits:",
                ["All Subreddits", "Select Specific"]
            )
            
            if subreddit_option == "Select Specific":
                selected_subreddits = st.multiselect(
                    "Choose subreddits:",
                    available_subreddits,
                    default=[]
                )
            else:
                selected_subreddits = []
        else:
            selected_subreddits = []
        
        # Analysis type
        st.markdown("### 📊 Analysis Type")
        if gold_available:
            analysis_type = st.radio(
                "Choose analysis level:",
                ["Basic Metrics", "Sentiment Analysis", "Emotion Analysis", "Topic Analysis", "All Insights"]
            )
        else:
            analysis_type = "Basic Metrics"
            st.info("ℹ️ Run Gold Layer pipeline to unlock AI-powered insights")
        
        # Refresh button
        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Main content area
    # Summary metrics
    st.markdown("## 📈 Key Metrics")
    
    df_daily = get_daily_metrics(start_date, end_date, selected_subreddits)
    
    if not df_daily.empty:
        total_posts = df_daily['post_count'].sum()
        total_score = df_daily['total_score'].sum()
        avg_score = df_daily['avg_score'].mean()
        total_comments = df_daily['total_comments'].sum()
        avg_comments = df_daily['avg_comments'].mean()
        unique_subreddits = df_daily['unique_subreddits'].max()
        unique_authors = df_daily['unique_authors'].sum()
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Total Posts", f"{total_posts:,}")
            st.metric("👥 Unique Authors", f"{int(unique_authors):,}")
        
        with col2:
            st.metric("⬆️ Total Score", f"{int(total_score):,}")
            st.metric("📊 Avg Score/Post", f"{int(avg_score):,}")
        
        with col3:
            st.metric("💬 Total Comments", f"{int(total_comments):,}")
            st.metric("📈 Avg Comments/Post", f"{int(avg_comments):,}")
        
        with col4:
            st.metric("🏷️ Unique Subreddits", f"{int(unique_subreddits):,}")
            engagement_rate = (total_comments / total_posts) if total_posts > 0 else 0
            st.metric("🔥 Engagement Rate", f"{engagement_rate:.1f}")
        
        st.markdown("---")
        
        # Daily trends
        st.markdown("## 📊 Daily Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Posts over time
            fig_posts = px.line(
                df_daily, 
                x='post_date', 
                y='post_count',
                title='Daily Post Volume',
                labels={'post_count': 'Number of Posts', 'post_date': 'Date'}
            )
            fig_posts.update_traces(line_color=REDDIT_ORANGE, line_width=3)
            fig_posts.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif"),
                hovermode='x unified'
            )
            st.plotly_chart(fig_posts, use_container_width=True)
        
        with col2:
            # Score trends
            fig_score = px.line(
                df_daily,
                x='post_date',
                y='avg_score',
                title='Average Post Score',
                labels={'avg_score': 'Average Score', 'post_date': 'Date'}
            )
            fig_score.update_traces(line_color=REDDIT_BLUE, line_width=3)
            fig_score.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif"),
                hovermode='x unified'
            )
            st.plotly_chart(fig_score, use_container_width=True)
        
        # Subreddit analysis
        st.markdown("## 🏆 Top Subreddits")
        
        df_subreddits = get_subreddit_metrics(start_date, end_date, selected_subreddits)
        
        if not df_subreddits.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Top subreddits by post count
                fig_sub_count = px.bar(
                    df_subreddits.head(10),
                    x='post_count',
                    y='subreddit',
                    orientation='h',
                    title='Top 10 Subreddits by Post Count',
                    labels={'post_count': 'Number of Posts', 'subreddit': 'Subreddit'}
                )
                fig_sub_count.update_traces(marker_color=REDDIT_ORANGE)
                fig_sub_count.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_sub_count, use_container_width=True)
            
            with col2:
                # Top subreddits by total score
                fig_sub_score = px.bar(
                    df_subreddits.head(10),
                    x='total_score',
                    y='subreddit',
                    orientation='h',
                    title='Top 10 Subreddits by Total Score',
                    labels={'total_score': 'Total Score', 'subreddit': 'Subreddit'}
                )
                fig_sub_score.update_traces(marker_color=REDDIT_BLUE)
                fig_sub_score.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_sub_score, use_container_width=True)
        
        # AI-powered insights (if Gold layer available)
        if gold_available and analysis_type != "Basic Metrics":
            st.markdown("---")
            st.markdown("## 🤖 AI-Powered Insights")
            
            # Sentiment Analysis
            if analysis_type in ["Sentiment Analysis", "All Insights"]:
                st.markdown("### 😊 Sentiment Distribution")
                df_sentiment = get_sentiment_distribution(start_date, end_date, selected_subreddits)
                
                if not df_sentiment.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment pie chart
                        colors = {'positive': '#46d160', 'negative': '#ff4757', 'neutral': '#747d8c'}
                        fig_sentiment = px.pie(
                            df_sentiment,
                            values='count',
                            names='sentiment',
                            title='Sentiment Distribution',
                            color='sentiment',
                            color_discrete_map=colors
                        )
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                    with col2:
                        # Sentiment confidence
                        fig_conf = px.bar(
                            df_sentiment,
                            x='sentiment',
                            y='avg_confidence',
                            title='Average Sentiment Confidence',
                            labels={'avg_confidence': 'Confidence Score', 'sentiment': 'Sentiment'}
                        )
                        fig_conf.update_traces(marker_color=REDDIT_ORANGE)
                        st.plotly_chart(fig_conf, use_container_width=True)
            
            # Emotion Analysis
            if analysis_type in ["Emotion Analysis", "All Insights"]:
                st.markdown("### 😮 Emotion Distribution")
                df_emotion = get_emotion_distribution(start_date, end_date, selected_subreddits)
                
                if not df_emotion.empty:
                    # Emotion bar chart
                    emotion_colors = {
                        'joy': '#feca57',
                        'anger': '#ff4757',
                        'fear': '#5f27cd',
                        'sadness': '#48dbfb',
                        'surprise': '#ff9ff3',
                        'disgust': '#00d2d3',
                        'neutral': '#747d8c'
                    }
                    
                    fig_emotion = px.bar(
                        df_emotion,
                        x='emotion',
                        y='count',
                        title='Emotion Distribution Across Posts',
                        labels={'count': 'Number of Posts', 'emotion': 'Emotion'},
                        color='emotion',
                        color_discrete_map=emotion_colors
                    )
                    fig_emotion.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        showlegend=False
                    )
                    st.plotly_chart(fig_emotion, use_container_width=True)
            
            # Topic Analysis
            if analysis_type in ["Topic Analysis", "All Insights"]:
                st.markdown("### 📌 Topic Distribution")
                df_topic = get_topic_distribution(start_date, end_date, selected_subreddits)
                
                if not df_topic.empty:
                    # Topic treemap
                    fig_topic = px.treemap(
                        df_topic,
                        path=['topic'],
                        values='count',
                        title='Topic Distribution (Top 15)',
                        color='count',
                        color_continuous_scale='Oranges'
                    )
                    st.plotly_chart(fig_topic, use_container_width=True)
        
        # Recent posts table
        st.markdown("---")
        st.markdown("## 📄 Recent Posts")
        
        df_posts = get_filtered_data(start_date, end_date, selected_subreddits, use_gold=gold_available)
        
        if not df_posts.empty:
            # Display options
            show_count = st.slider("Number of posts to display:", 10, 100, 25)
            
            # Format and display table
            display_df = df_posts.head(show_count)[[
                'post_date', 'subreddit', 'title', 'author', 'score', 'num_comments'
            ]].copy()
            
            display_df['score'] = display_df['score'].apply(lambda x: f"⬆️ {x:,}")
            display_df['num_comments'] = display_df['num_comments'].apply(lambda x: f"💬 {x:,}")
            display_df.columns = ['Date', 'Subreddit', 'Title', 'Author', 'Score', 'Comments']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Title": st.column_config.TextColumn(
                        "Title",
                        width="large"
                    )
                }
            )
            
            # Download button
            csv = df_posts.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset (CSV)",
                data=csv,
                file_name=f"reddit_data_{start_date}_{end_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("ℹ️ No posts found for the selected filters.")
    else:
        st.warning("⚠️ No data available for the selected date range.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Reddit Recon Analytics</strong> | Powered by Databricks & Streamlit</p>
        <p style='font-size: 0.9rem;'>Data refreshes automatically every 5 minutes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
