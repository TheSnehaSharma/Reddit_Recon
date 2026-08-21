import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from databricks import sql
import os
from datetime import datetime, timedelta

# ================================
# PAGE CONFIGURATION
# ================================

st.set_page_config(
    page_title="Reddit Recon | Sentiment Analytics",
    page_icon="\ue060",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# REDDIT THEME & STYLING
# ================================

REDDIT_ORANGE = "#FF4500"
REDDIT_BLUE = "#0079D3"
REDDIT_DARK = "#1A1A1B"

st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #F6F7F8;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FF4500 0%, #FF6B35 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] button {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);
        font-weight: 600;
        transition: all 0.3s;
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: rgba(255, 255, 255, 0.2);
        border-color: white;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #E5E7EB;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Filter section */
    .filter-section {
        background-color: white;
        padding: 1rem 1.15rem;
        border-radius: 8px;
        margin-bottom: 1.1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #FF4500;
    }
    
    /* Section headers - make them visible with white background */
    .section-header {
        color: #1A1A1B;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding: 0.8rem 1rem;
        background-color: white;
        border-left: 4px solid #FF4500;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton button {
        background-color: #FF4500;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #FF6B35;
        transform: translateY(-2px);
    }
    
    /* DataFrame styling */
    .dataframe th {
        background-color: #FF4500 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* About page content */
    .about-section {
        background-color: white;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    
    .about-section h2 {
        color: #FF4500;
        margin-bottom: 1rem;
    }
    
    .about-section h3 {
        color: #1A1A1B;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Footer link */
    .footer-link {
        color: #FF4500;
        text-decoration: none;
        font-weight: 600;
    }
    
    .footer-link:hover {
        color: #FF6B35;
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# DATABASE CONNECTION
# ================================

@st.cache_resource
def get_databricks_connection():
    try:
        return sql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_TOKEN")
        )
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        return None

@st.cache_data(ttl=86400)
def run_query(query):
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
    query = """
    SELECT 
        MIN(DATE(created_at)) as min_date,
        MAX(DATE(created_at)) as max_date
    FROM workspace.redditrecon.posts_silver
    """
    df = run_query(query)
    if not df.empty:
        return df.iloc[0]['min_date'], df.iloc[0]['max_date']
    return datetime.now().date() - timedelta(days=30), datetime.now().date()

def get_available_subreddits():
    query = """
    SELECT DISTINCT subreddit
    FROM workspace.redditrecon.posts_silver
    WHERE subreddit IS NOT NULL
    ORDER BY subreddit
    """
    df = run_query(query)
    return df['subreddit'].tolist() if not df.empty else []

def check_gold_layer():
    query = "SHOW TABLES IN workspace.redditrecon LIKE 'posts_gold'"
    df = run_query(query)
    return not df.empty

def get_kpi_metrics(start_date, end_date, subreddits):
    sub_filter = ""
    if subreddits and "All Subreddits" not in subreddits:
        subs = "','".join(subreddits)
        sub_filter = f"AND subreddit IN ('{subs}')"
    
    query = f"""
    SELECT 
        COUNT(*) as total_posts,
        SUM(score) as total_score,
        ROUND(AVG(score), 1) as avg_score,
        SUM(num_comments) as total_comments,
        ROUND(AVG(num_comments), 1) as avg_comments,
        COUNT(DISTINCT subreddit) as unique_subreddits,
        COUNT(DISTINCT author) as unique_authors
    FROM workspace.redditrecon.posts_silver
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {sub_filter}
    """
    return run_query(query)

def get_daily_trends(start_date, end_date, subreddits):
    sub_filter = ""
    if subreddits and "All Subreddits" not in subreddits:
        subs = "','".join(subreddits)
        sub_filter = f"AND subreddit IN ('{subs}')"
    
    query = f"""
    SELECT 
        DATE(created_at) as date,
        COUNT(*) as posts,
        SUM(score) as total_score,
        SUM(num_comments) as comments
    FROM workspace.redditrecon.posts_silver
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {sub_filter}
    GROUP BY DATE(created_at)
    ORDER BY date
    """
    return run_query(query)

def get_top_subreddits(start_date, end_date, limit=10):
    query = f"""
    SELECT 
        subreddit,
        COUNT(*) as post_count,
        SUM(score) as total_score,
        ROUND(AVG(score), 1) as avg_score
    FROM workspace.redditrecon.posts_silver
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY subreddit
    ORDER BY post_count DESC
    LIMIT {limit}
    """
    return run_query(query)

def get_sentiment_distribution(start_date, end_date, subreddits):
    sub_filter = ""
    if subreddits and "All Subreddits" not in subreddits:
        subs = "','".join(subreddits)
        sub_filter = f"AND subreddit IN ('{subs}')"
    
    query = f"""
    SELECT 
        sentiment,
        COUNT(*) as count
    FROM workspace.redditrecon.posts_gold
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {sub_filter}
    GROUP BY sentiment
    ORDER BY count DESC
    """
    return run_query(query)

def get_emotion_distribution(start_date, end_date, subreddits):
    sub_filter = ""
    if subreddits and "All Subreddits" not in subreddits:
        subs = "','".join(subreddits)
        sub_filter = f"AND subreddit IN ('{subs}')"
    
    query = f"""
    SELECT 
        emotion,
        COUNT(*) as count
    FROM workspace.redditrecon.posts_gold
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {sub_filter}
    GROUP BY emotion
    ORDER BY count DESC
    LIMIT 10
    """
    return run_query(query)

def get_topic_distribution(start_date, end_date, subreddits):
    sub_filter = ""
    if subreddits and "All Subreddits" not in subreddits:
        subs = "','".join(subreddits)
        sub_filter = f"AND subreddit IN ('{subs}')"
    
    query = f"""
    SELECT 
        topic,
        COUNT(*) as count
    FROM workspace.redditrecon.posts_gold
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {sub_filter}
    GROUP BY topic
    ORDER BY count DESC
    LIMIT 10
    """
    return run_query(query)

def get_sentiment_by_subreddit(start_date, end_date):
    query = f"""
    WITH sentiment_counts AS (
        SELECT 
            subreddit,
            sentiment,
            COUNT(*) as count,
            ROW_NUMBER() OVER (PARTITION BY subreddit ORDER BY COUNT(*) DESC) as rn
        FROM workspace.redditrecon.posts_gold
        WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY subreddit, sentiment
    )
    SELECT subreddit, sentiment, count
    FROM sentiment_counts
    WHERE rn = 1
    ORDER BY count DESC
    LIMIT 10
    """
    return run_query(query)

def get_raw_data(start_date, end_date, subreddits, limit=100):
    sub_filter = ""
    if subreddits and "All Subreddits" not in subreddits:
        subs = "','".join(subreddits)
        sub_filter = f"AND subreddit IN ('{subs}')"
    
    query = f"""
    SELECT 
        id,
        subreddit,
        author,
        title,
        score,
        num_comments,
        DATE(created_at) as post_date,
        url
    FROM workspace.redditrecon.posts_silver
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    {sub_filter}
    ORDER BY created_at DESC
    LIMIT {limit}
    """
    return run_query(query)

# ================================
# SIDEBAR
# ================================

with st.sidebar:
    st.markdown("""
<<<<<<< HEAD
    <div style='text-align: center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🔍</div>
        <div style='font-size: 1.8rem; font-weight: 700; color: white;'>Reddit Recon</div>
        <div style='font-size: 0.9rem; color: rgba(255,255,255,0.85); margin-top: 0.3rem;'>Sentiment Analytics Dashboard</div>
=======
    <div style='text-align: center; padding: 0.35rem 0 0.7rem 0;'>
        <div class='reddit-icon' style='font-size: 2.25rem; margin-bottom: 0.2rem;'>&#xe060;</div>
        <div style='font-size: 1.45rem; font-weight: 700; color: white;'>Reddit Recon</div>
        <div style='font-size: 0.78rem; color: rgba(255,255,255,0.8); margin-top: 0.15rem;'>Sentiment Analytics Dashboard</div>
>>>>>>> origin/main
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
<<<<<<< HEAD
    
    # Navigation
    if st.button("📊 KPIs & Metrics", key="nav_kpis", use_container_width=True):
        st.session_state.page = "kpis"
=======
    st.markdown("### Navigation")

    nav_items = [
        ("📊 KPIs & Metrics", "kpis"),
        ("💭 Sentiment Analysis", "sentiment"),
        ("📋 Raw Data Explorer", "raw"),
    ]

    if "page" not in st.session_state:
        st.session_state.page = "kpis"

    for label, key in nav_items:
        active = st.session_state.page == key
        button_class = "nav-active" if active else ""
        st.markdown(f"<div class='{button_class}'>", unsafe_allow_html=True)
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    page = {
        "kpis": "📊 KPIs & Metrics",
        "sentiment": "💭 Sentiment Analysis",
        "raw": "📋 Raw Data Explorer",
    }[st.session_state.page]

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Reddit Recon** analyzes Reddit posts with:

    🎯 **KPI Tracking**  
    Monitor posts, scores, comments

    💭 **Sentiment AI**  
    Positive, negative, neutral analysis

    🎨 **Emotion Detection**  
    7 emotion categories

    📚 **Topic Classification**  
    15 content categories
    """)

    st.markdown("---")
    st.markdown("**Data Refresh:** Daily at midnight UTC")
    st.markdown("**Cache TTL:** 24 hours")


# ================================
# MAIN AREA
# ================================

st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
st.markdown("### 🔧 Filters")

min_date, max_date = get_available_dates()

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    start_date = st.date_input(
        "📅 Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

with col2:
    end_date = st.date_input(
        "📅 End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

with col3:
    st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
>>>>>>> origin/main
        st.rerun()
    
    if st.button("💭 Sentiment Analysis", key="nav_sentiment", use_container_width=True):
        st.session_state.page = "sentiment"
        st.rerun()
    
    if st.button("📋 Raw Data Explorer", key="nav_raw", use_container_width=True):
        st.session_state.page = "raw"
        st.rerun()
    
    if st.button("ℹ️ About", key="nav_about", use_container_width=True):
        st.session_state.page = "about"
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("""
    <div style='color: rgba(255,255,255,0.85); font-size: 0.85rem; padding: 0.5rem 0;'>
        <strong>Data Refresh:</strong> Daily at midnight UTC<br>
        <strong>Cache TTL:</strong> 24 hours
    </div>
    """, unsafe_allow_html=True)

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "kpis"

page = st.session_state.page

# ================================
# PAGE ROUTING
# ================================

if page == "about":
    # About page
    st.markdown("""
    <div class='about-section'>
        <h2>🔍 About Reddit Recon</h2>
        <p style='font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem;'>
            A comprehensive Reddit sentiment analytics platform powered by AI and built on Databricks.
        </p>
        
        <h3>🎯 What We Do</h3>
        <p><strong>Reddit Recon</strong> analyzes Reddit posts using state-of-the-art AI models to extract insights from social media conversations. Our platform processes Reddit data through three analytical layers:</p>
        
        <ul style='line-height: 1.8;'>
            <li><strong>KPI Tracking:</strong> Monitor post volume, engagement rates, scores, and community metrics</li>
            <li><strong>Sentiment Analysis:</strong> Classify posts as positive, negative, or neutral using RoBERTa models</li>
            <li><strong>Emotion Detection:</strong> Identify 7 core emotions (joy, anger, fear, sadness, surprise, disgust, neutral)</li>
            <li><strong>Topic Classification:</strong> Categorize content across 15 topic areas using zero-shot learning</li>
        </ul>
        
        <h3>🤖 AI Models</h3>
        <p>We use cutting-edge transformer models from Hugging Face:</p>
        <ul style='line-height: 1.8;'>
            <li><strong>Sentiment:</strong> cardiffnlp/twitter-roberta-base-sentiment-latest</li>
            <li><strong>Emotion:</strong> j-hartmann/emotion-english-distilroberta-base</li>
            <li><strong>Topic:</strong> facebook/bart-large-mnli (zero-shot classification)</li>
        </ul>
        
        <h3>🏗️ Architecture</h3>
        <p>Built on <strong>Databricks Lakehouse Platform</strong> with a three-layer architecture:</p>
        <ul style='line-height: 1.8;'>
            <li><strong>Bronze Layer:</strong> Raw Reddit data ingestion from Archive Shift</li>
            <li><strong>Silver Layer:</strong> Data cleaning, deduplication, and transformation</li>
            <li><strong>Gold Layer:</strong> AI-enriched data with sentiment, emotion, and topic analysis</li>
        </ul>
        
        <h3>📊 Data Pipeline</h3>
        <ul style='line-height: 1.8;'>
            <li><strong>Source:</strong> Reddit posts from Archive Shift</li>
            <li><strong>Update Frequency:</strong> Daily at midnight UTC</li>
            <li><strong>Processing:</strong> Automated ETL pipeline with Databricks Jobs</li>
            <li><strong>Storage:</strong> Delta Lake tables with ACID transactions</li>
            <li><strong>Compute:</strong> Serverless Databricks SQL Warehouse</li>
        </ul>
        
        <h3>🛠️ Technology Stack</h3>
        <div style='background-color: #F6F7F8; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
            <strong>Data & ML:</strong> Databricks, Apache Spark, Delta Lake, MLflow<br>
            <strong>AI Models:</strong> Transformers, PyTorch, Hugging Face<br>
            <strong>Visualization:</strong> Streamlit, Plotly, Pandas<br>
            <strong>Version Control:</strong> Git, GitHub
        </div>
        
        <h3>👨‍💻 Created By</h3>
        <p>Developed by <strong>Sneha Sharma</strong> as a demonstration of end-to-end data engineering and ML on Databricks.</p>
        
        <h3>📝 License & Credits</h3>
        <p>This project uses public Reddit data and open-source AI models. All model credits go to their respective creators.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Main content pages with filters
    st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
    
    min_date, max_date = get_available_dates()
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        start_date = st.date_input(
            "📅 Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            label_visibility="visible"
        )
    
    with col2:
        end_date = st.date_input(
            "📅 End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            label_visibility="visible"
        )
    
    with col3:
        st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    available_subs = get_available_subreddits()
    selected_subs = st.multiselect(
        "🎯 Filter by Subreddit",
        options=["All Subreddits"] + available_subs,
        default=["All Subreddits"]
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    has_gold = check_gold_layer()
    
    # ================================
    # PAGE CONTENT
    # ================================
    
    if page == "kpis":
        st.markdown("<div class='section-header'>📊 Key Performance Indicators</div>", unsafe_allow_html=True)
        
        kpi_df = get_kpi_metrics(start_date, end_date, selected_subs)
        
        if not kpi_df.empty:
            row = kpi_df.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📝 Total Posts", f"{int(row['total_posts']):,}")
            with col2:
                st.metric("⬆️ Total Score", f"{int(row['total_score']):,}")
            with col3:
                st.metric("💬 Total Comments", f"{int(row['total_comments']):,}")
            with col4:
                engagement = (row['total_comments'] / row['total_posts']) if row['total_posts'] > 0 else 0
                st.metric("🔥 Engagement Rate", f"{engagement:.1f}")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📌 Unique Subreddits", int(row['unique_subreddits']))
            with col2:
                st.metric("👥 Unique Authors", f"{int(row['unique_authors']):,}")
            with col3:
                st.metric("⭐ Avg Score/Post", f"{row['avg_score']:.1f}")
            
            st.markdown("<div class='section-header'>📈 Daily Trends</div>", unsafe_allow_html=True)
            
            trends_df = get_daily_trends(start_date, end_date, selected_subs)
            
            if not trends_df.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=trends_df['date'],
                    y=trends_df['posts'],
                    mode='lines+markers',
                    name='Posts',
                    line=dict(color=REDDIT_ORANGE, width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Daily Post Volume",
                    xaxis_title="Date",
                    yaxis_title="Number of Posts",
                    height=400,
                    hovermode='x unified',
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_score = px.line(
                        trends_df,
                        x='date',
                        y='total_score',
                        title='Daily Total Score',
                        markers=True
                    )
                    fig_score.update_traces(line_color=REDDIT_BLUE)
                    fig_score.update_layout(height=350, plot_bgcolor='white')
                    st.plotly_chart(fig_score, use_container_width=True)
                
                with col2:
                    fig_comments = px.line(
                        trends_df,
                        x='date',
                        y='comments',
                        title='Daily Comments',
                        markers=True
                    )
                    fig_comments.update_traces(line_color='#10B981')
                    fig_comments.update_layout(height=350, plot_bgcolor='white')
                    st.plotly_chart(fig_comments, use_container_width=True)
            
            st.markdown("<div class='section-header'>🏆 Top 10 Subreddits</div>", unsafe_allow_html=True)
            
            top_subs = get_top_subreddits(start_date, end_date, 10)
            
            if not top_subs.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_posts = px.bar(
                        top_subs,
                        x='post_count',
                        y='subreddit',
                        orientation='h',
                        title='By Post Count',
                        color='post_count',
                        color_continuous_scale='Oranges'
                    )
                    fig_posts.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_posts, use_container_width=True)
                
                with col2:
                    fig_score = px.bar(
                        top_subs,
                        x='total_score',
                        y='subreddit',
                        orientation='h',
                        title='By Total Score',
                        color='total_score',
                        color_continuous_scale='Blues'
                    )
                    fig_score.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_score, use_container_width=True)
        else:
            st.warning("⚠️ No data available for the selected filters.")
    
    elif page == "sentiment":
        if not has_gold:
            st.warning("""
            ### 🔒 Gold Layer Not Available
            
            **Run the Gold Layer pipeline to enable sentiment analysis.**
            
            💡 **The Gold Layer adds:**
            - 😊 Sentiment classification (positive, negative, neutral)
            - 🎭 Emotion detection (7 categories)
            - 📚 Topic categorization (15 topics)
            
            Once the pipeline completes, this page will display rich sentiment insights!
            """)
        else:
            st.markdown("<div class='section-header'>💭 Sentiment Analysis Dashboard</div>", unsafe_allow_html=True)
            
            sent_df = get_sentiment_distribution(start_date, end_date, selected_subs)
            
            if not sent_df.empty:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    colors = {'positive': '#10B981', 'negative': '#EF4444', 'neutral': '#6B7280'}
                    sent_df['color'] = sent_df['sentiment'].map(colors)
                    
                    fig_sent = px.pie(
                        sent_df,
                        values='count',
                        names='sentiment',
                        title='Sentiment Distribution',
                        color='sentiment',
                        color_discrete_map=colors,
                        hole=0.4
                    )
                    fig_sent.update_layout(height=400)
                    st.plotly_chart(fig_sent, use_container_width=True)
                
                with col2:
                    sub_sent = get_sentiment_by_subreddit(start_date, end_date)
                    if not sub_sent.empty:
                        fig_sub = px.bar(
                            sub_sent,
                            x='count',
                            y='subreddit',
                            color='sentiment',
                            orientation='h',
                            title='Dominant Sentiment by Subreddit (Top 10)',
                            color_discrete_map=colors
                        )
                        fig_sub.update_layout(height=400)
                        st.plotly_chart(fig_sub, use_container_width=True)
                
                st.markdown("<div class='section-header'>🎭 Emotion & Topic Analysis</div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    emot_df = get_emotion_distribution(start_date, end_date, selected_subs)
                    if not emot_df.empty:
                        fig_emot = px.bar(
                            emot_df,
                            x='emotion',
                            y='count',
                            title='Top 10 Emotions',
                            color='count',
                            color_continuous_scale='Reds'
                        )
                        fig_emot.update_layout(height=350, showlegend=False)
                        st.plotly_chart(fig_emot, use_container_width=True)
                
                with col2:
                    topic_df = get_topic_distribution(start_date, end_date, selected_subs)
                    if not topic_df.empty:
                        fig_topic = px.bar(
                            topic_df,
                            x='topic',
                            y='count',
                            title='Top 10 Topics',
                            color='count',
                            color_continuous_scale='Purples'
                        )
                        fig_topic.update_layout(height=350, showlegend=False)
                        fig_topic.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_topic, use_container_width=True)
            else:
                st.warning("⚠️ No sentiment data available for the selected filters.")
    
    elif page == "raw":
        st.markdown("<div class='section-header'>📋 Raw Post Data Explorer</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 2])
        
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                ["Most Recent", "Highest Score", "Most Comments"]
            )
        
        with col2:
            limit = st.select_slider(
                "Number of records",
                options=[50, 100, 200, 500, 1000],
                value=100
            )
        
        raw_df = get_raw_data(start_date, end_date, selected_subs, limit)
        
        if not raw_df.empty:
            if sort_by == "Highest Score":
                raw_df = raw_df.sort_values('score', ascending=False)
            elif sort_by == "Most Comments":
                raw_df = raw_df.sort_values('num_comments', ascending=False)
            
            st.markdown(f"**Showing {len(raw_df):,} posts**")
            
            st.dataframe(
                raw_df,
                column_config={
                    "url": st.column_config.LinkColumn("URL"),
                    "score": st.column_config.NumberColumn("Score", format="%d ⬆️"),
                    "num_comments": st.column_config.NumberColumn("Comments", format="%d 💬"),
                },
                use_container_width=True,
                height=600
            )
            
            csv = raw_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"reddit_recon_{start_date}_{end_date}.csv",
                mime="text/csv",
            )
        else:
            st.warning("⚠️ No data available for the selected filters.")

# Footer
st.markdown("---")
st.markdown("""
<<<<<<< HEAD
<div style='text-align: center; color: #6B7280; padding: 2rem 0 1rem 0;'>
    <p style='font-size: 1.1rem;'><span style='font-size: 1.5rem;'>🔍</span> <strong style='color: #FF4500;'>Reddit Recon</strong> | Powered by Databricks & Streamlit</p>
    <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Real-time Reddit Analytics with AI-Powered Insights</p>
    <p style='margin-top: 1rem;'>
        <a href='https://github.com/TheSnehaSharma/Reddit_Recon' target='_blank' class='footer-link'>📁 View on GitHub</a>
    </p>
=======
<div style='text-align: center; color: #6B7280; padding: 2rem 0;'>
    <p><span class='reddit-icon'>&#xe060;</span> <strong>Reddit Recon</strong> | Powered by Databricks & Streamlit</p>
    <p style='font-size: 0.9rem;'>Real-time Reddit Analytics with AI-Powered Insights</p>
>>>>>>> origin/main
</div>
""", unsafe_allow_html=True)
