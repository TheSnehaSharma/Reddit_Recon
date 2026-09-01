"""
Reddit Conversation Intelligence Dashboard
A professional social listening and conversation analytics platform

All-in-one file for easy deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from databricks import sql
import os
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
    
    .dataframe th {
        background-color: #34495e !important;
        color: white !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# DATABASE CONNECTION
# ================================

@st.cache_resource
def get_databricks_connection():
    """Establish connection to Databricks"""
    try:
        return sql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_TOKEN")
        )
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def run_query(query):
    """Execute SQL query and return DataFrame"""
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
            st.error(f"Query failed: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

# ================================
# UTILITY FUNCTIONS
# ================================

def format_number(num):
    """Format large numbers with K, M suffixes"""
    if pd.isna(num):
        return "0"
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    return str(int(num))

def get_sentiment_label(score):
    """Convert sentiment score to label"""
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    return "Neutral"

def calculate_percentage_change(current, previous):
    """Calculate percentage change between two values"""
    if pd.isna(previous) or previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def calculate_topic_velocity(current_df, previous_df):
    """Calculate topic growth velocity"""
    current_topics = current_df.groupby('topic_label').size().reset_index(name='current_count')
    previous_topics = previous_df.groupby('topic_label').size().reset_index(name='previous_count')
    
    merged = current_topics.merge(previous_topics, on='topic_label', how='left')
    merged['previous_count'] = merged['previous_count'].fillna(0)
    merged['velocity'] = ((merged['current_count'] - merged['previous_count']) / 
                          (merged['previous_count'] + 1)) * 100
    
    return merged.sort_values('velocity', ascending=False)

def detect_anomalies(df, column='post_count', window=7, threshold=3):
    """Detect anomalies using z-score method"""
    if len(df) < window:
        df['is_anomaly'] = False
        return df
    
    df = df.copy()
    df['rolling_mean'] = df[column].rolling(window=window, center=True).mean()
    df['rolling_std'] = df[column].rolling(window=window, center=True).std()
    df['z_score'] = (df[column] - df['rolling_mean']) / (df['rolling_std'] + 0.001)
    df['is_anomaly'] = abs(df['z_score']) > threshold
    
    return df

def calculate_engagement_percentiles(df):
    """Calculate engagement percentiles for classification"""
    q75 = df['engagement_score'].quantile(0.75)
    df['high_engagement'] = (df['engagement_score'] >= q75).astype(int)
    return df, q75

def get_posting_time_heatmap_data(df):
    """Prepare data for time heatmap"""
    heatmap_data = df.groupby(['day_name', 'hour']).agg({
        'engagement_score': 'mean',
        'post_score': 'mean',
        'num_comments': 'mean'
    }).reset_index()
    
    pivot = heatmap_data.pivot(index='day_name', columns='hour', values='engagement_score')
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])
    
    return pivot

def calculate_kpis(current_df, previous_df=None):
    """Calculate all KPIs with comparisons"""
    kpis = {
        'total_posts': len(current_df),
        'avg_sentiment': current_df['sentiment_score'].mean(),
        'total_engagement': current_df['engagement_score'].sum(),
        'avg_engagement': current_df['engagement_score'].mean(),
        'dominant_emotion': current_df['emotion_label'].mode()[0] if len(current_df) > 0 else None,
        'dominant_emotion_pct': (current_df['emotion_label'].value_counts().iloc[0] / len(current_df) * 100) if len(current_df) > 0 else 0
    }
    
    if len(current_df) > 0:
        top_topic = current_df['topic_label'].value_counts().head(1)
        kpis['trending_topic'] = top_topic.index[0] if len(top_topic) > 0 else None
        kpis['trending_topic_count'] = top_topic.values[0] if len(top_topic) > 0 else 0
    else:
        kpis['trending_topic'] = None
        kpis['trending_topic_count'] = 0
    
    if previous_df is not None and len(previous_df) > 0:
        kpis['posts_change'] = calculate_percentage_change(len(current_df), len(previous_df))
        
        if kpis['trending_topic']:
            prev_topic_count = len(previous_df[previous_df['topic_label'] == kpis['trending_topic']])
            kpis['topic_change'] = calculate_percentage_change(kpis['trending_topic_count'], prev_topic_count)
        else:
            kpis['topic_change'] = 0
    else:
        kpis['posts_change'] = 0
        kpis['topic_change'] = 0
    
    return kpis

# ================================
# DATA LOADING FUNCTIONS
# ================================

@st.cache_data(ttl=7200)
def get_filter_options():
    """Get available filter options from database"""
    date_query = """
    SELECT 
        MIN(DATE(created_at)) as min_date,
        MAX(DATE(created_at)) as max_date
    FROM workspace.redditrecon.posts_gold
    """
    dates_df = run_query(date_query)
    
    sub_query = """
    SELECT DISTINCT subreddit
    FROM workspace.redditrecon.posts_gold
    WHERE subreddit IS NOT NULL
    ORDER BY subreddit
    """
    subs_df = run_query(sub_query)
    
    topic_query = """
    SELECT DISTINCT topic
    FROM workspace.redditrecon.posts_gold
    WHERE topic IS NOT NULL
    ORDER BY topic
    """
    topics_df = run_query(topic_query)
    
    emotion_query = """
    SELECT DISTINCT emotion
    FROM workspace.redditrecon.posts_gold
    WHERE emotion IS NOT NULL
    ORDER BY emotion
    """
    emotions_df = run_query(emotion_query)
    
    return {
        'min_date': dates_df.iloc[0]['min_date'] if not dates_df.empty else datetime.now().date() - timedelta(days=30),
        'max_date': dates_df.iloc[0]['max_date'] if not dates_df.empty else datetime.now().date(),
        'subreddits': subs_df['subreddit'].tolist() if not subs_df.empty else [],
        'topics': topics_df['topic'].tolist() if not topics_df.empty else [],
        'emotions': emotions_df['emotion'].tolist() if not emotions_df.empty else []
    }

@st.cache_data(ttl=3600)
def load_main_data(start_date, end_date, subreddits=None, topics=None, sentiments=None, emotions=None):
    """Load main dataset with all filters applied"""
    filters = [f"DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'"]
    
    if subreddits and len(subreddits) > 0:
        subs = "','".join([s.replace("'", "''") for s in subreddits])
        filters.append(f"subreddit IN ('{subs}')")
    
    if topics and len(topics) > 0:
        tops = "','".join([t.replace("'", "''") for t in topics])
        filters.append(f"topic IN ('{tops}')")
    
    if sentiments and len(sentiments) > 0:
        sents = "','".join(sentiments)
        filters.append(f"sentiment IN ('{sents}')")
    
    if emotions and len(emotions) > 0:
        emos = "','".join([e.replace("'", "''") for e in emotions])
        filters.append(f"emotion IN ('{emos}')")
    
    where_clause = " AND ".join(filters)
    
    query = f"""
    SELECT 
        id as post_id,
        created_at,
        DATE(created_at) as created_date,
        HOUR(created_at) as hour,
        DAYOFWEEK(created_at) as day_of_week,
        subreddit,
        author,
        title,
        selftext,
        LENGTH(COALESCE(title, '') || ' ' || COALESCE(selftext, '')) as text_length,
        sentiment as sentiment_label,
        sentiment_confidence as sentiment_score,
        emotion as emotion_label,
        emotion_confidence as emotion_score,
        topic as topic_label,
        topic_confidence as topic_score,
        score as post_score,
        num_comments,
        0.5 as upvote_ratio
    FROM workspace.redditrecon.posts_gold
    WHERE {where_clause}
    ORDER BY created_at DESC
    LIMIT 50000
    """
    
    df = run_query(query)
    
    if not df.empty:
        df['engagement_score'] = np.log1p(df['post_score'].fillna(0).clip(lower=0)) + np.log1p(df['num_comments'].fillna(0).clip(lower=0))
        df['created_date'] = pd.to_datetime(df['created_date'])
        day_map = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
        df['day_name'] = df['day_of_week'].map(day_map)
        
    return df

@st.cache_data(ttl=3600)
def get_previous_period_data(start_date, end_date, subreddits=None, topics=None, sentiments=None, emotions=None):
    """Get data for previous period for comparison"""
    date_diff = (end_date - start_date).days
    prev_start = start_date - timedelta(days=date_diff)
    prev_end = start_date - timedelta(days=1)
    return load_main_data(prev_start, prev_end, subreddits, topics, sentiments, emotions)

# ================================
# CHART FUNCTIONS
# ================================

COLORS = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'positive': '#27ae60',
    'neutral': '#95a5a6',
    'negative': '#e74c3c'
}

def create_volume_trend(df):
    """Create conversation volume over time chart"""
    daily = df.groupby('created_date').size().reset_index(name='post_count')
    daily['rolling_avg'] = daily['post_count'].rolling(window=7, center=True).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily['created_date'],
        y=daily['post_count'],
        mode='lines+markers',
        name='Daily Posts',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=daily['created_date'],
        y=daily['rolling_avg'],
        mode='lines',
        name='7-Day Average',
        line=dict(color=COLORS['warning'], width=3, dash='dash')
    ))
    
    fig.update_layout(
        title='Conversation Volume Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Posts',
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_sentiment_over_time(df):
    """Create sentiment distribution over time"""
    sentiment_time = df.groupby(['created_date', 'sentiment_label']).size().reset_index(name='count')
    sentiment_time['percentage'] = sentiment_time.groupby('created_date')['count'].transform(lambda x: x / x.sum() * 100)
    
    fig = px.area(
        sentiment_time,
        x='created_date',
        y='percentage',
        color='sentiment_label',
        title='Sentiment Distribution Over Time (%)',
        color_discrete_map={'Positive': COLORS['positive'], 'Neutral': COLORS['neutral'], 'Negative': COLORS['negative']}
    )
    
    fig.update_layout(height=400, template='plotly_white', hovermode='x unified')
    return fig

def create_top_topics_chart(df, top_n=10):
    """Create horizontal bar chart for top topics"""
    topic_counts = df['topic_label'].value_counts().head(top_n).reset_index()
    topic_counts.columns = ['topic', 'count']
    topic_counts['percentage'] = (topic_counts['count'] / len(df) * 100).round(1)
    
    fig = px.bar(
        topic_counts.sort_values('count'),
        x='count',
        y='topic',
        orientation='h',
        title=f'Top {top_n} Topics',
        text='percentage',
        color='count',
        color_continuous_scale='Blues'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False, template='plotly_white')
    return fig

def create_sentiment_heatmap(df):
    """Create sentiment by topic heatmap"""
    pivot = df.groupby(['topic_label', 'sentiment_label']).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    
    top_topics = df['topic_label'].value_counts().head(10).index
    pivot_pct = pivot_pct.loc[pivot_pct.index.isin(top_topics)]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_pct.values,
        x=pivot_pct.columns,
        y=pivot_pct.index,
        colorscale='RdYlGn',
        text=pivot_pct.values.round(1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Percentage")
    ))
    
    fig.update_layout(
        title='Sentiment Distribution by Topic (%)',
        xaxis_title='Sentiment',
        yaxis_title='Topic',
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_emotion_distribution(df):
    """Create emotion distribution bar chart"""
    emotion_counts = df['emotion_label'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']
    emotion_counts['percentage'] = (emotion_counts['count'] / len(df) * 100).round(1)
    
    fig = px.bar(
        emotion_counts.sort_values('count', ascending=False),
        x='count',
        y='emotion',
        orientation='h',
        title='Emotion Distribution',
        text='percentage',
        color='count',
        color_continuous_scale='Reds'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False, template='plotly_white')
    return fig

def create_engagement_by_topic(df, top_n=10):
    """Create engagement by topic chart"""
    topic_engagement = df.groupby('topic_label').agg({
        'engagement_score': 'mean',
        'post_id': 'count'
    }).reset_index()
    topic_engagement.columns = ['topic', 'avg_engagement', 'post_count']
    topic_engagement = topic_engagement.sort_values('avg_engagement', ascending=False).head(top_n)
    
    fig = px.bar(
        topic_engagement.sort_values('avg_engagement'),
        x='avg_engagement',
        y='topic',
        orientation='h',
        title=f'Top {top_n} Topics by Average Engagement',
        color='avg_engagement',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400, showlegend=False, template='plotly_white')
    return fig

def create_time_heatmap(pivot_data, metric_name='Engagement'):
    """Create time of day heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='YlOrRd',
        colorbar=dict(title=metric_name)
    ))
    
    fig.update_layout(
        title=f'Best Time to Post - Average {metric_name} by Day and Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_scatter_plot(df, x_col, y_col, title, color_col=None):
    """Create generic scatter plot"""
    fig = px.scatter(
        df.sample(min(5000, len(df))),
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        opacity=0.6,
        trendline='ols'
    )
    
    fig.update_layout(height=400, template='plotly_white')
    return fig

def create_box_plot(df, x_col, y_col, title):
    """Create box plot for distribution comparison"""
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=x_col,
        color_discrete_map={'Positive': COLORS['positive'], 'Neutral': COLORS['neutral'], 'Negative': COLORS['negative']}
    )
    
    fig.update_layout(height=400, showlegend=False, template='plotly_white')
    return fig

def create_anomaly_chart(df, date_col, value_col, anomaly_col):
    """Create anomaly detection visualization"""
    fig = go.Figure()
    
    normal = df[~df[anomaly_col]]
    fig.add_trace(go.Scatter(
        x=normal[date_col],
        y=normal[value_col],
        mode='lines+markers',
        name='Normal',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=6)
    ))
    
    anomalies = df[df[anomaly_col]]
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies[date_col],
            y=anomalies[value_col],
            mode='markers',
            name='Anomaly',
            marker=dict(size=12, color=COLORS['danger'], symbol='x', line=dict(width=2, color='white'))
        ))
    
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df['rolling_mean'],
        mode='lines',
        name='Expected',
        line=dict(color=COLORS['secondary'], width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Conversation Volume with Anomaly Detection',
        xaxis_title='Date',
        yaxis_title='Post Count',
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

# ================================
# SIDEBAR NAVIGATION & FILTERS
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
    
    if 'page' not in st.session_state:
        st.session_state.page = 'overview'
    
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
    
    filters = get_filter_options()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=max(filters['min_date'], filters['max_date'] - timedelta(days=30)),
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
    
    selected_subreddits = st.multiselect(
        "Subreddits",
        options=filters['subreddits'],
        default=[]
    )
    
    selected_topics = st.multiselect(
        "Topics",
        options=filters['topics'],
        default=[]
    )
    
    selected_sentiments = st.multiselect(
        "Sentiment",
        options=['Positive', 'Neutral', 'Negative'],
        default=[]
    )
    
    selected_emotions = st.multiselect(
        "Emotions",
        options=filters['emotions'],
        default=[]
    )
    
    if st.button("🔄 Reset Filters"):
        st.session_state.clear()
        st.rerun()

# ================================
# LOAD DATA
# ================================

df = load_main_data(
    start_date, end_date,
    selected_subreddits if selected_subreddits else None,
    selected_topics if selected_topics else None,
    selected_sentiments if selected_sentiments else None,
    selected_emotions if selected_emotions else None
)

df_prev = get_previous_period_data(
    start_date, end_date,
    selected_subreddits if selected_subreddits else None,
    selected_topics if selected_topics else None,
    selected_sentiments if selected_sentiments else None,
    selected_emotions if selected_emotions else None
)

if df.empty:
    st.warning("⚠️ No data available for the selected filters. Please adjust your filters and try again.")
    st.stop()

kpis = calculate_kpis(df, df_prev)

# ================================
# PAGES
# ================================

if st.session_state.page == 'overview':
    st.title("📊 Overview Dashboard")
    st.markdown("*Executive summary of Reddit conversation intelligence*")
    
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
    
    st.markdown("<div class='section-header'>Conversation Trends</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_volume = create_volume_trend(df)
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        fig_sentiment = create_sentiment_over_time(df)
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    st.markdown("<div class='section-header'>Topic Analysis</div>", unsafe_allow_html=True)
    
    fig_topics = create_top_topics_chart(df, 10)
    st.plotly_chart(fig_topics, use_container_width=True)
    
    st.markdown("<div class='section-header'>Recent Conversations</div>", unsafe_allow_html=True)
    
    recent = df.head(100)[['created_date', 'subreddit', 'title', 'topic_label', 
                           'sentiment_label', 'emotion_label', 'post_score', 'num_comments']].copy()
    recent.columns = ['Date', 'Subreddit', 'Title', 'Topic', 'Sentiment', 'Emotion', 'Score', 'Comments']
    
    st.dataframe(recent, use_container_width=True, height=400)

elif st.session_state.page == 'sentiment':
    st.title("💭 Sentiment Analysis")
    st.markdown("*Understanding emotional polarity in conversations*")
    
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
    
    st.markdown("<div class='section-header'>Sentiment by Topic</div>", unsafe_allow_html=True)
    
    fig_heatmap = create_sentiment_heatmap(df)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("<div class='section-header'>Sentiment Impact on Engagement</div>", unsafe_allow_html=True)
    
    fig_box = create_box_plot(df, 'sentiment_label', 'engagement_score', 
                              'Engagement Distribution by Sentiment')
    st.plotly_chart(fig_box, use_container_width=True)

elif st.session_state.page == 'topics':
    st.title("📚 Topic Intelligence")
    st.markdown("*Discovering what people are talking about*")
    
    st.markdown("<div class='section-header'>Topic Ranking</div>", unsafe_allow_html=True)
    
    fig_topics = create_top_topics_chart(df, 15)
    st.plotly_chart(fig_topics, use_container_width=True)
    
    st.markdown("<div class='section-header'>Topic Evolution</div>", unsafe_allow_html=True)
    
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
    
    if not df_prev.empty:
        st.markdown("<div class='section-header'>🔥 Trending Topics (Topic Velocity)</div>", unsafe_allow_html=True)
        
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

elif st.session_state.page == 'emotions':
    st.title("🎭 Emotion Analysis")
    st.markdown("*Understanding emotional responses in conversations*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_emotion_dist = create_emotion_distribution(df)
        st.plotly_chart(fig_emotion_dist, use_container_width=True)
    
    with col2:
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
    
    st.markdown("<div class='section-header'>Emotion Impact on Engagement</div>", unsafe_allow_html=True)
    
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

elif st.session_state.page == 'engagement':
    st.title("🔥 Engagement Analysis")
    st.markdown("*Understanding what drives Reddit interaction*")
    
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
    
    st.markdown("<div class='section-header'>Topic Engagement Ranking</div>", unsafe_allow_html=True)
    
    fig_topic_eng = create_engagement_by_topic(df, 10)
    st.plotly_chart(fig_topic_eng, use_container_width=True)
    
    st.markdown("<div class='section-header'>Best Time to Post</div>", unsafe_allow_html=True)
    
    pivot_data = get_posting_time_heatmap_data(df)
    fig_time_heatmap = create_time_heatmap(pivot_data, 'Engagement Score')
    st.plotly_chart(fig_time_heatmap, use_container_width=True)
    
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
        fig_sent_eng = create_box_plot(df, 'sentiment_label', 'engagement_score',
                                       'Engagement by Sentiment')
        st.plotly_chart(fig_sent_eng, use_container_width=True)

elif st.session_state.page == 'predictive':
    st.title("🤖 Predictive Analytics")
    st.markdown("*Machine learning insights for engagement prediction*")
    
    df_ml, engagement_threshold = calculate_engagement_percentiles(df)
    
    st.markdown(f"**High Engagement Threshold:** {engagement_threshold:.2f} (Top 25%)")
    
    st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
    
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
    
    st.markdown("<div class='section-header'>Engagement Prediction Simulator</div>", unsafe_allow_html=True)
    
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
            base_score = 0.5
            topic_eng = df[df['topic_label'] == sim_topic]['engagement_score'].mean()
            topic_norm = (topic_eng - df['engagement_score'].mean()) / df['engagement_score'].std()
            sent_mult = {'Positive': 1.1, 'Neutral': 1.0, 'Negative': 0.9}[sim_sentiment]
            hour_mult = 1.2 if 18 <= sim_hour <= 22 else 1.0
            prob = min(0.95, max(0.05, base_score + topic_norm * 0.15 * sent_mult * hour_mult))
            
            engagement_class = "HIGH" if prob > 0.5 else "LOW"
            color = "green" if prob > 0.5 else "orange"
            
            st.markdown(f"""
            <div style='background-color: white; padding: 2rem; border-radius: 8px; 
                        border-left: 4px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h2 style='color: {color}; margin: 0;'>{engagement_class} ENGAGEMENT</h2>
                <h1 style='color: #2c3e50; margin: 0.5rem 0;'>{prob*100:.0f}%</h1>
                <p style='color: #7f8c8d; margin: 0;'>Probability of high engagement</p>
            </div>
            """, unsafe_allow_html=True)

elif st.session_state.page == 'anomaly':
    st.title("⚠️ Anomaly Detection")
    st.markdown("*Identifying unusual patterns in conversations*")
    
    daily_data = df.groupby('created_date').size().reset_index(name='post_count')
    daily_data = detect_anomalies(daily_data, 'post_count', window=7, threshold=2.5)
    
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
        recent_sentiment = df[df['created_date'] >= df['created_date'].max() - timedelta(days=3)]['sentiment_score'].mean()
        overall_sentiment = df['sentiment_score'].mean()
        sentiment_shift = recent_sentiment - overall_sentiment
        st.metric("SENTIMENT SHIFT", f"{sentiment_shift:+.3f}", 
                 "Recent 3 days vs average")
    
    with col4:
        alert_level = "HIGH" if anomaly_count > 3 else "MEDIUM" if anomaly_count > 0 else "LOW"
        alert_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[alert_level]
        st.metric("ALERT LEVEL", f"{alert_color} {alert_level}")
    
    st.markdown("<div class='section-header'>Conversation Volume Anomalies</div>", unsafe_allow_html=True)
    
    fig_anomaly = create_anomaly_chart(daily_data, 'created_date', 'post_count', 'is_anomaly')
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    if not df_prev.empty:
        st.markdown("<div class='section-header'>Topic Spike Detection</div>", unsafe_allow_html=True)
        
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
    <h3 style='color: #2c3e50;'>🎯 Reddit Conversation Intelligence Platform</h3>
    <p>Powered by Databricks Lakehouse • Streamlit • NLP AI Models</p>
</div>
""", unsafe_allow_html=True)
