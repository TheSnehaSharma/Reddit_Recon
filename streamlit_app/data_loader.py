"""
Data Loading and Filtering Module
Handles all database queries and data transformations
"""

import pandas as pd
import numpy as np
from databricks import sql
import os
import streamlit as st
from datetime import datetime, timedelta

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
    SELECT DISTINCT topic_label
    FROM workspace.redditrecon.posts_gold
    WHERE topic_label IS NOT NULL
    ORDER BY topic_label
    """
    topics_df = run_query(topic_query)
    
    emotion_query = """
    SELECT DISTINCT emotion_label
    FROM workspace.redditrecon.posts_gold
    WHERE emotion_label IS NOT NULL
    ORDER BY emotion_label
    """
    emotions_df = run_query(emotion_query)
    
    return {
        'min_date': dates_df.iloc[0]['min_date'] if not dates_df.empty else datetime.now().date() - timedelta(days=30),
        'max_date': dates_df.iloc[0]['max_date'] if not dates_df.empty else datetime.now().date(),
        'subreddits': subs_df['subreddit'].tolist() if not subs_df.empty else [],
        'topics': topics_df['topic_label'].tolist() if not topics_df.empty else [],
        'emotions': emotions_df['emotion_label'].tolist() if not emotions_df.empty else []
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
        filters.append(f"topic_label IN ('{tops}')")
    
    if sentiments and len(sentiments) > 0:
        sents = "','".join(sentiments)
        filters.append(f"sentiment_label IN ('{sents}')")
    
    if emotions and len(emotions) > 0:
        emos = "','".join([e.replace("'", "''") for e in emotions])
        filters.append(f"emotion_label IN ('{emos}')")
    
    where_clause = " AND ".join(filters)
    
    query = f"""
    SELECT 
        post_id,
        created_at,
        DATE(created_at) as created_date,
        HOUR(created_at) as hour,
        DAYOFWEEK(created_at) as day_of_week,
        subreddit,
        author,
        title,
        selftext,
        LENGTH(COALESCE(title, '') || ' ' || COALESCE(selftext, '')) as text_length,
        sentiment_label,
        sentiment_score,
        emotion_label,
        emotion_score,
        topic_label,
        topic_score,
        score as post_score,
        num_comments,
        upvote_ratio
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
