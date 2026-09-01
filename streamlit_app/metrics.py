"""
Metrics Calculation Module
Handles all metric calculations and aggregations
"""

import pandas as pd
import numpy as np
from scipy import stats

def format_number(num):
    """Format large numbers with K, M suffixes"""
    if pd.isna(num):
        return "0"
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    return str(int(num))

def calculate_percentage_change(current, previous):
    """Calculate percentage change between two values"""
    if pd.isna(previous) or previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def get_sentiment_label(score):
    """Convert sentiment score to label"""
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    return "Neutral"

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
    
    # Pivot for heatmap
    pivot = heatmap_data.pivot(index='day_name', columns='hour', values='engagement_score')
    
    # Reorder days
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
    
    # Calculate trending topic
    if len(current_df) > 0:
        top_topic = current_df['topic_label'].value_counts().head(1)
        kpis['trending_topic'] = top_topic.index[0] if len(top_topic) > 0 else None
        kpis['trending_topic_count'] = top_topic.values[0] if len(top_topic) > 0 else 0
    else:
        kpis['trending_topic'] = None
        kpis['trending_topic_count'] = 0
    
    # Calculate percentage changes if previous period provided
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
