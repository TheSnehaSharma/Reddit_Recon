"""
Charts and Visualization Module
Contains all Plotly chart creation functions
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Professional color scheme
COLORS = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#9b59b6',
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
    """Create sentiment distribution over time (stacked area)"""
    sentiment_time = df.groupby(['created_date', 'sentiment_label']).size().reset_index(name='count')
    sentiment_time['percentage'] = sentiment_time.groupby('created_date')['count'].transform(lambda x: x / x.sum() * 100)
    
    fig = px.area(
        sentiment_time,
        x='created_date',
        y='percentage',
        color='sentiment_label',
        title='Sentiment Distribution Over Time (%)',
        color_discrete_map={'Positive': COLORS['positive'], 'Neutral': COLORS['neutral'], 'Negative': COLORS['negative']},
        labels={'percentage': 'Percentage (%)', 'created_date': 'Date', 'sentiment_label': 'Sentiment'}
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
    
    # Get top 10 topics
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
        color_continuous_scale='Viridis',
        hover_data=['post_count']
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
    
    # Normal points
    normal = df[~df[anomaly_col]]
    fig.add_trace(go.Scatter(
        x=normal[date_col],
        y=normal[value_col],
        mode='lines+markers',
        name='Normal',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=6)
    ))
    
    # Anomalies
    anomalies = df[df[anomaly_col]]
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies[date_col],
            y=anomalies[value_col],
            mode='markers',
            name='Anomaly',
            marker=dict(size=12, color=COLORS['danger'], symbol='x', line=dict(width=2, color='white'))
        ))
    
    # Rolling average
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
