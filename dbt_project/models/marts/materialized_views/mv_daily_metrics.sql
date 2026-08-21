-- Materialized View: Daily Metrics
-- Time-series aggregation of posts by day

{{ config(
    materialized='materialized_view',
    schema='analytics'
) }}

SELECT 
    DATE(created_at) as post_date,
    COUNT(*) as total_posts,
    
    -- Sentiment metrics
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive_posts,
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative_posts,
    SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_posts,
    
    -- Engagement metrics
    ROUND(AVG(score), 2) as avg_score,
    MAX(score) as top_score,
    SUM(score) as total_score,
    ROUND(AVG(num_comments), 2) as avg_comments,
    
    -- Unique counts
    COUNT(DISTINCT subreddit) as unique_subreddits,
    COUNT(DISTINCT author) as unique_authors,
    COUNT(DISTINCT topic) as unique_topics,
    
    -- Quality metrics
    ROUND(AVG(sentiment_confidence), 3) as avg_sentiment_confidence,
    ROUND(AVG(emotion_confidence), 3) as avg_emotion_confidence,
    ROUND(AVG(topic_confidence), 3) as avg_topic_confidence
    
FROM {{ ref('stg_reddit_gold') }}
WHERE created_at IS NOT NULL
GROUP BY DATE(created_at)
ORDER BY post_date DESC