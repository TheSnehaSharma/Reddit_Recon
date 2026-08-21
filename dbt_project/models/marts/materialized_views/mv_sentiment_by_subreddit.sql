-- Materialized View: Sentiment Analysis by Subreddit
-- Aggregates sentiment metrics for each subreddit

{{ config(
    materialized='materialized_view',
    schema='analytics'
) }}

SELECT 
    subreddit,
    COUNT(*) as total_posts,
    
    -- Sentiment distribution
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive_count,
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count,
    SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
    
    -- Sentiment percentages
    ROUND(SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as positive_pct,
    ROUND(SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as negative_pct,
    ROUND(SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as neutral_pct,
    
    -- Average confidence and engagement
    ROUND(AVG(sentiment_confidence), 3) as avg_sentiment_confidence,
    ROUND(AVG(score), 2) as avg_score,
    MAX(score) as max_score,
    SUM(score) as total_score,
    
    -- Engagement metrics
    ROUND(AVG(num_comments), 2) as avg_comments,
    MAX(num_comments) as max_comments
    
FROM {{ ref('stg_reddit_gold') }}
WHERE subreddit IS NOT NULL
GROUP BY subreddit
ORDER BY total_posts DESC