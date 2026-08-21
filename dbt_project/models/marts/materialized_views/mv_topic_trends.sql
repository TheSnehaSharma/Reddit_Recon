-- Materialized View: Topic Trends Analysis
-- Aggregates metrics by topic category

{{ config(
    materialized='materialized_view',
    schema='analytics'
) }}

SELECT 
    topic,
    COUNT(*) as post_count,
    
    -- Sentiment breakdown by topic
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive_posts,
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative_posts,
    SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_posts,
    
    -- Average confidences
    ROUND(AVG(topic_confidence), 3) as avg_topic_confidence,
    ROUND(AVG(sentiment_confidence), 3) as avg_sentiment_confidence,
    
    -- Engagement metrics
    ROUND(AVG(score), 2) as avg_score,
    MAX(score) as top_score,
    ROUND(AVG(num_comments), 2) as avg_comments,
    
    -- Top subreddits for this topic (concatenated)
    COLLECT_SET(subreddit) as related_subreddits
    
FROM {{ ref('stg_reddit_gold') }}
WHERE topic IS NOT NULL
GROUP BY topic
ORDER BY post_count DESC