-- Materialized View: Emotion Analysis Overview
-- Aggregates emotional tone across posts

{{ config(
    materialized='materialized_view',
    schema='analytics'
) }}

SELECT 
    emotion,
    COUNT(*) as post_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage,
    
    -- Confidence metrics
    ROUND(AVG(emotion_confidence), 3) as avg_confidence,
    ROUND(MIN(emotion_confidence), 3) as min_confidence,
    ROUND(MAX(emotion_confidence), 3) as max_confidence,
    
    -- Associated sentiments
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive_sentiment_count,
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative_sentiment_count,
    SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_sentiment_count,
    
    -- Engagement
    ROUND(AVG(score), 2) as avg_score,
    MAX(score) as highest_score,
    ROUND(AVG(num_comments), 2) as avg_comments
    
FROM {{ ref('stg_reddit_gold') }}
WHERE emotion IS NOT NULL
GROUP BY emotion
ORDER BY post_count DESC