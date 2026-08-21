-- Staging model: Reference to Gold layer table
-- This is a simple passthrough to make the gold table available for downstream models

{{ config(
    materialized='view',
    schema='staging'
) }}

SELECT 
    id,
    author,
    subreddit,
    title,
    selftext,
    score,
    created_at,
    created_utc,
    num_comments,
    url,
    over_18,
    link_flair_text,
    author_flair_text,
    is_bot,
    sentiment,
    sentiment_confidence,
    emotion,
    emotion_confidence,
    topic,
    topic_confidence,
    load_date
FROM {{ var('gold_table') }}