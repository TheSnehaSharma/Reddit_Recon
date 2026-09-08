-- ============================================================================
-- Reddit Pulse — Databricks curated view
-- Run this once (or via a Databricks Job / dbt model) against the schema
-- named in .streamlit/secrets.toml ([databricks].catalog / .schema).
--
-- Assumes a raw Delta table `reddit_posts_raw` with columns matching the
-- source CSV: author, author_flair_text, created_at, created_utc, emotion,
-- emotion_confidence, id, is_bot, link_flair_text, load_date, num_comments,
-- over_18, score, selftext, sentiment, sentiment_confidence, subreddit,
-- title, title_length, topic, topic_confidence, url
-- ============================================================================

CREATE OR REPLACE VIEW workspace.redditrecon.vw_reddit_posts AS
SELECT
    id,
    author,
    author_flair_text,
    created_at,
    load_date,
    subreddit,
    title,
    title_length,
    selftext,
    url,
    link_flair_text,
    score,
    num_comments,
    is_bot,
    over_18,
    LOWER(sentiment)                   AS sentiment,
    sentiment_confidence,
    LOWER(emotion)                     AS emotion,
    emotion_confidence,
    topic,
    topic_confidence
FROM workspace.redditrecon.posts_gold
WHERE id IS NOT NULL
  AND created_at IS NOT NULL;

-- Helpful indexes/optimizations on the underlying Delta table (run once):
OPTIMIZE workspace.redditrecon.posts_gold ZORDER BY (subreddit, created_at);