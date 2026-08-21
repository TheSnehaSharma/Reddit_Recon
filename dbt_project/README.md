# Reddit Analytics dbt Project

This dbt project creates materialized views on top of the Reddit gold layer data for analytics and reporting.

## Prerequisites

1. **Python 3.8+** installed
2. **Databricks SQL Warehouse** running
3. **Databricks Personal Access Token**

## Setup Instructions

### 1. Install dbt with Databricks adapter

```bash
pip install dbt-databricks
```

### 2. Configure Databricks Connection

#### Get your SQL Warehouse HTTP Path:
1. Go to your Databricks workspace
2. Navigate to SQL Warehouses
3. Click on your warehouse
4. Copy the **HTTP Path** (format: `/sql/1.0/warehouses/xxxxx`)

#### Create Personal Access Token:
1. Click your username in top-right corner
2. Settings → Developer → Access Tokens
3. Generate New Token
4. Copy and save the token securely

#### Set Environment Variable:
```bash
export DATABRICKS_TOKEN="your_token_here"
```

### 3. Update profiles.yml

Edit `profiles.yml` and update:
- `host`: Your Databricks workspace URL (already set)
- `http_path`: Your SQL Warehouse HTTP path

### 4. Copy profiles.yml to dbt directory

```bash
mkdir -p ~/.dbt
cp profiles.yml ~/.dbt/
```

## Running dbt

### Test connection
```bash
dbt debug
```

### Install dependencies (if any)
```bash
dbt deps
```

### Run all models
```bash
dbt run
```

### Run specific models
```bash
# Run only materialized views
dbt run --select marts.materialized_views

# Run specific model
dbt run --select mv_sentiment_by_subreddit
```

### Test models
```bash
dbt test
```

### Generate documentation
```bash
dbt docs generate
dbt docs serve
```

## Project Structure

```
dbt_project/
├── dbt_project.yml          # Project configuration
├── profiles.yml             # Connection configuration
├── models/
│   ├── staging/
│   │   └── stg_reddit_gold.sql     # Staging view of gold table
│   └── marts/
│       └── materialized_views/
│           ├── mv_sentiment_by_subreddit.sql   # Sentiment by subreddit
│           ├── mv_topic_trends.sql             # Topic trends analysis
│           ├── mv_emotion_analysis.sql         # Emotion analysis
│           └── mv_daily_metrics.sql            # Daily time-series metrics
```

## Materialized Views Created

### 1. `workspace.analytics.mv_sentiment_by_subreddit`
- Sentiment distribution by subreddit
- Engagement metrics (score, comments)
- Confidence scores

### 2. `workspace.analytics.mv_topic_trends`
- Post counts by topic
- Sentiment breakdown per topic
- Average engagement by topic

### 3. `workspace.analytics.mv_emotion_analysis`
- Emotion distribution across posts
- Confidence metrics
- Sentiment associations

### 4. `workspace.analytics.mv_daily_metrics`
- Time-series daily aggregations
- Sentiment trends over time
- Engagement metrics by day

## Querying Materialized Views

Once created, query the views directly:

```sql
-- Top subreddits by sentiment
SELECT subreddit, positive_pct, negative_pct, total_posts
FROM workspace.analytics.mv_sentiment_by_subreddit
ORDER BY total_posts DESC
LIMIT 10;

-- Topic trends
SELECT topic, post_count, avg_score
FROM workspace.analytics.mv_topic_trends
ORDER BY post_count DESC;

-- Daily sentiment trend
SELECT post_date, positive_posts, negative_posts, neutral_posts
FROM workspace.analytics.mv_daily_metrics
ORDER BY post_date DESC;
```

## Refreshing Materialized Views

Materialized views in Databricks refresh automatically, but you can manually trigger:

```bash
# Refresh all models
dbt run

# Refresh specific view
dbt run --select mv_sentiment_by_subreddit
```

## Troubleshooting

### Connection issues:
- Verify `DATABRICKS_TOKEN` environment variable is set
- Check SQL Warehouse is running
- Verify HTTP path is correct

### Permission issues:
- Ensure token has access to workspace catalog
- Verify CREATE privileges on analytics schema

### Model failures:
- Check gold table exists: `workspace.redditrecon.posts_gold`
- Verify all required columns exist
- Review dbt logs: `logs/dbt.log`

## Next Steps

1. Add dbt tests for data quality
2. Set up dbt Cloud for scheduled runs
3. Create additional mart models
4. Build Streamlit dashboard on top of materialized views