from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from databricks import sql as dbsql
except ImportError:  # lets the UI be previewed even without the connector installed
    dbsql = None


# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + REDDIT-THEMED STYLING
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Reddit Recon",
    page_icon="🔶",
    layout="wide",
    initial_sidebar_state="expanded",
)

ORANGE = "#FF4500"
UPVOTE = "#FF8717"
BLUE = "#0079D3"
GREEN = "#46D160"
BG = "#0B1416"
CARD_BG = "#1A1A1B"
TEXT = "#D7DADC"
MUTED = "#818384"
PLOTLY_TEMPLATE = "plotly_dark"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; }}
    section[data-testid="stSidebar"] {{
        background-color: {CARD_BG}; border-right: 1px solid #343536;
    }}
    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid #343536;
        border-radius: 10px;
        padding: 14px 16px 8px 16px;
    }}
    div[data-testid="stMetricValue"] {{ color: {ORANGE}; font-weight: 700; }}
    h1, h2, h3 {{ color: {TEXT}; font-family: "IBM Plex Sans", sans-serif; }}
    h1 span.accent {{ color: {ORANGE}; }}
    .rp-badge {{
        display:inline-block; background:{ORANGE}; color:white; font-size:12px;
        padding:2px 10px; border-radius:12px; font-weight:600; margin-left:8px;
        vertical-align:middle;
    }}
    hr {{ border-color: #343536; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <h1>🔶 Reddit <span class="accent">Recon</span>
    <span class="rp-badge">LIVE · DATABRICKS</span></h1>
    <p style="color:{MUTED}; margin-top:-8px;">
    Sentiment, emotion &amp; engagement analytics across Reddit posts,
    served entirely from Databricks SQL views.
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════
# DATABRICKS CONNECTION LAYER
# ══════════════════════════════════════════════════════════════════════════
def _secret(key: str, default: str | None = None):
    """Read from secrets.toml or environment variables."""
    # Try secrets.toml first (for local development)
    try:
        return st.secrets["databricks"][key]
    except Exception:
        pass
    
    # Fall back to environment variables (for GitHub deployment)
    env_map = {
        "server_hostname": "DATABRICKS_SERVER_HOSTNAME",
        "http_path": "DATABRICKS_HTTP_PATH",
        "access_token": "DATABRICKS_TOKEN",
    }
    env_key = env_map.get(key)
    if env_key:
        return os.environ.get(env_key, default)
    return default


# Hardcoded for deployment - these match your Databricks workspace
CATALOG = "workspace"
SCHEMA = "redditrecon"
BASE_VIEW = f"{CATALOG}.{SCHEMA}.vw_reddit_posts"


@st.cache_resource(show_spinner=False)
def get_connection():
    """One warehouse connection per app process, reused across sessions.
    Note: Connection itself is thread-safe, but cursors are NOT.
    Each thread must create its own cursor.
    """
    if dbsql is None:
        raise RuntimeError(
            "databricks-sql-connector is not installed. "
            "Run: pip install databricks-sql-connector"
        )
    
    # Validate connection parameters
    hostname = _secret("server_hostname")
    http_path = _secret("http_path")
    token = _secret("access_token")
    
    if not all([hostname, http_path, token]):
        raise ValueError(
            "Missing Databricks credentials. Set environment variables:\n"
            "  - DATABRICKS_SERVER_HOSTNAME\n"
            "  - DATABRICKS_HTTP_PATH\n"
            "  - DATABRICKS_TOKEN"
        )
    
    try:
        return dbsql.connect(
            server_hostname=hostname,
            http_path=http_path,
            access_token=token,
        )
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to Databricks SQL Warehouse.\n"
            f"Hostname: {hostname}\n"
            f"Error: {str(e)}"
        )


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerics and category-encode low-cardinality strings so the
    Streamlit process holds as little in memory as possible."""
    if df.empty:
        return df
    
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object"]).columns:
        n = len(df)
        if n and df[col].nunique() / n < 0.5:
            df[col] = df[col].astype("category")
    return df


def health_check() -> bool:
    """Verify database connection and view availability."""
    try:
        df = run_query(f"SELECT COUNT(*) as cnt FROM {BASE_VIEW} LIMIT 1", retry_count=1)
        return not df.empty and df.iloc[0]['cnt'] > 0
    except Exception:
        return False


@st.cache_data(ttl=600, show_spinner=False)
def run_query(sql_text: str, params: dict | None = None, retry_count: int = 2) -> pd.DataFrame:
    """Execute SQL against the warehouse, return a memory-optimized frame.
    Thread-safe: Each call creates its own cursor.
    Cached 10 min per unique (sql, params) — repeated filter states are free.
    """
    last_error = None
    
    for attempt in range(retry_count):
        cursor = None
        try:
            conn = get_connection()
            # IMPORTANT: Each thread gets its own cursor for thread safety
            cursor = conn.cursor()
            cursor.execute(sql_text, params or {})
            cols = [c[0] for c in cursor.description]
            rows = cursor.fetchall()
            return optimize_dtypes(pd.DataFrame(rows, columns=cols))
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # Only retry on transient errors
            is_transient = any(x in error_msg for x in [
                'timeout', 'connection', 'temporarily unavailable',
                'too many requests', 'throttled'
            ])
            
            if attempt < retry_count - 1 and is_transient:
                import time
                # Shorter retry delay: 1s, then 2s
                time.sleep(1 * (attempt + 1))
            elif not is_transient:
                # Non-transient error - fail immediately
                raise RuntimeError(
                    f"Query failed with non-retryable error.\n"
                    f"Error: {str(last_error)}"
                ) from last_error
            else:
                # Exhausted retries
                raise RuntimeError(
                    f"Query failed after {retry_count} attempts.\n"
                    f"Error: {str(last_error)}"
                ) from last_error
        finally:
            # Clean up cursor
            if cursor is not None:
                try:
                    cursor.close()
                except:
                    pass


# ══════════════════════════════════════════════════════════════════════════
# FILTER STATE → SQL WHERE CLAUSE (pushdown — never filter in pandas)
# ══════════════════════════════════════════════════════════════════════════
def build_where(start_d, end_d, subreddits, sentiments, hide_bots, hide_nsfw):
    # OPTIMIZED: Use timestamp range instead of CAST for better performance
    # Convert date to timestamp range (inclusive of entire end day)
    start_ts = datetime.combine(start_d, datetime.min.time())
    end_ts = datetime.combine(end_d, datetime.max.time())
    
    clauses = ["created_at >= %(start_ts)s AND created_at <= %(end_ts)s"]
    params = {"start_ts": start_ts, "end_ts": end_ts}

    if subreddits:
        keys = []
        for i, sr in enumerate(subreddits):
            k = f"sr{i}"
            keys.append(f"%({k})s")
            params[k] = sr
        clauses.append(f"subreddit IN ({', '.join(keys)})")

    if sentiments:
        keys = []
        for i, s in enumerate(sentiments):
            k = f"sent{i}"
            keys.append(f"%({k})s")
            params[k] = s
        clauses.append(f"sentiment IN ({', '.join(keys)})")

    if hide_bots:
        clauses.append("is_bot = false")
    if hide_nsfw:
        clauses.append("over_18 = false")

    return " AND ".join(clauses), params


@st.cache_data(ttl=1800)
def get_filter_options():
    """Get available subreddits for filter dropdown."""
    try:
        df = run_query(
            f"SELECT DISTINCT subreddit FROM {BASE_VIEW} "
            f"WHERE subreddit IS NOT NULL ORDER BY subreddit LIMIT 200"
        )
        return sorted(df["subreddit"].tolist()) if not df.empty else []
    except Exception as e:
        st.warning(f"Could not load filter options: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILTERS (FORM-BASED TO PREVENT IMMEDIATE RERUNS)
# ══════════════════════════════════════════════════════════════════════════

# Initialize session state for filters on first load
if 'filters_applied' not in st.session_state:
    # Default to July 2024 where the data actually exists
    st.session_state.filters_applied = {
        'start_d': date(2024, 7, 1),
        'end_d': date(2024, 7, 31),
        'subreddits': [],
        'sentiments': [],
        'hide_bots': True,
        'hide_nsfw': True,
        'top_n': 10
    }

with st.sidebar:
    st.markdown("### ⚙️ Filters")
    
    with st.form(key="filters_form"):
        # Get filter options (cached, won't rerun frequently)
        sub_options = get_filter_options()
        
        # Date range
        date_range = st.date_input(
            "Date range",
            value=(st.session_state.filters_applied['start_d'], 
                   st.session_state.filters_applied['end_d']),
            help="Data available: July 2024. Adjust pipeline to load more dates."
        )
        
        # Subreddits
        subreddits = st.multiselect(
            "Subreddits",
            options=sub_options,
            default=st.session_state.filters_applied['subreddits'],
            help="Select specific subreddits to filter. Leave empty for all."
        )
        
        # Sentiments
        sentiments = st.multiselect(
            "Sentiment",
            options=["positive", "neutral", "negative"],
            default=st.session_state.filters_applied['sentiments'],
            help="Filter by sentiment. Leave empty for all."
        )
        
        # Bot filter
        hide_bots = st.checkbox(
            "Exclude bot authors",
            value=st.session_state.filters_applied['hide_bots'],
            help="Remove posts identified as bot-generated."
        )
        
        # NSFW filter
        hide_nsfw = st.checkbox(
            "Exclude NSFW (18+)",
            value=st.session_state.filters_applied['hide_nsfw'],
            help="Remove posts marked as NSFW/adult content."
        )
        
        # Top N slider
        top_n = st.slider(
            "Top-N for rankings",
            min_value=5,
            max_value=25,
            value=st.session_state.filters_applied['top_n'],
            help="Number of items to show in leaderboards."
        )
        
        # Submit button
        submitted = st.form_submit_button(
            "🔍 Apply Filters",
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            # Update session state with new filters
            start_d, end_d = date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (date(2024, 7, 1), date(2024, 7, 31))
            st.session_state.filters_applied = {
                'start_d': start_d,
                'end_d': end_d,
                'subreddits': subreddits,
                'sentiments': sentiments,
                'hide_bots': hide_bots,
                'hide_nsfw': hide_nsfw,
                'top_n': top_n
            }
    
    st.caption("💡 Change filters and click 'Apply' to refresh. "
               "All aggregation happens in Databricks SQL.")

# Use the applied filters from session state
filters = st.session_state.filters_applied
start_d = filters['start_d']
end_d = filters['end_d']
subreddits = filters['subreddits']
sentiments = filters['sentiments']
hide_bots = filters['hide_bots']
hide_nsfw = filters['hide_nsfw']
top_n = filters['top_n']

WHERE_SQL, PARAMS = build_where(start_d, end_d, subreddits, sentiments, hide_bots, hide_nsfw)


# ══════════════════════════════════════════════════════════════════════════
# QUERIES — every one is a bounded aggregate, computed in Databricks SQL
# ══════════════════════════════════════════════════════════════════════════
def q_kpis():
    return run_query(f"""
        SELECT
            COUNT(*)                                                            AS total_posts,
            COUNT(DISTINCT author)                                              AS unique_authors,
            COUNT(DISTINCT subreddit)                                           AS unique_subreddits,
            SUM(num_comments)                                                   AS total_comments,
            ROUND(AVG(score), 1)                                                AS avg_score,
            ROUND(AVG(sentiment_confidence), 3)                                 AS avg_sent_conf,
            ROUND(100.0 * SUM(CASE WHEN is_bot THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_bot,
            ROUND(100.0 * SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_positive
        FROM {BASE_VIEW} WHERE {WHERE_SQL}
    """, PARAMS)


def q_sentiment_dist():
    return run_query(f"""
        SELECT sentiment, COUNT(*) AS n FROM {BASE_VIEW} WHERE {WHERE_SQL}
        GROUP BY sentiment ORDER BY n DESC
    """, PARAMS)


def q_emotion_dist():
    return run_query(f"""
        SELECT emotion, COUNT(*) AS n, ROUND(AVG(emotion_confidence), 3) AS avg_conf
        FROM {BASE_VIEW} WHERE {WHERE_SQL} GROUP BY emotion ORDER BY n DESC
    """, PARAMS)


def q_topic_dist():
    return run_query(f"""
        SELECT topic, COUNT(*) AS n, ROUND(AVG(score), 1) AS avg_score
        FROM {BASE_VIEW} WHERE {WHERE_SQL} GROUP BY topic ORDER BY n DESC LIMIT 12
    """, PARAMS)


def q_subreddit_stats(n):
    return run_query(f"""
        SELECT subreddit,
               COUNT(*)                            AS posts,
               ROUND(AVG(score), 1)                AS avg_score,
               ROUND(AVG(num_comments), 1)         AS avg_comments,
               ROUND(AVG(sentiment_confidence), 3) AS avg_sent_conf
        FROM {BASE_VIEW} WHERE {WHERE_SQL}
        GROUP BY subreddit ORDER BY posts DESC LIMIT %(n)s
    """, {**PARAMS, "n": n})


def q_timeseries():
    return run_query(f"""
        SELECT CAST(created_at AS DATE) AS day,
               COUNT(*) AS posts, ROUND(AVG(score), 1) AS avg_score, SUM(num_comments) AS comments
        FROM {BASE_VIEW} WHERE {WHERE_SQL}
        GROUP BY CAST(created_at AS DATE) ORDER BY day
    """, PARAMS)


def q_sentiment_by_subreddit(n):
    return run_query(f"""
        WITH top_subs AS (
            SELECT subreddit FROM {BASE_VIEW} WHERE {WHERE_SQL}
            GROUP BY subreddit ORDER BY COUNT(*) DESC LIMIT %(n)s
        )
        SELECT p.subreddit, p.sentiment, COUNT(*) AS n
        FROM {BASE_VIEW} p JOIN top_subs t ON t.subreddit = p.subreddit
        WHERE {WHERE_SQL} GROUP BY p.subreddit, p.sentiment
    """, {**PARAMS, "n": n})


def q_score_vs_comments_sample():
    # OPTIMIZED: Deterministic hash-based sampling instead of ORDER BY RAND()
    # This is 10-100x faster and gives consistent results
    return run_query(f"""
        SELECT score, num_comments, subreddit, sentiment, title_length
        FROM {BASE_VIEW} 
        WHERE {WHERE_SQL} 
          AND MOD(ABS(HASH(id)), 100) < 75
        LIMIT 1500
    """, PARAMS)


def q_top_posts(n):
    return run_query(f"""
        SELECT title, subreddit, score, num_comments, sentiment, emotion, url
        FROM {BASE_VIEW} WHERE {WHERE_SQL} ORDER BY score DESC LIMIT %(n)s
    """, {**PARAMS, "n": n})


# ══════════════════════════════════════════════════════════════════════════
# PARALLEL QUERY EXECUTION
# ══════════════════════════════════════════════════════════════════════════

def execute_queries_parallel() -> dict[str, pd.DataFrame]:
    """Execute all dashboard queries in parallel using ThreadPoolExecutor.
    
    This reduces total latency by running independent queries concurrently.
    Each query gets its own cursor for thread safety.
    
    Returns:
        Dictionary mapping query names to their result DataFrames
    
    Raises:
        RuntimeError: If any query fails
    """
    # Define all queries as (name, callable) tuples
    queries = [
        ('kpis', q_kpis),
        ('sentiment_df', q_sentiment_dist),
        ('emotion_df', q_emotion_dist),
        ('topic_df', q_topic_dist),
        ('subreddit_df', lambda: q_subreddit_stats(top_n)),
        ('ts_df', q_timeseries),
        ('heat_df', lambda: q_sentiment_by_subreddit(top_n)),
        ('scatter_df', q_score_vs_comments_sample),
        ('top_posts_df', lambda: q_top_posts(top_n)),
    ]
    
    results = {}
    errors = {}
    
    # Use ThreadPoolExecutor with limited workers
    # Max workers = 5 is safe for most SQL Warehouse concurrency limits
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all queries
        future_to_name = {executor.submit(fn): name for name, fn in queries}
        
        # Collect results as they complete
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                errors[name] = e
    
    # If ANY query failed, raise immediately with all errors
    if errors:
        error_msg = "\n".join([f"  - {name}: {str(e)}" for name, e in errors.items()])
        raise RuntimeError(f"Query execution failed:\n{error_msg}")
    
    return results


try:
    with st.spinner("🚀 Loading data from Databricks (parallel execution)..."):
        # Execute all queries concurrently
        query_results = execute_queries_parallel()
        
        # Unpack results (preserves original variable names)
        kpis = query_results['kpis']
        sentiment_df = query_results['sentiment_df']
        emotion_df = query_results['emotion_df']
        topic_df = query_results['topic_df']
        subreddit_df = query_results['subreddit_df']
        ts_df = query_results['ts_df']
        heat_df = query_results['heat_df']
        scatter_df = query_results['scatter_df']
        top_posts_df = query_results['top_posts_df']
    
    DATA_OK = True
except ConnectionError as e:
    DATA_OK = False
    st.error(
        "❌ **Connection Failed**\n\n"
        "Cannot connect to Databricks SQL Warehouse. Please check:\n"
        "1. Environment variables are set (DATABRICKS_SERVER_HOSTNAME, etc.)\n"
        "2. SQL Warehouse is running\n"
        "3. Access token is valid\n\n"
        f"**Details:** {e}"
    )
except RuntimeError as e:
    DATA_OK = False
    st.error(
        "❌ **Query Failed**\n\n"
        "Failed to query data after multiple retries. Possible causes:\n"
        "1. SQL Warehouse is overwhelmed\n"
        f"2. View `{BASE_VIEW}` doesn't exist\n"
        "3. Network timeout\n\n"
        f"**Details:** {e}"
    )
except Exception as e:
    DATA_OK = False
    st.error(
        "❌ **Unexpected Error**\n\n"
        f"**Details:** {type(e).__name__}: {e}"
    )

if not DATA_OK:
    st.stop()

if kpis.empty or int(kpis.iloc[0].total_posts or 0) == 0:
    st.warning("No rows match the current filters. Widen the date range or clear filters.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════
k = kpis.iloc[0]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Posts", f"{int(k.total_posts):,}")
c2.metric("Total Comments", f"{int(k.total_comments or 0):,}")
c3.metric("Avg. Score", f"{k.avg_score:,.1f}")
c4.metric("Unique Subreddits", f"{int(k.unique_subreddits):,}")
c5.metric("% Positive Sentiment", f"{k.pct_positive:.1f}%")
c6.metric("% Bot Authors", f"{k.pct_bot:.1f}%")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# ROW 1 — SENTIMENT / EMOTION / TOPIC
# ══════════════════════════════════════════════════════════════════════════
r1c1, r1c2, r1c3 = st.columns([1, 1, 1.3])

with r1c1:
    st.subheader("Sentiment Split")
    fig = px.pie(
        sentiment_df, names="sentiment", values="n", hole=0.55, color="sentiment",
        color_discrete_map={"positive": GREEN, "neutral": MUTED, "negative": ORANGE},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    st.subheader("Emotion Breakdown")
    fig = px.bar(
        emotion_df.sort_values("n"), x="n", y="emotion", orientation="h",
        color="n", color_continuous_scale=[BLUE, ORANGE], template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                       coloraxis_showscale=False, xaxis_title="Posts", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with r1c3:
    st.subheader("Top Topics")
    fig = px.treemap(
        topic_df, path=["topic"], values="n", color="avg_score",
        color_continuous_scale=[BLUE, "#343536", ORANGE], template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# ROW 2 — ACTIVITY OVER TIME + SUBREDDIT LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════
r2c1, r2c2 = st.columns([1.6, 1])

with r2c1:
    st.subheader("Activity Over Time")
    fig = go.Figure()
    fig.add_bar(x=ts_df["day"], y=ts_df["posts"], name="Posts", marker_color=ORANGE, opacity=0.85)
    fig.add_scatter(x=ts_df["day"], y=ts_df["avg_score"], name="Avg. Score",
                     yaxis="y2", mode="lines+markers", line=dict(color=BLUE, width=2))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=360, margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(title="Posts"),
        yaxis2=dict(title="Avg. Score", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

with r2c2:
    st.subheader(f"Top {top_n} Subreddits")
    fig = px.bar(
        subreddit_df.sort_values("posts"), x="posts", y="subreddit", orientation="h",
        color="avg_sent_conf", color_continuous_scale=[MUTED, ORANGE], template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=360,
                       coloraxis_colorbar_title="Sent. Conf.", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# ROW 3 — SENTIMENT HEATMAP + SCORE/ENGAGEMENT SCATTER
# ══════════════════════════════════════════════════════════════════════════
r3c1, r3c2 = st.columns(2)

with r3c1:
    st.subheader("Sentiment Mix by Subreddit")
    if not heat_df.empty:
        pivot = heat_df.pivot(index="subreddit", columns="sentiment", values="n").fillna(0)
        fig = px.imshow(
            pivot, color_continuous_scale=[BG, ORANGE], aspect="auto",
            template=PLOTLY_TEMPLATE, labels=dict(color="Posts"),
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for the current filter selection.")

with r3c2:
    st.subheader("Score vs. Engagement (sampled)")
    fig = px.scatter(
        scatter_df, x="num_comments", y="score", color="sentiment", size="title_length",
        color_discrete_map={"positive": GREEN, "neutral": MUTED, "negative": ORANGE},
        opacity=0.65, template=PLOTLY_TEMPLATE, hover_data=["subreddit"],
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=360,
                       xaxis_title="Comments", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# ROW 4 — TOP POSTS TABLE
# ══════════════════════════════════════════════════════════════════════════
st.subheader(f"🏆 Top {top_n} Posts by Score")
st.dataframe(
    top_posts_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "url": st.column_config.LinkColumn("Link"),
        "score": st.column_config.NumberColumn("Score", format="%d"),
        "num_comments": st.column_config.NumberColumn("Comments", format="%d"),
    },
)

st.caption(
    "Data source: Databricks SQL views over curated Reddit post data · "
    "All aggregation runs in Databricks, not in this app · "
    "Results cached 10 min per filter combination."
)