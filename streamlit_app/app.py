import os
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from databricks import sql

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Reddit Intelligence",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# REDDIT / DARK HIGH-TECH THEME
# ============================================================

COLORS = {
    "reddit": "#FF4500",
    "reddit_light": "#FF6A33",
    "background": "#0B0D0F",
    "surface": "#111417",
    "surface_2": "#171A1E",
    "surface_3": "#1D2126",
    "border": "#292D33",
    "text": "#F1F3F5",
    "muted": "#8B949E",
    "green": "#46D369",
    "red": "#FF4D5A",
    "yellow": "#FFB020",
    "blue": "#4DA3FF",
    "purple": "#A970FF",
    "cyan": "#35D0BA",
}


st.markdown(
    f"""
<style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {{
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(255,69,0,0.08),
                transparent 28%
            ),
            {COLORS["background"]};
        color: {COLORS["text"]};
    }}

    .main {{
        background-color: {COLORS["background"]};
    }}

    .block-container {{
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }}

    h1, h2, h3, h4 {{
        color: {COLORS["text"]} !important;
        letter-spacing: -0.02em;
    }}

    p, span, label, div {{
        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }}

    .stMarkdown {{
        color: {COLORS["text"]};
    }}

    /* =========================
       SIDEBAR
       ========================= */

    [data-testid="stSidebar"] {{
        background:
            linear-gradient(
                180deg,
                #0A0C0E 0%,
                #101316 100%
            );
        border-right: 1px solid {COLORS["border"]};
    }}

    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1rem;
    }}

    [data-testid="stSidebar"] label {{
        color: {COLORS["muted"]} !important;
    }}

    [data-testid="stSidebar"] .stMarkdown {{
        color: {COLORS["text"]};
    }}

    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {{
        width: 100%;
        background: transparent;
        color: {COLORS["muted"]};
        border: 1px solid transparent;
        border-radius: 7px;
        font-weight: 600;
        text-align: left;
        transition: all 0.15s ease;
    }}

    .stButton > button:hover {{
        color: white;
        background: rgba(255,69,0,0.10);
        border-color: rgba(255,69,0,0.25);
    }}

    /* =========================
       METRICS
       ========================= */

    [data-testid="metric-container"] {{
        background: linear-gradient(
            145deg,
            {COLORS["surface"]},
            {COLORS["surface_2"]}
        );
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLORS["muted"]} !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    [data-testid="stMetricValue"] {{
        color: {COLORS["text"]} !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: 0.78rem !important;
    }}

    /* =========================
       INPUTS
       ========================= */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background-color: {COLORS["surface_2"]} !important;
        border-color: {COLORS["border"]} !important;
        color: {COLORS["text"]} !important;
    }}

    input {{
        color: {COLORS["text"]} !important;
    }}

    /* =========================
       DATAFRAME
       ========================= */

    [data-testid="stDataFrame"] {{
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* =========================
       SECTION HEADERS
       ========================= */

    .section-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 2rem 0 0.8rem 0;
        padding-bottom: 0.55rem;
        border-bottom: 1px solid {COLORS["border"]};
        color: {COLORS["text"]};
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }}

    .section-header::before {{
        content: "";
        display: block;
        width: 4px;
        height: 20px;
        background: {COLORS["reddit"]};
        border-radius: 4px;
    }}

    /* =========================
       BRAND
       ========================= */

    .brand {{
        padding: 0.7rem 0 1.2rem 0;
    }}

    .brand-icon {{
        font-size: 2.3rem;
        line-height: 1;
    }}

    .brand-title {{
        margin-top: 0.5rem;
        font-size: 1.25rem;
        font-weight: 800;
        color: white;
        letter-spacing: -0.03em;
    }}

    .brand-subtitle {{
        margin-top: 0.25rem;
        font-size: 0.72rem;
        color: {COLORS["muted"]};
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    /* =========================
       STATUS
       ========================= */

    .status {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 4px 9px;
        border-radius: 999px;
        background: rgba(70,211,105,0.08);
        border: 1px solid rgba(70,211,105,0.2);
        color: {COLORS["green"]};
        font-size: 0.72rem;
        font-weight: 700;
    }}

    .status-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: {COLORS["green"]};
        box-shadow: 0 0 8px {COLORS["green"]};
    }}

    /* =========================
       ALERT CARDS
       ========================= */

    .insight-card {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-left: 3px solid {COLORS["reddit"]};
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.7rem;
    }}

    .insight-title {{
        font-weight: 700;
        color: white;
        margin-bottom: 0.25rem;
    }}

    .insight-text {{
        color: {COLORS["muted"]};
        font-size: 0.85rem;
        line-height: 1.5;
    }}

    /* =========================
       TABS
       ========================= */

    button[data-baseweb="tab"] {{
        color: {COLORS["muted"]} !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {COLORS["reddit"]} !important;
    }}

    /* =========================
       DOWNLOAD BUTTON
       ========================= */

    .stDownloadButton > button {{
        background: {COLORS["reddit"]};
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 700;
    }}

    .stDownloadButton > button:hover {{
        background: {COLORS["reddit_light"]};
    }}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PLOTLY THEME
# ============================================================

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLORS["surface"],
        "plot_bgcolor": COLORS["surface"],
        "font": {
            "color": COLORS["text"],
            "family": "Arial, sans-serif",
        },
        "title": {
            "font": {
                "size": 16,
                "color": COLORS["text"],
            }
        },
        "xaxis": {
            "gridcolor": COLORS["border"],
            "linecolor": COLORS["border"],
            "zerolinecolor": COLORS["border"],
        },
        "yaxis": {
            "gridcolor": COLORS["border"],
            "linecolor": COLORS["border"],
            "zerolinecolor": COLORS["border"],
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": COLORS["muted"]},
        },
        "coloraxis": {
            "colorbar": {
                "tickfont": {"color": COLORS["muted"]},
            }
        },
    }
}


def apply_dark_theme(fig, height=400):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=20, r=20, t=55, b=30),
    )
    return fig


# ============================================================
# DATABASE
# ============================================================

@st.cache_resource
def get_databricks_connection():
    """Create a Databricks SQL connection."""

    required = [
        "DATABRICKS_SERVER_HOSTNAME",
        "DATABRICKS_HTTP_PATH",
        "DATABRICKS_TOKEN",
    ]

    missing = [key for key in required if not os.getenv(key)]

    if missing:
        st.error(
            "Missing Databricks environment variables: "
            + ", ".join(missing)
        )
        return None

    try:
        return sql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_TOKEN"),
        )
    except Exception as exc:
        st.error(f"Databricks connection failed: {exc}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def run_query(query):
    """Execute SQL and return a DataFrame."""

    conn = get_databricks_connection()

    if conn is None:
        return pd.DataFrame()

    cursor = None

    try:
        cursor = conn.cursor()
        cursor.execute(query)

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        return pd.DataFrame(rows, columns=columns)

    except Exception as exc:
        st.error(f"Query failed: {exc}")
        return pd.DataFrame()

    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass


# ============================================================
# HELPERS
# ============================================================

def sql_quote(value):
    """Safely quote a string for SQL IN clauses."""
    return "'" + str(value).replace("'", "''") + "'"


def sql_in(values):
    """Create a SQL IN list."""
    return ", ".join(sql_quote(value) for value in values)


def format_number(value):
    """Human-readable number formatting."""

    if value is None or pd.isna(value):
        return "0"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def format_percent(value):
    if value is None or pd.isna(value):
        return "0.0%"

    return f"{value:+.1f}%"


def sentiment_to_score(label):
    """
    Convert categorical sentiment to signed polarity.

    This is intentionally separate from model confidence.
    """

    mapping = {
        "Positive": 1.0,
        "Neutral": 0.0,
        "Negative": -1.0,
    }

    return mapping.get(str(label), 0.0)


def get_sentiment_label(score):
    if pd.isna(score):
        return "Neutral"

    if score > 0.2:
        return "Positive"

    if score < -0.2:
        return "Negative"

    return "Neutral"


def percentage_change(current, previous):
    if previous is None or pd.isna(previous) or previous == 0:
        return 0.0

    return ((current - previous) / previous) * 100


def safe_mean(series, default=0.0):
    if series is None or len(series) == 0:
        return default

    value = series.mean()

    if pd.isna(value):
        return default

    return float(value)


def safe_std(series, default=1.0):
    if series is None or len(series) == 0:
        return default

    value = series.std()

    if pd.isna(value) or value == 0:
        return default

    return float(value)


# ============================================================
# FILTER OPTIONS
# ============================================================

@st.cache_data(ttl=7200, show_spinner=False)
def get_filter_options():

    date_query = """
        SELECT
            MIN(DATE(created_at)) AS min_date,
            MAX(DATE(created_at)) AS max_date
        FROM workspace.redditrecon.posts_gold
        WHERE created_at IS NOT NULL
    """

    dates = run_query(date_query)

    if dates.empty:
        today = date.today()

        min_date = today - timedelta(days=30)
        max_date = today

    else:
        min_date = dates.iloc[0]["min_date"]
        max_date = dates.iloc[0]["max_date"]

        if pd.isna(min_date):
            min_date = date.today() - timedelta(days=30)

        if pd.isna(max_date):
            max_date = date.today()

        if isinstance(min_date, pd.Timestamp):
            min_date = min_date.date()

        if isinstance(max_date, pd.Timestamp):
            max_date = max_date.date()

    def get_distinct(column):
        query = f"""
            SELECT DISTINCT {column}
            FROM workspace.redditrecon.posts_gold
            WHERE {column} IS NOT NULL
            ORDER BY {column}
        """

        result = run_query(query)

        if result.empty:
            return []

        return result[column].dropna().tolist()

    return {
        "min_date": min_date,
        "max_date": max_date,
        "subreddits": get_distinct("subreddit"),
        "topics": get_distinct("topic"),
        "emotions": get_distinct("emotion"),
    }


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_main_data(
    start_date,
    end_date,
    subreddits=None,
    topics=None,
    sentiments=None,
    emotions=None,
):
    """
    Load ALL records in the selected date range.

    There is intentionally NO LIMIT clause here.

    The selected month/date range therefore determines the
    complete analytical population.
    """

    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    filters = [
        f"DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'"
    ]

    if subreddits:
        filters.append(
            f"subreddit IN ({sql_in(subreddits)})"
        )

    if topics:
        filters.append(
            f"topic IN ({sql_in(topics)})"
        )

    if sentiments:
        filters.append(
            f"sentiment IN ({sql_in(sentiments)})"
        )

    if emotions:
        filters.append(
            f"emotion IN ({sql_in(emotions)})"
        )

    where_clause = "\nAND ".join(filters)

    query = f"""
        SELECT
            id AS post_id,
            created_at,
            DATE(created_at) AS created_date,
            HOUR(created_at) AS hour,
            DAYOFWEEK(created_at) AS day_of_week,

            subreddit,
            author,
            title,
            selftext,

            LENGTH(
                CONCAT(
                    COALESCE(title, ''),
                    ' ',
                    COALESCE(selftext, '')
                )
            ) AS text_length,

            sentiment AS sentiment_label,
            sentiment_confidence AS sentiment_confidence,

            emotion AS emotion_label,
            emotion_confidence AS emotion_confidence,

            topic AS topic_label,
            topic_confidence AS topic_confidence,

            COALESCE(score, 0) AS post_score,
            COALESCE(num_comments, 0) AS num_comments,
            COALESCE(upvote_ratio, 0.5) AS upvote_ratio

        FROM workspace.redditrecon.posts_gold

        WHERE {where_clause}

        ORDER BY created_at DESC
    """

    df = run_query(query)

    if df.empty:
        return df

    # --------------------------------------------------------
    # Type cleanup
    # --------------------------------------------------------

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
    )

    df["created_date"] = pd.to_datetime(
        df["created_date"],
        errors="coerce",
    )

    numeric_columns = [
        "text_length",
        "sentiment_confidence",
        "emotion_confidence",
        "topic_confidence",
        "post_score",
        "num_comments",
        "upvote_ratio",
        "hour",
        "day_of_week",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Correct sentiment polarity
    # --------------------------------------------------------

    df["sentiment_score"] = (
        df["sentiment_label"]
        .map(
            {
                "Positive": 1.0,
                "Neutral": 0.0,
                "Negative": -1.0,
            }
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Engagement score
    #
    # Log transform prevents highly viral posts from
    # completely dominating averages.
    # --------------------------------------------------------

    df["post_score"] = df["post_score"].fillna(0).clip(lower=0)
    df["num_comments"] = df["num_comments"].fillna(0).clip(lower=0)

    df["engagement_score"] = (
        np.log1p(df["post_score"])
        + np.log1p(df["num_comments"])
    )

    # --------------------------------------------------------
    # Calendar dimensions
    # --------------------------------------------------------

    df["day_name"] = df["created_at"].dt.day_name()

    # --------------------------------------------------------
    # Text cleanup
    # --------------------------------------------------------

    df["title"] = df["title"].fillna("")
    df["subreddit"] = df["subreddit"].fillna("Unknown")
    df["topic_label"] = df["topic_label"].fillna("Unknown")
    df["emotion_label"] = df["emotion_label"].fillna("Unknown")
    df["sentiment_label"] = df["sentiment_label"].fillna("Neutral")

    return df


@st.cache_data(ttl=1800, show_spinner=False)
def get_previous_period_data(
    start_date,
    end_date,
    subreddits=None,
    topics=None,
    sentiments=None,
    emotions=None,
):
    """
    Get the immediately preceding period of identical duration.
    """

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    days = (end - start).days + 1

    previous_end = start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=days - 1)

    return load_main_data(
        previous_start,
        previous_end,
        subreddits,
        topics,
        sentiments,
        emotions,
    )


# ============================================================
# KPI CALCULATIONS
# ============================================================

def calculate_kpis(current_df, previous_df=None):

    if current_df.empty:
        return {
            "total_posts": 0,
            "avg_sentiment": 0,
            "total_engagement": 0,
            "avg_engagement": 0,
            "dominant_emotion": "N/A",
            "dominant_emotion_pct": 0,
            "trending_topic": "N/A",
            "trending_topic_count": 0,
            "posts_change": 0,
            "topic_change": 0,
        }

    topic_counts = (
        current_df["topic_label"]
        .value_counts()
    )

    emotion_counts = (
        current_df["emotion_label"]
        .value_counts()
    )

    trending_topic = (
        topic_counts.index[0]
        if not topic_counts.empty
        else "N/A"
    )

    dominant_emotion = (
        emotion_counts.index[0]
        if not emotion_counts.empty
        else "N/A"
    )

    dominant_emotion_pct = (
        emotion_counts.iloc[0] / len(current_df) * 100
        if not emotion_counts.empty
        else 0
    )

    kpis = {
        "total_posts": len(current_df),
        "avg_sentiment": safe_mean(
            current_df["sentiment_score"]
        ),
        "total_engagement": current_df[
            "engagement_score"
        ].sum(),
        "avg_engagement": safe_mean(
            current_df["engagement_score"]
        ),
        "dominant_emotion": dominant_emotion,
        "dominant_emotion_pct": dominant_emotion_pct,
        "trending_topic": trending_topic,
        "trending_topic_count": (
            int(topic_counts.iloc[0])
            if not topic_counts.empty
            else 0
        ),
        "posts_change": 0,
        "topic_change": 0,
    }

    if previous_df is not None and not previous_df.empty:

        kpis["posts_change"] = percentage_change(
            len(current_df),
            len(previous_df),
        )

        previous_topic_count = len(
            previous_df[
                previous_df["topic_label"]
                == trending_topic
            ]
        )

        kpis["topic_change"] = percentage_change(
            kpis["trending_topic_count"],
            previous_topic_count,
        )

    return kpis


# ============================================================
# TOPIC VELOCITY
# ============================================================

def calculate_topic_velocity(current_df, previous_df):

    current = (
        current_df.groupby("topic_label")
        .size()
        .reset_index(name="current_count")
    )

    previous = (
        previous_df.groupby("topic_label")
        .size()
        .reset_index(name="previous_count")
    )

    merged = current.merge(
        previous,
        on="topic_label",
        how="left",
    )

    merged["previous_count"] = (
        merged["previous_count"]
        .fillna(0)
    )

    # Absolute growth is useful when previous count = 0.
    merged["absolute_growth"] = (
        merged["current_count"]
        - merged["previous_count"]
    )

    # Smoothed percentage avoids divide-by-zero.
    merged["velocity"] = (
        (
            merged["current_count"]
            - merged["previous_count"]
        )
        /
        (
            merged["previous_count"] + 1
        )
        * 100
    )

    return merged.sort_values(
        "velocity",
        ascending=False,
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomalies(
    df,
    column="post_count",
    window=7,
    threshold=2.5,
):

    result = df.copy()

    if len(result) < 3:
        result["rolling_mean"] = result[column]
        result["rolling_std"] = 0
        result["z_score"] = 0
        result["is_anomaly"] = False
        return result

    actual_window = min(window, len(result))

    result["rolling_mean"] = (
        result[column]
        .rolling(
            window=actual_window,
            center=True,
            min_periods=2,
        )
        .mean()
    )

    result["rolling_std"] = (
        result[column]
        .rolling(
            window=actual_window,
            center=True,
            min_periods=2,
        )
        .std()
    )

    result["rolling_mean"] = (
        result["rolling_mean"]
        .fillna(result[column].mean())
    )

    result["rolling_std"] = (
        result["rolling_std"]
        .fillna(0)
    )

    result["z_score"] = (
        result[column]
        - result["rolling_mean"]
    ) / (
        result["rolling_std"] + 0.001
    )

    result["is_anomaly"] = (
        result["z_score"].abs()
        > threshold
    )

    return result


# ============================================================
# HEATMAP DATA
# ============================================================

def get_posting_time_heatmap_data(df):

    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby(
            ["day_name", "hour"],
            observed=True,
        )["engagement_score"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(
        index="day_name",
        columns="hour",
        values="engagement_score",
    )

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    pivot = pivot.reindex(
        [day for day in day_order if day in pivot.index]
    )

    return pivot


# ============================================================
# CHARTS
# ============================================================

def create_volume_trend(df):

    daily = (
        df.groupby("created_date")
        .size()
        .reset_index(name="post_count")
        .sort_values("created_date")
    )

    daily["rolling_avg"] = (
        daily["post_count"]
        .rolling(
            window=7,
            min_periods=1,
        )
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily["created_date"],
            y=daily["post_count"],
            mode="lines",
            name="Posts",
            line=dict(
                color=COLORS["reddit"],
                width=2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily["created_date"],
            y=daily["rolling_avg"],
            mode="lines",
            name="7D average",
            line=dict(
                color=COLORS["blue"],
                width=2,
                dash="dot",
            ),
        )
    )

    fig.update_layout(
        title="Conversation Volume",
        xaxis_title="",
        yaxis_title="Posts",
        hovermode="x unified",
    )

    return apply_dark_theme(fig)


def create_sentiment_over_time(df):

    grouped = (
        df.groupby(
            ["created_date", "sentiment_label"]
        )
        .size()
        .reset_index(name="count")
    )

    grouped["percentage"] = (
        grouped.groupby("created_date")["count"]
        .transform(
            lambda x: x / x.sum() * 100
        )
    )

    fig = px.area(
        grouped,
        x="created_date",
        y="percentage",
        color="sentiment_label",
        title="Sentiment Mix",
        color_discrete_map={
            "Positive": COLORS["green"],
            "Neutral": COLORS["muted"],
            "Negative": COLORS["red"],
        },
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Share (%)",
        hovermode="x unified",
    )

    return apply_dark_theme(fig)


def create_top_topics_chart(df, top_n=10):

    counts = (
        df["topic_label"]
        .value_counts()
        .head(top_n)
        .sort_values()
        .reset_index()
    )

    counts.columns = [
        "topic",
        "count",
    ]

    counts["percentage"] = (
        counts["count"]
        / len(df)
        * 100
    )

    fig = px.bar(
        counts,
        x="count",
        y="topic",
        orientation="h",
        title=f"Top {top_n} Topics",
        text="percentage",
    )

    fig.update_traces(
        marker_color=COLORS["reddit"],
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Posts",
        yaxis_title="",
    )

    return apply_dark_theme(fig)


def create_sentiment_heatmap(df):

    pivot = (
        df.groupby(
            ["topic_label", "sentiment_label"]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    if pivot.empty:
        return go.Figure()

    pivot_pct = (
        pivot
        .div(
            pivot.sum(axis=1),
            axis=0,
        )
        * 100
    )

    top_topics = (
        df["topic_label"]
        .value_counts()
        .head(10)
        .index
    )

    pivot_pct = pivot_pct.loc[
        pivot_pct.index.intersection(top_topics)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_pct.values,
            x=pivot_pct.columns,
            y=pivot_pct.index,
            colorscale=[
                [0.0, "#5A1820"],
                [0.5, "#292D33"],
                [1.0, "#1D8A4A"],
            ],
            text=np.round(
                pivot_pct.values,
                1,
            ),
            texttemplate="%{text}%",
            colorbar=dict(
                title="%",
            ),
        )
    )

    fig.update_layout(
        title="Sentiment by Topic",
        xaxis_title="",
        yaxis_title="",
    )

    return apply_dark_theme(fig, 480)


def create_emotion_distribution(df):

    counts = (
        df["emotion_label"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "emotion",
        "count",
    ]

    counts["percentage"] = (
        counts["count"]
        / len(df)
        * 100
    )

    fig = px.bar(
        counts,
        x="count",
        y="emotion",
        orientation="h",
        title="Emotion Distribution",
        text="percentage",
    )

    fig.update_traces(
        marker_color=COLORS["purple"],
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Posts",
        yaxis_title="",
    )

    return apply_dark_theme(fig)


def create_engagement_by_topic(df, top_n=10):

    grouped = (
        df.groupby("topic_label")
        .agg(
            avg_engagement=(
                "engagement_score",
                "mean",
            ),
            post_count=(
                "post_id",
                "count",
            ),
        )
        .reset_index()
        .sort_values(
            "avg_engagement",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "avg_engagement"
        )
    )

    fig = px.bar(
        grouped,
        x="avg_engagement",
        y="topic_label",
        orientation="h",
        title=f"Highest Engagement Topics",
    )

    fig.update_traces(
        marker_color=COLORS["blue"]
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Average Engagement",
        yaxis_title="",
    )

    return apply_dark_theme(fig)


def create_time_heatmap(pivot):

    if pivot.empty:
        return go.Figure()

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0.0, "#111417"],
                [0.45, "#6B2715"],
                [1.0, COLORS["reddit"]],
            ],
            colorbar=dict(
                title="Engagement"
            ),
        )
    )

    fig.update_layout(
        title="Engagement by Day & Hour",
        xaxis_title="Hour",
        yaxis_title="",
    )

    return apply_dark_theme(fig)


def create_box_plot(
    df,
    x_col,
    y_col,
    title,
):

    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        color=x_col,
        title=title,
        color_discrete_map={
            "Positive": COLORS["green"],
            "Neutral": COLORS["muted"],
            "Negative": COLORS["red"],
        },
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title="Engagement",
    )

    return apply_dark_theme(fig)


def create_scatter_plot(
    df,
    x_col,
    y_col,
    title,
    color_col=None,
):

    sample_size = min(
        5000,
        len(df),
    )

    sample = (
        df.sample(
            sample_size,
            random_state=42,
        )
        if len(df) > sample_size
        else df
    )

    fig = px.scatter(
        sample,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        opacity=0.55,
    )

    fig.update_layout(
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
    )

    return apply_dark_theme(fig)


def create_anomaly_chart(df):

    fig = go.Figure()

    normal = df[
        ~df["is_anomaly"]
    ]

    anomalies = df[
        df["is_anomaly"]
    ]

    fig.add_trace(
        go.Scatter(
            x=normal["created_date"],
            y=normal["post_count"],
            mode="lines+markers",
            name="Normal",
            line=dict(
                color=COLORS["blue"],
                width=2,
            ),
        )
    )

    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["created_date"],
                y=anomalies["post_count"],
                mode="markers",
                name="Anomaly",
                marker=dict(
                    color=COLORS["reddit"],
                    size=12,
                    symbol="x",
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df["created_date"],
            y=df["rolling_mean"],
            mode="lines",
            name="Expected",
            line=dict(
                color=COLORS["muted"],
                width=2,
                dash="dot",
            ),
        )
    )

    fig.update_layout(
        title="Conversation Volume Anomalies",
        xaxis_title="",
        yaxis_title="Posts",
        hovermode="x unified",
    )

    return apply_dark_theme(fig)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🔴</div>
            <div class="brand-title">REDDIT INTELLIGENCE</div>
            <div class="brand-subtitle">
                Conversation Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="status">
            <span class="status-dot"></span>
            DATA PIPELINE ONLINE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    if "page" not in st.session_state:
        st.session_state.page = "overview"

    pages = {
        "overview": "◉  Overview",
        "sentiment": "◌  Sentiment",
        "topics": "◇  Topics",
        "emotions": "◈  Emotions",
        "engagement": "⚡  Engagement",
        "predictive": "△  Prediction",
        "anomaly": "⚠  Anomalies",
    }

    st.markdown(
        '<div style="color:#8B949E;font-size:.68rem;'
        'font-weight:700;letter-spacing:.1em;margin-bottom:.5rem">'
        'NAVIGATION</div>',
        unsafe_allow_html=True,
    )

    for key, label in pages.items():

        if st.button(
            label,
            key=f"nav_{key}",
        ):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    st.markdown(
        '<div style="color:#8B949E;font-size:.68rem;'
        'font-weight:700;letter-spacing:.1em;margin-bottom:.7rem">'
        'ANALYSIS WINDOW</div>',
        unsafe_allow_html=True,
    )

    filter_options = get_filter_options()

    default_start = max(
        filter_options["min_date"],
        filter_options["max_date"]
        - timedelta(days=29),
    )

    start_date = st.date_input(
        "Start",
        value=default_start,
        min_value=filter_options["min_date"],
        max_value=filter_options["max_date"],
    )

    end_date = st.date_input(
        "End",
        value=filter_options["max_date"],
        min_value=filter_options["min_date"],
        max_value=filter_options["max_date"],
    )

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    selected_subreddits = st.multiselect(
        "Subreddits",
        filter_options["subreddits"],
    )

    selected_topics = st.multiselect(
        "Topics",
        filter_options["topics"],
    )

    selected_sentiments = st.multiselect(
        "Sentiment",
        [
            "Positive",
            "Neutral",
            "Negative",
        ],
    )

    selected_emotions = st.multiselect(
        "Emotions",
        filter_options["emotions"],
    )

    st.markdown("---")

    if st.button("↻  Reset Filters"):
        for key in [
            "selected_subreddits",
            "selected_topics",
            "selected_sentiments",
            "selected_emotions",
        ]:
            st.session_state.pop(
                key,
                None,
            )

        st.session_state.page = "overview"
        st.rerun()


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner("Loading conversation intelligence..."):

    df = load_main_data(
        start_date,
        end_date,
        selected_subreddits or None,
        selected_topics or None,
        selected_sentiments or None,
        selected_emotions or None,
    )

    df_prev = get_previous_period_data(
        start_date,
        end_date,
        selected_subreddits or None,
        selected_topics or None,
        selected_sentiments or None,
        selected_emotions or None,
    )


if df.empty:

    st.markdown(
        """
        <div style="
            margin-top:4rem;
            text-align:center;
            padding:3rem;
            border:1px solid #292D33;
            border-radius:12px;
            background:#111417;
        ">
            <div style="font-size:3rem;">◌</div>
            <h2>No conversations found</h2>
            <p style="color:#8B949E;">
                No Reddit posts match the selected analysis window
                and filters.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


kpis = calculate_kpis(
    df,
    df_prev,
)


# ============================================================
# GLOBAL HEADER
# ============================================================

st.markdown(
    f"""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:flex-end;
        margin-bottom:1.2rem;
    ">
        <div>
            <div style="
                color:{COLORS["reddit"]};
                font-size:.7rem;
                font-weight:800;
                letter-spacing:.14em;
                margin-bottom:.35rem;
            ">
                REDDIT / CONVERSATION INTELLIGENCE
            </div>

            <h1 style="
                margin:0;
                font-size:2rem;
            ">
                {pages[st.session_state.page].replace("◉ ","").replace("◌ ","").replace("◇ ","").replace("◈ ","").replace("⚡ ","").replace("△ ","").replace("⚠ ","")}
            </h1>

            <div style="
                color:{COLORS["muted"]};
                margin-top:.35rem;
                font-size:.85rem;
            ">
                {start_date.strftime("%d %b %Y")}
                →
                {end_date.strftime("%d %b %Y")}
                &nbsp; • &nbsp;
                {format_number(len(df))} conversations
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if st.session_state.page == "overview":

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "CONVERSATIONS",
            format_number(
                kpis["total_posts"]
            ),
            format_percent(
                kpis["posts_change"]
            ),
        )

    with col2:
        sentiment = get_sentiment_label(
            kpis["avg_sentiment"]
        )

        st.metric(
            "AVG SENTIMENT",
            f"{kpis['avg_sentiment']:+.2f}",
            sentiment,
        )

    with col3:
        st.metric(
            "TOTAL ENGAGEMENT",
            format_number(
                kpis["total_engagement"]
            ),
        )

    with col4:
        st.metric(
            "TRENDING TOPIC",
            str(
                kpis["trending_topic"]
            )[:24],
            format_percent(
                kpis["topic_change"]
            ),
        )

    with col5:
        st.metric(
            "DOMINANT EMOTION",
            str(
                kpis["dominant_emotion"]
            )[:20],
            f"{kpis['dominant_emotion_pct']:.0f}% share",
        )

    st.markdown(
        '<div class="section-header">'
        'Conversation Pulse'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            create_volume_trend(df),
            use_container_width=True,
        )

    with c2:
        st.plotly_chart(
            create_sentiment_over_time(df),
            use_container_width=True,
        )

    st.markdown(
        '<div class="section-header">'
        'Topic Landscape'
        '</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        create_top_topics_chart(
            df,
            15,
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Full dataset
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        'All Conversations'
        '</div>',
        unsafe_allow_html=True,
    )

    display_columns = [
        "created_at",
        "subreddit",
        "title",
        "topic_label",
        "sentiment_label",
        "emotion_label",
        "post_score",
        "num_comments",
        "engagement_score",
    ]

    display_df = df[
        [
            column
            for column in display_columns
            if column in df.columns
        ]
    ].copy()

    display_df.columns = [
        "Created",
        "Subreddit",
        "Title",
        "Topic",
        "Sentiment",
        "Emotion",
        "Score",
        "Comments",
        "Engagement",
    ]

    st.caption(
        f"Showing all {len(display_df):,} conversations "
        "matching the selected filters. No top-N row limit is applied."
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        hide_index=True,
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "↓  Download Full Filtered Dataset",
        csv_data,
        file_name=(
            f"reddit_intelligence_"
            f"{start_date}_"
            f"{end_date}.csv"
        ),
        mime="text/csv",
    )


# ============================================================
# SENTIMENT
# ============================================================

elif st.session_state.page == "sentiment":

    c1, c2 = st.columns(2)

    with c1:

        sentiment_dist = (
            df["sentiment_label"]
            .value_counts()
            .reset_index()
        )

        sentiment_dist.columns = [
            "sentiment",
            "count",
        ]

        sentiment_dist["percentage"] = (
            sentiment_dist["count"]
            / len(df)
            * 100
        )

        fig = px.bar(
            sentiment_dist.sort_values(
                "count"
            ),
            x="count",
            y="sentiment",
            orientation="h",
            title="Sentiment Distribution",
            text="percentage",
            color="sentiment",
            color_discrete_map={
                "Positive": COLORS["green"],
                "Neutral": COLORS["muted"],
                "Negative": COLORS["red"],
            },
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        fig.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            apply_dark_theme(fig, 380),
            use_container_width=True,
        )

    with c2:

        trend = (
            df.groupby("created_date")
            ["sentiment_score"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            trend,
            x="created_date",
            y="sentiment_score",
            title="Sentiment Polarity Over Time",
            markers=True,
        )

        fig.update_traces(
            line_color=COLORS["reddit"]
        )

        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color=COLORS["muted"],
        )

        fig.update_layout(
            yaxis_title="Polarity",
            xaxis_title="",
        )

        st.plotly_chart(
            apply_dark_theme(fig, 380),
            use_container_width=True,
        )

    st.markdown(
        '<div class="section-header">'
        'Sentiment by Topic'
        '</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        create_sentiment_heatmap(df),
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-header">'
        'Engagement vs Sentiment'
        '</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        create_box_plot(
            df,
            "sentiment_label",
            "engagement_score",
            "Engagement Distribution by Sentiment",
        ),
        use_container_width=True,
    )


# ============================================================
# TOPICS
# ============================================================

elif st.session_state.page == "topics":

    st.plotly_chart(
        create_top_topics_chart(
            df,
            15,
        ),
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-header">'
        'Topic Evolution'
        '</div>',
        unsafe_allow_html=True,
    )

    top_topics = (
        df["topic_label"]
        .value_counts()
        .head(8)
        .index
        .tolist()
    )

    topic_time = (
        df[
            df["topic_label"]
            .isin(top_topics)
        ]
        .groupby(
            [
                "created_date",
                "topic_label",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    fig = px.line(
        topic_time,
        x="created_date",
        y="count",
        color="topic_label",
        title="Topic Volume Over Time",
    )

    fig.update_layout(
        hovermode="x unified",
        xaxis_title="",
        yaxis_title="Posts",
    )

    st.plotly_chart(
        apply_dark_theme(fig),
        use_container_width=True,
    )

    if not df_prev.empty:

        st.markdown(
            '<div class="section-header">'
            'Topic Velocity'
            '</div>',
            unsafe_allow_html=True,
        )

        velocity = calculate_topic_velocity(
            df,
            df_prev,
        ).head(15)

        fig = px.bar(
            velocity.sort_values(
                "velocity"
            ),
            x="velocity",
            y="topic_label",
            orientation="h",
            text="velocity",
            title="Topic Growth vs Previous Period",
        )

        fig.update_traces(
            marker_color=COLORS["reddit"],
            texttemplate="%{text:+.0f}%",
            textposition="outside",
        )

        fig.update_layout(
            xaxis_title="Growth",
            yaxis_title="",
        )

        st.plotly_chart(
            apply_dark_theme(fig, 500),
            use_container_width=True,
        )


# ============================================================
# EMOTIONS
# ============================================================

elif st.session_state.page == "emotions":

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            create_emotion_distribution(df),
            use_container_width=True,
        )

    with c2:

        top_emotions = (
            df["emotion_label"]
            .value_counts()
            .head(6)
            .index
        )

        emotion_time = (
            df[
                df["emotion_label"]
                .isin(top_emotions)
            ]
            .groupby(
                [
                    "created_date",
                    "emotion_label",
                ]
            )
            .size()
            .reset_index(name="count")
        )

        emotion_time["percentage"] = (
            emotion_time.groupby(
                "created_date"
            )["count"]
            .transform(
                lambda x: x / x.sum() * 100
            )
        )

        fig = px.area(
            emotion_time,
            x="created_date",
            y="percentage",
            color="emotion_label",
            title="Emotion Mix Over Time",
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Share (%)",
        )

        st.plotly_chart(
            apply_dark_theme(fig),
            use_container_width=True,
        )

    st.markdown(
        '<div class="section-header">'
        'Emotion × Engagement'
        '</div>',
        unsafe_allow_html=True,
    )

    emotion_engagement = (
        df.groupby("emotion_label")
        .agg(
            avg_engagement=(
                "engagement_score",
                "mean",
            ),
            avg_score=(
                "post_score",
                "mean",
            ),
            avg_comments=(
                "num_comments",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "avg_engagement",
            ascending=False,
        )
    )

    fig = px.bar(
        emotion_engagement,
        x="avg_engagement",
        y="emotion_label",
        orientation="h",
        title="Average Engagement by Emotion",
    )

    fig.update_traces(
        marker_color=COLORS["purple"]
    )

    st.plotly_chart(
        apply_dark_theme(fig),
        use_container_width=True,
    )


# ============================================================
# ENGAGEMENT
# ============================================================

elif st.session_state.page == "engagement":

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "AVG SCORE",
            f"{safe_mean(df['post_score']):.1f}",
        )

    with c2:
        st.metric(
            "AVG COMMENTS",
            f"{safe_mean(df['num_comments']):.1f}",
        )

    with c3:
        st.metric(
            "AVG ENGAGEMENT",
            f"{safe_mean(df['engagement_score']):.2f}",
        )

    with c4:
        topic_engagement = (
            df.groupby("topic_label")
            ["engagement_score"]
            .mean()
        )

        best_topic = (
            topic_engagement.idxmax()
            if not topic_engagement.empty
            else "N/A"
        )

        st.metric(
            "TOP ENGAGEMENT TOPIC",
            str(best_topic)[:20],
        )

    st.markdown(
        '<div class="section-header">'
        'Topic Engagement'
        '</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        create_engagement_by_topic(df),
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-header">'
        'Optimal Posting Windows'
        '</div>',
        unsafe_allow_html=True,
    )

    pivot = get_posting_time_heatmap_data(df)

    st.plotly_chart(
        create_time_heatmap(pivot),
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-header">'
        'Content Characteristics'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            create_scatter_plot(
                df,
                "text_length",
                "engagement_score",
                "Text Length vs Engagement",
                "sentiment_label",
            ),
            use_container_width=True,
        )

    with c2:
        st.plotly_chart(
            create_box_plot(
                df,
                "sentiment_label",
                "engagement_score",
                "Engagement by Sentiment",
            ),
            use_container_width=True,
        )


# ============================================================
# PREDICTIVE ANALYTICS
# ============================================================

elif st.session_state.page == "predictive":

    st.markdown(
        '<div class="section-header">'
        'Engagement Intelligence'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Data-driven drivers
    # --------------------------------------------------------

    feature_scores = []

    overall_engagement = safe_mean(
        df["engagement_score"]
    )

    # Topic effect
    topic_means = (
        df.groupby("topic_label")
        ["engagement_score"]
        .mean()
    )

    if len(topic_means) > 1:
        feature_scores.append(
            (
                "Topic",
                abs(
                    topic_means.std()
                    / (
                        abs(
                            overall_engagement
                        ) + 0.001
                    )
                ),
            )
        )

    # Hour effect
    hour_means = (
        df.groupby("hour")
        ["engagement_score"]
        .mean()
    )

    if len(hour_means) > 1:
        feature_scores.append(
            (
                "Posting Hour",
                abs(
                    hour_means.std()
                    / (
                        abs(
                            overall_engagement
                        ) + 0.001
                    )
                ),
            )
        )

    # Text length correlation
    if (
        df["text_length"].nunique() > 1
        and df["engagement_score"].nunique() > 1
    ):
        corr = (
            df[
                [
                    "text_length",
                    "engagement_score",
                ]
            ]
            .corr()
            .iloc[0, 1]
        )

        feature_scores.append(
            (
                "Text Length",
                abs(float(corr)),
            )
        )

    # Sentiment effect
    sentiment_means = (
        df.groupby("sentiment_label")
        ["engagement_score"]
        .mean()
    )

    if len(sentiment_means) > 1:
        feature_scores.append(
            (
                "Sentiment",
                abs(
                    sentiment_means.std()
                    / (
                        abs(
                            overall_engagement
                        ) + 0.001
                    )
                ),
            )
        )

    # Emotion effect
    emotion_means = (
        df.groupby("emotion_label")
        ["engagement_score"]
        .mean()
    )

    if len(emotion_means) > 1:
        feature_scores.append(
            (
                "Emotion",
                abs(
                    emotion_means.std()
                    / (
                        abs(
                            overall_engagement
                        ) + 0.001
                    )
                ),
            )
        )

    feature_importance = pd.DataFrame(
        feature_scores,
        columns=[
            "Feature",
            "Importance",
        ],
    )

    if not feature_importance.empty:

        feature_importance["Importance"] = (
            feature_importance["Importance"]
            /
            feature_importance["Importance"].sum()
        )

        feature_importance = (
            feature_importance
            .sort_values("Importance")
        )

        fig = px.bar(
            feature_importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Observed Engagement Drivers",
        )

        fig.update_traces(
            marker_color=COLORS["reddit"]
        )

        st.plotly_chart(
            apply_dark_theme(fig),
            use_container_width=True,
        )

        st.caption(
            "These are descriptive effect measures derived "
            "from the selected data, not trained ML feature "
            "importance scores."
        )

    # --------------------------------------------------------
    # Simulator
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        'Engagement Simulator'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    topics = (
        df["topic_label"]
        .dropna()
        .unique()
        .tolist()
    )

    emotions = (
        df["emotion_label"]
        .dropna()
        .unique()
        .tolist()
    )

    with c1:

        sim_topic = st.selectbox(
            "Topic",
            topics[:50],
        )

        sim_sentiment = st.selectbox(
            "Sentiment",
            [
                "Positive",
                "Neutral",
                "Negative",
            ],
        )

        sim_emotion = st.selectbox(
            "Emotion",
            emotions[:50],
        )

    with c2:

        sim_hour = st.slider(
            "Posting Hour",
            0,
            23,
            18,
        )

        sim_day = st.selectbox(
            "Day",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )

        sim_length = st.slider(
            "Text Length",
            10,
            5000,
            300,
        )

    with c3:

        st.write("")

        if st.button(
            "◉  Estimate Engagement",
            use_container_width=True,
        ):

            # Topic effect
            topic_baseline = (
                safe_mean(
                    df[
                        df["topic_label"]
                        == sim_topic
                    ]["engagement_score"]
                )
            )

            # Hour effect
            hour_baseline = (
                safe_mean(
                    df[
                        df["hour"]
                        == sim_hour
                    ]["engagement_score"],
                    overall_engagement,
                )
            )

            # Day effect
            day_baseline = (
                safe_mean(
                    df[
                        df["day_name"]
                        == sim_day
                    ]["engagement_score"],
                    overall_engagement,
                )
            )

            # Sentiment effect
            sentiment_baseline = (
                safe_mean(
                    df[
                        df["sentiment_label"]
                        == sim_sentiment
                    ]["engagement_score"],
                    overall_engagement,
                )
            )

            # Text length effect
            length_corr = 0

            if (
                df["text_length"].nunique() > 1
                and df["engagement_score"].nunique() > 1
            ):
                length_corr = (
                    df[
                        [
                            "text_length",
                            "engagement_score",
                        ]
                    ]
                    .corr()
                    .iloc[0, 1]
                )

            median_length = (
                df["text_length"]
                .median()
            )

            length_effect = (
                length_corr
                * (
                    sim_length
                    - median_length
                )
                / (
                    median_length
                    + 1
                )
            )

            estimated = (
                overall_engagement
                + 0.35
                * (
                    topic_baseline
                    - overall_engagement
                )
                + 0.20
                * (
                    hour_baseline
                    - overall_engagement
                )
                + 0.15
                * (
                    day_baseline
                    - overall_engagement
                )
                + 0.15
                * (
                    sentiment_baseline
                    - overall_engagement
                )
                + length_effect
            )

            percentile = (
                df["engagement_score"]
                <= estimated
            ).mean() * 100

            st.metric(
                "ESTIMATED ENGAGEMENT",
                f"{estimated:.2f}",
            )

            st.metric(
                "ESTIMATED PERCENTILE",
                f"{percentile:.0f}th",
            )


# ============================================================
# ANOMALIES
# ============================================================

elif st.session_state.page == "anomaly":

    daily = (
        df.groupby("created_date")
        .size()
        .reset_index(name="post_count")
        .sort_values("created_date")
    )

    daily = detect_anomalies(
        daily,
        "post_count",
    )

    anomaly_count = int(
        daily["is_anomaly"].sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "ANOMALIES",
            anomaly_count,
        )

    with c2:

        if anomaly_count:
            largest = daily[
                daily["is_anomaly"]
            ]["post_count"].max()

            st.metric(
                "LARGEST SPIKE",
                format_number(largest),
            )

        else:
            st.metric(
                "LARGEST SPIKE",
                "N/A",
            )

    with c3:

        recent_cutoff = (
            df["created_date"].max()
            - timedelta(days=2)
        )

        recent_sentiment = safe_mean(
            df[
                df["created_date"]
                >= recent_cutoff
            ]["sentiment_score"]
        )

        overall_sentiment = safe_mean(
            df["sentiment_score"]
        )

        shift = (
            recent_sentiment
            - overall_sentiment
        )

        st.metric(
            "SENTIMENT SHIFT",
            f"{shift:+.3f}",
        )

    with c4:

        if anomaly_count >= 4:
            level = "HIGH"
            icon = "🔴"
        elif anomaly_count > 0:
            level = "MEDIUM"
            icon = "🟠"
        else:
            level = "LOW"
            icon = "🟢"

        st.metric(
            "ALERT LEVEL",
            f"{icon} {level}",
        )

    st.markdown(
        '<div class="section-header">'
        'Conversation Volume'
        '</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        create_anomaly_chart(daily),
        use_container_width=True,
    )

    if anomaly_count:

        st.markdown(
            '<div class="section-header">'
            'Detected Events'
            '</div>',
            unsafe_allow_html=True,
        )

        anomaly_table = daily[
            daily["is_anomaly"]
        ][
            [
                "created_date",
                "post_count",
                "rolling_mean",
                "z_score",
            ]
        ].copy()

        anomaly_table.columns = [
            "Date",
            "Posts",
            "Expected",
            "Z Score",
        ]

        st.dataframe(
            anomaly_table,
            use_container_width=True,
            hide_index=True,
        )

    if not df_prev.empty:

        st.markdown(
            '<div class="section-header">'
            'Topic Spikes'
            '</div>',
            unsafe_allow_html=True,
        )

        velocity = calculate_topic_velocity(
            df,
            df_prev,
        )

        spikes = velocity[
            velocity["velocity"] > 100
        ].head(10)

        if not spikes.empty:

            fig = px.bar(
                spikes.sort_values(
                    "velocity"
                ),
                x="velocity",
                y="topic_label",
                orientation="h",
                text="velocity",
                title="Topics Growing >100%",
            )

            fig.update_traces(
                marker_color=COLORS["red"],
                texttemplate="%{text:+.0f}%",
                textposition="outside",
            )

            st.plotly_chart(
                apply_dark_theme(fig),
                use_container_width=True,
            )

        else:
            st.info(
                "No topics exceeded 100% growth "
                "versus the previous period."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div style="
        margin-top:3rem;
        padding-top:1rem;
        border-top:1px solid {COLORS["border"]};
        display:flex;
        justify-content:space-between;
        color:{COLORS["muted"]};
        font-size:.72rem;
    ">
        <span>🔴 REDDIT INTELLIGENCE</span>
        <span>
            Databricks Lakehouse • Streamlit • NLP Analytics
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
