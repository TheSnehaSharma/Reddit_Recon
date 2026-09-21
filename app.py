# =============================================================================
# REDDIT RECON
# Reddit Community Intelligence Dashboard
# =============================================================================
#
# Pipeline:
# Reddit API
#     ↓
# Cleaning
#     ↓
# Sentiment Classification
#     ↓
# Emotion Classification
#     ↓
# Sentence Embeddings
#     ↓
# KMeans Topic Discovery
#     ↓
# Topic Keywords
#     ↓
# Groq Topic Analysis
#     ↓
# Streamlit Dashboard
#
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import torch

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.metrics import silhouette_score

from transformers import (
    AutoModel,
    AutoTokenizer,
    pipeline,
)

from wordcloud import WordCloud


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

ARCTIC_URL = (
    "https://arctic-shift.photon-reddit.com/api/posts/search"
)

SENTIMENT_MODEL = (
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

EMOTION_MODEL = (
    "j-hartmann/emotion-english-distilroberta-base"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = "openai/gpt-oss-20b"


DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 3
RANDOM_STATE = 42

MAX_FETCH_RETRIES = 4
MAX_TEXT_CHARS = 2500
EMBEDDING_BATCH_SIZE = 32

ROLLING_WINDOW = "1D"


# =============================================================================
# 3. DASHBOARD COLORS
# =============================================================================

BG = "#111111"
CARD_BG = "#151515"
BORDER = "#2A2A2A"
TEXT_MUTED = "#A3A3A3"


# -----------------------------------------------------------------------------
# Topic colors
# -----------------------------------------------------------------------------

TOPIC_PALETTE = [
    "#06B6D4",
    "#A855F7",
    "#F97316",
    "#14B8A6",
    "#EAB308",
    "#EC4899",
    "#0EA5E9",
    "#84CC16",
    "#8B5CF6",
    "#F43F5E",
]


# -----------------------------------------------------------------------------
# Sentiment colors
# -----------------------------------------------------------------------------

SENTIMENT_COLORS = {
    "Positive": "#22C55E",
    "Neutral": "#64748B",
    "Negative": "#EF4444",
}


# -----------------------------------------------------------------------------
# Emotion colors
# -----------------------------------------------------------------------------

EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


EMOTION_COLORS = {
    "joy": "#F59E0B",
    "surprise": "#8B5CF6",
    "sadness": "#3B82F6",
    "anger": "#DC2626",
    "fear": "#6366F1",
    "disgust": "#10B981",
    "neutral": "#94A3B8",
}


# -----------------------------------------------------------------------------
# Stopwords
# -----------------------------------------------------------------------------

EXTRA_STOPWORDS = {
    "reddit",
    "post",
    "posts",
    "people",
    "really",
    "just",
    "like",
    "think",
    "thing",
    "things",
    "want",
    "got",
    "get",
    "going",
    "does",
    "did",
    "said",
    "say",
    "know",
    "use",
    "used",
    "using",
}


# =============================================================================
# 4. STREAMLIT PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Reddit Recon",
    page_icon="https://www.reddit.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 5. GLOBAL CSS
# =============================================================================

def inject_css():
    """Inject dashboard-wide CSS."""

    st.html(
        f"""
        <style>

        .stApp,
        .main {{
            background: {BG};
        }}

        html,
        body,
        [class*="css"],
        p,
        span,
        div,
        label,
        li,
        td,
        th {{
            color: #FFFFFF;
        }}

        h1,
        h2,
        h3,
        h4,
        h5,
        h6 {{
            color: #FFFFFF !important;
        }}

        section[data-testid="stSidebar"] {{
            background: {BG};
            border-right: 1px solid {BORDER};
        }}

        section[data-testid="stSidebar"] * {{
            color: #FFFFFF !important;
        }}

        .reddit-header {{
            display: flex;
            align-items: center;
            padding: 1.2rem 0;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid {BORDER};
        }}

        .reddit-header .reddit-icon {{
            font-size: 58px;
            margin-right: 20px;
            width: 58px;
            flex-shrink: 0;
        }}

        .reddit-header h1 {{
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
        }}

        .reddit-header p {{
            margin: 5px 0 0;
            font-size: .95rem;
            color: #A3A3A3 !important;
        }}

        .reddit-card {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }}

        .reddit-card h2 {{
            margin-top: 0;
            font-size: 1.4rem;
        }}

        .topic-title {{
            font-weight: 700;
            font-size: 1.35rem;
            margin-bottom: .75rem;
        }}

        .topic-title::before {{
            content: "\\25CF";
            margin-right: 10px;
            color: #FF4500;
        }}

        .badge {{
            display: inline-block;
            padding: .3rem .7rem;
            border-radius: 999px;
            font-size: .75rem;
            font-weight: 700;
            margin: 0 .5rem .4rem 0;
            border: 1px solid #444;
            background: #1C1C1C;
        }}

        a,
        a:hover {{
            color: #FFFFFF !important;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        input,
        textarea,
        select,
        button {{
            color: #FFFFFF !important;
            background-color: #181818 !important;
        }}

        hr {{
            border-color: {BORDER} !important;
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
        }}

        [data-testid="stMetric"] {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1rem;
        }}

        [data-testid="stMetricValue"] {{
            color: #FFFFFF !important;
            font-size: 2rem;
            font-weight: 700;
        }}

        [data-testid="stMetricLabel"] {{
            color: #BDBDBD !important;
        }}

        div[data-baseweb="select"] *,
        div[data-baseweb="input"] * {{
            color: #FFFFFF !important;
            background-color: #181818 !important;
        }}

        .js-plotly-plot {{
            border-radius: 8px;
        }}

        .reddit-sidebar-icon {{
            font-size: 46px;
            text-align: center;
            margin: 10px 0 20px;
        }}

        .section-label {{
            color: #AAAAAA !important;
            font-size: .78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .08em;
        }}

        .top-post-link {{
            color: #FFFFFF !important;
            text-decoration: none;
        }}

        .top-post-link:hover {{
            text-decoration: underline;
        }}

        .confidence-wrapper {{
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
        }}

        .confidence-title {{
            font-size: .82rem;
            font-weight: 600;
            color: #A3A3A3 !important;
            margin-bottom: .45rem;
        }}

        .confidence-track {{
            display: flex;
            width: 100%;
            height: 18px;
            border-radius: 999px;
            overflow: hidden;
            background: #252525;
            border: 1px solid #333;
        }}

        .confidence-segment {{
            flex: 1;
            position: relative;
            background: #222;
            border-right: 1px solid #111;
            overflow: hidden;
        }}

        .confidence-segment:last-child {{
            border-right: none;
        }}

        .confidence-fill {{
            height: 100%;
            border-radius: 999px;
        }}

        .confidence-labels {{
            display: flex;
            width: 100%;
            margin-top: .4rem;
        }}

        .confidence-label {{
            flex: 1;
            text-align: center;
            font-size: .72rem;
            color: #A3A3A3 !important;
        }}

        .legend-row {{
            display: flex;
            flex-wrap: wrap;
            gap: .65rem 1rem;
            margin-top: .5rem;
            margin-bottom: .4rem;
        }}

        .legend-item {{
            font-size: .78rem;
            white-space: nowrap;
        }}

        .legend-dot {{
            font-size: 1rem;
        }}

        </style>
        """
    )


inject_css()


# =============================================================================
# 6. GENERIC HELPERS
# =============================================================================

def safe_int(value):
    """Safely convert a value to integer."""

    try:
        return int(value)
    except Exception:
        return 0


def normalize_subreddit(value):
    """Normalize a subreddit name."""

    value = value.strip()

    if value.lower().startswith("r/"):
        value = value[2:]

    return value.strip().replace(" ", "")


def get_groq_key():
    """Read the Groq API key from Streamlit secrets."""

    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""


@contextmanager
def card():
    """Render a dashboard card."""

    st.html('<div class="reddit-card">')

    yield

    st.html("</div>")


def style_fig(fig, **overrides):
    """Apply consistent Plotly styling."""

    layout = dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color="white"),
        margin=dict(
            t=40,
            b=40,
            l=40,
            r=20,
        ),
        xaxis=dict(
            gridcolor=BORDER,
        ),
        yaxis=dict(
            gridcolor=BORDER,
        ),
    )

    layout.update(overrides)

    fig.update_layout(**layout)

    return fig


def topic_color_map(names):
    """Assign stable colors to topic names."""

    return {
        name: TOPIC_PALETTE[index % len(TOPIC_PALETTE)]
        for index, name in enumerate(names)
    }


# =============================================================================
# 7. HTTP CLIENT
# =============================================================================

@st.cache_resource
def get_http_session():
    """Create a reusable HTTP session with retry support."""

    retry = Retry(
        total=MAX_FETCH_RETRIES,
        backoff_factor=0.6,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )

    session = requests.Session()

    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=retry,
            pool_connections=4,
            pool_maxsize=4,
        ),
    )

    session.headers.update(
        {
            "User-Agent": "reddit-recon-ui/4.0"
        }
    )

    return session


# =============================================================================
# 8. MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    """Load sentiment classifier."""

    return pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        truncation=True,
        max_length=256,
        device=-1,
    )


@st.cache_resource(show_spinner="Loading emotion model...")
def load_emotion_model():
    """Load emotion classifier."""

    return pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        tokenizer=EMOTION_MODEL,
        truncation=True,
        max_length=256,
        top_k=None,
        device=-1,
    )


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Load MiniLM embedding model."""

    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL
    )

    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL
    )

    model.eval()

    return tokenizer, model


# =============================================================================
# 9. REDDIT DATA FETCHING
# =============================================================================

@st.cache_data(
    ttl=900,
    show_spinner="Fetching Reddit Data",
)
def fetch_reddit_posts(
    subreddit,
    posts_to_fetch,
    days_back,
):
    """Fetch Reddit posts using Arctic Shift."""

    now = int(time.time())

    cutoff = (
        now
        - days_back * 24 * 60 * 60
    )

    before = now

    page_size = 100

    max_pages = int(
        np.ceil(
            posts_to_fetch / page_size
        )
    )

    session = get_http_session()

    fields = (
        "id,"
        "created_utc,"
        "score,"
        "num_comments,"
        "subreddit,"
        "title,"
        "selftext,"
        "url"
    )

    posts = []
    seen_ids = set()

    for _ in range(max_pages):

        params = {
            "subreddit": subreddit,
            "after": cutoff,
            "before": before,
            "limit": page_size,
            "sort": "desc",
            "over_18": "false",
            "fields": fields,
        }

        try:
            response = session.get(
                ARCTIC_URL,
                params=params,
                timeout=45,
            )

            response.raise_for_status()

            payload = response.json()

        except requests.RequestException as exc:
            st.warning(
                f"Reddit data request failed: {exc}"
            )
            break

        except ValueError:
            st.warning(
                "Reddit API returned invalid JSON."
            )
            break

        batch = payload.get("data", [])

        if not batch:
            break

        oldest_timestamp = None

        for item in batch:

            post_id = item.get("id")

            if (
                not post_id
                or post_id in seen_ids
            ):
                continue

            created = safe_int(
                item.get("created_utc", 0)
            )

            if created < cutoff:
                continue

            seen_ids.add(post_id)

            posts.append(
                {
                    "id": post_id,
                    "title": str(
                        item.get("title") or ""
                    ),
                    "selftext": str(
                        item.get("selftext") or ""
                    ),
                    "score": safe_int(
                        item.get("score", 0)
                    ),
                    "num_comments": safe_int(
                        item.get(
                            "num_comments",
                            0,
                        )
                    ),
                    "created_utc": created,
                    "url": str(
                        item.get("url") or ""
                    ),
                    "subreddit": str(
                        item.get(
                            "subreddit"
                        )
                        or subreddit
                    ),
                }
            )

            oldest_timestamp = (
                created
                if oldest_timestamp is None
                else min(
                    oldest_timestamp,
                    created,
                )
            )

            if len(posts) >= posts_to_fetch:
                break

        if (
            len(posts) >= posts_to_fetch
            or oldest_timestamp is None
            or oldest_timestamp <= cutoff
        ):
            break

        before = oldest_timestamp - 1

        time.sleep(0.15)

    if not posts:
        return pd.DataFrame()

    return (
        pd.DataFrame(posts)
        .drop_duplicates("id")
        .reset_index(drop=True)
    )


# =============================================================================
# 10. TEXT CLEANING
# =============================================================================

def clean_posts(
    df,
    min_text_length=20,
):
    """Prepare post text for NLP."""

    if df.empty:
        return df.copy()

    result = df.copy()

    result["title"] = (
        result["title"]
        .fillna("")
        .astype(str)
    )

    result["selftext"] = (
        result["selftext"]
        .fillna("")
        .astype(str)
    )

    result["text"] = (
        result["title"]
        + " "
        + result["selftext"]
    )

    result["text"] = (
        result["text"]
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    result = result[
        result["text"].str.len()
        >= min_text_length
    ].copy()

    result["model_text"] = (
        result["text"]
        .str.slice(
            0,
            MAX_TEXT_CHARS,
        )
    )

    return (
        result
        .drop_duplicates("id")
        .reset_index(drop=True)
    )


def select_posts_for_analysis(
    df,
    top_posts,
):
    """Select the posts used by the NLP pipeline."""

    if df.empty:
        return df.copy()

    return (
        df.sort_values(
            ["score", "num_comments"],
            ascending=False,
            kind="stable",
        )
        .head(
            min(
                top_posts,
                len(df),
            )
        )
        .reset_index(drop=True)
    )


# =============================================================================
# 11. SENTIMENT ANALYSIS
# =============================================================================

def analyze_sentiment(
    df,
    batch_size,
):
    """Run sentiment classification."""

    if df.empty:
        return df.copy()

    model = load_sentiment_model()

    predictions = model(
        df["model_text"].tolist(),
        batch_size=batch_size,
    )

    result = df.copy()

    result["sentiment"] = [
        str(
            prediction["label"]
        )
        .lower()
        .strip()
        for prediction in predictions
    ]

    result["sentiment_confidence"] = [
        float(
            prediction["score"]
        )
        for prediction in predictions
    ]

    return result


# =============================================================================
# 12. EMOTION ANALYSIS
# =============================================================================

def analyze_emotions(
    df,
    batch_size,
):
    """Run emotion classification and retain all class probabilities."""

    if df.empty:
        return df.copy()

    model = load_emotion_model()

    texts = (
        df["model_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    predictions = model(
        texts,
        batch_size=batch_size,
    )

    result = df.copy()

    emotion_scores = []
    emotion_labels = []
    emotion_confidences = []

    for prediction in predictions:

        if isinstance(
            prediction,
            dict,
        ):
            prediction = [prediction]

        scores = {
            str(item["label"]).lower():
            float(item["score"])
            for item in prediction
        }

        for label in EMOTION_LABELS:
            scores.setdefault(
                label,
                0.0,
            )

        dominant = max(
            scores,
            key=scores.get,
        )

        emotion_labels.append(
            dominant
        )

        emotion_confidences.append(
            scores[dominant]
        )

        emotion_scores.append(
            scores
        )

    result["emotion"] = emotion_labels

    result["emotion_confidence"] = (
        emotion_confidences
    )

    for label in EMOTION_LABELS:

        result[
            f"emotion_{label}"
        ] = [
            scores[label]
            for scores
            in emotion_scores
        ]

    return result


# =============================================================================
# 13. EMBEDDINGS
# =============================================================================

def mean_pool(
    last_hidden_state,
    attention_mask,
):
    """Mean-pool transformer embeddings."""

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(
            last_hidden_state.size()
        )
        .float()
    )

    summed = torch.sum(
        last_hidden_state * mask,
        dim=1,
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9,
    )

    return summed / counts


def encode_minilm(texts):
    """Encode text using MiniLM."""

    tokenizer, model = (
        load_embedding_model()
    )

    vectors = []

    for start in range(
        0,
        len(texts),
        EMBEDDING_BATCH_SIZE,
    ):

        batch = texts[
            start:
            start + EMBEDDING_BATCH_SIZE
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.inference_mode():

            output = model(
                **encoded
            )

            pooled = mean_pool(
                output.last_hidden_state,
                encoded["attention_mask"],
            )

            pooled = (
                torch.nn.functional.normalize(
                    pooled,
                    p=2,
                    dim=1,
                )
            )

        vectors.append(
            pooled
            .cpu()
            .numpy()
        )

    return np.vstack(vectors)


# =============================================================================
# 14. TOPIC DISCOVERY
# =============================================================================

def discover_topics(df):
    """Discover semantic topics using KMeans and silhouette selection."""

    if len(df) < 3:

        result = df.copy()

        result["topic_id"] = 0

        return (
            result,
            pd.DataFrame(),
            1,
        )

    embeddings = encode_minilm(
        df["model_text"].tolist()
    )

    max_k = min(
        8,
        len(df) - 1,
    )

    if max_k < 3:

        result = df.copy()

        result["topic_id"] = 0

        return (
            result,
            pd.DataFrame(),
            1,
        )

    scores = []

    for k in range(
        3,
        max_k + 1,
    ):

        try:

            labels = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=5,
            ).fit_predict(
                embeddings
            )

            if len(
                np.unique(labels)
            ) < 2:
                continue

            scores.append(
                {
                    "k": k,
                    "silhouette_score": float(
                        silhouette_score(
                            embeddings,
                            labels,
                            metric="cosine",
                        )
                    ),
                }
            )

        except Exception:
            continue

    if scores:

        silhouette_df = (
            pd.DataFrame(scores)
        )

        best_k = int(
            silhouette_df.loc[
                silhouette_df[
                    "silhouette_score"
                ].idxmax(),
                "k",
            ]
        )

    else:

        silhouette_df = (
            pd.DataFrame()
        )

        best_k = min(
            5,
            len(df) - 1,
        )

    result = df.copy()

    result["topic_id"] = (
        KMeans(
            n_clusters=best_k,
            random_state=RANDOM_STATE,
            n_init=5,
        )
        .fit_predict(embeddings)
    )

    return (
        result,
        silhouette_df,
        best_k,
    )


def extract_topic_keywords(
    df,
    top_n=10,
):
    """Extract TF-IDF keywords for each topic."""

    if (
        df.empty
        or "topic_id" not in df.columns
    ):
        return {}

    stopwords = (
        set(ENGLISH_STOP_WORDS)
        | EXTRA_STOPWORDS
    )

    try:

        vectorizer = TfidfVectorizer(
            stop_words=list(stopwords),
            max_features=4000,
            ngram_range=(1, 2),
            min_df=2,
        )

        matrix = vectorizer.fit_transform(
            df["model_text"]
        )

    except ValueError:
        return {}

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    keywords = {}

    for topic_id in sorted(
        df["topic_id"].unique()
    ):

        indexes = np.where(
            df["topic_id"].values
            == topic_id
        )[0]

        topic_scores = (
            matrix[indexes]
            .mean(axis=0)
            .A1
        )

        top_indexes = (
            topic_scores
            .argsort()[::-1][:top_n]
        )

        keywords[int(topic_id)] = (
            feature_names[
                top_indexes
            ].tolist()
        )

    return keywords


# =============================================================================
# 15. ENGAGEMENT
# =============================================================================

def calculate_engagement(df):
    """
    Calculate the dashboard engagement index.

    Engagement = Score + 2 × Comments
    """

    result = df.copy()

    result["score"] = pd.to_numeric(
        result["score"],
        errors="coerce",
    ).fillna(0)

    result["num_comments"] = pd.to_numeric(
        result["num_comments"],
        errors="coerce",
    ).fillna(0)

    result["engagement"] = (
        result["score"]
        + (
            2
            * result["num_comments"]
        )
    )

    return result


def get_top_engaged_posts(
    df,
    n=10,
):
    """Return top posts by engagement."""

    columns = [
        "title",
        "score",
        "num_comments",
        "sentiment",
        "emotion",
        "topic_id",
        "engagement",
        "url",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    return (
        df.sort_values(
            "engagement",
            ascending=False,
        )[available]
        .head(n)
        .reset_index(drop=True)
    )


# =============================================================================
# 16. SENTIMENT SUMMARIES
# =============================================================================

def sentiment_summary(df):
    """Create sentiment distribution summary."""

    if (
        df.empty
        or "sentiment" not in df.columns
    ):
        return pd.DataFrame(
            columns=[
                "sentiment",
                "count",
                "percentage",
            ]
        )

    summary = (
        df["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(
            name="count"
        )
    )

    summary["percentage"] = (
        summary["count"]
        / summary["count"].sum()
        * 100
    )

    return summary


# =============================================================================
# 17. TOPIC STATISTICS
# =============================================================================

def build_topic_stats(
    analysis_df,
    name_map,
):
    """Build aggregate statistics for every topic."""

    stats = (
        analysis_df
        .groupby("topic_id")
        .agg(
            posts=("id", "count"),
            avg_score=("score", "mean"),
            avg_comments=(
                "num_comments",
                "mean",
            ),
            total_engagement=(
                "engagement",
                "sum",
            ),
            avg_engagement=(
                "engagement",
                "mean",
            ),
        )
        .reset_index()
    )

    stats["Topic Name"] = (
        stats["topic_id"]
        .map(name_map)
        .fillna(
            stats["topic_id"].apply(
                lambda x:
                f"Topic {x}"
            )
        )
    )

    return stats


# =============================================================================
# 18. TOPIC EVIDENCE
# =============================================================================

def create_topic_evidence(
    df,
    keywords,
    top_n_posts=5,
):
    """Prepare evidence for LLM topic analysis."""

    evidence = []

    for topic_id in sorted(
        df["topic_id"].unique()
    ):

        topic_df = df[
            df["topic_id"]
            == topic_id
        ]

        representative = (
            topic_df
            .sort_values(
                [
                    "score",
                    "num_comments",
                ],
                ascending=False,
            )
            .head(top_n_posts)
        )

        posts = [
            {
                "title": str(
                    row.title
                ),
                "text": str(
                    row.selftext
                )[:1000],
                "score": safe_int(
                    row.score
                ),
                "comments": safe_int(
                    row.num_comments
                ),
            }
            for row in representative.itertuples()
        ]

        evidence.append(
            {
                "topic_id": int(
                    topic_id
                ),
                "post_count": int(
                    len(topic_df)
                ),
                "avg_score": round(
                    float(
                        topic_df[
                            "score"
                        ].mean()
                    ),
                    2,
                ),
                "avg_comments": round(
                    float(
                        topic_df[
                            "num_comments"
                        ].mean()
                    ),
                    2,
                ),
                "keywords": keywords.get(
                    int(topic_id),
                    [],
                ),
                "representative_posts": posts,
            }
        )

    return evidence


# =============================================================================
# 19. GROQ / AI ANALYSIS
# =============================================================================

def extract_json_object(content):
    """Parse JSON directly or extract the first JSON object."""

    if not content:
        raise ValueError(
            "The AI returned an empty response."
        )

    content = content.strip()

    try:
        return json.loads(content)

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        content,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(
            "The AI response did not contain a JSON object."
        )

    return json.loads(
        match.group(0)
    )


def analyze_topic_with_groq(
    topic_evidence,
    api_key,
):
    """Generate an AI description for one topic."""

    topic_id = topic_evidence[
        "topic_id"
    ]

    fallback = {
        "topic_id": topic_id,
        "name": (
            f"Topic {topic_id}"
        ),
        "description": "",
        "main_reaction": "",
    }

    if not api_key:

        return {
            **fallback,
            "error": (
                "GROQ_API_KEY is not configured."
            ),
        }

    try:

        from groq import Groq

        client = Groq(
            api_key=api_key
        )

        prompt = f"""
You are an expert Reddit community analyst.

Analyze Topic {topic_id} using the evidence below.

Return exactly one valid JSON object with these four keys:

- topic_id: integer
- name: concise meaningful topic name
- description: concise explanation of the discussion
- main_reaction: dominant reaction or attitude

Use double quotes for JSON keys and string values.

Do not include Markdown fences, comments, or text outside the JSON.

Do not invent facts not supported by the evidence.

Do not call the topic "Topic {topic_id}" unless no meaningful
information is available.

Evidence:
{json.dumps(topic_evidence, ensure_ascii=False)}
"""

        response = (
            client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return only a valid JSON object. "
                            "Do not include Markdown."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
                max_completion_tokens=500,
                response_format={
                    "type": "json_object"
                },
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        parsed = extract_json_object(
            content
        )

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                "The AI response was not a JSON object."
            )

        name = parsed.get(
            "name"
        )

        description = parsed.get(
            "description"
        )

        main_reaction = parsed.get(
            "main_reaction"
        )

        if not all(
            isinstance(
                value,
                str,
            )
            for value in (
                name,
                description,
                main_reaction,
            )
        ):
            raise ValueError(
                "The AI response has invalid field types."
            )

        return {
            "topic_id": topic_id,
            "name": (
                name.strip()
                or fallback["name"]
            ),
            "description": (
                description.strip()
            ),
            "main_reaction": (
                main_reaction.strip()
            ),
        }

    except Exception as exc:

        return {
            **fallback,
            "error": str(exc),
        }


def generate_ai_topic_insights(
    evidence,
    workers=3,
):
    """Generate AI analysis for all topics."""

    if not evidence:
        return []

    api_key = get_groq_key()

    if not api_key:

        return [
            {
                "topic_id": item[
                    "topic_id"
                ],
                "name": (
                    f"Topic {item['topic_id']}"
                ),
                "description": "",
                "main_reaction": "",
                "error": (
                    "GROQ_API_KEY is missing."
                ),
            }
            for item in evidence
        ]

    results = []

    with ThreadPoolExecutor(
        max_workers=min(
            workers,
            len(evidence),
        )
    ) as executor:

        futures = {
            executor.submit(
                analyze_topic_with_groq,
                item,
                api_key,
            ): item
            for item in evidence
        }

        for future in as_completed(
            futures
        ):

            item = futures[future]

            try:
                results.append(
                    future.result()
                )

            except Exception as exc:

                results.append(
                    {
                        "topic_id": item[
                            "topic_id"
                        ],
                        "name": (
                            f"Topic "
                            f"{item['topic_id']}"
                        ),
                        "description": "",
                        "main_reaction": "",
                        "error": str(exc),
                    }
                )

    return sorted(
        results,
        key=lambda x:
        x.get(
            "topic_id",
            999,
        ),
    )


def generate_overall_review(
    evidence,
):
    """Generate an overall subreddit intelligence summary."""

    api_key = get_groq_key()

    if not api_key:
        return (
            "GROQ_API_KEY is not configured."
        )

    try:

        from groq import Groq

        client = Groq(
            api_key=api_key
        )

        prompt = f"""
Analyze this Reddit subreddit sample.

Provide a concise 2-3 paragraph intelligence summary covering:

- Overall community consensus
- Major sentiment drivers
- Important recurring discussions
- Unusual or niche observations

Use only the supplied evidence.

Evidence:
{json.dumps(
    evidence,
    ensure_ascii=False
)}
"""

        response = (
            client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.3,
                max_completion_tokens=1000,
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as exc:

        return (
            f"Error generating review: {exc}"
        )


# =============================================================================
# 20. WORD CLOUD
# =============================================================================

def generate_word_cloud(df):
    """Generate the Review tab word cloud."""

    if df.empty:
        return None

    text = " ".join(
        (
            df["title"].fillna("")
            + " "
            + df["selftext"].fillna("")
        )
    )

    stopwords = (
        set(ENGLISH_STOP_WORDS)
        | EXTRA_STOPWORDS
    )

    wordcloud = WordCloud(
        width=1200,
        height=550,
        background_color=BG,
        stopwords=stopwords,
        colormap="turbo",
        max_words=150,
        min_font_size=10,
        max_font_size=100,
        prefer_horizontal=0.9,
    ).generate(text)

    fig, ax = plt.subplots(
        figsize=(12, 5.5)
    )

    fig.patch.set_facecolor(
        BG
    )

    ax.set_facecolor(BG)

    ax.imshow(
        wordcloud,
        interpolation="bilinear",
    )

    ax.axis("off")

    plt.tight_layout(
        pad=0
    )

    return fig


# =============================================================================
# 21. UI HELPERS
# =============================================================================

def page_header(
    title,
    subtitle,
):
    """Render application header."""

    st.html(
        f"""
        <div class="reddit-header">

            <div class="reddit-icon">
                <i
                    class="fa-brands fa-reddit"
                    aria-hidden="true">
                </i>
            </div>

            <div>

                <h1>
                    {html.escape(title)}
                </h1>

                <p>
                    {html.escape(subtitle)}
                </p>

            </div>

        </div>
        """
    )


def render_color_key():
    """Render the dashboard semantic color legend."""

    with card():

        st.markdown(
            '<div class="section-label">'
            'Color Key'
            '</div>',
            unsafe_allow_html=True,
        )

        sentiment_items = []

        for label, color in (
            SENTIMENT_COLORS.items()
        ):

            sentiment_items.append(
                f"""
                <span class="legend-item">
                    <span
                        class="legend-dot"
                        style="color:{color};">
                        ●
                    </span>
                    {html.escape(label)}
                </span>
                """
            )

        emotion_items = []

        for label, color in (
            EMOTION_COLORS.items()
        ):

            emotion_items.append(
                f"""
                <span class="legend-item">
                    <span
                        class="legend-dot"
                        style="color:{color};">
                        ●
                    </span>
                    {html.escape(
                        label.capitalize()
                    )}
                </span>
                """
            )

        st.html(
            f"""
            <div class="legend-row">
                <strong>Sentiment</strong>
                {''.join(sentiment_items)}
            </div>

            <div class="legend-row">
                <strong>Emotion</strong>
                {''.join(emotion_items)}
            </div>
            """
        )


def render_confidence_strip(
    values,
    colors,
    labels,
    title,
):
    """
    Render one confidence bar.

    Each class gets one equal-width section.
    Within each section, the colored fill represents its
    prediction confidence from 0-100%.
    """

    if not values:
        return

    segments = []

    labels_html = []

    for label, value in zip(
        labels,
        values,
    ):

        value = max(
            0,
            min(
                100,
                float(value),
            ),
        )

        color = colors.get(
            label,
            "#64748B",
        )

        segments.append(
            f"""
            <div
                class="confidence-segment"
                title="{html.escape(label)}: "
                      f"{value:.1f}%">

                <div
                    class="confidence-fill"
                    style="
                        width:{value:.1f}%;
                        background:{color};
                    ">
                </div>

            </div>
            """
        )

        labels_html.append(
            f"""
            <div class="confidence-label">
                <span style="color:{color};">
                    ●
                </span>
                {html.escape(label)}
                <b>{value:.0f}%</b>
            </div>
            """
        )

    st.html(
        f"""
        <div class="confidence-wrapper">

            <div class="confidence-title">
                {html.escape(title)}
            </div>

            <div class="confidence-track">
                {''.join(segments)}
            </div>

            <div class="confidence-labels">
                {''.join(labels_html)}
            </div>

        </div>
        """
    )


# =============================================================================
# 22. FILTERING
# =============================================================================

def apply_dashboard_filters(
    analysis_df,
    name_map,
):
    """ Apply global dashboard filters."""

    filter1, filter2, filter3 = (
        st.columns(3)
    )

    with filter1:

        topic_options = [
            "All Topics"
        ] + sorted(
            name_map.values()
        )

        selected_topic = st.selectbox(
            "Topic",
            topic_options,
            key="overview_topic_filter",
        )

    with filter2:

        selected_sentiment = (
            st.selectbox(
                "Sentiment",
                [
                    "All Sentiments",
                    "Positive",
                    "Neutral",
                    "Negative",
                ],
                key="overview_sentiment_filter",
            )
        )

    with filter3:

        selected_emotion = (
            st.selectbox(
                "Emotion",
                [
                    "All Emotions"
                ]
                + [
                    label.capitalize()
                    for label
                    in EMOTION_LABELS
                ],
                key="overview_emotion_filter",
            )
        )

    filtered_df = (
        analysis_df.copy()
    )

    if selected_topic != "All Topics":

        topic_id_lookup = {
            name: topic_id
            for topic_id, name
            in name_map.items()
        }

        selected_topic_id = (
            topic_id_lookup.get(
                selected_topic
            )
        )

        if selected_topic_id is not None:

            filtered_df = (
                filtered_df[
                    filtered_df[
                        "topic_id"
                    ]
                    == selected_topic_id
                ]
            )

    if (
        selected_sentiment
        != "All Sentiments"
    ):

        filtered_df = (
            filtered_df[
                filtered_df[
                    "sentiment"
                ]
                == selected_sentiment.lower()
            ]
        )

    if (
        selected_emotion
        != "All Emotions"
    ):

        filtered_df = (
            filtered_df[
                filtered_df[
                    "emotion"
                ]
                == selected_emotion.lower()
            ]
        )

    return filtered_df


# =============================================================================
# 23. OVERVIEW CHARTS
# =============================================================================

def render_sentiment_distribution(
    df,
):
    """Render sentiment donut and confidence strip."""

    st.subheader(
        "Sentiment Distribution"
    )

    if df.empty:

        st.info(
            "No posts match the selected filters."
        )

        return

    counts = (
        df["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(
            name="count"
        )
    )

    counts["label"] = (
        counts["sentiment"]
        .str.capitalize()
    )

    fig = px.pie(
        counts,
        values="count",
        names="label",
        hole=0.55,
        color="label",
        color_discrete_map=(
            SENTIMENT_COLORS
        ),
    )

    fig.update_traces(
        textinfo="percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Posts: %{value:,}<br>"
            "Share: %{percent}"
            "<extra></extra>"
        ),
    )

    st.plotly_chart(
        style_fig(
            fig,
            margin=dict(
                t=10,
                b=10,
                l=10,
                r=10,
            ),
            legend=dict(
                font=dict(
                    color="white"
                )
            ),
        ),
        use_container_width=True,
    )

    confidence_values = []

    confidence_labels = []

    for sentiment in [
        "positive",
        "neutral",
        "negative",
    ]:

        subset = df[
            df["sentiment"]
            == sentiment
        ]

        confidence_values.append(
            (
                subset[
                    "sentiment_confidence"
                ].mean()
                * 100
                if not subset.empty
                else 0
            )
        )

        confidence_labels.append(
            sentiment.capitalize()
        )

    render_confidence_strip(
        confidence_values,
        SENTIMENT_COLORS,
        confidence_labels,
        "Average Prediction Confidence",
    )


def render_emotion_distribution(
    df,
):
    """Render emotion distribution and confidence strip."""

    st.subheader(
        "Emotion Distribution"
    )

    if df.empty:

        st.info(
            "No posts match the selected filters."
        )

        return

    counts = (
        df["emotion"]
        .value_counts()
        .rename_axis("emotion")
        .reset_index(
            name="count"
        )
    )

    counts["label"] = (
        counts["emotion"]
        .str.capitalize()
    )

    fig = px.bar(
        counts.sort_values(
            "count",
            ascending=True,
        ),
        x="count",
        y="label",
        orientation="h",
        color="emotion",
        color_discrete_map=(
            EMOTION_COLORS
        ),
        labels={
            "count": "Posts",
            "label": "",
        },
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Posts: %{x:,}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(
        style_fig(
            fig,
            margin=dict(
                t=10,
                b=10,
                l=30,
                r=20,
            ),
            showlegend=False,
        ),
        use_container_width=True,
    )

    confidence_values = []

    confidence_labels = []

    confidence_colors = {}

    for emotion in EMOTION_LABELS:

        label = emotion.capitalize()

        subset = df[
            df["emotion"]
            == emotion
        ]

        confidence_values.append(
            (
                subset[
                    "emotion_confidence"
                ].mean()
                * 100
                if not subset.empty
                else 0
            )
        )

        confidence_labels.append(
            label
        )

        confidence_colors[
            label
        ] = EMOTION_COLORS[
            emotion
        ]

    render_confidence_strip(
        confidence_values,
        confidence_colors,
        confidence_labels,
        "Average Prediction Confidence",
    )


def render_engagement_chart(
    df,
):
    """Render engagement by sentiment or emotion."""

    st.subheader(
        "Engagement Performance"
    )

    mode = st.radio(
        "View",
        [
            "Sentiment",
            "Emotion",
        ],
        horizontal=True,
        key="engagement_mode",
    )

    if df.empty:

        st.info(
            "No posts match the selected filters."
        )

        return

    if mode == "Sentiment":

        grouped = (
            df.groupby(
                "sentiment"
            )
            .agg(
                avg_engagement=(
                    "engagement",
                    "mean",
                ),
                median_engagement=(
                    "engagement",
                    "median",
                ),
                posts=(
                    "id",
                    "count",
                ),
                avg_comments=(
                    "num_comments",
                    "mean",
                ),
            )
            .reset_index()
        )

        grouped["label"] = (
            grouped["sentiment"]
            .str.capitalize()
        )

        fig = px.bar(
            grouped.sort_values(
                "avg_engagement",
                ascending=False,
            ),
            x="label",
            y="avg_engagement",
            text="avg_engagement",
            color="label",
            color_discrete_map=(
                SENTIMENT_COLORS
            ),
            labels={
                "label": "",
                "avg_engagement":
                    "Average Engagement",
            },
        )

    else:

        grouped = (
            df.groupby(
                "emotion"
            )
            .agg(
                avg_engagement=(
                    "engagement",
                    "mean",
                ),
                median_engagement=(
                    "engagement",
                    "median",
                ),
                posts=(
                    "id",
                    "count",
                ),
                avg_comments=(
                    "num_comments",
                    "mean",
                ),
            )
            .reset_index()
        )

        grouped["label"] = (
            grouped["emotion"]
            .str.capitalize()
        )

        fig = px.bar(
            grouped.sort_values(
                "avg_engagement",
                ascending=False,
            ),
            x="label",
            y="avg_engagement",
            text="avg_engagement",
            color="emotion",
            color_discrete_map=(
                EMOTION_COLORS
            ),
            labels={
                "label": "",
                "avg_engagement":
                    "Average Engagement",
            },
        )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    st.plotly_chart(
        style_fig(
            fig,
            margin=dict(
                t=30,
                b=50,
                l=50,
                r=20,
            ),
            showlegend=False,
            xaxis=dict(
                gridcolor=BORDER,
                tickangle=-20,
            ),
        ),
        use_container_width=True,
    )


def render_topic_performance(
    df,
    name_map,
):
    """Render topic volume vs engagement scatter."""

    st.subheader(
        "Topic Performance"
    )

    if df.empty:

        st.info(
            "No topic data matches the selected filters."
        )

        return

    topic_stats = (
        df.groupby("topic_id")
        .agg(
            posts=(
                "id",
                "count",
            ),
            avg_engagement=(
                "engagement",
                "mean",
            ),
            total_comments=(
                "num_comments",
                "sum",
            ),
        )
        .reset_index()
    )

    topic_stats["Topic Name"] = (
        topic_stats["topic_id"]
        .map(name_map)
        .fillna(
            topic_stats[
                "topic_id"
            ].apply(
                lambda x:
                f"Topic {x}"
            )
        )
    )

    colors = topic_color_map(
        topic_stats[
            "Topic Name"
        ]
    )

    fig = px.scatter(
        topic_stats,
        x="posts",
        y="avg_engagement",
        size="total_comments",
        hover_name="Topic Name",
        text="Topic Name",
        color="Topic Name",
        color_discrete_map=colors,
        labels={
            "posts": "Post Volume",
            "avg_engagement":
                "Average Engagement",
            "total_comments":
                "Total Comments",
        },
    )

    fig.update_traces(
        textposition="top center"
    )

    st.plotly_chart(
        style_fig(
            fig,
            margin=dict(
                t=30,
                b=50,
                l=50,
                r=20,
            ),
            showlegend=False,
        ),
        use_container_width=True,
    )


def render_topic_distribution(
    df,
    name_map,
):
    """Render topic volume distribution."""

    st.subheader(
        "Topic Distribution"
    )

    if df.empty:

        st.info(
            "No posts match the selected filters."
        )

        return

    distribution = (
        df.groupby(
            "topic_id"
        )
        .size()
        .reset_index(
            name="posts"
        )
    )

    distribution["Topic Name"] = (
        distribution["topic_id"]
        .map(name_map)
        .fillna(
            distribution[
                "topic_id"
            ].apply(
                lambda x:
                f"Topic {x}"
            )
        )
    )

    distribution = (
        distribution
        .sort_values(
            "posts",
            ascending=False,
        )
    )

    colors = topic_color_map(
        distribution[
            "Topic Name"
        ]
    )

    fig = px.bar(
        distribution,
        x="Topic Name",
        y="posts",
        text="posts",
        color="Topic Name",
        color_discrete_map=colors,
        labels={
            "Topic Name": "",
            "posts": "Number of Posts",
        },
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        style_fig(
            fig,
            margin=dict(
                t=30,
                b=80,
                l=40,
                r=20,
            ),
            showlegend=False,
            xaxis=dict(
                tickangle=-30,
                gridcolor=BORDER,
            ),
        ),
        use_container_width=True,
    )


def render_trends(df):
    """Render sentiment or emotion distribution over time."""

    st.subheader(
        "Community Sentiment & Emotion Over Time"
    )

    mode = st.radio(
        "Trend View",
        [
            "Sentiment",
            "Emotion",
        ],
        horizontal=True,
        key="trend_mode",
    )

    if df.empty:

        st.info(
            "No posts match the selected filters."
        )

        return

    trend_df = df.copy()

    trend_df["date"] = (
        pd.to_datetime(
            trend_df[
                "created_utc"
            ],
            unit="s",
            utc=True,
        )
        .dt.floor(
            ROLLING_WINDOW
        )
    )

    if mode == "Sentiment":

        counts = (
            trend_df
            .groupby(
                [
                    "date",
                    "sentiment",
                ]
            )
            .size()
            .reset_index(
                name="posts"
            )
        )

        totals = (
            trend_df
            .groupby("date")
            .size()
            .reset_index(
                name="total"
            )
        )

        counts = counts.merge(
            totals,
            on="date",
            how="left",
        )

        counts["percentage"] = (
            counts["posts"]
            / counts["total"]
            * 100
        )

        counts["label"] = (
            counts["sentiment"]
            .str.capitalize()
        )

        fig = px.line(
            counts,
            x="date",
            y="percentage",
            color="label",
            color_discrete_map=(
                SENTIMENT_COLORS
            ),
            markers=True,
            labels={
                "date": "Date",
                "percentage":
                    "Share of Posts (%)",
                "label": "Sentiment",
            },
        )

    else:

        counts = (
            trend_df
            .groupby(
                [
                    "date",
                    "emotion",
                ]
            )
            .size()
            .reset_index(
                name="posts"
            )
        )

        totals = (
            trend_df
            .groupby("date")
            .size()
            .reset_index(
                name="total"
            )
        )

        counts = counts.merge(
            totals,
            on="date",
            how="left",
        )

        counts["percentage"] = (
            counts["posts"]
            / counts["total"]
            * 100
        )

        counts["label"] = (
            counts["emotion"]
            .str.capitalize()
        )

        emotion_colors = {
            label.capitalize():
                color
            for label, color
            in EMOTION_COLORS.items()
        }

        fig = px.line(
            counts,
            x="date",
            y="percentage",
            color="label",
            color_discrete_map=(
                emotion_colors
            ),
            markers=True,
            labels={
                "date": "Date",
                "percentage":
                    "Share of Posts (%)",
                "label": "Emotion",
            },
        )

    st.plotly_chart(
        style_fig(
            fig,
            margin=dict(
                t=30,
                b=50,
                l=50,
                r=20,
            ),
            yaxis=dict(
                title="Share of Posts (%)",
                range=[0, 100],
                gridcolor=BORDER,
            ),
        ),
        use_container_width=True,
    )


def render_top_posts(
    df,
):
    """Render the final Top 5 Posts table."""

    st.subheader(
        "Top 5 Posts by Engagement"
    )

    if df.empty:

        st.info(
            "No posts match the selected filters."
        )

        return

    top5 = (
        df.sort_values(
            "engagement",
            ascending=False,
        )
        .head(5)
        .copy()
    )

    rows = []

    for rank, row in enumerate(
        top5.itertuples(),
        start=1,
    ):

        rows.append(
            {
                "Rank": rank,
                "Post": str(
                    row.title
                ),
                "Sentiment": (
                    str(
                        row.sentiment
                    ).capitalize()
                ),
                "Emotion": (
                    str(
                        row.emotion
                    ).capitalize()
                ),
                "Score": safe_int(
                    row.score
                ),
                "Comments": safe_int(
                    row.num_comments
                ),
                "Engagement": float(
                    row.engagement
                ),
            }
        )

    display_df = pd.DataFrame(
        rows
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(
                "Rank",
                width="small",
            ),
            "Post": st.column_config.TextColumn(
                "Post",
                width="large",
            ),
            "Engagement": (
                st.column_config.NumberColumn(
                    "Engagement",
                    format="%.2f",
                )
            ),
        },
    )


# =============================================================================
# 24. TOPIC TAB HELPERS
# =============================================================================

def get_prominent_non_neutral_sentiment(
    topic_df,
):
    """Return the dominant non-neutral sentiment."""

    if (
        topic_df.empty
        or "sentiment"
        not in topic_df.columns
    ):
        return None, 0.0

    counts = (
        topic_df[
            "sentiment"
        ]
        .value_counts(
            normalize=True
        )
        * 100
    )

    non_neutral = counts[
        counts.index.isin(
            [
                "positive",
                "negative",
            ]
        )
    ]

    if non_neutral.empty:
        return None, 0.0

    label = (
        non_neutral
        .idxmax()
    )

    return (
        label,
        float(
            non_neutral[label]
        ),
    )


def get_prominent_non_neutral_emotion(
    topic_df,
):
    """Return the dominant non-neutral emotion."""

    if (
        topic_df.empty
        or "emotion"
        not in topic_df.columns
    ):
        return None, 0.0

    counts = (
        topic_df[
            "emotion"
        ]
        .value_counts(
            normalize=True
        )
        * 100
    )

    non_neutral = counts[
        counts.index != "neutral"
    ]

    if non_neutral.empty:
        return None, 0.0

    label = (
        non_neutral
        .idxmax()
    )

    return (
        label,
        float(
            non_neutral[label]
        ),
    )


def render_topic_reaction_badges(
    topic_df,
):
    """Render prominent non-neutral sentiment and emotion."""

    sentiment_label, sentiment_pct = (
        get_prominent_non_neutral_sentiment(
            topic_df
        )
    )

    emotion_label, emotion_pct = (
        get_prominent_non_neutral_emotion(
            topic_df
        )
    )

    badges = []

    if sentiment_label:

        color = SENTIMENT_COLORS.get(
            sentiment_label.capitalize(),
            "#64748B",
        )

        badges.append(
            f"""
            <span
                class="badge"
                style="
                    border-color:{color};
                    color:{color};
                "
            >
                Sentiment:
                {html.escape(
                    sentiment_label.capitalize()
                )}
                {sentiment_pct:.0f}%
            </span>
            """
        )

    if emotion_label:

        color = EMOTION_COLORS.get(
            emotion_label,
            "#64748B",
        )

        badges.append(
            f"""
            <span
                class="badge"
                style="
                    border-color:{color};
                    color:{color};
                "
            >
                Emotion:
                {html.escape(
                    emotion_label.capitalize()
                )}
                {emotion_pct:.0f}%
            </span>
            """
        )

    st.html(
        "".join(badges)
    )


# =============================================================================
# 25. NLP PIPELINE
# =============================================================================

def run_nlp_pipeline(
    raw_df,
    top_posts,
):
    """Run the complete Reddit NLP pipeline."""

    empty = {
        "clean": pd.DataFrame(),
        "analysis": pd.DataFrame(),
        "best_k": None,
        "keywords": {},
        "evidence": [],
        "top_engaged": pd.DataFrame(),
        "sentiment": pd.DataFrame(),
        "ai_insights": [],
        "overall_review": "",
    }

    clean_df = clean_posts(
        raw_df
    )

    if clean_df.empty:
        return empty

    analysis_df = (
        select_posts_for_analysis(
            clean_df,
            top_posts,
        )
    )

    if analysis_df.empty:
        return empty

    analysis_df = analyze_sentiment(
        analysis_df,
        DEFAULT_BATCH_SIZE,
    )

    analysis_df = analyze_emotions(
        analysis_df,
        DEFAULT_BATCH_SIZE,
    )

    (
        analysis_df,
        _,
        best_k,
    ) = discover_topics(
        analysis_df
    )

    keywords = extract_topic_keywords(
        analysis_df
    )

    analysis_df = (
        calculate_engagement(
            analysis_df
        )
    )

    evidence = create_topic_evidence(
        analysis_df,
        keywords,
    )

    ai_insights = (
        generate_ai_topic_insights(
            evidence,
            DEFAULT_WORKERS,
        )
    )

    return {
        "clean": clean_df,
        "analysis": analysis_df,
        "best_k": best_k,
        "keywords": keywords,
        "evidence": evidence,
        "top_engaged": (
            get_top_engaged_posts(
                analysis_df
            )
        ),
        "sentiment": (
            sentiment_summary(
                analysis_df
            )
        ),
        "ai_insights": ai_insights,
        "overall_review": (
            generate_overall_review(
                evidence
            )
        ),
    }


# =============================================================================
# 26. SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar controls."""

    st.markdown(
        '<link rel="stylesheet" '
        'href="https://cdnjs.cloudflare.com/ajax/libs/'
        'font-awesome/6.7.2/css/all.min.css">',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="reddit-sidebar-icon">
            <i
                class="fa-brands fa-reddit"
                aria-hidden="true">
            </i>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title(
        "Recon Settings"
    )

    with st.sidebar.form(
        "subreddit_form",
        clear_on_submit=False,
    ):

        subreddit_input = (
            st.text_input(
                "Subreddit",
                placeholder=(
                    "e.g. technology, "
                    "Python, gaming"
                ),
                help=(
                    "Enter a subreddit "
                    "without r/"
                ),
            )
        )

        days_back = st.slider(
            "Days to analyze",
            1,
            30,
            1,
        )

        posts_to_fetch = st.slider(
            "Posts to fetch",
            100,
            1000,
            300,
            50,
        )

        top_posts = st.slider(
            "Posts for NLP",
            10,
            500,
            100,
            10,
        )

        submitted = (
            st.form_submit_button(
                "Run Recon",
                type="primary",
                use_container_width=True,
            )
        )

    return (
        subreddit_input,
        days_back,
        posts_to_fetch,
        top_posts,
        submitted,
    )


def render_about_sidebar(
    subreddit,
    days_back,
    posts_to_fetch,
    top_posts,
):
    """Render model/pipeline information."""

    with st.sidebar.expander(
        "About & Pipeline"
    ):

        st.html(
            f"""
            <strong>Current Analysis</strong>

            <ul>
                <li>
                    <strong>Subreddit:</strong>
                    r/{html.escape(
                        subreddit
                    )}
                </li>

                <li>
                    <strong>Days:</strong>
                    {days_back}
                </li>

                <li>
                    <strong>Posts fetched:</strong>
                    {posts_to_fetch}
                </li>

                <li>
                    <strong>Posts analyzed:</strong>
                    {top_posts}
                </li>
            </ul>

            <strong>Models</strong>

            <ul>

                <li>
                    <strong>Sentiment:</strong>
                    <code>
                        {html.escape(
                            SENTIMENT_MODEL
                        )}
                    </code>
                </li>

                <li>
                    <strong>Emotion:</strong>
                    <code>
                        {html.escape(
                            EMOTION_MODEL
                        )}
                    </code>
                </li>

                <li>
                    <strong>Embeddings:</strong>
                    <code>
                        {html.escape(
                            EMBEDDING_MODEL
                        )}
                    </code>
                </li>

                <li>
                    <strong>AI:</strong>
                    <code>
                        {html.escape(
                            LLM_MODEL
                        )}
                    </code>
                </li>

            </ul>

            <strong>Engagement Index</strong>

            <p>
                Score + (2 × Comments)
            </p>

            <strong>Pipeline</strong>

            <p>
                Reddit → Sentiment → Emotion →
                Embeddings → Clustering →
                Topic Keywords → GPT Topic Analysis →
                Community Review
            </p>
            """
        )


# =============================================================================
# 27. RUN RECON
# =============================================================================

def execute_recon(
    subreddit_input,
    days_back,
    posts_to_fetch,
    top_posts,
):
    """Fetch data and execute NLP pipeline."""

    subreddit = normalize_subreddit(
        subreddit_input
    )

    if (
        not subreddit
        or not subreddit.replace(
            "_",
            "",
        ).isalnum()
    ):

        st.sidebar.error(
            "Use a valid subreddit name."
        )

        return

    st.session_state.pop(
        "result",
        None,
    )

    st.session_state[
        "selected_subreddit"
    ] = subreddit

    progress = st.progress(0)

    status = st.empty()

    try:

        status.write(
            f"Fetching posts from "
            f"r/{subreddit}..."
        )

        raw_df = fetch_reddit_posts(
            subreddit,
            posts_to_fetch,
            days_back,
        )

        progress.progress(35)

        if raw_df.empty:

            status.empty()
            progress.empty()

            st.error(
                f"No posts were fetched from "
                f"r/{subreddit}. "
                "Check the subreddit name or "
                "try a longer date range."
            )

            st.stop()

        status.write(
            f"Fetched {len(raw_df):,} posts. "
            "Running sentiment, emotion and "
            "topic analysis..."
        )

        result = run_nlp_pipeline(
            raw_df,
            top_posts,
        )

        progress.progress(85)

        if (
            not isinstance(
                result,
                dict,
            )
            or "analysis"
            not in result
        ):

            raise RuntimeError(
                "NLP pipeline returned "
                "an invalid result."
            )

        result["raw"] = raw_df

        st.session_state[
            "result"
        ] = result

        progress.progress(100)

        status.empty()
        progress.empty()

        st.success(
            f"Recon complete for "
            f"r/{subreddit}."
        )

    except Exception as exc:

        status.empty()
        progress.empty()

        st.error(
            "Recon failed."
        )

        st.exception(exc)


# =============================================================================
# 28. OVERVIEW TAB
# =============================================================================

def render_overview(
    raw_df,
    analysis_df,
    best_k,
    name_map,
):
    """Render the complete executive overview dashboard."""

    # -------------------------------------------------------------------------
    # Filters
    # -------------------------------------------------------------------------

    st.subheader(
        "Dashboard Filters"
    )

    filtered_df = (
        apply_dashboard_filters(
            analysis_df,
            name_map,
        )
    )

    # -------------------------------------------------------------------------
    # KPIs
    # -------------------------------------------------------------------------

    k1, k2, k3, k4 = (
        st.columns(4)
    )

    k1.metric(
        "Total Posts",
        f"{len(filtered_df):,}",
    )

    k2.metric(
        "Avg Comments",
        (
            f"{filtered_df['num_comments'].mean():.1f}"
            if not filtered_df.empty
            else "0.0"
        ),
    )

    k3.metric(
        "Avg Score",
        (
            f"{filtered_df['score'].mean():.1f}"
            if not filtered_df.empty
            else "0.0"
        ),
    )

    k4.metric(
        "Avg Engagement",
        (
            f"{filtered_df['engagement'].mean():.2f}"
            if not filtered_df.empty
            else "0.00"
        ),
    )

    # -------------------------------------------------------------------------
    # Color Key
    # -------------------------------------------------------------------------

    render_color_key()

    # -------------------------------------------------------------------------
    # Sentiment / Emotion
    # -------------------------------------------------------------------------

    sentiment_col, emotion_col = (
        st.columns(2)
    )

    with sentiment_col, card():

        render_sentiment_distribution(
            filtered_df
        )

    with emotion_col, card():

        render_emotion_distribution(
            filtered_df
        )

    # -------------------------------------------------------------------------
    # Engagement
    # -------------------------------------------------------------------------

    st.markdown("---")

    render_engagement_chart(
        filtered_df
    )

    # -------------------------------------------------------------------------
    # Topic Performance + Topic Distribution
    # -------------------------------------------------------------------------

    st.markdown("---")

    topic_col1, topic_col2 = (
        st.columns(2)
    )

    with topic_col1:

        render_topic_performance(
            filtered_df,
            name_map,
        )

    with topic_col2:

        render_topic_distribution(
            filtered_df,
            name_map,
        )

    # -------------------------------------------------------------------------
    # Trends
    # -------------------------------------------------------------------------

    st.markdown("---")

    render_trends(
        filtered_df
    )

    # -------------------------------------------------------------------------
    # Top 5
    # -------------------------------------------------------------------------

    st.markdown("---")

    render_top_posts(
        filtered_df
    )


# =============================================================================
# 29. TOPICS TAB
# =============================================================================

def render_topics_tab(
    analysis_df,
    ai_insights,
):
    """Render topic-level AI analysis."""

    if not ai_insights:

        st.warning(
            "No AI insights were generated."
        )

        return

    for item in ai_insights:

        topic_id = item.get(
            "topic_id"
        )

        topic_df = analysis_df[
            analysis_df[
                "topic_id"
            ]
            == topic_id
        ]

        if topic_df.empty:
            continue

        topic_name = item.get(
            "name",
            f"Topic {topic_id}",
        )

        representative = (
            topic_df
            .sort_values(
                [
                    "score",
                    "num_comments",
                ],
                ascending=False,
            )
            .head(5)
        )

        rep_html = "".join(
            f"""
            <li
                style="margin-bottom:.7rem;"
            >

                <a
                    class="top-post-link"
                    href="{html.escape(
                        str(row.url)
                    )}"
                    target="_blank"
                    rel="noopener noreferrer"
                >

                    <b>
                        {html.escape(
                            str(row.title)
                        )}
                    </b>

                </a>

                <span>
                    (
                    Score:
                    {safe_int(row.score):,}
                    |
                    Comments:
                    {safe_int(row.num_comments):,}
                    )
                </span>

            </li>
            """
            for row
            in representative.itertuples()
        )

        error = item.get(
            "error"
        )

        error_html = (
            f"""
            <p>
                <b>AI status:</b>
                {html.escape(
                    str(error)
                )}
            </p>
            """
            if error
            else ""
        )

        with card():

            st.html(
                f"""
                <div class="topic-title">
                    {html.escape(
                        str(topic_name)
                    )}
                </div>
                """
            )

            render_topic_reaction_badges(
                topic_df
            )

            st.markdown(
                f"""
                **Analysis:** {
                    item.get(
                        "description",
                        "",
                    )
                }
                """
            )

            st.markdown(
                f"""
                **Reaction Context:** {
                    item.get(
                        "main_reaction",
                        "",
                    )
                }
                """
            )

            if error_html:
                st.html(
                    error_html
                )

            st.markdown(
                "---"
            )

            st.markdown(
                "**TOP POSTS IN TOPIC**"
            )

            st.html(
                f"""
                <ul
                    style="
                        list-style-type:none;
                        padding-left:0;
                    "
                >
                    {rep_html}
                </ul>
                """
            )


# =============================================================================
# 30. REVIEW TAB
# =============================================================================

def render_review_tab(
    subreddit,
    overall_review,
    analysis_df,
):
    """Render overall AI review and word cloud."""

    with card():

        st.subheader(
            f"General Consensus: r/{subreddit}"
        )

        if overall_review:

            st.write(
                overall_review
            )

        else:

            st.info(
                "LLM summary is not available."
            )

    with card():

        st.subheader(
            "What they are talking about"
        )

        st.caption(
            "Common terms after removing "
            "standard stopwords."
        )

        wc_fig = (
            generate_word_cloud(
                analysis_df
            )
        )

        if wc_fig is not None:

            st.pyplot(
                wc_fig,
                clear_figure=True,
            )


# =============================================================================
# 31. DATA TAB
# =============================================================================

def render_data_tab(
    analysis_df,
):
    """Render analyzed post dataset."""

    with card():

        st.subheader(
            "Top Posts Dataset"
        )

        display_cols = [
            "score",
            "num_comments",
            "title",
            "sentiment",
            "sentiment_confidence",
            "emotion",
            "emotion_confidence",
            "topic_id",
            "engagement",
            "url",
        ]

        display_cols.extend(
            f"emotion_{label}"
            for label
            in EMOTION_LABELS
        )

        available_cols = [
            column
            for column
            in display_cols
            if column in analysis_df.columns
        ]

        st.dataframe(
            analysis_df.sort_values(
                "score",
                ascending=False,
            )[available_cols],
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# 32. APPLICATION ENTRY POINT
# =============================================================================

def main():
    """Main Streamlit application."""

    (
        subreddit_input,
        days_back,
        posts_to_fetch,
        top_posts,
        submitted,
    ) = render_sidebar()

    # -------------------------------------------------------------------------
    # Execute analysis
    # -------------------------------------------------------------------------

    if submitted:

        execute_recon(
            subreddit_input,
            days_back,
            posts_to_fetch,
            top_posts,
        )

    # -------------------------------------------------------------------------
    # Empty state
    # -------------------------------------------------------------------------

    if "result" not in st.session_state:

        page_header(
            "Reddit Recon",
            (
                "Community intelligence, semantic "
                "topics, sentiment and engagement analysis."
            ),
        )

        st.write(
            "Enter a subreddit in the sidebar "
            "and click **Run Recon**."
        )

        return

    # -------------------------------------------------------------------------
    # Load result
    # -------------------------------------------------------------------------

    result = (
        st.session_state.get(
            "result"
        )
    )

    if not isinstance(
        result,
        dict,
    ):

        st.error(
            "Invalid analysis result."
        )

        st.session_state.pop(
            "result",
            None,
        )

        return

    raw_df = result.get(
        "raw",
        pd.DataFrame(),
    )

    analysis_df = result.get(
        "analysis",
        pd.DataFrame(),
    )

    best_k = result.get(
        "best_k"
    )

    ai_insights = result.get(
        "ai_insights",
        [],
    )

    overall_review = result.get(
        "overall_review",
        "",
    )

    if analysis_df.empty:

        st.error(
            "No usable posts were available "
            "for NLP analysis."
        )

        return

    subreddit = (
        st.session_state.get(
            "selected_subreddit",
            subreddit_input,
        )
    )

    # -------------------------------------------------------------------------
    # Topic name mapping
    # -------------------------------------------------------------------------

    name_map = {
        item.get(
            "topic_id"
        ): item.get(
            "name",
            f"Topic {item.get('topic_id')}",
        )
        for item in ai_insights
    }

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------

    page_header(
        f"Reddit Recon: r/{subreddit}",
        (
            f"Analyzed top "
            f"{len(analysis_df):,} posts."
        ),
    )

    # -------------------------------------------------------------------------
    # Sidebar information
    # -------------------------------------------------------------------------

    render_about_sidebar(
        subreddit,
        days_back,
        posts_to_fetch,
        top_posts,
    )

    # -------------------------------------------------------------------------
    # Tabs
    # -------------------------------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Overview",
            "Topics",
            "Review",
            "Data",
        ]
    )

    # -------------------------------------------------------------------------
    # Overview
    # -------------------------------------------------------------------------

    with tab1:

        render_overview(
            raw_df,
            analysis_df,
            best_k,
            name_map,
        )

    # -------------------------------------------------------------------------
    # Topics
    # -------------------------------------------------------------------------

    with tab2:

        render_topics_tab(
            analysis_df,
            ai_insights,
        )

    # -------------------------------------------------------------------------
    # Review
    # -------------------------------------------------------------------------

    with tab3:

        render_review_tab(
            subreddit,
            overall_review,
            analysis_df,
        )

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    with tab4:

        render_data_tab(
            analysis_df
        )


# =============================================================================
# 33. START APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
