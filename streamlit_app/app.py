```python
import html
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from transformers import AutoModel, AutoTokenizer, pipeline
from wordcloud import WordCloud


# ============================================================
# CONFIG
# ============================================================

ARCTIC_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Fixed AI model — no model selection in UI
LLM_MODEL = "openai/gpt-oss-20b"

DEFAULT_DAYS_BACK = 1
DEFAULT_POSTS_TO_FETCH = 300
DEFAULT_TOP_POSTS = 100

# Fixed internal performance settings
SENTIMENT_BATCH_SIZE = 16
AI_WORKER_THREADS = 3
EMBEDDING_BATCH_SIZE = 32

RANDOM_STATE = 42
MAX_FETCH_RETRIES = 4
MAX_TEXT_CHARS = 2500
TOP_POSTS_PER_TOPIC = 5

REDDIT_ORANGE = "#FF4500"
REDDIT_BLUE = "#0079D3"
REDDIT_DARK = "#1A1A1B"
REDDIT_BG = "#DAE0E6"
CARD_BORDER = "#EDEFF1"

REDDIT_ICON_URL = (
    "https://www.redditstatic.com/desktop2x/img/favicon/"
    "apple-icon-180x180.png"
)

SENTIMENT_COLORS = {
    "positive": "#2AA96D",
    "neutral": "#878A8C",
    "negative": REDDIT_ORANGE,
}

CUSTOM_STOPWORDS = {
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
    "amp",
    "https",
    "http",
    "www",
    "com",
    "removed",
    "deleted",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Reddit Recon",
    page_icon=REDDIT_ICON_URL,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# REDDIT UI THEME
# ============================================================

st.markdown(
    f"""
    <style>

    html, body, [class*="css"] {{
        font-family:
            "IBM Plex Sans",
            "Noto Sans",
            Arial,
            sans-serif;
    }}

    .stApp {{
        background-color: {REDDIT_BG};
    }}

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {{
        background-color: {REDDIT_DARK};
        border-right: 1px solid #343536;
    }}

    section[data-testid="stSidebar"] * {{
        color: #D7DADC !important;
    }}

    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {{
        color: #1A1A1B !important;
        background-color: white !important;
    }}

    section[data-testid="stSidebar"] [data-baseweb="select"] {{
        background-color: #272729;
    }}

    section[data-testid="stSidebar"] .stButton button {{
        border-radius: 7px;
    }}

    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */

    .hero {{
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding: 1.15rem 1.5rem;
        border-radius: 10px;
        background: {REDDIT_ORANGE};
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,.16);
    }}

    .hero img {{
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: white;
        padding: 4px;
        flex-shrink: 0;
    }}

    .hero h1 {{
        margin: 0;
        font-size: 1.75rem;
        line-height: 1.2;
        font-weight: 700;
    }}

    .hero p {{
        margin: .25rem 0 0;
        opacity: .95;
        font-size: .9rem;
    }}

    /* --------------------------------------------------------
       METRIC CARDS
    -------------------------------------------------------- */

    .metric-card {{
        border: 1px solid {CARD_BORDER};
        border-radius: 9px;
        padding: 1rem 1.1rem;
        background: white;
        min-height: 105px;
        box-shadow: 0 1px 2px rgba(0,0,0,.06);
        transition: transform .15s ease, box-shadow .15s ease;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,.08);
    }}

    .metric-label {{
        color: #787C7E;
        font-size: .78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .035em;
        margin-bottom: .25rem;
    }}

    .metric-value {{
        color: {REDDIT_DARK};
        font-size: 1.7rem;
        line-height: 1.2;
        font-weight: 750;
    }}

    /* --------------------------------------------------------
       CARDS
    -------------------------------------------------------- */

    .reddit-card {{
        border: 1px solid {CARD_BORDER};
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-bottom: .8rem;
        background: white;
        box-shadow: 0 1px 1px rgba(0,0,0,.05);
    }}

    .reddit-card:hover {{
        border-color: #898989;
    }}

    .chart-card {{
        border: 1px solid {CARD_BORDER};
        border-radius: 9px;
        padding: .85rem 1rem .5rem;
        background: white;
        box-shadow: 0 1px 2px rgba(0,0,0,.05);
        margin-bottom: .8rem;
    }}

    /* --------------------------------------------------------
       TOPIC CARDS
    -------------------------------------------------------- */

    .topic-card {{
        border: 1px solid {CARD_BORDER};
        border-left: 4px solid {REDDIT_ORANGE};
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
        background: white;
        box-shadow: 0 1px 2px rgba(0,0,0,.05);
    }}

    .topic-card:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,.07);
    }}

    .topic-title {{
        color: {REDDIT_DARK};
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: .45rem;
    }}

    .topic-meta {{
        color: #787C7E;
        font-size: .82rem;
        margin-bottom: .7rem;
    }}

    .topic-section-label {{
        color: #787C7E;
        font-size: .72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .04em;
        margin-top: .85rem;
        margin-bottom: .35rem;
    }}

    /* --------------------------------------------------------
       BADGES
    -------------------------------------------------------- */

    .badge {{
        display: inline-block;
        padding: .17rem .62rem;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 650;
        color: white;
        margin-right: .3rem;
        margin-bottom: .25rem;
    }}

    /* --------------------------------------------------------
       KEYWORD CHIPS
    -------------------------------------------------------- */

    .keyword-chip {{
        display: inline-block;
        padding: .13rem .55rem;
        border-radius: 999px;
        background: #FFE8DE;
        color: #B33500;
        font-size: .76rem;
        margin: .13rem .2rem .13rem 0;
        font-weight: 600;
    }}

    /* --------------------------------------------------------
       AI CARD
    -------------------------------------------------------- */

    .ai-card {{
        border-left: 4px solid {REDDIT_BLUE};
        padding: .95rem 1.1rem;
        margin: .6rem 0 .9rem;
        background: #EEF7FF;
        border-radius: 8px;
    }}

    .ai-label {{
        color: #0066A1;
        font-weight: 700;
        font-size: .8rem;
        text-transform: uppercase;
        letter-spacing: .025em;
    }}

    /* --------------------------------------------------------
       POST LIST
    -------------------------------------------------------- */

    .post-item {{
        padding: .65rem 0;
        border-bottom: 1px solid #F0F1F2;
    }}

    .post-item:last-child {{
        border-bottom: none;
    }}

    .post-title {{
        color: #0079D3;
        font-weight: 650;
        text-decoration: none;
    }}

    .post-title:hover {{
        text-decoration: underline;
    }}

    .post-meta {{
        color: #878A8C;
        font-size: .77rem;
        margin-top: .15rem;
    }}

    /* --------------------------------------------------------
       STATUS / EMPTY STATE
    -------------------------------------------------------- */

    .empty-state {{
        background: white;
        border: 1px solid {CARD_BORDER};
        border-radius: 10px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,.05);
    }}

    .empty-state-icon {{
        font-size: 2.3rem;
        margin-bottom: .5rem;
    }}

    /* --------------------------------------------------------
       STREAMLIT POLISH
    -------------------------------------------------------- */

    div[data-testid="stMetric"] {{
        background: white;
        border: 1px solid {CARD_BORDER};
        border-radius: 8px;
        padding: .75rem;
    }}

    div[data-testid="stTabs"] button {{
        font-weight: 600;
    }}

    div[data-testid="stDownloadButton"] button {{
        border-radius: 7px;
    }}

    @media (max-width: 900px) {{
        .hero {{
            padding: 1rem;
        }}

        .hero h1 {{
            font-size: 1.4rem;
        }}

        .metric-card {{
            min-height: 90px;
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_int(value):
    try:
        return int(value)
    except Exception:
        return 0


def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""


def normalize_subreddit(value):
    value = value.strip()

    if value.lower().startswith("r/"):
        value = value[2:]

    return value.strip().replace(" ", "")


def sentiment_badge(label, pct=None):
    color = SENTIMENT_COLORS.get(label, "#878A8C")

    text = (
        label.capitalize()
        if pct is None
        else f"{label.capitalize()} {pct:.0f}%"
    )

    return (
        f'<span class="badge" '
        f'style="background:{color};">'
        f'{html.escape(text)}'
        f'</span>'
    )


def fallback_topic_name(keywords):
    if not keywords:
        return "Uncategorized discussion"

    return " / ".join(
        str(k).title()
        for k in keywords[:3]
    )


def format_number(value):
    try:
        value = float(value)

        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"

        if abs(value) >= 1_000:
            return f"{value / 1_000:.1f}K"

        return f"{value:,.0f}"

    except Exception:
        return "0"


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_http_session():

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

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=4,
        pool_maxsize=4,
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent": "reddit-recon-streamlit/3.0"
        }
    )

    return session


# ============================================================
# MODELS
# ============================================================

@st.cache_resource(
    show_spinner="Loading sentiment model..."
)
def load_sentiment_model():

    return pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        truncation=True,
        max_length=256,
        device=-1,
    )


@st.cache_resource(
    show_spinner="Loading topic embedding model..."
)
def load_embedding_model():

    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL
    )

    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL
    )

    model.eval()

    return tokenizer, model


# ============================================================
# REDDIT DATA
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def fetch_reddit_posts(
    subreddit,
    posts_to_fetch,
    days_back,
):

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

    posts = []
    seen_ids = set()

    fields = (
        "id,created_utc,score,"
        "num_comments,subreddit,"
        "title,selftext,url"
    )

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
                f"Arctic Shift request failed: {exc}"
            )

            break

        batch = payload.get(
            "data",
            []
        )

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
                item.get(
                    "created_utc",
                    0,
                )
            )

            if created < cutoff:
                continue

            seen_ids.add(post_id)

            posts.append(
                {
                    "id": post_id,
                    "title": str(
                        item.get("title")
                        or ""
                    ),
                    "selftext": str(
                        item.get("selftext")
                        or ""
                    ),
                    "score": safe_int(
                        item.get(
                            "score",
                            0,
                        )
                    ),
                    "num_comments": safe_int(
                        item.get(
                            "num_comments",
                            0,
                        )
                    ),
                    "created_utc": created,
                    "url": str(
                        item.get("url")
                        or ""
                    ),
                    "subreddit": str(
                        item.get(
                            "subreddit"
                        )
                        or subreddit
                    ),
                }
            )

            if (
                oldest_timestamp is None
                or created < oldest_timestamp
            ):
                oldest_timestamp = created

            if len(posts) >= posts_to_fetch:
                break

        if len(posts) >= posts_to_fetch:
            break

        if (
            oldest_timestamp is None
            or oldest_timestamp <= cutoff
        ):
            break

        before = (
            oldest_timestamp - 1
        )

        time.sleep(0.15)

    return (
        pd.DataFrame(posts)
        .drop_duplicates("id")
        .reset_index(drop=True)
    )


# ============================================================
# CLEANING / SELECTION
# ============================================================

def clean_posts(
    df,
    min_text_length=20,
):

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
    ).str.replace(
        r"\s+",
        " ",
        regex=True,
    ).str.strip()

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


def select_top_posts(
    df,
    top_posts,
):

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


# ============================================================
# SENTIMENT
# ============================================================

def analyze_sentiment(df):

    if df.empty:
        return df.copy()

    model = load_sentiment_model()

    result = df.copy()

    predictions = model(
        result["model_text"].tolist(),
        batch_size=SENTIMENT_BATCH_SIZE,
    )

    result["sentiment"] = [
        p["label"]
        .lower()
        .strip()
        for p in predictions
    ]

    result["sentiment_confidence"] = [
        float(p["score"])
        for p in predictions
    ]

    return result


# ============================================================
# EMBEDDINGS
# ============================================================

def mean_pool(
    last_hidden_state,
    attention_mask,
):

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
            start:start
            + EMBEDDING_BATCH_SIZE
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
                encoded[
                    "attention_mask"
                ],
            )

            pooled = (
                torch.nn.functional
                .normalize(
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


# ============================================================
# TOPIC DISCOVERY
# ============================================================

def discover_topics(df):

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

    silhouette_results = []

    for k in range(
        3,
        max_k + 1,
    ):

        try:

            km = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=5,
            )

            labels = km.fit_predict(
                embeddings
            )

            if len(
                np.unique(labels)
            ) < 2:
                continue

            score = silhouette_score(
                embeddings,
                labels,
                metric="cosine",
            )

            silhouette_results.append(
                {
                    "k": k,
                    "silhouette_score": float(
                        score
                    ),
                }
            )

        except Exception:
            continue

    if not silhouette_results:

        best_k = min(
            5,
            len(df) - 1,
        )

        final_model = KMeans(
            n_clusters=best_k,
            random_state=RANDOM_STATE,
            n_init=5,
        )

        result = df.copy()

        result["topic_id"] = (
            final_model.fit_predict(
                embeddings
            )
        )

        return (
            result,
            pd.DataFrame(),
            best_k,
        )

    silhouette_df = pd.DataFrame(
        silhouette_results
    )

    best_k = int(
        silhouette_df.loc[
            silhouette_df[
                "silhouette_score"
            ].idxmax(),
            "k",
        ]
    )

    final_model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=5,
    )

    result = df.copy()

    result["topic_id"] = (
        final_model.fit_predict(
            embeddings
        )
    )

    return (
        result,
        silhouette_df,
        best_k,
    )


# ============================================================
# TF-IDF KEYWORDS
# ============================================================

def extract_topic_keywords(
    df,
    top_n=10,
):

    if (
        df.empty
        or "topic_id" not in df.columns
    ):
        return {}

    stopwords = (
        set(ENGLISH_STOP_WORDS)
        | CUSTOM_STOPWORDS
    )

    try:

        vectorizer = TfidfVectorizer(
            stop_words=list(
                stopwords
            ),
            max_features=4000,
            ngram_range=(1, 2),
            min_df=2,
        )

        matrix = (
            vectorizer.fit_transform(
                df["model_text"]
            )
        )

    except ValueError:
        return {}

    feature_names = np.array(
        vectorizer
        .get_feature_names_out()
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
            topic_scores.argsort()[
                ::-1
            ][:top_n]
        )

        keywords[
            int(topic_id)
        ] = (
            feature_names[
                top_indexes
            ].tolist()
        )

    return keywords


# ============================================================
# ENGAGEMENT
# ============================================================

def calculate_engagement(df):

    result = df.copy()

    result["score"] = (
        pd.to_numeric(
            result["score"],
            errors="coerce",
        )
        .fillna(0)
    )

    result["num_comments"] = (
        pd.to_numeric(
            result["num_comments"],
            errors="coerce",
        )
        .fillna(0)
    )

    result["log_score"] = np.log1p(
        result["score"].clip(
            lower=0
        )
    )

    result["log_comments"] = np.log1p(
        result["num_comments"].clip(
            lower=0
        )
    )

    result["engagement"] = (
        result["log_score"]
        + result["log_comments"]
    )

    return result


def get_top_engaged_posts(
    df,
    n=10,
):

    columns = [
        "title",
        "score",
        "num_comments",
        "sentiment",
        "topic_id",
        "topic_name",
        "engagement",
        "url",
    ]

    available = [
        c
        for c in columns
        if c in df.columns
    ]

    return (
        df.sort_values(
            "engagement",
            ascending=False,
        )
        [available]
        .head(n)
        .reset_index(drop=True)
    )


def sentiment_summary(df):

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


# ============================================================
# TOPIC EVIDENCE
# ============================================================

def create_topic_evidence(
    df,
    keywords,
    top_n_posts=TOP_POSTS_PER_TOPIC,
):

    evidence = []

    for topic_id in sorted(
        df["topic_id"].unique()
    ):

        topic_df = df[
            df["topic_id"] == topic_id
        ].copy()

        representative = (
            topic_df
            .sort_values(
                ["score", "num_comments"],
                ascending=False,
            )
            .head(top_n_posts)
        )

        posts = []

        for _, row in representative.iterrows():

            posts.append(
                {
                    "title": str(
                        row.get(
                            "title",
                            "",
                        )
                    ),
                    "text": str(
                        row.get(
                            "selftext",
                            "",
                        )
                    )[:1000],
                    "score": safe_int(
                        row.get(
                            "score",
                            0,
                        )
                    ),
                    "comments": safe_int(
                        row.get(
                            "num_comments",
                            0,
                        )
                    ),
                }
            )

        topic_sentiment = (
            topic_df[
                "sentiment"
            ]
            .value_counts(
                normalize=True
            )
            * 100
        ).round(1).to_dict()

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
                "sentiment_breakdown":
                    topic_sentiment,
                "keywords":
                    keywords.get(
                        int(topic_id),
                        [],
                    ),
                "representative_posts":
                    posts,
            }
        )

    return evidence


# ============================================================
# GROQ / GPT-OSS
# ============================================================

def groq_extra_kwargs():

    return {
        "reasoning_effort": "low",
        "include_reasoning": False,
    }


def analyze_topic_with_groq(
    topic_evidence,
    api_key,
):

    topic_id = topic_evidence[
        "topic_id"
    ]

    if not api_key:

        return {
            "topic_id": topic_id,
            "error":
                "GROQ_API_KEY is not configured.",
            "name":
                f"Topic {topic_id}",
        }

    try:

        from groq import Groq

        client = Groq(
            api_key=api_key
        )

        prompt = f"""
You are an expert Reddit community analyst.

Analyze Topic {topic_id} using ONLY the evidence supplied below.

Give this topic a short, human-readable name, similar to a news
headline or community discussion title.

Then explain:
1. What people are discussing.
2. The dominant reaction.
3. Why people appear to be reacting that way.
4. Any important concerns.
5. Whether there is meaningful disagreement.

Rules:
- Do not invent facts.
- Do not claim the entire subreddit agrees.
- The evidence is a sample.
- Base the interpretation primarily on representative posts.
- Keywords are supporting context only.
- Do not calculate sentiment percentages yourself.
- Mention disagreement only if the evidence supports it.
- Keep the response concise and useful.

Evidence:
{json.dumps(topic_evidence, indent=2)}
"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_completion_tokens=500,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name":
                        "reddit_topic_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "topic_id":
                                {
                                    "type":
                                        "integer"
                                },
                            "name":
                                {
                                    "type":
                                        "string"
                                },
                            "description":
                                {
                                    "type":
                                        "string"
                                },
                            "main_reaction":
                                {
                                    "type":
                                        "string"
                                },
                            "key_concerns":
                                {
                                    "type":
                                        "array",
                                    "items":
                                        {
                                            "type":
                                                "string"
                                        },
                                },
                            "disagreement":
                                {
                                    "type":
                                        "string"
                                },
                        },
                        "required": [
                            "topic_id",
                            "name",
                            "description",
                            "main_reaction",
                            "key_concerns",
                            "disagreement",
                        ],
                        "additionalProperties":
                            False,
                    },
                },
            },
            **groq_extra_kwargs(),
        )

        return json.loads(
            response
            .choices[0]
            .message
            .content
        )

    except Exception as exc:

        return {
            "topic_id": topic_id,
            "error": str(exc),
            "name":
                f"Topic {topic_id}",
        }


def generate_ai_topic_insights(
    evidence,
):

    if not evidence:
        return []

    api_key = get_groq_key()

    if not api_key:

        return [
            {
                "topic_id":
                    item["topic_id"],
                "name":
                    f"Topic {item['topic_id']}",
                "error":
                    "GROQ_API_KEY is not configured.",
            }
            for item in evidence
        ]

    results = []

    with ThreadPoolExecutor(
        max_workers=min(
            AI_WORKER_THREADS,
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
                        "topic_id":
                            item[
                                "topic_id"
                            ],
                        "name":
                            f"Topic {item['topic_id']}",
                        "error":
                            str(exc),
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


# ============================================================
# OVERALL REVIEW
# ============================================================

def generate_overall_review(
    subreddit,
    sentiment_df,
    topic_summaries,
):

    api_key = get_groq_key()

    fallback = {
        "consensus":
            "AI review is unavailable.",
        "overall_sentiment":
            "See the sentiment distribution for the statistical breakdown.",
        "notable_details": [],
    }

    if not api_key:

        fallback["error"] = (
            "GROQ_API_KEY is not configured."
        )

        return fallback

    try:

        from groq import Groq

        client = Groq(
            api_key=api_key
        )

        evidence = {
            "subreddit":
                subreddit,
            "sentiment_breakdown":
                sentiment_df.to_dict(
                    orient="records"
                ),
            "topics":
                topic_summaries,
        }

        prompt = f"""
You are summarizing a Reddit community analysis.

The analysis is based on a sample of recent high-engagement posts.

Use ONLY the evidence below.

Write:
1. consensus — 3-5 sentences describing the apparent consensus,
   mood, or dominant discussion patterns.
2. overall_sentiment — 1-2 sentences describing the overall sentiment
   and its likely drivers.
3. notable_details — 3-6 concise bullet-style observations about
   interesting, niche, surprising, or disputed topics.

Rules:
- Do not claim this represents the entire subreddit.
- Explicitly treat it as a sample.
- Do not invent facts.
- Do not infer details that are not supported by the evidence.

Evidence:
{json.dumps(evidence, indent=2)}
"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_completion_tokens=700,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name":
                        "reddit_overall_summary",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "consensus":
                                {
                                    "type":
                                        "string"
                                },
                            "overall_sentiment":
                                {
                                    "type":
                                        "string"
                                },
                            "notable_details":
                                {
                                    "type":
                                        "array",
                                    "items":
                                        {
                                            "type":
                                                "string"
                                        },
                                },
                        },
                        "required": [
                            "consensus",
                            "overall_sentiment",
                            "notable_details",
                        ],
                        "additionalProperties":
                            False,
                    },
                },
            },
            **groq_extra_kwargs(),
        )

        return json.loads(
            response
            .choices[0]
            .message
            .content
        )

    except Exception as exc:

        fallback["error"] = str(exc)

        return fallback


# ============================================================
# WORD CLOUD
# ============================================================

def generate_word_cloud(df):

    if df.empty:
        return None

    text = " ".join(
        df["title"]
        .fillna("")
        .astype(str)
        + " "
        + df["selftext"]
        .fillna("")
        .astype(str)
    )

    stopwords = (
        set(ENGLISH_STOP_WORDS)
        | CUSTOM_STOPWORDS
    )

    if not text.strip():
        return None

    wordcloud = WordCloud(
        width=1000,
        height=420,
        background_color="white",
        stopwords=stopwords,
        collocations=False,
        max_words=120,
        color_func=lambda *args, **kwargs:
            REDDIT_ORANGE
            if np.random.rand() > 0.5
            else REDDIT_DARK,
    ).generate(text)

    fig, ax = plt.subplots(
        figsize=(10, 4.2)
    )

    ax.imshow(
        wordcloud,
        interpolation="bilinear",
    )

    ax.axis("off")

    fig.tight_layout(
        pad=0
    )

    return fig


# ============================================================
# FULL NLP PIPELINE
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def run_nlp_pipeline(
    raw_df,
    top_posts,
):

    empty = {
        "clean": pd.DataFrame(),
        "analysis": pd.DataFrame(),
        "best_k": None,
        "keywords": {},
        "evidence": [],
        "top_engaged":
            pd.DataFrame(),
        "sentiment":
            pd.DataFrame(),
        "ai_insights": [],
        "overall_review": {},
    }

    if raw_df.empty:
        return empty

    clean_df = clean_posts(
        raw_df
    )

    if clean_df.empty:
        return empty

    analysis_df = select_top_posts(
        clean_df,
        top_posts,
    )

    if analysis_df.empty:
        return empty

    analysis_df = analyze_sentiment(
        analysis_df
    )

    (
        analysis_df,
        silhouette_df,
        best_k,
    ) = discover_topics(
        analysis_df
    )

    keywords = extract_topic_keywords(
        analysis_df
    )

    analysis_df = calculate_engagement(
        analysis_df
    )

    evidence = create_topic_evidence(
        analysis_df,
        keywords,
    )

    sentiment = sentiment_summary(
        analysis_df
    )

    ai_insights = (
        generate_ai_topic_insights(
            evidence
        )
    )

    # --------------------------------------------------------
    # Topic names
    # --------------------------------------------------------

    topic_names = {}

    for item in ai_insights:

        tid = item.get(
            "topic_id"
        )

        if (
            "error" not in item
            and item.get("name")
        ):

            topic_names[tid] = (
                item["name"]
            )

    for item in evidence:

        tid = item[
            "topic_id"
        ]

        if tid not in topic_names:

            topic_names[tid] = (
                fallback_topic_name(
                    item.get(
                        "keywords",
                        [],
                    )
                )
            )

    analysis_df[
        "topic_name"
    ] = (
        analysis_df[
            "topic_id"
        ].map(topic_names)
    )

    top_engaged = (
        get_top_engaged_posts(
            analysis_df
        )
    )

    # --------------------------------------------------------
    # Overall review evidence
    # --------------------------------------------------------

    topic_summaries = []

    for item in evidence:

        tid = item[
            "topic_id"
        ]

        topic_summaries.append(
            {
                "topic_id": tid,
                "name":
                    topic_names.get(
                        tid,
                        f"Topic {tid}",
                    ),
                "post_count":
                    item["post_count"],
                "avg_score":
                    item["avg_score"],
                "avg_comments":
                    item["avg_comments"],
                "sentiment_breakdown":
                    item[
                        "sentiment_breakdown"
                    ],
                "keywords":
                    item["keywords"],
            }
        )

    subreddit = ""

    if (
        not raw_df.empty
        and "subreddit"
        in raw_df.columns
    ):

        subreddit = str(
            raw_df[
                "subreddit"
            ].iloc[0]
        )

    overall_review = (
        generate_overall_review(
            subreddit,
            sentiment,
            topic_summaries,
        )
    )

    return {
        "clean":
            clean_df,
        "analysis":
            analysis_df,
        "best_k":
            best_k,
        "keywords":
            keywords,
        "evidence":
            evidence,
        "top_engaged":
            top_engaged,
        "sentiment":
            sentiment,
        "ai_insights":
            {
                i.get("topic_id"):
                    i
                for i in ai_insights
            },
        "topic_names":
            topic_names,
        "overall_review":
            overall_review,
        "silhouette":
            silhouette_df,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:.55rem;
            margin-bottom:.8rem;
        ">
            <img
                src="{REDDIT_ICON_URL}"
                style="
                    width:34px;
                    height:34px;
                    border-radius:50%;
                    background:white;
                    padding:3px;
                "
            >
            <div>
                <div style="
                    font-size:1.2rem;
                    font-weight:750;
                    color:white;
                ">
                    Reddit Recon
                </div>
                <div style="
                    font-size:.72rem;
                    color:#878A8C;
                ">
                    Community intelligence
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form(
        "subreddit_form"
    ):

        subreddit_input = st.text_input(
            "Subreddit",
            placeholder=(
                "GTA6, gaming, technology, stocks..."
            ),
            help=(
                "Enter a subreddit name. "
                "You can include or omit r/."
            ),
        )

        days_back = st.slider(
            "Days to analyze",
            1,
            30,
            DEFAULT_DAYS_BACK,
        )

        posts_to_fetch = st.slider(
            "Posts to fetch",
            100,
            1000,
            DEFAULT_POSTS_TO_FETCH,
            50,
        )

        top_posts = st.slider(
            "Top posts for NLP",
            10,
            500,
            DEFAULT_TOP_POSTS,
            10,
        )

        submitted = st.form_submit_button(
            "🚀 Run Recon",
            type="primary",
            use_container_width=True,
        )

    st.divider()

    st.markdown(
        """
        <div style="
            font-size:.72rem;
            color:#878A8C;
            text-transform:uppercase;
            font-weight:700;
            letter-spacing:.05em;
            margin-bottom:.35rem;
        ">
            AI Engine
        </div>
        <div style="
            background:#272729;
            border:1px solid #343536;
            border-radius:7px;
            padding:.65rem .75rem;
            margin-bottom:.7rem;
        ">
            <div style="
                color:white;
                font-size:.85rem;
                font-weight:650;
            ">
                GPT-OSS 20B
            </div>
            <div style="
                color:#878A8C;
                font-size:.7rem;
                margin-top:.15rem;
            ">
                Automatic topic interpretation
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(
        "ℹ️ About & Pipeline"
    ):

        st.markdown(
            f"""
            **Current Analysis**

            - **Data source:** Arctic Shift Reddit API
            - **Sentiment:** `{SENTIMENT_MODEL}`
            - **Embeddings:** `{EMBEDDING_MODEL}`
            - **Clustering:** K-Means
            - **Topic selection:** Silhouette score
            - **Keywords:** TF-IDF
            - **AI:** `{LLM_MODEL}` via Groq
            - **AI workers:** {AI_WORKER_THREADS}
            - **Sentiment batch:** {SENTIMENT_BATCH_SIZE}

            **Pipeline**

            Reddit posts → cleaning → engagement selection →
            sentiment → semantic embeddings → K-Means topics →
            TF-IDF keywords → GPT-OSS topic analysis →
            subreddit review.
            """
        )


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <img src="{REDDIT_ICON_URL}">
        <div>
            <h1>Reddit Recon</h1>
            <p>
                Sentiment, semantic topics, engagement and
                AI-powered community intelligence.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIAL STATE
# ============================================================

if "selected_subreddit" not in st.session_state:

    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">🔎</div>
            <h3>Ready for reconnaissance</h3>
            <p>
                Enter a public subreddit in the sidebar,
                configure the analysis window, and run Recon.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# SUBREDDIT SUBMISSION
# ============================================================

if submitted:

    subreddit = normalize_subreddit(
        subreddit_input
    )

    if not subreddit:

        st.sidebar.error(
            "Enter a subreddit name."
        )

        st.stop()

    if not subreddit.replace(
        "_",
        "",
    ).isalnum():

        st.sidebar.error(
            "Use a valid subreddit name, "
            "e.g. GTA6 or r/GTA6."
        )

        st.stop()

    st.session_state[
        "selected_subreddit"
    ] = subreddit

    st.session_state.pop(
        "result",
        None,
    )

    progress = st.progress(
        0
    )

    status = st.empty()

    status.write(
        f"Fetching recent posts from "
        f"r/{subreddit}..."
    )

    raw_df = fetch_reddit_posts(
        subreddit=subreddit,
        posts_to_fetch=posts_to_fetch,
        days_back=days_back,
    )

    progress.progress(
        0.3
    )

    if raw_df.empty:

        progress.empty()
        status.empty()

        st.error(
            "No posts were fetched. "
            "Check the subreddit name or try again later."
        )

        st.stop()

    status.write(
        f"Fetched {len(raw_df):,} posts. "
        "Running NLP + GPT-OSS analysis..."
    )

    result = run_nlp_pipeline(
        raw_df=raw_df,
        top_posts=top_posts,
    )

    progress.progress(
        1.0
    )

    progress.empty()
    status.empty()

    result["raw"] = raw_df

    result["config"] = {
        "days_back":
            days_back,
        "posts_to_fetch":
            posts_to_fetch,
        "top_posts":
            top_posts,
        "sentiment_model":
            SENTIMENT_MODEL,
        "embedding_model":
            EMBEDDING_MODEL,
        "llm_model":
            LLM_MODEL,
    }

    st.session_state[
        "result"
    ] = result

    st.toast(
        f"Recon complete for r/{subreddit}!"
    )


# ============================================================
# RESULT STATE
# ============================================================

if "result" not in st.session_state:

    st.info(
        "Enter a subreddit in the sidebar and "
        "click **Run Recon** to get started."
    )

    st.stop()


result = st.session_state[
    "result"
]

SUBREDDIT = st.session_state[
    "selected_subreddit"
]

raw_df = result[
    "raw"
]

analysis_df = result[
    "analysis"
]

best_k = result[
    "best_k"
]

keywords = result[
    "keywords"
]

top_engaged = result[
    "top_engaged"
]

sentiment = result[
    "sentiment"
]

ai_insights = result[
    "ai_insights"
]

topic_names = result.get(
    "topic_names",
    {},
)

overall_review = result[
    "overall_review"
]

cfg = result[
    "config"
]

if analysis_df.empty:

    st.error(
        "No usable posts were available "
        "for NLP analysis."
    )

    st.stop()


# ============================================================
# CONTEXT BAR
# ============================================================

st.caption(
    f"Analyzing **r/{SUBREDDIT}** · "
    f"last {cfg['days_back']} day"
    f"{'s' if cfg['days_back'] != 1 else ''} · "
    f"{len(raw_df):,} fetched · "
    f"{len(analysis_df):,} analyzed · "
    f"GPT-OSS 20B"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Overview",
        "🧠 Topics",
        "📝 Review",
        "📁 Data",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.subheader(
        "Executive Summary"
    )

    positive_pct = 0.0

    if not sentiment.empty:

        positive_pct = float(
            sentiment.loc[
                sentiment["sentiment"]
                == "positive",
                "percentage",
            ].sum()
        )

    avg_score = (
        analysis_df["score"]
        .mean()
    )

    avg_comments = (
        analysis_df[
            "num_comments"
        ].mean()
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    metrics = [
        (
            "Posts fetched",
            f"{len(raw_df):,}",
        ),
        (
            "Posts analyzed",
            f"{len(analysis_df):,}",
        ),
        (
            "Topics discovered",
            str(
                best_k
                if best_k
                else "N/A"
            ),
        ),
        (
            "Positive sentiment",
            f"{positive_pct:.1f}%",
        ),
        (
            "Avg. score",
            format_number(
                avg_score
            ),
        ),
    ]

    for col, (
        label,
        value,
    ) in zip(
        [
            col1,
            col2,
            col3,
            col4,
            col5,
        ],
        metrics,
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        {html.escape(label)}
                    </div>
                    <div class="metric-value">
                        {html.escape(value)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # --------------------------------------------------------
    # SENTIMENT / TOPIC VOLUME
    # --------------------------------------------------------

    chart_col1, chart_col2 = (
        st.columns(2)
    )

    with chart_col1:

        st.markdown(
            '<div class="chart-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "**Sentiment distribution**"
        )

        if not sentiment.empty:

            sentiment_plot = (
                sentiment
                .sort_values(
                    "percentage",
                    ascending=False,
                )
            )

            fig = px.bar(
                sentiment_plot,
                x="sentiment",
                y="percentage",
                color="sentiment",
                color_discrete_map=
                    SENTIMENT_COLORS,
                text=(
                    sentiment_plot[
                        "percentage"
                    ]
                    .round(1)
                    .astype(str)
                    + "%"
                ),
                labels={
                    "sentiment":
                        "Sentiment",
                    "percentage":
                        "Percentage",
                },
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                showlegend=False,
                margin=dict(
                    t=10,
                    r=10,
                    b=10,
                    l=10,
                ),
                height=350,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                },
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with chart_col2:

        st.markdown(
            '<div class="chart-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "**Posts by topic**"
        )

        topic_counts = (
            analysis_df
            .groupby(
                [
                    "topic_id",
                    "topic_name",
                ]
            )
            .size()
            .reset_index(
                name="posts"
            )
            .sort_values(
                "posts",
                ascending=True,
            )
        )

        fig_topic = px.bar(
            topic_counts,
            x="posts",
            y="topic_name",
            orientation="h",
            text="posts",
            color_discrete_sequence=[
                REDDIT_ORANGE
            ],
            labels={
                "posts":
                    "Posts",
                "topic_name":
                    "Topic",
            },
        )

        fig_topic.update_traces(
            textposition="outside"
        )

        fig_topic.update_layout(
            margin=dict(
                t=10,
                r=10,
                b=10,
                l=10,
            ),
            height=350,
        )

        st.plotly_chart(
            fig_topic,
            use_container_width=True,
            config={
                "displayModeBar": False
            },
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # ENGAGEMENT / VOLUME
    # --------------------------------------------------------

    chart_col3, chart_col4 = (
        st.columns(2)
    )

    with chart_col3:

        st.markdown(
            '<div class="chart-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "**Topic engagement**"
        )

        topic_eng = (
            analysis_df
            .groupby(
                [
                    "topic_id",
                    "topic_name",
                ]
            )
            .agg(
                posts=(
                    "id",
                    "count",
                ),
                avg_score=(
                    "score",
                    "mean",
                ),
                avg_comments=(
                    "num_comments",
                    "mean",
                ),
            )
            .reset_index()
        )

        fig_eng = px.scatter(
            topic_eng,
            x="avg_score",
            y="avg_comments",
            size="posts",
            color="topic_name",
            hover_data=[
                "posts",
                "avg_score",
                "avg_comments",
            ],
            labels={
                "avg_score":
                    "Average score",
                "avg_comments":
                    "Average comments",
                "topic_name":
                    "Topic",
            },
        )

        fig_eng.update_layout(
            height=360,
            margin=dict(
                t=10,
                r=10,
                b=10,
                l=10,
            ),
            showlegend=False,
        )

        st.plotly_chart(
            fig_eng,
            use_container_width=True,
            config={
                "displayModeBar": False
            },
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with chart_col4:

        st.markdown(
            '<div class="chart-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "**Post volume over time**"
        )

        volume_df = raw_df.copy()

        volume_df["date"] = (
            pd.to_datetime(
                volume_df[
                    "created_utc"
                ],
                unit="s",
            ).dt.date
        )

        daily = (
            volume_df
            .groupby("date")
            .size()
            .reset_index(
                name="posts"
            )
        )

        fig_vol = px.line(
            daily,
            x="date",
            y="posts",
            markers=True,
            color_discrete_sequence=[
                REDDIT_ORANGE
            ],
            labels={
                "date":
                    "Date",
                "posts":
                    "Posts fetched",
            },
        )

        fig_vol.update_layout(
            height=360,
            margin=dict(
                t=10,
                r=10,
                b=10,
                l=10,
            ),
        )

        st.plotly_chart(
            fig_vol,
            use_container_width=True,
            config={
                "displayModeBar": False
            },
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # TOP TOPICS
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "**Top topics at a glance**"
    )

    top_topics = (
        analysis_df
        .groupby(
            [
                "topic_id",
                "topic_name",
            ]
        )
        .agg(
            posts=(
                "id",
                "count",
            ),
            avg_score=(
                "score",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "posts",
            ascending=False,
        )
        .head(3)
    )

    if not top_topics.empty:

        preview_cols = st.columns(
            len(top_topics)
        )

        for col, (
            _,
            row,
        ) in zip(
            preview_cols,
            top_topics.iterrows(),
        ):

            tid = int(
                row["topic_id"]
            )

            chips = "".join(
                f"""
                <span class="keyword-chip">
                    {html.escape(str(kw))}
                </span>
                """
                for kw in keywords.get(
                    tid,
                    [],
                )[:5]
            )

            with col:

                st.markdown(
                    f"""
                    <div class="reddit-card">
                        <div style="
                            font-weight:700;
                            font-size:1rem;
                            color:{REDDIT_DARK};
                        ">
                            {html.escape(
                                str(
                                    row[
                                        "topic_name"
                                    ]
                                )
                            )}
                        </div>

                        <div style="
                            color:#787C7E;
                            font-size:.8rem;
                            margin-top:.25rem;
                        ">
                            {int(row["posts"]):,} posts
                            · avg score
                            {row["avg_score"]:,.0f}
                        </div>

                        <div style="
                            margin-top:.55rem;
                        ">
                            {chips}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # --------------------------------------------------------
    # TOP ENGAGED POSTS
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "**Highest-engagement posts**"
    )

    if not top_engaged.empty:

        for _, row in (
            top_engaged.head(5)
            .iterrows()
        ):

            title = html.escape(
                str(
                    row.get(
                        "title",
                        "Untitled",
                    )
                )
            )

            url = html.escape(
                str(
                    row.get(
                        "url",
                        "",
                    )
                ),
                quote=True,
            )

            topic = html.escape(
                str(
                    row.get(
                        "topic_name",
                        "Unknown",
                    )
                )
            )

            st.markdown(
                f"""
                <div class="reddit-card">
                    <a
                        class="post-title"
                        href="{url}"
                        target="_blank"
                    >
                        {title}
                    </a>

                    <div class="post-meta">
                        ↑ {safe_int(row.get("score", 0)):,}
                        · 💬 {safe_int(row.get("num_comments", 0)):,}
                        · Topic: {topic}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# TAB 2 — TOPICS
# ============================================================

with tab2:

    st.subheader(
        "Semantic Topics"
    )

    st.caption(
        "Topics are discovered using MiniLM embeddings + K-Means. "
        "GPT-OSS 20B automatically names and interprets each cluster."
    )

    if not get_groq_key():

        st.warning(
            "GROQ_API_KEY is not configured. "
            "Keyword-based topic names are being shown."
        )

    for topic_id in sorted(
        analysis_df[
            "topic_id"
        ].unique()
    ):

        topic_id = int(
            topic_id
        )

        topic_df = (
            analysis_df[
                analysis_df[
                    "topic_id"
                ]
                == topic_id
            ]
            .copy()
        )

        name = topic_names.get(
            topic_id,
            fallback_topic_name(
                keywords.get(
                    topic_id,
                    [],
                )
            ),
        )

        insight = ai_insights.get(
            topic_id,
            {},
        )

        topic_sentiment = (
            topic_df[
                "sentiment"
            ]
            .value_counts(
                normalize=True
            )
            * 100
        )

        badges = "".join(
            sentiment_badge(
                label,
                pct,
            )
            for label, pct
            in topic_sentiment
                .sort_values(
                    ascending=False
                )
                .items()
        )

        chips = "".join(
            f"""
            <span class="keyword-chip">
                {html.escape(str(kw))}
            </span>
            """
            for kw in keywords.get(
                topic_id,
                [],
            )
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
            .head(
                TOP_POSTS_PER_TOPIC
            )
        )

        posts_html = ""

        for _, row in representative.iterrows():

            title = html.escape(
                str(
                    row.get(
                        "title",
                        "",
                    )
                )
            )

            url = html.escape(
                str(
                    row.get(
                        "url",
                        "",
                    )
                ),
                quote=True,
            )

            posts_html += f"""
            <div class="post-item">
                <a
                    class="post-title"
                    href="{url}"
                    target="_blank"
                >
                    {title}
                </a>

                <div class="post-meta">
                    ↑ {safe_int(row.get("score", 0)):,}
                    · 💬 {safe_int(row.get("num_comments", 0)):,}
                </div>
            </div>
            """

        st.markdown(
            f"""
            <div class="topic-card">

                <div class="topic-title">
                    {html.escape(str(name))}
                </div>

                <div style="
                    margin-bottom:.45rem;
                ">
                    {badges}
                </div>

                <div class="topic-meta">
                    {len(topic_df):,} posts
                    · avg score
                    {topic_df["score"].mean():,.0f}
                    · avg comments
                    {topic_df["num_comments"].mean():,.0f}
                </div>

                <div>
                    {chips}
                </div>

                <div class="topic-section-label">
                    Top posts
                </div>

                {posts_html}

            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # AI INTERPRETATION
        # ----------------------------------------------------

        if "error" in insight:

            st.caption(
                "AI interpretation unavailable: "
                + str(
                    insight.get(
                        "error",
                        "Unknown error",
                    )
                )
            )

        elif insight:

            description = html.escape(
                str(
                    insight.get(
                        "description",
                        "",
                    )
                )
            )

            reaction = html.escape(
                str(
                    insight.get(
                        "main_reaction",
                        "",
                    )
                )
            )

            disagreement = html.escape(
                str(
                    insight.get(
                        "disagreement",
                        "",
                    )
                )
            )

            st.markdown(
                f"""
                <div class="ai-card">

                    <div class="ai-label">
                        GPT-OSS interpretation
                    </div>

                    <div style="
                        margin-top:.45rem;
                    ">
                        <b>What this is about:</b>
                        {description}
                    </div>

                    <div style="
                        margin-top:.55rem;
                    ">
                        <b>Main reaction and why:</b>
                        {reaction}
                    </div>

                    <div style="
                        margin-top:.55rem;
                    ">
                        <b>Disagreement:</b>
                        {disagreement}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            concerns = insight.get(
                "key_concerns",
                [],
            )

            if concerns:

                st.markdown(
                    "**Key concerns / reasons**"
                )

                for concern in concerns:

                    st.markdown(
                        f"- {html.escape(str(concern))}"
                    )

        st.divider()


# ============================================================
# TAB 3 — REVIEW
# ============================================================

with tab3:

    st.subheader(
        f"Subreddit Review · r/{SUBREDDIT}"
    )

    if (
        overall_review
        and "error"
        not in overall_review
    ):

        review_col1, review_col2 = (
            st.columns(
                [1.7, 1]
            )
        )

        with review_col1:

            st.markdown(
                '<div class="reddit-card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                "**General consensus**"
            )

            st.write(
                overall_review.get(
                    "consensus",
                    "",
                )
            )

            st.markdown(
                "**Overall sentiment**"
            )

            st.write(
                overall_review.get(
                    "overall_sentiment",
                    "",
                )
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        with review_col2:

            st.markdown(
                '<div class="reddit-card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                "**Notable / niche details**"
            )

            details = (
                overall_review.get(
                    "notable_details",
                    [],
                )
            )

            if details:

                for detail in details:

                    st.markdown(
                        f"- {html.escape(str(detail))}"
                    )

            else:

                st.caption(
                    "No additional details were generated."
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

    else:

        st.info(
            "AI review summary is unavailable. "
            "Showing the statistical sentiment breakdown instead."
        )

        if not sentiment.empty:

            st.dataframe(
                sentiment,
                hide_index=True,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # SENTIMENT SUMMARY CARD
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "**Sentiment breakdown**"
    )

    if not sentiment.empty:

        sentiment_display = sentiment.copy()

        sentiment_display[
            "percentage"
        ] = sentiment_display[
            "percentage"
        ].round(1)

        sentiment_display = (
            sentiment_display.rename(
                columns={
                    "sentiment":
                        "Sentiment",
                    "count":
                        "Posts",
                    "percentage":
                        "Percentage",
                }
            )
        )

        st.dataframe(
            sentiment_display,
            hide_index=True,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # WORD CLOUD
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "**What people are talking about**"
    )

    st.caption(
        "Common terms from the analyzed posts after removing "
        "English and Reddit-specific stopwords."
    )

    wc_fig = generate_word_cloud(
        analysis_df
    )

    if wc_fig is not None:

        st.pyplot(
            wc_fig,
            use_container_width=True,
        )

        plt.close(wc_fig)

    else:

        st.caption(
            "Not enough text to build a word cloud."
        )


# ============================================================
# TAB 4 — DATA
# ============================================================

with tab4:

    st.subheader(
        "Top Posts Dataset"
    )

    st.caption(
        "The table contains the high-engagement posts selected "
        "for NLP analysis."
    )

    display_df = (
        analysis_df.copy()
    )

    display_df["created"] = (
        pd.to_datetime(
            display_df[
                "created_utc"
            ],
            unit="s",
        )
    )

    display_df = (
        display_df
        .sort_values(
            "score",
            ascending=False,
        )
    )

    display_cols = [
        "title",
        "score",
        "num_comments",
        "sentiment",
        "topic_name",
        "created",
        "url",
    ]

    display_cols = [
        c
        for c in display_cols
        if c in display_df.columns
    ]

    st.dataframe(
        display_df[
            display_cols
        ],
        use_container_width=True,
        height=560,
        hide_index=True,
        column_config={
            "title":
                st.column_config.TextColumn(
                    "Title",
                    width="large",
                ),
            "score":
                st.column_config.NumberColumn(
                    "Score",
                    format="%d",
                ),
            "num_comments":
                st.column_config.NumberColumn(
                    "Comments",
                    format="%d",
                ),
            "sentiment":
                st.column_config.TextColumn(
                    "Sentiment",
                ),
            "topic_name":
                st.column_config.TextColumn(
                    "Topic",
                    width="medium",
                ),
            "created":
                st.column_config.DatetimeColumn(
                    "Created",
                ),
            "url":
                st.column_config.LinkColumn(
                    "Post",
                    display_text="Open ↗",
                ),
        },
    )

    csv_data = (
        display_df[
            display_cols
        ].to_csv(
            index=False
        )
    )

    st.download_button(
        "⬇️ Download table as CSV",
        data=csv_data,
        file_name=(
            f"{SUBREDDIT.lower()}"
            "_reddit_recon.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"Reddit Recon · r/{SUBREDDIT} · "
    f"{SENTIMENT_MODEL} + "
    f"{EMBEDDING_MODEL} + "
    f"K-Means + TF-IDF + "
    f"{LLM_MODEL} via Groq"
)

st.caption(
    "Results describe a high-engagement sample of fetched posts, "
    "not the entire subreddit population."
)
```
