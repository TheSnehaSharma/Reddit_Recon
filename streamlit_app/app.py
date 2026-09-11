import html
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from wordcloud import WordCloud


# ============================================================
# PAGE CONFIG
# ============================================================

# Font Awesome Reddit icon used as the browser-tab favicon.
REDDIT_ICON_URL = (
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/"
    "svgs/brands/reddit.svg"
)

st.set_page_config(
    page_title="Reddit Recon",
    page_icon="https://www.reddit.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

ARCTIC_URL = (
    "https://arctic-shift.photon-reddit.com/api/posts/search"
)

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Fixed GPT-OSS model. No model selector in the UI.
LLM_MODEL = "openai/gpt-oss-20b"

DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 3
DEFAULT_RANDOM_STATE = 42
DEFAULT_RETRIES = 4
DEFAULT_MAX_TEXT = 2500
DEFAULT_EMBED_BATCH = 32


# ============================================================
# DARK / WHITE UI
# ============================================================

st.markdown(
    """
    <link
        rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css"
    >

    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background: #111111 !important;
        color: #ffffff !important;
    }

    [data-testid="stHeader"] {
        background: #111111 !important;
    }

    [data-testid="stToolbar"] {
        background: #111111 !important;
    }

    [data-testid="stDecoration"] {
        background: #111111 !important;
    }

    * {
        color: #ffffff !important;
    }

    p,
    span,
    div,
    label,
    li,
    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #ffffff !important;
    }

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    [data-testid="stSidebar"] {
        background: #111111 !important;
        border-right: 1px solid #333333 !important;
    }

    [data-testid="stSidebarContent"] {
        background: #111111 !important;
    }

    /* --------------------------------------------------------
       INPUTS
    -------------------------------------------------------- */

    input,
    textarea,
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background: #181818 !important;
        color: #ffffff !important;
        border-color: #444444 !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #888888 !important;
    }

    [data-baseweb="select"] *,
    [data-baseweb="popover"] * {
        color: #ffffff !important;
    }

    [role="option"] {
        background: #181818 !important;
        color: #ffffff !important;
    }

    [role="option"]:hover {
        background: #292929 !important;
    }

    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    button {
        color: #ffffff !important;
    }

    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button {
        background: #181818 !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }

    [data-testid="stButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: #252525 !important;
        border-color: #777777 !important;
    }

    /* --------------------------------------------------------
       TABS
    -------------------------------------------------------- */

    [data-baseweb="tab-list"] {
        background: #111111 !important;
        border-bottom: 1px solid #333333 !important;
    }

    [data-baseweb="tab"] {
        color: #aaaaaa !important;
    }

    [aria-selected="true"] {
        color: #ffffff !important;
    }

    /* --------------------------------------------------------
       CARDS
    -------------------------------------------------------- */

    .reddit-card {
        background: #181818;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .reddit-card:hover {
        border-color: #555555;
    }

    .topic-title {
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
        color: #ffffff !important;
    }

    .reddit-header {
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 1.25rem 0 1.5rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid #333333;
    }

    .reddit-icon {
        width: 58px;
        height: 58px;
        min-width: 58px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #444444;
        border-radius: 50%;
        background: #181818;
    }

    .reddit-icon i {
        font-size: 34px !important;
        color: #ffffff !important;
    }

    .reddit-header h1 {
        margin: 0;
        padding: 0;
        font-size: 2rem;
        line-height: 1.15;
        color: #ffffff !important;
    }

    .reddit-header p {
        margin: 0.35rem 0 0 0;
        color: #aaaaaa !important;
        font-size: 0.95rem;
    }

    /* --------------------------------------------------------
       BADGES
    -------------------------------------------------------- */

    .badge {
        display: inline-block;
        padding: 4px 9px;
        margin-right: 6px;
        margin-bottom: 5px;
        border: 1px solid #555555;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        background: #222222;
        color: #ffffff !important;
    }

    /* --------------------------------------------------------
       POSTS
    -------------------------------------------------------- */

    .post-item {
        padding: 0.75rem 0;
        border-bottom: 1px solid #333333;
    }

    .post-item:last-child {
        border-bottom: none;
    }

    .post-title {
        font-weight: 700;
        line-height: 1.4;
    }

    .post-meta {
        color: #999999 !important;
        font-size: 0.82rem;
        margin-top: 4px;
    }

    .post-link {
        color: #ffffff !important;
        text-decoration: none !important;
    }

    .post-link:hover {
        text-decoration: underline !important;
    }

    /* --------------------------------------------------------
       METRICS
    -------------------------------------------------------- */

    .metric-card {
        background: #181818;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
    }

    .metric-label {
        color: #999999 !important;
        font-size: 0.8rem;
        margin-top: 3px;
    }

    /* --------------------------------------------------------
       DIVIDERS / DATAFRAME
    -------------------------------------------------------- */

    hr {
        border-color: #333333 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #333333 !important;
    }

    /* --------------------------------------------------------
       EXPANDERS
    -------------------------------------------------------- */

    [data-testid="stExpander"] {
        background: #181818 !important;
        border: 1px solid #333333 !important;
        border-radius: 10px !important;
    }

    /* --------------------------------------------------------
       ALERTS
    -------------------------------------------------------- */

    [data-testid="stAlert"] {
        background: #181818 !important;
        border: 1px solid #444444 !important;
    }

    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "selected_subreddit" not in st.session_state:
    st.session_state["selected_subreddit"] = None

if "result" not in st.session_state:
    st.session_state["result"] = None


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource(show_spinner=False)
def get_http_session():
    session = requests.Session()

    retry = Retry(
        total=DEFAULT_RETRIES,
        connect=DEFAULT_RETRIES,
        read=DEFAULT_RETRIES,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=10,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "RedditRecon/1.0 "
                "(Streamlit Reddit analysis application)"
            )
        }
    )

    return session


# ============================================================
# HELPERS
# ============================================================

def normalize_subreddit(value):
    value = str(value).strip()

    if value.startswith("https://www.reddit.com/r/"):
        value = value.split("/r/", 1)[1]

    if value.startswith("https://reddit.com/r/"):
        value = value.split("/r/", 1)[1]

    value = value.strip("/")

    if value.lower().startswith("r/"):
        value = value[2:]

    return value.strip()


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clean_text(value):
    if value is None:
        return ""

    text = str(value)

    if text.lower() in {
        "[removed]",
        "[deleted]",
        "nan",
        "none",
    }:
        return ""

    return text.strip()


def truncate_text(text, max_chars=DEFAULT_MAX_TEXT):
    text = clean_text(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def json_safe(value):
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return None

    return value


# ============================================================
# REDDIT FETCH
# ============================================================

def fetch_reddit_posts(
    subreddit,
    days_back=5,
    posts_to_fetch=100,
):
    subreddit = normalize_subreddit(subreddit)

    if not subreddit:
        raise ValueError("Please enter a subreddit.")

    session = get_http_session()

    before = datetime.now(timezone.utc)
    after = before - timedelta(days=int(days_back))

    params = {
        "subreddit": subreddit,
        "after": after.isoformat(),
        "before": before.isoformat(),
        "sort": "desc",
        "limit": min(int(posts_to_fetch), 100),
    }

    response = session.get(
        ARCTIC_URL,
        params=params,
        timeout=45,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Arctic Shift returned HTTP "
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    payload = response.json()

    if isinstance(payload, dict):
        rows = payload.get("data", [])

        if rows is None:
            rows = []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    return df


# ============================================================
# CLEAN REDDIT DATA
# ============================================================

def prepare_posts(df):
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()

    required_defaults = {
        "id": "",
        "title": "",
        "selftext": "",
        "score": 0,
        "num_comments": 0,
        "created_utc": None,
        "author": "[deleted]",
        "url": "",
        "subreddit": "",
    }

    for column, default in required_defaults.items():
        if column not in work.columns:
            work[column] = default

    work["title"] = work["title"].fillna("").astype(str)
    work["selftext"] = work["selftext"].fillna("").astype(str)

    work["text"] = (
        work["title"].str.strip()
        + "\n\n"
        + work["selftext"].str.strip()
    )

    work["text"] = work["text"].apply(
        lambda x: truncate_text(x, DEFAULT_MAX_TEXT)
    )

    work = work[work["title"].str.strip() != ""]

    work = work[
        work["text"].str.len() > 10
    ].copy()

    work["score"] = pd.to_numeric(
        work["score"],
        errors="coerce",
    ).fillna(0)

    work["num_comments"] = pd.to_numeric(
        work["num_comments"],
        errors="coerce",
    ).fillna(0)

    work["engagement"] = (
        work["score"].clip(lower=0)
        + work["num_comments"].clip(lower=0) * 2
    )

    if "created_utc" in work.columns:
        numeric_created = pd.to_numeric(
            work["created_utc"],
            errors="coerce",
        )

        if numeric_created.notna().any():
            work["created"] = pd.to_datetime(
                numeric_created,
                unit="s",
                errors="coerce",
                utc=True,
            )

    if "created" not in work.columns:
        work["created"] = pd.NaT

    work = work.sort_values(
        "engagement",
        ascending=False,
    )

    work = work.drop_duplicates(
        subset=["id"],
        keep="first",
    )

    return work.reset_index(drop=True)


# ============================================================
# SENTIMENT
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_MODEL
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL
    )

    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
    )


def normalize_sentiment(label):
    label = str(label).lower()

    if "positive" in label:
        return "Positive"

    if "negative" in label:
        return "Negative"

    return "Neutral"


def analyze_sentiment(texts):
    if not texts:
        return []

    classifier = load_sentiment_pipeline()

    results = []

    for start in range(
        0,
        len(texts),
        DEFAULT_BATCH_SIZE,
    ):
        batch = texts[
            start:start + DEFAULT_BATCH_SIZE
        ]

        output = classifier(batch)

        for item in output:
            results.append(
                {
                    "label": normalize_sentiment(
                        item.get("label", "Neutral")
                    ),
                    "score": safe_float(
                        item.get("score", 0)
                    ),
                }
            )

    return results


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL
    )

    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL
    )

    model.eval()

    return tokenizer, model


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    return torch.sum(
        token_embeddings * input_mask_expanded,
        dim=1,
    ) / torch.clamp(
        input_mask_expanded.sum(dim=1),
        min=1e-9,
    )


def create_embeddings(texts):
    if not texts:
        return np.empty((0, 384))

    tokenizer, model = load_embedding_model()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = model.to(device)

    all_embeddings = []

    with torch.no_grad():
        for start in range(
            0,
            len(texts),
            DEFAULT_EMBED_BATCH,
        ):
            batch = texts[
                start:start + DEFAULT_EMBED_BATCH
            ]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            output = model(**encoded)

            embeddings = mean_pooling(
                output,
                encoded["attention_mask"],
            )

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1,
            )

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

    return np.vstack(all_embeddings)


# ============================================================
# TOPIC DISCOVERY
# ============================================================

def discover_topics(
    embeddings,
    min_k=2,
    max_k=8,
):
    n = len(embeddings)

    if n < 4:
        labels = np.zeros(n, dtype=int)
        return labels, 1

    upper = min(
        max_k,
        n - 1,
    )

    if upper < min_k:
        min_k = 2

    best_score = -1
    best_k = 1
    best_labels = np.zeros(n, dtype=int)

    for k in range(
        min_k,
        upper + 1,
    ):
        try:
            model = KMeans(
                n_clusters=k,
                random_state=DEFAULT_RANDOM_STATE,
                n_init=10,
            )

            labels = model.fit_predict(
                embeddings
            )

            if len(set(labels)) < 2:
                continue

            score = silhouette_score(
                embeddings,
                labels,
            )

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        except Exception:
            continue

    if best_k == 1:
        model = KMeans(
            n_clusters=2,
            random_state=DEFAULT_RANDOM_STATE,
            n_init=10,
        )

        best_labels = model.fit_predict(
            embeddings
        )

        best_k = 2

    return best_labels, best_k


# ============================================================
# TOPIC KEYWORDS
# ============================================================

def extract_topic_keywords(
    texts,
    labels,
    top_n=8,
):
    if len(texts) == 0:
        return {}

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
    )

    matrix = vectorizer.fit_transform(texts)

    terms = np.array(
        vectorizer.get_feature_names_out()
    )

    keywords = {}

    for topic_id in sorted(set(labels)):
        indices = np.where(
            labels == topic_id
        )[0]

        if len(indices) == 0:
            keywords[int(topic_id)] = []
            continue

        topic_scores = np.asarray(
            matrix[indices].mean(axis=0)
        ).ravel()

        order = topic_scores.argsort()[::-1]

        selected = []

        for idx in order:
            term = terms[idx]

            if term not in selected:
                selected.append(term)

            if len(selected) >= top_n:
                break

        keywords[int(topic_id)] = selected

    return keywords


# ============================================================
# TOPIC EVIDENCE
# ============================================================

def build_topic_evidence(
    df,
    labels,
    keywords,
    max_posts=5,
):
    evidence = {}

    work = df.copy()
    work["topic"] = labels

    for topic_id in sorted(
        work["topic"].unique()
    ):
        topic_df = work[
            work["topic"] == topic_id
        ].sort_values(
            "engagement",
            ascending=False,
        )

        posts = []

        for _, row in topic_df.head(
            max_posts
        ).iterrows():

            posts.append(
                {
                    "title": clean_text(
                        row.get("title", "")
                    ),
                    "text": truncate_text(
                        row.get("text", ""),
                        700,
                    ),
                    "score": safe_float(
                        row.get("score", 0)
                    ),
                    "comments": safe_float(
                        row.get(
                            "num_comments",
                            0,
                        )
                    ),
                    "url": clean_text(
                        row.get("url", "")
                    ),
                }
            )

        evidence[int(topic_id)] = {
            "keywords": keywords.get(
                int(topic_id),
                [],
            ),
            "posts": posts,
        }

    return evidence


# ============================================================
# GROQ
# ============================================================

def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def groq_request(
    messages,
    response_schema,
    temperature=0.2,
):
    api_key = get_groq_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing from Streamlit secrets."
        )

    endpoint = (
        "https://api.groq.com/openai/v1/chat/completions"
    )

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema["name"],
                "strict": True,
                "schema": response_schema["schema"],
            },
        },
    }

    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Groq API error "
            f"{response.status_code}: "
            f"{response.text[:1000]}"
        )

    data = response.json()

    content = (
        data["choices"][0]["message"]
        .get("content", "")
    )

    if not content:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    return json.loads(content)


# ============================================================
# AI TOPIC ANALYSIS
# ============================================================

TOPIC_SCHEMA = {
    "name": "reddit_topic_analysis",
    "schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "description": {
                "type": "string"
            },
            "reaction": {
                "type": "string"
            },
        },
        "required": [
            "name",
            "description",
            "reaction",
        ],
        "additionalProperties": False,
    },
}


def analyze_topic_with_llm(
    topic_id,
    topic_data,
):
    keywords = topic_data.get(
        "keywords",
        [],
    )

    posts = topic_data.get(
        "posts",
        [],
    )

    evidence_text = []

    for post in posts:
        evidence_text.append(
            "\n".join(
                [
                    f"Title: {post.get('title', '')}",
                    f"Text: {post.get('text', '')}",
                    f"Score: {post.get('score', 0)}",
                    (
                        "Comments: "
                        f"{post.get('comments', 0)}"
                    ),
                ]
            )
        )

    prompt = f"""
You are analyzing a Reddit discussion topic.

Topic cluster ID:
{topic_id}

Extracted keywords:
{", ".join(keywords)}

Representative Reddit posts:
{chr(10).join(evidence_text)}

Give this topic:
1. A concise human-readable name.
2. A factual description of what users are discussing.
3. A concise description of the reaction/context visible in the posts.

Do not invent facts.
Do not claim something is confirmed unless the evidence says so.
Keep the output concise.
"""

    return groq_request(
        [
            {
                "role": "system",
                "content": (
                    "You are a precise Reddit research analyst."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        TOPIC_SCHEMA,
    )


# ============================================================
# OVERALL AI REVIEW
# ============================================================

OVERALL_SCHEMA = {
    "name": "reddit_overall_review",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string"
            },
            "dominant_topics": {
                "type": "string"
            },
            "sentiment": {
                "type": "string"
            },
            "notable_signals": {
                "type": "string"
            },
        },
        "required": [
            "summary",
            "dominant_topics",
            "sentiment",
            "notable_signals",
        ],
        "additionalProperties": False,
    },
}


def generate_overall_review(
    analysis_df,
    topic_analysis,
    sentiment_summary,
):
    topic_lines = []

    for topic_id, info in topic_analysis.items():
        topic_lines.append(
            f"""
Topic {topic_id}:
Name: {info.get('name', '')}
Description: {info.get('description', '')}
Reaction: {info.get('reaction', '')}
"""
        )

    prompt = f"""
You are reviewing a Reddit community.

Number of analyzed posts:
{len(analysis_df)}

Sentiment distribution:
{json.dumps(sentiment_summary)}

Topic analysis:
{''.join(topic_lines)}

Produce a concise overall Reddit community review.

Focus on:
- what the community is discussing
- dominant topics
- overall sentiment
- meaningful signals or patterns

Do not invent information.
"""

    return groq_request(
        [
            {
                "role": "system",
                "content": (
                    "You are a rigorous Reddit intelligence analyst."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        OVERALL_SCHEMA,
    )


# ============================================================
# WORD CLOUD
# ============================================================

def create_wordcloud(text):
    if not text.strip():
        return None

    wordcloud = WordCloud(
        width=1200,
        height=500,
        background_color="#111111",
        color_func=lambda *args, **kwargs: "white",
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    ax.imshow(
        wordcloud,
        interpolation="bilinear",
    )

    ax.axis("off")

    return fig


# ============================================================
# COMPLETE NLP PIPELINE
# ============================================================

def run_nlp_pipeline(
    raw_df,
    top_posts,
):
    clean_df = prepare_posts(raw_df)

    if clean_df.empty:
        raise ValueError(
            "No usable Reddit posts were found "
            "for the selected time range."
        )

    analysis_df = clean_df.head(
        int(top_posts)
    ).copy()

    texts = analysis_df[
        "text"
    ].tolist()

    # --------------------------------------------------------
    # Sentiment
    # --------------------------------------------------------

    sentiment_results = analyze_sentiment(
        texts
    )

    analysis_df["sentiment"] = [
        item["label"]
        for item in sentiment_results
    ]

    analysis_df["sentiment_score"] = [
        item["score"]
        for item in sentiment_results
    ]

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        texts
    )

    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------

    labels, best_k = discover_topics(
        embeddings
    )

    analysis_df["topic"] = labels

    # --------------------------------------------------------
    # Keywords
    # --------------------------------------------------------

    keywords = extract_topic_keywords(
        texts,
        labels,
    )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence = build_topic_evidence(
        analysis_df,
        labels,
        keywords,
    )

    # --------------------------------------------------------
    # Engagement
    # --------------------------------------------------------

    top_engaged = (
        analysis_df.sort_values(
            "engagement",
            ascending=False,
        )
        .head(10)
        .copy()
    )

    # --------------------------------------------------------
    # AI topic analysis
    # --------------------------------------------------------

    ai_insights = {}

    groq_available = bool(
        get_groq_key()
    )

    if groq_available:
        progress = st.progress(
            0,
            text="Generating AI topic analysis...",
        )

        topic_ids = sorted(
            evidence.keys()
        )

        completed = 0

        for topic_id in topic_ids:
            try:
                ai_insights[topic_id] = (
                    analyze_topic_with_llm(
                        topic_id,
                        evidence[topic_id],
                    )
                )
            except Exception as exc:
                ai_insights[topic_id] = {
                    "name": (
                        "Topic "
                        f"{topic_id + 1}"
                    ),
                    "description": (
                        "AI analysis failed for "
                        "this topic."
                    ),
                    "reaction": str(exc),
                    "error": True,
                }

            completed += 1

            progress.progress(
                completed / max(
                    len(topic_ids),
                    1,
                ),
                text=(
                    "Generating AI topic analysis "
                    f"({completed}/{len(topic_ids)})"
                ),
            )

        progress.empty()

    else:
        for topic_id, topic_info in evidence.items():
            ai_insights[topic_id] = {
                "name": (
                    " • ".join(
                        topic_info.get(
                            "keywords",
                            [],
                        )[:3]
                    )
                    or f"Topic {topic_id + 1}"
                ),
                "description": (
                    "Add GROQ_API_KEY to enable "
                    "GPT-OSS topic analysis."
                ),
                "reaction": (
                    "AI analysis is disabled."
                ),
                "error": True,
            }

    # --------------------------------------------------------
    # Sentiment summary
    # --------------------------------------------------------

    sentiment_summary = (
        analysis_df["sentiment"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Overall review
    # --------------------------------------------------------

    overall_review = None

    if groq_available:
        try:
            overall_review = (
                generate_overall_review(
                    analysis_df,
                    ai_insights,
                    sentiment_summary,
                )
            )
        except Exception as exc:
            overall_review = {
                "summary": (
                    "Overall AI review failed."
                ),
                "dominant_topics": "",
                "sentiment": "",
                "notable_signals": str(exc),
            }

    return {
        "clean": clean_df,
        "analysis": analysis_df,
        "best_k": best_k,
        "keywords": keywords,
        "evidence": evidence,
        "top_engaged": top_engaged,
        "sentiment": sentiment_summary,
        "ai_insights": ai_insights,
        "overall_review": overall_review,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div style="
            display:flex;
            align-items:center;
            gap:12px;
            margin-bottom:1rem;
        ">
            <i
                class="fa-brands fa-reddit"
                style="font-size:34px;"
            ></i>

            <div style="
                font-size:1.25rem;
                font-weight:800;
            ">
                Reddit Recon
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.html(
        """
        <div style="
            color:#999999 !important;
            font-size:0.85rem;
            margin-bottom:1.25rem;
        ">
            Reddit community intelligence and
            topic analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("subreddit_form"):

        subreddit_input = st.text_input(
            "Subreddit",
            value=(
                st.session_state.get(
                    "selected_subreddit"
                )
                or ""
            ),
            placeholder="e.g. GTA6",
        )

        days_back = st.number_input(
            "Days back",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
        )

        posts_to_fetch = st.number_input(
            "Posts to fetch",
            min_value=10,
            max_value=100,
            value=100,
            step=10,
        )

        top_posts = st.number_input(
            "Posts to analyze",
            min_value=10,
            max_value=100,
            value=100,
            step=10,
        )

        submitted = st.form_submit_button(
            "Run Recon",
            use_container_width=True,
        )

    st.markdown(
        "<hr>",
        unsafe_allow_html=True,
    )

    st.html(
        f"""
        <div style="
            font-size:0.8rem;
            color:#888888 !important;
            line-height:1.6;
        ">
            <b>AI:</b> {html.escape(LLM_MODEL)}<br>
            <b>Embeddings:</b> MiniLM<br>
            <b>Sentiment:</b> CardiffNLP RoBERTa
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RUN RECON
# ============================================================

if submitted:

    subreddit = normalize_subreddit(
        subreddit_input
    )

    if not subreddit:
        st.error(
            "Enter a subreddit first."
        )
        st.stop()

    st.session_state[
        "selected_subreddit"
    ] = subreddit

    # Important: clear stale results.
    st.session_state["result"] = None

    try:

        with st.status(
            "Running Reddit Recon...",
            expanded=True,
        ) as status:

            st.write(
                f"Fetching r/{subreddit}..."
            )

            raw_df = fetch_reddit_posts(
                subreddit=subreddit,
                days_back=days_back,
                posts_to_fetch=posts_to_fetch,
            )

            if raw_df.empty:
                raise ValueError(
                    "No Reddit posts were returned. "
                    "Try a larger time range or "
                    "another subreddit."
                )

            st.write(
                f"Fetched {len(raw_df):,} posts."
            )

            st.write(
                "Running sentiment, embeddings, "
                "topic discovery, and AI analysis..."
            )

            result = run_nlp_pipeline(
                raw_df,
                top_posts,
            )

            result["raw"] = raw_df

            st.session_state[
                "result"
            ] = result

            status.update(
                label="Recon complete.",
                state="complete",
                expanded=False,
            )

    except Exception as exc:

        st.error(
            "Recon failed."
        )

        st.exception(exc)

        st.stop()


# ============================================================
# NO RESULT YET
# ============================================================

result = st.session_state.get(
    "result"
)

selected_subreddit = (
    st.session_state.get(
        "selected_subreddit"
    )
)

if result is None:

    st.markdown(
        """
        <div style="
            min-height:60vh;
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
        ">
            <div>

                <div class="reddit-icon"
                     style="
                        margin:0 auto 1.25rem auto;
                     ">
                    <i class="fa-brands fa-reddit"></i>
                </div>

                <h1 style="
                    font-size:2.4rem;
                    margin-bottom:0.5rem;
                ">
                    Reddit Recon
                </h1>

                <p style="
                    color:#999999 !important;
                    font-size:1rem;
                ">
                    Enter a subreddit and click
                    <b>Run Recon</b> to begin.
                </p>

            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# LOAD RESULT
# ============================================================

raw_df = result.get(
    "raw",
    pd.DataFrame(),
)

analysis_df = result.get(
    "analysis",
    pd.DataFrame(),
)

best_k = result.get(
    "best_k",
    1,
)

keywords = result.get(
    "keywords",
    {},
)

evidence = result.get(
    "evidence",
    {},
)

top_engaged = result.get(
    "top_engaged",
    pd.DataFrame(),
)

sentiment_summary = result.get(
    "sentiment",
    {},
)

ai_insights = result.get(
    "ai_insights",
    {},
)

overall_review = result.get(
    "overall_review"
)


# ============================================================
# MAIN HEADER
# ============================================================

st.html(
    f"""
    <div class="reddit-header">

        <div class="reddit-icon">
            <i class="fa-brands fa-reddit"></i>
        </div>

        <div>
            <h1>
                Reddit Recon:
                r/{html.escape(
                    str(selected_subreddit)
                )}
            </h1>

            <p>
                Analyzed top
                {len(analysis_df):,}
                posts from the last
                {days_back if 'days_back' in locals() else '?'}
                day(s).
            </p>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Overview",
        "Topics",
        "Review",
        "Data",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    total_posts = len(
        analysis_df
    )

    positive = sentiment_summary.get(
        "Positive",
        0,
    )

    negative = sentiment_summary.get(
        "Negative",
        0,
    )

    neutral = sentiment_summary.get(
        "Neutral",
        0,
    )

    avg_score = (
        analysis_df["score"].mean()
        if "score" in analysis_df.columns
        else 0
    )

    avg_comments = (
        analysis_df["num_comments"].mean()
        if "num_comments"
        in analysis_df.columns
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {total_posts:,}
                </div>
                <div class="metric-label">
                    Posts analyzed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.html(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {best_k}
                    </div>
                    <div class="metric-label">
                        Topics discovered
                    </div>
                </div>
                """,
            unsafe_allow_html=True,
        )

    with col3:
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {avg_score:,.1f}
                </div>
                <div class="metric-label">
                    Average score
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {avg_comments:,.1f}
                </div>
                <div class="metric-label">
                    Avg. comments
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)

    with col_left:

        st.markdown(
            "### Sentiment"
        )

        sentiment_df = pd.DataFrame(
            {
                "Sentiment": [
                    "Positive",
                    "Neutral",
                    "Negative",
                ],
                "Posts": [
                    positive,
                    neutral,
                    negative,
                ],
            }
        )

        fig = px.bar(
            sentiment_df,
            x="Sentiment",
            y="Posts",
            template="plotly_dark",
        )

        fig.update_layout(
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font_color="white",
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col_right:

        st.markdown(
            "### Topic Distribution"
        )

        topic_counts = (
            analysis_df[
                "topic"
            ]
            .value_counts()
            .sort_index()
        )

        topic_df = pd.DataFrame(
            {
                "Topic": [
                    f"Topic {x + 1}"
                    for x in topic_counts.index
                ],
                "Posts": topic_counts.values,
            }
        )

        fig = px.bar(
            topic_df,
            x="Topic",
            y="Posts",
            template="plotly_dark",
        )

        fig.update_layout(
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font_color="white",
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        "### Word Cloud"
    )

    all_text = " ".join(
        analysis_df["text"].astype(str)
    )

    wordcloud_fig = create_wordcloud(
        all_text
    )

    if wordcloud_fig is not None:
        st.pyplot(
            wordcloud_fig,
            use_container_width=True,
        )
        plt.close(
            wordcloud_fig
        )


# ============================================================
# TOPICS
# ============================================================

with tabs[1]:

    st.markdown(
        "## Topic Analysis"
    )

    st.caption(
        "Topics are discovered using MiniLM embeddings "
        "and K-Means clustering."
    )

    if not evidence:
        st.info(
            "No topics were available."
        )

    for topic_id in sorted(
        evidence.keys()
    ):

        topic_info = evidence[
            topic_id
        ]

        ai_info = ai_insights.get(
            topic_id,
            {},
        )

        ai_name = ai_info.get(
            "name",
            f"Topic {topic_id + 1}",
        )

        ai_desc = ai_info.get(
            "description",
            "",
        )

        ai_reaction = ai_info.get(
            "reaction",
            "",
        )

        has_error = ai_info.get(
            "error",
            False,
        )

        # ----------------------------------------------------
        # Sentiment for topic
        # ----------------------------------------------------

        topic_df = analysis_df[
            analysis_df["topic"]
            == topic_id
        ]

        topic_sentiment = (
            topic_df["sentiment"]
            .value_counts()
            .to_dict()
        )

        badges = ""

        for label in [
            "Positive",
            "Neutral",
            "Negative",
        ]:
            count = topic_sentiment.get(
                label,
                0,
            )

            if count:
                badges += (
                    f'<span class="badge">'
                    f'{html.escape(label)}: '
                    f'{count}'
                    f'</span>'
                )

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        error_html = ""

        if has_error:
            error_html = f"""
            <div style="
                margin:1rem 0;
                padding:0.75rem;
                border:1px solid #555555;
                border-radius:8px;
                color:#aaaaaa !important;
            ">
                {html.escape(
                    str(ai_reaction)
                )}
            </div>
            """

        # ----------------------------------------------------
        # Representative posts
        # ----------------------------------------------------

        rep_html = ""

        for post in topic_info.get(
            "posts",
            [],
        ):

            title = html.escape(
                str(
                    post.get(
                        "title",
                        "Untitled",
                    )
                )
            )

            url = html.escape(
                str(
                    post.get(
                        "url",
                        "",
                    )
                ),
                quote=True,
            )

            score = safe_float(
                post.get(
                    "score",
                    0,
                )
            )

            comments = safe_float(
                post.get(
                    "comments",
                    0,
                )
            )

            if url:
                title_html = (
                    f'<a class="post-link" '
                    f'href="{url}" '
                    f'target="_blank">'
                    f'{title}'
                    f'</a>'
                )
            else:
                title_html = title

            rep_html += f"""
            <li class="post-item">
                <div class="post-title">
                    {title_html}
                </div>

                <div class="post-meta">
                    Score: {score:,.0f}
                    &nbsp; • &nbsp;
                    Comments: {comments:,.0f}
                </div>
            </li>
            """

        # ----------------------------------------------------
        # TOPIC CARD
        #
        # IMPORTANT:
        # This is intentionally st.markdown with
        # unsafe_allow_html=True.
        # Do NOT replace this with st.write(),
        # st.text(), or st.code().
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="reddit-card">

                <div class="topic-title">
                    {html.escape(
                        str(ai_name)
                    )}
                </div>

                <div style="
                    margin-bottom:1rem;
                ">
                    {badges}
                </div>

                <p>
                    <b>Analysis:</b>
                    {html.escape(
                        str(ai_desc)
                    )}
                </p>

                <p>
                    <b>Reaction Context:</b>
                    {html.escape(
                        str(ai_reaction)
                    )}
                </p>

                {error_html}

                <hr/>

                <p>
                    <b>TOP POSTS IN TOPIC</b>
                </p>

                <ul style="
                    list-style-type:none;
                    padding-left:0;
                ">
                    {rep_html}
                </ul>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# REVIEW
# ============================================================

with tabs[2]:

    st.markdown(
        "## Overall Reddit Review"
    )

    if overall_review:

        summary = overall_review.get(
            "summary",
            "",
        )

        dominant_topics = (
            overall_review.get(
                "dominant_topics",
                "",
            )
        )

        sentiment_review = (
            overall_review.get(
                "sentiment",
                "",
            )
        )

        notable_signals = (
            overall_review.get(
                "notable_signals",
                "",
            )
        )

        st.markdown(
            f"""
            <div class="reddit-card">

                <div class="topic-title">
                    Summary
                </div>

                <p>
                    {html.escape(
                        str(summary)
                    )}
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class="reddit-card">

                    <div class="topic-title">
                        Dominant Topics
                    </div>

                    <p>
                        {html.escape(
                            str(
                                dominant_topics
                            )
                        )}
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
                <div class="reddit-card">

                    <div class="topic-title">
                        Sentiment
                    </div>

                    <p>
                        {html.escape(
                            str(
                                sentiment_review
                            )
                        )}
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="reddit-card">

                <div class="topic-title">
                    Notable Signals
                </div>

                <p>
                    {html.escape(
                        str(
                            notable_signals
                        )
                    )}
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.info(
            "Overall AI review is unavailable. "
            "Make sure GROQ_API_KEY is configured."
        )


# ============================================================
# DATA
# ============================================================

with tabs[3]:

    st.markdown(
        "## Reddit Data"
    )

    if not analysis_df.empty:

        display_columns = [
            column
            for column in [
                "id",
                "title",
                "score",
                "num_comments",
                "engagement",
                "sentiment",
                "sentiment_score",
                "topic",
                "created",
                "url",
            ]
            if column in analysis_df.columns
        ]

        st.dataframe(
            analysis_df[
                display_columns
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No analysis data available."
        )

    st.markdown(
        "### Top Engaged Posts"
    )

    if not top_engaged.empty:

        columns = [
            column
            for column in [
                "title",
                "score",
                "num_comments",
                "engagement",
                "sentiment",
                "url",
            ]
            if column in top_engaged.columns
        ]

        st.dataframe(
            top_engaged[
                columns
            ],
            use_container_width=True,
            hide_index=True,
        )
