import html
import json
import time
import re
from contextlib import contextmanager
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


# ---------------------------------------------------------------- CONFIG ---

ARCTIC_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 3
RANDOM_STATE = 42
MAX_FETCH_RETRIES = 4
MAX_TEXT_CHARS = 2500
EMBEDDING_BATCH_SIZE = 32

ROLLING_WINDOW = "1D"

BG = "#111111"
CARD_BG = "#151515"
BORDER = "#2a2a2a"

TOPIC_PALETTE = [
    "#ff6b6b", "#4dabf7", "#51cf66", "#fcc419", "#cc5de8",
    "#20c997", "#ff922b", "#845ef7", "#22b8cf", "#f06595"
]

SENTIMENT_COLORS = {
    "Positive": "#2ecc71",
    "Neutral": "#f1c40f",
    "Negative": "#e74c3c"
}

EMOTION_LABELS = [
    "anger", "disgust", "fear", "joy",
    "neutral", "sadness", "surprise"
]

EXTRA_STOPWORDS = {
    "reddit", "post", "posts", "people", "really", "just", "like",
    "think", "thing", "things", "want", "got", "get", "going",
    "does", "did", "said", "say", "know", "use", "used", "using"
}

st.set_page_config(
    page_title="Reddit Recon",
    page_icon="https://www.reddit.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.html(f"""
<style>
.stApp, .main {{ background: {BG}; }}
html, body, [class*="css"], p, span, div, label, li, td, th {{ color: #fff; }}
h1,h2,h3,h4,h5,h6 {{ color: #fff !important; }}

section[data-testid="stSidebar"] {{
    background: {BG};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: #fff !important; }}

.reddit-header {{
    display:flex; align-items:center; padding:1.2rem 0;
    margin-bottom:1.5rem; border-bottom:1px solid {BORDER};
}}
.reddit-header .reddit-icon {{
    font-size:58px; margin-right:20px; width:58px; flex-shrink:0;
}}
.reddit-header h1 {{ margin:0; font-size:2.2rem; font-weight:700; }}
.reddit-header p {{ margin:5px 0 0; font-size:.95rem; }}

.reddit-card {{
    background:{CARD_BG}; border:1px solid {BORDER};
    border-radius:10px; padding:1.25rem; margin-bottom:1rem;
}}
.reddit-card h2 {{ margin-top:0; font-size:2rem; }}

.topic-title {{
    font-weight:700; font-size:1.35rem; margin-bottom:.75rem;
}}
.topic-title::before {{ content:"\\25CF"; margin-right:10px; color:#ff4500; }}

.badge {{
    display:inline-block; padding:.25rem .7rem; border-radius:999px;
    font-size:.75rem; font-weight:700; margin:0 .5rem .4rem 0;
    border:1px solid #444; background:#1c1c1c;
}}

a, a:hover {{ color:#fff !important; }}
a:hover {{ text-decoration:underline; }}
input, textarea, select, button {{
    color:#fff !important; background-color:#181818 !important;
}}
hr {{ border-color:{BORDER} !important; }}

[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; }}
[data-testid="stMetric"] {{
    background:{CARD_BG}; border:1px solid {BORDER};
    border-radius:10px; padding:1rem;
}}
[data-testid="stMetricValue"] {{
    color:#fff !important; font-size:2rem; font-weight:700;
}}
[data-testid="stMetricLabel"] {{ color:#bdbdbd !important; }}

div[data-baseweb="select"] *, div[data-baseweb="input"] * {{
    color:#fff !important; background-color:#181818 !important;
}}
.js-plotly-plot {{ border-radius:8px; }}

.reddit-sidebar-icon {{
    font-size:46px; text-align:center; margin:10px 0 20px;
}}
.section-label {{
    color:#aaa !important; font-size:.78rem; font-weight:600;
    text-transform:uppercase; letter-spacing:.08em;
}}
.top-post-link {{ color:#fff !important; text-decoration:none; }}
.top-post-link:hover {{ text-decoration:underline; }}
</style>
""")


# --------------------------------------------------------------- HELPERS ---

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


def sentiment_badge(label, pct=None):
    text = label.capitalize() if pct is None else f"{label.capitalize()} {pct:.0f}%"
    return f'<span class="badge">{html.escape(text)}</span>'


def normalize_subreddit(value):
    value = value.strip()
    if value.lower().startswith("r/"):
        value = value[2:]
    return value.strip().replace(" ", "")


@contextmanager
def card():
    st.html('<div class="reddit-card">')
    yield
    st.html("</div>")


def style_fig(fig, **overrides):
    layout = dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color="white"),
        margin=dict(t=40, b=40, l=40, r=20),
        xaxis=dict(gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER)
    )
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


def topic_color_map(names):
    return {
        name: TOPIC_PALETTE[i % len(TOPIC_PALETTE)]
        for i, name in enumerate(names)
    }


def build_topic_stats(analysis_df, name_map):
    stats = (
        analysis_df.groupby("topic_id")
        .agg(
            posts=("id", "count"),
            avg_score=("score", "mean"),
            avg_comments=("num_comments", "mean"),
            total_engagement=("engagement", "sum"),
            avg_engagement=("engagement", "mean")
        )
        .reset_index()
    )

    stats["Topic Name"] = stats["topic_id"].map(name_map).fillna(
        stats["topic_id"].apply(lambda x: f"Topic {x}")
    )
    return stats


# ------------------------------------------------------------------ HTTP ---

@st.cache_resource
def get_http_session():
    retry = Retry(
        total=MAX_FETCH_RETRIES,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True
    )

    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    )
    session.headers.update({"User-Agent": "reddit-recon-ui/3.0"})
    return session


# ---------------------------------------------------------------- MODELS ---

@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        truncation=True,
        max_length=256,
        device=-1
    )


@st.cache_resource(show_spinner="Loading emotion model...")
def load_emotion_model():
    return pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        tokenizer=EMOTION_MODEL,
        truncation=True,
        max_length=256,
        top_k=None,
        device=-1
    )


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    model.eval()
    return tokenizer, model


# --------------------------------------------------------- FETCH REDDIT ---

@st.cache_data(ttl=900, show_spinner="Fetching Reddit Data")
def fetch_reddit_posts(subreddit, posts_to_fetch, days_back):
    now = int(time.time())
    cutoff = now - days_back * 24 * 60 * 60
    before = now
    page_size = 100
    max_pages = int(np.ceil(posts_to_fetch / page_size))
    session = get_http_session()

    fields = (
        "id,created_utc,score,num_comments,subreddit,"
        "title,selftext,url"
    )

    posts, seen_ids = [], set()

    for _ in range(max_pages):
        params = {
            "subreddit": subreddit,
            "after": cutoff,
            "before": before,
            "limit": page_size,
            "sort": "desc",
            "over_18": "false",
            "fields": fields
        }

        try:
            response = session.get(ARCTIC_URL, params=params, timeout=45)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            st.warning(f"Reddit data request failed: {exc}")
            break
        except ValueError:
            st.warning("Reddit API returned invalid JSON.")
            break

        batch = payload.get("data", [])
        if not batch:
            break

        oldest_timestamp = None

        for item in batch:
            post_id = item.get("id")
            if not post_id or post_id in seen_ids:
                continue

            created = safe_int(item.get("created_utc", 0))
            if created < cutoff:
                continue

            seen_ids.add(post_id)
            posts.append({
                "id": post_id,
                "title": str(item.get("title") or ""),
                "selftext": str(item.get("selftext") or ""),
                "score": safe_int(item.get("score", 0)),
                "num_comments": safe_int(item.get("num_comments", 0)),
                "created_utc": created,
                "url": str(item.get("url") or ""),
                "subreddit": str(item.get("subreddit") or subreddit)
            })

            oldest_timestamp = (
                created if oldest_timestamp is None
                else min(oldest_timestamp, created)
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

    return pd.DataFrame(posts).drop_duplicates("id").reset_index(drop=True)


# -------------------------------------------------------------- CLEANING ---

def clean_posts(df, min_text_length=20):
    if df.empty:
        return df.copy()

    result = df.copy()
    result["title"] = result["title"].fillna("").astype(str)
    result["selftext"] = result["selftext"].fillna("").astype(str)

    result["text"] = (
        result["title"] + " " + result["selftext"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    result = result[result["text"].str.len() >= min_text_length].copy()
    result["model_text"] = result["text"].str.slice(0, MAX_TEXT_CHARS)

    return result.drop_duplicates("id").reset_index(drop=True)


def select_top_posts(df, top_posts):
    if df.empty:
        return df.copy()

    return (
        df.sort_values(
            ["score", "num_comments"],
            ascending=False,
            kind="stable"
        )
        .head(min(top_posts, len(df)))
        .reset_index(drop=True)
    )


# -------------------------------------------------------------- SENTIMENT ---

def analyze_sentiment(df, batch_size):
    if df.empty:
        return df.copy()

    model = load_sentiment_model()
    predictions = model(df["model_text"].tolist(), batch_size=batch_size)

    result = df.copy()
    result["sentiment"] = [
        str(p["label"]).lower().strip() for p in predictions
    ]
    result["sentiment_confidence"] = [
        float(p["score"]) for p in predictions
    ]

    return result


# --------------------------------------------------------------- EMOTIONS ---

def analyze_emotions(df, batch_size):
    """Add DistilRoBERTa emotion labels, confidence and per-class scores."""
    if df.empty:
        return df.copy()

    model = load_emotion_model()
    texts = df["model_text"].fillna("").astype(str).tolist()

    predictions = model(texts, batch_size=batch_size)

    result = df.copy()
    emotion_scores = []
    emotion_labels = []
    emotion_confidences = []

    for prediction in predictions:
        # top_k=None returns one score dictionary per emotion.
        if isinstance(prediction, dict):
            prediction = [prediction]

        scores = {
            str(item["label"]).lower(): float(item["score"])
            for item in prediction
        }

        # Ensure every expected class has a value.
        for label in EMOTION_LABELS:
            scores.setdefault(label, 0.0)

        dominant = max(scores, key=scores.get)

        emotion_labels.append(dominant)
        emotion_confidences.append(scores[dominant])
        emotion_scores.append(scores)

    result["emotion"] = emotion_labels
    result["emotion_confidence"] = emotion_confidences

    for label in EMOTION_LABELS:
        result[f"emotion_{label}"] = [
            scores[label] for scores in emotion_scores
        ]

    return result


# -------------------------------------------------------------- EMBEDDINGS ---

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()

    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    return summed / counts


def encode_minilm(texts):
    tokenizer, model = load_embedding_model()
    vectors = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        with torch.inference_mode():
            output = model(**encoded)
            pooled = mean_pool(
                output.last_hidden_state,
                encoded["attention_mask"]
            )
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        vectors.append(pooled.cpu().numpy())

    return np.vstack(vectors)


# ----------------------------------------------------------- TOPIC MODEL ---

def discover_topics(df):
    if len(df) < 3:
        result = df.copy()
        result["topic_id"] = 0
        return result, pd.DataFrame(), 1

    embeddings = encode_minilm(df["model_text"].tolist())
    max_k = min(8, len(df) - 1)

    if max_k < 3:
        result = df.copy()
        result["topic_id"] = 0
        return result, pd.DataFrame(), 1

    scores = []

    for k in range(3, max_k + 1):
        try:
            labels = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=5
            ).fit_predict(embeddings)

            if len(np.unique(labels)) < 2:
                continue

            scores.append({
                "k": k,
                "silhouette_score": float(
                    silhouette_score(
                        embeddings, labels, metric="cosine"
                    )
                )
            })
        except Exception:
            continue

    if scores:
        silhouette_df = pd.DataFrame(scores)
        best_k = int(
            silhouette_df.loc[
                silhouette_df["silhouette_score"].idxmax(), "k"
            ]
        )
    else:
        silhouette_df = pd.DataFrame()
        best_k = min(5, len(df) - 1)

    result = df.copy()
    result["topic_id"] = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=5
    ).fit_predict(embeddings)

    return result, silhouette_df, best_k


def extract_topic_keywords(df, top_n=10):
    if df.empty or "topic_id" not in df.columns:
        return {}

    stopwords = set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS

    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(stopwords),
            max_features=4000,
            ngram_range=(1, 2),
            min_df=2
        )
        matrix = vectorizer.fit_transform(df["model_text"])
    except ValueError:
        return {}

    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords = {}

    for topic_id in sorted(df["topic_id"].unique()):
        indexes = np.where(df["topic_id"].values == topic_id)[0]
        topic_scores = matrix[indexes].mean(axis=0).A1
        top_indexes = topic_scores.argsort()[::-1][:top_n]
        keywords[int(topic_id)] = feature_names[top_indexes].tolist()

    return keywords


# -------------------------------------------------------------- ENGAGEMENT ---

def calculate_engagement(df):
    result = df.copy()

    result["score"] = pd.to_numeric(
        result["score"], errors="coerce"
    ).fillna(0)

    result["num_comments"] = pd.to_numeric(
        result["num_comments"], errors="coerce"
    ).fillna(0)

    result["log_score"] = np.log1p(result["score"].clip(lower=0))
    result["log_comments"] = np.log1p(
        result["num_comments"].clip(lower=0)
    )
    result["engagement"] = (
        result["log_score"] + result["log_comments"]
    )

    return result


def get_top_engaged_posts(df, n=10):
    columns = [
        "title", "score", "num_comments", "sentiment",
        "emotion", "topic_id", "engagement", "url"
    ]
    available = [c for c in columns if c in df.columns]

    return (
        df.sort_values("engagement", ascending=False)[available]
        .head(n)
        .reset_index(drop=True)
    )


def sentiment_summary(df):
    if df.empty or "sentiment" not in df.columns:
        return pd.DataFrame(
            columns=["sentiment", "count", "percentage"]
        )

    summary = (
        df["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )

    summary["percentage"] = (
        summary["count"] / summary["count"].sum() * 100
    )

    return summary


def create_topic_evidence(df, keywords, top_n_posts=5):
    evidence = []

    for topic_id in sorted(df["topic_id"].unique()):
        topic_df = df[df["topic_id"] == topic_id]

        representative = (
            topic_df.sort_values(
                ["score", "num_comments"], ascending=False
            )
            .head(top_n_posts)
        )

        posts = [
            {
                "title": str(row.title),
                "text": str(row.selftext)[:1000],
                "score": safe_int(row.score),
                "comments": safe_int(row.num_comments)
            }
            for row in representative.itertuples()
        ]

        evidence.append({
            "topic_id": int(topic_id),
            "post_count": int(len(topic_df)),
            "avg_score": round(float(topic_df["score"].mean()), 2),
            "avg_comments": round(
                float(topic_df["num_comments"].mean()), 2
            ),
            "keywords": keywords.get(int(topic_id), []),
            "representative_posts": posts
        })

    return evidence


# --------------------------------------------------------------- GROQ/LLM ---

def extract_json_object(content):
    """Parse JSON directly, or extract the first JSON object from text."""
    if not content:
        raise ValueError("The AI returned an empty response.")

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Handle accidental Markdown fences or surrounding text.
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError("The AI response did not contain a JSON object.")

    return json.loads(match.group(0))


def analyze_topic_with_groq(topic_evidence, api_key):
    topic_id = topic_evidence["topic_id"]

    fallback = {
        "topic_id": topic_id,
        "name": f"Topic {topic_id}",
        "description": "",
        "main_reaction": ""
    }

    if not api_key:
        return {
            **fallback,
            "error": "GROQ_API_KEY is not configured."
        }

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

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

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only a valid JSON object. "
                        "Do not include Markdown."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_completion_tokens=500,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        parsed = extract_json_object(content)

        if not isinstance(parsed, dict):
            raise ValueError("The AI response was not a JSON object.")

        # Validate and normalize all required fields.
        name = parsed.get("name")
        description = parsed.get("description")
        main_reaction = parsed.get("main_reaction")

        if not all(
            isinstance(value, str)
            for value in (name, description, main_reaction)
        ):
            raise ValueError("The AI response has invalid field types.")

        return {
            "topic_id": topic_id,
            "name": name.strip() or fallback["name"],
            "description": description.strip(),
            "main_reaction": main_reaction.strip()
        }

    except Exception as exc:
        return {**fallback, "error": str(exc)}


def generate_ai_topic_insights(evidence, workers=3):
    if not evidence:
        return []

    api_key = get_groq_key()

    if not api_key:
        return [
            {
                "topic_id": item["topic_id"],
                "name": f"Topic {item['topic_id']}",
                "description": "",
                "main_reaction": "",
                "error": "GROQ_API_KEY is missing."
            }
            for item in evidence
        ]

    results = []

    with ThreadPoolExecutor(
        max_workers=min(workers, len(evidence))
    ) as executor:
        futures = {
            executor.submit(
                analyze_topic_with_groq, item, api_key
            ): item
            for item in evidence
        }

        for future in as_completed(futures):
            item = futures[future]

            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "topic_id": item["topic_id"],
                    "name": f"Topic {item['topic_id']}",
                    "description": "",
                    "main_reaction": "",
                    "error": str(exc)
                })

    return sorted(
        results, key=lambda x: x.get("topic_id", 999)
    )


def generate_overall_review(evidence):
    api_key = get_groq_key()

    if not api_key:
        return "GROQ_API_KEY is not configured."

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        prompt = f"""
Analyze this Reddit subreddit sample.

Provide a concise 2-3 paragraph intelligence summary covering:
- Overall community consensus
- Major sentiment drivers
- Important recurring discussions
- Unusual or niche observations

Use only the supplied evidence.

Evidence:
{json.dumps(evidence, ensure_ascii=False)}
"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as exc:
        return f"Error generating review: {exc}"


# ----------------------------------------------------------- NEW CHARTS ---

def create_topic_emotion_heatmap(df, name_map):
    """Mean emotion-class probabilities for every discovered topic."""
    emotion_cols = [f"emotion_{label}" for label in EMOTION_LABELS]

    if df.empty or not all(c in df.columns for c in emotion_cols):
        return None

    means = df.groupby("topic_id")[emotion_cols].mean()
    means.index = [
        name_map.get(int(topic_id), f"Topic {topic_id}")
        for topic_id in means.index
    ]

    means.columns = EMOTION_LABELS

    fig = px.imshow(
        means * 100,
        labels={
            "x": "Emotion",
            "y": "Topic",
            "color": "Mean probability (%)"
        },
        color_continuous_scale="RdYlBu_r",
        aspect="auto",
        text_auto=".1f"
    )

    return style_fig(
        fig,
        title="Topic × Emotion Heatmap",
        margin=dict(t=60, b=40, l=100, r=20)
    )


def create_emotion_radar(df, topic_id, topic_name):
    """One radar chart showing mean emotion probabilities for a topic."""
    emotion_cols = [f"emotion_{label}" for label in EMOTION_LABELS]

    topic_df = df[df["topic_id"] == topic_id]

    if topic_df.empty or not all(c in topic_df.columns for c in emotion_cols):
        return None

    values = [
        float(topic_df[f"emotion_{label}"].mean()) * 100
        for label in EMOTION_LABELS
    ]

    labels = EMOTION_LABELS + [EMOTION_LABELS[0]]
    values = values + [values[0]]

    fig = px.line_polar(
        r=values,
        theta=labels,
        line_close=True,
        range_r=[0, 100],
        title=f"Emotion Radar — {topic_name}"
    )

    fig.update_traces(fill="toself")

    return style_fig(fig, margin=dict(t=60, b=30, l=30, r=30))


def create_rolling_trends(df):
    """
    Aggregate sentiment and emotion probabilities by day.
    Uses a rolling 7-day mean when enough dates are available.
    """
    if df.empty:
        return None, None

    result = df.copy()
    result["date"] = pd.to_datetime(
        result["created_utc"], unit="s", utc=True
    ).dt.floor(ROLLING_WINDOW)

    # Sentiment polarity: positive = +1, neutral = 0, negative = -1.
    polarity_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    result["sentiment_polarity"] = (
        result["sentiment"].map(polarity_map).fillna(0)
    )

    daily = (
        result.groupby("date")
        .agg(
            sentiment=("sentiment_polarity", "mean"),
            posts=("id", "count")
        )
        .sort_index()
    )

    daily["rolling_sentiment"] = daily["sentiment"].rolling(
        window=7, min_periods=1
    ).mean()

    sentiment_fig = px.line(
        daily.reset_index(),
        x="date",
        y="rolling_sentiment",
        title="Rolling Sentiment Trend",
        labels={
            "date": "Date",
            "rolling_sentiment": "Mean sentiment polarity"
        }
    )

    sentiment_fig.add_hline(y=0, line_dash="dash", line_color="#888")

    sentiment_fig = style_fig(sentiment_fig)

    emotion_cols = [f"emotion_{label}" for label in EMOTION_LABELS]

    if not all(c in result.columns for c in emotion_cols):
        return sentiment_fig, None

    daily_emotions = (
        result.groupby("date")[emotion_cols]
        .mean()
        .sort_index()
    )

    rolling_emotions = daily_emotions.rolling(
        window=7, min_periods=1
    ).mean()

    rolling_emotions.columns = EMOTION_LABELS

    emotion_fig = px.line(
        rolling_emotions.reset_index(),
        x="date",
        y=EMOTION_LABELS,
        title="Rolling Emotion Trend",
        labels={
            "date": "Date",
            "value": "Mean emotion probability",
            "variable": "Emotion"
        }
    )

    emotion_fig = style_fig(emotion_fig)

    return sentiment_fig, emotion_fig


# ---------------------------------------------------------------- PIPELINE ---

def run_nlp_pipeline(raw_df, top_posts):
    empty = {
        "clean": pd.DataFrame(),
        "analysis": pd.DataFrame(),
        "best_k": None,
        "keywords": {},
        "evidence": [],
        "top_engaged": pd.DataFrame(),
        "sentiment": pd.DataFrame(),
        "ai_insights": [],
        "overall_review": ""
    }

    clean_df = clean_posts(raw_df)

    if clean_df.empty:
        return empty

    analysis_df = select_top_posts(clean_df, top_posts)

    if analysis_df.empty:
        return empty

    analysis_df = analyze_sentiment(
        analysis_df, DEFAULT_BATCH_SIZE
    )

    # NEW: emotion classification before topic discovery.
    analysis_df = analyze_emotions(
        analysis_df, DEFAULT_BATCH_SIZE
    )

    analysis_df, _, best_k = discover_topics(analysis_df)

    keywords = extract_topic_keywords(analysis_df)
    analysis_df = calculate_engagement(analysis_df)

    evidence = create_topic_evidence(analysis_df, keywords)

    return {
        "clean": clean_df,
        "analysis": analysis_df,
        "best_k": best_k,
        "keywords": keywords,
        "evidence": evidence,
        "top_engaged": get_top_engaged_posts(analysis_df),
        "sentiment": sentiment_summary(analysis_df),
        "ai_insights": generate_ai_topic_insights(
            evidence, DEFAULT_WORKERS
        ),
        "overall_review": generate_overall_review(evidence)
    }


def generate_word_cloud(df):
    if df.empty:
        return None

    text = " ".join(
        df["title"].fillna("") + " " + df["selftext"].fillna("")
    )

    stopwords = set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS

    wordcloud = WordCloud(
        width=1200,
        height=550,
        background_color=BG,
        stopwords=stopwords,
        colormap="turbo",
        max_words=150,
        min_font_size=10,
        max_font_size=100,
        prefer_horizontal=0.9
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)

    return fig


def page_header(title, subtitle):
    st.html(f"""
    <div class="reddit-header">
        <div class="reddit-icon">
            <i class="fa-brands fa-reddit" aria-hidden="true"></i>
        </div>
        <div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
    </div>
    """)


# ---------------------------------------------------------------- SIDEBAR ---

st.markdown(
    '<link rel="stylesheet" '
    'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="reddit-sidebar-icon">'
    '<i class="fa-brands fa-reddit" aria-hidden="true"></i></div>',
    unsafe_allow_html=True
)

st.sidebar.title("Recon Settings")

with st.sidebar.form("subreddit_form", clear_on_submit=False):
    subreddit_input = st.text_input(
        "Subreddit",
        placeholder="e.g. technology, Python, gaming",
        help="Enter a subreddit without r/"
    )

    days_back = st.slider("Days to analyze", 1, 30, 1)
    posts_to_fetch = st.slider("Posts to fetch", 100, 1000, 300, 50)
    top_posts = st.slider("Top posts for NLP", 10, 500, 100, 10)

    submitted = st.form_submit_button(
        "Run Recon",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------------------- RUN RECON ---

if submitted:
    subreddit = normalize_subreddit(subreddit_input)

    if not subreddit or not subreddit.replace("_", "").isalnum():
        st.sidebar.error("Use a valid subreddit name.")
    else:
        st.session_state.pop("result", None)
        st.session_state["selected_subreddit"] = subreddit

        progress = st.progress(0)
        status = st.empty()

        try:
            status.write(f"Fetching posts from r/{subreddit}...")

            raw_df = fetch_reddit_posts(
                subreddit, posts_to_fetch, days_back
            )

            progress.progress(35)

            if raw_df.empty:
                status.empty()
                progress.empty()

                st.error(
                    f"No posts were fetched from r/{subreddit}. "
                    "Check the subreddit name or try a longer date range."
                )
                st.stop()

            status.write(
                f"Fetched {len(raw_df):,} posts. "
                "Running sentiment, emotion and topic analysis..."
            )

            result = run_nlp_pipeline(raw_df, top_posts)
            progress.progress(85)

            if not isinstance(result, dict) or "analysis" not in result:
                raise RuntimeError(
                    "NLP pipeline returned an invalid result."
                )

            result["raw"] = raw_df
            st.session_state["result"] = result

            progress.progress(100)
            status.empty()
            progress.empty()

            st.success(f"Recon complete for r/{subreddit}.")

        except Exception as exc:
            status.empty()
            progress.empty()
            st.error("Recon failed.")
            st.exception(exc)


# ------------------------------------------------------------- NO RESULT ---

if "result" not in st.session_state:
    page_header(
        "Reddit Recon",
        "Community intelligence, semantic topics, "
        "sentiment and engagement analysis."
    )
    st.write("Enter a subreddit in the sidebar and click **Run Recon**.")
    st.stop()

result = st.session_state.get("result")

if not isinstance(result, dict):
    st.error("Invalid analysis result.")
    st.session_state.pop("result", None)
    st.stop()

raw_df = result.get("raw", pd.DataFrame())
analysis_df = result.get("analysis", pd.DataFrame())
best_k = result.get("best_k")
ai_insights = result.get("ai_insights", [])
overall_review = result.get("overall_review", "")
sentiment = result.get("sentiment", pd.DataFrame())

if analysis_df.empty:
    st.error("No usable posts were available for NLP analysis.")
    st.stop()

SUBREDDIT = st.session_state.get(
    "selected_subreddit", subreddit_input
)

name_map = {
    item.get("topic_id"): item.get(
        "name", f"Topic {item.get('topic_id')}"
    )
    for item in ai_insights
}

stats = build_topic_stats(analysis_df, name_map)
color_map = topic_color_map(stats["Topic Name"])

page_header(
    f"Reddit Recon: r/{html.escape(SUBREDDIT)}",
    f"Analyzed top {len(analysis_df):,} posts."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Topics", "Review", "Data"]
)


# ------------------------------------------------------------------ TAB 1 ---

with tab1:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Fetched", f"{len(raw_df):,}")
    c2.metric("NLP Analyzed", f"{len(analysis_df):,}")
    c3.metric("Topics Discovered", best_k if best_k else "N/A")

    pos_pct = (
        sentiment.loc[
            sentiment["sentiment"] == "positive", "percentage"
        ].sum()
        if not sentiment.empty else 0
    )

    c4.metric("Positivity", f"{pos_pct:.1f}%")

    col1, col2 = st.columns(2)

    with col1, card():
        st.subheader("Sentiment Distribution")

        if not sentiment.empty:
            fig = px.pie(
                sentiment,
                values="percentage",
                names="sentiment",
                hole=0.4
            )

            st.plotly_chart(
                style_fig(
                    fig,
                    margin=dict(t=0, b=0, l=0, r=0),
                    legend=dict(font=dict(color="white"))
                ),
                use_container_width=True
            )

    with col2, card():
        st.subheader("Topic Engagement")

        fig = px.scatter(
            stats,
            x="avg_score",
            y="posts",
            size="posts",
            hover_name="Topic Name",
            text="Topic Name",
            color="Topic Name",
            color_discrete_map=color_map,
            labels={
                "avg_score": "Average Score",
                "posts": "Volume of Posts"
            }
        )

        fig.update_traces(textposition="top center")

        st.plotly_chart(
            style_fig(
                fig,
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20)
            ),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Community Activity")

    col3, col4 = st.columns(2)

    with col3:
        fig = px.bar(
            stats.sort_values("posts", ascending=False),
            x="Topic Name",
            y="posts",
            text="posts",
            title="Posts by Topic",
            color="Topic Name",
            color_discrete_map=color_map,
            labels={"posts": "Number of Posts", "Topic Name": ""}
        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(
            style_fig(
                fig,
                margin=dict(t=60, b=70, l=40, r=20),
                showlegend=False,
                xaxis=dict(tickangle=-25, gridcolor=BORDER)
            ),
            use_container_width=True
        )

    with col4:
        fig = px.bar(
            stats.sort_values("avg_comments", ascending=False),
            x="Topic Name",
            y="avg_comments",
            text="avg_comments",
            title="Average Comments by Topic",
            color="Topic Name",
            color_discrete_map=color_map,
            labels={
                "avg_comments": "Average Comments",
                "Topic Name": ""
            }
        )

        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )

        st.plotly_chart(
            style_fig(
                fig,
                margin=dict(t=60, b=70, l=40, r=20),
                showlegend=False,
                xaxis=dict(tickangle=-25, gridcolor=BORDER)
            ),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Sentiment & Engagement")

    col5, col6 = st.columns(2)

    with col5:
        counts = analysis_df["sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        counts["Sentiment"] = counts["sentiment"].str.capitalize()

        fig = px.bar(
            counts,
            x="Sentiment",
            y="count",
            text="count",
            title="Number of Posts by Sentiment",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            labels={"count": "Posts", "Sentiment": ""}
        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(
            style_fig(
                fig,
                margin=dict(t=60, b=40, l=40, r=20),
                showlegend=False
            ),
            use_container_width=True
        )

    with col6:
        fig = px.scatter(
            stats,
            x="posts",
            y="avg_engagement",
            size="total_engagement",
            hover_name="Topic Name",
            text="Topic Name",
            color="Topic Name",
            color_discrete_map=color_map,
            title="Topic Engagement vs. Volume",
            labels={
                "posts": "Number of Posts",
                "avg_engagement": "Average Engagement",
                "total_engagement": "Total Engagement"
            }
        )

        fig.update_traces(textposition="top center")

        st.plotly_chart(
            style_fig(
                fig,
                margin=dict(t=60, b=50, l=50, r=30),
                legend=dict(font=dict(color="white"))
            ),
            use_container_width=True
        )

    # NEW: topic × emotion heatmap.
    st.markdown("---")

    with card():
        st.subheader("Topic × Emotion Heatmap")
        heatmap = create_topic_emotion_heatmap(analysis_df, name_map)

        if heatmap is not None:
            st.plotly_chart(heatmap, use_container_width=True)
        else:
            st.info("Emotion scores are not available.")

    # NEW: rolling sentiment and emotion trends.
    st.markdown("---")
    st.subheader("Rolling Sentiment & Emotion Trends")

    trend1, trend2 = st.columns(2)

    sentiment_trend, emotion_trend = create_rolling_trends(analysis_df)

    with trend1:
        if sentiment_trend is not None:
            st.plotly_chart(
                sentiment_trend,
                use_container_width=True
            )

    with trend2:
        if emotion_trend is not None:
            st.plotly_chart(
                emotion_trend,
                use_container_width=True
            )


# ------------------------------------------------------------------ TAB 2 ---

with tab2:
    if not ai_insights:
        st.warning("No AI insights were generated.")

    for item in ai_insights:
        topic_id = item.get("topic_id")
        topic_df = analysis_df[
            analysis_df["topic_id"] == topic_id
        ]

        if topic_df.empty:
            continue

        topic_sentiment = (
            topic_df["sentiment"]
            .value_counts(normalize=True) * 100
        ).sort_values(ascending=False)

        badges = "".join(
            sentiment_badge(label, pct)
            for label, pct in topic_sentiment.items()
        )

        representative = (
            topic_df.sort_values(
                ["score", "num_comments"], ascending=False
            )
            .head(5)
        )

        rep_html = "".join(
            f"""
            <li style="margin-bottom:.7rem;">
                <a class="top-post-link"
                   href="{html.escape(str(row.url))}"
                   target="_blank" rel="noopener noreferrer">
                    <b>{html.escape(str(row.title))}</b>
                </a>
                <span>
                    (Score: {safe_int(row.score):,} |
                    Comments: {safe_int(row.num_comments):,})
                </span>
            </li>
            """
            for row in representative.itertuples()
        )

        error = item.get("error")
        error_html = (
            f"<p><b>AI status:</b> {html.escape(str(error))}</p>"
            if error else ""
        )

        topic_name = item.get("name", f"Topic {topic_id}")

        st.html(f"""
        <div class="reddit-card">
            <div class="topic-title">
                {html.escape(str(topic_name))}
            </div>

            <div style="margin-bottom:1rem;">{badges}</div>

            <p>
                <b>Analysis:</b>
                {html.escape(str(item.get('description', '')))}
            </p>

            <p>
                <b>Reaction Context:</b>
                {html.escape(str(item.get('main_reaction', '')))}
            </p>

            {error_html}

            <hr/>

            <p><b>TOP POSTS IN TOPIC</b></p>

            <ul style="list-style-type:none; padding-left:0;">
                {rep_html}
            </ul>
        </div>
        """)

        # NEW: emotion radar for each topic.
        radar = create_emotion_radar(
            analysis_df, topic_id, str(topic_name)
        )

        if radar is not None:
            with card():
                st.plotly_chart(
                    radar,
                    use_container_width=True
                )


# ------------------------------------------------------------------ TAB 3 ---

with tab3:
    with card():
        st.subheader(f"General Consensus: r/{SUBREDDIT}")

        if overall_review:
            st.write(overall_review)
        else:
            st.info("LLM summary is not available.")

    with card():
        st.subheader("What they are talking about")
        st.caption("Common terms after removing standard stopwords.")

        wc_fig = generate_word_cloud(analysis_df)

        if wc_fig is not None:
            st.pyplot(wc_fig, clear_figure=True)


# ------------------------------------------------------------------ TAB 4 ---

with tab4:
    with card():
        st.subheader("Top Posts Dataset")

        display_cols = [
            "score",
            "num_comments",
            "title",
            "sentiment",
            "sentiment_confidence",
            "emotion",
            "emotion_confidence",
            "topic_id",
            "url"
        ]

        # Include all individual emotion probability columns.
        display_cols.extend(
            f"emotion_{label}" for label in EMOTION_LABELS
        )

        available_cols = [
            c for c in display_cols if c in analysis_df.columns
        ]

        st.dataframe(
            analysis_df.sort_values(
                "score", ascending=False
            )[available_cols],
            use_container_width=True,
            hide_index=True
        )


# ------------------------------------------------------------------- ABOUT ---

with st.sidebar.expander("About & Pipeline"):
    st.html(f"""
    <strong>Current Analysis</strong>
    <ul>
        <li><strong>Subreddit:</strong> r/{html.escape(SUBREDDIT)}</li>
        <li><strong>Days:</strong> {days_back}</li>
        <li><strong>Posts fetched:</strong> {posts_to_fetch}</li>
        <li><strong>Posts analyzed:</strong> {top_posts}</li>
    </ul>

    <strong>Models</strong>
    <ul>
        <li><strong>Sentiment:</strong>
            <code>{html.escape(SENTIMENT_MODEL)}</code>
        </li>
        <li><strong>Emotion:</strong>
            <code>{html.escape(EMOTION_MODEL)}</code>
        </li>
        <li><strong>Embeddings:</strong>
            <code>{html.escape(EMBEDDING_MODEL)}</code>
        </li>
        <li><strong>AI:</strong>
            <code>{html.escape(LLM_MODEL)}</code>
        </li>
    </ul>

    <strong>Description</strong>
    <p>
        Reddit → Sentiment → Emotion → Embeddings → Clustering →
        Topic Keywords → GPT-OSS Topic Analysis → Community Review
    </p>
    """)
