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

# Fixed GPT-OSS model — no model selector in UI
LLM_MODEL = "openai/gpt-oss-20b"

DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 3

RANDOM_STATE = 42
MAX_FETCH_RETRIES = 4
MAX_TEXT_CHARS = 2500
EMBEDDING_BATCH_SIZE = 32


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Reddit Recon",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DARK THEME — WHITE TEXT ONLY
# ============================================================

st.html(
    """
<style>

/* ========================================================
   GLOBAL
   ======================================================== */

.stApp {
    background: #111111;
    color: #ffffff;
}

.main {
    background: #111111;
}

/* Force normal Streamlit text to white */
html,
body,
[class*="css"],
p,
span,
div,
label,
li,
td,
th {
    color: #ffffff;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* ========================================================
   SIDEBAR
   ======================================================== */

section[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #2a2a2a;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* ========================================================
   HEADER
   ======================================================== */

.reddit-header {
    display: flex;
    align-items: center;
    background: #111111;
    padding: 1.2rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #2a2a2a;
}

.reddit-header .reddit-icon {
    font-size: 58px;
    margin-right: 20px;
    color: #ffffff;
    line-height: 1;
}

.reddit-header h1 {
    margin: 0;
    padding: 0;
    font-size: 2.2rem;
    color: #ffffff !important;
    font-weight: 700;
}

.reddit-header p {
    margin: 5px 0 0 0;
    color: #ffffff !important;
    font-size: 0.95rem;
}

/* ========================================================
   CARDS
   ======================================================== */

.reddit-card {
    background: #111111;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.topic-title {
    color: #ffffff !important;
    font-weight: 700;
    font-size: 1.35rem;
    margin-bottom: 0.5rem;
}

/* ========================================================
   BADGES
   ======================================================== */

.badge {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #ffffff !important;
    margin-right: 0.5rem;
    border: 1px solid #444444;
}

/* ========================================================
   LINKS
   ======================================================== */

a {
    color: #ffffff !important;
}

a:hover {
    color: #ffffff !important;
    text-decoration: underline;
}

/* ========================================================
   INPUTS
   ======================================================== */

input,
textarea,
select {
    color: #ffffff !important;
    background-color: #181818 !important;
}

/* ========================================================
   STREAMLIT BUTTON
   ======================================================== */

button {
    color: #ffffff !important;
}

/* ========================================================
   DIVIDERS
   ======================================================== */

hr {
    border-color: #2a2a2a !important;
}

/* ========================================================
   DATAFRAME
   ======================================================== */

[data-testid="stDataFrame"] {
    border: 1px solid #2a2a2a;
}

/* ========================================================
   METRICS
   ======================================================== */

[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
    color: #ffffff !important;
}

/* ========================================================
   CAPTIONS / INFO
   ======================================================== */

.stCaption {
    color: #ffffff !important;
}

/* ========================================================
   FILE / SELECT / FORM ELEMENTS
   ======================================================== */

div[data-baseweb="select"] * {
    color: #ffffff !important;
    background-color: #181818 !important;
}

div[data-baseweb="input"] * {
    color: #ffffff !important;
}

</style>

<!-- Font Awesome -->
<link
    rel="stylesheet"
    href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css"
>
"""
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


def sentiment_badge(label, pct=None):
    text = label.capitalize() if pct is None else f"{label.capitalize()} {pct:.0f}%"

    # White-only theme
    return (
        f'<span class="badge">{html.escape(text)}</span>'
    )


def normalize_subreddit(value):
    value = value.strip()

    if value.lower().startswith("r/"):
        value = value[2:]

    return value.strip().replace(" ", "")


# ============================================================
# HTTP
# ============================================================

@st.cache_resource
def get_http_session():
    retry = Retry(
        total=MAX_FETCH_RETRIES,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=4,
        pool_maxsize=4,
    )

    session = requests.Session()
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": "reddit-recon-ui/3.0"
    })

    return session


# ============================================================
# MODELS
# ============================================================

@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        truncation=True,
        max_length=256,
        device=-1,
    )


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)

    model.eval()

    return tokenizer, model


# ============================================================
# FETCH REDDIT
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_reddit_posts(subreddit, posts_to_fetch, days_back):

    now = int(time.time())

    cutoff = now - days_back * 24 * 60 * 60
    before = now

    page_size = 100
    max_pages = int(np.ceil(posts_to_fetch / page_size))

    session = get_http_session()

    posts = []
    seen_ids = set()

    fields = (
        "id,created_utc,score,num_comments,"
        "subreddit,title,selftext,url"
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

            created = safe_int(
                item.get("created_utc", 0)
            )

            if created < cutoff:
                continue

            seen_ids.add(post_id)

            posts.append({
                "id": post_id,
                "title": str(item.get("title") or ""),
                "selftext": str(item.get("selftext") or ""),
                "score": safe_int(item.get("score", 0)),
                "num_comments": safe_int(
                    item.get("num_comments", 0)
                ),
                "created_utc": created,
                "url": str(item.get("url") or ""),
                "subreddit": str(
                    item.get("subreddit") or subreddit
                ),
            })

            if (
                oldest_timestamp is None
                or created < oldest_timestamp
            ):
                oldest_timestamp = created

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


# ============================================================
# CLEANING
# ============================================================

def clean_posts(df, min_text_length=20):

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
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    result = result[
        result["text"].str.len() >= min_text_length
    ].copy()

    result["model_text"] = (
        result["text"]
        .str.slice(0, MAX_TEXT_CHARS)
    )

    return (
        result
        .drop_duplicates("id")
        .reset_index(drop=True)
    )


def select_top_posts(df, top_posts):

    if df.empty:
        return df.copy()

    return (
        df.sort_values(
            ["score", "num_comments"],
            ascending=False,
            kind="stable",
        )
        .head(min(top_posts, len(df)))
        .reset_index(drop=True)
    )


# ============================================================
# SENTIMENT
# ============================================================

def analyze_sentiment(df, batch_size):

    if df.empty:
        return df.copy()

    model = load_sentiment_model()

    result = df.copy()

    predictions = model(
        result["model_text"].tolist(),
        batch_size=batch_size,
    )

    result["sentiment"] = [
        str(p["label"]).lower().strip()
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

def mean_pool(last_hidden_state, attention_mask):

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
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

    tokenizer, model = load_embedding_model()

    vectors = []

    for start in range(
        0,
        len(texts),
        EMBEDDING_BATCH_SIZE,
    ):

        batch = texts[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.inference_mode():

            output = model(**encoded)

            pooled = mean_pool(
                output.last_hidden_state,
                encoded["attention_mask"],
            )

            pooled = torch.nn.functional.normalize(
                pooled,
                p=2,
                dim=1,
            )

        vectors.append(
            pooled.cpu().numpy()
        )

    return np.vstack(vectors)


# ============================================================
# TOPIC DISCOVERY
# ============================================================

def discover_topics(df):

    if len(df) < 3:

        result = df.copy()
        result["topic_id"] = 0

        return result, pd.DataFrame(), 1

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

        return result, pd.DataFrame(), 1

    silhouette_results = []

    for k in range(3, max_k + 1):

        try:

            km = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=5,
            )

            labels = km.fit_predict(
                embeddings
            )

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(
                embeddings,
                labels,
                metric="cosine",
            )

            silhouette_results.append({
                "k": k,
                "silhouette_score": float(score),
            })

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
            final_model.fit_predict(embeddings)
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
            silhouette_df["silhouette_score"].idxmax(),
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
        final_model.fit_predict(embeddings)
    )

    return (
        result,
        silhouette_df,
        best_k,
    )


# ============================================================
# TOPIC KEYWORDS
# ============================================================

def extract_topic_keywords(df, top_n=10):

    if (
        df.empty
        or "topic_id" not in df.columns
    ):
        return {}

    stopwords = set(
        ENGLISH_STOP_WORDS
    )

    stopwords.update({
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
    })

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
            df["topic_id"].values == topic_id
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
            feature_names[top_indexes].tolist()
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
        result["score"].clip(lower=0)
    )

    result["log_comments"] = np.log1p(
        result["num_comments"].clip(lower=0)
    )

    result["engagement"] = (
        result["log_score"]
        + result["log_comments"]
    )

    return result


def get_top_engaged_posts(df, n=10):

    columns = [
        "title",
        "score",
        "num_comments",
        "sentiment",
        "topic_id",
        "engagement",
        "url",
    ]

    available = [
        c for c in columns
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


# ============================================================
# SENTIMENT SUMMARY
# ============================================================

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
        .reset_index(name="count")
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
    top_n_posts=5,
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

            posts.append({
                "title": str(
                    row.get("title", "")
                ),
                "text": str(
                    row.get("selftext", "")
                )[:1000],
                "score": safe_int(
                    row.get("score", 0)
                ),
                "comments": safe_int(
                    row.get("num_comments", 0)
                ),
            })

        evidence.append({
            "topic_id": int(topic_id),
            "post_count": int(
                len(topic_df)
            ),
            "avg_score": round(
                float(topic_df["score"].mean()),
                2,
            ),
            "avg_comments": round(
                float(
                    topic_df["num_comments"].mean()
                ),
                2,
            ),
            "keywords": keywords.get(
                int(topic_id),
                [],
            ),
            "representative_posts": posts,
        })

    return evidence


# ============================================================
# GROQ / GPT-OSS
# ============================================================

def analyze_topic_with_groq(
    topic_evidence,
    api_key,
):

    topic_id = topic_evidence["topic_id"]

    if not api_key:

        return {
            "topic_id": topic_id,
            "name": f"Topic {topic_id}",
            "description": "",
            "main_reaction": "",
            "error": "GROQ_API_KEY is not configured.",
        }

    try:

        from groq import Groq

        client = Groq(
            api_key=api_key
        )

        prompt = f"""
You are an expert Reddit community analyst.

Analyze Topic {topic_id} using the evidence below.

Return:
1. A concise, meaningful topic name.
2. A concise explanation of what users are discussing.
3. The dominant reaction or attitude toward the topic.

Do not call it "Topic {topic_id}" unless there is genuinely no
meaningful information available.

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
                    "name": "reddit_topic_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "topic_id": {
                                "type": "integer"
                            },
                            "name": {
                                "type": "string"
                            },
                            "description": {
                                "type": "string"
                            },
                            "main_reaction": {
                                "type": "string"
                            },
                        },
                        "required": [
                            "topic_id",
                            "name",
                            "description",
                            "main_reaction",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        return json.loads(content)

    except Exception as exc:

        return {
            "topic_id": topic_id,
            "name": f"Topic {topic_id}",
            "description": "",
            "main_reaction": "",
            "error": str(exc),
        }


def generate_ai_topic_insights(
    evidence,
    workers=3,
):

    api_key = get_groq_key()

    if not evidence:
        return []

    if not api_key:

        return [
            {
                "topic_id": item["topic_id"],
                "name": f"Topic {item['topic_id']}",
                "description": "",
                "main_reaction": "",
                "error": "GROQ_API_KEY is missing.",
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

        for future in as_completed(futures):

            item = futures[future]

            try:
                results.append(
                    future.result()
                )

            except Exception as exc:

                results.append({
                    "topic_id": item["topic_id"],
                    "name": (
                        f"Topic {item['topic_id']}"
                    ),
                    "description": "",
                    "main_reaction": "",
                    "error": str(exc),
                })

    return sorted(
        results,
        key=lambda x: x.get(
            "topic_id",
            999,
        ),
    )


def generate_overall_review(
    evidence,
):

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

Provide a concise 2–3 paragraph intelligence summary covering:

- Overall community consensus
- Major sentiment drivers
- The most important recurring discussions
- Any unusual or niche observations

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
            temperature=0.3,
            max_completion_tokens=800,
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as exc:

        return f"Error generating review: {exc}"


# ============================================================
# NLP PIPELINE
# ============================================================

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
        "top_engaged": pd.DataFrame(),
        "sentiment": pd.DataFrame(),
        "ai_insights": [],
        "overall_review": "",
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
        analysis_df,
        DEFAULT_BATCH_SIZE,
    )

    analysis_df, _, best_k = (
        discover_topics(
            analysis_df
        )
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

    top_engaged = get_top_engaged_posts(
        analysis_df
    )

    sentiment = sentiment_summary(
        analysis_df
    )

    # GPT-OSS analysis
    ai_insights = generate_ai_topic_insights(
        evidence,
        DEFAULT_WORKERS,
    )

    overall_review = generate_overall_review(
        evidence,
    )

    return {
        "clean": clean_df,
        "analysis": analysis_df,
        "best_k": best_k,
        "keywords": keywords,
        "evidence": evidence,
        "top_engaged": top_engaged,
        "sentiment": sentiment,
        "ai_insights": ai_insights,
        "overall_review": overall_review,
    }


# ============================================================
# WORD CLOUD
# ============================================================

def generate_word_cloud(df):

    if df.empty:
        return None

    text = " ".join(
        df["title"].fillna("")
        + " "
        + df["selftext"].fillna("")
    )

    stopwords = set(
        ENGLISH_STOP_WORDS
    )

    stopwords.update({
        "reddit",
        "post",
        "posts",
        "people",
        "just",
        "like",
        "think",
        "thing",
        "really",
    })

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="#111111",
        stopwords=stopwords,
        color_func=lambda *args, **kwargs: "white",
    ).generate(text)

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    fig.patch.set_facecolor="#111111"
    ax.set_facecolor("#111111")

    ax.imshow(
        wordcloud,
        interpolation="bilinear",
    )

    ax.axis("off")

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.html(
    """
<div style="
    font-size: 46px;
    text-align: center;
    margin: 10px 0 20px 0;
">
    <i class="fa-brands fa-reddit"></i>
</div>
"""
)

st.sidebar.title("Recon Settings")


with st.sidebar.form(
    "subreddit_form",
    clear_on_submit=False,
):

    subreddit_input = st.text_input(
        "Subreddit",
        placeholder="e.g. technology, Python, gaming",
        help="Enter a subreddit without r/",
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
        "Top posts for NLP",
        10,
        500,
        100,
        10,
    )

    submitted = st.form_submit_button(
        "Run Recon",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# RUN RECON
# ============================================================

if submitted:

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

    else:

        # Clear old result first
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
                f"Fetching posts from r/{subreddit}..."
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
                    f"No posts were fetched from r/{subreddit}. "
                    "Check the subreddit name or try a longer date range."
                )

                st.stop()

            status.write(
                f"Fetched {len(raw_df):,} posts. "
                "Running sentiment and topic analysis..."
            )

            result = run_nlp_pipeline(
                raw_df,
                top_posts,
            )

            progress.progress(85)

            if (
                not isinstance(result, dict)
                or "analysis" not in result
            ):

                raise RuntimeError(
                    "NLP pipeline returned an invalid result."
                )

            result["raw"] = raw_df

            # IMPORTANT:
            # Store the complete result in one place.
            st.session_state[
                "result"
            ] = result

            progress.progress(100)

            status.empty()
            progress.empty()

            st.success(
                f"Recon complete for r/{subreddit}."
            )

        except Exception as exc:

            status.empty()
            progress.empty()

            st.error(
                "Recon failed."
            )

            st.exception(exc)


# ============================================================
# NO RESULT YET
# ============================================================

if "result" not in st.session_state:

    st.html(
        """
<div class="reddit-header">
    <div class="reddit-icon">
        <i class="fa-brands fa-reddit"></i>
    </div>

    <div>
        <h1>Reddit Recon</h1>

        <p>
            Community intelligence, semantic topics,
            sentiment and engagement analysis.
        </p>
    </div>
</div>
"""
    )

    st.write(
        "Enter a subreddit in the sidebar and click **Run Recon**."
    )

    st.stop()


# ============================================================
# SAFELY LOAD RESULT
# ============================================================

result = st.session_state.get(
    "result"
)

if not isinstance(result, dict):

    st.error(
        "Invalid analysis result."
    )

    st.session_state.pop(
        "result",
        None,
    )

    st.stop()


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

keywords = result.get(
    "keywords",
    {},
)

top_engaged = result.get(
    "top_engaged",
    pd.DataFrame(),
)

sentiment = result.get(
    "sentiment",
    pd.DataFrame(),
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
        "No usable posts were available for NLP analysis."
    )

    st.stop()


SUBREDDIT = st.session_state.get(
    "selected_subreddit",
    subreddit_input,
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
            Reddit Recon: r/{html.escape(SUBREDDIT)}
        </h1>

        <p>
            Analyzed top {len(analysis_df):,} posts
            from the last {days_back} day(s).
        </p>

    </div>

</div>
"""
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Topics",
    "Review",
    "Data",
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.html(
            f"""
<div class="reddit-card">
    <h4>Total Fetched</h4>
    <h2>{len(raw_df):,}</h2>
</div>
"""
        )

    with col2:

        st.html(
            f"""
<div class="reddit-card">
    <h4>NLP Analyzed</h4>
    <h2>{len(analysis_df):,}</h2>
</div>
"""
        )

    with col3:

        st.html(
            f"""
<div class="reddit-card">
    <h4>Topics Discovered</h4>
    <h2>{best_k if best_k else "N/A"}</h2>
</div>
"""
        )

    with col4:

        pos_pct = 0

        if not sentiment.empty:

            pos_pct = sentiment.loc[
                sentiment["sentiment"] == "positive",
                "percentage",
            ].sum()

        st.html(
            f"""
<div class="reddit-card">
    <h4>Positivity</h4>
    <h2>{pos_pct:.1f}%</h2>
</div>
"""
        )

    c1, c2 = st.columns(2)

    with c1:

        st.html(
            '<div class="reddit-card">'
        )

        st.subheader(
            "Sentiment Distribution"
        )

        if not sentiment.empty:

            fig = px.pie(
                sentiment,
                values="percentage",
                names="sentiment",
                hole=0.4,
            )

            fig.update_layout(
                margin=dict(
                    t=0,
                    b=0,
                    l=0,
                    r=0,
                ),
                paper_bgcolor="#111111",
                plot_bgcolor="#111111",
                font=dict(color="white"),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.html(
            "</div>"
        )

    with c2:

        st.html(
            '<div class="reddit-card">'
        )

        st.subheader(
            "Topic Engagement"
        )

        topic_eng = (
            analysis_df
            .groupby("topic_id")
            .agg(
                posts=("id", "count"),
                avg_score=("score", "mean"),
            )
            .reset_index()
        )

        topic_names_map = {
            item.get("topic_id"): item.get(
                "name",
                f"Topic {item.get('topic_id')}",
            )
            for item in ai_insights
        }

        topic_eng["Topic Name"] = (
            topic_eng["topic_id"]
            .map(topic_names_map)
        )

        fig4 = px.scatter(
            topic_eng,
            x="avg_score",
            y="posts",
            size="posts",
            hover_name="Topic Name",
            labels={
                "avg_score": "Average Score",
                "posts": "Volume of Posts",
            },
        )

        fig4.update_layout(
            margin=dict(
                t=0,
                b=0,
                l=0,
                r=0,
            ),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font=dict(color="white"),
            showlegend=False,
        )

        st.plotly_chart(
            fig4,
            use_container_width=True,
        )

        st.html(
            "</div>"
        )


# ============================================================
# TAB 2 — TOPICS
# ============================================================

with tab2:

    if not ai_insights:

        st.warning(
            "No AI insights were generated."
        )

    for item in ai_insights:

        topic_id = item.get(
            "topic_id"
        )

        ai_name = item.get(
            "name",
            f"Topic {topic_id}",
        )

        ai_desc = item.get(
            "description",
            "",
        )

        ai_reaction = item.get(
            "main_reaction",
            "",
        )

        topic_df = analysis_df[
            analysis_df["topic_id"] == topic_id
        ].copy()

        if topic_df.empty:
            continue

        topic_sentiment = (
            topic_df["sentiment"]
            .value_counts(normalize=True)
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

        representative = (
            topic_df
            .sort_values(
                ["score", "num_comments"],
                ascending=False,
            )
            .head(5)
        )

        rep_html = ""

        for _, row in representative.iterrows():

            title = html.escape(
                str(row["title"])
            )

            url = html.escape(
                str(row["url"])
            )

            rep_html += (
                f"""
<li style="margin-bottom:0.7rem;">
    <a
        href="{url}"
        target="_blank"
    >
        <b>{title}</b>
    </a>

    <span>
        (
        Score: {safe_int(row["score"]):,}
        |
        Comments: {safe_int(row["num_comments"]):,}
        )
    </span>
</li>
"""
            )

        error = item.get(
            "error"
        )

        error_html = ""

        if error:

            error_html = f"""
<p>
    <b>AI status:</b>
    {html.escape(str(error))}
</p>
"""

        st.html(
            f"""
<div class="reddit-card">

    <div class="topic-title">
        {html.escape(str(ai_name))}
    </div>

    <div style="margin-bottom:1rem;">
        {badges}
    </div>

    <p>
        <b>Analysis:</b>
        {html.escape(str(ai_desc))}
    </p>

    <p>
        <b>Reaction Context:</b>
        {html.escape(str(ai_reaction))}
    </p>

    {error_html}

    <hr/>

    <p>
        <b>TOP POSTS IN TOPIC</b>
    </p>

    <ul
        style="
            list-style-type:none;
            padding-left:0;
        "
    >
        {rep_html}
    </ul>

</div>
"""
        )


# ============================================================
# TAB 3 — REVIEW
# ============================================================

with tab3:

    st.html(
        '<div class="reddit-card">'
    )

    st.subheader(
        f"General Consensus: r/{SUBREDDIT}"
    )

    if overall_review:

        st.write(
            overall_review
        )

    else:

        st.info(
            "LLM summary is not available."
        )

    st.html(
        "</div>"
    )

    st.html(
        '<div class="reddit-card">'
    )

    st.subheader(
        "What they are talking about"
    )

    st.caption(
        "Common terms after removing standard stopwords."
    )

    wc_fig = generate_word_cloud(
        analysis_df
    )

    if wc_fig is not None:
        st.pyplot(
            wc_fig,
            clear_figure=True,
        )

    st.html(
        "</div>"
    )


# ============================================================
# TAB 4 — DATA
# ============================================================

with tab4:

    st.html(
        '<div class="reddit-card">'
    )

    st.subheader(
        "Top Posts Dataset"
    )

    display_cols = [
        "score",
        "num_comments",
        "title",
        "sentiment",
        "topic_id",
        "url",
    ]

    available_cols = [
        c for c in display_cols
        if c in analysis_df.columns
    ]

    display_df = (
        analysis_df
        .sort_values(
            by="score",
            ascending=False,
        )[available_cols]
        .copy()
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.html(
        "</div>"
    )


# ============================================================
# ABOUT
# ============================================================

with st.sidebar.expander(
    "About & Pipeline"
):

    st.html(
        f"""
**Current Analysis**

- **Subreddit:** r/{SUBREDDIT}
- **Days:** {days_back}
- **Posts fetched:** {posts_to_fetch}
- **Posts analyzed:** {top_posts}

**Models**

- **Sentiment:** `{SENTIMENT_MODEL}`
- **Embeddings:** `{EMBEDDING_MODEL}`
- **AI:** `{LLM_MODEL}`

**Pipeline**

Reddit → Sentiment → Embeddings →
Clustering → Topic Keywords →
GPT-OSS Topic Analysis → Community Review
"""
    )
