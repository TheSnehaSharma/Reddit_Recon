%%writefile app.py

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score
from transformers import pipeline


# ============================================================
# CONFIG
# ============================================================

SUBREDDIT = "GTA6"

ARCTIC_URL = (
    "https://arctic-shift.photon-reddit.com/api/posts/search"
)

SENTIMENT_MODEL = (
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "openai/gpt-oss-20b"

DEFAULT_DAYS_BACK = 7
DEFAULT_POSTS_TO_FETCH = 1000
DEFAULT_TOP_POSTS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_WORKERS = 3

RANDOM_STATE = 42
MAX_FETCH_RETRIES = 5

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral": "#94a3b8",
    "negative": "#ef4444",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GTA6 Reddit Intelligence",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
<style>

.hero {
    padding: 1.75rem 2rem;
    border-radius: 16px;
    background: linear-gradient(
        135deg,
        #7c3aed 0%,
        #db2777 100%
    );
    color: white;
    margin-bottom: 1.25rem;
}

.hero h1 {
    margin: 0 0 0.25rem 0;
    font-size: 1.9rem;
}

.hero p {
    margin: 0;
    opacity: 0.92;
    font-size: 0.95rem;
}

.topic-card {
    border: 1px solid rgba(127, 127, 127, 0.18);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    background: rgba(127, 127, 127, 0.04);
}

.badge {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: white;
    margin-right: 0.4rem;
}

.keyword-chip {
    display: inline-block;
    padding: 0.1rem 0.55rem;
    border-radius: 6px;
    background: rgba(124, 58, 237, 0.12);
    color: #7c3aed;
    font-size: 0.8rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
}

.ai-card {
    border-left: 4px solid #7c3aed;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    background: rgba(124, 58, 237, 0.05);
    border-radius: 8px;
}

</style>
"""

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sentiment_badge(label, pct=None):
    color = SENTIMENT_COLORS.get(
        label,
        "#94a3b8"
    )

    if pct is None:
        text = label.capitalize()
    else:
        text = f"{label.capitalize()} {pct:.0f}%"

    return (
        f'<span class="badge" '
        f'style="background:{color};">'
        f'{text}'
        f'</span>'
    )


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return 0


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


# ============================================================
# SECRETS
# ============================================================

def get_groq_key():
    """
    Streamlit Community Cloud:

    App Settings
    -> Secrets

    Add:

    GROQ_API_KEY = "your-key"
    """

    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(
    show_spinner="Loading sentiment model..."
)
def load_sentiment_model():

    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        truncation=True,
        max_length=512,
    )


@st.cache_resource(
    show_spinner="Loading embedding model..."
)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# ARCTIC SHIFT DATA COLLECTION
# ============================================================

def fetch_reddit_posts(
    subreddit,
    posts_to_fetch,
    days_back,
    progress_cb=None,
):
    """
    Fetch recent Reddit posts through Arctic Shift.

    Arctic Shift pagination is handled by moving the
    `before` timestamp backward through the requested
    time window.
    """

    now = int(time.time())

    cutoff = int(
        now - days_back * 24 * 60 * 60
    )

    before = now

    posts = []
    seen_ids = set()

    page_size = 100

    max_pages = int(
        np.ceil(
            posts_to_fetch / page_size
        )
    )

    for page in range(max_pages):

        params = {
            "subreddit": subreddit,
            "after": cutoff,
            "before": before,
            "limit": page_size,
            "sort": "desc",
            "over_18": "false",
            "fields": (
                "id,"
                "created_utc,"
                "score,"
                "num_comments,"
                "subreddit,"
                "title,"
                "selftext,"
                "url"
            ),
        }

        payload = None

        for attempt in range(
            MAX_FETCH_RETRIES
        ):

            try:

                response = requests.get(
                    ARCTIC_URL,
                    params=params,
                    timeout=60,
                )

                if response.status_code in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):

                    wait = min(
                        2 ** attempt,
                        30,
                    )

                    if progress_cb:
                        progress_cb(
                            len(posts)
                            / max(
                                posts_to_fetch,
                                1,
                            ),
                            (
                                f"Arctic Shift HTTP "
                                f"{response.status_code}. "
                                f"Retrying in {wait}s..."
                            ),
                        )

                    time.sleep(wait)
                    continue

                response.raise_for_status()

                payload = response.json()

                break

            except requests.RequestException as exc:

                wait = min(
                    2 ** attempt,
                    30,
                )

                if attempt == (
                    MAX_FETCH_RETRIES - 1
                ):

                    if progress_cb:
                        progress_cb(
                            1.0,
                            f"Request failed: {exc}",
                        )

                    payload = None

                else:
                    time.sleep(wait)

        if not payload:
            break

        batch = payload.get(
            "data",
            [],
        )

        if not batch:
            break

        oldest_timestamp = None

        for item in batch:

            post_id = item.get("id")

            if not post_id:
                continue

            if post_id in seen_ids:
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
                    "title": item.get(
                        "title",
                        "",
                    ),
                    "selftext": item.get(
                        "selftext",
                        "",
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
                    "url": item.get(
                        "url",
                        "",
                    ),
                    "subreddit": item.get(
                        "subreddit",
                        subreddit,
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

        if progress_cb:

            progress_cb(
                min(
                    (page + 1) / max_pages,
                    1.0,
                ),
                (
                    f"Fetched "
                    f"{len(posts):,} posts..."
                ),
            )

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

        time.sleep(0.25)

    df = pd.DataFrame(posts)

    if df.empty:
        return df

    df = (
        df
        .drop_duplicates(
            subset=["id"]
        )
        .reset_index(drop=True)
    )

    return df.head(
        posts_to_fetch
    )


# ============================================================
# TEXT CLEANING
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

    result = (
        result
        .drop_duplicates(
            subset=["id"]
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# TOP POST SELECTION
# ============================================================

def select_top_posts(
    df,
    top_posts,
):

    if df.empty:
        return df.copy()

    result = (
        df
        .sort_values(
            [
                "score",
                "num_comments",
            ],
            ascending=False,
        )
        .head(
            min(
                top_posts,
                len(df),
            )
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# SENTIMENT
# ============================================================

def analyze_sentiment(
    df,
    batch_size,
):

    if df.empty:
        return df.copy()

    model = load_sentiment_model()

    result = df.copy()

    texts = (
        result["text"]
        .tolist()
    )

    labels = []
    confidences = []

    for start in range(
        0,
        len(texts),
        batch_size,
    ):

        batch = texts[
            start:start + batch_size
        ]

        predictions = model(
            batch,
            batch_size=batch_size,
        )

        for prediction in predictions:

            labels.append(
                prediction["label"]
                .lower()
                .strip()
            )

            confidences.append(
                float(
                    prediction["score"]
                )
            )

    result["sentiment"] = labels

    result["sentiment_confidence"] = (
        confidences
    )

    return result


# ============================================================
# TOPIC DISCOVERY
# ============================================================

def discover_topics(
    df,
    batch_size,
):

    if len(df) < 3:

        return (
            df.copy(),
            pd.DataFrame(),
            None,
            None,
        )

    model = load_embedding_model()

    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    max_k = min(
        10,
        len(df) - 1,
    )

    if max_k < 3:

        return (
            df.copy(),
            pd.DataFrame(),
            None,
            embeddings,
        )

    silhouette_results = []

    for k in range(
        3,
        max_k + 1,
    ):

        try:

            kmeans = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=10,
            )

            labels = (
                kmeans.fit_predict(
                    embeddings
                )
            )

            # Silhouette requires at least
            # two actual clusters.
            if len(
                np.unique(labels)
            ) < 2:
                continue

            score = silhouette_score(
                embeddings,
                labels,
            )

            silhouette_results.append(
                {
                    "k": k,
                    "silhouette_score": score,
                }
            )

        except Exception:
            continue

    if not silhouette_results:

        fallback_k = min(
            5,
            len(df) - 1,
        )

        model = KMeans(
            n_clusters=fallback_k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        result = df.copy()

        result["topic_id"] = (
            model.fit_predict(
                embeddings
            )
        )

        return (
            result,
            pd.DataFrame(),
            fallback_k,
            embeddings,
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
        n_init=10,
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
        embeddings,
    )


# ============================================================
# TF-IDF TOPIC KEYWORDS
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

    stopwords = set(
        ENGLISH_STOP_WORDS
    )

    stopwords.update(
        {
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
    )

    try:

        vectorizer = (
            TfidfVectorizer(
                stop_words=list(
                    stopwords
                ),
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
            )
        )

        matrix = (
            vectorizer.fit_transform(
                df["text"]
            )
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

        if len(indexes) == 0:
            continue

        topic_scores = (
            matrix[indexes]
            .mean(axis=0)
            .A1
        )

        top_indexes = (
            topic_scores
            .argsort()[::-1]
            [:top_n]
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

    result["score"] = pd.to_numeric(
        result["score"],
        errors="coerce",
    ).fillna(0)

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

    result["log_comments"] = (
        np.log1p(
            result[
                "num_comments"
            ].clip(
                lower=0
            )
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
        "engagement",
        "permalink",
    ]

    # permalink isn't available from
    # Arctic Shift fields, so use url instead.
    if "permalink" not in df.columns:
        columns = [
            c
            for c in columns
            if c != "permalink"
        ]

    available = [
        c
        for c in columns
        if c in df.columns
    ]

    return (
        df
        .sort_values(
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
# LLM EVIDENCE
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

        topic_df = (
            df[
                df["topic_id"]
                == topic_id
            ]
            .copy()
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
                    )[:1200],
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


# ============================================================
# GROQ
# ============================================================

def analyze_topic_with_groq(
    topic_evidence,
    api_key,
):

    if not api_key:

        return {
            "topic_id": topic_evidence[
                "topic_id"
            ],
            "error": (
                "GROQ_API_KEY is not configured."
            ),
        }

    try:

        from groq import Groq

        client = Groq(
            api_key=api_key
        )

        topic_id = topic_evidence[
            "topic_id"
        ]

        prompt = f"""
You are analyzing Topic {topic_id}
from a Reddit community.

Use ONLY the evidence below.

Your task is to interpret what people
are discussing and explain the reactions
visible in the representative posts.

Important rules:

1. Do not invent facts.
2. Do not claim the entire subreddit agrees.
3. The evidence is a sample.
4. Do not calculate sentiment percentages.
5. Do not report sentiment percentages.
6. Explain WHY people appear supportive,
   negative, frustrated, excited, concerned,
   or otherwise reactive.
7. Base the explanation primarily on the
   actual representative posts.
8. Use keywords as supporting context.
9. Engagement numbers indicate which topics
   or posts attracted attention.
10. Mention disagreement only when the
    representative posts actually support it.

Evidence:

{json.dumps(topic_evidence, indent=2)}

Return a concise interpretation.
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
            reasoning_effort="low",
            include_reasoning=False,
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
                            "key_concerns": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "disagreement": {
                                "type": "string"
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
            "topic_id": topic_evidence[
                "topic_id"
            ],
            "error": str(exc),
        }


def generate_ai_topic_insights(
    evidence,
    workers=3,
):

    api_key = get_groq_key()

    if not api_key:

        return [
            {
                "topic_id": item[
                    "topic_id"
                ],
                "error": (
                    "GROQ_API_KEY is not "
                    "configured in Streamlit Secrets."
                ),
            }
            for item in evidence
        ]

    if not evidence:
        return []

    results = []

    max_workers = min(
        workers,
        len(evidence),
    )

    with ThreadPoolExecutor(
        max_workers=max_workers
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

            item = futures[
                future
            ]

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
                        "error": str(exc),
                    }
                )

    return sorted(
        results,
        key=lambda x: x.get(
            "topic_id",
            999,
        ),
    )


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
    batch_size,
):

    empty_result = {
        "clean": pd.DataFrame(),
        "analysis": pd.DataFrame(),
        "silhouette": pd.DataFrame(),
        "best_k": None,
        "keywords": {},
        "evidence": [],
        "top_engaged": pd.DataFrame(),
        "sentiment": pd.DataFrame(),
    }

    if raw_df.empty:
        return empty_result

    clean_df = clean_posts(
        raw_df
    )

    if clean_df.empty:
        return empty_result

    analysis_df = select_top_posts(
        clean_df,
        top_posts,
    )

    if analysis_df.empty:
        return empty_result

    analysis_df = analyze_sentiment(
        analysis_df,
        batch_size,
    )

    (
        analysis_df,
        silhouette_df,
        best_k,
        _,
    ) = discover_topics(
        analysis_df,
        batch_size,
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
        top_n_posts=5,
    )

    top_engaged = (
        get_top_engaged_posts(
            analysis_df,
            n=10,
        )
    )

    sentiment = sentiment_summary(
        analysis_df
    )

    return {
        "clean": clean_df,
        "analysis": analysis_df,
        "silhouette": silhouette_df,
        "best_k": best_k,
        "keywords": keywords,
        "evidence": evidence,
        "top_engaged": top_engaged,
        "sentiment": sentiment,
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Analysis Settings"
)

days_back = st.sidebar.slider(
    "Days to analyze",
    min_value=1,
    max_value=30,
    value=DEFAULT_DAYS_BACK,
)

posts_to_fetch = st.sidebar.slider(
    "Posts to fetch",
    min_value=100,
    max_value=2000,
    value=DEFAULT_POSTS_TO_FETCH,
    step=100,
)

top_posts = st.sidebar.slider(
    "Top posts for NLP",
    min_value=30,
    max_value=200,
    value=DEFAULT_TOP_POSTS,
    step=10,
)

batch_size = st.sidebar.select_slider(
    "NLP batch size",
    options=[
        8,
        16,
        32,
        64,
    ],
    value=DEFAULT_BATCH_SIZE,
)

workers = st.sidebar.slider(
    "AI worker threads",
    min_value=1,
    max_value=6,
    value=DEFAULT_WORKERS,
)

st.sidebar.divider()

if get_groq_key():

    st.sidebar.success(
        "Groq API key configured"
    )

else:

    st.sidebar.warning(
        "Groq API key not configured"
    )

st.sidebar.caption(
    "Add GROQ_API_KEY under "
    "Streamlit Secrets to enable "
    "AI topic insights."
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎮 GTA6 Reddit Intelligence</h1>
        <p>
            NLP, semantic topic discovery,
            sentiment analysis, engagement
            analytics, and LLM-assisted
            interpretation of r/GTA6.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "📋 Methodology",
    expanded=False,
):

    st.markdown(
        """
        **Pipeline**

        Reddit posts → text cleaning → top
        posts by score → RoBERTa sentiment →
        sentence embeddings → K-Means topic
        discovery → TF-IDF keywords →
        representative posts → optional Groq
        interpretation.

        **Important separation of responsibilities**

        - **RoBERTa** determines sentiment.
        - **MiniLM embeddings + K-Means** discover
          semantic topics.
        - **TF-IDF** identifies topic keywords.
        - **Engagement metrics** measure attention.
        - **LLM** interprets the topics and explains
          reactions using representative posts.

        The LLM does not determine sentiment or
        clustering.

        Results describe the high-engagement sample
        selected from the fetched posts, not the
        entire subreddit.
        """
    )


# ============================================================
# RUN ANALYSIS
# ============================================================

run_analysis = st.button(
    "🚀 Run Analysis",
    type="primary",
    use_container_width=True,
)

if run_analysis:

    # Clear previous AI results
    st.session_state.pop(
        "ai_results",
        None,
    )

    progress = st.progress(
        0.0
    )

    status = st.empty()

    def progress_callback(
        fraction,
        message,
    ):

        progress.progress(
            min(
                max(
                    fraction,
                    0.0,
                ),
                1.0,
            )
        )

        status.write(
            message
        )

    with st.spinner(
        "Fetching Reddit posts from Arctic Shift..."
    ):

        raw_df = fetch_reddit_posts(
            subreddit=SUBREDDIT,
            posts_to_fetch=posts_to_fetch,
            days_back=days_back,
            progress_cb=progress_callback,
        )

    progress.empty()
    status.empty()

    if raw_df.empty:

        st.error(
            "No posts were fetched. "
            "Try increasing the date range "
            "or try again later."
        )

        st.session_state.pop(
            "result",
            None,
        )

    else:

        st.success(
            f"Fetched {len(raw_df):,} posts."
        )

        with st.spinner(
            "Running sentiment, semantic clustering, "
            "and topic extraction..."
        ):

            result = run_nlp_pipeline(
                raw_df=raw_df,
                top_posts=top_posts,
                batch_size=batch_size,
            )

        result["raw"] = raw_df

        st.session_state[
            "result"
        ] = result


# ============================================================
# CHECK RESULT
# ============================================================

if "result" not in st.session_state:

    st.info(
        "Configure the analysis settings and "
        "click **Run Analysis** to begin."
    )

    st.stop()


result = st.session_state[
    "result"
]

raw_df = result["raw"]

analysis_df = result[
    "analysis"
]

silhouette_df = result[
    "silhouette"
]

best_k = result[
    "best_k"
]

keywords = result[
    "keywords"
]

evidence = result[
    "evidence"
]

top_engaged = result[
    "top_engaged"
]

sentiment = result[
    "sentiment"
]


if analysis_df.empty:

    st.error(
        "No usable posts were available "
        "for NLP analysis."
    )

    st.stop()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview",
        "🧠 Topics",
        "🔥 Engagement",
        "🤖 AI Insights",
        "📁 Data",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.subheader(
        "Community Overview"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    col1.metric(
        "Posts fetched",
        f"{len(raw_df):,}",
    )

    col2.metric(
        "Posts analyzed",
        f"{len(analysis_df):,}",
    )

    col3.metric(
        "Topics discovered",
        best_k
        if best_k is not None
        else "N/A",
    )

    positive_pct = 0

    if not sentiment.empty:

        positive_pct = (
            sentiment.loc[
                sentiment["sentiment"]
                == "positive",
                "percentage",
            ].sum()
        )

    col4.metric(
        "Positive sentiment",
        f"{positive_pct:.1f}%",
    )

    st.divider()

    st.subheader(
        "Sentiment Distribution"
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
            color_discrete_map=(
                SENTIMENT_COLORS
            ),
            text=(
                sentiment_plot[
                    "percentage"
                ]
                .round(1)
                .astype(str)
                + "%"
            ),
            labels={
                "sentiment": "Sentiment",
                "percentage": "Percentage",
            },
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    if not silhouette_df.empty:

        st.subheader(
            "K-Means Model Selection"
        )

        fig2 = px.line(
            silhouette_df,
            x="k",
            y="silhouette_score",
            markers=True,
            labels={
                "k": "Number of Topics",
                "silhouette_score":
                    "Silhouette Score",
            },
        )

        if best_k is not None:

            fig2.add_vline(
                x=best_k,
                line_dash="dash",
            )

        st.plotly_chart(
            fig2,
            use_container_width=True,
        )

        st.caption(
            f"Selected k: **{best_k}** "
            "based on the highest silhouette score."
        )


# ============================================================
# TAB 2 — TOPICS
# ============================================================

with tab2:

    st.subheader(
        "Semantic Topics"
    )

    if best_k:

        st.caption(
            f"K-Means discovered {best_k} "
            "semantic topic clusters."
        )

    for topic_id in sorted(
        analysis_df[
            "topic_id"
        ].unique()
    ):

        topic_df = (
            analysis_df[
                analysis_df[
                    "topic_id"
                ]
                == topic_id
            ]
            .copy()
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

        topic_keywords = keywords.get(
            int(topic_id),
            [],
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
            f'<span class="keyword-chip">'
            f'{kw}'
            f'</span>'
            for kw in topic_keywords
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
            .head(3)
        )

        representative_html = ""

        for _, row in (
            representative.iterrows()
        ):

            title = (
                str(
                    row["title"]
                )
                .replace(
                    "<",
                    "&lt;",
                )
                .replace(
                    ">",
                    "&gt;",
                )
            )

            representative_html += (
                f"<li>"
                f"<b>{title}</b>"
                f" — "
                f"{safe_int(row['score']):,} "
                f"points, "
                f"{safe_int(row['num_comments']):,} "
                f"comments"
                f"</li>"
            )

        st.markdown(
            f"""
            <div class="topic-card">

                <h4>
                    Topic {topic_id}
                    &nbsp;
                    {badges}
                </h4>

                <p style="opacity:0.75;">
                    {len(topic_df)} posts
                    · avg score
                    {topic_df['score'].mean():,.0f}
                    · avg comments
                    {topic_df['num_comments'].mean():,.0f}
                </p>

                <div style="margin-bottom:0.6rem;">
                    {chips}
                </div>

                <b>Representative posts:</b>

                <ul>
                    {representative_html}
                </ul>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 3 — ENGAGEMENT
# ============================================================

with tab3:

    st.subheader(
        "Engagement Analysis"
    )

    col1, col2, col3 = st.columns(
        3
    )

    col1.metric(
        "Highest score",
        f"{analysis_df['score'].max():,}",
    )

    col2.metric(
        "Highest comments",
        f"{analysis_df['num_comments'].max():,}",
    )

    col3.metric(
        "Average score",
        f"{analysis_df['score'].mean():,.0f}",
    )

    st.divider()

    st.subheader(
        "Most Engaged Posts"
    )

    display_df = (
        top_engaged.copy()
    )

    if "title" in display_df.columns:

        display_df["title"] = (
            display_df["title"]
            .astype(str)
            .str.slice(
                0,
                100,
            )
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Score vs Comments"
    )

    fig3 = px.scatter(
        analysis_df,
        x="score",
        y="num_comments",
        color="sentiment",
        color_discrete_map=(
            SENTIMENT_COLORS
        ),
        hover_data=[
            "title",
            "topic_id",
        ],
        opacity=0.75,
        labels={
            "score": "Reddit Score",
            "num_comments":
                "Number of Comments",
        },
    )

    st.plotly_chart(
        fig3,
        use_container_width=True,
    )

    st.subheader(
        "Topic Engagement"
    )

    topic_engagement = (
        analysis_df
        .groupby("topic_id")
        .agg(
            posts=("id", "count"),
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

    fig4 = px.scatter(
        topic_engagement,
        x="avg_score",
        y="avg_comments",
        size="posts",
        hover_name="topic_id",
        labels={
            "avg_score":
                "Average Score",
            "avg_comments":
                "Average Comments",
            "topic_id":
                "Topic",
        },
    )

    st.plotly_chart(
        fig4,
        use_container_width=True,
    )


# ============================================================
# TAB 4 — AI INSIGHTS
# ============================================================

with tab4:

    st.subheader(
        "🤖 AI Topic Insights"
    )

    st.caption(
        "The LLM is an interpretation layer. "
        "It receives topic keywords, engagement "
        "statistics, and the top 5 representative "
        "posts for each topic. It does not "
        "calculate sentiment or discover clusters."
    )

    if not get_groq_key():

        st.warning(
            "Groq is not configured. Add "
            "`GROQ_API_KEY` under Streamlit Secrets."
        )

    else:

        if st.button(
            "✨ Generate AI Insights",
            type="primary",
        ):

            with st.spinner(
                "Generating AI topic insights..."
            ):

                ai_results = (
                    generate_ai_topic_insights(
                        evidence,
                        workers=workers,
                    )
                )

                st.session_state[
                    "ai_results"
                ] = ai_results

    if (
        "ai_results"
        in st.session_state
    ):

        ai_results = st.session_state[
            "ai_results"
        ]

        if not ai_results:

            st.info(
                "No AI topic results were generated."
            )

        for item in ai_results:

            st.divider()

            if "error" in item:

                st.error(
                    f"Topic "
                    f"{item.get('topic_id')}: "
                    f"{item['error']}"
                )

                continue

            topic_id = item.get(
                "topic_id",
                "N/A",
            )

            name = item.get(
                "name",
                "Unnamed Topic",
            )

            description = item.get(
                "description",
                "",
            )

            reaction = item.get(
                "main_reaction",
                "N/A",
            )

            concerns = item.get(
                "key_concerns",
                [],
            )

            disagreement = item.get(
                "disagreement",
                "N/A",
            )

            st.markdown(
                f"""
                <div class="ai-card">
                    <h3>
                        Topic {topic_id}:
                        {name}
                    </h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "**What this topic is about**"
            )

            st.write(
                description
            )

            st.markdown(
                "**Main reaction and why**"
            )

            st.write(
                reaction
            )

            if concerns:

                st.markdown(
                    "**Key concerns / reasons**"
                )

                for concern in concerns:

                    st.markdown(
                        f"- {concern}"
                    )

            st.markdown(
                "**Disagreement**"
            )

            st.write(
                disagreement
            )


# ============================================================
# TAB 5 — DATA
# ============================================================

with tab5:

    st.subheader(
        "Analysis Dataset"
    )

    st.dataframe(
        analysis_df,
        use_container_width=True,
        height=500,
        hide_index=True,
    )

    st.divider()

    st.subheader(
        "Topic Evidence Sent to LLM"
    )

    evidence_json = json.dumps(
        {
            "subreddit": SUBREDDIT,
            "posts_fetched": len(
                raw_df
            ),
            "posts_analyzed": len(
                analysis_df
            ),
            "best_k": best_k,
            "topics": evidence,
        },
        indent=2,
    )

    st.code(
        evidence_json,
        language="json",
    )

    st.download_button(
        "⬇️ Download Evidence JSON",
        data=evidence_json,
        file_name=(
            "gta6_evidence.json"
        ),
        mime="application/json",
        use_container_width=True,
    )

    analysis_csv = (
        analysis_df
        .to_csv(
            index=False
        )
    )

    st.download_button(
        "⬇️ Download Analysis CSV",
        data=analysis_csv,
        file_name=(
            "gta6_analysis.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GTA6 Reddit Intelligence • "
    "NLP + Semantic Clustering + "
    "Engagement Analytics + "
    "Optional LLM Interpretation"
)

st.caption(
    "Results describe a high-engagement "
    "sample of fetched posts, not the "
    "entire subreddit."
)
