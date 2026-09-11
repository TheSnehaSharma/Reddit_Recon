
import html
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# ============================================================
# CONFIG — deliberately small for Streamlit Community Cloud
# ============================================================

ARCTIC_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

DEFAULT_DAYS_BACK = 1
DEFAULT_POSTS_TO_FETCH = 300
DEFAULT_TOP_POSTS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 3

RANDOM_STATE = 42
MAX_FETCH_RETRIES = 4
MAX_TEXT_CHARS = 2500
EMBEDDING_BATCH_SIZE = 32

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral": "#94a3b8",
    "negative": "#ef4444",
}

st.set_page_config(
    page_title="Reddit Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0 0 .25rem 0; font-size: 1.9rem; }
    .hero p { margin: 0; opacity: .92; font-size: .95rem; }
    .topic-card {
        border: 1px solid rgba(127,127,127,.18);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: .9rem;
        background: rgba(127,127,127,.04);
    }
    .badge {
        display: inline-block;
        padding: .15rem .65rem;
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 600;
        color: white;
        margin-right: .35rem;
    }
    .keyword-chip {
        display: inline-block;
        padding: .1rem .55rem;
        border-radius: 6px;
        background: rgba(124,58,237,.12);
        color: #7c3aed;
        font-size: .8rem;
        margin: .15rem .25rem .15rem 0;
    }
    .ai-card {
        border-left: 4px solid #7c3aed;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        background: rgba(124,58,237,.05);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SMALL HELPERS
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
    color = SENTIMENT_COLORS.get(label, "#94a3b8")
    text = label.capitalize() if pct is None else f"{label.capitalize()} {pct:.0f}%"
    return f'<span class="badge" style="background:{color};">{text}</span>'


def normalize_subreddit(value):
    value = value.strip()
    if value.lower().startswith("r/"):
        value = value[2:]
    return value.strip().replace(" ", "")


# ============================================================
# HTTP — one reusable session per process
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
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "reddit-intelligence-streamlit/1.0"})
    return session


# ============================================================
# MODELS
#
# IMPORTANT:
# sentence-transformers is intentionally NOT imported.
# MiniLM is loaded directly through transformers + torch.
# This avoids the sentence_transformers import failure while
# preserving the same all-MiniLM-L6-v2 embedding model.
# ============================================================

@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        truncation=True,
        max_length=256,
        device=-1,  # CPU on Community Cloud
    )


@st.cache_resource(show_spinner="Loading MiniLM topic model...")
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    model.eval()
    return tokenizer, model


# ============================================================
# REDDIT DATA
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
        "id,created_utc,score,num_comments,subreddit,"
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
            response = session.get(ARCTIC_URL, params=params, timeout=45)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            st.warning(f"Arctic Shift request failed: {exc}")
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
            posts.append(
                {
                    "id": post_id,
                    "title": str(item.get("title") or ""),
                    "selftext": str(item.get("selftext") or ""),
                    "score": safe_int(item.get("score", 0)),
                    "num_comments": safe_int(item.get("num_comments", 0)),
                    "created_utc": created,
                    "url": str(item.get("url") or ""),
                    "subreddit": str(item.get("subreddit") or subreddit),
                }
            )

            if oldest_timestamp is None or created < oldest_timestamp:
                oldest_timestamp = created

            if len(posts) >= posts_to_fetch:
                break

        if len(posts) >= posts_to_fetch:
            break

        if oldest_timestamp is None or oldest_timestamp <= cutoff:
            break

        before = oldest_timestamp - 1
        time.sleep(0.15)

    return pd.DataFrame(posts).drop_duplicates("id").reset_index(drop=True)


# ============================================================
# TEXT + SELECTION
# ============================================================

def clean_posts(df, min_text_length=20):
    if df.empty:
        return df.copy()

    result = df.copy()
    result["title"] = result["title"].fillna("").astype(str)
    result["selftext"] = result["selftext"].fillna("").astype(str)

    # Keep the model input bounded. Title is always retained.
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

    result["sentiment"] = [p["label"].lower().strip() for p in predictions]
    result["sentiment_confidence"] = [float(p["score"]) for p in predictions]
    return result


# ============================================================
# MINI-LM EMBEDDINGS
# ============================================================

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
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
            return_tensors="pt",
        )

        with torch.inference_mode():
            output = model(**encoded)
            pooled = mean_pool(
                output.last_hidden_state,
                encoded["attention_mask"],
            )
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        vectors.append(pooled.cpu().numpy())

    return np.vstack(vectors)


# ============================================================
# TOPIC DISCOVERY
# ============================================================

def discover_topics(df):
    if len(df) < 3:
        return df.copy(), pd.DataFrame(), None

    embeddings = encode_minilm(df["model_text"].tolist())

    # 3–8 is enough for a 100-post dashboard and is much cheaper
    # than repeatedly testing up to 10 clusters.
    max_k = min(8, len(df) - 1)
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
            labels = km.fit_predict(embeddings)

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(
                embeddings,
                labels,
                metric="cosine",
            )
            silhouette_results.append(
                {"k": k, "silhouette_score": float(score)}
            )
        except Exception:
            continue

    if not silhouette_results:
        best_k = min(5, len(df) - 1)
        final_model = KMeans(
            n_clusters=best_k,
            random_state=RANDOM_STATE,
            n_init=5,
        )
        result = df.copy()
        result["topic_id"] = final_model.fit_predict(embeddings)
        return result, pd.DataFrame(), best_k

    silhouette_df = pd.DataFrame(silhouette_results)
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
    result["topic_id"] = final_model.fit_predict(embeddings)

    return result, silhouette_df, best_k


# ============================================================
# TF-IDF KEYWORDS
# ============================================================

def extract_topic_keywords(df, top_n=10):
    if df.empty or "topic_id" not in df.columns:
        return {}

    stopwords = set(ENGLISH_STOP_WORDS)
    stopwords.update(
        {
            "reddit", "post", "posts", "people", "really",
            "just", "like", "think", "thing", "things",
            "want", "got", "get", "going", "does", "did",
            "said", "say", "know", "use", "used", "using",
        }
    )

    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(stopwords),
            max_features=4000,
            ngram_range=(1, 2),
            min_df=2,
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


# ============================================================
# ENGAGEMENT + SUMMARIES
# ============================================================

def calculate_engagement(df):
    result = df.copy()
    result["score"] = pd.to_numeric(result["score"], errors="coerce").fillna(0)
    result["num_comments"] = pd.to_numeric(
        result["num_comments"], errors="coerce"
    ).fillna(0)

    result["log_score"] = np.log1p(result["score"].clip(lower=0))
    result["log_comments"] = np.log1p(result["num_comments"].clip(lower=0))
    result["engagement"] = result["log_score"] + result["log_comments"]
    return result


def get_top_engaged_posts(df, n=10):
    columns = [
        "title", "score", "num_comments", "sentiment",
        "topic_id", "engagement", "url",
    ]
    available = [c for c in columns if c in df.columns]

    return (
        df.sort_values("engagement", ascending=False)
        [available]
        .head(n)
        .reset_index(drop=True)
    )


def sentiment_summary(df):
    if df.empty or "sentiment" not in df.columns:
        return pd.DataFrame(columns=["sentiment", "count", "percentage"])

    summary = (
        df["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )
    summary["percentage"] = summary["count"] / summary["count"].sum() * 100
    return summary


def create_topic_evidence(df, keywords, top_n_posts=5):
    evidence = []

    for topic_id in sorted(df["topic_id"].unique()):
        topic_df = df[df["topic_id"] == topic_id].copy()
        representative = (
            topic_df.sort_values(
                ["score", "num_comments"],
                ascending=False,
            )
            .head(top_n_posts)
        )

        posts = []
        for _, row in representative.iterrows():
            posts.append(
                {
                    "title": str(row.get("title", "")),
                    "text": str(row.get("selftext", ""))[:1000],
                    "score": safe_int(row.get("score", 0)),
                    "comments": safe_int(row.get("num_comments", 0)),
                }
            )

        evidence.append(
            {
                "topic_id": int(topic_id),
                "post_count": int(len(topic_df)),
                "avg_score": round(float(topic_df["score"].mean()), 2),
                "avg_comments": round(
                    float(topic_df["num_comments"].mean()), 2
                ),
                "keywords": keywords.get(int(topic_id), []),
                "representative_posts": posts,
            }
        )

    return evidence


# ============================================================
# OPTIONAL GROQ INTERPRETATION
# ============================================================

def analyze_topic_with_groq(topic_evidence, api_key):
    if not api_key:
        return {
            "topic_id": topic_evidence["topic_id"],
            "error": "GROQ_API_KEY is not configured.",
        }

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        topic_id = topic_evidence["topic_id"]

        prompt = f"""
You are analyzing Topic {topic_id} from a Reddit community.

Use ONLY the evidence below.

Interpret what people are discussing and explain the reactions visible
in the representative posts.

Rules:
1. Do not invent facts.
2. Do not claim the entire subreddit agrees.
3. The evidence is a sample.
4. Do not calculate or report sentiment percentages.
5. Explain WHY people appear supportive, negative, frustrated,
   excited, concerned, or otherwise reactive.
6. Base the explanation primarily on representative posts.
7. Use keywords only as supporting context.
8. Engagement numbers indicate attention.
9. Mention disagreement only when the evidence supports it.

Evidence:
{json.dumps(topic_evidence, indent=2)}

Return a concise interpretation.
"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=500,
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
                            "topic_id": {"type": "integer"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "main_reaction": {"type": "string"},
                            "key_concerns": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "disagreement": {"type": "string"},
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

        return json.loads(response.choices[0].message.content)

    except Exception as exc:
        return {
            "topic_id": topic_evidence["topic_id"],
            "error": str(exc),
        }


def generate_ai_topic_insights(evidence, workers=3):
    api_key = get_groq_key()

    if not api_key:
        return [
            {
                "topic_id": item["topic_id"],
                "error": "GROQ_API_KEY is not configured in Streamlit Secrets.",
            }
            for item in evidence
        ]

    if not evidence:
        return []

    results = []
    max_workers = min(workers, len(evidence))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {"topic_id": item["topic_id"], "error": str(exc)}
                )

    return sorted(results, key=lambda x: x.get("topic_id", 999))


# ============================================================
# FULL PIPELINE
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def run_nlp_pipeline(raw_df, top_posts, batch_size):
    empty = {
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
        return empty

    clean_df = clean_posts(raw_df)
    if clean_df.empty:
        return empty

    analysis_df = select_top_posts(clean_df, top_posts)
    if analysis_df.empty:
        return empty

    analysis_df = analyze_sentiment(analysis_df, batch_size)
    analysis_df, silhouette_df, best_k = discover_topics(analysis_df)
    keywords = extract_topic_keywords(analysis_df)
    analysis_df = calculate_engagement(analysis_df)

    evidence = create_topic_evidence(analysis_df, keywords)
    top_engaged = get_top_engaged_posts(analysis_df)
    sentiment = sentiment_summary(analysis_df)

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

st.sidebar.title("⚙️ Analysis Settings")

days_back = st.sidebar.slider(
    "Days to analyze",
    1, 7, DEFAULT_DAYS_BACK,
)

posts_to_fetch = st.sidebar.slider(
    "Posts to fetch",
    100, 500, DEFAULT_POSTS_TO_FETCH, 50,
)

top_posts = st.sidebar.slider(
    "Top posts for NLP",
    30, 150, DEFAULT_TOP_POSTS, 10,
)

batch_size = st.sidebar.select_slider(
    "Sentiment batch size",
    options=[8, 16, 32],
    value=DEFAULT_BATCH_SIZE,
)

workers = st.sidebar.slider(
    "AI worker threads",
    1, 4, DEFAULT_WORKERS,
)

st.sidebar.divider()

if get_groq_key():
    st.sidebar.success("Groq API key configured")
else:
    st.sidebar.warning("Groq API key not configured")

st.sidebar.caption(
    "Add GROQ_API_KEY under Streamlit Secrets to enable AI topic insights."
)


# ============================================================
# HEADER + FORM
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📊 Reddit Intelligence</h1>
        <p>
            Sentiment, semantic topic discovery, engagement analytics,
            and optional LLM interpretation for any public subreddit.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("subreddit_form"):
    subreddit_input = st.text_input(
        "Subreddit",
        placeholder="GTA6, gaming, technology, stocks, etc.",
        help="Enter a subreddit name. You can include or omit r/.",
    )
    submitted = st.form_submit_button(
        "🚀 Analyze subreddit",
        type="primary",
        use_container_width=True,
    )

if submitted:
    subreddit = normalize_subreddit(subreddit_input)

    if not subreddit:
        st.error("Enter a subreddit name.")
        st.stop()

    if not subreddit.replace("_", "").isalnum():
        st.error("Use a valid subreddit name, e.g. GTA6 or r/GTA6.")
        st.stop()

    st.session_state["selected_subreddit"] = subreddit
    st.session_state.pop("result", None)
    st.session_state.pop("ai_results", None)


if "selected_subreddit" not in st.session_state:
    with st.expander("📋 Methodology", expanded=False):
        st.markdown(
            """
            **Pipeline**

            Reddit posts → text cleaning → top posts by score →
            RoBERTa sentiment → MiniLM embeddings → K-Means topic
            discovery → TF-IDF keywords → representative posts →
            optional Groq interpretation.

            **Responsibilities**

            - **RoBERTa** determines sentiment.
            - **MiniLM + K-Means** discovers semantic topics.
            - **TF-IDF** identifies topic keywords.
            - **Engagement metrics** measure attention.
            - **Groq LLM** interprets topics from representative evidence.

            Results describe the high-engagement sample selected from the
            fetched posts, not the entire subreddit.
            """
        )

    st.info("Enter a subreddit above and click **Analyze subreddit**.")
    st.stop()


SUBREDDIT = st.session_state["selected_subreddit"]

st.caption(
    f"Analyzing **r/{SUBREDDIT}** • "
    f"last {days_back} day{'s' if days_back != 1 else ''} • "
    f"top {top_posts} posts for NLP"
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if submitted:
    progress = st.progress(0)
    status = st.empty()

    status.write("Fetching recent Reddit posts...")

    raw_df = fetch_reddit_posts(
        subreddit=SUBREDDIT,
        posts_to_fetch=posts_to_fetch,
        days_back=days_back,
    )
    progress.progress(0.35)

    if raw_df.empty:
        progress.empty()
        status.empty()
        st.error(
            "No posts were fetched. Check the subreddit name or try again later."
        )
        st.stop()

    status.write(
        f"Fetched {len(raw_df):,} posts. Running NLP pipeline..."
    )

    result = run_nlp_pipeline(
        raw_df=raw_df,
        top_posts=top_posts,
        batch_size=batch_size,
    )
    progress.progress(1.0)
    progress.empty()
    status.empty()

    result["raw"] = raw_df
    st.session_state["result"] = result

    st.success(
        f"Analyzed r/{SUBREDDIT}: "
        f"{len(result['analysis']):,} high-engagement posts."
    )


# ============================================================
# RESULT
# ============================================================

if "result" not in st.session_state:
    st.info("Click **Analyze subreddit** to run the pipeline.")
    st.stop()

result = st.session_state["result"]
raw_df = result["raw"]
analysis_df = result["analysis"]
silhouette_df = result["silhouette"]
best_k = result["best_k"]
keywords = result["keywords"]
evidence = result["evidence"]
top_engaged = result["top_engaged"]
sentiment = result["sentiment"]

if analysis_df.empty:
    st.error("No usable posts were available for NLP analysis.")
    st.stop()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🧠 Topics", "🔥 Engagement", "🤖 AI Insights", "📁 Data"]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab1:
    st.subheader("Community Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Posts fetched", f"{len(raw_df):,}")
    col2.metric("Posts analyzed", f"{len(analysis_df):,}")
    col3.metric(
        "Topics discovered",
        best_k if best_k is not None else "N/A",
    )

    positive_pct = 0.0
    if not sentiment.empty:
        positive_pct = float(
            sentiment.loc[
                sentiment["sentiment"] == "positive",
                "percentage",
            ].sum()
        )
    col4.metric("Positive sentiment", f"{positive_pct:.1f}%")

    st.divider()
    st.subheader("Sentiment Distribution")

    if not sentiment.empty:
        sentiment_plot = sentiment.sort_values(
            "percentage", ascending=False
        )

        fig = px.bar(
            sentiment_plot,
            x="sentiment",
            y="percentage",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            text=sentiment_plot["percentage"].round(1).astype(str) + "%",
            labels={
                "sentiment": "Sentiment",
                "percentage": "Percentage",
            },
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if not silhouette_df.empty:
        st.subheader("K-Means Model Selection")

        fig2 = px.line(
            silhouette_df,
            x="k",
            y="silhouette_score",
            markers=True,
            labels={
                "k": "Number of Topics",
                "silhouette_score": "Silhouette Score",
            },
        )

        if best_k is not None:
            fig2.add_vline(x=best_k, line_dash="dash")

        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            f"Selected k: **{best_k}** based on the highest silhouette score."
        )


# ============================================================
# TOPICS
# ============================================================

with tab2:
    st.subheader("Semantic Topics")

    if best_k:
        st.caption(f"K-Means discovered {best_k} semantic topic clusters.")

    for topic_id in sorted(analysis_df["topic_id"].unique()):
        topic_df = analysis_df[analysis_df["topic_id"] == topic_id].copy()

        topic_sentiment = (
            topic_df["sentiment"].value_counts(normalize=True) * 100
        )
        topic_keywords = keywords.get(int(topic_id), [])

        badges = "".join(
            sentiment_badge(label, pct)
            for label, pct in topic_sentiment.sort_values(
                ascending=False
            ).items()
        )

        chips = "".join(
            f'<span class="keyword-chip">{html.escape(str(kw))}</span>'
            for kw in topic_keywords
        )

        representative = (
            topic_df.sort_values(
                ["score", "num_comments"],
                ascending=False,
            )
            .head(3)
        )

        representative_html = "".join(
            f"<li><b>{html.escape(str(row['title']))}</b> — "
            f"{safe_int(row['score']):,} points, "
            f"{safe_int(row['num_comments']):,} comments</li>"
            for _, row in representative.iterrows()
        )

        st.markdown(
            f"""
            <div class="topic-card">
                <h4>Topic {topic_id} &nbsp; {badges}</h4>
                <p style="opacity:.75;">
                    {len(topic_df)} posts ·
                    avg score {topic_df['score'].mean():,.0f} ·
                    avg comments {topic_df['num_comments'].mean():,.0f}
                </p>
                <div style="margin-bottom:.6rem;">{chips}</div>
                <b>Representative posts:</b>
                <ul>{representative_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ENGAGEMENT
# ============================================================

with tab3:
    st.subheader("Engagement Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Highest score", f"{analysis_df['score'].max():,}")
    col2.metric(
        "Highest comments",
        f"{analysis_df['num_comments'].max():,}",
    )
    col3.metric(
        "Average score",
        f"{analysis_df['score'].mean():,.0f}",
    )

    st.divider()
    st.subheader("Most Engaged Posts")

    display_df = top_engaged.copy()
    if "title" in display_df.columns:
        display_df["title"] = display_df["title"].astype(str).str.slice(0, 100)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Score vs Comments")

    fig3 = px.scatter(
        analysis_df,
        x="score",
        y="num_comments",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        hover_data=["title", "topic_id"],
        opacity=0.75,
        labels={
            "score": "Reddit Score",
            "num_comments": "Number of Comments",
        },
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Topic Engagement")

    topic_engagement = (
        analysis_df.groupby("topic_id")
        .agg(
            posts=("id", "count"),
            avg_score=("score", "mean"),
            avg_comments=("num_comments", "mean"),
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
            "avg_score": "Average Score",
            "avg_comments": "Average Comments",
            "topic_id": "Topic",
        },
    )
    st.plotly_chart(fig4, use_container_width=True)


# ============================================================
# AI INSIGHTS
# ============================================================

with tab4:
    st.subheader("🤖 AI Topic Insights")

    st.caption(
        "The LLM is an interpretation layer. It receives topic keywords, "
        "engagement statistics, and representative posts. It does not "
        "calculate sentiment or discover clusters."
    )

    if not get_groq_key():
        st.warning(
            "Groq is not configured. Add GROQ_API_KEY under Streamlit Secrets."
        )
    else:
        if st.button(
            "✨ Generate AI Insights",
            type="primary",
            key="generate_ai",
        ):
            with st.spinner("Generating AI topic insights..."):
                st.session_state["ai_results"] = (
                    generate_ai_topic_insights(evidence, workers)
                )

    if "ai_results" in st.session_state:
        for item in st.session_state["ai_results"]:
            st.divider()

            if "error" in item:
                st.error(
                    f"Topic {item.get('topic_id')}: {item['error']}"
                )
                continue

            topic_id = item.get("topic_id", "N/A")
            name = item.get("name", "Unnamed Topic")
            description = item.get("description", "")
            reaction = item.get("main_reaction", "N/A")
            concerns = item.get("key_concerns", [])
            disagreement = item.get("disagreement", "N/A")

            st.markdown(
                f"""
                <div class="ai-card">
                    <h3>Topic {topic_id}: {html.escape(str(name))}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**What this topic is about**")
            st.write(description)

            st.markdown("**Main reaction and why**")
            st.write(reaction)

            if concerns:
                st.markdown("**Key concerns / reasons**")
                for concern in concerns:
                    st.markdown(f"- {concern}")

            st.markdown("**Disagreement**")
            st.write(disagreement)


# ============================================================
# DATA
# ============================================================

with tab5:
    st.subheader("Analysis Dataset")

    st.dataframe(
        analysis_df,
        use_container_width=True,
        height=500,
        hide_index=True,
    )

    st.divider()
    st.subheader("Topic Evidence Sent to LLM")

    evidence_json = json.dumps(
        {
            "subreddit": SUBREDDIT,
            "posts_fetched": len(raw_df),
            "posts_analyzed": len(analysis_df),
            "best_k": best_k,
            "topics": evidence,
        },
        indent=2,
    )

    st.code(evidence_json, language="json")

    safe_name = SUBREDDIT.lower().replace(" ", "_")

    st.download_button(
        "⬇️ Download Evidence JSON",
        data=evidence_json,
        file_name=f"{safe_name}_evidence.json",
        mime="application/json",
        use_container_width=True,
    )

    analysis_csv = analysis_df.to_csv(index=False)

    st.download_button(
        "⬇️ Download Analysis CSV",
        data=analysis_csv,
        file_name=f"{safe_name}_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.divider()
st.caption(
    f"Reddit Intelligence • r/{SUBREDDIT} • "
    "RoBERTa + MiniLM + K-Means + TF-IDF + optional Groq"
)
st.caption(
    "Results describe a high-engagement sample of fetched posts, "
    "not the entire subreddit."
)
