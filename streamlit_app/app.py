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

# Removed batch and workers from UI as requested, setting sane defaults
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 3

RANDOM_STATE = 42
MAX_FETCH_RETRIES = 4
MAX_TEXT_CHARS = 2500
EMBEDDING_BATCH_SIZE = 32

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral": "#878A8C",
    "negative": "#FF4500", # Reddit orange/red for negative
}

st.set_page_config(
    page_title="Reddit Recon",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Reddit Theme Core Background */
    .stApp {
        background-color: #DAE0E6;
    }
    
    /* Header UI */
    .reddit-header {
        display: flex;
        align-items: center;
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #ccc;
        margin-bottom: 1.5rem;
    }
    .reddit-header img {
        width: 64px;
        height: 64px;
        margin-right: 1.5rem;
    }
    .reddit-header h1 {
        margin: 0;
        font-size: 2.2rem;
        color: #1c1c1c;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
    }
    .reddit-header p {
        margin: 0;
        color: #7c7c7c;
        font-size: 1rem;
    }
    
    /* Reddit-style Cards */
    .reddit-card {
        background-color: white;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Topic specific styling */
    .topic-title {
        color: #1a1a1b;
        font-weight: 600;
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
        margin-right: 0.5rem;
    }
    
    .keyword-chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        background: #F6F7F8;
        color: #0079D3;
        font-size: 0.8rem;
        margin: 0.2rem 0.2rem 0.2rem 0;
        border: 1px solid #EDEFF1;
    }
    
    hr {
        border-top: 1px solid #EDEFF1;
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
    color = SENTIMENT_COLORS.get(label, "#878A8C")
    text = label.capitalize() if pct is None else f"{label.capitalize()} {pct:.0f}%"
    return f'<span class="badge" style="background:{color};">{text}</span>'


def normalize_subreddit(value):
    value = value.strip()
    if value.lower().startswith("r/"):
        value = value[2:]
    return value.strip().replace(" ", "")


# ============================================================
# HTTP & MODELS
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
    session.headers.update({"User-Agent": "reddit-recon-ui/2.0"})
    return session


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


@st.cache_resource(show_spinner="Loading MiniLM topic model...")
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    model.eval()
    return tokenizer, model


# ============================================================
# DATA PROCESSING
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

    fields = "id,created_utc,score,num_comments,subreddit,title,selftext,url"

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
        except requests.RequestException:
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
                "subreddit": str(item.get("subreddit") or subreddit),
            })

            if oldest_timestamp is None or created < oldest_timestamp:
                oldest_timestamp = created

            if len(posts) >= posts_to_fetch:
                break

        if len(posts) >= posts_to_fetch or oldest_timestamp is None or oldest_timestamp <= cutoff:
            break

        before = oldest_timestamp - 1
        time.sleep(0.15)

    return pd.DataFrame(posts).drop_duplicates("id").reset_index(drop=True)


def clean_posts(df, min_text_length=20):
    if df.empty: return df.copy()
    result = df.copy()
    result["title"] = result["title"].fillna("").astype(str)
    result["selftext"] = result["selftext"].fillna("").astype(str)
    result["text"] = (result["title"] + " " + result["selftext"]).str.replace(r"\s+", " ", regex=True).str.strip()
    result = result[result["text"].str.len() >= min_text_length].copy()
    result["model_text"] = result["text"].str.slice(0, MAX_TEXT_CHARS)
    return result.drop_duplicates("id").reset_index(drop=True)


def select_top_posts(df, top_posts):
    if df.empty: return df.copy()
    return df.sort_values(["score", "num_comments"], ascending=False, kind="stable").head(min(top_posts, len(df))).reset_index(drop=True)


def analyze_sentiment(df, batch_size):
    if df.empty: return df.copy()
    model = load_sentiment_model()
    result = df.copy()
    predictions = model(result["model_text"].tolist(), batch_size=batch_size)
    result["sentiment"] = [p["label"].lower().strip() for p in predictions]
    result["sentiment_confidence"] = [float(p["score"]) for p in predictions]
    return result


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
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.inference_mode():
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors)


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

    silhouette_results = []
    for k in range(3, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
            labels = km.fit_predict(embeddings)
            if len(np.unique(labels)) < 2: continue
            score = silhouette_score(embeddings, labels, metric="cosine")
            silhouette_results.append({"k": k, "silhouette_score": float(score)})
        except Exception:
            continue

    if not silhouette_results:
        best_k = min(5, len(df) - 1)
        final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=5)
        result = df.copy()
        result["topic_id"] = final_model.fit_predict(embeddings)
        return result, pd.DataFrame(), best_k

    silhouette_df = pd.DataFrame(silhouette_results)
    best_k = int(silhouette_df.loc[silhouette_df["silhouette_score"].idxmax(), "k"])
    final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=5)
    result = df.copy()
    result["topic_id"] = final_model.fit_predict(embeddings)

    return result, silhouette_df, best_k


def extract_topic_keywords(df, top_n=10):
    if df.empty or "topic_id" not in df.columns: return {}
    stopwords = set(ENGLISH_STOP_WORDS)
    stopwords.update({"reddit", "post", "posts", "people", "really", "just", "like", "think", "thing", "things", "want", "got", "get", "going", "does", "did", "said", "say", "know", "use", "used", "using"})
    try:
        vectorizer = TfidfVectorizer(stop_words=list(stopwords), max_features=4000, ngram_range=(1, 2), min_df=2)
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


def calculate_engagement(df):
    result = df.copy()
    result["score"] = pd.to_numeric(result["score"], errors="coerce").fillna(0)
    result["num_comments"] = pd.to_numeric(result["num_comments"], errors="coerce").fillna(0)
    result["log_score"] = np.log1p(result["score"].clip(lower=0))
    result["log_comments"] = np.log1p(result["num_comments"].clip(lower=0))
    result["engagement"] = result["log_score"] + result["log_comments"]
    return result


def get_top_engaged_posts(df, n=10):
    columns = ["title", "score", "num_comments", "sentiment", "topic_id", "engagement", "url"]
    available = [c for c in columns if c in df.columns]
    return df.sort_values("engagement", ascending=False)[available].head(n).reset_index(drop=True)


def sentiment_summary(df):
    if df.empty or "sentiment" not in df.columns:
        return pd.DataFrame(columns=["sentiment", "count", "percentage"])
    summary = df["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    summary["percentage"] = summary["count"] / summary["count"].sum() * 100
    return summary


def create_topic_evidence(df, keywords, top_n_posts=5):
    evidence = []
    for topic_id in sorted(df["topic_id"].unique()):
        topic_df = df[df["topic_id"] == topic_id].copy()
        representative = topic_df.sort_values(["score", "num_comments"], ascending=False).head(top_n_posts)
        posts = [{"title": str(row.get("title", "")), "text": str(row.get("selftext", ""))[:1000], "score": safe_int(row.get("score", 0)), "comments": safe_int(row.get("num_comments", 0))} for _, row in representative.iterrows()]
        evidence.append({
            "topic_id": int(topic_id),
            "post_count": int(len(topic_df)),
            "avg_score": round(float(topic_df["score"].mean()), 2),
            "avg_comments": round(float(topic_df["num_comments"].mean()), 2),
            "keywords": keywords.get(int(topic_id), []),
            "representative_posts": posts,
        })
    return evidence


# ============================================================
# LLM AI INTERPRETATION
# ============================================================

def analyze_topic_with_groq(topic_evidence, api_key, model):
    if not api_key:
        return {"topic_id": topic_evidence["topic_id"], "error": "GROQ_API_KEY is not configured.", "name": f"Topic {topic_evidence['topic_id']}"}
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        topic_id = topic_evidence["topic_id"]

        prompt = f"""
        You are an expert community manager analyzing Topic {topic_id} from a Reddit community.
        Provide a concise actual name/title for this topic (e.g. "Game Performance Issues" or "Meme Culture")
        and explain what people are discussing.
        Evidence: {json.dumps(topic_evidence, indent=2)}
        """

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
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
                            "topic_id": {"type": "integer"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "main_reaction": {"type": "string"},
                        },
                        "required": ["topic_id", "name", "description", "main_reaction"],
                        "additionalProperties": False,
                    }
                }
            }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as exc:
        return {"topic_id": topic_evidence["topic_id"], "error": str(exc), "name": f"Topic {topic_evidence['topic_id']}"}


def generate_ai_topic_insights(evidence, llm_model, workers=3):
    api_key = get_groq_key()
    if not api_key or not evidence:
        return [{"topic_id": item["topic_id"], "name": f"Topic {item['topic_id']}", "error": "LLM Key Missing"} for item in evidence]

    results = []
    with ThreadPoolExecutor(max_workers=min(workers, len(evidence))) as executor:
        futures = {executor.submit(analyze_topic_with_groq, item, api_key, llm_model): item for item in evidence}
        for future in as_completed(futures):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"topic_id": item["topic_id"], "name": f"Topic {item['topic_id']}", "error": str(exc)})
    return sorted(results, key=lambda x: x.get("topic_id", 999))


def generate_overall_review(evidence, df, llm_model):
    api_key = get_groq_key()
    if not api_key: return "LLM API Key missing for overall review."
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = f"""
        Analyze the following Reddit subreddit sample evidence. 
        Provide a 2-3 paragraph summary detailing the general consensus in the subreddit, 
        the general sentiment drivers, and any niche details that stand out.
        Evidence Summary: {json.dumps(evidence, indent=2)}
        """
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=800
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"Error generating review: {str(exc)}"


# ============================================================
# PIPELINE
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def run_nlp_pipeline(raw_df, top_posts, llm_model):
    empty = {"clean": pd.DataFrame(), "analysis": pd.DataFrame(), "best_k": None, "keywords": {}, "evidence": [], "top_engaged": pd.DataFrame(), "sentiment": pd.DataFrame(), "ai_insights": [], "overall_review": ""}
    if raw_df.empty: return empty

    clean_df = clean_posts(raw_df)
    if clean_df.empty: return empty

    analysis_df = select_top_posts(clean_df, top_posts)
    if analysis_df.empty: return empty

    analysis_df = analyze_sentiment(analysis_df, DEFAULT_BATCH_SIZE)
    analysis_df, _, best_k = discover_topics(analysis_df)
    keywords = extract_topic_keywords(analysis_df)
    analysis_df = calculate_engagement(analysis_df)

    evidence = create_topic_evidence(analysis_df, keywords)
    top_engaged = get_top_engaged_posts(analysis_df)
    sentiment = sentiment_summary(analysis_df)
    
    # Auto-run AI Inference
    ai_insights = generate_ai_topic_insights(evidence, llm_model, DEFAULT_WORKERS)
    overall_review = generate_overall_review(evidence, analysis_df, llm_model)

    return {
        "clean": clean_df,
        "analysis": analysis_df,
        "best_k": best_k,
        "keywords": keywords,
        "evidence": evidence,
        "top_engaged": top_engaged,
        "sentiment": sentiment,
        "ai_insights": ai_insights,
        "overall_review": overall_review
    }


def generate_word_cloud(df):
    text = " ".join(df["title"].fillna("") + " " + df["selftext"].fillna(""))
    stopwords = set(ENGLISH_STOP_WORDS).union({"reddit", "post", "posts", "people", "just", "like", "think", "thing", "really"})
    wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=stopwords, colormap="inferno").generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image("https://www.redditinc.com/assets/images/site/reddit-logo.png", width=120)
st.sidebar.title("⚙️ Recon Settings")

with st.sidebar.form("subreddit_form"):
    subreddit_input = st.text_input("Subreddit", placeholder="e.g. technology, Python, gaming", help="Enter a subreddit without r/")
    days_back = st.slider("Days to analyze", 1, 30, 1)
    posts_to_fetch = st.slider("Posts to fetch", 100, 1000, 300, 50)
    top_posts = st.slider("Top posts for NLP", 10, 500, 100, 10)
    
    submitted = st.form_submit_button("🚀 Run Recon", type="primary", use_container_width=True)

with st.sidebar.expander("🛠️ Advanced Model Settings"):
    selected_llm = st.selectbox(
        "AI Analysis Model", 
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"], 
        index=0
    )


# Run Analysis Logic
if submitted:
    subreddit = normalize_subreddit(subreddit_input)
    if not subreddit or not subreddit.replace("_", "").isalnum():
        st.sidebar.error("Use a valid subreddit name.")
        st.stop()
        
    st.session_state["selected_subreddit"] = subreddit
    st.session_state.pop("result", None)


if "selected_subreddit" not in st.session_state:
    st.markdown(
        """
        <div class="reddit-header">
            <img src="https://www.redditstatic.com/desktop2x/img/favicon/apple-icon-120x120.png" alt="Reddit Logo">
            <div>
                <h1>Reddit Recon</h1>
                <p>Enter a subreddit in the sidebar to extract intelligence, semantic topics, and engagement metrics.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("👈 Enter a subreddit on the sidebar and click **Run Recon**.")
    st.stop()


SUBREDDIT = st.session_state["selected_subreddit"]

if submitted:
    progress = st.progress(0)
    status = st.empty()
    status.write(f"Fetching posts from r/{SUBREDDIT}...")

    raw_df = fetch_reddit_posts(SUBREDDIT, posts_to_fetch, days_back)
    progress.progress(0.4)

    if raw_df.empty:
        progress.empty(); status.empty()
        st.error("No posts fetched. Check the subreddit name.")
        st.stop()

    status.write(f"Fetched {len(raw_df):,} posts. Extracting intelligence with AI inference...")
    result = run_nlp_pipeline(raw_df, top_posts, selected_llm)
    progress.progress(1.0)
    progress.empty(); status.empty()

    result["raw"] = raw_df
    st.session_state["result"] = result
    st.toast(f"Recon complete for r/{SUBREDDIT}!")


if "result" not in st.session_state:
    st.stop()

result = st.session_state["result"]
raw_df = result["raw"]
analysis_df = result["analysis"]
best_k = result["best_k"]
keywords = result["keywords"]
top_engaged = result["top_engaged"]
sentiment = result["sentiment"]
ai_insights = result["ai_insights"]
overall_review = result["overall_review"]

if analysis_df.empty:
    st.error("No usable posts were available for NLP analysis.")
    st.stop()


# ============================================================
# MAIN CONTENT / HEADER
# ============================================================

st.markdown(
    f"""
    <div class="reddit-header">
        <img src="https://www.redditstatic.com/desktop2x/img/favicon/apple-icon-120x120.png" alt="Reddit Logo">
        <div>
            <h1>Reddit Recon: r/{SUBREDDIT}</h1>
            <p>Analyzed top {len(analysis_df)} posts from the last {days_back} days.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧠 Topics", "📝 Review", "📁 Data"])


# ============================================================
# TAB 1: OVERVIEW DASHBOARD
# ============================================================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='reddit-card'><h4>Total Fetched</h4><h2>{len(raw_df):,}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='reddit-card'><h4>NLP Analyzed</h4><h2>{len(analysis_df):,}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='reddit-card'><h4>Topics Discovered</h4><h2>{best_k if best_k else 'N/A'}</h2></div>", unsafe_allow_html=True)
    with col4:
        pos_pct = sentiment.loc[sentiment["sentiment"] == "positive", "percentage"].sum() if not sentiment.empty else 0
        st.markdown(f"<div class='reddit-card'><h4>Positivity</h4><h2>{pos_pct:.1f}%</h2></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='reddit-card'>", unsafe_allow_html=True)
        st.subheader("Sentiment Distribution")
        if not sentiment.empty:
            fig = px.pie(sentiment, values="percentage", names="sentiment", color="sentiment", 
                         color_discrete_map=SENTIMENT_COLORS, hole=0.4)
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='reddit-card'>", unsafe_allow_html=True)
        st.subheader("Topic Engagement")
        topic_eng = analysis_df.groupby("topic_id").agg(posts=("id", "count"), avg_score=("score", "mean")).reset_index()
        # Map topic IDs to inferred actual names
        topic_names_map = {item.get("topic_id"): item.get("name", f"Topic {item.get('topic_id')}") for item in ai_insights}
        topic_eng["Topic Name"] = topic_eng["topic_id"].map(topic_names_map)
        
        fig4 = px.scatter(topic_eng, x="avg_score", y="posts", size="posts", color="Topic Name",
                          labels={"avg_score": "Average Score", "posts": "Volume of Posts"})
        fig4.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 2: TOPICS (AI INFERRED)
# ============================================================
with tab2:
    if not ai_insights:
        st.warning("No AI insights generated. Please ensure LLM API is configured.")
        
    for item in ai_insights:
        topic_id = item.get("topic_id")
        ai_name = item.get("name", f"Topic {topic_id}")
        ai_desc = item.get("description", "")
        ai_reaction = item.get("main_reaction", "")
        
        topic_df = analysis_df[analysis_df["topic_id"] == topic_id].copy()
        if topic_df.empty: continue
        
        topic_sentiment = (topic_df["sentiment"].value_counts(normalize=True) * 100)
        badges = "".join(sentiment_badge(label, pct) for label, pct in topic_sentiment.sort_values(ascending=False).items())
        
        representative = topic_df.sort_values(["score", "num_comments"], ascending=False).head(5)
        rep_html = "".join(
            f"<li style='margin-bottom:0.5rem;'><a href='{row['url']}' target='_blank' style='color:#0079D3;text-decoration:none;'><b>{html.escape(str(row['title']))}</b></a> "
            f"<span style='color:#7c7c7c;font-size:0.85rem;'>(Score: {safe_int(row['score']):,} | Comments: {safe_int(row['num_comments']):,})</span></li>"
            for _, row in representative.iterrows()
        )
        
        st.markdown(
            f"""
            <div class="reddit-card">
                <div class="topic-title">{html.escape(ai_name)}</div>
                <div style="margin-bottom: 1rem;">{badges}</div>
                <p><b>Analysis:</b> {html.escape(ai_desc)}</p>
                <p><b>Reaction Context:</b> {html.escape(ai_reaction)}</p>
                <hr/>
                <p style="font-size: 0.9rem; font-weight: 600; color: #7c7c7c;">TOP POSTS IN TOPIC</p>
                <ul style="list-style-type:none; padding-left:0;">{rep_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 3: REVIEW (SUMMARY & WORDCLOUD)
# ============================================================
with tab3:
    st.markdown("<div class='reddit-card'>", unsafe_allow_html=True)
    st.subheader(f"General Consensus: r/{SUBREDDIT}")
    if overall_review:
        st.write(overall_review)
    else:
        st.info("LLM summary is not available.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='reddit-card'>", unsafe_allow_html=True)
    st.subheader("What they are talking about")
    st.caption("Common terms after removing standard stopwords.")
    wc_fig = generate_word_cloud(analysis_df)
    st.pyplot(wc_fig)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 4: DATA
# ============================================================
with tab4:
    st.markdown("<div class='reddit-card'>", unsafe_allow_html=True)
    st.subheader("Top Posts Dataset")
    display_cols = ["score", "num_comments", "title", "sentiment", "topic_id", "url"]
    display_df = analysis_df.sort_values(by="score", ascending=False)[display_cols].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SIDEBAR ABOUT EXPANDER
# ============================================================
with st.sidebar.expander("ℹ️ About & Pipeline Details"):
    st.markdown(f"""
    **Current Analysis Context**
    - **Subreddit:** r/{SUBREDDIT if 'SUBREDDIT' in locals() else 'None'}
    - **Days Fetched:** {days_back}
    - **Target Limit:** {posts_to_fetch}
    - **Analysis Limit:** {top_posts}
    
    **Models in Use:**
    - **Sentiment:** {SENTIMENT_MODEL}
    - **Embeddings:** {EMBEDDING_MODEL}
    - **LLM Selection:** {selected_llm}
    
    *This is a Data Science project designed to emulate Reddit's environment while injecting machine learning insights.*
    """)
