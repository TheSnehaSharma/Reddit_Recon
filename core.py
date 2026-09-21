from __future__ import annotations

import json
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
try:
    import streamlit as st
except ImportError:
    class _DummyCache:
        def __call__(self, *args, **kwargs):
            def deco(fn):
                return fn
            return deco
    class _DummySecrets(dict):
        pass
    class _DummyStreamlit:
        cache_resource = _DummyCache()
        cache_data = _DummyCache()
        secrets = _DummySecrets()
    st = _DummyStreamlit()
from requests.adapters import HTTPAdapter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from urllib3.util.retry import Retry

from config import (
    ARCTIC_URL, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, EMOTION_LABELS,
    EMOTION_MODEL, EXTRA_STOPWORDS, LLM_MODEL, MAX_CLUSTER_K,
    MAX_GROQ_EVIDENCE_CHARS, MAX_SILHOUETTE_SAMPLE, MAX_TEXT_CHARS,
    MODEL_MAX_LENGTH, RANDOM_STATE, SENTIMENT_MODEL, TOPIC_EVIDENCE_CHARS,
    TOPIC_EVIDENCE_POSTS, DEFAULT_BATCH_SIZE, MAX_FETCH_RETRIES,
)


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def normalize_subreddit(value: str) -> str:
    value = str(value or "").strip()
    if value.lower().startswith("r/"):
        value = value[2:]
    return value.strip().replace(" ", "")


@st.cache_resource(show_spinner=False)
def get_http_session() -> requests.Session:
    retry = Retry(
        total=MAX_FETCH_RETRIES,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
    session.headers.update({"User-Agent": "reddit-recon/5.0"})
    return session


@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
    model.eval()
    if hasattr(torch, "ao"):
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=MODEL_MAX_LENGTH,
        device=-1,
    )


@st.cache_resource(show_spinner="Loading emotion model...")
def load_emotion_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
    model.eval()
    if hasattr(torch, "ao"):
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=MODEL_MAX_LENGTH,
        top_k=None,
        device=-1,
    )


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    model.eval()
    if hasattr(torch, "ao"):
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return tokenizer, model


@st.cache_data(ttl=900, show_spinner="Fetching Reddit Data")
def fetch_reddit_posts(subreddit: str, posts_to_fetch: int, days_back: int) -> pd.DataFrame:
    now = int(time.time())
    cutoff = now - days_back * 24 * 60 * 60
    before = now
    page_size = 100
    max_pages = int(np.ceil(posts_to_fetch / page_size))
    session = get_http_session()
    fields = "id,created_utc,score,num_comments,subreddit,title,selftext,url"

    posts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

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
            response = session.get(ARCTIC_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
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
            oldest_timestamp = created if oldest_timestamp is None else min(oldest_timestamp, created)
            if len(posts) >= posts_to_fetch:
                break

        if len(posts) >= posts_to_fetch or oldest_timestamp is None or oldest_timestamp <= cutoff:
            break
        before = oldest_timestamp - 1
        time.sleep(0.1)

    if not posts:
        return pd.DataFrame()
    return pd.DataFrame(posts).drop_duplicates("id").reset_index(drop=True)


def clean_posts(df: pd.DataFrame, min_text_length: int = 20) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    result = df.copy()
    result["title"] = result["title"].fillna("").astype(str)
    result["selftext"] = result["selftext"].fillna("").astype(str)
    result["text"] = (result["title"] + " " + result["selftext"]).str.replace(r"\s+", " ", regex=True).str.strip()
    result = result[result["text"].str.len() >= min_text_length].copy()
    result["model_text"] = result["text"].str.slice(0, MAX_TEXT_CHARS)
    return result.drop_duplicates("id").reset_index(drop=True)


def select_posts_for_analysis(df: pd.DataFrame, top_posts: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values(["score", "num_comments"], ascending=False, kind="stable").head(min(top_posts, len(df))).reset_index(drop=True)


def analyze_sentiment(df: pd.DataFrame, batch_size: int = DEFAULT_BATCH_SIZE) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    predictions = load_sentiment_model()(df["model_text"].tolist(), batch_size=batch_size)
    result = df.copy()
    result["sentiment"] = [str(p["label"]).lower().strip() for p in predictions]
    result["sentiment_confidence"] = [float(p["score"]) for p in predictions]
    return result


def analyze_emotions(df: pd.DataFrame, batch_size: int = DEFAULT_BATCH_SIZE) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    predictions = load_emotion_model()(df["model_text"].tolist(), batch_size=batch_size)
    result = df.copy()
    emotion_scores = []
    labels = []
    confidences = []
    
    model_order = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    for prediction in predictions:
        if isinstance(prediction, dict):
            prediction = [prediction]
        scores = {label: 0.0 for label in EMOTION_LABELS}
        for item in prediction:
            raw_label = str(item["label"]).lower()
            if raw_label.startswith("label_"):
                idx = safe_int(raw_label.split("_")[-1])
                label = model_order[idx] if 0 <= idx < len(model_order) else None
            else:
                label = raw_label if raw_label in scores else None
            if label:
                scores[label] = float(item["score"])
        dominant = max(scores, key=scores.get)
        labels.append(dominant)
        confidences.append(scores[dominant])
        emotion_scores.append(scores)

    result["emotion"] = labels
    result["emotion_confidence"] = confidences
    for label in EMOTION_LABELS:
        result[f"emotion_{label}"] = [scores[label] for scores in emotion_scores]
    return result


def mean_pool(last_hidden_state, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_minilm(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    tokenizer, model = load_embedding_model()
    vectors = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=MODEL_MAX_LENGTH, return_tensors="pt")
        import torch
        with torch.inference_mode():
            output = model(**encoded)
            pooled = torch.nn.functional.normalize(mean_pool(output.last_hidden_state, encoded["attention_mask"]), p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype(np.float32, copy=False))
    return np.vstack(vectors)


def discover_topics(df: pd.DataFrame):
    if len(df) < 3:
        result = df.copy(); result["topic_id"] = 0
        return result, pd.DataFrame(), 1

    embeddings = encode_minilm(df["model_text"].tolist())
    max_k = min(MAX_CLUSTER_K, len(df) - 1)
    if max_k < 3:
        result = df.copy(); result["topic_id"] = 0
        return result, pd.DataFrame(), 1

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(embeddings), size=min(len(embeddings), MAX_SILHOUETTE_SAMPLE), replace=False)
    sample = embeddings[sample_idx]
    scores = []
    for k in range(3, max_k + 1):
        labels_sample = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=3, algorithm="lloyd").fit_predict(sample)
        if len(np.unique(labels_sample)) < 2:
            continue
        try:
            scores.append({"k": k, "silhouette_score": float(silhouette_score(sample, labels_sample, metric="cosine"))})
        except ValueError:
            continue

    silhouette_df = pd.DataFrame(scores)
    if silhouette_df.empty:
        best_k = min(5, len(df) - 1)
    else:
        best_k = int(silhouette_df.loc[silhouette_df["silhouette_score"].idxmax(), "k"])

    labels = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=3, algorithm="lloyd").fit_predict(embeddings)
    result = df.copy(); result["topic_id"] = labels
    return result, silhouette_df, best_k


def extract_topic_keywords(df: pd.DataFrame, top_n: int = 10) -> dict[int, list[str]]:
    if df.empty or "topic_id" not in df.columns:
        return {}
    stopwords = set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS
    try:
        vectorizer = TfidfVectorizer(stop_words=list(stopwords), max_features=2500, ngram_range=(1, 2), min_df=2)
        matrix = vectorizer.fit_transform(df["model_text"])
    except ValueError:
        return {}
    features = np.array(vectorizer.get_feature_names_out())
    keywords = {}
    for topic_id in sorted(df["topic_id"].unique()):
        indexes = np.where(df["topic_id"].values == topic_id)[0]
        scores = matrix[indexes].mean(axis=0).A1
        keywords[int(topic_id)] = features[scores.argsort()[::-1][:top_n]].tolist()
    return keywords


def calculate_engagement(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["score"] = pd.to_numeric(result["score"], errors="coerce").fillna(0)
    result["num_comments"] = pd.to_numeric(result["num_comments"], errors="coerce").fillna(0)
    result["engagement"] = result["score"] + 2 * result["num_comments"]
    return result


def get_top_engaged_posts(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["title", "score", "num_comments", "sentiment", "emotion", "topic_id", "engagement", "url"]
    available = [c for c in cols if c in df.columns]
    return df.sort_values("engagement", ascending=False)[available].head(n).reset_index(drop=True)


def sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "sentiment" not in df.columns:
        return pd.DataFrame(columns=["sentiment", "count", "percentage"])
    summary = df["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    summary["percentage"] = summary["count"] / summary["count"].sum() * 100
    return summary


def create_topic_evidence(df: pd.DataFrame, keywords: dict[int, list[str]], top_n_posts: int = TOPIC_EVIDENCE_POSTS) -> list[dict[str, Any]]:
    evidence = []
    for topic_id in sorted(df["topic_id"].unique()):
        topic_df = df[df["topic_id"] == topic_id]
        representative = topic_df.sort_values(["score", "num_comments"], ascending=False).head(top_n_posts)
        posts = [{
            "title": str(row.title)[:TOPIC_EVIDENCE_CHARS],
            "text": str(row.selftext)[:TOPIC_EVIDENCE_CHARS],
            "score": safe_int(row.score),
            "comments": safe_int(row.num_comments),
        } for row in representative.itertuples()]
        evidence.append({
            "topic_id": int(topic_id),
            "post_count": int(len(topic_df)),
            "avg_score": round(float(topic_df["score"].mean()), 2),
            "avg_comments": round(float(topic_df["num_comments"].mean()), 2),
            "keywords": keywords.get(int(topic_id), []),
            "representative_posts": posts,
        })
    return evidence


def _groq_client(api_key: str):
    from groq import Groq
    return Groq(api_key=api_key)


def get_groq_key() -> str:
    try:
        return str(st.secrets["GROQ_API_KEY"])
    except Exception:
        return ""


def _bounded_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = json.dumps(evidence, ensure_ascii=False)
    if len(raw) <= MAX_GROQ_EVIDENCE_CHARS:
        return evidence
    compact = []
    for item in evidence:
        clone = dict(item)
        clone["representative_posts"] = [
            {**post, "title": str(post["title"])[:200], "text": str(post["text"])[:200]}
            for post in item.get("representative_posts", [])[:2]
        ]
        compact.append(clone)
    return compact


@st.cache_data(show_spinner=False)
def generate_groq_analysis(evidence: list[dict[str, Any]], api_key_present: bool) -> dict[str, Any]:
    fallback_topics = [{
        "topic_id": item["topic_id"], "name": f"Topic {item['topic_id']}",
        "description": "", "main_reaction": "",
        "error": "GROQ_API_KEY is missing.",
    } for item in evidence]
    if not api_key_present:
        return {"ai_insights": fallback_topics, "overall_review": "GROQ_API_KEY is not configured."}

    api_key = get_groq_key()
    if not api_key:
        return {"ai_insights": fallback_topics, "overall_review": "GROQ_API_KEY is not configured."}

    bounded = _bounded_evidence(evidence)
    prompt = f"""You are an expert Reddit community analyst. Use only the supplied evidence.
Return exactly one JSON object with keys `topics` and `overall_review`.
`topics` must be an array with one object per topic, each containing:
- topic_id: integer
- name: concise meaningful topic name
- description: concise evidence-grounded explanation
- main_reaction: dominant reaction or attitude
`overall_review` must be a concise 2-3 paragraph summary covering community consensus, major sentiment drivers, recurring discussions, and unusual/niche observations.
Do not invent facts. Do not use Markdown fences.

Evidence:
{json.dumps(bounded, ensure_ascii=False)}"""
    try:
        response = _groq_client(api_key).chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Return only one valid JSON object. Do not include Markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=1800,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        parsed = json.loads(content)
        topics = parsed.get("topics", [])
        by_id = {int(item.get("topic_id")): item for item in topics if isinstance(item, dict) and str(item.get("topic_id", "")).isdigit()}
        insights = []
        for item in evidence:
            topic_id = int(item["topic_id"])
            ai = by_id.get(topic_id, {})
            insights.append({
                "topic_id": topic_id,
                "name": str(ai.get("name") or f"Topic {topic_id}").strip(),
                "description": str(ai.get("description") or "").strip(),
                "main_reaction": str(ai.get("main_reaction") or "").strip(),
            })
        return {"ai_insights": insights, "overall_review": str(parsed.get("overall_review") or "").strip()}
    except Exception as exc:
        return {
            "ai_insights": [dict(item, error=str(exc)) for item in fallback_topics],
            "overall_review": f"Error generating review: {exc}",
        }


@st.cache_data(show_spinner=False)
def run_nlp_pipeline(raw_df: pd.DataFrame, top_posts: int, api_key_present: bool) -> dict[str, Any]:
    """Run the complete NLP pipeline; cached by data + settings."""
    empty = {"clean": pd.DataFrame(), "analysis": pd.DataFrame(), "best_k": None,
             "keywords": {}, "evidence": [], "top_engaged": pd.DataFrame(),
             "sentiment": pd.DataFrame(), "ai_insights": [], "overall_review": ""}
    clean_df = clean_posts(raw_df)
    if clean_df.empty:
        return empty
    analysis_df = select_posts_for_analysis(clean_df, top_posts)
    if analysis_df.empty:
        return empty
    analysis_df = analyze_sentiment(analysis_df)
    analysis_df = analyze_emotions(analysis_df)
    analysis_df, silhouette_df, best_k = discover_topics(analysis_df)
    keywords = extract_topic_keywords(analysis_df)
    analysis_df = calculate_engagement(analysis_df)
    evidence = create_topic_evidence(analysis_df, keywords)
    groq = generate_groq_analysis(evidence, api_key_present)
    return {
        "clean": clean_df,
        "analysis": analysis_df,
        "best_k": best_k,
        "silhouette": silhouette_df,
        "keywords": keywords,
        "evidence": evidence,
        "top_engaged": get_top_engaged_posts(analysis_df),
        "sentiment": sentiment_summary(analysis_df),
        "ai_insights": groq["ai_insights"],
        "overall_review": groq["overall_review"],
    }
