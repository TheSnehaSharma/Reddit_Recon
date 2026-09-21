"""Application configuration. Keep secrets out of source control."""

ARCTIC_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"

# Keep the Groq model exactly as supplied by the original project.
LLM_MODEL = "openai/gpt-oss-20b"

# Local models: chosen for CPU/RAM efficiency while preserving the dashboard's
# 3-class sentiment + 7-class emotion contracts.
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "Frankhihi/fast-emotion-classifier"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_BATCH_SIZE = 16
EMBEDDING_BATCH_SIZE = 32
RANDOM_STATE = 42
MAX_FETCH_RETRIES = 4
MAX_TEXT_CHARS = 1500
MODEL_MAX_LENGTH = 192
ROLLING_WINDOW = "1D"

# Free-tier safeguards.
MAX_SILHOUETTE_SAMPLE = 500
MIN_CLUSTER_K = 3
MAX_CLUSTER_K = 6
TOPIC_EVIDENCE_POSTS = 3
TOPIC_EVIDENCE_CHARS = 450
MAX_GROQ_EVIDENCE_CHARS = 18000

BG = "#111111"
CARD_BG = "#151515"
BORDER = "#2A2A2A"
TEXT_MUTED = "#A3A3A3"

TOPIC_PALETTE = [
    "#06B6D4", "#A855F7", "#F97316", "#14B8A6", "#EAB308",
    "#EC4899", "#0EA5E9", "#84CC16", "#8B5CF6", "#F43F5E",
]

SENTIMENT_COLORS = {
    "Positive": "#22C55E",
    "Neutral": "#64748B",
    "Negative": "#EF4444",
}

EMOTION_LABELS = [
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",
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

EXTRA_STOPWORDS = {
    "reddit", "post", "posts", "people", "really", "just", "like",
    "think", "thing", "things", "want", "got", "get", "going", "does",
    "did", "said", "say", "know", "use", "used", "using",
}
