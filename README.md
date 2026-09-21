# Reddit Recon — optimized Streamlit portfolio version

Reddit Community Intelligence Dashboard using Reddit archive data, local NLP, semantic topic discovery, and Groq-assisted topic/community analysis.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For Groq features, add the existing secret to Streamlit secrets without changing its name or value:

```toml
GROQ_API_KEY = "your-existing-key"
```

The app remains deployable on Streamlit Community Cloud. No local model files are committed; Hugging Face downloads the models on first use and Streamlit caches them for the process.

## Architecture

```text
Streamlit UI
    |
    +--> Arctic Shift Reddit API (cached HTTP data)
    |
    +--> Cleaning / selection
    |
    +--> Local sentiment model (CPU, int8 dynamic quantization)
    |
    +--> Local emotion model (CPU, 66M DistilBERT, int8 dynamic quantization)
    |
    +--> MiniLM embeddings (cached, normalized)
    |
    +--> sampled silhouette + KMeans topic discovery
    |
    +--> TF-IDF topic keywords
    |
    +--> one batched Groq analysis request
    |
    +--> unchanged dashboard tabs
```

## Why the UI was left alone

The dashboard controls, four tabs, metrics, charts, topic cards, review, word cloud, data table, CSS and overall workflow remain the same. The refactor is primarily a separation of UI from compute and a reduction in repeated work.

## Tests

```bash
pytest -q
```

The test suite focuses on deterministic, dependency-light logic: input normalization, text cleaning, selection, engagement calculation and keyword extraction. Remote Reddit/Groq calls and model downloads are intentionally not part of unit tests.
