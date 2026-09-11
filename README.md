# 🔎 Reddit Recon

AI-powered Reddit community intelligence dashboard for sentiment analysis, semantic topic discovery, engagement analysis, and AI-generated insights.

Reddit Recon analyzes Reddit discussions and transforms them into meaningful insights about what a community is discussing, how users feel, and which topics receive the most engagement.

**Live Demo:** https://redditrecon.streamlit.app

<!-- Add main project screenshot here -->

## Features

### 📊 Reddit Community Analysis

Reddit Recon fetches Reddit posts and analyzes recent community activity.

Users can configure:

- Days to analyze
- Number of posts to fetch
- Number of posts used for NLP analysis

<!-- Add dashboard screenshot here -->

### ❤️ Sentiment Analysis

Reddit posts are classified into three sentiment categories:

- 🟢 Positive
- ⚪ Neutral
- 🔴 Negative

Sentiment analysis is powered by the Cardiff NLP Twitter-RoBERTa sentiment model, which is designed for social-media text.

<!-- Add sentiment analysis screenshot here -->

### 🧠 Semantic Topic Discovery

Reddit Recon automatically discovers discussion topics using semantic embeddings and K-Means clustering.

Posts are converted into semantic embeddings using Sentence Transformers and grouped according to their semantic similarity.

The application evaluates different numbers of clusters using the Silhouette Score to determine the most suitable topic structure.

This allows topics to emerge naturally from the discussions instead of relying entirely on predefined categories.

<!-- Add topic discovery screenshot here -->

### 🔥 Engagement Analysis

Reddit Recon analyzes both Reddit post scores and comment activity to identify discussions generating significant community engagement.

This makes it possible to distinguish between topics that are frequently discussed and topics that generate particularly strong interaction.

<!-- Add engagement chart screenshot here -->

### 🤖 AI-Powered Topic Intelligence

Each discovered topic is analyzed using an AI language model through Groq.

Reddit Recon generates:

- Topic names
- Topic descriptions
- Community reactions
- Main discussion points
- Representative posts

This transforms automatically discovered clusters into easy-to-understand community insights.

<!-- Add AI topic analysis screenshot here -->

### 📝 Overall Community Review

Reddit Recon generates an AI-powered review of the overall Reddit community.

The review focuses on:

- Overall community consensus
- Major sentiment drivers
- Recurring discussions
- Important themes
- Unusual or niche observations

<!-- Add community review screenshot here -->

### ☁️ Word Cloud

The Review section includes a word cloud showing frequently discussed terms across the analyzed Reddit posts.

Common Reddit-specific filler terms are removed to make the visualization more meaningful.

<!-- Add word cloud screenshot here -->

## Dashboard

Reddit Recon is organized into four main sections.

### Overview

The Overview section provides a high-level summary of the analyzed Reddit community.

It includes:

- Total posts analyzed
- Total comments
- Total score
- Sentiment distribution
- Topic volume
- Topic engagement
- Interactive charts

<!-- Add Overview screenshot here -->

### Topics

The Topics section provides detailed information about each discovered topic.

Each topic includes:

- AI-generated topic name
- Topic description
- Community reaction
- Sentiment distribution
- Representative Reddit posts
- Post score
- Number of comments
- Reddit URL

<!-- Add Topics screenshot here -->

### Review

The Review section provides an AI-generated summary of the overall community discussion.

It also includes a word cloud generated from the analyzed Reddit content.

<!-- Add Review screenshot here -->

### Data

The Data section provides access to the processed Reddit dataset.

The displayed information includes:

- Post score
- Number of comments
- Post title
- Sentiment
- Topic ID
- Reddit URL

<!-- Add Data screenshot here -->

## How It Works

Reddit Recon follows a multi-stage analysis pipeline:

**Reddit Posts → Data Cleaning → Post Selection → Sentiment Analysis → Semantic Embeddings → Topic Clustering → Topic Evaluation → Keyword Extraction → Engagement Analysis → Representative Posts → AI Analysis → Community Intelligence**

### Reddit Data Collection

Reddit posts are collected using the Arctic Shift Reddit API.

The application includes retry handling for temporary API errors and caching to reduce repeated API requests.

### Text Processing

Reddit post titles and text are cleaned and prepared before being passed through the NLP pipeline.

The processed content is used for sentiment analysis, semantic topic discovery, and word-cloud generation.

### Sentiment Classification

Each selected Reddit post is classified as positive, neutral, or negative using a social-media sentiment model.

### Semantic Embeddings

Posts are converted into numerical semantic representations using the Sentence Transformers framework.

These embeddings allow Reddit Recon to identify posts that discuss similar concepts even when they use different words.

### Topic Clustering

Semantic embeddings are grouped using K-Means clustering.

The application evaluates feasible cluster counts between 3 and 8 and uses the cosine Silhouette Score to select the most appropriate clustering configuration.

### Topic Keywords

TF-IDF is used to identify representative keywords associated with each discovered topic.

These keywords help describe the underlying discussion represented by each cluster.

### Engagement Analysis

Reddit score and comment activity are combined to estimate discussion engagement.

This allows topics to be compared not only by how frequently they appear, but also by how much interaction they generate.

### AI Analysis

The discovered topics, representative posts, sentiment information, and engagement data are analyzed using GPT-OSS through Groq.

The AI produces human-readable topic descriptions, community reactions, and an overall community review.

## Technology Stack

- Python
- Streamlit
- Pandas
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Scikit-learn
- Plotly
- WordCloud
- Groq
- Arctic Shift Reddit API
- Font Awesome

## Models Used

| Model | Purpose |
|---|---|
| Cardiff NLP Twitter-RoBERTa Sentiment | Sentiment classification |
| Sentence Transformers MiniLM | Semantic embeddings |
| GPT-OSS-20B | AI topic and community analysis |

## Configuration

The Reddit Recon sidebar provides three main controls:

| Setting | Range | Default |
|---|---:|---:|
| Days to Analyze | 1–30 | 1 |
| Posts to Fetch | 100–1000 | 300 |
| Top Posts for NLP | 10–500 | 100 |

The application uses a Groq API key for AI-powered topic analysis and community review.

## API & Data Handling

Reddit Recon uses the Arctic Shift Reddit API as its Reddit data source.

The application includes:

- API retry handling
- Response caching
- Configurable data collection
- Configurable NLP processing
- AI-powered analysis through Groq

## Performance

Reddit Recon includes several optimizations to improve dashboard performance:

- Cached NLP models
- Cached Reddit API responses
- CPU-based sentiment inference
- Configurable NLP dataset size
- Parallel AI topic analysis
- Retry handling for temporary API errors

## Screenshots

### Dashboard

<!-- Add screenshot here -->

### Overview

<!-- Add screenshot here -->

### Topics

<!-- Add screenshot here -->

### Review

<!-- Add screenshot here -->

### Data

<!-- Add screenshot here -->

## Use Cases

Reddit Recon can be used for:

- 📈 Market research
- 🧠 Community research
- 📊 Social listening
- 🏷️ Brand monitoring
- 🧑‍💻 Product feedback
- 🎮 Gaming community analysis
- 📱 Product research
- 🔎 Competitive intelligence
- 💬 Customer sentiment analysis
- 📢 Trend discovery

## Limitations

- Sentiment models may not always correctly interpret sarcasm, slang, memes, or highly contextual language.
- Topic clustering is unsupervised, so some discovered topics may overlap.
- AI-generated descriptions are interpretations and should not be treated as verified facts.
- Small datasets may not produce meaningful topic clusters.
- Analysis quality depends on the number and diversity of retrieved Reddit posts.
- Reddit API availability and behavior may change.

## Project Structure

The project contains the main Streamlit application, dependency configuration, Streamlit secrets, and optional assets for README screenshots.

<!-- Add project structure screenshot here if desired -->

## Contributing

Contributions, suggestions, and improvements are welcome.

If you have ideas for improving Reddit Recon, feel free to open an issue or submit a pull request.

## License

No specific license has currently been specified for this project.

## Acknowledgments

- Arctic Shift Reddit API
- Hugging Face Transformers
- Sentence Transformers
- Scikit-learn
- Streamlit
- Groq
- OpenAI GPT-OSS
- Plotly
- WordCloud

## About

Reddit Recon combines Reddit data collection, natural language processing, machine learning, data visualization, and generative AI to transform Reddit discussions into actionable community intelligence.

**Live Demo:** https://redditrecon.streamlit.app
