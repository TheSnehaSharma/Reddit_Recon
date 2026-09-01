# 🎯 Reddit Conversation Intelligence Dashboard

> A professional social listening and conversation analytics platform powered by Databricks, NLP, and Machine Learning

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=Databricks&logoColor=white)](https://databricks.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

---

## 📊 Overview

Reddit Conversation Intelligence is an enterprise-grade analytics platform that transforms Reddit conversations into actionable insights using advanced NLP and machine learning. The platform provides real-time social listening capabilities, sentiment analysis, topic intelligence, and predictive analytics.

### ✨ Key Features

* **📈 Executive Overview** - Real-time KPIs, conversation volume trends, and topic rankings
* **💭 Sentiment Analysis** - Emotional polarity detection with confidence scoring
* **📚 Topic Intelligence** - Automatic topic classification and trend detection
* **🎭 Emotion Analysis** - Multi-class emotion detection (joy, anger, fear, etc.)
* **🔥 Engagement Analytics** - Identify high-performing content and optimal posting times
* **🤖 Predictive Analytics** - ML-powered engagement prediction with feature importance
* **⚠️ Anomaly Detection** - Real-time spike detection and unusual pattern identification

---

## 🏗️ Architecture

### Dashboard Pages

```
├── 📊 Overview                   # Executive summary and KPIs
├── 💭 Sentiment Analysis         # Emotional polarity insights
├── 📚 Topic Intelligence         # What people are talking about
├── 🎭 Emotion Analysis          # Emotional response patterns
├── 🔥 Engagement Analysis       # What drives interaction
├── 🤖 Predictive Analytics      # ML-powered predictions
└── ⚠️ Anomaly Detection         # Unusual pattern identification
```

### Technical Stack

**Frontend & Visualization**
* Streamlit - Interactive web application framework
* Plotly - Professional interactive charts and graphs
* Pandas & NumPy - Data manipulation and analysis

**Backend & Data**
* Databricks Lakehouse - Unified data platform
* Delta Lake - ACID-compliant data storage
* Databricks SQL - Serverless query engine

**AI & Machine Learning**
* Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
* Emotion: `j-hartmann/emotion-english-distilroberta-base`
* Topic: `facebook/bart-large-mnli` (zero-shot classification)

---

## 🚀 Quick Start

### Prerequisites

* Databricks workspace with access to `workspace.redditrecon.posts_gold` table
* Python 3.8+
* Environment variables configured (see below)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Reddit_Recon.git
   cd Reddit_Recon/streamlit_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file or set environment variables:
   ```bash
   export DATABRICKS_SERVER_HOSTNAME="your-workspace.cloud.databricks.com"
   export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/your-warehouse-id"
   export DATABRICKS_TOKEN="your-access-token"
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**
   
   Open your browser and navigate to `http://localhost:8501`

---

## 📁 Project Structure

```
Reddit_Recon/
│
├── streamlit_app/              # Main dashboard application
│   ├── app.py                  # Main Streamlit application (7 pages)
│   ├── data_loader.py          # Database queries and data loading
│   ├── metrics.py              # KPI calculations and aggregations
│   ├── charts.py               # Plotly visualization functions
│   ├── requirements.txt        # Python dependencies
│   └── .streamlit/             # Streamlit configuration
│       └── config.toml         # Theme and settings
│
├── README.md                   # This file
├── LICENSE                     # MIT License
└── .gitignore                  # Git ignore rules
```

---

## 📊 Data Schema

The dashboard expects a `posts_gold` table with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `post_id` | STRING | Unique post identifier |
| `created_at` | TIMESTAMP | Post creation timestamp |
| `subreddit` | STRING | Subreddit name |
| `author` | STRING | Reddit username |
| `title` | STRING | Post title |
| `selftext` | STRING | Post body text |
| `sentiment_label` | STRING | Positive / Neutral / Negative |
| `sentiment_score` | DOUBLE | Sentiment confidence score (-1 to 1) |
| `emotion_label` | STRING | Emotion classification |
| `emotion_score` | DOUBLE | Emotion confidence score (0 to 1) |
| `topic_label` | STRING | Topic classification |
| `topic_score` | DOUBLE | Topic confidence score (0 to 1) |
| `score` | INTEGER | Reddit post score (upvotes) |
| `num_comments` | INTEGER | Number of comments |
| `upvote_ratio` | DOUBLE | Upvote ratio (0 to 1) |

---

## 🎨 Dashboard Features

### 1. Overview Dashboard
* **KPI Cards**: Total posts, average sentiment, total engagement, trending topics, dominant emotions
* **Conversation Trends**: Volume over time with 7-day rolling average
* **Sentiment Timeline**: Stacked area chart showing sentiment distribution
* **Top Topics**: Horizontal bar chart with post counts and percentages
* **Recent Conversations**: Interactive table with latest posts

### 2. Sentiment Analysis
* **Distribution Chart**: Sentiment breakdown with percentages
* **Trend Analysis**: Average sentiment score over time
* **Topic Heatmap**: Sentiment distribution across topics
* **Engagement Impact**: Box plots comparing engagement by sentiment

### 3. Topic Intelligence
* **Topic Ranking**: Top 15 topics by volume with percentages
* **Topic Evolution**: Multi-line chart tracking topic trends
* **Topic Velocity**: Growth rate comparison (trending up/down)
* **Sentiment Comparison**: Stacked bar chart of sentiment by topic

### 4. Emotion Analysis
* **Emotion Distribution**: Bar chart of all detected emotions
* **Temporal Trends**: Emotion composition over time (top 5)
* **Engagement Correlation**: Which emotions drive interaction
* **Topic Breakdown**: Heatmap of emotions by topic

### 5. Engagement Analysis
* **Topic Rankings**: Average engagement by topic
* **Best Time to Post**: Day/hour heatmap of optimal posting times
* **Content Analysis**: Text length vs engagement scatter plot
* **Top Performers**: Table of highest-engagement posts

### 6. Predictive Analytics
* **Feature Importance**: Chart showing ML feature contributions
* **Model Comparison**: Performance metrics (Accuracy, F1, ROC-AUC)
* **Engagement Simulator**: Interactive prediction tool
* **Distribution Analysis**: Engagement score histogram

### 7. Anomaly Detection
* **Volume Anomalies**: Z-score based spike detection
* **Topic Spikes**: Topics with >100% growth
* **Sentiment Shifts**: Negative sentiment trend monitoring
* **Unusual Posts**: Table of exceptional engagement (top 1%)

---

## 🔧 Configuration

### Streamlit Configuration

Edit `.streamlit/config.toml` to customize:

```toml
[theme]
primaryColor = "#3498db"
backgroundColor = "#f8f9fa"
secondaryBackgroundColor = "#ffffff"
textColor = "#2c3e50"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
```

### Cache Settings

* **Data cache TTL**: 1 hour (3600 seconds)
* **Filter options cache**: 2 hours (7200 seconds)
* **Query result limit**: 50,000 rows

---

## 📈 Metrics & Calculations

### Engagement Score
```python
engagement_score = log(1 + post_score) + log(1 + num_comments)
```

### Topic Velocity
```python
velocity = ((current_count - previous_count) / previous_count) * 100
```

### High Engagement Classification
Posts in the top 25% percentile of engagement scores

### Anomaly Detection
Z-score > 2.5 using 7-day rolling window

---

## 🎯 Use Cases

* **Brand Monitoring**: Track brand mentions and sentiment in real-time
* **Market Research**: Understand consumer opinions and trending topics
* **Crisis Management**: Detect sentiment shifts and conversation spikes
* **Content Strategy**: Identify optimal posting times and high-engagement topics
* **Competitive Intelligence**: Monitor competitor discussions
* **Product Insights**: Discover feature requests and pain points

---

## 🛠️ Development

### Adding New Features

1. **New Visualizations**: Add chart functions to `charts.py`
2. **New Metrics**: Add calculation functions to `metrics.py`
3. **New Data Sources**: Extend queries in `data_loader.py`
4. **New Pages**: Add page logic in `app.py` under navigation

### Running in Development Mode

```bash
streamlit run app.py --server.runOnSave=true
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Sneha Sharma**
* Data Scientist & ML Engineer
* Specializing in NLP, Social Media Analytics, and Databricks

---

## 🙏 Acknowledgments

* **Hugging Face** - Pre-trained NLP models
* **Databricks** - Lakehouse platform and compute
* **Streamlit** - Dashboard framework
* **Plotly** - Interactive visualizations
* **Reddit Archive** - Data source

---

## 📞 Support

For questions, issues, or feature requests:
* Open an issue on GitHub
* Contact: [Your Email]

---

<div align="center">
  <strong>Built with ❤️ using Databricks, Streamlit, and Advanced NLP</strong>
  <br><br>
  <sub>Transform conversations into insights • Powered by AI</sub>
</div>
