import html
from contextlib import contextmanager

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import (
    BG, BORDER, CARD_BG, EMOTION_COLORS, EMOTION_LABELS,
    EMOTION_MODEL, EMBEDDING_MODEL, EXTRA_STOPWORDS, LLM_MODEL,
    ROLLING_WINDOW, SENTIMENT_COLORS, SENTIMENT_MODEL, TEXT_MUTED,
    TOPIC_PALETTE,
)
from core import (
    calculate_engagement, fetch_reddit_posts, get_groq_key,
    get_top_engaged_posts, normalize_subreddit, run_nlp_pipeline,
    safe_int,
)

st.set_page_config(
    page_title="Reddit Recon",
    page_icon="https://www.reddit.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.html(f"""
    <style>
    .stApp,.main{{background:{BG};}}
    html,body,[class*="css"],p,span,div,label,li,td,th{{color:#FFFFFF;}}
    h1,h2,h3,h4,h5,h6{{color:#FFFFFF!important;}}
    section[data-testid="stSidebar"]{{background:{BG};border-right:1px solid {BORDER};}}
    section[data-testid="stSidebar"] *{{color:#FFFFFF!important;}}
    .reddit-header{{display:flex;align-items:center;padding:1.2rem 0;margin-bottom:1.5rem;border-bottom:1px solid {BORDER};}}
    .reddit-header .reddit-icon{{font-size:58px;margin-right:20px;width:58px;flex-shrink:0;}}
    .reddit-header h1{{margin:0;font-size:2.2rem;font-weight:700;}}
    .reddit-header p{{margin:5px 0 0;font-size:.95rem;color:#A3A3A3!important;}}
    .reddit-card{{background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;padding:1.25rem;margin-bottom:1rem;}}
    .reddit-card h2{{margin-top:0;font-size:1.4rem;}}
    .topic-title{{font-weight:700;font-size:1.35rem;margin-bottom:.75rem;}}
    .topic-title::before{{content:"\\25CF";margin-right:10px;color:#FF4500;}}
    .badge{{display:inline-block;padding:.3rem .7rem;border-radius:999px;font-size:.75rem;font-weight:700;margin:0 .5rem .4rem 0;border:1px solid #444;background:#1C1C1C;}}
    a,a:hover{{color:#FFFFFF!important;}} a:hover{{text-decoration:underline;}}
    input,textarea,select,button{{color:#FFFFFF!important;background-color:#181818!important;}}
    hr{{border-color:{BORDER}!important;}}
    [data-testid="stDataFrame"]{{border:1px solid {BORDER};}}
    [data-testid="stMetric"]{{background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;padding:1rem;}}
    [data-testid="stMetricValue"]{{color:#FFFFFF!important;font-size:2rem;font-weight:700;}}
    [data-testid="stMetricLabel"]{{color:#BDBDBD!important;}}
    div[data-baseweb="select"] *,div[data-baseweb="input"] *{{color:#FFFFFF!important;background-color:#181818!important;}}
    .js-plotly-plot{{border-radius:8px;}}
    .reddit-sidebar-icon{{font-size:46px;text-align:center;margin:10px 0 20px;}}
    .section-label{{color:#AAAAAA!important;font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;}}
    .top-post-link{{color:#FFFFFF!important;text-decoration:none;}}
    .top-post-link:hover{{text-decoration:underline;}}
    .confidence-wrapper{{margin-top:.75rem;margin-bottom:.75rem;}}
    .confidence-title{{font-size:.82rem;font-weight:600;color:#A3A3A3!important;margin-bottom:.45rem;}}
    .confidence-track{{display:flex;width:100%;height:18px;border-radius:999px;overflow:hidden;background:#252525;border:1px solid #333;}}
    .confidence-segment{{flex:1;position:relative;background:#222;border-right:1px solid #111;overflow:hidden;}}
    .confidence-segment:last-child{{border-right:none;}}
    .confidence-fill{{height:100%;border-radius:999px;}}
    .confidence-labels{{display:flex;width:100%;margin-top:.4rem;}}
    .confidence-label{{flex:1;text-align:center;font-size:.72rem;color:#A3A3A3!important;}}
    .legend-row{{display:flex;flex-wrap:wrap;gap:.65rem 1rem;margin-top:.5rem;margin-bottom:.4rem;}}
    .legend-item{{font-size:.78rem;white-space:nowrap;}}
    .legend-dot{{font-size:1rem;}}
    </style>
    """)


inject_css()


@contextmanager
def card():
    st.html('<div class="reddit-card">')
    yield
    st.html('</div>')


def style_fig(fig, **overrides):
    layout = dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color="white"),
        margin=dict(t=40, b=40, l=40, r=20),
        xaxis=dict(gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER),
    )
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


def topic_color_map(names):
    return {name: TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i, name in enumerate(names)}


def page_header(title, subtitle):
    st.html(f"""
    <div class="reddit-header">
        <div class="reddit-icon"><i class="fa-brands fa-reddit" aria-hidden="true"></i></div>
        <div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>
    </div>
    """)


def render_confidence_strip(values, colors, labels, title):
    if not values:
        return
    segments, labels_html = [], []
    for label, value in zip(labels, values):
        value = max(0, min(100, float(value)))
        color = colors.get(label, "#64748B")
        segments.append(f'<div class="confidence-segment" title="{html.escape(label)}: {value:.1f}%"><div class="confidence-fill" style="width:{value:.1f}%;background:{color};"></div></div>')
        labels_html.append(f'<div class="confidence-label"><span style="color:{color};">●</span> {html.escape(label)} <b>{value:.0f}%</b></div>')
    st.html(f'<div class="confidence-wrapper"><div class="confidence-title">{html.escape(title)}</div><div class="confidence-track">{"".join(segments)}</div><div class="confidence-labels">{"".join(labels_html)}</div></div>')


def apply_dashboard_filters(analysis_df, name_map):
    filter1, filter2, filter3 = st.columns(3)
    with filter1:
        topic_options = ["All Topics"] + sorted(name_map.values())
        selected_topic = st.selectbox("Topic", topic_options, key="overview_topic_filter")
    with filter2:
        selected_sentiment = st.selectbox("Sentiment", ["All Sentiments", "Positive", "Neutral", "Negative"], key="overview_sentiment_filter")
    with filter3:
        selected_emotion = st.selectbox("Emotion", ["All Emotions"] + [x.capitalize() for x in EMOTION_LABELS], key="overview_emotion_filter")

    filtered = analysis_df.copy()
    if selected_topic != "All Topics":
        lookup = {name: topic_id for topic_id, name in name_map.items()}
        topic_id = lookup.get(selected_topic)
        if topic_id is not None:
            filtered = filtered[filtered["topic_id"] == topic_id]
    if selected_sentiment != "All Sentiments":
        filtered = filtered[filtered["sentiment"] == selected_sentiment.lower()]
    if selected_emotion != "All Emotions":
        filtered = filtered[filtered["emotion"] == selected_emotion.lower()]
    return filtered


def render_sentiment_distribution(df):
    st.subheader("Sentiment Distribution")
    if df.empty:
        st.info("No posts match the selected filters."); return
    counts = df["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    counts["label"] = counts["sentiment"].str.capitalize()
    fig = px.pie(counts, values="count", names="label", hole=.55, color="label", color_discrete_map=SENTIMENT_COLORS)
    fig.update_traces(textinfo="percent", hovertemplate="<b>%{label}</b><br>Posts: %{value:,}<br>Share: %{percent}<extra></extra>")
    st.plotly_chart(style_fig(fig, margin=dict(t=10,b=10,l=10,r=10), legend=dict(font=dict(color="white"))), use_container_width=True)
    values = [df.loc[df.sentiment == s, "sentiment_confidence"].mean() * 100 if not df.loc[df.sentiment == s].empty else 0 for s in ["positive","neutral","negative"]]
    render_confidence_strip(values, SENTIMENT_COLORS, ["Positive","Neutral","Negative"], "Average Prediction Confidence")


def render_emotion_distribution(df):
    st.subheader("Emotion Distribution")
    if df.empty:
        st.info("No posts match the selected filters."); return
    counts = df["emotion"].value_counts().rename_axis("emotion").reset_index(name="count")
    counts["label"] = counts["emotion"].str.capitalize()
    fig = px.bar(counts.sort_values("count"), x="count", y="label", orientation="h", color="emotion", color_discrete_map=EMOTION_COLORS, labels={"count":"Posts","label":""})
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Posts: %{x:,}<extra></extra>")
    st.plotly_chart(style_fig(fig, margin=dict(t=10,b=10,l=30,r=20), showlegend=False), use_container_width=True)
    values, labels, colors = [], [], {}
    for emotion in EMOTION_LABELS:
        subset = df[df.emotion == emotion]
        values.append(subset["emotion_confidence"].mean() * 100 if not subset.empty else 0)
        label = emotion.capitalize(); labels.append(label); colors[label] = EMOTION_COLORS[emotion]
    render_confidence_strip(values, colors, labels, "Average Prediction Confidence")


def render_engagement_chart(df):
    st.subheader("Engagement Performance")
    mode = st.radio("View", ["Sentiment", "Emotion"], horizontal=True, key="engagement_mode")
    if df.empty:
        st.info("No posts match the selected filters."); return
    column = "sentiment" if mode == "Sentiment" else "emotion"
    grouped = df.groupby(column).agg(avg_engagement=("engagement","mean"), median_engagement=("engagement","median"), posts=("id","count"), avg_comments=("num_comments","mean")).reset_index()
    grouped["label"] = grouped[column].str.capitalize()
    colors = SENTIMENT_COLORS if mode == "Sentiment" else EMOTION_COLORS
    fig = px.bar(grouped.sort_values("avg_engagement", ascending=False), x="label", y="avg_engagement", text="avg_engagement", color=column, color_discrete_map=colors, labels={"label":"","avg_engagement":"Average Engagement"})
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(style_fig(fig, margin=dict(t=30,b=50,l=50,r=20), showlegend=False, xaxis=dict(gridcolor=BORDER,tickangle=-20)), use_container_width=True)


def render_topic_performance(df, name_map):
    st.subheader("Topic Performance")
    if df.empty:
        st.info("No topic data matches the selected filters."); return
    stats = df.groupby("topic_id").agg(posts=("id","count"), avg_engagement=("engagement","mean"), total_comments=("num_comments","sum")).reset_index()
    stats["Topic Name"] = stats.topic_id.map(name_map).fillna(stats.topic_id.apply(lambda x: f"Topic {x}"))
    colors = topic_color_map(stats["Topic Name"])
    fig = px.scatter(stats, x="posts", y="avg_engagement", size="total_comments", hover_name="Topic Name", text="Topic Name", color="Topic Name", color_discrete_map=colors, labels={"posts":"Post Volume","avg_engagement":"Average Engagement","total_comments":"Total Comments"})
    fig.update_traces(textposition="top center")
    st.plotly_chart(style_fig(fig, margin=dict(t=30,b=50,l=50,r=20), showlegend=False), use_container_width=True)


def render_topic_distribution(df, name_map):
    st.subheader("Topic Distribution")
    if df.empty:
        st.info("No posts match the selected filters."); return
    distribution = df.groupby("topic_id").size().reset_index(name="posts")
    distribution["Topic Name"] = distribution.topic_id.map(name_map).fillna(distribution.topic_id.apply(lambda x: f"Topic {x}"))
    distribution = distribution.sort_values("posts", ascending=False)
    colors = topic_color_map(distribution["Topic Name"])
    fig = px.bar(distribution, x="Topic Name", y="posts", text="posts", color="Topic Name", color_discrete_map=colors, labels={"Topic Name":"","posts":"Number of Posts"})
    fig.update_traces(textposition="outside")
    st.plotly_chart(style_fig(fig, margin=dict(t=30,b=80,l=40,r=20), showlegend=False, xaxis=dict(tickangle=-30,gridcolor=BORDER)), use_container_width=True)


def render_trends(df):
    st.subheader("Community Sentiment & Emotion Over Time")
    mode = st.radio("Trend View", ["Sentiment", "Emotion"], horizontal=True, key="trend_mode")
    if df.empty:
        st.info("No posts match the selected filters."); return
    trend = df.copy()
    trend["date"] = pd.to_datetime(trend["created_utc"], unit="s", utc=True).dt.floor(ROLLING_WINDOW)
    column = "sentiment" if mode == "Sentiment" else "emotion"
    counts = trend.groupby(["date", column]).size().reset_index(name="posts")
    totals = trend.groupby("date").size().reset_index(name="total")
    counts = counts.merge(totals, on="date", how="left")
    counts["percentage"] = counts.posts / counts.total * 100
    counts["label"] = counts[column].str.capitalize()
    colors = SENTIMENT_COLORS if mode == "Sentiment" else EMOTION_COLORS
    fig = px.line(counts, x="date", y="percentage", color="label", color_discrete_map=colors, markers=True, labels={"date":"Date","percentage":"Share of Posts (%)","label":mode})
    st.plotly_chart(style_fig(fig, margin=dict(t=30,b=50,l=50,r=20), yaxis=dict(gridcolor=BORDER,title="Share of Posts (%)")), use_container_width=True)


def render_top_posts(df):
    st.subheader("Top 5 Posts by Engagement")
    if df.empty:
        st.info("No posts match the selected filters."); return
    top5 = df.sort_values("engagement", ascending=False).head(5)
    rows = []
    for rank, row in enumerate(top5.itertuples(), 1):
        rows.append({"Rank":rank,"Post":str(row.title),"Sentiment":str(row.sentiment).capitalize(),"Emotion":str(row.emotion).capitalize(),"Score":safe_int(row.score),"Comments":safe_int(row.num_comments),"Engagement":float(row.engagement)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config={"Rank":st.column_config.NumberColumn("Rank",width="small"),"Post":st.column_config.TextColumn("Post",width="large"),"Engagement":st.column_config.NumberColumn("Engagement",format="%.2f")})


def get_prominent_non_neutral_sentiment(topic_df):
    if topic_df.empty: return None, 0.0
    counts = topic_df.sentiment.value_counts(normalize=True) * 100
    counts = counts[counts.index.isin(["positive","negative"])]
    if counts.empty: return None, 0.0
    label = counts.idxmax(); return label, float(counts[label])


def get_prominent_non_neutral_emotion(topic_df):
    if topic_df.empty: return None, 0.0
    counts = topic_df.emotion.value_counts(normalize=True) * 100
    counts = counts[counts.index != "neutral"]
    if counts.empty: return None, 0.0
    label = counts.idxmax(); return label, float(counts[label])


def render_topic_reaction_badges(topic_df):
    sentiment_label, sentiment_pct = get_prominent_non_neutral_sentiment(topic_df)
    emotion_label, emotion_pct = get_prominent_non_neutral_emotion(topic_df)
    badges = []
    if sentiment_label:
        color = SENTIMENT_COLORS.get(sentiment_label.capitalize(), "#64748B")
        badges.append(f'<span class="badge" style="border-color:{color};color:{color};">Sentiment: {html.escape(sentiment_label.capitalize())} {sentiment_pct:.0f}%</span>')
    if emotion_label:
        color = EMOTION_COLORS.get(emotion_label, "#64748B")
        badges.append(f'<span class="badge" style="border-color:{color};color:{color};">Emotion: {html.escape(emotion_label.capitalize())} {emotion_pct:.0f}%</span>')
    st.html("".join(badges))


def generate_word_cloud(df):
    if df.empty: return None
    text = " ".join(df.title.fillna("") + " " + df.selftext.fillna(""))
    stopwords = set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS
    wordcloud = WordCloud(width=1200,height=550,background_color=BG,stopwords=stopwords,colormap="turbo",max_words=150,min_font_size=10,max_font_size=100,prefer_horizontal=.9).generate(text)
    fig, ax = plt.subplots(figsize=(12,5.5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.imshow(wordcloud, interpolation="bilinear"); ax.axis("off"); plt.tight_layout(pad=0)
    return fig


def render_sidebar():
    st.sidebar.title("Recon Settings")
    with st.sidebar.form("subreddit_form", clear_on_submit=False):
        subreddit_input = st.text_input("Subreddit", placeholder="e.g. technology, Python, gaming", help="Enter a subreddit without r/")
        days_back = st.slider("Days to analyze", 1, 30, 1)
        posts_to_fetch = st.slider("Posts to fetch", 100, 1000, 300, 50)
        top_posts = st.slider("Posts for NLP", 10, 500, 100, 10)
        submitted = st.form_submit_button("Run Recon", type="primary", use_container_width=True)
    return subreddit_input, days_back, posts_to_fetch, top_posts, submitted


def execute_recon(subreddit_input, days_back, posts_to_fetch, top_posts):
    subreddit = normalize_subreddit(subreddit_input)
    if not subreddit or not subreddit.replace("_", "").isalnum():
        st.sidebar.error("Use a valid subreddit name."); return
    st.session_state.pop("result", None)
    st.session_state["selected_subreddit"] = subreddit
    progress, status = st.progress(0), st.empty()
    try:
        status.write(f"Fetching posts from r/{subreddit}...")
        raw_df = fetch_reddit_posts(subreddit, posts_to_fetch, days_back)
        progress.progress(35)
        if raw_df.empty:
            status.empty(); progress.empty(); st.error(f"No posts were fetched from r/{subreddit}. Check the subreddit name or try a longer date range."); return
        status.write(f"Fetched {len(raw_df):,} posts. Running sentiment, emotion and topic analysis...")
        result = run_nlp_pipeline(raw_df, top_posts, bool(get_groq_key()))
        progress.progress(85)
        if not isinstance(result, dict) or "analysis" not in result:
            raise RuntimeError("NLP pipeline returned an invalid result.")
        result["raw"] = raw_df
        st.session_state["result"] = result
        progress.progress(100); status.empty(); progress.empty()
        st.success(f"Recon complete for r/{subreddit}.")
    except Exception as exc:
        status.empty(); progress.empty(); st.error("Recon failed."); st.exception(exc)


def render_overview(raw_df, analysis_df, best_k, name_map):
    st.subheader("Dashboard Filters")
    filtered_df = apply_dashboard_filters(analysis_df, name_map)
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Posts", f"{len(filtered_df):,}")
    k2.metric("Avg Comments", f"{filtered_df.num_comments.mean():.1f}" if not filtered_df.empty else "0.0")
    k3.metric("Avg Score", f"{filtered_df.score.mean():.1f}" if not filtered_df.empty else "0.0")
    k4.metric("Avg Engagement", f"{filtered_df.engagement.mean():.2f}" if not filtered_df.empty else "0.00")
    c1,c2 = st.columns(2)
    with c1: render_sentiment_distribution(filtered_df)
    with c2: render_emotion_distribution(filtered_df)
    st.markdown("---"); render_engagement_chart(filtered_df)
    st.markdown("---"); c1,c2 = st.columns(2)
    with c1: render_topic_performance(filtered_df, name_map)
    with c2: render_topic_distribution(filtered_df, name_map)
    st.markdown("---"); render_trends(filtered_df)
    st.markdown("---"); render_top_posts(filtered_df)


def render_topics_tab(analysis_df, ai_insights):
    if not ai_insights:
        st.warning("No AI insights were generated."); return
    for item in ai_insights:
        topic_id = item.get("topic_id")
        topic_df = analysis_df[analysis_df.topic_id == topic_id]
        if topic_df.empty: continue
        topic_name = item.get("name", f"Topic {topic_id}")
        representative = topic_df.sort_values(["score","num_comments"], ascending=False).head(5)
        rep_html = "".join(f'<li style="margin-bottom:.7rem;"><a class="top-post-link" href="{html.escape(str(row.url))}" target="_blank" rel="noopener noreferrer"><b>{html.escape(str(row.title))}</b></a> <span>(Score: {safe_int(row.score):,} | Comments: {safe_int(row.num_comments):,})</span></li>' for row in representative.itertuples())
        st.markdown("---"); st.html(f'<div class="topic-title">{html.escape(str(topic_name))}</div>')
        render_topic_reaction_badges(topic_df)
        st.markdown(f'**Analysis:** {item.get("description", "")}')
        st.markdown(f'**Reaction Context:** {item.get("main_reaction", "")}')
        if item.get("error"): st.html(f'<p><b>AI status:</b> {html.escape(str(item["error"]))}</p>')
        st.markdown("---"); st.markdown("**TOP POSTS IN TOPIC**")
        st.html(f'<ul style="list-style-type:none;padding-left:0;">{rep_html}</ul>')


def render_review_tab(subreddit, overall_review, analysis_df):
        st.subheader(f"General Consensus: r/{subreddit}")
        st.write(overall_review) if overall_review else st.info("LLM summary is not available.")
        st.markdown("---")
        st.subheader("Commonly Used Words")
        fig = generate_word_cloud(analysis_df)
        if fig is not None: st.pyplot(fig, clear_figure=True)


def render_data_tab(analysis_df):
    st.subheader("Top Posts Dataset")
    display_cols = ["score","num_comments","title","sentiment","sentiment_confidence","emotion","emotion_confidence","topic_id","engagement","url"] + [f"emotion_{x}" for x in EMOTION_LABELS]
    available = [c for c in display_cols if c in analysis_df.columns]
    st.dataframe(analysis_df.sort_values("score", ascending=False)[available], use_container_width=True, hide_index=True)


def main():
    subreddit_input, days_back, posts_to_fetch, top_posts, submitted = render_sidebar()
    if submitted:
        execute_recon(subreddit_input, days_back, posts_to_fetch, top_posts)
    if "result" not in st.session_state:
        page_header("Reddit Recon", "Community intelligence, semantic topics, sentiment and engagement analysis.")
        st.write("Enter a subreddit in the sidebar and click **Run Recon**.")
        return
    result = st.session_state.get("result")
    if not isinstance(result, dict):
        st.error("Invalid analysis result."); st.session_state.pop("result", None); return
    raw_df = result.get("raw", pd.DataFrame())
    analysis_df = result.get("analysis", pd.DataFrame())
    best_k = result.get("best_k")
    ai_insights = result.get("ai_insights", [])
    overall_review = result.get("overall_review", "")
    if analysis_df.empty:
        st.error("No usable posts were available for NLP analysis."); return
    subreddit = st.session_state.get("selected_subreddit", subreddit_input)
    name_map = {item.get("topic_id"): item.get("name", f"Topic {item.get('topic_id')}") for item in ai_insights}
    page_header(f"Reddit Recon: r/{subreddit}", f"Analyzed top {len(analysis_df):,} posts.")
    tab1,tab2,tab3,tab4 = st.tabs(["Overview","Topics","AI Review","Data"])
    with tab1: render_overview(raw_df, analysis_df, best_k, name_map)
    with tab2: render_topics_tab(analysis_df, ai_insights)
    with tab3: render_review_tab(subreddit, overall_review, analysis_df)
    with tab4: render_data_tab(analysis_df)


if __name__ == "__main__":
    main()
