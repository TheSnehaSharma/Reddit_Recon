import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from core import (
    calculate_engagement,
    clean_posts,
    extract_topic_keywords,
    normalize_subreddit,
    safe_int,
    select_posts_for_analysis,
)


def test_safe_int_handles_bad_values():
    assert safe_int("42") == 42
    assert safe_int("not-a-number") == 0
    assert safe_int(None) == 0


def test_normalize_subreddit():
    assert normalize_subreddit(" r/Python ") == "Python"
    assert normalize_subreddit("machine learning") == "machinelearning"


def test_clean_posts_builds_bounded_model_text():
    df = pd.DataFrame({
        "id": ["1", "2"],
        "title": ["Useful title", "x"],
        "selftext": ["A useful body with enough words to pass the minimum text length.", "short"],
        "score": [10, 1],
        "num_comments": [2, 0],
        "created_utc": [1, 2],
        "url": ["u1", "u2"],
    })
    out = clean_posts(df, min_text_length=20)
    assert list(out["id"]) == ["1"]
    assert "model_text" in out.columns
    assert len(out.iloc[0].model_text) <= 1500


def test_select_posts_is_deterministic():
    df = pd.DataFrame({"id": ["a", "b", "c"], "score": [3, 9, 5], "num_comments": [1, 0, 4]})
    out = select_posts_for_analysis(df, 2)
    assert list(out.id) == ["b", "c"]


def test_engagement_formula():
    df = pd.DataFrame({"score": [10, 4], "num_comments": [2, 3]})
    out = calculate_engagement(df)
    assert list(out.engagement) == [14, 10]


def test_topic_keywords_returns_dict():
    df = pd.DataFrame({
        "topic_id": [0, 0, 1, 1],
        "model_text": ["python code api", "python api tool", "gaming game console", "game console gaming"],
    })
    result = extract_topic_keywords(df, top_n=3)
    assert set(result) == {0, 1}
    assert all(isinstance(v, list) for v in result.values())
