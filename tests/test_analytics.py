"""
Unit tests for analytics calculations, daily trends, topic breakdowns, and News Mood score.
"""

import pytest
import pandas as pd
from datetime import datetime
from src.analytics import (
    compute_sentiment_distribution,
    compute_daily_sentiment_trends,
    compute_topic_sentiment_breakdown,
    compute_period_comparison
)


@pytest.fixture
def sample_analytics_df():
    data = [
        {"published_at": "2026-09-01", "category": "Technology", "sentiment": "Positive"},
        {"published_at": "2026-09-01", "category": "Technology", "sentiment": "Positive"},
        {"published_at": "2026-09-01", "category": "Business", "sentiment": "Negative"},
        {"published_at": "2026-09-01", "category": "Politics", "sentiment": "Neutral"},
        {"published_at": "2026-09-02", "category": "Business", "sentiment": "Negative"},
        {"published_at": "2026-09-02", "category": "Politics", "sentiment": "Negative"},
        {"published_at": "2026-09-02", "category": "Sports", "sentiment": "Positive"},
        {"published_at": "2026-09-02", "category": "Sports", "sentiment": "Neutral"},
    ]
    df = pd.DataFrame(data)
    df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def test_compute_sentiment_distribution(sample_analytics_df):
    dist = compute_sentiment_distribution(sample_analytics_df)
    assert dist["total"] == 8
    assert dist["positive_count"] == 3
    assert dist["negative_count"] == 3
    assert dist["neutral_count"] == 2
    # 3/8 = 37.5%, 3/8 = 37.5% -> net score = 0.0
    assert dist["net_sentiment_score"] == 0.0
    assert "Balanced" in dist["mood_label"]


def test_compute_sentiment_distribution_empty():
    dist = compute_sentiment_distribution(pd.DataFrame())
    assert dist["total"] == 0
    assert dist["net_sentiment_score"] == 0.0
    assert dist["mood_label"] == "No Data"


def test_compute_daily_sentiment_trends(sample_analytics_df):
    daily = compute_daily_sentiment_trends(sample_analytics_df)
    assert len(daily) == 2
    day1 = daily.iloc[0]
    assert day1["total_count"] == 4
    assert day1["positive_count"] == 2
    assert day1["negative_count"] == 1
    assert day1["neutral_count"] == 1
    assert day1["positive_pct"] == 50.0
    assert day1["negative_pct"] == 25.0

    day2 = daily.iloc[1]
    assert day2["negative_pct"] == 50.0
    # Day-over-day negative pct change from 25.0 to 50.0 is +25.0
    assert day2["dod_neg_pct_change"] == 25.0


def test_compute_topic_sentiment_breakdown(sample_analytics_df):
    breakdown, top_neg = compute_topic_sentiment_breakdown(sample_analytics_df)
    assert not breakdown.empty
    assert top_neg is not None
    # In sample_analytics_df, Business has 2 negative out of 2 samples (100% negative)
    assert top_neg["category"] == "Business"
    assert top_neg["negative_pct"] == 100.0


def test_compute_period_comparison(sample_analytics_df):
    comp = compute_period_comparison(sample_analytics_df)
    assert comp["status"] == "success"
    # First half has 1 neg out of 4 (25%), second half has 2 neg out of 4 (50%)
    assert comp["direction"] == "increased"
    assert comp["neg_pct_change"] == 25.0
