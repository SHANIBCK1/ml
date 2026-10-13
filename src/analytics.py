"""
Analytics engine for NewsInsight AI.
Computes daily sentiment aggregation, day-over-day shifts, topic-wise sentiment distributions,
and the Net Sentiment Score (News Mood index).
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


def compute_sentiment_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes overall counts and percentages of Positive, Neutral, Negative records.
    """
    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "positive_count": 0, "positive_pct": 0.0,
            "neutral_count": 0, "neutral_pct": 0.0,
            "negative_count": 0, "negative_pct": 0.0,
            "net_sentiment_score": 0.0,
            "mood_label": "No Data"
        }

    pos = int((df["sentiment"] == "Positive").sum())
    neu = int((df["sentiment"] == "Neutral").sum())
    neg = int((df["sentiment"] == "Negative").sum())

    pos_pct = round((pos / total) * 100.0, 1)
    neu_pct = round((neu / total) * 100.0, 1)
    neg_pct = round((neg / total) * 100.0, 1)
    net_score = round(pos_pct - neg_pct, 1)

    # Interpret Net Sentiment Score
    if net_score >= 30.0:
        mood = "Strongly Optimistic / Positive"
    elif net_score >= 10.0:
        mood = "Moderately Optimistic"
    elif net_score >= -10.0:
        mood = "Balanced / Neutral"
    elif net_score >= -30.0:
        mood = "Moderately Cautious / Negative"
    else:
        mood = "Strongly Pessimistic / Negative"

    return {
        "total": total,
        "positive_count": pos, "positive_pct": pos_pct,
        "neutral_count": neu, "neutral_pct": neu_pct,
        "negative_count": neg, "negative_pct": neg_pct,
        "net_sentiment_score": net_score,
        "mood_label": mood
    }


def compute_daily_sentiment_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates news records by publication date.
    Returns DataFrame with columns:
    [date, total_count, positive_count, neutral_count, negative_count,
     positive_pct, neutral_pct, negative_pct, dod_neg_pct_change]
    """
    if df.empty or "published_at" not in df.columns:
        return pd.DataFrame()

    valid_dates_df = df[df["published_at"].notna()].copy()
    if valid_dates_df.empty:
        return pd.DataFrame()

    valid_dates_df["date"] = valid_dates_df["published_at"].dt.date

    grouped = valid_dates_df.groupby("date")
    daily_records = []

    for dt, group in grouped:
        total = len(group)
        pos = int((group["sentiment"] == "Positive").sum())
        neu = int((group["sentiment"] == "Neutral").sum())
        neg = int((group["sentiment"] == "Negative").sum())

        daily_records.append({
            "date": dt,
            "total_count": total,
            "positive_count": pos,
            "neutral_count": neu,
            "negative_count": neg,
            "positive_pct": round((pos / total) * 100.0, 1) if total > 0 else 0.0,
            "neutral_pct": round((neu / total) * 100.0, 1) if total > 0 else 0.0,
            "negative_pct": round((neg / total) * 100.0, 1) if total > 0 else 0.0,
            "net_sentiment": round(((pos - neg) / total) * 100.0, 1) if total > 0 else 0.0
        })

    daily_df = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)
    if not daily_df.empty:
        # Calculate Day-over-Day shift in negative news percentage
        daily_df["dod_neg_pct_change"] = daily_df["negative_pct"].diff().round(1).fillna(0.0)

    return daily_df


def compute_topic_sentiment_breakdown(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Computes topic-wise sentiment breakdown (counts and percentages)
    along with the category that exhibits the highest negative sentiment percentage.
    Returns:
        (breakdown_df, highest_neg_summary)
    """
    if df.empty or "category" not in df.columns or "sentiment" not in df.columns:
        return pd.DataFrame(), None

    valid_df = df[df["category"].notna() & df["sentiment"].notna()].copy()
    if valid_df.empty:
        return pd.DataFrame(), None

    grouped = valid_df.groupby("category")
    rows = []

    for cat, group in grouped:
        total = len(group)
        pos = int((group["sentiment"] == "Positive").sum())
        neu = int((group["sentiment"] == "Neutral").sum())
        neg = int((group["sentiment"] == "Negative").sum())

        rows.append({
            "category": cat,
            "sample_size": total,
            "positive_count": pos,
            "neutral_count": neu,
            "negative_count": neg,
            "positive_pct": round((pos / total) * 100.0, 1) if total > 0 else 0.0,
            "neutral_pct": round((neu / total) * 100.0, 1) if total > 0 else 0.0,
            "negative_pct": round((neg / total) * 100.0, 1) if total > 0 else 0.0,
        })

    breakdown_df = pd.DataFrame(rows).sort_values("sample_size", ascending=False).reset_index(drop=True)

    # Identify category with highest negative sentiment (minimum 2 samples for statistical sense)
    qualifying = breakdown_df[breakdown_df["sample_size"] >= 2]
    highest_neg_summary = None
    if not qualifying.empty:
        top_neg_row = qualifying.sort_values("negative_pct", ascending=False).iloc[0]
        highest_neg_summary = {
            "category": top_neg_row["category"],
            "negative_pct": top_neg_row["negative_pct"],
            "sample_size": int(top_neg_row["sample_size"]),
            "negative_count": int(top_neg_row["negative_count"])
        }
    elif not breakdown_df.empty:
        top_neg_row = breakdown_df.sort_values("negative_pct", ascending=False).iloc[0]
        highest_neg_summary = {
            "category": top_neg_row["category"],
            "negative_pct": top_neg_row["negative_pct"],
            "sample_size": int(top_neg_row["sample_size"]),
            "negative_count": int(top_neg_row["negative_count"])
        }

    return breakdown_df, highest_neg_summary


def compute_period_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compares the current recent half of dated data with the previous half
    to calculate if negative sentiment increased or decreased.
    """
    if df.empty or "published_at" not in df.columns:
        return {"status": "insufficient_data"}

    dated_df = df[df["published_at"].notna()].sort_values("published_at").copy()
    if len(dated_df) < 4:
        return {"status": "insufficient_data"}

    midpoint = len(dated_df) // 2
    prev_period = dated_df.iloc[:midpoint]
    curr_period = dated_df.iloc[midpoint:]

    prev_dist = compute_sentiment_distribution(prev_period)
    curr_dist = compute_sentiment_distribution(curr_period)

    neg_diff = round(curr_dist["negative_pct"] - prev_dist["negative_pct"], 1)
    net_score_diff = round(curr_dist["net_sentiment_score"] - prev_dist["net_sentiment_score"], 1)

    return {
        "status": "success",
        "current_neg_pct": curr_dist["negative_pct"],
        "previous_neg_pct": prev_dist["negative_pct"],
        "neg_pct_change": neg_diff,
        "current_net_score": curr_dist["net_sentiment_score"],
        "previous_net_score": prev_dist["net_sentiment_score"],
        "net_score_change": net_score_diff,
        "direction": "increased" if neg_diff > 0 else ("decreased" if neg_diff < 0 else "unchanged")
    }
