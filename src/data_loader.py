"""
Dataset loading, validation, and standardization module for NewsInsight AI.
Ensures schemas are consistent, handles missing data, normalizes labels, and prevents duplicates.
"""

import hashlib
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from config.settings import (
    CATEGORIES,
    CATEGORY_MAPPING,
    SENTIMENT_CLASSES,
    SENTIMENT_MAPPING,
    SAMPLE_DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)

# Standardized column definitions
REQUIRED_COLUMNS = ["headline"]
RECOMMENDED_COLUMNS = [
    "article_id", "published_at", "headline", "article_text",
    "category", "sentiment", "source"
]


def generate_article_id(headline: str, source: str = "", date_str: str = "") -> str:
    """Generates a reproducible unique ID using MD5 hash of headline and source."""
    raw = f"{str(headline).strip()}_{str(source).strip()}_{str(date_str).strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def normalize_category_label(raw_cat: Any) -> Optional[str]:
    """Maps varied category strings (e.g., 'sci/tech', 'world') to standard categories."""
    if pd.isna(raw_cat):
        return None
    val = str(raw_cat).strip().lower()
    if val in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[val]
    # Check direct title case match
    val_title = str(raw_cat).strip().capitalize()
    if val_title in CATEGORIES:
        return val_title
    return None


def normalize_sentiment_label(raw_sent: Any) -> Optional[str]:
    """Maps varied sentiment strings/codes to ['Negative', 'Neutral', 'Positive']."""
    if pd.isna(raw_sent):
        return None
    val = str(raw_sent).strip().lower()
    if val in SENTIMENT_MAPPING:
        return SENTIMENT_MAPPING[val]
    val_title = str(raw_sent).strip().capitalize()
    if val_title in SENTIMENT_CLASSES:
        return val_title
    return None


def validate_and_standardize_df(
    df: pd.DataFrame,
    require_labels: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates and standardizes a news dataframe.
    Returns:
        (standardized_df, validation_report)
    """
    report = {
        "initial_rows": len(df),
        "valid_rows": 0,
        "dropped_empty_text": 0,
        "dropped_duplicates": 0,
        "dropped_invalid_labels": 0,
        "warnings": []
    }

    if df.empty:
        report["warnings"].append("Provided dataframe is empty.")
        return pd.DataFrame(columns=RECOMMENDED_COLUMNS), report

    # Make working copy
    clean_df = df.copy()

    # Identify headline or text column
    # Fallback mappings for common CSV headers
    col_aliases = {
        "title": "headline",
        "Title": "headline",
        "Headline": "headline",
        "HEADLINE": "headline",
        "news_headline": "headline",
        "text": "article_text",
        "content": "article_text",
        "article": "article_text",
        "description": "article_text",
        "topic": "category",
        "label": "category",
        "Class Index": "category",
        "date": "published_at",
        "Date": "published_at",
        "published": "published_at",
        "publication_date": "published_at",
    }
    clean_df.rename(columns=col_aliases, inplace=True)

    if "headline" not in clean_df.columns:
        if "article_text" in clean_df.columns:
            # Derive headline from first sentence of article_text
            clean_df["headline"] = clean_df["article_text"].apply(
                lambda x: str(x).split(".")[0][:120] if pd.notna(x) else ""
            )
        else:
            report["warnings"].append("Missing required 'headline' column.")
            return pd.DataFrame(columns=RECOMMENDED_COLUMNS), report

    if "article_text" not in clean_df.columns:
        clean_df["article_text"] = clean_df["headline"]

    # Filter out empty or whitespace-only headlines
    clean_df["headline"] = clean_df["headline"].astype(str).str.strip()
    valid_text_mask = clean_df["headline"].str.len() > 3
    dropped_empty = len(clean_df) - valid_text_mask.sum()
    report["dropped_empty_text"] = int(dropped_empty)
    clean_df = clean_df[valid_text_mask].copy()

    # Remove duplicates based on headline text
    initial_before_dedup = len(clean_df)
    clean_df.drop_duplicates(subset=["headline"], keep="first", inplace=True)
    report["dropped_duplicates"] = int(initial_before_dedup - len(clean_df))

    # Standardize category labels
    if "category" in clean_df.columns:
        clean_df["category"] = clean_df["category"].apply(normalize_category_label)
        if require_labels:
            clean_df = clean_df[clean_df["category"].notna()].copy()
    else:
        clean_df["category"] = None

    # Standardize sentiment labels
    if "sentiment" in clean_df.columns:
        clean_df["sentiment"] = clean_df["sentiment"].apply(normalize_sentiment_label)
        if require_labels:
            clean_df = clean_df[clean_df["sentiment"].notna()].copy()
    else:
        clean_df["sentiment"] = None

    # Standardize published_at dates
    if "published_at" in clean_df.columns:
        clean_df["published_at"] = pd.to_datetime(clean_df["published_at"], errors="coerce")
    else:
        clean_df["published_at"] = pd.NaT

    # Standardize source
    if "source" not in clean_df.columns:
        clean_df["source"] = "Unknown"
    else:
        clean_df["source"] = clean_df["source"].fillna("Unknown").astype(str).str.strip()

    # Ensure article_id exists
    if "article_id" not in clean_df.columns or clean_df["article_id"].isna().any():
        clean_df["article_id"] = [
            generate_article_id(h, s, str(d))
            for h, s, d in zip(clean_df["headline"], clean_df["source"], clean_df["published_at"])
        ]

    # Reorder/ensure recommended columns
    for col in RECOMMENDED_COLUMNS:
        if col not in clean_df.columns:
            clean_df[col] = None

    clean_df = clean_df[RECOMMENDED_COLUMNS].reset_index(drop=True)
    report["valid_rows"] = int(len(clean_df))

    return clean_df, report


def load_news_dataset(
    filepath: Path,
    require_labels: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads a CSV dataset from disk and standardizes it."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    return validate_and_standardize_df(df, require_labels=require_labels)
