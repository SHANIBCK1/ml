"""
Local SQLite database storage module for NewsInsight AI.
Provides persistent storage across app restarts, batch insertions,
duplicate avoidance, filtering queries, and CSV export.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from config.settings import DB_PATH
from src.data_loader import generate_article_id


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with row dictionary access."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Creates the necessary tables and indexes if they do not exist."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyzed_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id TEXT UNIQUE NOT NULL,
                published_at TEXT,
                headline TEXT NOT NULL,
                article_text TEXT,
                category TEXT,
                sentiment TEXT,
                topic_confidence REAL,
                sentiment_confidence REAL,
                model_version TEXT,
                source TEXT DEFAULT 'Manual Input',
                analysis_timestamp TEXT NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_published_at ON analyzed_news(published_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON analyzed_news(category);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON analyzed_news(sentiment);")
    conn.close()


def save_article(record: Dict[str, Any], db_path: Path = DB_PATH) -> bool:
    """
    Saves a single analyzed article record.
    Prevents duplicates based on article_id.
    Returns True if inserted, False if duplicate or failed.
    """
    init_db(db_path)
    headline = str(record.get("headline", "")).strip()
    if not headline:
        return False

    source = str(record.get("source", "Manual Input")).strip()
    pub_date = str(record.get("published_at", "")).strip()
    
    article_id = record.get("article_id") or generate_article_id(headline, source, pub_date)
    analysis_ts = record.get("analysis_timestamp") or datetime.now(timezone.utc).isoformat()

    conn = get_db_connection(db_path)
    inserted = False
    try:
        with conn:
            conn.execute("""
                INSERT OR IGNORE INTO analyzed_news (
                    article_id, published_at, headline, article_text,
                    category, sentiment, topic_confidence, sentiment_confidence,
                    model_version, source, analysis_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article_id,
                pub_date if pub_date and pub_date != "NaT" else None,
                headline,
                record.get("article_text", headline),
                record.get("category"),
                record.get("sentiment"),
                float(record.get("topic_confidence", 0.0)),
                float(record.get("sentiment_confidence", 0.0)),
                record.get("model_version", "v1.0"),
                source,
                analysis_ts
            ))
            inserted = conn.total_changes > 0
    except Exception as e:
        print(f"Error saving article: {e}")
        inserted = False
    finally:
        conn.close()

    return inserted


def save_articles_batch(records: List[Dict[str, Any]], db_path: Path = DB_PATH) -> int:
    """
    Batch inserts multiple analyzed articles, ignoring duplicates.
    Returns the count of newly inserted rows.
    """
    if not records:
        return 0

    init_db(db_path)
    conn = get_db_connection(db_path)
    inserted_count = 0

    try:
        with conn:
            cursor = conn.cursor()
            for r in records:
                headline = str(r.get("headline", "")).strip()
                if not headline:
                    continue
                source = str(r.get("source", "Dataset/Batch")).strip()
                pub_date = str(r.get("published_at", "")).strip()
                article_id = r.get("article_id") or generate_article_id(headline, source, pub_date)
                analysis_ts = r.get("analysis_timestamp") or datetime.now(timezone.utc).isoformat()

                cursor.execute("""
                    INSERT OR IGNORE INTO analyzed_news (
                        article_id, published_at, headline, article_text,
                        category, sentiment, topic_confidence, sentiment_confidence,
                        model_version, source, analysis_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article_id,
                    pub_date if pub_date and pub_date != "NaT" else None,
                    headline,
                    r.get("article_text", headline),
                    r.get("category"),
                    r.get("sentiment"),
                    float(r.get("topic_confidence", 0.0)),
                    float(r.get("sentiment_confidence", 0.0)),
                    r.get("model_version", "v1.0"),
                    source,
                    analysis_ts
                ))
            inserted_count = conn.total_changes
    except Exception as e:
        print(f"Error in batch insertion: {e}")
    finally:
        conn.close()

    return inserted_count


def get_all_records(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Retrieves all analyzed news records as a pandas DataFrame."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM analyzed_news ORDER BY published_at DESC, id DESC",
            conn
        )
        if not df.empty and "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        return df
    finally:
        conn.close()


def get_filtered_records(
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    categories: Optional[List[str]] = None,
    sentiments: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    db_path: Path = DB_PATH
) -> pd.DataFrame:
    """
    Queries records filtered by dates, categories, sentiments, and sources.
    """
    df = get_all_records(db_path)
    if df.empty:
        return df

    # Date filter
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df = df[df["published_at"].isna() | (df["published_at"] >= start_dt)]
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        df = df[df["published_at"].isna() | (df["published_at"] <= end_dt)]

    # Category filter
    if categories:
        df = df[df["category"].isin(categories)]

    # Sentiment filter
    if sentiments:
        df = df[df["sentiment"].isin(sentiments)]

    # Source filter
    if sources:
        df = df[df["source"].isin(sources)]

    return df


def clear_all_records(db_path: Path = DB_PATH) -> int:
    """Deletes all records from the database. Returns count of deleted rows."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    deleted_count = 0
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM analyzed_news")
            deleted_count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM analyzed_news")
    finally:
        conn.close()
    return deleted_count


def get_record_count(db_path: Path = DB_PATH) -> int:
    """Returns the total number of records currently in the database."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analyzed_news")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def seed_sample_data_if_empty(sample_csv: Optional[Path] = None, db_path: Path = DB_PATH) -> int:
    """Seeds the database with sample news records if it currently has 0 records."""
    count = get_record_count(db_path)
    if count > 0:
        return 0

    from config.settings import SAMPLE_DATA_DIR
    csv_file = sample_csv or (SAMPLE_DATA_DIR / "sample_news.csv")
    if not csv_file.exists():
        return 0

    from src.data_loader import load_news_dataset
    df, report = load_news_dataset(csv_file)
    if df.empty:
        return 0

    records = df.to_dict(orient="records")
    for r in records:
        r["topic_confidence"] = 0.88
        r["sentiment_confidence"] = 0.85
        r["model_version"] = "v1.0-benchmark"
        r["analysis_timestamp"] = str(r.get("published_at") or datetime.utcnow().isoformat())

    return save_articles_batch(records, db_path=db_path)

