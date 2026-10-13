"""
Unit tests for SQLite storage, batch inserting, deduplication, and queries.
"""

import pytest
import tempfile
from pathlib import Path
from src.storage import (
    init_db,
    save_article,
    save_articles_batch,
    get_all_records,
    get_filtered_records,
    clear_all_records,
    get_record_count
)


@pytest.fixture
def temp_db():
    temp_dir = tempfile.TemporaryDirectory()
    db_file = Path(temp_dir.name) / "test_news.db"
    init_db(db_file)
    yield db_file
    temp_dir.cleanup()


def test_save_article_and_prevent_duplicate(temp_db):
    record = {
        "article_id": "test_001",
        "headline": "Historic scientific breakthrough in renewable energy",
        "category": "Science",
        "sentiment": "Positive",
        "source": "ScienceDaily",
        "topic_confidence": 0.92,
        "sentiment_confidence": 0.88
    }

    # First insert should succeed
    inserted = save_article(record, db_path=temp_db)
    assert inserted is True
    assert get_record_count(db_path=temp_db) == 1

    # Second insert with identical article_id should be ignored
    dup_inserted = save_article(record, db_path=temp_db)
    assert dup_inserted is False
    assert get_record_count(db_path=temp_db) == 1


def test_save_articles_batch_and_query(temp_db):
    batch = [
        {"article_id": "t1", "headline": "Headline A", "category": "Politics", "sentiment": "Neutral", "source": "Reuters"},
        {"article_id": "t2", "headline": "Headline B", "category": "Sports", "sentiment": "Positive", "source": "ESPN"},
        {"article_id": "t3", "headline": "Headline C", "category": "Business", "sentiment": "Negative", "source": "Bloomberg"},
    ]
    saved_count = save_articles_batch(batch, db_path=temp_db)
    assert saved_count == 3
    assert get_record_count(db_path=temp_db) == 3

    # Filter by category
    politics_df = get_filtered_records(categories=["Politics"], db_path=temp_db)
    assert len(politics_df) == 1
    assert politics_df.iloc[0]["headline"] == "Headline A"

    # Filter by sentiment
    pos_df = get_filtered_records(sentiments=["Positive"], db_path=temp_db)
    assert len(pos_df) == 1
    assert pos_df.iloc[0]["headline"] == "Headline B"


def test_clear_all_records(temp_db):
    record = {"article_id": "t_del", "headline": "Sample headline to delete"}
    save_article(record, db_path=temp_db)
    assert get_record_count(db_path=temp_db) == 1

    deleted = clear_all_records(db_path=temp_db)
    assert deleted == 1
    assert get_record_count(db_path=temp_db) == 0
