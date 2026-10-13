"""
Unit tests for text preprocessing, tokenization, and feature explanation.
"""

import pytest
from src.preprocessing import clean_text, remove_stopwords, preprocess_text, get_tfidf_vectorizer


def test_clean_text_removes_urls_and_html():
    raw = "Breaking: <p>Visit https://example.com/live or www.news.org for updates!</p> #BreakingNews"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "www" not in cleaned
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "breaking visit or for updates breakingnews" in cleaned


def test_clean_text_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n\t ") == ""


def test_remove_stopwords():
    text = "the president of the nation announced a new economic policy"
    filtered = remove_stopwords(text)
    # 'the', 'of', 'a', 'new' should be removed
    assert "president" in filtered
    assert "nation" in filtered
    assert "announced" in filtered
    assert "economic" in filtered
    assert "policy" in filtered
    assert "the" not in filtered.split()


def test_preprocess_text_pipeline():
    raw = "ISRO successfully launches a new satellite into space!"
    result = preprocess_text(raw)
    assert "isro" in result
    assert "successfully" in result
    assert "launches" in result
    assert "satellite" in result
    assert "space" in result


def test_tfidf_vectorizer_fit_transform():
    corpus = [
        "Economy records rapid quarterly growth in manufacturing sector.",
        "Scientists discover novel exoplanet orbiting nearby star system.",
        "National soccer squad wins international championship cup final."
    ]
    vec = get_tfidf_vectorizer(max_features=100)
    matrix = vec.fit_transform(corpus)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] > 0
