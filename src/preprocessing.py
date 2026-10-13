"""
Text preprocessing and feature extraction module for NewsInsight AI.
Ensures identical preprocessing across training, evaluation, and real-time inference.
"""

import re
import string
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Built-in robust English stopword list ensuring zero runtime failure even offline
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
    "yours", "yourself", "yourselves", "also", "said", "new", "says", "inc", "corp", "reuters",
    "ap", "year", "years", "day", "days", "time", "times"
}

# Try importing NLTK stopwords if available, fall back cleanly
try:
    import nltk
    from nltk.corpus import stopwords
    nltk_stops = set(stopwords.words('english'))
    STOPWORDS = STANDARD_STOPWORDS.union(nltk_stops)
except Exception:
    STOPWORDS = STANDARD_STOPWORDS


def clean_text(text: str) -> str:
    """
    Cleans raw text by:
    - Normalizing whitespace and lowercasing
    - Stripping URLs, HTML tags, and email addresses
    - Removing non-alphabetic characters
    - Stripping redundant spaces
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove numbers and non-alphabetic characters (keep letters and spaces)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str, custom_stopwords: Optional[set] = None) -> str:
    """Removes standard and custom stopwords from a cleaned string."""
    if not text:
        return ""
    
    stops = custom_stopwords if custom_stopwords is not None else STOPWORDS
    tokens = text.split()
    filtered = [token for token in tokens if token not in stops and len(token) > 2]
    return " ".join(filtered)


def preprocess_text(text: str) -> str:
    """
    Standard full preprocessing pipeline for inference and training:
    1. Clean text (lowercase, regex scrubbing)
    2. Filter stopwords and short tokens
    """
    cleaned = clean_text(text)
    filtered = remove_stopwords(cleaned)
    return filtered


def get_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 1,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Instantiates a standardized TF-IDF Vectorizer with sublinear term frequency.
    """
    return TfidfVectorizer(
        preprocessor=preprocess_text,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf
    )


def extract_top_features(
    text: str,
    vectorizer: TfidfVectorizer,
    model: Any,
    predicted_class: str,
    top_n: int = 6
) -> List[Dict[str, Any]]:
    """
    Explains the model prediction by identifying the top most influential
    features (words/n-grams) present in the input text.
    Works for LogisticRegression, MultinomialNB, and falls back to TF-IDF weights.
    """
    preprocessed = preprocess_text(text)
    if not preprocessed:
        return []

    # Transform input text into TF-IDF sparse vector
    tfidf_vec = vectorizer.transform([preprocessed])
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    nonzero_indices = tfidf_vec.nonzero()[1]
    if len(nonzero_indices) == 0:
        return []

    scores = []
    classes = list(getattr(model, "classes_", []))

    if hasattr(model, "coef_") and predicted_class in classes:
        # Logistic Regression: feature coefficient * tfidf weight
        class_idx = classes.index(predicted_class)
        # Handle binary vs multiclass coef shape
        if model.coef_.shape[0] == 1 and len(classes) == 2:
            coef = model.coef_[0] if class_idx == 1 else -model.coef_[0]
        else:
            coef = model.coef_[class_idx]
            
        for idx in nonzero_indices:
            term = feature_names[idx]
            tfidf_weight = tfidf_vec[0, idx]
            importance = float(coef[idx] * tfidf_weight)
            scores.append({"feature": term, "score": round(importance, 4), "tfidf": round(float(tfidf_weight), 4)})

    elif hasattr(model, "feature_log_prob_") and predicted_class in classes:
        # Multinomial Naive Bayes: feature_log_prob * tfidf weight
        class_idx = classes.index(predicted_class)
        log_prob = model.feature_log_prob_[class_idx]
        for idx in nonzero_indices:
            term = feature_names[idx]
            tfidf_weight = tfidf_vec[0, idx]
            # Higher log prob indicates higher probability under this class
            importance = float(log_prob[idx] * tfidf_weight)
            scores.append({"feature": term, "score": round(importance, 4), "tfidf": round(float(tfidf_weight), 4)})
            
    else:
        # Generic fallback: rank by TF-IDF magnitude in input
        for idx in nonzero_indices:
            term = feature_names[idx]
            tfidf_weight = float(tfidf_vec[0, idx])
            scores.append({"feature": term, "score": round(tfidf_weight, 4), "tfidf": round(tfidf_weight, 4)})

    # Sort by absolute or positive score descending
    scores = sorted(scores, key=lambda x: abs(x["score"]), reverse=True)[:top_n]
    return scores
