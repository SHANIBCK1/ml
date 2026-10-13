"""
Configuration settings for NewsInsight AI.
Centralizes paths, constants, label mappings, and styling configurations.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "newsinsight.db"
EVAL_RESULTS_PATH = MODELS_DIR / "evaluation_results.json"

# Ensure essential directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Standardized Categories
# Primary categories supported: Politics, Sports, Business, Technology, Science, Entertainment
CATEGORIES = [
    "Business",
    "Entertainment",
    "Politics",
    "Science",
    "Sports",
    "Technology",
]

# Standard category alias/normalization map
CATEGORY_MAPPING = {
    "world": "Politics",
    "politics": "Politics",
    "political": "Politics",
    "national": "Politics",
    "sports": "Sports",
    "sport": "Sports",
    "business": "Business",
    "economy": "Business",
    "finance": "Business",
    "sci/tech": "Technology",
    "science/technology": "Technology",
    "technology": "Technology",
    "tech": "Technology",
    "science": "Science",
    "sci": "Science",
    "entertainment": "Entertainment",
    "arts": "Entertainment",
    "culture": "Entertainment",
}

# Standardized Sentiment Classes
SENTIMENT_CLASSES = ["Negative", "Neutral", "Positive"]

SENTIMENT_MAPPING = {
    "positive": "Positive",
    "pos": "Positive",
    "1": "Positive",
    "negative": "Negative",
    "neg": "Negative",
    "-1": "Negative",
    "0": "Neutral",
    "neutral": "Neutral",
    "neu": "Neutral",
}

# Machine Learning Reproducibility & Pipeline Defaults
RANDOM_SEED = 42
TEST_SIZE = 0.20
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Theme & Palette Settings for Consistent UI Visualization
COLOR_PALETTE = {
    "background": "#0F172A",      # Slate 900
    "surface": "#1E293B",         # Slate 800
    "surface_border": "#334155",  # Slate 700
    "primary": "#3B82F6",         # Blue 500
    "primary_dark": "#1D4ED8",    # Blue 700
    "secondary": "#8B5CF6",       # Violet 500
    "positive": "#10B981",        # Emerald 500
    "negative": "#EF4444",        # Red 500
    "neutral": "#64748B",         # Slate 500
    "text_main": "#F8FAFC",       # Slate 50
    "text_muted": "#94A3B8",      # Slate 400
}

# Sentiment Color Mapping for Plotly Charts
SENTIMENT_COLORS = {
    "Positive": "#10B981",
    "Neutral": "#64748B",
    "Negative": "#EF4444",
}

# Category Color Mapping for Charts
CATEGORY_COLORS = {
    "Politics": "#F59E0B",       # Amber 500
    "Sports": "#06B6D4",         # Cyan 500
    "Business": "#3B82F6",       # Blue 500
    "Technology": "#8B5CF6",     # Violet 500
    "Science": "#10B981",        # Emerald 500
    "Entertainment": "#EC4899",  # Pink 500
}
