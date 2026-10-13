"""
Model training and evaluation pipeline for NewsInsight AI.
Trains topic models (MNB, LogReg, KNN) and sentiment models (MNB, LogReg).
Saves models, vectorizers, and metrics to the models/ directory.

Usage:
    python scripts/train_models.py
"""

import sys
from pathlib import Path
import json
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    EVAL_RESULTS_PATH
)
from src.topic_classifier import TopicClassificationEngine
from src.sentiment_classifier import SentimentClassificationEngine
from src.model_evaluation import save_evaluation_results


def train_all_models():
    print("=" * 65)
    print("NewsInsight AI - Machine Learning Model Training Pipeline")
    print("=" * 65)

    topic_file = PROCESSED_DATA_DIR / "topic_news.csv"
    sentiment_file = PROCESSED_DATA_DIR / "sentiment_news.csv"

    if not topic_file.exists() or not sentiment_file.exists():
        print("Processed datasets not found. Running dataset preparation first...")
        from scripts.prepare_data import main as prep_main
        prep_main()

    # 1. Train Topic Classification Models
    print("\n[Stage 1/2] Training Topic Classifiers (MNB, Logistic Regression, KNN)...")
    topic_df = pd.read_csv(topic_file)
    print(f"Loaded {len(topic_df)} records for topic classification.")
    
    topic_engine = TopicClassificationEngine()
    topic_summary = topic_engine.train_and_compare(topic_df)
    
    print("\n--- Topic Classification Results ---")
    for name, res in topic_summary["models"].items():
        print(f"  * {name:25s} | Acc: {res['accuracy']:.4f} | Macro-F1: {res['macro_f1']:.4f} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f}")
    print(f"\n=> Best Topic Model Selected: {topic_summary['selected_model']} (Macro-F1: {topic_summary['best_macro_f1']:.4f})")

    # 2. Train Sentiment Classification Models
    print("\n[Stage 2/2] Training Sentiment Classifiers (MNB, Logistic Regression)...")
    sentiment_df = pd.read_csv(sentiment_file)
    print(f"Loaded {len(sentiment_df)} records for sentiment classification.")
    
    sentiment_engine = SentimentClassificationEngine()
    sentiment_summary = sentiment_engine.train_and_compare(sentiment_df)
    
    print("\n--- Sentiment Classification Results ---")
    for name, res in sentiment_summary["models"].items():
        print(f"  * {name:25s} | Acc: {res['accuracy']:.4f} | Macro-F1: {res['macro_f1']:.4f} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f}")
    print(f"\n=> Best Sentiment Model Selected: {sentiment_summary['selected_model']} (Macro-F1: {sentiment_summary['best_macro_f1']:.4f})")

    # 3. Combine and Save Consolidated Evaluation Metadata
    consolidated_eval = {
        "topic_classification": {
            "selected_model": topic_summary["selected_model"],
            "models": topic_summary["models"]
        },
        "sentiment_classification": {
            "selected_model": sentiment_summary["selected_model"],
            "models": sentiment_summary["models"]
        }
    }
    save_evaluation_results(consolidated_eval)
    print(f"\nConsolidated evaluation metrics saved to {EVAL_RESULTS_PATH}")
    print("Model training pipeline finished successfully!")


if __name__ == "__main__":
    train_all_models()
