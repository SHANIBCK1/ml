"""
NewsInsight AI - Main Application Landing Page
AI-Based News Classification & Sentiment Intelligence
"""

import streamlit as st
from pathlib import Path
import pandas as pd

from config.settings import CATEGORIES, SENTIMENT_CLASSES, EVAL_RESULTS_PATH
from src.ui_components import render_header, render_metric_card, apply_custom_theme
from src.storage import get_record_count, seed_sample_data_if_empty
from src.model_evaluation import load_evaluation_results

st.set_page_config(
    page_title="NewsInsight AI | News Intelligence Platform",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_theme()

# Header banner
render_header(
    title="NewsInsight AI",
    subtitle="AI-Based News Classification & Sentiment Intelligence Platform",
    badge_text="Academic Machine Learning Lab Project"
)

# Ensure sample data is seeded and models exist for zero-setup cloud deployments
seed_sample_data_if_empty()

from src.topic_classifier import TOPIC_MODEL_PATH
from src.sentiment_classifier import SENTIMENT_MODEL_PATH

if not TOPIC_MODEL_PATH.exists() or not SENTIMENT_MODEL_PATH.exists():
    try:
        from scripts.prepare_data import main as prep_main
        from scripts.train_models import train_all_models
        prep_main()
        train_all_models()
    except Exception as e:
        print(f"Notice during first-time cloud setup: {e}")

# Load real metrics
total_db_records = get_record_count()
eval_data = load_evaluation_results()

topic_best_model = "Trained"
topic_acc = "N/A"
topic_f1 = "N/A"
if eval_data and "topic_classification" in eval_data:
    t_info = eval_data["topic_classification"]
    topic_best_model = t_info.get("selected_model", "Trained")
    best_t_dict = t_info.get("models", {}).get(topic_best_model, {})
    topic_acc = f"{best_t_dict.get('accuracy', 0.0) * 100:.1f}%"
    topic_f1 = f"{best_t_dict.get('macro_f1', 0.0):.3f}"

sent_best_model = "Trained"
sent_acc = "N/A"
sent_f1 = "N/A"
if eval_data and "sentiment_classification" in eval_data:
    s_info = eval_data["sentiment_classification"]
    sent_best_model = s_info.get("selected_model", "Trained")
    best_s_dict = s_info.get("models", {}).get(sent_best_model, {})
    sent_acc = f"{best_s_dict.get('accuracy', 0.0) * 100:.1f}%"
    sent_f1 = f"{best_s_dict.get('macro_f1', 0.0):.3f}"

# Top KPI row
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card(
        title="Stored News Records",
        value=f"{total_db_records:,}",
        subtitle="SQLite Local Storage",
        border_color="#3B82F6"
    )
with col2:
    render_metric_card(
        title="Best Topic Classifier",
        value=topic_acc,
        subtitle=f"{topic_best_model} (Macro-F1: {topic_f1})",
        border_color="#8B5CF6"
    )
with col3:
    render_metric_card(
        title="Best Sentiment Model",
        value=sent_acc,
        subtitle=f"{sent_best_model} (Macro-F1: {sent_f1})",
        border_color="#10B981"
    )
with col4:
    render_metric_card(
        title="Active Categories",
        value=f"{len(CATEGORIES)}",
        subtitle="Politics, Sports, Tech, etc.",
        border_color="#F59E0B"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main project description & features
left_col, right_col = st.columns([1.8, 1.2])

with left_col:
    st.markdown("""
        ### 🔍 Project Overview
        **NewsInsight AI** is an academic Machine Learning Lab project developed by a three-member student engineering team. 
        It integrates Natural Language Processing (NLP) and supervised Machine Learning algorithms to analyze, categorize, 
        and monitor temporal sentiment patterns in news headlines and articles.

        Unlike conventional systems that stop at one-off text categorization, NewsInsight AI captures the **temporal trajectory** 
        of news reporting, calculating **daily sentiment shifts**, **topic-wise sentiment distributions**, and an empirical 
        **Net Sentiment Score (News Mood Index)**.
    """)

    st.markdown("""
        <div class="callout-box">
            <b>💡 Explainability & Integrity:</b> All predictions are computed in real time using scikit-learn TF-IDF pipelines 
            trained on verified benchmark splits. The platform explicitly highlights top contributing vocabulary features 
            for each prediction and treats predictions as model estimates rather than absolute factual truths.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚀 Core Modules")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("""
            **1. [News Analyzer](News_Analyzer)**
            - Interactive headline & article analysis
            - Real-time dual topic & sentiment inference
            - Model confidence scores & influential feature keywords
            - Batch CSV processing & local SQLite storage
        """)
        st.markdown("""
            **2. [Analytics Dashboard](Analytics_Dashboard)**
            - Daily news volume and sentiment trends
            - Day-over-Day changes in negative reporting
            - Stacked topic-wise sentiment distributions
            - Net Sentiment Score & News Mood gauge
        """)
    with f_col2:
        st.markdown("""
            **3. [Model Evaluation](Model_Evaluation)**
            - Head-to-head comparison: Naive Bayes, Logistic Regression, KNN
            - Confusion matrices with annotated class heatmaps
            - Macro-F1, Precision, Recall, and Support reports
            - Interactive retrain trigger for updated data
        """)
        st.markdown("""
            **4. [About Project](About_Project)**
            - Complete academic documentation & viva voce guide
            - 3-Member work division breakdown
            - Mathematical formulations & algorithm details
            - Ethical considerations & system limitations
        """)

with right_col:
    st.markdown("### ⚙️ System Status")
    
    st.markdown(f"""
        <div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 20px;">
            <div style="margin-bottom: 12px;">
                <span style="color: #94A3B8; font-size: 0.85rem;">TOPIC CLASSIFIER PIPELINE</span><br>
                <b style="color: #38BDF8; font-size: 1.1rem;">TF-IDF + {topic_best_model}</b>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="color: #94A3B8; font-size: 0.85rem;">SENTIMENT CLASSIFIER PIPELINE</span><br>
                <b style="color: #34D399; font-size: 1.1rem;">TF-IDF + {sent_best_model}</b>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="color: #94A3B8; font-size: 0.85rem;">LOCAL STORAGE</span><br>
                <b style="color: #F8FAFC; font-size: 1.1rem;">SQLite (newsinsight.db)</b>
            </div>
            <div>
                <span style="color: #94A3B8; font-size: 0.85rem;">STATUS</span><br>
                <span class="badge badge-positive">✓ Active & Ready</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if total_db_records == 0:
        st.warning("Database currently has 0 records. Seed with sample data?")
        if st.button("🌱 Seed Sample News Dataset"):
            count = seed_sample_data_if_empty()
            st.success(f"Seeded {count} records! Refresh to view updated metrics.")
            st.rerun()

# Sidebar Navigation and Information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/news.png", width=64)
    st.markdown("### **NewsInsight AI**")
    st.caption("AI-Based News Classification & Sentiment Intelligence")
    st.markdown("---")
    st.markdown("**Team Work Division:**")
    st.markdown("- **Member 1:** Data, Preprocessing, TF-IDF")
    st.markdown("- **Member 2:** ML Models, Training, Evaluation")
    st.markdown("- **Member 3:** Dashboard, SQLite, Visuals")
    st.markdown("---")
    st.caption("Machine Learning Lab | Academic Project 2026")
