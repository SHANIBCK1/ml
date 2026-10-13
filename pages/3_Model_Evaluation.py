"""
Model Evaluation and Algorithm Comparison Page for NewsInsight AI.
Displays verified evaluation metrics, confusion matrices, and classification reports
for both Topic Classification and Sentiment Classification.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.settings import EVAL_RESULTS_PATH, MODELS_DIR
from src.ui_components import render_header, render_metric_card, apply_custom_theme
from src.model_evaluation import (
    load_evaluation_results,
    create_model_comparison_bar_chart,
    create_confusion_matrix_heatmap,
    create_classification_report_table
)
from src.topic_classifier import TopicClassificationEngine
from src.sentiment_classifier import SentimentClassificationEngine

st.set_page_config(page_title="Model Evaluation | NewsInsight AI", page_icon="🧪", layout="wide")
apply_custom_theme()

render_header(
    title="Model Evaluation & Benchmark Comparison",
    subtitle="Rigorous empirical evaluation on stratified held-out test datasets",
    badge_text="Scikit-Learn Evaluation"
)

# Load existing verified evaluation results
eval_data = load_evaluation_results()

# Re-train trigger in top section
top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    st.markdown("""
        <div class="callout-box">
            <b>Evaluation Protocol:</b> All reported metrics are computed on a strictly stratified 20% held-out test split. 
            TF-IDF vectorizers are fit solely on the training partition to eliminate data leakage. 
            Topic classification and sentiment classification are evaluated independently.
        </div>
    """, unsafe_allow_html=True)

with top_col2:
    if st.button("🔄 Retrain & Re-evaluate Models", key="btn_retrain"):
        with st.spinner("Training models and computing evaluation metrics..."):
            from scripts.train_models import train_all_models
            train_all_models()
            eval_data = load_evaluation_results()
        st.success("Retraining complete! Evaluation metrics updated.")
        st.rerun()

if not eval_data:
    st.warning("⚠️ Models have not been evaluated yet. Please click 'Retrain & Re-evaluate Models' above or execute `python scripts/train_models.py`.")
    st.stop()

tab_topic, tab_sent = st.tabs(["📚 News Topic Classification", "💬 Supervised Sentiment Classification"])

# ==========================================
# TAB 1: TOPIC CLASSIFICATION EVALUATION
# ==========================================
with tab_topic:
    st.markdown("### 📚 Topic Classification: MNB vs Logistic Regression vs KNN")
    
    topic_eval = eval_data.get("topic_classification", {})
    topic_models = topic_eval.get("models", {})
    selected_topic_model = topic_eval.get("selected_model", "Multinomial Naive Bayes")

    if not topic_models:
        st.info("Topic models not evaluated.")
    else:
        # Top KPI summary for best topic model
        best_t_dict = topic_models.get(selected_topic_model, {})
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            render_metric_card("Selected Model", selected_topic_model, "Highest Macro-F1", "#3B82F6")
        with m_c2:
            render_metric_card("Test Accuracy", f"{best_t_dict.get('accuracy', 0.0) * 100:.2f}%", "Overall Correct", "#10B981")
        with m_c3:
            render_metric_card("Macro-F1 Score", f"{best_t_dict.get('macro_f1', 0.0):.4f}", "Balanced Metric", "#8B5CF6")
        with m_c4:
            render_metric_card("Test Set Size", f"{best_t_dict.get('test_size', 0)}", f"Train size: {best_t_dict.get('train_size', 0)}", "#F59E0B")

        st.markdown("<br>", unsafe_allow_html=True)

        # Comparison Bar Chart
        fig_comp = create_model_comparison_bar_chart(
            topic_models,
            title="Topic Classification: Performance Across Algorithms"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Confusion Matrix & Classification Report for selected model
        st.markdown("#### 🔍 Detailed Diagnostics for Selected Topic Model")
        sub_c1, sub_c2 = st.columns([1.2, 1.8])

        with sub_c1:
            cm = best_t_dict.get("confusion_matrix", [])
            labels = best_t_dict.get("labels", [])
            if cm and labels:
                fig_cm = create_confusion_matrix_heatmap(cm, labels, selected_topic_model)
                st.plotly_chart(fig_cm, use_container_width=True)

        with sub_c2:
            report_dict = best_t_dict.get("classification_report", {})
            if report_dict:
                rep_df = create_classification_report_table(report_dict)
                st.markdown("**Per-Class Classification Report:**")
                st.dataframe(rep_df, use_container_width=True)


# ==========================================
# TAB 2: SENTIMENT CLASSIFICATION EVALUATION
# ==========================================
with tab_sent:
    st.markdown("### 💬 Sentiment Classification: MNB vs Logistic Regression")

    sent_eval = eval_data.get("sentiment_classification", {})
    sent_models = sent_eval.get("models", {})
    selected_sent_model = sent_eval.get("selected_model", "Multinomial Naive Bayes")

    if not sent_models:
        st.info("Sentiment models not evaluated.")
    else:
        # Top KPI summary for best sentiment model
        best_s_dict = sent_models.get(selected_sent_model, {})
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        with s_c1:
            render_metric_card("Selected Model", selected_sent_model, "Highest Macro-F1", "#10B981")
        with s_c2:
            render_metric_card("Test Accuracy", f"{best_s_dict.get('accuracy', 0.0) * 100:.2f}%", "Overall Correct", "#3B82F6")
        with s_c3:
            render_metric_card("Macro-F1 Score", f"{best_s_dict.get('macro_f1', 0.0):.4f}", "Class-Balanced Metric", "#8B5CF6")
        with s_c4:
            render_metric_card("Test Set Size", f"{best_s_dict.get('test_size', 0)}", f"Train size: {best_s_dict.get('train_size', 0)}", "#F59E0B")

        st.markdown("<br>", unsafe_allow_html=True)

        # Comparison Bar Chart
        fig_s_comp = create_model_comparison_bar_chart(
            sent_models,
            title="Sentiment Classification: Performance Across Algorithms"
        )
        st.plotly_chart(fig_s_comp, use_container_width=True)

        # Confusion Matrix & Classification Report for selected sentiment model
        st.markdown("#### 🔍 Detailed Diagnostics for Selected Sentiment Model")
        s_sub1, s_sub2 = st.columns([1.2, 1.8])

        with s_sub1:
            cm = best_s_dict.get("confusion_matrix", [])
            labels = best_s_dict.get("labels", [])
            if cm and labels:
                fig_s_cm = create_confusion_matrix_heatmap(cm, labels, selected_sent_model)
                st.plotly_chart(fig_s_cm, use_container_width=True)

        with s_sub2:
            report_dict = best_s_dict.get("classification_report", {})
            if report_dict:
                rep_df = create_classification_report_table(report_dict)
                st.markdown("**Per-Class Classification Report (Positive, Negative, Neutral):**")
                st.dataframe(rep_df, use_container_width=True)

        st.markdown("""
            <div class="callout-box">
                <b>⚖️ Class Imbalance Handling:</b> In Logistic Regression, <code>class_weight='balanced'</code> is used 
                to penalize misclassifications proportionally to class frequencies, preventing majority class bias.
            </div>
        """, unsafe_allow_html=True)
