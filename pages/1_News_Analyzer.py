"""
News Analyzer Page for NewsInsight AI.
Provides single headline/article inference with model explainability,
batch CSV upload and processing, and local database saving.
"""

from datetime import datetime, timezone
import io
import streamlit as st
import pandas as pd
import plotly.express as px

from config.settings import CATEGORIES, SENTIMENT_CLASSES, CATEGORY_COLORS, SENTIMENT_COLORS
from src.ui_components import render_header, render_metric_card, apply_custom_theme
from src.topic_classifier import TopicClassificationEngine
from src.sentiment_classifier import SentimentClassificationEngine
from src.storage import save_article, save_articles_batch
from src.data_loader import validate_and_standardize_df

st.set_page_config(page_title="News Analyzer | NewsInsight AI", page_icon="📰", layout="wide")
apply_custom_theme()

render_header(
    title="News Analyzer",
    subtitle="Real-time multi-task classification for news headlines and articles",
    badge_text="Inference & Explainability"
)

# Load engines
@st.cache_resource
def get_engines():
    topic_eng = TopicClassificationEngine()
    sent_eng = SentimentClassificationEngine()
    return topic_eng, sent_eng

topic_engine, sentiment_engine = get_engines()

# Check model availability
if topic_engine.model is None or sentiment_engine.model is None:
    st.error("⚠️ Trained models are not loaded. Please ensure models are trained in the 'Model Evaluation' page or via `python scripts/train_models.py`.")

tab_single, tab_batch = st.tabs(["📝 Single News Analyzer", "📁 Batch CSV Analyzer"])

# ==========================================
# TAB 1: SINGLE NEWS ANALYZER
# ==========================================
with tab_single:
    st.markdown("#### Enter News Headline or Article")
    
    # Pre-canned examples for quick testing
    example_prompts = [
        "-- Select an illustrative example --",
        "ISRO successfully launches a new satellite into space.",
        "Global stock markets plunge following unexpected interest rate hike.",
        "National football team secures thrilling victory in championship qualifier.",
        "Cybersecurity agency warns of critical zero-day vulnerability in banking software.",
        "Parliament approves landmark bipartisan renewable energy transition bill."
    ]
    selected_example = st.selectbox("Quick Test Examples:", example_prompts)

    default_text = ""
    if selected_example != "-- Select an illustrative example --":
        default_text = selected_example

    user_input = st.text_area(
        "Headline or Article Content:",
        value=default_text,
        height=110,
        placeholder="e.g. Prime Minister announces comprehensive infrastructure investment package..."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        pub_date = st.date_input("Publication Date:", value=datetime.now())
    with c2:
        news_source = st.text_input("News Source (Optional):", value="Manual Input")
    with c3:
        auto_save = st.checkbox("Save analysis to Local Database", value=True)

    analyze_btn = st.button("🚀 Analyze News", key="btn_single_analyze")

    if analyze_btn:
        if not user_input or len(user_input.strip()) < 4:
            st.warning("⚠️ Input is too short or empty. Please enter a meaningful news headline or sentence.")
        else:
            with st.spinner("Analyzing text using NLP pipelines..."):
                topic_res = topic_engine.predict(user_input)
                sent_res = sentiment_engine.predict(user_input)

            if topic_res["status"] == "error" or sent_res["status"] == "error":
                st.error(f"Error during inference: {topic_res.get('message', '')} {sent_res.get('message', '')}")
            else:
                st.success("Analysis Complete!")

                # Display Results in 2 Metric Cards
                res_col1, res_col2 = st.columns(2)

                # 1. Topic Result
                with res_col1:
                    topic_label = topic_res["predicted_category"]
                    topic_conf = topic_res["confidence"] * 100
                    st.markdown(f"""
                        <div class="metric-card" style="border-top: 3px solid #3B82F6;">
                            <div class="metric-title">Predicted News Category</div>
                            <div class="metric-value" style="color: #60A5FA;">{topic_label}</div>
                            <div class="metric-subtitle">
                                Model: <b>{topic_res['model_name']}</b> | Confidence: <b>{topic_conf:.1f}%</b>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(topic_res["confidence"], text=f"Category Confidence: {topic_conf:.1f}%")

                # 2. Sentiment Result
                with res_col2:
                    sent_label = sent_res["predicted_sentiment"]
                    sent_conf = sent_res["confidence"] * 100
                    badge_style = "badge-positive" if sent_label == "Positive" else ("badge-negative" if sent_label == "Negative" else "badge-neutral")
                    color = "#10B981" if sent_label == "Positive" else ("#EF4444" if sent_label == "Negative" else "#94A3B8")
                    
                    st.markdown(f"""
                        <div class="metric-card" style="border-top: 3px solid {color};">
                            <div class="metric-title">Predicted Sentiment</div>
                            <div class="metric-value" style="color: {color};">{sent_label}</div>
                            <div class="metric-subtitle">
                                Model: <b>{sent_res['model_name']}</b> | Confidence: <b>{sent_conf:.1f}%</b>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(sent_res["confidence"], text=f"Sentiment Confidence: {sent_conf:.1f}%")

                # Explanatory features row
                st.markdown("#### 🔍 Explainability & Influential Words")
                exp_c1, exp_c2 = st.columns(2)

                with exp_c1:
                    st.markdown("**Top Category Indicative Words:**")
                    top_t_words = topic_res.get("top_features", [])
                    if top_t_words:
                        badges_html = " ".join([
                            f'<span class="badge badge-category" title="Importance: {w["score"]}">{w["feature"]}</span>'
                            for w in top_t_words
                        ])
                        st.markdown(badges_html, unsafe_allow_html=True)
                    else:
                        st.caption("No strong individual TF-IDF terms isolated from input.")

                with exp_c2:
                    st.markdown("**Top Sentiment Indicative Words:**")
                    top_s_words = sent_res.get("top_features", [])
                    if top_s_words:
                        badges_html = " ".join([
                            f'<span class="badge {badge_style}" title="Importance: {w["score"]}">{w["feature"]}</span>'
                            for w in top_s_words
                        ])
                        st.markdown(badges_html, unsafe_allow_html=True)
                    else:
                        st.caption("No strong individual sentiment terms isolated from input.")

                # Explanatory scientific disclaimer
                st.markdown("""
                    <div class="callout-box">
                        <b>📌 Model Estimation Disclaimer:</b> These predictions are mathematical estimates derived from statistical 
                        patterns in training datasets. They reflect text associations rather than verified factual truth or subjective intent.
                    </div>
                """, unsafe_allow_html=True)

                # Database saving
                if auto_save:
                    record = {
                        "headline": user_input.strip(),
                        "article_text": user_input.strip(),
                        "published_at": str(pub_date),
                        "category": topic_label,
                        "sentiment": sent_label,
                        "topic_confidence": topic_res["confidence"],
                        "sentiment_confidence": sent_res["confidence"],
                        "source": news_source.strip() or "Manual Input",
                        "model_version": f"{topic_res['model_name']} + {sent_res['model_name']}",
                        "analysis_timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    saved = save_article(record)
                    if saved:
                        st.toast("✅ Analysis record saved to SQLite database!", icon="💾")
                    else:
                        st.info("ℹ️ Record already exists in local database (duplicate detected).")

# ==========================================
# TAB 2: BATCH CSV ANALYZER
# ==========================================
with tab_batch:
    st.markdown("#### Upload News Dataset (CSV)")
    st.markdown("""
        Upload a CSV file containing news headlines or articles.
        The file should preferably contain columns such as `headline`, `published_at`, and `source`.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_uploader")

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        # Provide sample CSV template
        from config.settings import SAMPLE_DATA_DIR
        sample_path = SAMPLE_DATA_DIR / "sample_news.csv"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.download_button(
                    "📥 Download Sample CSV",
                    data=f.read(),
                    file_name="sample_news_template.csv",
                    mime="text/csv"
                )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(raw_df)} raw records.** Validating schema...")
            clean_df, report = validate_and_standardize_df(raw_df)

            st.write(f"Validation Result: **{report['valid_rows']} valid records** (Dropped {report['dropped_empty_text']} empty, {report['dropped_duplicates']} duplicates).")
            
            if not clean_df.empty:
                st.dataframe(clean_df.head(5), use_container_width=True)

                if st.button("⚡ Run Batch Analysis on All Records", key="btn_run_batch"):
                    progress_bar = st.progress(0, text="Starting batch inference...")
                    
                    results = []
                    total = len(clean_df)
                    
                    for i, row in clean_df.iterrows():
                        text = str(row["headline"])
                        t_pred = topic_engine.predict(text)
                        s_pred = sentiment_engine.predict(text)

                        results.append({
                            "article_id": row.get("article_id"),
                            "published_at": str(row.get("published_at")),
                            "headline": text,
                            "article_text": row.get("article_text"),
                            "category": t_pred["predicted_category"] if t_pred["status"] == "success" else "Unknown",
                            "sentiment": s_pred["predicted_sentiment"] if s_pred["status"] == "success" else "Neutral",
                            "topic_confidence": t_pred.get("confidence", 0.0),
                            "sentiment_confidence": s_pred.get("confidence", 0.0),
                            "source": row.get("source", "Uploaded CSV"),
                            "model_version": "v1.0-batch",
                            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
                        })

                        if (i + 1) % max(1, total // 20) == 0 or (i + 1) == total:
                            progress_bar.progress((i + 1) / total, text=f"Analyzed {i + 1}/{total} records...")

                    res_df = pd.DataFrame(results)
                    st.success(f"Successfully analyzed {len(res_df)} records!")

                    st.dataframe(res_df[["headline", "category", "sentiment", "topic_confidence", "sentiment_confidence"]].head(10), use_container_width=True)

                    # Bulk save button
                    if st.button("💾 Save Batch to Local Database", key="btn_save_batch"):
                        inserted = save_articles_batch(results)
                        st.success(f"Saved {inserted} new records to SQLite database!")

                    # Download Enriched CSV
                    csv_buffer = io.StringIO()
                    res_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="⬇️ Download Enriched Results CSV",
                        data=csv_buffer.getvalue(),
                        file_name="analyzed_news_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
