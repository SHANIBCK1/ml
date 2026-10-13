"""
About Project Page for NewsInsight AI.
Provides academic project documentation, system architecture, three-member work division,
mathematical formulations, viva voce exam preparation guide, and dataset licenses.
"""

import streamlit as st
from src.ui_components import render_header, apply_custom_theme

st.set_page_config(page_title="About Project | NewsInsight AI", page_icon="📘", layout="wide")
apply_custom_theme()

render_header(
    title="About Project & Academic Documentation",
    subtitle="Machine Learning Lab Project Specifications, Architecture, and Viva Voce Guide",
    badge_text="Lab Viva Documentation"
)

tab_overview, tab_team, tab_math, tab_viva, tab_license = st.tabs([
    "📖 Project Abstract & Architecture",
    "👥 Three-Member Work Division",
    "📐 Algorithms & Formulations",
    "🎓 Viva Voce Exam Q&A",
    "📄 Datasets & Limitations"
])

# ==========================================
# TAB 1: ABSTRACT & ARCHITECTURE
# ==========================================
with tab_overview:
    st.markdown("""
        ### 🎓 Academic Project Abstract
        In the era of information abundance, news media significantly influences societal perceptions, capital markets, and 
        public policymaking. Traditional news categorization systems primarily focus on standalone topic tagging, ignoring the 
        dynamic emotional undercurrents and longitudinal shifts embedded in coverage over time.

        **NewsInsight AI** introduces a dual-task natural language processing and supervised machine learning architecture 
        that simultaneously predicts news categories (Politics, Sports, Business, Technology, Science, Entertainment) and 
        multi-class sentiment (Positive, Negative, Neutral). Crucially, the system links predictions with publication timestamps 
        to track **daily sentiment distributions**, compute **day-over-day reporting volatility**, assess **topic-wise negative concentrations**, 
        and formulate an empirical **Net Sentiment Score (News Mood Index)**.
    """)

    st.markdown("### 🔄 Existing vs. Proposed System")
    st.markdown("""
        | Attribute | Existing News Classifier Systems | Proposed NewsInsight AI Platform |
        | :--- | :--- | :--- |
        | **Task Scope** | Single-task (topic classification only) | Dual-task (Topic Categorization + 3-Class Sentiment) |
        | **Temporal Intelligence** | Isolated, static prediction without historical tracking | Full daily timeline tracking with Day-over-Day shift metrics |
        | **Topic Sentiment Correlation** | Unavailable | Stacked topic-wise sentiment breakdown with sample size ($N$) |
        | **Net Sentiment Index** | None | Empirical Net Sentiment Score: $\\text{Positive \\%} - \\text{Negative \\%}$ |
        | **Explainability** | Black-box output without word importance | Word-level TF-IDF and coefficient feature extraction |
        | **Model Diversity** | Single heuristic or hardcoded model | Rigorous comparison of MNB, Logistic Regression, and KNN |
        | **Data Governance** | Frequently reliant on fragile external APIs | Local-first architecture with SQLite persistent database |
    """)

    st.markdown("### 🏗️ System Architecture Flowchart")
    st.code("""
    +-----------------------------------------------------------------------------------+
    |                             NewsInsight AI Architecture                           |
    +-----------------------------------------------------------------------------------+
    
        [Raw News / CSV / Headline / RSS Feeds]
                          │
                          ▼
            +───────────────────────────+
            |    Data Preprocessing     |  ──> [Lowercase, Regex Clean, Stopwords]
            +───────────────────────────+
                          │
                          ▼
            +───────────────────────────+
            |  Feature Extraction:      |  ──> [Sublinear TF-IDF (1-gram, 2-gram)]
            |       TfidfVectorizer     |
            +───────────────────────────+
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
    +───────────────────+     +──────────────────────+
    | Topic Classifier  |     | Sentiment Classifier |
    | (MNB, LogReg, KNN)|     | (MNB, Logistic Reg)  |
    +───────────────────+     +──────────────────────+
             │                         │
             └────────────┬────────────┘
                          ▼
            +───────────────────────────+
            |  Inference & Storage      |  ──> [Confidence, Top Features, SQLite DB]
            +───────────────────────────+
                          │
                          ▼
            +───────────────────────────+
            |  Interactive Streamlit    |  ──> [Daily Sentiment Trend, Topic Bars,
            |         Dashboard         |       News Mood Gauge, Model Evaluation]
            +───────────────────────────+
    """, language="text")

# ==========================================
# TAB 2: THREE-MEMBER WORK DIVISION
# ==========================================
with tab_team:
    st.markdown("### 👥 Three-Member Student Work Division Plan")
    st.markdown("""
        To ensure structured execution, clear accountability, and thorough peer understanding for the lab evaluation, 
        responsibilities are segmented across three specialized engineering roles:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="metric-card" style="border-top: 3px solid #3B82F6;">
                <div class="metric-title">Team Member 1</div>
                <div class="metric-value" style="font-size: 1.3rem; color: #60A5FA;">Data & NLP Engineer</div>
                <div class="metric-subtitle" style="margin-top: 10px; color: #CBD5E1;">
                    <b>Core Responsibilities:</b><br>
                    • Dataset acquisition & documentation<br>
                    • Schema validation & deduplication<br>
                    • Text normalization (regex, clean_text)<br>
                    • Stopword curation & NLTK fallback<br>
                    • TF-IDF feature extraction pipeline<br>
                    • Data leakage prevention & split logic<br>
                    • Unit testing: <code>test_preprocessing.py</code>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-card" style="border-top: 3px solid #8B5CF6;">
                <div class="metric-title">Team Member 2</div>
                <div class="metric-value" style="font-size: 1.3rem; color: #A78BFA;">ML Modeling Engineer</div>
                <div class="metric-subtitle" style="margin-top: 10px; color: #CBD5E1;">
                    <b>Core Responsibilities:</b><br>
                    • Multinomial Naive Bayes implementation<br>
                    • Logistic Regression (with balanced weights)<br>
                    • K-Nearest Neighbors (cosine metric)<br>
                    • Model comparison & selection scripts<br>
                    • Confusion matrix generation<br>
                    • Macro-F1, Precision, Recall metrics<br>
                    • Feature explainability extraction
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-card" style="border-top: 3px solid #10B981;">
                <div class="metric-title">Team Member 3</div>
                <div class="metric-value" style="font-size: 1.3rem; color: #34D399;">Full-Stack & UI/UX Engineer</div>
                <div class="metric-subtitle" style="margin-top: 10px; color: #CBD5E1;">
                    <b>Core Responsibilities:</b><br>
                    • Streamlit multi-page interface<br>
                    • SQLite database persistence & queries<br>
                    • Daily sentiment trend aggregation<br>
                    • Topic-wise stacked bar charts (Plotly)<br>
                    • News Mood gauge & DoD metrics<br>
                    • Batch CSV uploader & export engine<br>
                    • Unit testing: <code>test_analytics.py</code>, <code>test_storage.py</code>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 3: MATHEMATICAL FORMULATIONS
# ==========================================
with tab_math:
    st.markdown("### 📐 Mathematical Formulations & Algorithms")

    st.markdown("""
        #### 1. Term Frequency - Inverse Document Frequency (TF-IDF)
        TF-IDF balances local term occurrence with corpus-wide discriminative power:
        $$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\text{IDF}(t, D)$$
        $$\\text{TF}(t, d) = 1 + \\log(\\text{count}(t, d)) \\quad \\text{if count} > 0$$
        $$\\text{IDF}(t, D) = \\log\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$$
    """)

    st.markdown("""
        #### 2. Multinomial Naive Bayes (MNB)
        Applying Bayes' Theorem with conditional feature independence assumptions:
        $$P(C_k \\mid x) \\propto P(C_k) \\prod_{i=1}^n P(x_i \\mid C_k)$$
        With Laplace smoothing ($\\alpha = 0.5$):
        $$P(x_i \\mid C_k) = \\frac{N_{ki} + \\alpha}{N_k + \\alpha \\cdot |V|}$$
    """)

    st.markdown("""
        #### 3. Logistic Regression with Class Weighting
        Multinomial Softmax formulation minimizing cross-entropy loss:
        $$P(Y = k \\mid x) = \\frac{e^{w_k^T x + b_k}}{\\sum_{j=1}^K e^{w_j^T x + b_j}}$$
        Class balancing adjustment weights the loss by inverse frequency:
        $$w_k = \\frac{N}{K \\times N_k}$$
    """)

    st.markdown("""
        #### 4. Net Sentiment Score (News Mood Index)
        Empirically measures the surplus of optimistic reporting over pessimistic coverage:
        $$\\text{Net Sentiment Score} = \\text{Positive Percentage} - \\text{Negative Percentage} = \\left(\\frac{N_{\\text{pos}} - N_{\\text{neg}}}{N_{\\text{total}}}\\right) \\times 100$$
    """)

# ==========================================
# TAB 4: VIVA VOCE EXAM Q&A
# ==========================================
with tab_viva:
    st.markdown("### 🎓 Lab Viva Voce Preparation Guide")
    st.markdown("Essential questions frequently asked by academic lab evaluators:")

    with st.expander("Q1: Why use TF-IDF instead of simple Bag of Words (CountVectorizer)?"):
        st.write("""
            **Answer:** CountVectorizer only counts raw frequency, causing common words that appear across all documents 
            to dominate the feature space. TF-IDF downweights non-informative words that appear across many documents (via IDF) 
            while prioritizing terms that are uniquely discriminative for a specific category or sentiment class.
        """)

    with st.expander("Q2: How is data leakage prevented during vectorization?"):
        st.write("""
            **Answer:** The `TfidfVectorizer` is fit exclusively on the training partition (`X_train`) after a stratified 
            train/test split. The test set (`X_test`) and inference texts are strictly transformed using the learned training 
            vocabulary, ensuring test distributions do not influence training weights.
        """)

    with st.expander("Q3: Why choose Macro-F1 over Accuracy as the model selection metric?"):
        st.write("""
            **Answer:** Accuracy is misleading when classes are imbalanced; a model predicting only the majority class 
            can achieve high accuracy while failing on minority categories. Macro-F1 computes the harmonic mean of Precision 
            and Recall independently for each class and calculates their unweighted average, ensuring every class is evaluated equally.
        """)

    with st.expander("Q4: How does the system handle class imbalance in sentiment data?"):
        st.write("""
            **Answer:** In Logistic Regression, we configure `class_weight='balanced'`. In Naive Bayes, prior probabilities 
            reflect class distributions, and Laplace smoothing stabilizes minority class likelihoods.
        """)

    with st.expander("Q5: What are the differences between probability, confidence, and accuracy?"):
        st.write("""
            **Answer:** 
            - *Accuracy* is the empirical proportion of correct predictions evaluated against true labels on a test set.
            - *Probability / Confidence* is the model's internal score (e.g. Softmax or normalized Bayes likelihood) for an individual input. 
            A high confidence score does not guarantee that the prediction is factually accurate.
        """)

# ==========================================
# TAB 5: DATASETS & ETHICAL CONSIDERATIONS
# ==========================================
with tab_license:
    st.markdown("### 📄 Datasets, Licenses, & Limitations")

    st.markdown("""
        #### Documented Datasets & Licenses:
        1. **AG News Corpus**:
           - Source: Academic news collection created by Antonio Gulli (1M+ articles from 2,000+ news sources).
           - License: Non-commercial academic research and educational use.
        2. **Standardized Benchmark Corpus**:
           - Source: Standardized multi-category and 3-class sentiment benchmark maintained locally in `data/processed/`.
           - Structure: `article_id`, `published_at`, `headline`, `article_text`, `category`, `sentiment`, `source`.

        #### Critical System Limitations:
        - **Sarcasm & Irony:** Linear and bag-of-words models can struggle with complex linguistic subtleties, figurative language, and rhetorical irony.
        - **Domain Bias:** Financial news terminology (e.g. "bear market", "rate cut", "slump") carries distinct sentiment connotations compared to general news.
        - **Temporal Drift:** Terminology evolves over time. Models should be periodically retrained on contemporary data to maintain classification fidelity.
    """)
