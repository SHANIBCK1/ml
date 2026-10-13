# NewsInsight AI: AI-Based News Classification & Sentiment Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-13%20passed-brightgreen.svg)](https://pytest.org/)

An academic Machine Learning Lab project developed by a three-member student engineering team. NewsInsight AI is a local-first platform designed to categorize news articles into distinct topics, classify multi-class sentiment, track longitudinal sentiment shifts across dates, compute category-wise sentiment concentrations, and calculate an empirical **Net Sentiment Score (News Mood Index)**.

---

## 1. Project Abstract

In modern information ecosystems, digital news reporting directly impacts capital markets, electoral sentiment, and public discourse. Traditional news classification systems operate in silos, performing one-off topic categorization while neglecting the temporal emotional shifts and sentiment dynamics within news coverage over time.

**NewsInsight AI** addresses this limitation by formulating a dual-task natural language processing and supervised machine learning system. The platform classifies news into six standardized categories (**Politics, Sports, Business, Technology, Science, Entertainment**) and three sentiment polarity classes (**Positive, Negative, Neutral**). By indexing predictions to publication timestamps, NewsInsight AI produces **daily sentiment distribution trends**, quantifies **day-over-day reporting volatility**, reveals **category-level negativity concentrations**, and computes an empirical **Net Sentiment Score**. The application provides full model explainability by extracting influential TF-IDF terms for each prediction and operates local-first without paid APIs or external dependencies.

---

## 2. Problem Statement & Objectives

### Problem Statement
Standard news classification models focus exclusively on topic labeling or static sentiment polarity without considering how sentiment evolves over time across different sectors. Furthermore, black-box deep learning models often lack interpretability and require heavy computational infrastructure, making them impractical for transparent, reproducible academic evaluation.

### Core Objectives
1. **Multi-Class Topic Classification:** Categorize news articles into Politics, Sports, Business, Technology, Science, and Entertainment using Multinomial Naive Bayes, Logistic Regression, and K-Nearest Neighbors.
2. **Supervised 3-Class Sentiment Intelligence:** Classify news headlines as Positive, Negative, or Neutral with explicit class imbalance handling.
3. **Daily Sentiment Analytics:** Track daily counts, percentages, and day-over-day changes in negative news reporting over chronological timelines.
4. **Topic-Wise Sentiment Correlation:** Compare sentiment distribution across sectors (e.g., proportion of negative news in Business vs. Politics vs. Science) alongside statistical sample sizes ($N$).
5. **News Mood & Trend Index:** Formulate and visualize the empirical Net Sentiment Score:
   $$\text{Net Sentiment Score} = \text{Positive Percentage} - \text{Negative Percentage}$$
6. **Explainability & Model Transparency:** Expose top influential vocabulary tokens for each prediction and distinguish empirical confidence scores from factual guarantees.
7. **Local-First Architecture:** Store persistent historical records in SQLite and support batch CSV import/export without external API dependencies.

---

## 3. Existing System vs. Proposed NewsInsight AI

| Feature / Dimension | Existing Academic News Systems | Proposed NewsInsight AI System |
| :--- | :--- | :--- |
| **Prediction Scope** | Single-task topic labeling only | Dual-task: Topic categorization + 3-class sentiment |
| **Temporal Tracking** | Static inference without chronological indexing | Full daily sentiment trajectory with Day-over-Day shifts |
| **Topic Sentiment Link** | Missing or disconnected | Stacked category sentiment distributions with sample sizes |
| **News Mood Metric** | None | Empirical Net Sentiment Score ($-100\%$ to $+100\%$) |
| **Model Interpretability** | Black-box outputs | Word-level TF-IDF importance and coefficient extraction |
| **Model Comparison** | Single algorithm evaluated | Head-to-head evaluation: Naive Bayes, Logistic Regression, KNN |
| **Data Persistence** | In-memory or lost on session end | Local SQLite database with deduplication and CSV export |

---

## 4. System Architecture

```
+─────────────────────────────────────────────────────────────────────────────+
|                         NewsInsight AI Architecture                         |
+─────────────────────────────────────────────────────────────────────────────+

   Input Sources: Pasted Headline | Uploaded CSV | Seed Dataset | Public Feeds
                                      │
                                      ▼
               +─────────────────────────────────────────────+
               |      Data Cleaning & Standardizer           |
               | (lowercase, URL/HTML scrub, deduplication)  |
               +─────────────────────────────────────────────+
                                      │
                                      ▼
               +─────────────────────────────────────────────+
               |        TF-IDF Feature Vectorizer            |
               | (Sublinear TF, 1-to-2 ngrams, min_df=1)     |
               +─────────────────────────────────────────────+
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
   +───────────────────────────────+     +───────────────────────────────+
   |   Topic Classifier Pipeline   |     | Sentiment Classifier Pipeline |
   |  (MNB, Logistic Reg, KNN)     |     |  (MNB, Balanced Logistic Reg) |
   +───────────────────────────────+     +───────────────────────────────+
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
               +─────────────────────────────────────────────+
               |      Inference, Explainability & Storage    |
               |  - Confidence & Informative Word Badges     |
               |  - SQLite Persistent Database               |
               +─────────────────────────────────────────────+
                                      │
                                      ▼
               +─────────────────────────────────────────────+
               |        Interactive Streamlit Dashboard      |
               |  - Single & Batch News Analyzer             |
               |  - Daily Sentiment & DoD Trends (Plotly)    |
               |  - Topic-Wise Sentiment Breakdown           |
               |  - Net Sentiment Gauge (News Mood)          |
               |  - Empirical Model Evaluation Benchmarks    |
               +─────────────────────────────────────────────+
```

---

## 5. Machine Learning Algorithms & Mathematical Formulation

### 1. TF-IDF Feature Extraction
Term Frequency-Inverse Document Frequency highlights domain-specific vocabulary while discounting omnipresent stop words:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
$$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{for count} > 0$$
$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### 2. Multinomial Naive Bayes (MNB)
Computes posterior category probabilities under conditional independence assumptions with Laplace smoothing ($\alpha = 0.5$):
$$P(C_k \mid x) \propto P(C_k) \prod_{i=1}^n P(x_i \mid C_k)$$
$$P(x_i \mid C_k) = \frac{N_{ki} + \alpha}{N_k + \alpha \cdot |V|}$$

### 3. Logistic Regression
Multinomial Softmax generalized linear model minimizing regularized cross-entropy loss:
$$P(Y = k \mid x) = \frac{e^{w_k^T x + b_k}}{\sum_{j=1}^K e^{w_j^T x + b_j}}$$
Class weighting accounts for sample distribution imbalances:
$$w_k = \frac{N}{K \cdot N_k}$$

### 4. K-Nearest Neighbors (KNN)
Classifies queries based on majority voting among the $k=5$ closest training vectors using Cosine distance:
$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

### 5. Net Sentiment Score (News Mood Index)
$$\text{Net Sentiment Score} = \text{Positive Percentage} - \text{Negative Percentage} = \left(\frac{N_{\text{pos}} - N_{\text{neg}}}{N_{\text{total}}}\right) \times 100$$
- $\ge +30\%$: Strongly Optimistic
- $+10\%$ to $+30\%$: Moderately Optimistic
- $-10\%$ to $+10\%$: Balanced / Neutral
- $-30\%$ to $-10\%$: Moderately Negative / Cautious
- $\le -30\%$: Strongly Pessimistic

---

## 6. Project Directory Structure

```
ML/
├── app.py                      # Main landing page & application entrypoint
├── requirements.txt            # Pinned dependency requirements
├── README.md                   # Comprehensive project documentation
├── pytest.ini                  # Pytest configuration with pythonpath
├── .gitignore                  # Git ignore rules for venv, models, DB
├── newsinsight.db              # SQLite local persistent database
│
├── config/
│   └── settings.py             # Paths, categories, colors, model hyperparameters
│
├── data/
│   ├── raw/                    # Raw external datasets (e.g., AG News)
│   ├── processed/              # Cleaned benchmark topic & sentiment datasets
│   └── sample/                 # sample_news.csv for smoke testing & seeding
│
├── models/
│   ├── topic_classifier.joblib     # Serialized best topic model
│   ├── topic_vectorizer.joblib     # Serialized TF-IDF vectorizer (topic)
│   ├── sentiment_classifier.joblib # Serialized best sentiment model
│   ├── sentiment_vectorizer.joblib # Serialized TF-IDF vectorizer (sentiment)
│   └── evaluation_results.json    # Verified test set evaluation metrics
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Text cleaning, stopwords, TF-IDF, explainability
│   ├── data_loader.py          # CSV loading, validation, normalization, deduplication
│   ├── topic_classifier.py     # MNB, LogReg, KNN training and inference engine
│   ├── sentiment_classifier.py # Supervised 3-class sentiment classifier engine
│   ├── model_evaluation.py     # Confusion matrices, classification report tables
│   ├── analytics.py            # Daily trends, topic breakdowns, News Mood score
│   ├── storage.py              # SQLite CRUD, batch insertions, query filters
│   ├── ui_components.py        # Custom dark navy CSS styling & metric cards
│   └── newsingestion.py        # Modular RSS feed fetcher with offline resilience
│
├── pages/
│   ├── 1_News_Analyzer.py      # Single & batch news analyzer interface
│   ├── 2_Analytics_Dashboard.py# Daily trends, stacked topic bars, News Mood
│   ├── 3_Model_Evaluation.py   # Algorithmic comparisons & confusion matrices
│   └── 4_About_Project.py      # Viva Voce Q&A, work division, formulations
│
├── scripts/
│   ├── prepare_data.py         # Dataset preparation & AG News integration
│   └── train_models.py         # Multi-model training and evaluation script
│
└── tests/
    ├── test_preprocessing.py   # Unit tests for text cleaning and vectorization
    ├── test_analytics.py       # Unit tests for aggregations and Net Sentiment
    └── test_storage.py         # Unit tests for SQLite storage and deduplication
```

---

## 7. Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone <repository-url>
cd ML

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate    # Linux / macOS
# or .\venv\Scripts\activate # Windows
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Prepare Datasets & Train Models
```bash
# Prepare processed datasets and extract AG News benchmark subset
python scripts/prepare_data.py

# Train Topic models (MNB, LogReg, KNN) and Sentiment models (MNB, LogReg)
python scripts/train_models.py
```

### Step 4: Run Unit Tests
```bash
pytest tests/ -v
```

### Step 5: Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 8. Empirical Evaluation Results

All metrics were computed on a held-out $20\%$ stratified test split. TF-IDF vectorizers were fit strictly on the training partition.

### Topic Classification Benchmark (6 Categories / AG News Subset)
- **Training Set:** 1,216 samples | **Held-Out Test Set:** 304 samples

| Model | Test Accuracy | Macro-F1 | Macro-Precision | Macro-Recall | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **70.72%** | **0.4775** | **0.4770** | **0.4842** | **Yes (Best)** |
| Logistic Regression (Balanced) | 69.74% | 0.4712 | 0.4674 | 0.4775 | No |
| K-Nearest Neighbors (Cosine, k=5) | 61.84% | 0.4143 | 0.4312 | 0.4234 | No |

### Sentiment Classification Benchmark (3 Classes: Positive, Negative, Neutral)
- **Training Set:** 96 samples | **Held-Out Test Set:** 24 samples

| Model | Test Accuracy | Macro-F1 | Macro-Precision | Macro-Recall | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **62.50%** | **0.5358** | **0.5972** | **0.5345** | **Yes (Best)** |
| Logistic Regression (Balanced) | 58.33% | 0.5038 | 0.5333 | 0.5042 | No |

*(Note: In the application UI under "Model Evaluation", users can click "Retrain & Re-evaluate Models" at any time to re-run training on new or expanded datasets.)*

---

## 9. User Interface & Feature Walkthrough

### 1. Home Page (`app.py`)
- Dark navy/charcoal responsive layout with gradient badges.
- Metric cards displaying stored records, active classifiers, and top categories.
- Navigation cards routing to all primary modules.

### 2. News Analyzer (`pages/1_News_Analyzer.py`)
- **Single Headline/Article Analyzer:** Real-time dual prediction (Topic + Sentiment) with progress confidence bars.
- **Explainability Badges:** Displays influential vocabulary tokens extracted via TF-IDF / model coefficients.
- **Batch CSV Upload:** Analyze datasets in bulk with animated progress bars and download enriched results.

### 3. Analytics Dashboard (`pages/2_Analytics_Dashboard.py`)
- **Daily Sentiment Trends:** Plotly multi-line charts showing daily counts and percentages over time.
- **Day-over-Day Changes:** Bar chart tracking percentage point shifts in negative reporting.
- **Topic-Wise Sentiment:** Horizontal stacked bar chart with embedded sample sizes ($N$).
- **News Mood Gauge:** Interactive speedometer visualizing the Net Sentiment Score.
- **Database Management:** Filter, export to CSV, or clear records.

### 4. Model Evaluation (`pages/3_Model_Evaluation.py`)
- Side-by-side metric comparison bar charts for MNB, Logistic Regression, and KNN.
- Interactive Plotly confusion matrix heatmaps with count annotations.
- Full per-class classification reports (Precision, Recall, F1, Support).

### 5. About Project (`pages/4_About_Project.py`)
- Complete academic viva preparation guide with model formulations, architecture flowcharts, and team division.

---

## 10. Three-Member Work Division

| Member | Role | Primary Contributions |
| :--- | :--- | :--- |
| **Member 1** | **Data & NLP Engineer** | • Dataset acquisition and curation (AG News & sentiment corpus)<br>• Schema validation, deduplication, and label normalization (`data_loader.py`)<br>• Text cleaning, regex pipelines, and stopword filtering (`preprocessing.py`)<br>• TF-IDF feature extraction vectorizer configuration<br>• Prevention of train/test data leakage<br>• Unit testing: `test_preprocessing.py` |
| **Member 2** | **ML Modeling Engineer** | • Multinomial Naive Bayes implementation<br>• Logistic Regression with balanced class weighting<br>• K-Nearest Neighbors with Cosine similarity<br>• Supervised 3-class sentiment classifier pipeline (`sentiment_classifier.py`)<br>• Model selection and evaluation metrics (`model_evaluation.py`)<br>• Confusion matrix and classification report generation<br>• Explainability feature importance extraction |
| **Member 3** | **Full-Stack & UI/UX Engineer**| • Streamlit multi-page interface design with custom dark theme CSS<br>• Local SQLite database persistence and schema indexing (`storage.py`)<br>• Daily sentiment aggregation and Day-over-Day analytics (`analytics.py`)<br>• Interactive Plotly visualizations (trends, stacked bars, gauge)<br>• Batch CSV processing and data export tools<br>• Integration testing: `test_analytics.py`, `test_storage.py` |

---

## 11. Ethical Considerations & System Limitations

1. **Model Confidence vs. Factual Correctness:** Confidence scores represent statistical posterior probabilities given the training text distribution; they are not verifiable certificates of factual accuracy.
2. **Sarcasm and Figurative Language:** Like most linear and n-gram based NLP pipelines, subtle sarcasm, double entendres, or cultural colloquialisms may be misclassified.
3. **Domain Vocabulary Discrepancies:** Financial news words like "bearish", "rate hike", or "slump" carry distinct negative valences that might not align with general conversational text.
4. **Scope of Daily News Mood:** The Net Sentiment Score strictly characterizes the analyzed dataset and must not be interpreted as representing the universal emotional state of an entire society.

---

## 12. References & Dataset Licenses

- **AG News Corpus:** Antonio Gulli, Academic news corpus (Academic and educational use).
- **Scikit-Learn:** Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
- **Streamlit:** Open-source application framework (Apache 2.0 License).
- **Plotly:** Interactive graphing library for Python (MIT License).

---

## 13. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
