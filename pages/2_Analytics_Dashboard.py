"""
Analytics Dashboard for NewsInsight AI.
Provides daily sentiment trends, topic-wise sentiment distributions,
and News Mood index (Net Sentiment Score) analytics.
"""

from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config.settings import CATEGORIES, SENTIMENT_CLASSES, SENTIMENT_COLORS, CATEGORY_COLORS
from src.ui_components import render_header, render_metric_card, apply_custom_theme
from src.storage import get_all_records, get_filtered_records, clear_all_records, seed_sample_data_if_empty
from src.analytics import (
    compute_sentiment_distribution,
    compute_daily_sentiment_trends,
    compute_topic_sentiment_breakdown,
    compute_period_comparison
)

st.set_page_config(page_title="Analytics Dashboard | NewsInsight AI", page_icon="📊", layout="wide")
apply_custom_theme()

render_header(
    title="Analytics Dashboard",
    subtitle="Temporal sentiment intelligence, topic distribution, and News Mood indices",
    badge_text="Temporal & Categorical Analytics"
)

# Fetch all records from SQLite
df_all = get_all_records()

if df_all.empty:
    st.info("ℹ️ The database currently contains no analyzed records.")
    if st.button("🌱 Load Sample Historical News Dataset"):
        seed_sample_data_if_empty()
        st.success("Sample records loaded!")
        st.rerun()
    st.stop()

# ==========================================
# SIDEBAR FILTERS
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Filters")

    # Date Range Filter
    has_dates = "published_at" in df_all.columns and df_all["published_at"].notna().any()
    start_date = None
    end_date = None

    if has_dates:
        valid_dates = df_all["published_at"].dropna()
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
        date_selection = st.date_input(
            "Publication Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if isinstance(date_selection, tuple) and len(date_selection) == 2:
            start_date, end_date = date_selection

    # Category Filter
    available_categories = sorted([str(c) for c in df_all["category"].dropna().unique()])
    selected_categories = st.multiselect(
        "News Categories:",
        options=available_categories,
        default=available_categories
    )

    # Sentiment Filter
    available_sentiments = [s for s in SENTIMENT_CLASSES if s in df_all["sentiment"].dropna().unique()]
    selected_sentiments = st.multiselect(
        "Sentiment Classes:",
        options=available_sentiments,
        default=available_sentiments
    )

    # Source Filter
    available_sources = sorted([str(s) for s in df_all["source"].dropna().unique()])
    selected_sources = st.multiselect(
        "News Sources:",
        options=available_sources,
        default=available_sources
    )

    st.markdown("---")
    st.caption("Filters apply dynamically to all visual analytics.")

# Apply Filters
filtered_df = df_all.copy()
if start_date:
    filtered_df = filtered_df[filtered_df["published_at"].isna() | (filtered_df["published_at"].dt.date >= start_date)]
if end_date:
    filtered_df = filtered_df[filtered_df["published_at"].isna() | (filtered_df["published_at"].dt.date <= end_date)]
if selected_categories:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
if selected_sentiments:
    filtered_df = filtered_df[filtered_df["sentiment"].isin(selected_sentiments)]
if selected_sources:
    filtered_df = filtered_df[filtered_df["source"].isin(selected_sources)]

if filtered_df.empty:
    st.warning("⚠️ No records match the active filter criteria. Try expanding your date range or category filters.")
    st.stop()

# ==========================================
# SUMMARY METRICS ROW
# ==========================================
dist = compute_sentiment_distribution(filtered_df)
breakdown_df, top_neg_summary = compute_topic_sentiment_breakdown(filtered_df)

dominant_topic = filtered_df["category"].value_counts().index[0] if not filtered_df["category"].dropna().empty else "N/A"
top_neg_topic = f"{top_neg_summary['category']} ({top_neg_summary['negative_pct']}%)" if top_neg_summary else "N/A"

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_metric_card(
        title="Total Analyzed",
        value=f"{dist['total']:,}",
        subtitle=f"Of {len(df_all):,} database records",
        border_color="#3B82F6"
    )
with col2:
    render_metric_card(
        title="Positive News",
        value=f"{dist['positive_pct']}%",
        subtitle=f"{dist['positive_count']} records",
        border_color="#10B981"
    )
with col3:
    render_metric_card(
        title="Negative News",
        value=f"{dist['negative_pct']}%",
        subtitle=f"{dist['negative_count']} records",
        border_color="#EF4444"
    )
with col4:
    render_metric_card(
        title="Dominant Category",
        value=dominant_topic,
        subtitle=f"Most frequent topic",
        border_color="#8B5CF6"
    )
with col5:
    render_metric_card(
        title="Net Sentiment Score",
        value=f"{dist['net_sentiment_score']:+.1f}",
        subtitle=dist["mood_label"].split("/")[0].strip(),
        border_color="#F59E0B"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Tabs for Organized Analytics
tab_daily, tab_topic, tab_mood, tab_records = st.tabs([
    "📈 Daily Sentiment Trends",
    "📊 Topic-Wise Sentiment",
    "🧠 News Mood Index",
    "📋 Data Records & Management"
])

# ==========================================
# TAB 1: DAILY SENTIMENT TRENDS
# ==========================================
with tab_daily:
    st.markdown("### 📈 Daily News Sentiment Analytics")
    st.markdown("""
        <div class="callout-box">
            <b>Temporal Context:</b> Daily trend analytics track reporting shifts across publication dates in the selected dataset. 
            These describe the analyzed sample and are not an automatic proxy for all global media reporting.
        </div>
    """, unsafe_allow_html=True)

    daily_df = compute_daily_sentiment_trends(filtered_df)

    if daily_df.empty:
        st.info("ℹ️ Publication date information is not available in the filtered dataset to plot daily trends.")
    else:
        # Chart 1: Daily Counts (Positive, Negative, Neutral)
        fig_counts = go.Figure()
        fig_counts.add_trace(go.Scatter(
            x=daily_df["date"], y=daily_df["positive_count"],
            name="Positive News", mode="lines+markers",
            line=dict(color="#10B981", width=3),
            marker=dict(size=7)
        ))
        fig_counts.add_trace(go.Scatter(
            x=daily_df["date"], y=daily_df["negative_count"],
            name="Negative News", mode="lines+markers",
            line=dict(color="#EF4444", width=3),
            marker=dict(size=7)
        ))
        fig_counts.add_trace(go.Scatter(
            x=daily_df["date"], y=daily_df["neutral_count"],
            name="Neutral News", mode="lines+markers",
            line=dict(color="#64748B", width=2, dash="dot"),
            marker=dict(size=6)
        ))

        fig_counts.update_layout(
            title=dict(text="Daily News Volume by Sentiment", font=dict(color="#F8FAFC", size=16)),
            plot_bgcolor="#1E293B", paper_bgcolor="#1E293B",
            font=dict(color="#94A3B8"),
            xaxis=dict(gridcolor="#334155", title="Date"),
            yaxis=dict(gridcolor="#334155", title="Article Count"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_counts, use_container_width=True)

        # Chart 2: Daily Sentiment Percentages over time
        fig_pct = go.Figure()
        fig_pct.add_trace(go.Scatter(
            x=daily_df["date"], y=daily_df["positive_pct"],
            name="Positive %", mode="lines+markers",
            line=dict(color="#10B981", width=2),
            stackgroup="one"
        ))
        fig_pct.add_trace(go.Scatter(
            x=daily_df["date"], y=daily_df["neutral_pct"],
            name="Neutral %", mode="lines+markers",
            line=dict(color="#64748B", width=2),
            stackgroup="one"
        ))
        fig_pct.add_trace(go.Scatter(
            x=daily_df["date"], y=daily_df["negative_pct"],
            name="Negative %", mode="lines+markers",
            line=dict(color="#EF4444", width=2),
            stackgroup="one"
        ))

        fig_pct.update_layout(
            title=dict(text="Daily Sentiment Distribution Proportions (%)", font=dict(color="#F8FAFC", size=16)),
            plot_bgcolor="#1E293B", paper_bgcolor="#1E293B",
            font=dict(color="#94A3B8"),
            xaxis=dict(gridcolor="#334155", title="Date"),
            yaxis=dict(gridcolor="#334155", title="Percentage (%)", range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_pct, use_container_width=True)

        # Day-over-Day Negative Change Bar Chart
        if "dod_neg_pct_change" in daily_df.columns:
            st.markdown("#### ⚡ Day-over-Day Shift in Negative News (%)")
            colors = ["#EF4444" if val > 0 else "#10B981" for val in daily_df["dod_neg_pct_change"]]
            fig_dod = go.Figure(go.Bar(
                x=daily_df["date"],
                y=daily_df["dod_neg_pct_change"],
                marker_color=colors,
                text=[f"{v:+.1f}%" for v in daily_df["dod_neg_pct_change"]],
                textposition="auto"
            ))
            fig_dod.update_layout(
                title=dict(text="Day-over-Day Shift in Negative News Proportion (Percentage Points)", font=dict(color="#F8FAFC", size=15)),
                plot_bgcolor="#1E293B", paper_bgcolor="#1E293B",
                font=dict(color="#94A3B8"),
                xaxis=dict(gridcolor="#334155", title="Date"),
                yaxis=dict(gridcolor="#334155", title="DoD Change (% points)"),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_dod, use_container_width=True)

# ==========================================
# TAB 2: TOPIC-WISE SENTIMENT
# ==========================================
with tab_topic:
    st.markdown("### 📊 Topic-Wise Sentiment Breakdown")

    if breakdown_df.empty:
        st.warning("No category breakdown data available for the active filters.")
    else:
        # Category with highest negative sentiment alert
        if top_neg_summary:
            st.markdown(f"""
                <div class="callout-box" style="border-left-color: #EF4444;">
                    <b>⚠️ Highest Negative Sentiment Category:</b> 
                    <b>{top_neg_summary['category']}</b> exhibits the highest proportion of negative reporting at 
                    <b>{top_neg_summary['negative_pct']}%</b> (Sample size: <i>N={top_neg_summary['sample_size']}</i>).
                </div>
            """, unsafe_allow_html=True)

        # Create category labels with sample size embedded e.g. "Science (N=15)"
        breakdown_df["category_label"] = [
            f"{row['category']} (N={row['sample_size']})"
            for _, row in breakdown_df.iterrows()
        ]

        # Stacked Bar Chart of Percentages
        fig_topic = go.Figure()
        fig_topic.add_trace(go.Bar(
            name="Positive %",
            y=breakdown_df["category_label"],
            x=breakdown_df["positive_pct"],
            orientation="h",
            marker_color="#10B981",
            text=[f"{v}%" for v in breakdown_df["positive_pct"]],
            textposition="auto"
        ))
        fig_topic.add_trace(go.Bar(
            name="Neutral %",
            y=breakdown_df["category_label"],
            x=breakdown_df["neutral_pct"],
            orientation="h",
            marker_color="#64748B",
            text=[f"{v}%" for v in breakdown_df["neutral_pct"]],
            textposition="auto"
        ))
        fig_topic.add_trace(go.Bar(
            name="Negative %",
            y=breakdown_df["category_label"],
            x=breakdown_df["negative_pct"],
            orientation="h",
            marker_color="#EF4444",
            text=[f"{v}%" for v in breakdown_df["negative_pct"]],
            textposition="auto"
        ))

        fig_topic.update_layout(
            barmode="stack",
            title=dict(text="Sentiment Breakdown by Topic (with Sample Sizes)", font=dict(color="#F8FAFC", size=16)),
            plot_bgcolor="#1E293B", paper_bgcolor="#1E293B",
            font=dict(color="#94A3B8"),
            xaxis=dict(gridcolor="#334155", title="Proportion (%)", range=[0, 100]),
            yaxis=dict(gridcolor="#334155", autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_topic, use_container_width=True)

        # Detailed Breakdown Table
        st.markdown("#### 📋 Detailed Category Statistics")
        disp_cols = ["category", "sample_size", "positive_count", "neutral_count", "negative_count", "positive_pct", "negative_pct"]
        st.dataframe(
            breakdown_df[disp_cols].rename(columns={
                "category": "Topic", "sample_size": "Sample Size (N)",
                "positive_count": "Positive (Count)", "neutral_count": "Neutral (Count)",
                "negative_count": "Negative (Count)", "positive_pct": "Positive (%)",
                "negative_pct": "Negative (%)"
            }),
            use_container_width=True
        )

# ==========================================
# TAB 3: NEWS MOOD INDEX
# ==========================================
with tab_mood:
    st.markdown("### 🧠 News Mood and Net Sentiment Index")
    st.markdown("""
        The **Net Sentiment Score** measures the balance of positive versus negative sentiment:
        $$\\text{Net Sentiment Score} = \\text{Positive \\%} - \\text{Negative \\%}$$
        Values range from **-100% (entirely negative)** to **+100% (entirely positive)**.
    """)

    mood_col1, mood_col2 = st.columns([1.2, 1.8])
    with mood_col1:
        net_score = dist["net_sentiment_score"]
        # Gauge Meter for Net Sentiment Score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=net_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Net Sentiment Score", 'font': {'color': "#F8FAFC", 'size': 18}},
            gauge={
                'axis': {'range': [-100, 100], 'tickcolor': "#94A3B8"},
                'bar': {'color': "#3B82F6"},
                'steps': [
                    {'range': [-100, -20], 'color': "rgba(239, 68, 68, 0.3)"},
                    {'range': [-20, 20], 'color': "rgba(100, 116, 139, 0.3)"},
                    {'range': [20, 100], 'color': "rgba(16, 185, 129, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "#F8FAFC", 'width': 4},
                    'thickness': 0.75,
                    'value': net_score
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#1E293B", plot_bgcolor="#1E293B",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=50, b=20),
            height=280
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
            <div style="background-color: #0F172A; padding: 14px; border-radius: 8px; border: 1px solid #334155; text-align: center;">
                <span style="color: #94A3B8; font-size: 0.85rem;">INTERPRETATION</span><br>
                <b style="color: #38BDF8; font-size: 1.15rem;">{dist['mood_label']}</b>
            </div>
        """, unsafe_allow_html=True)

    with mood_col2:
        # Period comparison
        period_res = compute_period_comparison(filtered_df)
        st.markdown("#### ⏳ Historical Period Shift")

        if period_res["status"] == "success":
            neg_shift = period_res["neg_pct_change"]
            direction_color = "#EF4444" if neg_shift > 0 else "#10B981"
            direction_arrow = "▲" if neg_shift > 0 else "▼"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Shift in Negative Sentiment vs Previous Period</div>
                    <div class="metric-value" style="color: {direction_color};">
                        {direction_arrow} {abs(neg_shift):.1f}% points ({period_res['direction']})
                    </div>
                    <div class="metric-subtitle">
                        Current Period Negative: <b>{period_res['current_neg_pct']}%</b> | 
                        Previous Period: <b>{period_res['previous_neg_pct']}%</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Insufficient chronological records to compute period-over-period shift.")

        st.markdown("""
            <div class="callout-box">
                <b>Scope Boundary:</b> This score strictly reflects the sampled text dataset and must not be 
                conflated with the broader mood of an entire country or society.
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 4: DATA RECORDS & MANAGEMENT
# ==========================================
with tab_records:
    st.markdown("### 📋 Filtered Records Explorer & Database Management")
    st.write(f"Showing **{len(filtered_df)}** of **{len(df_all)}** database records.")

    # Table preview
    st.dataframe(
        filtered_df[[
            "published_at", "headline", "category", "sentiment",
            "topic_confidence", "sentiment_confidence", "source"
        ]].head(50),
        use_container_width=True
    )

    col_exp, col_clear, col_seed = st.columns([1, 1, 1])
    
    with col_exp:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Export Filtered Records (CSV)",
            data=csv_data,
            file_name="newsinsight_filtered_records.csv",
            mime="text/csv"
        )

    with col_clear:
        if st.button("🗑️ Clear All Database Records", key="btn_clear_db"):
            deleted = clear_all_records()
            st.warning(f"Cleared {deleted} records from database.")
            st.rerun()

    with col_seed:
        if st.button("🌱 Reset / Re-seed Sample Records", key="btn_reseed_db"):
            clear_all_records()
            count = seed_sample_data_if_empty()
            st.success(f"Database reset and seeded with {count} sample records.")
            st.rerun()
