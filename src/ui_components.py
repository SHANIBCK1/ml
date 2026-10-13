"""
Shared UI components and styling utilities for NewsInsight AI.
Provides unified dark navy styling, metric cards, badges, and layout helpers across Streamlit pages.
"""

import streamlit as st
from config.settings import COLOR_PALETTE


def apply_custom_theme():
    """Applies modern dark navy styling across the Streamlit application."""
    st.markdown("""
        <style>
        /* Main background and font styling */
        .stApp {
            background-color: #0B1120;
            color: #F8FAFC;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid #1E293B;
        }

        /* Header styling */
        h1, h2, h3, h4 {
            color: #F8FAFC !important;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        
        /* Metric card styling */
        .metric-card {
            background: linear-gradient(145deg, #1E293B, #0F172A);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            border-color: #3B82F6;
            transform: translateY(-2px);
        }
        .metric-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #F8FAFC;
            line-height: 1.2;
        }
        .metric-subtitle {
            font-size: 0.8rem;
            color: #64748B;
            margin-top: 4px;
        }

        /* Sentiment Badges */
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .badge-positive {
            background-color: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge-negative {
            background-color: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .badge-neutral {
            background-color: rgba(100, 116, 139, 0.15);
            color: #94A3B8;
            border: 1px solid rgba(100, 116, 139, 0.3);
        }

        /* Category Badges */
        .badge-category {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15));
            color: #93C5FD;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        /* Form and input elements */
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }
        .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 1px #3B82F6 !important;
        }

        /* Custom buttons */
        .stButton > button {
            background: linear-gradient(135deg, #2563EB, #1D4ED8);
            color: #FFFFFF;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 8px 18px;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #3B82F6, #2563EB);
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #0F172A;
            padding: 6px;
            border-radius: 10px;
            border: 1px solid #1E293B;
        }
        .stTabs [data-baseweb="tab"] {
            color: #94A3B8;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E293B !important;
            color: #38BDF8 !important;
            font-weight: 700 !important;
        }

        /* Explanatory notes */
        .callout-box {
            background-color: rgba(30, 41, 59, 0.6);
            border-left: 4px solid #3B82F6;
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            margin: 12px 0;
            font-size: 0.9rem;
            color: #CBD5E1;
        }
        .callout-warning {
            border-left-color: #F59E0B;
        }
        .callout-success {
            border-left-color: #10B981;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badge_text: str = "Machine Learning Lab Project"):
    """Renders a standard branded header banner."""
    apply_custom_theme()
    st.markdown(f"""
        <div style="padding-bottom: 20px; border-bottom: 1px solid #1E293B; margin-bottom: 24px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h1 style="margin: 0; font-size: 2.2rem; background: linear-gradient(135deg, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        {title}
                    </h1>
                    <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 1.05rem;">
                        {subtitle}
                    </p>
                </div>
                <div>
                    <span class="badge" style="background: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3);">
                        ⚡ {badge_text}
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = "", border_color: str = "#334155"):
    """Renders an individual metric card component."""
    st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid {border_color};">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)
