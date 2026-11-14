"""Dark fintech styling and Plotly theme configuration for FraudPulse."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

COLORS = {
    "bg_navy": "#060B18",
    "card_slate": "#0F172A",
    "card_elevated": "#162036",
    "border": "rgba(59, 130, 246, 0.20)",
    "border_subtle": "rgba(148, 163, 184, 0.14)",
    "blue": "#3B82F6",
    "cyan": "#06B6D4",
    "green": "#10B981",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
}


def inject_custom_css() -> None:
    """Inject commercial dark fintech CSS into the Streamlit app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #F8FAFC;
        }

        .stApp {
            background: radial-gradient(circle at 12% 8%, rgba(59, 130, 246, 0.10), transparent 36%),
                        radial-gradient(circle at 88% 12%, rgba(6, 182, 212, 0.08), transparent 32%),
                        #060B18;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #091022 0%, #060B18 100%);
            border-right: 1px solid rgba(59, 130, 246, 0.18);
        }

        [data-testid="stHeader"] {
            background: rgba(6, 11, 24, 0.72);
            backdrop-filter: blur(10px);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
        }

        .fp-card {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.96), rgba(11, 18, 34, 0.96));
            border: 1px solid rgba(59, 130, 246, 0.20);
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.45);
            margin-bottom: 0.9rem;
        }

        .fp-kpi-card {
            background: linear-gradient(155deg, #0F172A 0%, #0B1326 100%);
            border: 1px solid rgba(59, 130, 246, 0.22);
            border-top: 3px solid var(--accent, #3B82F6);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 18px rgba(2, 6, 23, 0.35);
            height: 100%;
        }

        .fp-kpi-label {
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #94A3B8;
            margin-bottom: 0.35rem;
        }

        .fp-kpi-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.55rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.2;
        }

        .fp-kpi-sub {
            font-size: 0.78rem;
            color: #94A3B8;
            margin-top: 0.35rem;
        }

        .fp-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.24rem 0.65rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            border: 1px solid transparent;
        }

        .fp-badge-blue { background: rgba(59, 130, 246, 0.14); color: #60A5FA; border-color: rgba(59, 130, 246, 0.35); }
        .fp-badge-cyan { background: rgba(6, 182, 212, 0.14); color: #22D3EE; border-color: rgba(6, 182, 212, 0.35); }
        .fp-badge-green { background: rgba(16, 185, 129, 0.14); color: #34D399; border-color: rgba(16, 185, 129, 0.35); }
        .fp-badge-amber { background: rgba(245, 158, 11, 0.14); color: #FBBF24; border-color: rgba(245, 158, 11, 0.35); }
        .fp-badge-red { background: rgba(239, 68, 68, 0.14); color: #F87171; border-color: rgba(239, 68, 68, 0.35); }

        .fp-section-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #E2E8F0;
            margin: 0.4rem 0 0.75rem 0;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: rgba(15, 23, 42, 0.7);
            padding: 0.35rem;
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.14);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #94A3B8;
            font-weight: 500;
            padding: 0.45rem 0.9rem;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(59, 130, 246, 0.20) !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(59, 130, 246, 0.45);
        }

        div[data-testid="stExpander"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 10px;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563EB, #06B6D4);
            border: none;
            color: #FFFFFF;
            font-weight: 600;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply consistent dark fintech styling to any Plotly figure."""
    layout_kwargs: dict[str, Any] = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(15, 23, 42, 0.0)",
        "plot_bgcolor": "rgba(15, 23, 42, 0.55)",
        "font": {"family": "Inter, sans-serif", "color": "#E2E8F0", "size": 12},
        "height": height,
        "margin": {"l": 48, "r": 24, "t": 48 if title else 24, "b": 44},
        "legend": {
            "bgcolor": "rgba(6, 11, 24, 0.6)",
            "bordercolor": "rgba(148, 163, 184, 0.2)",
            "borderwidth": 1,
        },
        "xaxis": {
            "gridcolor": "rgba(148, 163, 184, 0.10)",
            "zerolinecolor": "rgba(148, 163, 184, 0.18)",
        },
        "yaxis": {
            "gridcolor": "rgba(148, 163, 184, 0.10)",
            "zerolinecolor": "rgba(148, 163, 184, 0.18)",
        },
    }
    if title:
        layout_kwargs["title"] = {"text": title, "font": {"size": 15, "color": "#F8FAFC"}}
    fig.update_layout(**layout_kwargs)
    return fig
