"""Consistent page header component for FraudPulse."""

from __future__ import annotations

import streamlit as st


def render_page_header(
    title: str,
    subtitle: str,
    badge_text: str = "Historical Experiment Telemetry",
    badge_variant: str = "cyan",
) -> None:
    """Render a consistent top header with title, subtitle, and provenance badge."""
    st.markdown(
        f"""
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-start;
                    gap:0.75rem;padding-bottom:1rem;margin-bottom:1.1rem;
                    border-bottom:1px solid rgba(59, 130, 246, 0.20);">
            <div>
                <h1 style="margin:0;font-size:1.75rem;font-weight:700;letter-spacing:-0.02em;color:#F8FAFC;">
                    {title}
                </h1>
                <p style="margin:0.3rem 0 0 0;font-size:0.92rem;color:#94A3B8;">
                    {subtitle}
                </p>
            </div>
            <div style="padding-top:0.25rem;">
                <span class="fp-badge fp-badge-{badge_variant}">{badge_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
