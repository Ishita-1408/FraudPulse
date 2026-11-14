"""Styled KPI card grid component."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class KpiItem:
    label: str
    value: str
    subtext: str = ""
    accent: str = "#3B82F6"


def render_kpi_row(items: list[KpiItem]) -> None:
    """Render a responsive row of dark-fintech KPI cards."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            sub_html = f"<div class='fp-kpi-sub'>{item.subtext}</div>" if item.subtext else ""
            st.markdown(
                f"""
                <div class="fp-kpi-card" style="--accent: {item.accent};">
                    <div class="fp-kpi-label">{item.label}</div>
                    <div class="fp-kpi-value">{item.value}</div>
                    {sub_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
