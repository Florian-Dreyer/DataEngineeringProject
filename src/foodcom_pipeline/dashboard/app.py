"""
app.py — thin orchestrator for the Food.com Analytics dashboard.

Run locally (from project root):
  FOODCOM_STAGING_DIR=./staging streamlit run src/foodcom_pipeline/dashboard/app.py
"""

import streamlit as st

from foodcom_pipeline.dashboard.theme import inject_theme
from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import render as render_recipe_explorer
from foodcom_pipeline.dashboard.tabs.tab_substitutions import render as render_substitutions
from foodcom_pipeline.dashboard.tabs.tab_market_intelligence import render as render_market_intelligence
from foodcom_pipeline.dashboard.tabs.tab_pipeline_status import render as render_pipeline_status


def main() -> None:
    st.set_page_config(
        page_title="Food.com Intelligence",
        page_icon="🍳",
        layout="wide",
    )
    inject_theme()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🍳 Recipe Explorer",
        "🔄 Smart Substitutions",
        "📊 Market Intelligence",
        "⚙️ Pipeline Status",
    ])

    with tab1:
        render_recipe_explorer()
    with tab2:
        render_substitutions()
    with tab3:
        render_market_intelligence()
    with tab4:
        render_pipeline_status()


if __name__ == "__main__":
    main()
