"""Smart Substitutions tab — polished Coming Soon stub."""

import streamlit as st


def render() -> None:
    st.header("🔄 Smart Substitutions")

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #f0fdf4, #dcfce7);
            border: 2px dashed #10b981;
            border-radius: 12px;
            padding: 48px 32px;
            text-align: center;
            margin-top: 32px;
        ">
            <div style="font-size: 48px; margin-bottom: 16px;">🔄</div>
            <h2 style="color: #065f46; margin-bottom: 12px;">Smart Substitutions</h2>
            <p style="color: #374151; max-width: 480px; margin: 0 auto; line-height: 1.8; font-size: 15px;">
                Ingredient swap cards with rating and nutritional deltas, real-time
                Bayesian recalculation, and native ad placement hooks.
                Detects underperforming ingredients and recommends trending alternatives.
            </p>
            <div style="
                display: inline-block;
                margin-top: 24px;
                background: #10b981;
                color: white;
                border-radius: 8px;
                padding: 10px 24px;
                font-weight: 700;
                font-size: 14px;
            ">Coming Soon</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
