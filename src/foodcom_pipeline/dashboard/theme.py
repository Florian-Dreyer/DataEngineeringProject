"""Global CSS theme — inject_theme() writes emerald palette styles."""

_CSS = """
<style>
/* Page background */
.stApp { background-color: #f8fafc; }

/* Metric cards — emerald top border */
[data-testid="stMetric"] {
    background: white;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    border-top: 3px solid #10b981;
    padding: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Tab bar */
.stTabs [data-baseweb="tab"] {
    background-color: white;
    border-radius: 6px 6px 0 0;
    border: 1px solid #e5e7eb;
    border-bottom: none;
    padding: 8px 18px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    border-top: 2px solid #10b981 !important;
    color: #10b981 !important;
}

/* Primary buttons */
.stButton > button[kind="primary"],
.stButton > button {
    background-color: #10b981;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #059669;
    border: none;
}

/* DataFrames */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Dividers */
hr { border-color: #e5e7eb; }
</style>
"""


def inject_theme() -> None:
    """Inject global Clean & Modern CSS into the Streamlit page."""
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)
