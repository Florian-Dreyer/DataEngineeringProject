"""Audience & Market Intelligence tab — CPG segment advertising tool."""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _resolve_staging_dir() -> Path:
    project_root = Path(__file__).resolve().parents[4]
    if env_val := os.getenv("FOODCOM_STAGING_DIR"):
        env_path = Path(env_val).expanduser()
        candidates = [env_path]
        if not env_path.is_absolute():
            # Support launching Streamlit from directories other than repo root.
            candidates.append((project_root / env_path).resolve())
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    candidate = project_root / "staging"
    if candidate.exists():
        return candidate
    return Path("/opt/airflow/staging")


STAGING_DIR = _resolve_staging_dir()
CLUSTER_PROFILE_PATH = STAGING_DIR / "cluster_profiles.json"

RADAR_FEATURES = ["avg_rating_dairy", "avg_rating_protein", "avg_rating_vegetable",
                  "avg_rating_baking", "avg_rating_international"]
RADAR_LABELS   = ["Dairy", "Protein", "Vegetable", "Baking", "International"]

# Emerald palette for segment colours
SEGMENT_COLORS = ["#10b981", "#059669", "#34d399", "#6ee7b7", "#a7f3d0"]

CPG_ADJACENCY: dict[str, dict] = {
    "Indulgent Baker":       {"top_categories": ["baking", "dairy"],
                              "brands": ["King Arthur Baking", "Ghirardelli", "Domino Sugar", "Land O'Lakes"],
                              "cpm_range": "$4 – $8"},
    "International Explorer":{"top_categories": ["international", "protein"],
                              "brands": ["Kikkoman", "Blue Dragon", "Goya Foods", "McCormick"],
                              "cpm_range": "$5 – $9"},
    "Protein-Forward Cook":  {"top_categories": ["protein", "dairy"],
                              "brands": ["Tyson Foods", "Beyond Meat", "Kirkland Signature", "Applegate"],
                              "cpm_range": "$6 – $10"},
    "Health-Conscious Cook": {"top_categories": ["vegetable", "international"],
                              "brands": ["Amy's Kitchen", "Whole Foods 365", "Green Giant", "Earthbound Farm"],
                              "cpm_range": "$7 – $12"},
    "General Cook":          {"top_categories": ["dairy", "vegetable"],
                              "brands": ["Heinz", "Kraft", "Campbell's", "Birds Eye"],
                              "cpm_range": "$3 – $6"},
}


@st.cache_data(ttl=300)
def _load_cluster_profiles() -> dict | None:
    if not CLUSTER_PROFILE_PATH.exists():
        return None
    try:
        with open(CLUSTER_PROFILE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _radar_chart(profiles: dict, selected_labels: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, (_, profile) in enumerate(profiles.items()):
        label = profile["cluster_label"]
        if label not in selected_labels:
            continue
        stats = profile.get("feature_stats", {})
        values = [stats.get(f, {}).get("mean", 0.0) for f in RADAR_FEATURES]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=RADAR_LABELS + [RADAR_LABELS[0]],
            fill="toself",
            name=label,
            line_color=SEGMENT_COLORS[i % len(SEGMENT_COLORS)],
            opacity=0.65,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[3.5, 5.0])),
        showlegend=True,
        title="Taste Profile Radar — Avg Rating per Ingredient Category",
        height=480,
    )
    return fig


def _segment_profile_table(profiles: dict, selected_labels: list[str]) -> pd.DataFrame:
    rows = []
    for profile in profiles.values():
        if profile["cluster_label"] not in selected_labels:
            continue
        rows.append({
            "Segment": profile["cluster_label"],
            "Users": f"{profile['n_users']:,}",
            "% of Total": f"{profile['pct_users']:.1f}%",
            "Peak Month": profile.get("peak_month") or "—",
            "Top Ingredients": ", ".join(profile.get("top_ingredients") or []) or "—",
            "Avg ! per Review": (f"{profile['avg_exclamation_count']:.2f}"
                                  if profile.get("avg_exclamation_count") is not None else "—"),
            "Sub. Exposure": (f"{profile['substitute_exposure_rate']:.1%}"
                               if profile.get("substitute_exposure_rate") is not None else "—"),
        })
    return pd.DataFrame(rows)


def _brand_adjacency_table(selected_labels: list[str]) -> pd.DataFrame:
    rows = []
    for label, data in CPG_ADJACENCY.items():
        if label not in selected_labels:
            continue
        rows.append({
            "Segment": label,
            "Key Categories": ", ".join(data["top_categories"]),
            "Recommended CPG Brands": ", ".join(data["brands"]),
            "Est. CPM Range": data["cpm_range"],
        })
    return pd.DataFrame(rows)


def _generate_pdf(profiles: dict, selected_labels: list[str]) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError:
        return b""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Food.com — Audience & Market Intelligence", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "CPG Advertising Segment Report", ln=True, align="C")
    pdf.ln(6)
    for profile in profiles.values():
        label = profile["cluster_label"]
        if label not in selected_labels:
            continue
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"{label}  ({profile['n_users']:,} users, {profile['pct_users']:.1f}%)", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  Top ingredients : {', '.join(profile.get('top_ingredients') or []) or '—'}", ln=True)
        pdf.cell(0, 6, f"  Peak month      : {profile.get('peak_month') or '—'}", ln=True)
        sub = profile.get("substitute_exposure_rate")
        pdf.cell(0, 6, f"  Sub. exposure   : {sub:.1%}" if sub is not None else "  Sub. exposure   : —", ln=True)
        adj = CPG_ADJACENCY.get(label, {})
        if adj:
            pdf.cell(0, 6, f"  CPG brands      : {', '.join(adj.get('brands', []))}", ln=True)
            pdf.cell(0, 6, f"  Est. CPM        : {adj.get('cpm_range', '—')}", ln=True)
        pdf.ln(4)
    return bytes(pdf.output())


def render() -> None:
    st.header("📊 Audience & Market Intelligence")

    profiles_data = _load_cluster_profiles()
    if profiles_data is None:
        st.warning(f"No cluster profiles found at `{CLUSTER_PROFILE_PATH}`. Run the batch pipeline first.")
        return

    profiles = profiles_data.get("clusters", {})
    all_labels = [p["cluster_label"] for p in profiles.values()]

    st.subheader("Segment Filter")
    selected_labels = st.multiselect("Select segments to display", options=all_labels, default=all_labels)
    if not selected_labels:
        st.warning("Select at least one segment.")
        return

    # KPI row — metric cards styled by theme.py
    total_users = profiles_data.get("total_users", 0)
    n_clusters  = profiles_data.get("n_clusters", len(profiles))
    c1, c2 = st.columns(2)
    c1.metric("Total Segmented Users", f"{total_users:,}")
    c2.metric("Number of Segments", n_clusters)

    st.divider()

    st.subheader("Taste Profile Radar")
    st.plotly_chart(_radar_chart(profiles, selected_labels), use_container_width=True)

    st.divider()

    st.subheader("Segment Profiles")
    st.dataframe(_segment_profile_table(profiles, selected_labels), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("CPG Brand Adjacency")
    st.caption("Static mapping of segments to recommended CPG brands and estimated programmatic CPM ranges.")
    st.dataframe(_brand_adjacency_table(selected_labels), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Export")
    if st.button("Prepare PDF Report"):
        pdf_bytes = _generate_pdf(profiles, selected_labels)
        if pdf_bytes:
            st.download_button("Download PDF Report", data=pdf_bytes,
                                file_name="foodcom_audience_report.pdf", mime="application/pdf")
        else:
            st.info("PDF export requires `fpdf2`. Install with `uv add fpdf2`.")
