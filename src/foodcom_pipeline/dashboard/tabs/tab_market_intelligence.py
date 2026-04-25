"""Audience & Market Intelligence tab — CPG segment advertising + search demand intelligence."""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────
# Staging directory resolution
# ─────────────────────────────────────────────────────────────────────────

def _resolve_staging_dir() -> Path:
    project_root = Path(__file__).resolve().parents[4]
    if env_val := os.getenv("FOODCOM_STAGING_DIR"):
        env_path = Path(env_val).expanduser()
        candidates = [env_path]
        if not env_path.is_absolute():
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

CLUSTER_PROFILE_PATH        = STAGING_DIR / "cluster_profiles.json"
AI_MODE_TERM_SCORES_PATH    = STAGING_DIR / "ai_mode_term_scores.parquet"
TRENDS_NORMALISED_PATH      = STAGING_DIR / "google_trends_normalised.parquet"
RECIPE_GAP_ANALYSIS_PATH    = STAGING_DIR / "recipe_gap_analysis.parquet"
EXTERNAL_RECIPE_TERMS_PATH  = STAGING_DIR / "external_recipe_terms.parquet"
RECIPE_TERM_INDEX_PATH      = STAGING_DIR / "recipe_term_index.parquet"
RECIPE_TERM_CLUSTERS_PATH   = STAGING_DIR / "recipe_term_clusters.parquet"
RECIPE_TAGS_PATH            = STAGING_DIR / "recipe_tags.parquet"

# ─────────────────────────────────────────────────────────────────────────
# CPG segment config
# ─────────────────────────────────────────────────────────────────────────

RADAR_FEATURES = ["avg_rating_dairy", "avg_rating_protein", "avg_rating_vegetable",
                  "avg_rating_baking", "avg_rating_international"]
RADAR_LABELS   = ["Dairy", "Protein", "Vegetable", "Baking", "International"]

SEGMENT_COLORS = ["#10b981", "#059669", "#34d399", "#6ee7b7", "#a7f3d0"]

CPG_ADJACENCY: dict[str, dict] = {
    "Indulgent Baker":        {"top_categories": ["baking", "dairy"],
                               "brands": ["King Arthur Baking", "Ghirardelli", "Domino Sugar", "Land O'Lakes"],
                               "cpm_range": "$4 – $8"},
    "International Explorer": {"top_categories": ["international", "protein"],
                               "brands": ["Kikkoman", "Blue Dragon", "Goya Foods", "McCormick"],
                               "cpm_range": "$5 – $9"},
    "Protein-Forward Cook":   {"top_categories": ["protein", "dairy"],
                               "brands": ["Tyson Foods", "Beyond Meat", "Kirkland Signature", "Applegate"],
                               "cpm_range": "$6 – $10"},
    "Health-Conscious Cook":  {"top_categories": ["vegetable", "international"],
                               "brands": ["Amy's Kitchen", "Whole Foods 365", "Green Giant", "Earthbound Farm"],
                               "cpm_range": "$7 – $12"},
    "General Cook":           {"top_categories": ["dairy", "vegetable"],
                               "brands": ["Heinz", "Kraft", "Campbell's", "Birds Eye"],
                               "cpm_range": "$3 – $6"},
}

# ─────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_cluster_profiles() -> dict | None:
    if not CLUSTER_PROFILE_PATH.exists():
        return None
    try:
        with open(CLUSTER_PROFILE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(ttl=300)
def _load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────
# CPG helpers
# ─────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────
# Demand gap helpers
# ─────────────────────────────────────────────────────────────────────────

_OPP_COLORS = {"High": "#ef4444", "Medium": "#f97316", "Low": "#9ca3af"}
_OPP_ICONS  = {"High": "🔴", "Medium": "🟠", "Low": "⚪"}


def _gap_bar_chart(gap_df: pd.DataFrame) -> go.Figure:
    """Bar chart of gap scores coloured by opportunity_label."""
    label_col = "opportunity_label" if "opportunity_label" in gap_df.columns else "opportunity_tier"
    term_col  = "raw_term"          if "raw_term"          in gap_df.columns else "term"

    fig = go.Figure()
    for tier in ("High", "Medium", "Low"):
        subset = gap_df[gap_df[label_col] == tier]
        if subset.empty:
            continue
        fig.add_trace(go.Bar(
            x=subset[term_col],
            y=subset["gap_score"],
            name=tier,
            marker_color=_OPP_COLORS[tier],
        ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Term",
        yaxis_title="Gap Score",
        xaxis_tickangle=-40,
        height=420,
        legend_title="Opportunity",
        margin=dict(l=40, r=20, t=30, b=120),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Recipe term cluster helpers
# ─────────────────────────────────────────────────────────────────────────

def _clusters_bar_chart(clusters_df: pd.DataFrame) -> go.Figure:
    top = clusters_df.dropna(subset=["max_gap_score"]).nlargest(20, "max_gap_score")
    fig = go.Figure(go.Bar(
        x=top["cluster_label"],
        y=top["max_gap_score"],
        marker_color="#6366f1",
        text=top["sources_present"],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Max Gap Score",
        xaxis_tickangle=-40,
        height=400,
        margin=dict(l=40, r=20, t=30, b=120),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.header("📊 Market Intelligence")

    # ── Section A: AI-Powered Search Intelligence ─────────────────────────
    st.subheader("What consumers are searching for right now")
    st.markdown("*Sourced from Google AI Mode — the AI answers users see before any search results*")

    ai_df = _load_parquet(AI_MODE_TERM_SCORES_PATH)
    if ai_df is None:
        st.warning("AI Mode data not yet available — run the pipeline first.")
    else:
        top15 = (
            ai_df.sort_values("normalised_score", ascending=False)
            .head(15)
            [["term", "normalised_score", "fetched_date"]]
            .rename(columns={"normalised_score": "AI Prominence Score"})
            .reset_index(drop=True)
        )
        st.dataframe(top15, use_container_width=True, hide_index=True)
        st.caption(
            "These are the food topics appearing in AI-generated search summaries — "
            "the first thing a user reads when they Google a meal idea."
        )

    # External terms breakdown (AI Mode + Trends combined)
    ext_df = _load_parquet(EXTERNAL_RECIPE_TERMS_PATH)
    if ext_df is not None and not ext_df.empty:
        with st.expander("View all normalised external terms (AI Mode + Trends)"):
            by_src = ext_df.groupby("source").size().reset_index(name="terms")
            st.dataframe(by_src, use_container_width=True, hide_index=True)
            display_cols = [c for c in
                ["source", "raw_term", "canonical_term", "source_score", "tags", "fetched_date"]
                if c in ext_df.columns]
            st.dataframe(
                ext_df[display_cols].sort_values("source_score", ascending=False),
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ── Section B: Demand vs Supply Gap ───────────────────────────────────
    st.subheader("Where consumer demand outpaces available recipes")

    gap_df = _load_parquet(RECIPE_GAP_ANALYSIS_PATH)
    if gap_df is None:
        st.warning("Gap analysis data not yet available — run the pipeline first.")
    else:
        opp_df = (
            gap_df[gap_df["match_status"].isin(["gap", "weak_match"])]
            .sort_values("gap_score", ascending=False)
            .reset_index(drop=True)
        )

        st.plotly_chart(_gap_bar_chart(opp_df.head(30)), use_container_width=True)
        st.info(
            "These are the cuisines and ingredients where consumer demand is outpacing "
            "available content — emerging markets you should move into."
        )

        # Top insight cards
        st.markdown("#### Top content opportunities")
        top_gaps = opp_df.head(10)
        for _, row in top_gaps.iterrows():
            insight = str(row.get("insight_summary") or "")
            label   = str(row.get("opportunity_label") or "")
            icon    = _OPP_ICONS.get(label, "•")
            raw     = str(row.get("raw_term", ""))
            if insight:
                st.info(f"{icon} **{raw}** — {insight}")

        # Full table
        with st.expander("View full gap analysis table"):
            display_cols = [c for c in [
                "raw_term", "match_status", "gap_score", "opportunity_label",
                "top_gap_rank", "best_foodcom_recipe_name", "best_foodcom_similarity",
                "matching_method", "source", "insight_summary",
            ] if c in gap_df.columns]
            st.dataframe(gap_df[display_cols], use_container_width=True, hide_index=True)

    st.divider()

    # ── Section C: Google Trends Signal ───────────────────────────────────
    st.subheader("Trending recipe searches on Google")

    trends_df = _load_parquet(TRENDS_NORMALISED_PATH)
    if trends_df is None:
        st.warning("Trends data not yet available — run the pipeline first.")
    else:
        top20 = (
            trends_df.sort_values("normalised_score", ascending=False)
            .head(20)
            [["related_query", "query_type", "normalised_score", "fetched_date"]]
            .reset_index(drop=True)
        )
        st.dataframe(top20, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section D: Recipe Term Clusters ───────────────────────────────────
    st.subheader("Recipe term clusters")
    st.markdown("*Semantic groups of external demand terms — fuzzy-merged by dish similarity*")

    clusters_df = _load_parquet(RECIPE_TERM_CLUSTERS_PATH)
    if clusters_df is None:
        st.warning("Cluster data not yet available — run `build_recipe_term_clusters` first.")
    else:
        st.plotly_chart(_clusters_bar_chart(clusters_df), use_container_width=True)

        display_cols = [c for c in [
            "cluster_label", "representative_term", "sources_present",
            "dominant_tags", "avg_gap_score", "max_gap_score",
            "best_foodcom_match", "foodcom_coverage_count",
        ] if c in clusters_df.columns]
        st.dataframe(
            clusters_df[display_cols].sort_values("max_gap_score", ascending=False, na_position="last"),
            use_container_width=True, hide_index=True,
        )

        with st.expander("Cluster explanations"):
            if "explanation" in clusters_df.columns:
                for _, row in clusters_df.dropna(subset=["max_gap_score"]).nlargest(15, "max_gap_score").iterrows():
                    label = str(row.get("cluster_label", ""))
                    expl  = str(row.get("explanation", ""))
                    if expl:
                        st.markdown(f"**{label}** — {expl}")

    st.divider()

    # ── Section E: Pipeline Health ────────────────────────────────────────
    st.subheader("🏥 Pipeline health")

    _files_to_check = [
        ("recipe_tags",           RECIPE_TAGS_PATH),
        ("recipe_term_index",     RECIPE_TERM_INDEX_PATH),
        ("external_terms",        EXTERNAL_RECIPE_TERMS_PATH),
        ("gap_analysis",          RECIPE_GAP_ANALYSIS_PATH),
        ("term_clusters",         RECIPE_TERM_CLUSTERS_PATH),
    ]

    health_cols = st.columns(len(_files_to_check))
    for col, (label, path) in zip(health_cols, _files_to_check):
        if path.is_file():
            try:
                n = len(pd.read_parquet(path))
                col.metric(label, f"{n:,}", "✓ present")
            except Exception:
                col.metric(label, "read error", "⚠")
        else:
            col.metric(label, "missing", "✗")

    if gap_df is not None and not gap_df.empty and "match_status" in gap_df.columns:
        total  = len(gap_df)
        strong = int((gap_df["match_status"] == "strong_match").sum())
        weak   = int((gap_df["match_status"] == "weak_match").sum())
        gaps   = int((gap_df["match_status"] == "gap").sum())

        st.markdown("**Gap analysis match breakdown**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Strong matches",      f"{strong:,}", f"{100*strong/total:.0f}% of total")
        c2.metric("Weak matches",        f"{weak:,}",   f"{100*weak/total:.0f}% of total")
        c3.metric("Gaps (opportunities)", f"{gaps:,}",  f"{100*gaps/total:.0f}% of total")

        if "insight_summary" in gap_df.columns:
            populated = int(gap_df["insight_summary"].notna().sum())
            if populated == total:
                st.success(f"All {total:,} rows have `insight_summary` populated.")
            else:
                st.warning(f"{populated:,} / {total:,} rows have `insight_summary`.")

        if "matching_method" in gap_df.columns:
            method = gap_df["matching_method"].mode().iloc[0] if not gap_df.empty else "unknown"
            st.caption(f"Matching method: **{method}**")

    st.divider()

    # ── Section F: Audience & CPG Segments ───────────────────────────────
    st.subheader("Audience & CPG Segments")

    profiles_data = _load_cluster_profiles()
    if profiles_data is None:
        st.warning(f"No cluster profiles found at `{CLUSTER_PROFILE_PATH}`. Run the batch pipeline first.")
        return

    profiles = profiles_data.get("clusters", {})
    all_labels = [p["cluster_label"] for p in profiles.values()]

    selected_labels = st.multiselect("Select segments to display", options=all_labels, default=all_labels)
    if not selected_labels:
        st.warning("Select at least one segment.")
        return

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
