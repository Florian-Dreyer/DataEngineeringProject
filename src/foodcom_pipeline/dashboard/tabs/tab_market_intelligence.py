"""Audience & Market Intelligence tab — CPG segment advertising + search demand intelligence.

REDESIGNED: April 2026
- Query control panel (sidebar) for user-defined queries
- Global filter bar for match status, gap score, tags, sources
- Sortable/filterable tables with column_config
- Explainability layer with expandable row details
- Preset visibility & traceability
- Improved charts with hover tooltips
- Enhanced pipeline transparency
- Tab-based navigation
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import dashboard components
from foodcom_pipeline.dashboard.components import (
    TOOLTIPS,
    tooltip_icon,
    render_tooltip,
    metric_with_tooltip,
    badge,
    progress_bar,
    insight_card,
    empty_state,
    detail_panel,
    health_card,
    enhance_gap_chart,
    enhance_cluster_chart,
)


# ─────────────────────────────────────────────────────────────────────────
# PRESET QUERIES CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

# Default preset queries used by the pipeline
DEFAULT_PRESET_QUERIES = [
    "dinner ideas",
    "easy weeknight meals",
    "healthy lunch",
    "quick breakfast",
    "vegetarian recipes",
    "chicken dinner",
    "pasta recipes",
    "salad ideas",
    "dessert recipes",
    "soup recipes",
    "healthy dinner",
    "low carb meals",
    "gluten free",
    "vegan recipes",
    "keto diet",
    "meal prep",
    "comfort food",
    "summer recipes",
    "winter meals",
    "holiday cooking",
]

# All available presets
ALL_PRESETS = sorted(set(DEFAULT_PRESET_QUERIES + [
    "air fryer recipes",
    "instant pot meals",
    "one pot pasta",
    "stir fry",
    "casserole",
    "slow cooker",
    "grilling recipes",
    "bbq ideas",
    "mexican food",
    "italian recipes",
    "asian cuisine",
    "indian food",
    "mediterranean diet",
    "american classics",
    "french cooking",
    "japanese recipes",
    "thai food",
    "chinese new year",
    "thanksgiving dinner",
    "christmas cookies",
    "easter brunch",
    "4th of july bbq",
    "halloween treats",
    "valentine's day dinner",
    "birthday cake",
    "appetizers",
    "snacks",
    "sides",
    "drinks",
    "smoothies",
    "cocktails",
    "coffee drinks",
    "tea recipes",
    "baking bread",
    "pizza dough",
    "homemade pasta",
    "sourdough recipes",
    "cake decorating",
    "pie recipes",
    "cookie exchange",
    "chocolate desserts",
    "ice cream",
    "pie",
    "tart",
    "muffins",
    "cupcakes",
    "brownies",
    "pancakes",
    "waffles",
    "french toast",
    "omelette",
    "scrambled eggs",
    "breakfast casserole",
    "overnight oats",
    "yogurt parfait",
    "acai bowl",
    "avocado toast",
    "breakfast burrito",
    "morning glory",
    "bran muffin",
    "granola",
]))


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
# QUERY CONTROL PANEL (SIDEBAR)
# ─────────────────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    """Initialize session state for query controls and filters."""
    if "selected_presets" not in st.session_state:
        st.session_state.selected_presets = DEFAULT_PRESET_QUERIES[:5]
    if "custom_queries" not in st.session_state:
        st.session_state.custom_queries = ""
    if "include_ai_mode" not in st.session_state:
        st.session_state.include_ai_mode = True
    if "include_trends" not in st.session_state:
        st.session_state.include_trends = True
    if "last_run_timestamp" not in st.session_state:
        st.session_state.last_run_timestamp = None
    if "active_queries" not in st.session_state:
        st.session_state.active_queries = []


def _render_query_control_tab() -> list[str]:
    """Render the Query Control sub-tab inside Market Intelligence.

    Returns:
        List of active queries (presets + custom)
    """
    st.markdown("## 🔍 Query Control")
    st.markdown("*Define which external demand signals to analyze*")

    st.info(
        "**How this works:** Changing the query set changes the external demand universe. "
        "Google AI Mode terms and Google Trends queries are normalized into canonical terms, "
        "matched against Food.com recipes, and scored for coverage gaps. "
        "Selections are stored in session state and applied consistently across all "
        "Insights tables and charts."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        # ── Section 1: Preset queries ──────────────────────────────────────
        st.markdown("### 📋 Preset Queries")

        selected_presets = st.multiselect(
            "Select preset queries",
            options=ALL_PRESETS,
            default=st.session_state.selected_presets,
            key="preset_multiselect",
        )
        st.session_state.selected_presets = selected_presets
        st.caption(f"{len(selected_presets)} preset(s) selected")

        # ── Section 2: Custom queries ──────────────────────────────────────
        st.markdown("### ✏️ Custom Queries")

        custom_input = st.text_area(
            "Add custom queries (comma-separated)",
            value=st.session_state.custom_queries,
            height=100,
            key="custom_queries_input",
            placeholder="e.g., air fryer chicken, keto pizza, vegan brownies",
        )
        custom_queries = (
            [q.strip().lower() for q in custom_input.split(",") if q.strip()]
            if custom_input.strip()
            else []
        )
        st.session_state.custom_queries = custom_input

    with col_right:
        # ── Section 3: Source toggles ──────────────────────────────────────
        st.markdown("### 📡 Data Sources")

        include_ai = st.checkbox(
            "Include Google AI Mode",
            value=st.session_state.include_ai_mode,
            key="include_ai_checkbox",
        )
        include_trends = st.checkbox(
            "Include Google Trends",
            value=st.session_state.include_trends,
            key="include_trends_checkbox",
        )
        st.session_state.include_ai_mode = include_ai
        st.session_state.include_trends = include_trends

        # ── Section 4: Run / Refresh ───────────────────────────────────────
        st.markdown("### 🔄 Run / Refresh")
        st.markdown(
            "Click **Run** to apply selected queries and sources to all "
            "Market Intelligence insights, charts, and tables."
        )

        if st.button("▶ Run Query / Refresh Data", type="primary", use_container_width=True):
            all_queries = list(set(selected_presets + custom_queries))
            st.session_state.active_queries = all_queries
            st.session_state.last_run_timestamp = datetime.now().isoformat()
            st.rerun()

    # ── Section 5: Active query set (traceability) ────────────────────────
    st.divider()
    st.markdown("### 📋 Active Query Set")

    if st.session_state.active_queries:
        st.caption(f"Last run: {st.session_state.last_run_timestamp or 'Never'}")
        st.code(st.session_state.active_queries, language="python")
    else:
        st.caption("No queries have been run yet. Select presets above and click Run.")
        st.info("Default presets will be used until you explicitly run a query set.")

    return st.session_state.active_queries


# ─────────────────────────────────────────────────────────────────────────
# GLOBAL FILTER BAR
# ─────────────────────────────────────────────────────────────────────────

def _render_global_filter_bar(gap_df: pd.DataFrame) -> tuple:
    """Render the global filter bar above tables and charts.
    
    Returns:
        Tuple of (match_status_filter, gap_score_range, selected_tags, selected_sources)
    """
    st.markdown("### 🎛️ Global Filters")
    st.markdown("*Apply filters to all tables and charts below*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        match_status_filter = st.selectbox(
            "Match Status",
            options=["All", "Strong match", "Weak match", "Gap"],
            index=0,
            key="match_status_filter",
        )
    
    with col2:
        gap_score_range = st.slider(
            "Gap Score Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.05,
            key="gap_score_filter",
        )
    
    # Get unique tags
    all_tags = []
    if "tags" in gap_df.columns:
        tags_series = gap_df["tags"].dropna()
        all_tags = sorted(set(
            tag for tags in tags_series 
            for tag in (tags if isinstance(tags, list) else str(tags).split(","))
        ))
    
    with col3:
        selected_tags = st.multiselect(
            "Tags",
            options=all_tags,
            default=[],
            key="tags_filter",
        )
    
    # Get unique sources
    all_sources = []
    if "source" in gap_df.columns:
        all_sources = sorted(gap_df["source"].dropna().unique().tolist())
    
    with col4:
        selected_sources = st.multiselect(
            "Sources",
            options=all_sources,
            default=all_sources,
            key="source_filter",
        )
    
    return match_status_filter, gap_score_range, selected_tags, selected_sources


def _apply_filters(
    df: pd.DataFrame,
    match_status_filter: str,
    gap_score_range: tuple,
    selected_tags: list,
    selected_sources: list,
) -> pd.DataFrame:
    """Apply global filters to a dataframe.
    
    Args:
        df: Input dataframe
        match_status_filter: Filter by match status
        gap_score_range: Tuple of (min, max) gap scores
        selected_tags: List of tags to filter by
        selected_sources: List of sources to filter by
    
    Returns:
        Filtered dataframe
    """
    filtered = df.copy()
    
    # Match status filter
    if match_status_filter != "All":
        status_map = {
            "Strong match": "strong_match",
            "Weak match": "weak_match",
            "Gap": "gap",
        }
        status_value = status_map.get(match_status_filter)
        if status_value and "match_status" in filtered.columns:
            filtered = filtered[filtered["match_status"] == status_value]
    
    # Gap score range filter
    if "gap_score" in filtered.columns:
        filtered = filtered[
            (filtered["gap_score"] >= gap_score_range[0]) &
            (filtered["gap_score"] <= gap_score_range[1])
        ]
    
    # Tags filter
    if selected_tags and "tags" in filtered.columns:
        def has_any_tag(tags):
            if pd.isna(tags):
                return False
            tag_list = tags if isinstance(tags, list) else str(tags).split(",")
            return any(t.strip() in selected_tags for t in tag_list)
        filtered = filtered[filtered["tags"].apply(has_any_tag)]
    
    # Sources filter
    if selected_sources and "source" in filtered.columns:
        filtered = filtered[filtered["source"].isin(selected_sources)]
    
    return filtered


# ─────────────────────────────────────────────────────────────────────────
# SORTABLE + FILTERABLE TABLES WITH COLUMN_CONFIG
# ─────────────────────────────────────────────────────────────────────────

def _render_gap_table(df: pd.DataFrame, max_rows: int = 20) -> None:
    """Render gap analysis table with sorting, filtering, and column config.
    
    Args:
        df: Gap analysis dataframe
        max_rows: Maximum rows to display
    """
    st.markdown("### 📊 Gap Analysis Table")
    st.markdown("*Sort by clicking column headers. Hover for tooltips.*")
    
    # Prepare display columns
    display_df = df.head(max_rows).copy()
    
    # Build column config
    column_config = {}
    
    if "gap_score" in display_df.columns:
        column_config["gap_score"] = st.column_config.ProgressColumn(
            "Gap Score",
            help=TOOLTIPS.get("gap_score", "Measures opportunity gap"),
            min_value=0,
            max_value=1,
            format="%.2f",
        )
    
    if "best_foodcom_similarity" in display_df.columns:
        column_config["best_foodcom_similarity"] = st.column_config.ProgressColumn(
            "Similarity",
            help=TOOLTIPS.get("similarity", "Semantic similarity to best match"),
            min_value=0,
            max_value=1,
            format="%.2f",
        )
    
    if "match_status" in display_df.columns:
        column_config["match_status"] = st.column_config.TextColumn(
            "Match Status",
            help=TOOLTIPS.get("match_status", "Strong ≥0.80, Weak 0.60–0.79, Gap <0.60"),
        )
    
    if "canonical_term" in display_df.columns:
        column_config["canonical_term"] = st.column_config.TextColumn(
            "Canonical Term",
            help="Normalized version of the query used for matching",
        )
    
    if "raw_term" in display_df.columns:
        column_config["raw_term"] = st.column_config.TextColumn(
            "Raw Term",
            help="Original search term from external source",
        )
    
    if "best_foodcom_recipe_name" in display_df.columns:
        column_config["best_foodcom_recipe_name"] = st.column_config.TextColumn(
            "Best Match",
            help="Closest Food.com recipe to the search term",
        )
    
    if "source_score" in display_df.columns:
        column_config["source_score"] = st.column_config.ProgressColumn(
            "Demand",
            help=TOOLTIPS.get("demand", "Consumer search demand"),
            min_value=0,
            max_value=1,
            format="%.2f",
        )
    
    if "opportunity_label" in display_df.columns:
        column_config["opportunity_label"] = st.column_config.TextColumn(
            "Priority",
            help=TOOLTIPS.get("opportunity_label", "High/Medium/Low priority"),
        )
    
    # Render table
    st.dataframe(
        display_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
    )
    
    # Full table expander
    with st.expander("View full gap analysis table"):
        st.dataframe(
            df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
        )


def _render_external_terms_table(df: pd.DataFrame, max_rows: int = 20) -> None:
    """Render external terms table with column config.
    
    Args:
        df: External terms dataframe
        max_rows: Maximum rows to display
    """
    st.markdown("### 📡 External Terms Table")
    
    display_df = df.head(max_rows).copy()
    
    column_config = {}
    
    if "source_score" in display_df.columns:
        column_config["source_score"] = st.column_config.ProgressColumn(
            "Demand Score",
            help=TOOLTIPS.get("demand", "Consumer search demand"),
            min_value=0,
            max_value=1,
            format="%.2f",
        )
    
    if "source" in display_df.columns:
        column_config["source"] = st.column_config.TextColumn(
            "Source",
            help="Data source: Google AI Mode or Google Trends",
        )
    
    if "raw_term" in display_df.columns:
        column_config["raw_term"] = st.column_config.TextColumn(
            "Search Term",
            help="Original search term",
        )
    
    if "canonical_term" in display_df.columns:
        column_config["canonical_term"] = st.column_config.TextColumn(
            "Canonical Term",
            help="Normalized version of the query",
        )
    
    if "fetched_date" in display_df.columns:
        column_config["fetched_date"] = st.column_config.DatetimeColumn(
            "Fetched",
            help="When this data was collected",
            format="MMM DD, YYYY",
        )
    
    st.dataframe(
        display_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
    )


def _render_clusters_table(df: pd.DataFrame, max_rows: int = 15) -> None:
    """Render clusters table with column config.
    
    Args:
        df: Clusters dataframe
        max_rows: Maximum rows to display
    """
    st.markdown("### 🔗 Clusters Table")
    
    display_df = df.head(max_rows).copy()
    
    column_config = {}
    
    if "max_gap_score" in display_df.columns:
        column_config["max_gap_score"] = st.column_config.ProgressColumn(
            "Max Gap",
            help="Highest gap score in this cluster",
            min_value=0,
            max_value=1,
            format="%.2f",
        )
    
    if "avg_gap_score" in display_df.columns:
        column_config["avg_gap_score"] = st.column_config.ProgressColumn(
            "Avg Gap",
            help="Average gap score in this cluster",
            min_value=0,
            max_value=1,
            format="%.2f",
        )
    
    if "cluster_label" in display_df.columns:
        column_config["cluster_label"] = st.column_config.TextColumn(
            "Cluster",
            help=TOOLTIPS.get("cluster", "Semantic group of similar terms"),
        )
    
    if "sources_present" in display_df.columns:
        column_config["sources_present"] = st.column_config.NumberColumn(
            "Sources",
            help="Number of source terms in this cluster",
            format="%d",
        )
    
    if "foodcom_coverage_count" in display_df.columns:
        column_config["foodcom_coverage_count"] = st.column_config.NumberColumn(
            "Food.com Coverage",
            help="Number of matching Food.com recipes",
            format="%d",
        )
    
    st.dataframe(
        display_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY LAYER
# ─────────────────────────────────────────────────────────────────────────

def _render_row_explanation(row: pd.Series) -> None:
    """Render expandable explanation for a single row.
    
    Args:
        row: DataFrame row
    """
    with st.expander("ℹ️ Why this matters"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Term Details**")
            st.markdown(f"- **Raw Term:** {row.get('raw_term', row.get('term', 'N/A'))}")
            st.markdown(f"- **Canonical Term:** {row.get('canonical_term', 'N/A')}")
            
            tags = row.get("tags")
            if tags is not None and not (isinstance(tags, float) and pd.isna(tags)) and tags:
                tag_str = tags if isinstance(tags, str) else ", ".join(str(t) for t in tags)
                st.markdown(f"- **Tags:** {tag_str}")
        
        with col2:
            st.markdown("**Match Details**")
            st.markdown(f"- **Best Match:** {row.get('best_foodcom_recipe_name', row.get('best_match', 'N/A'))}")
            sim = row.get("best_foodcom_similarity", row.get("similarity", 0))
            st.markdown(f"- **Similarity:** {sim:.2f}" if pd.notna(sim) else "- **Similarity:** N/A")
            st.markdown(f"- **Matching Method:** {row.get('matching_method', 'N/A')}")
        
        st.markdown("---")
        st.markdown("**Gap Analysis**")
        gap = row.get("gap_score", 0)
        st.markdown(f"- **Gap Score:** {gap:.2f}" if pd.notna(gap) else "- **Gap Score:** N/A")
        
        insight = row.get("insight_summary", row.get("insight", ""))
        if pd.notna(insight) and insight:
            st.markdown(f"- **Insight:** {insight}")


def _render_tooltip_glossary() -> None:
    """Render collapsible tooltip glossary."""
    with st.expander("❓ What do these metrics mean?"):
        st.markdown("""
        ### Metric Definitions
        
        **Gap Score** — Measures how much consumer demand exceeds Food.com recipe supply.
        - Higher = larger opportunity gap
        - Range: 0 to 1 (normalized)
        
        **Similarity Score** — Semantic similarity between search term and closest Food.com recipe.
        - Higher = closer match to existing recipe
        - Range: 0 to 1
        
        **Match Status** — How well Food.com covers a search term:
        - **Strong match:** similarity ≥ 0.80
        - **Weak match:** similarity 0.60–0.79
        - **Gap:** similarity < 0.60
        
        **Canonical Term** — Normalized version of the query used for matching and clustering.
        
        **Tags** — Categorization labels applied to search terms.
        
        **Clusters** — Semantic groupings of similar search terms representing distinct cuisines or dish categories.
        """)


# ─────────────────────────────────────────────────────────────────────────
# IMPROVED CHARTS WITH HOVER TOOLTIPS
# ─────────────────────────────────────────────────────────────────────────

def _gap_bar_chart_enhanced(gap_df: pd.DataFrame) -> go.Figure:
    """Enhanced gap bar chart with hover tooltips.
    
    Args:
        gap_df: Gap analysis dataframe
    
    Returns:
        Plotly figure
    """
    label_col = "opportunity_label" if "opportunity_label" in gap_df.columns else "opportunity_tier"
    term_col  = "raw_term"          if "raw_term"          in gap_df.columns else "term"

    fig = go.Figure()
    for tier in ("High", "Medium", "Low"):
        subset = gap_df[gap_df[label_col] == tier]
        if subset.empty:
            continue
        
        # Build hover data
        hover_texts = []
        for _, row in subset.iterrows():
            raw_term = row.get(term_col, "N/A")
            canonical = row.get("canonical_term", "")
            best_match = row.get("best_foodcom_recipe_name", row.get("best_match", "N/A"))
            similarity = row.get("best_foodcom_similarity", row.get("similarity", 0))
            gap_reason = row.get("insight_summary", row.get("insight", ""))
            
            hover_text = (
                f"<b>{raw_term}</b><br>"
                f"Canonical: {canonical}<br>"
                f"Best Match: {best_match}<br>"
                f"Similarity: {similarity:.2f}<br>"
                f"Gap Reason: {gap_reason}"
            )
            hover_texts.append(hover_text)
        
        fig.add_trace(go.Bar(
            x=subset[term_col],
            y=subset["gap_score"],
            name=tier,
            marker_color=_OPP_COLORS[tier],
            text=subset[term_col],
            textposition="outside",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts,
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


def _clusters_bar_chart_enhanced(clusters_df: pd.DataFrame) -> go.Figure:
    """Enhanced cluster bar chart with hover tooltips.
    
    Args:
        clusters_df: Clusters dataframe
    
    Returns:
        Plotly figure
    """
    top = clusters_df.dropna(subset=["max_gap_score"]).nlargest(20, "max_gap_score")
    
    # Build hover data
    hover_texts = []
    for _, row in top.iterrows():
        label = row.get("cluster_label", "N/A")
        rep_term = row.get("representative_term", "")
        best_match = row.get("best_foodcom_match", "N/A")
        coverage = row.get("foodcom_coverage_count", 0)
        
        hover_text = (
            f"<b>{label}</b><br>"
            f"Representative: {rep_term}<br>"
            f"Best Match: {best_match}<br>"
            f"Food.com Coverage: {coverage}"
        )
        hover_texts.append(hover_text)
    
    fig = go.Figure(go.Bar(
        x=top["cluster_label"],
        y=top["max_gap_score"],
        marker_color="#6366f1",
        text=top["sources_present"],
        textposition="outside",
        hovertemplate="%{x}<br>Max Gap: %{y:.2f}<br>Sources: %{text}<extra></extra>",
        customdata=hover_texts,
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
                                  if profile.get('avg_exclamation_count') is not None else "—"),
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
# Demand gap helpers (ENHANCED)
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
# Recipe term cluster helpers (ENHANCED)
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
# NEW: Hero Insight Panel
# ─────────────────────────────────────────────────────────────────────────

def _render_hero_insights(gap_df: pd.DataFrame, n: int = 5) -> None:
    """Render the hero insight panel using native Streamlit components."""
    st.markdown("### 🎯 Top Unmet Recipe Opportunities")
    st.markdown("*The highest-priority content gaps where demand exceeds supply*")

    opp_df = (
        gap_df[gap_df["match_status"].isin(["gap", "weak_match"])]
        .sort_values("gap_score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    if opp_df.empty:
        st.info("No significant gaps found. All search terms have good Food.com coverage.")
        return

    _STATUS = {
        "gap":          ("🔴", "GAP"),
        "weak_match":   ("🟡", "WEAK MATCH"),
        "strong_match": ("🟢", "STRONG MATCH"),
    }

    for _, row in opp_df.iterrows():
        term      = str(row.get("raw_term") or row.get("term") or "Unknown")
        gap_score = float(row.get("gap_score") or 0)
        status    = str(row.get("match_status") or "gap")
        best      = str(row.get("best_foodcom_recipe_name") or row.get("best_match") or "N/A")
        sim       = float(row.get("best_foodcom_similarity") or row.get("similarity") or 0)
        insight   = str(row.get("insight_summary") or "")
        src_raw   = row.get("source_score")
        src       = float(src_raw) if src_raw is not None and pd.notna(src_raw) else None
        tags_raw  = row.get("tags")

        icon, label = _STATUS.get(status, ("🔴", "GAP"))

        with st.container(border=True):
            title_col, badge_col = st.columns([5, 1])
            with title_col:
                st.markdown(f"**{icon} {term}**")
            with badge_col:
                st.markdown(f"`{label}`")

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Gap Score", f"{gap_score:.2f}",
                help="How much demand exceeds Food.com supply. Higher = bigger opportunity.",
            )
            m2.metric(
                "Similarity", f"{sim:.2f}",
                help="Semantic similarity to the closest Food.com recipe. Lower = weaker coverage.",
            )
            if src is not None:
                m3.metric(
                    "Demand", f"{src:.2f}",
                    help="Normalised search demand from Google AI Mode / Trends (0–1).",
                )

            if best and best != "N/A":
                st.caption(f"**Closest Food.com match:** {best[:70]}")
            if insight:
                st.caption(f"📈 {insight}")
            if (
                tags_raw is not None
                and not (isinstance(tags_raw, float) and pd.isna(tags_raw))
                and tags_raw
            ):
                tags = tags_raw if isinstance(tags_raw, list) else str(tags_raw).split(",")
                st.caption("Tags: " + " · ".join(str(t).strip() for t in tags[:5]))


# ─────────────────────────────────────────────────────────────────────────
# NEW: Scannable Decision Table
# ─────────────────────────────────────────────────────────────────────────

def _render_decision_table(gap_df: pd.DataFrame, max_rows: int = 20) -> None:
    """Render a scannable decision table with visual encodings."""
    st.markdown("### 📋 Gap Analysis — Decision Table")
    st.markdown("*Click any row to see why this opportunity was identified*")
    
    # Prepare display dataframe
    display_df = gap_df.copy()
    
    # Truncate long text
    if "raw_term" in display_df.columns:
        display_df["Term"] = display_df["raw_term"].str[:30]
    elif "term" in display_df.columns:
        display_df["Term"] = display_df["term"].str[:30]
    
    # Create demand bar (source score)
    if "source_score" in display_df.columns:
        display_df["Demand"] = display_df["source_score"].apply(
            lambda x: f"▓{'░' * int((1-x)*10)}" if pd.notna(x) else "—"
        )
    
    # Create best match column
    if "best_foodcom_recipe_name" in display_df.columns:
        display_df["Best Match"] = display_df["best_foodcom_recipe_name"].str[:35]
    elif "best_match" in display_df.columns:
        display_df["Best Match"] = display_df["best_match"].str[:35]
    
    # Create similarity bar
    sim_col = "best_foodcom_similarity" if "best_foodcom_similarity" in display_df.columns else "similarity"
    if sim_col in display_df.columns:
        display_df["Similarity"] = display_df[sim_col].apply(
            lambda x: f"▓{'░' * int((1-x)*10)} ({x:.2f})" if pd.notna(x) else "—"
        )
    
    # Create status label (plain text — st.dataframe does not render HTML)
    _STATUS_LABELS = {"gap": "🔴 Gap", "weak_match": "🟡 Weak Match", "strong_match": "🟢 Strong Match"}
    if "match_status" in display_df.columns:
        display_df["Status"] = display_df["match_status"].apply(
            lambda x: _STATUS_LABELS.get(str(x), str(x)) if pd.notna(x) else "—"
        )
    
    # Select columns for display
    cols_to_show = ["Term", "Demand", "Best Match", "Similarity", "Status"]
    available_cols = [c for c in cols_to_show if c in display_df.columns]
    
    if available_cols:
        st.dataframe(
            display_df[available_cols].head(max_rows),
            use_container_width=True,
            hide_index=True,
        )
    
    # Add expander for full table
    with st.expander("View full gap analysis table"):
        all_cols = ["raw_term", "match_status", "gap_score", "opportunity_label",
                    "top_gap_rank", "best_foodcom_recipe_name", "best_foodcom_similarity",
                    "matching_method", "source", "insight_summary", "canonical_term", "tags"]
        display_cols = [c for c in all_cols if c in gap_df.columns]
        st.dataframe(gap_df[display_cols], use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────
# NEW: Enhanced Empty State
# ─────────────────────────────────────────────────────────────────────────

def _render_empty_state(
    icon: str,
    title: str,
    description: str,
    cta_text: str = "Run Pipeline",
    cta_key: str = "run_pipeline_cta",
) -> None:
    """Render an enhanced empty state with CTA."""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button(cta_text, key=cta_key):
        st.code("docker compose exec airflow airflow dags trigger foodcom_batch_pipeline", 
                language="bash")
        st.info("Run this command in your terminal to trigger the pipeline.")


# ─────────────────────────────────────────────────────────────────────────
# NEW: Enhanced Pipeline Health
# ─────────────────────────────────────────────────────────────────────────

def _render_pipeline_health(gap_df: pd.DataFrame | None) -> None:
    """Render enhanced pipeline health with interpretations."""
    st.markdown("### 🏥 Pipeline Health")
    st.markdown("*Data freshness and quality indicators*")
    
    _files_to_check = [
        ("Recipe Tags", RECIPE_TAGS_PATH, "High coverage across Food.com corpus"),
        ("Recipe Term Index", RECIPE_TERM_INDEX_PATH, "Search index for recipe matching"),
        ("External Terms", EXTERNAL_RECIPE_TERMS_PATH, "Demand signals from external sources"),
        ("Gap Analysis", RECIPE_GAP_ANALYSIS_PATH, "Opportunity gap calculations"),
        ("Term Clusters", RECIPE_TERM_CLUSTERS_PATH, "Semantic groupings of demand"),
    ]
    
    # Render health cards
    health_cols = st.columns(len(_files_to_check))
    for col, (label, path, interpretation) in zip(health_cols, _files_to_check):
        with col:
            if path.is_file():
                try:
                    n = len(pd.read_parquet(path))
                    # Determine status based on count
                    if n > 100000:
                        status = "ok"
                    elif n > 50000:
                        status = "warning"
                    else:
                        status = "critical"
                    
                    health_card(label, n, status, interpretation)
                except Exception:
                    health_card(label, 0, "critical", "Read error — file may be corrupted")
            else:
                health_card(label, 0, "critical", "Missing — run pipeline to generate")
    
    # Gap analysis breakdown
    if gap_df is not None and not gap_df.empty and "match_status" in gap_df.columns:
        st.markdown("---")
        st.markdown("#### 📊 Gap Analysis Quality")
        
        total  = len(gap_df)
        strong = int((gap_df["match_status"] == "strong_match").sum())
        weak   = int((gap_df["match_status"] == "weak_match").sum())
        gaps   = int((gap_df["match_status"] == "gap").sum())
        
        strong_pct = 100 * strong / total if total > 0 else 0
        weak_pct = 100 * weak / total if total > 0 else 0
        gap_pct = 100 * gaps / total if total > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Strong Matches", f"{strong:,}", f"{strong_pct:.0f}% of total")
        c2.metric("Weak Matches", f"{weak:,}", f"{weak_pct:.0f}% of total")
        c3.metric("Gaps (Opportunities)", f"{gaps:,}", f"{gap_pct:.0f}% of total")
        
        # Health warnings
        if gaps == 0:
            st.warning("⚠️ **Warning:** Gap count is 0 — this may indicate an over-matching issue. Review similarity thresholds.")
        
        if weak_pct > 50:
            st.warning("⚠️ **Warning:** Weak match rate is above 50% — threshold may be too aggressive. Consider lowering strong match threshold from 0.80 to 0.75.")
        
        if strong_pct > 80:
            st.success("✅ Strong match rate is healthy (>80%) — good Food.com coverage overall.")
        
        # Insight summary check
        if "insight_summary" in gap_df.columns:
            populated = int(gap_df["insight_summary"].notna().sum())
            if populated == total:
                st.success(f"✅ All {total:,} rows have `insight_summary` populated.")
            else:
                st.warning(f"⚠️ Only {populated:,} / {total:,} rows have `insight_summary`. Some insights may be missing.")
        
        # Matching method
        if "matching_method" in gap_df.columns:
            method = gap_df["matching_method"].mode().iloc[0] if not gap_df.empty else "unknown"
            st.caption(f"**Primary matching method:** {method}")


# ─────────────────────────────────────────────────────────────────────────
# MARKET INTELLIGENCE SUB-TAB RENDERERS
# ─────────────────────────────────────────────────────────────────────────

def _render_insights_tab(
    gap_df: pd.DataFrame | None,
    ext_df: pd.DataFrame | None,
    ai_df: pd.DataFrame | None,
    trends_df: pd.DataFrame | None,
    clusters_df: pd.DataFrame | None,
) -> None:
    """Insights sub-tab: demand signals, hero gaps, chart, top clusters."""
    st.caption(
        "Modify the demand query set in the **Query Control** tab "
        "to change which terms are analyzed."
    )

    # ── Demand signals ────────────────────────────────────────────────────
    st.markdown(
        '<h2 style="display:flex;align-items:center;gap:6px;">'
        f'What Consumers Are Searching For {render_tooltip("source_score")}'
        "</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("*Real-time demand signals from Google AI Mode and Google Trends*")

    if ai_df is None:
        _render_empty_state(
            icon="🤖",
            title="AI Mode Data Not Available",
            description="Run the pipeline to populate Google AI Mode demand signals.",
            cta_text="🚀 Run Pipeline",
            cta_key="run_ai_mode_insights",
        )
    elif st.session_state.include_ai_mode:
        display_df = ai_df.sort_values("normalised_score", ascending=False).head(15)
        st.dataframe(
            display_df[["term", "normalised_score", "fetched_date"]].rename(columns={
                "term": "Term",
                "normalised_score": "Demand Score",
                "fetched_date": "Fetched",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Food topics appearing in Google AI-generated search summaries.")

    if ext_df is not None and not ext_df.empty:
        with st.expander("View all normalised external terms (AI Mode + Trends)"):
            by_src = ext_df.groupby("source").size().reset_index(name="terms")
            st.dataframe(by_src, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("## Trending Recipe Searches on Google")
    if trends_df is None:
        _render_empty_state(
            icon="📈",
            title="Google Trends Data Not Available",
            description="Run the pipeline to populate Google Trends demand signals.",
            cta_text="🚀 Run Pipeline",
            cta_key="run_trends_insights",
        )
    elif st.session_state.include_trends:
        top20 = (
            trends_df.sort_values("normalised_score", ascending=False)
            .head(20)
            [["related_query", "query_type", "normalised_score", "fetched_date"]]
            .rename(columns={
                "related_query": "Search Term",
                "query_type": "Type",
                "normalised_score": "Trend Score",
                "fetched_date": "Fetched",
            })
        )
        st.dataframe(top20, use_container_width=True, hide_index=True)

    st.divider()

    # ── Gap analysis ──────────────────────────────────────────────────────
    st.markdown("## Where Demand Exceeds Supply")
    st.markdown(
        "*Higher bars = bigger opportunity. "
        "Red = high priority, Orange = medium, Gray = low.*"
    )

    if gap_df is None:
        _render_empty_state(
            icon="📊",
            title="Gap Analysis Not Available",
            description="Run the pipeline to identify where consumer demand exceeds Food.com supply.",
            cta_text="🚀 Run Pipeline",
            cta_key="run_gap_insights",
        )
    else:
        match_filter, gap_range, selected_tags, selected_sources = _render_global_filter_bar(gap_df)
        filtered_gap = _apply_filters(gap_df, match_filter, gap_range, selected_tags, selected_sources)
        st.caption(f"Showing {len(filtered_gap)} of {len(gap_df)} terms")

        _render_hero_insights(filtered_gap, n=5)

        st.markdown("---")

        opp_df = (
            filtered_gap[filtered_gap["match_status"].isin(["gap", "weak_match"])]
            .sort_values("gap_score", ascending=False)
            .head(30)
            .reset_index(drop=True)
        )
        if not opp_df.empty:
            fig = _gap_bar_chart_enhanced(opp_df)
            fig.update_layout(title="Where Demand Exceeds Supply")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📖 Row Explanations")
        st.markdown("*Expand any row to see why this opportunity was identified*")
        for _, row in filtered_gap.head(10).iterrows():
            _render_row_explanation(row)

    # ── Clusters overview ─────────────────────────────────────────────────
    if clusters_df is not None:
        st.divider()
        st.markdown(
            '<h2 style="display:flex;align-items:center;gap:6px;">'
            f'Top Demand Clusters {render_tooltip("cluster")}'
            "</h2>",
            unsafe_allow_html=True,
        )
        fig = _clusters_bar_chart_enhanced(clusters_df)
        fig.update_layout(title="Demand Clusters (Grouped by Similar Dishes)")
        st.plotly_chart(fig, use_container_width=True)

        if "explanation" in clusters_df.columns:
            with st.expander("📖 Cluster Explanations"):
                for _, row in clusters_df.dropna(subset=["max_gap_score"]).nlargest(15, "max_gap_score").iterrows():
                    label = str(row.get("cluster_label", ""))
                    expl  = str(row.get("explanation", ""))
                    if expl:
                        st.markdown(f"**{label}** — {expl}")


def _render_raw_data_tab(
    gap_df: pd.DataFrame | None,
    ext_df: pd.DataFrame | None,
    clusters_df: pd.DataFrame | None,
) -> None:
    """Raw Data sub-tab: full sortable, filterable tables."""
    st.markdown("## Raw Data Tables")
    st.markdown("*Sort by clicking column headers. Hover column headers for metric definitions.*")

    if ext_df is not None and not ext_df.empty:
        st.markdown("### External Terms (AI Mode + Trends)")
        st.caption("All normalized terms collected from Google AI Mode and Google Trends.")
        _render_external_terms_table(ext_df, max_rows=100)
        st.divider()

    if gap_df is not None and not gap_df.empty:
        st.markdown("### Gap Analysis (Full Table)")
        st.caption("All evaluated terms with match status, gap score, and best Food.com match.")
        _render_gap_table(gap_df, max_rows=100)
        st.divider()

    if clusters_df is not None and not clusters_df.empty:
        st.markdown("### Term Clusters (Full Table)")
        st.caption("Semantic groupings of similar demand terms.")
        _render_clusters_table(clusters_df, max_rows=50)

    if ext_df is None and gap_df is None and clusters_df is None:
        _render_empty_state(
            icon="📋",
            title="No Data Available",
            description="Run the pipeline to generate raw data tables.",
            cta_text="🚀 Run Pipeline",
            cta_key="run_raw_data",
        )


def _render_methodology_tab() -> None:
    """Methodology sub-tab: metric definitions and matching process."""
    st.markdown("## Methodology & Definitions")
    st.markdown("*Every insight is traceable: raw query → canonical term → match → score → gap.*")

    _render_tooltip_glossary()

    st.divider()

    with st.expander("🔗 Pipeline data flow"):
        st.markdown("""
        ```
        Raw query  →  canonical term  →  source_score (demand)
                                      ↓
                          Food.com recipe index
                                      ↓
                          similarity score  →  match_status
                                      ↓
                              gap_score  →  opportunity_label
        ```
        - **Canonical term**: lowercased, stripped, deduplicated form of the query
        - **source_score**: normalised 0–1 demand signal from Google AI Mode or Trends
        - **similarity**: cosine similarity between term embedding and nearest Food.com recipe
        - **gap_score**: `normalize(source_score × (1 − similarity))`
        """)

    with st.expander("📐 Matching thresholds"):
        st.markdown("""
        | Status | Similarity | Meaning |
        |--------|-----------|---------|
        | **Strong match** | ≥ 0.80 | Good Food.com coverage exists |
        | **Weak match** | 0.60 – 0.79 | Partial coverage, room for improvement |
        | **Gap** | < 0.60 | Demand exceeds supply — content opportunity |

        **Primary method:** sentence-transformer embedding cosine similarity.
        **Fallback:** lexical matching (BM25 / token overlap) when embedding index is unavailable.
        """)

    with st.expander("📊 Gap score formula"):
        st.markdown("""
        ```python
        gap_score = normalize(source_score * (1 - similarity))
        ```
        Higher demand (`source_score`) × lower coverage (`similarity`) = higher gap.
        Scores are normalized 0–1 across all evaluated terms in the run.
        """)

    with st.expander("🏷️ Tag taxonomy"):
        st.markdown("""
        Tags are assigned via keyword matching against a curated taxonomy covering:
        - **Cuisine types**: mexican, italian, asian, indian, …
        - **Meal occasions**: breakfast, lunch, dinner, snack, dessert, …
        - **Dietary patterns**: vegan, keto, gluten-free, low-carb, …
        - **Cooking methods**: air fryer, slow cooker, instant pot, grilled, …
        """)


# ─────────────────────────────────────────────────────────────────────────
# Main render — Market Intelligence
# ─────────────────────────────────────────────────────────────────────────

def render() -> None:
    """Market Intelligence with sub-tabs: Insights / Query Control / Raw Data / Methodology."""
    _init_session_state()
    st.header("📊 Market Intelligence")

    gap_df      = _load_parquet(RECIPE_GAP_ANALYSIS_PATH)
    ext_df      = _load_parquet(EXTERNAL_RECIPE_TERMS_PATH)
    ai_df       = _load_parquet(AI_MODE_TERM_SCORES_PATH)
    trends_df   = _load_parquet(TRENDS_NORMALISED_PATH)
    clusters_df = _load_parquet(RECIPE_TERM_CLUSTERS_PATH)

    tab_insights, tab_query, tab_raw, tab_method = st.tabs([
        "📈 Insights",
        "🔍 Query Control",
        "📋 Raw Data",
        "🔬 Methodology",
    ])

    with tab_insights:
        _render_insights_tab(gap_df, ext_df, ai_df, trends_df, clusters_df)

    with tab_query:
        _render_query_control_tab()

    with tab_raw:
        _render_raw_data_tab(gap_df, ext_df, clusters_df)

    with tab_method:
        _render_methodology_tab()


# ─────────────────────────────────────────────────────────────────────────
# Audience & CPG Segments — separate top-level tab
# ─────────────────────────────────────────────────────────────────────────

def render_audience_cpg() -> None:
    """Audience & CPG Segments tab."""
    st.header("👥 Audience & CPG Segments")

    profiles_data = _load_cluster_profiles()
    if profiles_data is None:
        st.warning(
            f"No cluster profiles found at `{CLUSTER_PROFILE_PATH}`. "
            "Run the batch pipeline first."
        )
        return

    profiles   = profiles_data.get("clusters", {})
    all_labels = [p["cluster_label"] for p in profiles.values()]

    selected_labels = st.multiselect(
        "Select segments to display", options=all_labels, default=all_labels
    )
    if not selected_labels:
        st.warning("Select at least one segment.")
        return

    total_users = profiles_data.get("total_users", 0)
    n_clusters  = profiles_data.get("n_clusters", len(profiles))
    c1, c2 = st.columns(2)
    c1.metric("Total Segmented Users", f"{total_users:,}")
    c2.metric("Number of Segments", n_clusters)

    st.divider()
    st.markdown("### Taste Profile Radar")
    st.plotly_chart(_radar_chart(profiles, selected_labels), use_container_width=True)

    st.divider()
    st.markdown("### Segment Profiles")
    st.dataframe(
        _segment_profile_table(profiles, selected_labels),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("### CPG Brand Adjacency")
    st.caption("Recommended CPG brands and estimated programmatic CPM ranges per segment.")
    st.dataframe(
        _brand_adjacency_table(selected_labels),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("### Export")
    if st.button("Prepare PDF Report"):
        pdf_bytes = _generate_pdf(profiles, selected_labels)
        if pdf_bytes:
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="foodcom_audience_report.pdf",
                mime="application/pdf",
            )
        else:
            st.info("PDF export requires `fpdf2`. Install with `uv add fpdf2`.")


# ─────────────────────────────────────────────────────────────────────────
# Sidebar pipeline status — compact widget called from app.py
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR EXTRA DATA LOADERS  (ETL / USDA / sentiment / Airflow / clustering)
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_etl_stats_sidebar() -> dict:
    """Row counts for key ETL stages using fast Parquet metadata read."""
    staging = _resolve_staging_dir()
    keys = {
        "recipes_extracted":      staging / "recipes_extracted.parquet",
        "recipes_clean":          staging / "recipes_clean.parquet",
        "interactions_extracted": staging / "interactions_extracted.parquet",
        "interactions_clean":     staging / "interactions_clean.parquet",
    }
    out: dict = {}
    for key, path in keys.items():
        if path.is_file():
            try:
                import pyarrow.parquet as pq
                out[key] = pq.read_metadata(path).num_rows
            except Exception:
                out[key] = None
    return out


@st.cache_data(ttl=300)
def _load_usda_coverage_sidebar() -> dict | None:
    path = STAGING_DIR / "usda_nutrients.parquet"
    if not path.is_file():
        return None
    try:
        df = pd.read_parquet(path)
        total = len(df)
        if total == 0:
            return None
        cols = [c for c in df.columns if c.endswith("_per_100g")]
        matched = int(df[cols].notna().any(axis=1).sum()) if cols else 0
        return {"total": total, "matched": matched, "rate": matched / total}
    except Exception:
        return None


@st.cache_data(ttl=300)
def _load_sentiment_sidebar() -> dict | None:
    path = STAGING_DIR / "interactions_sentiment.parquet"
    if not path.is_file():
        return None
    try:
        df = pd.read_parquet(path, columns=["sentiment_score"])
        total = len(df)
        scored = int(df["sentiment_score"].notna().sum())
        return {"total": total, "scored": scored, "rate": scored / total if total else 0.0}
    except Exception:
        return None


@st.cache_data(ttl=300)
def _load_clustering_sidebar() -> dict | None:
    import json
    path = STAGING_DIR / "elbow_stats.json"
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=60)
def _load_airflow_sidebar() -> pd.DataFrame | None:
    _pg_user = os.getenv("POSTGRES_USER", "user")
    _pg_pass = os.getenv("POSTGRES_PASSWORD", "password")
    _pg_host = os.getenv("POSTGRES_HOST", "localhost")
    _pg_port = os.getenv("POSTGRES_PORT", "5432")
    _pg_db   = os.getenv("POSTGRES_DB", "foodcom")
    dsn = f"postgresql://{_pg_user}:{_pg_pass}@{_pg_host}:{_pg_port}/{_pg_db}"
    try:
        import psycopg2
        conn = psycopg2.connect(dsn, connect_timeout=3)
        query = """
            SELECT task_id, duration, state
            FROM task_instance
            WHERE dag_id = 'foodcom_batch_pipeline'
              AND run_id = (
                  SELECT run_id FROM task_instance
                  WHERE dag_id = 'foodcom_batch_pipeline'
                  ORDER BY start_date DESC LIMIT 1
              )
            ORDER BY start_date
        """
        try:
            df = pd.read_sql(query, conn)
        finally:
            conn.close()
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        return df
    except Exception:
        return None


def _file_age(path: Path) -> str:
    """Return a compact human-readable age string for a file's mtime."""
    try:
        delta = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)}m ago"
        if hours < 48:
            return f"{hours:.0f}h ago"
        return f"{delta.days}d ago"
    except OSError:
        return "unknown"


_STALE_HOURS = 25  # warn if newest staging file is older than this


def render_sidebar_pipeline_status() -> None:
    """Compact pipeline health panel rendered in the sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Pipeline Status")
        st.caption(
            "Pipeline Status shows whether the required intermediate datasets exist "
            "and how many records are available for insight generation."
        )

        # ── Freshness banner ──────────────────────────────────────────────
        _all_paths = [
            RECIPE_TAGS_PATH, RECIPE_TERM_INDEX_PATH, EXTERNAL_RECIPE_TERMS_PATH,
            RECIPE_GAP_ANALYSIS_PATH, RECIPE_TERM_CLUSTERS_PATH,
        ]
        existing_mtimes = [
            p.stat().st_mtime for p in _all_paths if p.is_file()
        ]
        if not existing_mtimes:
            st.warning("No staging files found — run the pipeline first.")
        else:
            newest = datetime.fromtimestamp(max(existing_mtimes))
            hours_ago = (datetime.now() - newest).total_seconds() / 3600
            ts = newest.strftime("%b %d, %H:%M")
            if hours_ago > _STALE_HOURS:
                st.warning(f"⚠️ Data is **{hours_ago:.0f}h old** ({ts}) — consider re-running.")
            else:
                st.success(f"Fresh — last updated {hours_ago:.0f}h ago ({ts})")

        st.markdown("**Datasets**")

        # ── Per-file status cards ─────────────────────────────────────────
        _status_files = [
            ("Recipe Tags",    RECIPE_TAGS_PATH,           "Tag assignments across Food.com recipes"),
            ("Term Index",     RECIPE_TERM_INDEX_PATH,     "Search index for recipe matching"),
            ("External Terms", EXTERNAL_RECIPE_TERMS_PATH, "Demand signals from Google AI + Trends"),
            ("Gap Analysis",   RECIPE_GAP_ANALYSIS_PATH,   "Coverage gap calculations"),
            ("Term Clusters",  RECIPE_TERM_CLUSTERS_PATH,  "Semantic groupings of demand terms"),
        ]

        for label, path, description in _status_files:
            if path.is_file():
                try:
                    n    = len(pd.read_parquet(path))
                    age  = _file_age(path)
                    icon = "✅"
                    note = f"{n:,} rows · {age}"
                except Exception:
                    icon, note = "⚠️", "read error"
            else:
                icon, note = "🔴", "missing"
            st.markdown(
                f"{icon} **{label}** — {note} {tooltip_icon(description)}",
                unsafe_allow_html=True,
            )

        # ── Gap quality summary ───────────────────────────────────────────
        gap_df = _load_parquet(RECIPE_GAP_ANALYSIS_PATH)
        if gap_df is not None and "match_status" in gap_df.columns:
            total  = len(gap_df)
            strong = int((gap_df["match_status"] == "strong_match").sum())
            weak   = int((gap_df["match_status"] == "weak_match").sum())
            gaps   = int((gap_df["match_status"] == "gap").sum())

            st.markdown("---")
            st.markdown(
                f'**Match breakdown** {render_tooltip("match_status")}',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"🟢 **{strong:,}** strong &nbsp;"
                f"🟡 **{weak:,}** weak &nbsp;"
                f"🔴 **{gaps:,}** gaps"
            )

            if "matching_method" in gap_df.columns and not gap_df.empty:
                method = gap_df["matching_method"].mode().iloc[0]
                st.caption(f"Matching method: {method}")

            # insight_summary coverage
            if "insight_summary" in gap_df.columns and total > 0:
                populated = int(gap_df["insight_summary"].notna().sum())
                if populated < total:
                    st.caption(f"⚠️ {populated:,}/{total:,} rows have insight summary")

            # Quality warnings
            if total > 0:
                weak_pct   = weak / total
                strong_pct = strong / total
                if gaps == 0:
                    st.warning(
                        "No gaps detected. This may indicate overly permissive "
                        "matching rather than full recipe coverage."
                    )
                elif weak_pct > 0.5:
                    st.warning(
                        f"Weak match rate is {weak_pct:.0%} — threshold may be "
                        "too aggressive. Consider lowering strong-match threshold."
                    )
                elif strong_pct > 0.8:
                    st.success(f"Strong match rate {strong_pct:.0%} — good coverage.")

        # ── Expandable ETL & data-quality section ─────────────────────────
        with st.expander("📋 ETL & data quality"):
            # ① Recipe / interaction ETL counts + loss
            etl = _load_etl_stats_sidebar()
            if etl:
                st.markdown("**ETL row counts**")
                raw_r   = etl.get("recipes_extracted")
                clean_r = etl.get("recipes_clean")
                raw_i   = etl.get("interactions_extracted")
                clean_i = etl.get("interactions_clean")
                if raw_r and clean_r:
                    loss = (raw_r - clean_r) / raw_r
                    st.caption(
                        f"Recipes: {clean_r:,} clean / {raw_r:,} raw "
                        f"({loss:.1%} loss)"
                    )
                if raw_i and clean_i:
                    loss = (raw_i - clean_i) / raw_i
                    st.caption(
                        f"Interactions: {clean_i:,} clean / {raw_i:,} raw "
                        f"({loss:.1%} loss)"
                    )
            else:
                st.caption("ETL files not found.")

            st.markdown("---")

            # ② USDA nutrient coverage
            usda = _load_usda_coverage_sidebar()
            st.markdown(
                f'**USDA coverage** {tooltip_icon("Fraction of ingredients matched to USDA nutrient database")}',
                unsafe_allow_html=True,
            )
            if usda:
                st.caption(
                    f"{usda['matched']:,} / {usda['total']:,} ingredients "
                    f"({usda['rate']:.1%})"
                )
            else:
                st.caption("usda_nutrients.parquet not found.")

            # ③ Sentiment coverage
            sent = _load_sentiment_sidebar()
            st.markdown(
                f'**Sentiment coverage** {tooltip_icon("Fraction of interactions scored by VADER sentiment analysis")}',
                unsafe_allow_html=True,
            )
            if sent:
                st.caption(
                    f"{sent['scored']:,} / {sent['total']:,} interactions "
                    f"({sent['rate']:.1%})"
                )
            else:
                st.caption("interactions_sentiment.parquet not found.")

            st.markdown("---")

            # ④ Clustering health
            clust = _load_clustering_sidebar()
            st.markdown(
                f'**Clustering** {tooltip_icon("K-means clustering health — optimal k chosen by silhouette score")}',
                unsafe_allow_html=True,
            )
            if clust:
                k = clust.get("chosen_k", "—")
                k_range = clust.get("k_range", [])
                silhouettes = clust.get("silhouettes", [])
                if k in k_range and silhouettes:
                    best_sil = silhouettes[k_range.index(k)]
                    st.caption(f"Optimal k={k}, silhouette={best_sil:.4f}")
                else:
                    st.caption(f"Optimal k={k}")
            else:
                st.caption("elbow_stats.json not found.")

            st.markdown("---")

            # ⑤ Airflow task runtimes (last run, text list — no chart in sidebar)
            _TASK_ICONS = {
                "success": "✅", "failed": "🔴", "running": "🟡", "skipped": "⚪",
            }
            af = _load_airflow_sidebar()
            st.markdown(
                f'**Airflow runtimes** {tooltip_icon("Task durations from the most recent DAG run")}',
                unsafe_allow_html=True,
            )
            if af is None:
                st.caption("Airflow DB not reachable.")
            elif af.empty:
                st.caption("No runs found for foodcom_batch_pipeline.")
            else:
                for _, row in af.iterrows():
                    dur  = f"{row['duration']:.0f}s" if pd.notna(row.get("duration")) else "—"
                    icon = _TASK_ICONS.get(str(row.get("state", "")), "⚪")
                    st.caption(f"{icon} {row['task_id']}: {dur}")