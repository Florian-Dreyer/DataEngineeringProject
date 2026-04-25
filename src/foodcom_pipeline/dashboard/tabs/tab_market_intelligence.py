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


def _render_sidebar_query_panel() -> list[str]:
    """Render the sidebar query control panel.
    
    Returns:
        List of active queries (presets + custom)
    """
    _init_session_state()
    
    with st.sidebar:
        st.markdown("## 🔍 Query Control Panel")
        st.markdown("*Define which demand signals to analyze*")
        
        # ── Section 1: Preset queries ───────────────────────────────────
        st.markdown("### 📋 Preset Queries")
        
        selected_presets = st.multiselect(
            "Select preset queries",
            options=ALL_PRESETS,
            default=st.session_state.selected_presets,
            key="preset_multiselect",
        )
        
        st.session_state.selected_presets = selected_presets
        st.caption(f"Selected: {len(selected_presets)} presets")
        
        # ── Section 2: Custom queries ───────────────────────────────────
        st.markdown("### ✏️ Custom Queries")
        
        custom_input = st.text_area(
            "Add custom queries (comma-separated)",
            value=st.session_state.custom_queries,
            height=80,
            key="custom_queries_input",
            placeholder="e.g., air fryer chicken, keto pizza, vegan brownies",
        )
        
        # Parse custom queries
        custom_queries = []
        if custom_input.strip():
            custom_queries = [
                q.strip().lower() 
                for q in custom_input.split(",") 
                if q.strip()
            ]
        
        st.session_state.custom_queries = custom_input
        
        # ── Section 3: Source toggles ───────────────────────────────────
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
        
        # ── Section 4: Run / Refresh ───────────────────────────────────
        st.markdown("### 🔄 Actions")
        
        if st.button("Run Query / Refresh Data", type="primary", use_container_width=True):
            # Merge presets and custom queries
            all_queries = list(set(selected_presets + custom_queries))
            st.session_state.active_queries = all_queries
            st.session_state.last_run_timestamp = datetime.now().isoformat()
            st.rerun()
        
        # ── Section 5: Transparency note ────────────────────────────────
        st.info(
            "📝 **Transparency Note**\n\n"
            "Queries determine which external demand signals are analyzed. "
            "Results reflect Google Trends and Google AI outputs for the selected terms."
        )
        
        # ── Section 6: Active query set (traceability) ─────────────────
        if st.session_state.active_queries:
            st.markdown("---")
            st.markdown("### 📋 Active Query Set")
            st.caption(f"Last run: {st.session_state.last_run_timestamp or 'Never'}")
            
            with st.expander("View active queries"):
                st.code(st.session_state.active_queries, language="python")
    
    # Return active queries
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
    """Render the hero insight panel with top opportunities."""
    st.markdown("### 🎯 Top Unmet Recipe Opportunities")
    st.markdown("*The highest-priority content gaps where demand exceeds supply*")
    
    # Get top gaps
    opp_df = (
        gap_df[gap_df["match_status"].isin(["gap", "weak_match"])]
        .sort_values("gap_score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    
    if opp_df.empty:
        st.info("No significant gaps found. All search terms have good Food.com coverage.")
        return
    
    # Render each insight card
    for idx, row in opp_df.iterrows():
        term = str(row.get("raw_term", row.get("term", "Unknown")))
        gap_score = float(row.get("gap_score", 0))
        match_status = str(row.get("match_status", "gap"))
        best_match = str(row.get("best_foodcom_recipe_name", row.get("best_match", "N/A")))
        similarity = float(row.get("best_foodcom_similarity", row.get("similarity", 0)))
        insight = str(row.get("insight_summary", ""))
        source_score = row.get("source_score")
        tags = row.get("tags", [])
        
        # Generate the card HTML
        card_html = insight_card(
            term=term,
            gap_score=gap_score,
            match_status=match_status,
            best_match=best_match[:50] + "..." if len(best_match) > 50 else best_match,
            similarity=similarity,
            insight=insight,
            source_score=source_score,
            tags=tags,
        )
        
        st.markdown(card_html, unsafe_allow_html=True)


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
# Main render (REDESIGNED)
# ─────────────────────────────────────────────────────────────────────────

def render() -> None:
    """Main render function with tab-based navigation."""
    st.header("📊 Market Intelligence")
    
    # ── Render sidebar query panel ───────────────────────────────────────
    active_queries = _render_sidebar_query_panel()
    
    # ── Tab navigation ───────────────────────────────────────────────────
    tab_market, tab_gap, tab_clusters, tab_health = st.tabs([
        "📈 Market Intelligence",
        "📊 Gap Analysis", 
        "🔗 Clusters",
        "🏥 Pipeline Health",
    ])
    
    # Load data
    gap_df = _load_parquet(RECIPE_GAP_ANALYSIS_PATH)
    ext_df = _load_parquet(EXTERNAL_RECIPE_TERMS_PATH)
    ai_df = _load_parquet(AI_MODE_TERM_SCORES_PATH)
    trends_df = _load_parquet(TRENDS_NORMALISED_PATH)
    clusters_df = _load_parquet(RECIPE_TERM_CLUSTERS_PATH)
    
    # ── TAB 1: Market Intelligence ───────────────────────────────────────
    with tab_market:
        st.markdown("## What Consumers Are Searching For")
        st.markdown("*Real-time demand signals from Google AI Mode and Google Trends*")
        
        # Tooltip explanation
        st.markdown(f"""
        <div style="background: #f0f9ff; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
            <span style="font-size: 12px; color: #0369a1;">
                ℹ️ {TOOLTIPS['source_score']}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Mode data
        if ai_df is None:
            _render_empty_state(
                icon="🤖",
                title="AI Mode Data Not Available",
                description="This section shows real-time consumer demand from Google AI Mode — the AI answers users see before any search results.",
                cta_text="🚀 Run Pipeline to Populate",
                cta_key="run_ai_mode_pipeline",
            )
        else:
            # Apply source filter
            if st.session_state.include_ai_mode:
                display_df = ai_df.sort_values("normalised_score", ascending=False).head(15)
                st.dataframe(
                    display_df[["term", "normalised_score", "fetched_date"]].rename(columns={
                        "term": "Term",
                        "normalised_score": "Demand Score",
                        "fetched_date": "Fetched"
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            st.caption("These are the food topics appearing in AI-generated search summaries.")
        
        # External terms breakdown
        if ext_df is not None and not ext_df.empty:
            with st.expander("View all normalised external terms (AI Mode + Trends)"):
                by_src = ext_df.groupby("source").size().reset_index(name="terms")
                st.dataframe(by_src, use_container_width=True, hide_index=True)
                _render_external_terms_table(ext_df, max_rows=50)
        
        st.divider()
        
        # Trending searches
        st.markdown("## Trending Recipe Searches on Google")
        
        if trends_df is None:
            _render_empty_state(
                icon="📈",
                title="Google Trends Data Not Available",
                description="This section shows what's trending on Google Searches related to food and recipes.",
                cta_text="🚀 Run Pipeline",
                cta_key="run_trends_pipeline",
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
                    "fetched_date": "Fetched"
                })
            )
            st.dataframe(top20, use_container_width=True, hide_index=True)
    
    # ── TAB 2: Gap Analysis ──────────────────────────────────────────────
    with tab_gap:
        st.markdown("## Where Demand Exceeds Supply")
        st.markdown("*Higher bars = bigger opportunity gaps. These are the cuisines and ingredients where consumer demand is outpacing available content.*")
        
        if gap_df is None:
            _render_empty_state(
                icon="📊",
                title="Gap Analysis Not Available",
                description="Run the pipeline to identify where consumer demand exceeds Food.com recipe supply.",
                cta_text="🚀 Run Pipeline",
                cta_key="run_gap_pipeline",
            )
        else:
            # Global filter bar
            match_filter, gap_range, selected_tags, selected_sources = _render_global_filter_bar(gap_df)
            
            # Apply filters
            filtered_gap = _apply_filters(gap_df, match_filter, gap_range, selected_tags, selected_sources)
            
            st.markdown(f"*Showing {len(filtered_gap)} of {len(gap_df)} terms*")
            
            # Hero insight panel
            _render_hero_insights(filtered_gap, n=5)
            
            st.markdown("---")
            
            # Enhanced gap bar chart
            opp_df = (
                filtered_gap[filtered_gap["match_status"].isin(["gap", "weak_match"])]
                .sort_values("gap_score", ascending=False)
                .head(30)
                .reset_index(drop=True)
            )
            
            fig = _gap_bar_chart_enhanced(opp_df)
            fig.update_layout(
                title="Where Demand Exceeds Supply",
                xaxis_title="Term",
                yaxis_title="Gap Score",
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Sortable/filterable gap table
            _render_gap_table(filtered_gap, max_rows=20)
            
            # Explainability: Row explanations
            st.markdown("### 📖 Row Explanations")
            st.markdown("*Expand any row below to see why this opportunity was identified*")
            
            for idx, row in filtered_gap.head(10).iterrows():
                _render_row_explanation(row)
            
            # Tooltip glossary
            _render_tooltip_glossary()
    
    # ── TAB 3: Clusters ──────────────────────────────────────────────────
    with tab_clusters:
        st.markdown("## How These Opportunities Were Identified")
        st.markdown("*Demand clusters group similar search terms using semantic similarity. Each cluster represents a distinct cuisine or dish category.*")
        
        # Cluster tooltip
        st.markdown(f"""
        <div style="background: #f0f9ff; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
            <span style="font-size: 12px; color: #0369a1;">
                ℹ️ {TOOLTIPS['cluster']}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        if clusters_df is None:
            _render_empty_state(
                icon="🔗",
                title="Cluster Data Not Available",
                description="Run the clustering pipeline to group similar demand terms together.",
                cta_text="🚀 Run Clustering",
                cta_key="run_cluster_pipeline",
            )
        else:
            # Enhanced cluster bar chart
            fig = _clusters_bar_chart_enhanced(clusters_df)
            fig.update_layout(
                title="Demand Clusters (Grouped by Similar Dishes)",
                xaxis_title="Cluster",
                yaxis_title="Max Gap Score",
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Clusters table
            _render_clusters_table(clusters_df, max_rows=15)
            
            # Cluster explanations
            if "explanation" in clusters_df.columns:
                with st.expander("📖 Cluster Explanations"):
                    for _, row in clusters_df.dropna(subset=["max_gap_score"]).nlargest(15, "max_gap_score").iterrows():
                        label = str(row.get("cluster_label", ""))
                        expl  = str(row.get("explanation", ""))
                        if expl:
                            st.markdown(f"**{label}** — {expl}")
            
            # Tooltip glossary
            _render_tooltip_glossary()
    
    # ── TAB 4: Pipeline Health ───────────────────────────────────────────
    with tab_health:
        _render_pipeline_health(gap_df)
    
    # ── Section 6: Audience & CPG Segments (existing, unchanged) ────────
    st.markdown("---")
    st.markdown("## Audience & CPG Segments")
    
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
    
    st.markdown("### Taste Profile Radar")
    st.plotly_chart(_radar_chart(profiles, selected_labels), use_container_width=True)
    
    st.divider()
    
    st.markdown("### Segment Profiles")
    st.dataframe(_segment_profile_table(profiles, selected_labels), use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.markdown("### CPG Brand Adjacency")
    st.caption("Static mapping of segments to recommended CPG brands and estimated programmatic CPM ranges.")
    st.dataframe(_brand_adjacency_table(selected_labels), use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.markdown("### Export")
    if st.button("Prepare PDF Report"):
        pdf_bytes = _generate_pdf(profiles, selected_labels)
        if pdf_bytes:
            st.download_button("Download PDF Report", data=pdf_bytes,
                                file_name="foodcom_audience_report.pdf", mime="application/pdf")
        else:
            st.info("PDF export requires `fpdf2`. Install with `uv add fpdf2`.")