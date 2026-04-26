"""Dashboard UI components — reusable widgets for the decision-focused dashboard."""

import re
import streamlit as st
import plotly.graph_objects as go
from typing import Optional


# =============================================================================
# TOOLTIP DEFINITIONS
# =============================================================================

TOOLTIPS = {
    "gap_score": """
    **Gap Score** measures how much consumer demand exceeds available Food.com recipes.
    
    • **Higher = larger opportunity** — more demand, less supply
    • **Range:** 0 to 1 (normalized)
    • **Formula:** (external demand - Food.com coverage) normalized
    """,
    
    "similarity": """
    **Similarity Score** is the semantic similarity between the search term and the closest Food.com recipe.
    
    • **Range:** 0 to 1
    • **Higher = closer match** to an existing recipe
    • **Method:** Embedding-based similarity with lexical fallback
    """,
    
    "match_status": """
    **Match Status** classifies how well Food.com covers a search term:
    
    • **Strong match:** similarity ≥ 0.80 — Good coverage exists
    • **Weak match:** similarity 0.60–0.79 — Partial coverage
    • **Gap:** similarity < 0.60 — Demand exceeds supply
    """,
    
    "source_score": """
    **Source Score** represents demand from Google Trends or AI Mode.
    
    • **Higher = more searched** or surfaced by Google
    • **Range:** 0 to 1 (normalized across all terms)
    • **Sources:** Google AI Mode, Google Trends
    """,
    
    "opportunity_label": """
    **Opportunity Label** categorizes the priority of a gap:
    
    • **High:** Gap score ≥ 0.70 — Priority content opportunity
    • **Medium:** Gap score 0.40–0.69 — Monitor and consider
    • **Low:** Gap score < 0.40 — Lower priority
    """,
    
    "demand": """
    **Demand** shows how much consumers are searching for this term.
    
    • Derived from Google Trends and AI Mode data
    • Higher bars = more consumer interest
    • Use this to prioritize content creation
    """,
    
    "cluster": """
    **Clusters** group similar search terms using semantic similarity.
    
    • Each cluster represents a distinct cuisine or dish category
    • Shows common demand patterns across related terms
    • Helps identify broader market trends
    """,
    
    "pipeline_health": """
    **Pipeline Health** shows the status of data processing.
    
    • **✓ present:** Data file exists and is readable
    • **⚠ warning:** Data exists but may be stale or incomplete
    • **✗ missing:** Data file not found — run pipeline
    """,
}


# =============================================================================
# COMPONENT FUNCTIONS
# =============================================================================

def tooltip_icon(text: str) -> str:
    """Return a (?) hover icon for arbitrary tooltip text.

    Uses the CSS classes from theme.py (.tooltip-container / .tooltip-icon /
    .tooltip-text) and adds a native `title` attribute as a reliable fallback.
    Markdown bold markers (**) are stripped so the raw text reads cleanly.
    """
    if not text:
        return ""
    clean = re.sub(r"\*+", "", text).strip()
    safe_title = clean.replace('"', "&quot;").replace("\n", " ")
    return (
        '<span class="tooltip-container" style="display:inline-flex;vertical-align:middle;">'
        f'<span class="tooltip-icon" title="{safe_title}">?</span>'
        f'<span class="tooltip-text">{clean}</span>'
        "</span>"
    )


def render_tooltip(key: str) -> str:
    """Get tooltip HTML for a given metric key from the TOOLTIPS dictionary."""
    return tooltip_icon(TOOLTIPS.get(key, ""))


def metric_with_tooltip(
    label: str,
    value: str,
    tooltip_key: str,
    delta: Optional[str] = None,
    status: Optional[str] = None,
) -> None:
    """Render a metric with an info tooltip."""
    col1, col2 = st.columns([1, 0.1])
    
    with col1:
        if delta:
            st.metric(label=label, value=value, delta=delta)
        else:
            st.metric(label=label, value=value)
    
    with col2:
        st.markdown(render_tooltip(tooltip_key), unsafe_allow_html=True)


def badge(status: str, status_type: str = "match") -> str:
    """Generate HTML for a status badge.
    
    Args:
        status: The status value (gap, weak, strong, high, medium, low)
        status_type: Type of status ('match' or 'opportunity')
    
    Returns:
        HTML string for the badge
    """
    status_lower = status.lower().replace(" ", "_")
    
    # Map status to CSS class
    if status_type == "match":
        class_map = {
            "gap": "badge-gap",
            "weak_match": "badge-weak",
            "strong_match": "badge-strong",
        }
    else:  # opportunity
        class_map = {
            "high": "badge-high",
            "medium": "badge-medium",
            "low": "badge-low",
        }
    
    css_class = class_map.get(status_lower, "badge-low")
    display_text = status.replace("_", " ").title()
    
    return f'<span class="badge {css_class}">{display_text}</span>'


def progress_bar(value: float, max_value: float = 1.0, bar_type: str = "default") -> str:
    """Generate HTML for a progress bar.
    
    Args:
        value: Current value (0 to 1)
        max_value: Maximum value (default 1.0)
        bar_type: Type of bar ('default', 'gap', 'weak', 'strong', 'demand')
    
    Returns:
        HTML string for the progress bar
    """
    percentage = min(100, max(0, (value / max_value) * 100))
    
    # Determine color class based on value and type
    if bar_type == "gap":
        color_class = "gap"
    elif bar_type == "similarity":
        if percentage < 60:
            color_class = "gap"
        elif percentage < 80:
            color_class = "weak"
        else:
            color_class = "strong"
    elif bar_type == "demand":
        color_class = "demand"
    else:
        color_class = "default"
    
    return f"""
    <div class="progress-bar-container">
        <div class="progress-bar">
            <div class="progress-bar-fill {color_class}" style="width: {percentage}%"></div>
        </div>
        <span style="font-size: 12px; color: #64748b; min-width: 40px;">{value:.2f}</span>
    </div>
    """


def insight_card(
    term: str,
    gap_score: float,
    match_status: str,
    best_match: str,
    similarity: float,
    insight: str,
    source_score: Optional[float] = None,
    tags: Optional[list] = None,
) -> str:
    """Generate HTML for an insight card.
    
    Args:
        term: The search term
        gap_score: Gap score (0-1)
        match_status: gap, weak_match, or strong_match
        best_match: Best matching Food.com recipe
        similarity: Similarity score (0-1)
        insight: One-line explanation
        source_score: Optional demand score
        tags: Optional list of tags
    
    Returns:
        HTML string for the card
    """
    # Determine status class
    status_class = match_status.lower().replace(" ", "_")
    # Map to CSS class names used in theme.py (.insight-card.gap/.weak/.strong)
    _CSS_CLASS = {"gap": "gap", "weak_match": "weak", "strong_match": "strong"}
    card_css_class = _CSS_CLASS.get(status_class, "gap")

    # Status icon and label
    if status_class == "gap":
        status_icon = "🔴"
        status_label = "GAP"
    elif status_class == "weak_match":
        status_icon = "🟡"
        status_label = "WEAK MATCH"
    else:
        status_icon = "🟢"
        status_label = "STRONG MATCH"
    
    # Build tags HTML
    tags_html = ""
    if tags:
        tags_list = tags if isinstance(tags, list) else str(tags).split(",")
        tags_html = " ".join([f"`{t.strip()}`" for t in tags_list[:4]])
    
    # Build source score bar if available
    source_bar = ""
    if source_score is not None:
        source_bar = f"""
        <div style="margin: 8px 0;">
            <span style="font-size: 12px; color: #64748b;">Demand: </span>
            {progress_bar(source_score, 1.0, "demand")}
        </div>
        """
    
    return f"""
    <div class="insight-card {card_css_class}">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <span style="font-size: 14px;">{status_icon}</span>
            <span class="badge badge-{status_class}">{status_label}</span>
            <span style="font-weight: 600; font-size: 16px;">{term}</span>
        </div>
        
        {source_bar}
        
        <div style="margin: 8px 0;">
            <span style="font-size: 12px; color: #64748b;">Gap Score: </span>
            {progress_bar(gap_score, 1.0, "gap")}
        </div>
        
        <div style="margin: 8px 0;">
            <span style="font-size: 12px; color: #64748b;">Best Match: </span>
            <span style="color: #334155;">{best_match}</span>
        </div>
        
        <div style="margin: 8px 0;">
            <span style="font-size: 12px; color: #64748b;">Similarity: </span>
            {progress_bar(similarity, 1.0, "similarity")}
        </div>
        
        {f'<div style="margin-top: 8px; font-size: 12px;">{tags_html}</div>' if tags_html else ''}
        
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e2e8f0;">
            <span style="font-size: 14px;">📈 {insight}</span>
        </div>
    </div>
    """


def empty_state(
    icon: str,
    title: str,
    description: str,
    cta_text: Optional[str] = None,
    cta_key: Optional[str] = None,
) -> None:
    """Render an empty state with optional CTA button.
    
    Args:
        icon: Emoji icon to display
        title: Title of the empty state
        description: Description of what's missing
        cta_text: Optional CTA button text
        cta_key: Optional key for the CTA button
    """
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if cta_text:
        if st.button(cta_text, key=cta_key):
            st.info("Run the pipeline: `docker compose exec airflow airflow dags trigger foodcom_batch_pipeline`")


def detail_panel(
    raw_term: str,
    canonical_term: str,
    tags: list,
    best_match: str,
    similarity: float,
    match_status: str,
    matching_method: str,
    explanation: str,
) -> None:
    """Render an expandable detail panel for a term.
    
    Args:
        raw_term: Original search term
        canonical_term: Normalized term
        tags: List of tags
        best_match: Best Food.com match
        similarity: Similarity score
        match_status: Match status
        matching_method: How similarity was calculated
        explanation: Why this classification was made
    """
    tags_html = ", ".join([f"`{t}`" for t in tags]) if tags else "—"
    
    status_class = match_status.lower().replace(" ", "_")
    
    st.markdown(f"""
    <div class="detail-panel">
        <div class="detail-section">
            <div class="detail-section-title">📋 Term Details</div>
            <div><strong>Raw Term:</strong> {raw_term}</div>
            <div><strong>Canonical Term:</strong> {canonical_term}</div>
            <div><strong>Tags:</strong> {tags_html}</div>
        </div>
        
        <div class="detail-section">
            <div class="detail-section-title">🍳 Food.com Match</div>
            <div><strong>Best Match:</strong> {best_match}</div>
            <div><strong>Similarity:</strong> {similarity:.2f} ({badge(status_class, 'match')})</div>
            <div><strong>Method:</strong> {matching_method}</div>
        </div>
        
        <div class="detail-section">
            <div class="detail-section-title">💡 Why This Classification</div>
            <div>{explanation}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def health_card(
    label: str,
    count: int,
    status: str = "ok",
    interpretation: str = "",
) -> None:
    """Render a pipeline health card with interpretation.
    
    Args:
        label: Name of the data asset
        count: Number of records
        status: Health status ('ok', 'warning', 'critical')
        interpretation: Human-readable interpretation
    """
    status_icons = {
        "ok": "✓",
        "warning": "⚠",
        "critical": "🔴",
    }
    
    status_text = {
        "ok": "Healthy",
        "warning": "Needs Attention",
        "critical": "Action Required",
    }
    
    icon = status_icons.get(status, "✓")
    text = status_text.get(status, "Healthy")
    
    st.markdown(f"""
    <div class="health-card">
        <div class="metric-value">{count:,}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-status {status}">{icon} {text}</div>
        {f'<div style="font-size: 11px; color: #64748b; margin-top: 8px;">{interpretation}</div>' if interpretation else ''}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CHART HELPERS
# =============================================================================

def enhance_gap_chart(fig, title: str = "Where Demand Exceeds Supply", subtitle: str = "Higher bars = bigger opportunity gaps") -> go.Figure:
    """Enhance a gap bar chart with better titles and annotations.
    
    Args:
        fig: Plotly figure
        title: Chart title
        subtitle: Chart subtitle
    
    Returns:
        Enhanced Plotly figure
    """
    fig.update_layout(
        title={
            "text": f"<b>{title}</b><br><span style='font-size: 12px; color: #64748b;'>{subtitle}</span>",
            "x": 0.5,
            "xanchor": "center",
        },
        height=420,
        margin=dict(l=40, r=20, t=80, b=120),
        xaxis_title=None,
        yaxis_title="Gap Score",
    )
    
    # Add annotations for top 3 bars
    if fig.data and len(fig.data[0].x) >= 3:
        for i in range(min(3, len(fig.data[0].x))):
            fig.add_annotation(
                x=fig.data[0].x[i],
                y=fig.data[0].y[i],
                text=f"{fig.data[0].y[i]:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=0,
                ay=-20,
                font=dict(size=10, color="#334155"),
            )
    
    return fig


def enhance_cluster_chart(fig, title: str = "Demand Clusters", subtitle: str = "Grouped by similar dishes") -> go.Figure:
    """Enhance a cluster bar chart with better titles and tooltips.
    
    Args:
        fig: Plotly figure
        title: Chart title
        subtitle: Chart subtitle
    
    Returns:
        Enhanced Plotly figure
    """
    fig.update_layout(
        title={
            "text": f"<b>{title}</b><br><span style='font-size: 12px; color: #64748b;'>{subtitle}</span>",
            "x": 0.5,
            "xanchor": "center",
        },
        height=400,
        margin=dict(l=40, r=20, t=80, b=120),
        xaxis_title=None,
        yaxis_title="Max Gap Score",
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Gap Score: %{y}<br>Terms: %{text}<extra></extra>",
    )
    
    return fig