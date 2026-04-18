"""Recipe Explorer tab — hero search, filters, recipe cards, Shop buttons, pagination."""

import html
import os
import urllib.parse
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _resolve_staging_dir() -> Path:
    if env_val := os.getenv("FOODCOM_STAGING_DIR"):
        return Path(env_val)
    project_root = Path(__file__).resolve().parents[4]
    candidate = project_root / "staging"
    if candidate.exists():
        return candidate
    return Path("/opt/airflow/staging")


STAGING_DIR = _resolve_staging_dir()

_PG_USER = os.getenv("POSTGRES_USER", "user")
_PG_PASS = os.getenv("POSTGRES_PASSWORD", "password")
_PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
_PG_PORT = os.getenv("POSTGRES_PORT", "5432")
_PG_DB   = os.getenv("POSTGRES_DB", "foodcom")
_DB_DSN  = f"postgresql://{_PG_USER}:{_PG_PASS}@{_PG_HOST}:{_PG_PORT}/{_PG_DB}"

PAGE_SIZE = 10

# Nutrition axes shown on the radar (all Food.com %DV columns)
_RADAR_NUTRIENTS = ["protein", "fat", "carbs", "sugar", "sodium"]
_RADAR_LABELS    = ["Protein %DV", "Fat %DV", "Carbs %DV", "Sugar %DV", "Sodium %DV"]


# ---------------------------------------------------------------------------
# Pure helpers (independently testable — no Streamlit dependency)
# ---------------------------------------------------------------------------

def build_amazon_url(ingredients: list[str]) -> str:
    """Build an Amazon Fresh search URL for the first 8 ingredients."""
    top = ingredients[:8]
    if not top:
        return "https://www.amazon.com/s?i=amazonfresh"
    query = "+".join(urllib.parse.quote_plus(ing) for ing in top)
    return f"https://www.amazon.com/s?k={query}&i=amazonfresh"


def build_instacart_url(recipe_name: str) -> str:
    """Build an Instacart search URL for a recipe name."""
    query = urllib.parse.quote_plus(f"{recipe_name} ingredients")
    return f"https://www.instacart.com/store/s?k={query}"


def apply_filters(
    df: pd.DataFrame,
    search: str,
    max_cook_time: int,
    min_rating: float,
    selected_tags: list[str],
) -> pd.DataFrame:
    """Filter the recipe DataFrame by search text, cook time, rating, and tags."""
    if search:
        mask = (
            df["name"].str.contains(search, case=False, na=False)
            | df["top_ingredients"].str.contains(search, case=False, na=False)
        )
        df = df[mask]
    df = df[df["avg_cook_minutes"].fillna(0) <= max_cook_time]
    df = df[df["display_rating"].fillna(0) >= min_rating]
    if selected_tags:
        def _has_tag(tag_str: str) -> bool:
            return any(t.lower() in str(tag_str).lower() for t in selected_tags)
        df = df[df["tags"].apply(_has_tag)]
    return df


def nutrition_bar_color(pct_dv: float, nutrient: str) -> str:
    """Return an emerald/amber/red hex colour based on nutrient and %DV level."""
    if nutrient == "sugar" and pct_dv > 30:
        return "#f59e0b"  # amber — high sugar
    if nutrient == "sodium" and pct_dv > 50:
        return "#ef4444"  # red — high sodium
    return "#10b981"      # emerald — default


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_recipes() -> pd.DataFrame:
    """
    Load recipe data. Tries PostgreSQL dim_recipe first; falls back to parquet.
    Returns a DataFrame with a normalised schema. Returns empty DataFrame on failure.
    """
    # Primary: PostgreSQL dim_recipe
    try:
        import psycopg2
        conn = psycopg2.connect(_DB_DSN, connect_timeout=3)
        try:
            df = pd.read_sql("""
                SELECT recipe_id, name,
                       COALESCE(avg_rating, 0)      AS avg_rating,
                       sentiment_rating,
                       COALESCE(avg_cook_minutes, 0) AS avg_cook_minutes,
                       COALESCE(top_ingredients, '') AS top_ingredients,
                       COALESCE(tags, '')            AS tags,
                       COALESCE(ingredient_count, 0) AS ingredient_count,
                       calories, protein, fat, sugar, sodium, carbs, saturated_fat
                FROM dim_recipe
                ORDER BY COALESCE(sentiment_rating, avg_rating) DESC NULLS LAST
            """, conn)
        finally:
            conn.close()
        if not df.empty:
            df["display_rating"] = df["sentiment_rating"].fillna(df["avg_rating"])
            return df
    except Exception:
        pass

    # Fallback: parquet staging files
    recipes_path   = STAGING_DIR / "recipes_clean.parquet"
    sentiment_path = STAGING_DIR / "recipe_sentiment_ratings.parquet"

    if not recipes_path.exists():
        return pd.DataFrame()

    nutrition_cols = ["calories", "protein", "fat", "sugar", "sodium", "carbs", "saturated_fat"]
    probe_df = pd.read_parquet(recipes_path, columns=[])
    keep = ["id", "name", "minutes", "tags", "ingredients", "n_ingredients"] + nutrition_cols
    available = [c for c in keep if c in probe_df.columns]

    df = pd.read_parquet(recipes_path, columns=available).rename(columns={
        "id": "recipe_id",
        "minutes": "avg_cook_minutes",
        "n_ingredients": "ingredient_count",
    })

    # Build top_ingredients from the ingredients list column
    if "ingredients" in df.columns:
        df["top_ingredients"] = df["ingredients"].apply(
            lambda x: "|".join(str(i) for i in x[:10]) if isinstance(x, list) else ""
        )
    else:
        df["top_ingredients"] = ""

    df["avg_rating"] = float("nan")

    if sentiment_path.exists():
        sent = pd.read_parquet(sentiment_path, columns=["recipe_id", "sentiment_rating"])
        df = df.merge(sent, on="recipe_id", how="left")
    else:
        df["sentiment_rating"] = float("nan")

    df["display_rating"] = df["sentiment_rating"].fillna(df["avg_rating"])
    df = df.sort_values("display_rating", ascending=False, na_position="last")

    # Ensure all expected nutrition columns exist (fill missing with NaN)
    for col in nutrition_cols:
        if col not in df.columns:
            df[col] = float("nan")

    return df


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _nutrition_radar(row: pd.Series) -> go.Figure:
    """Plotly polar chart for 5 nutrition axes (%DV values)."""
    def _safe_float(v) -> float:
        return 0.0 if v is None or pd.isna(v) else float(v)
    values = [_safe_float(row.get(n)) for n in _RADAR_NUTRIENTS]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=_RADAR_LABELS + [_RADAR_LABELS[0]],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.15)",
        line=dict(color="#10b981", width=2),
        name="%DV",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9))),
        showlegend=False,
        height=220,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    return fig


def _render_recipe_card(row: pd.Series) -> None:
    """Render one full-width recipe card inside a styled container."""
    name          = str(row.get("name", "Unknown"))
    cook_min      = row.get("avg_cook_minutes")
    n_ingredients = row.get("ingredient_count")
    bayesian      = row.get("sentiment_rating")
    raw_rating    = row.get("avg_rating")
    top_ingr_str  = str(row.get("top_ingredients") or "")
    ingredients   = [i.strip() for i in top_ingr_str.split("|") if i.strip()]
    trend_index   = row.get("trend_index")  # None/NaN when Google Trends not yet wired

    amazon_url    = build_amazon_url(ingredients)
    instacart_url = build_instacart_url(name)

    with st.container():
        st.markdown(
            '<div style="background:white;border-radius:10px;border:1px solid #e5e7eb;'
            'padding:20px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">',
            unsafe_allow_html=True,
        )

        # --- Top row: name + shop buttons ---
        col_info, col_shop = st.columns([3, 1])

        with col_info:
            subtitle_parts = []
            if cook_min is not None and not pd.isna(cook_min):
                subtitle_parts.append(f"⏱ {int(cook_min)} min")
            if n_ingredients is not None and not pd.isna(n_ingredients):
                subtitle_parts.append(f"{int(n_ingredients)} ingredients")
            st.markdown(f"### {name}")
            if subtitle_parts:
                st.caption(" · ".join(subtitle_parts))

            # Rating badges
            badge_parts = []
            if bayesian is not None and not pd.isna(bayesian):
                badge_parts.append(
                    f'<span style="background:#10b981;color:white;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;font-weight:700;">⭐ {bayesian:.1f} Bayesian</span>'
                )
            if raw_rating is not None and not pd.isna(raw_rating):
                badge_parts.append(
                    f'<span style="background:#f3f4f6;color:#374151;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;">{raw_rating:.1f} raw</span>'
                )
            if (bayesian is not None and not pd.isna(bayesian)
                    and raw_rating is not None and not pd.isna(raw_rating)):
                delta = bayesian - raw_rating
                if delta > 0:
                    badge_parts.append(
                        f'<span style="color:#059669;font-size:12px;font-weight:600;">'
                        f'↑ +{delta:.1f} sentiment boost</span>'
                    )
            if trend_index is not None and not pd.isna(trend_index):
                level = "🔥 High" if trend_index > 66 else ("📈 Medium" if trend_index > 33 else "📉 Low")
                badge_parts.append(
                    f'<span style="background:#fef3c7;color:#92400e;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;">{level} trend demand</span>'
                )
            if badge_parts:
                st.markdown(" &nbsp; ".join(badge_parts), unsafe_allow_html=True)

        with col_shop:
            st.markdown(
                f'<a href="{amazon_url}" target="_blank" style="display:block;background:#10b981;'
                f'color:white;text-align:center;border-radius:6px;padding:9px 12px;'
                f'font-weight:700;font-size:13px;text-decoration:none;margin-bottom:6px;">'
                f'🛒 Shop on Amazon Fresh</a>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<a href="{instacart_url}" target="_blank" style="display:block;background:white;'
                f'color:#374151;text-align:center;border-radius:6px;padding:8px 12px;'
                f'font-size:13px;text-decoration:none;border:1px solid #e5e7eb;">'
                f'🥬 Instacart</a>',
                unsafe_allow_html=True,
            )

        # --- Bottom row: radar | nutrition bars | ingredient pills ---
        col_radar, col_bars, col_pills = st.columns([1, 1.5, 1])

        with col_radar:
            st.plotly_chart(_nutrition_radar(row), use_container_width=True, key=f"radar_{name[:20]}")

        with col_bars:
            st.markdown("**Nutrition (% Daily Value)**")
            for nutrient, label in zip(_RADAR_NUTRIENTS, _RADAR_LABELS):
                val = row.get(nutrient)
                if val is None or pd.isna(val):
                    continue
                pct = float(val)
                color = nutrition_bar_color(pct, nutrient)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:11px;color:#6b7280;margin-bottom:2px;">'
                    f'<span>{label}</span><span>{pct:.0f}%</span></div>'
                    f'<div style="height:6px;border-radius:3px;background:#f3f4f6;margin-bottom:6px;">'
                    f'<div style="height:6px;border-radius:3px;background:{color};'
                    f'width:{min(pct, 100):.0f}%;"></div></div>',
                    unsafe_allow_html=True,
                )

        with col_pills:
            st.markdown("**Ingredients**")
            if ingredients:
                pills_html = "".join(
                    f'<span style="display:inline-block;background:#f3f4f6;border-radius:12px;'
                    f'padding:3px 10px;font-size:11px;color:#6b7280;margin:2px;">{html.escape(ing)}</span>'
                    for ing in ingredients[:8]
                )
                if len(ingredients) > 8:
                    pills_html += (
                        f'<span style="display:inline-block;background:#e5e7eb;border-radius:12px;'
                        f'padding:3px 10px;font-size:11px;color:#9ca3af;margin:2px;">'
                        f'+{len(ingredients)-8} more</span>'
                    )
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.caption("—")

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    # Hero header
    st.markdown(
        '<div style="background:linear-gradient(135deg,#10b981,#059669);border-radius:10px;'
        'padding:24px 28px;margin-bottom:16px;">'
        '<h1 style="color:white;margin:0;font-size:26px;">🍳 Recipe Explorer</h1>'
        '<p style="color:#d1fae5;margin:6px 0 14px;">Search 231,637 recipes — '
        'see Bayesian ratings, nutrition, and shop ingredients in one click.</p>',
        unsafe_allow_html=True,
    )
    search = st.text_input("", placeholder="Search recipes or ingredients...", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    df = load_recipes()
    if df.empty:
        st.warning(
            "No recipe data found. Run the batch pipeline first "
            "(or set `FOODCOM_STAGING_DIR` to your staging directory)."
        )
        return

    # Filter strip
    filter_col1, filter_col2, filter_col3 = st.columns([1.5, 1, 1.5])
    with filter_col1:
        max_cook = st.slider("Max cook time (min)", 5, 180, 60, step=5)
    with filter_col2:
        min_rating = st.select_slider("Min Bayesian rating", options=[1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0], value=3.0)
    with filter_col3:
        tag_counts = Counter(
            tag.strip()
            for tags_str in df["tags"].dropna()
            for tag in str(tags_str).split(",")
            if tag.strip()
        )
        all_tags = [tag for tag, _ in tag_counts.most_common(40)]
        selected_tags = st.multiselect("Cuisine / tags", options=all_tags, default=[])

    filtered = apply_filters(df, search, max_cook, min_rating, selected_tags)
    total = len(filtered)

    # Pagination state
    filter_hash = hash((search, max_cook, min_rating, tuple(sorted(selected_tags))))
    if st.session_state.get("_recipe_filter_hash") != filter_hash:
        st.session_state["_recipe_page"] = 0
        st.session_state["_recipe_filter_hash"] = filter_hash
    page = st.session_state.get("_recipe_page", 0)
    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, n_pages - 1))

    # Result count
    start = page * PAGE_SIZE
    end   = min(start + PAGE_SIZE, total)
    st.caption(f"{total:,} recipes · showing {start + 1}–{end}" if total else "No recipes match your filters.")

    # Recipe cards
    page_df = filtered.iloc[start:end]
    for _, row in page_df.iterrows():
        _render_recipe_card(row)

    # Pagination controls
    pc1, pc2, pc3 = st.columns([1, 2, 1])
    with pc1:
        if page > 0:
            if st.button("← Previous", key="prev_page"):
                st.session_state["_recipe_page"] = page - 1
                st.rerun()
    with pc2:
        st.caption(f"Page {page + 1} of {n_pages}")
    with pc3:
        if page < n_pages - 1:
            if st.button("Next →", key="next_page"):
                st.session_state["_recipe_page"] = page + 1
                st.rerun()
