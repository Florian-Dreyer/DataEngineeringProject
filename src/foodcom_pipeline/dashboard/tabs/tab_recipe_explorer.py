"""Recipe Explorer tab — hero search, filters, recipe cards, Shop buttons, pagination."""

import ast
import html
import os
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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

_PG_USER = os.getenv("POSTGRES_USER", "user")
_PG_PASS = os.getenv("POSTGRES_PASSWORD", "password")
_PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
_PG_PORT = os.getenv("POSTGRES_PORT", "5432")
_PG_DB   = os.getenv("POSTGRES_DB", "foodcom")
_DB_DSN  = f"postgresql://{_PG_USER}:{_PG_PASS}@{_PG_HOST}:{_PG_PORT}/{_PG_DB}"

PAGE_SIZE = 10

# Populated by load_recipes(); always a valid Series even before first load.
_RECIPE_MEDIAN: pd.Series = pd.Series(
    {n: 0.0 for n in ["protein", "fat", "carbs", "sugar", "sodium"]}
)

# Nutrition axes shown on the radar (all Food.com %DV columns)
_RADAR_NUTRIENTS = ["protein", "fat", "carbs", "sugar", "sodium"]
_RADAR_LABELS    = ["Protein %DV", "Fat %DV", "Carbs %DV", "Sugar %DV", "Sodium %DV"]

_TAG_BLOCKLIST = frozenset({
    "time-to-make", "course", "main-ingredient", "preparation",
    "occasion", "equipment", "dietary", "technique",
    "number-of-servings", "meat", "vegetables",
})


def _is_useful_tag(tag: str) -> bool:
    """Return True if a tag is meaningful for the cuisine/style filter."""
    if tag in _TAG_BLOCKLIST:
        return False
    if tag.startswith("for-"):
        return False
    if "-servings" in tag:
        return False
    if tag.isdigit():
        return False
    return True


# ---------------------------------------------------------------------------
# Pure helpers (independently testable — no Streamlit dependency)
# ---------------------------------------------------------------------------

def _restore_apostrophes(name: str) -> str:
    """Restore possessive apostrophes stripped by the Kaggle dataset (e.g. "mom s" → "mom's")."""
    # Match 2+ alpha chars followed by standalone " s " or " s" at end of string
    name = re.sub(r"\b([a-zA-Z]{2,}) s ([a-zA-Z])", r"\1's \2", name)
    name = re.sub(r"\b([a-zA-Z]{2,}) s$", r"\1's", name)
    return name


def build_amazon_url(_ingredients: list[str]) -> str:
    return "https://www.amazon.com/fresh"


def build_instacart_url(_recipe_name: str) -> str:
    return "https://www.instacart.com"


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


def _compute_affiliate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with affiliate scoring columns added.

    Only the top 1 000 recipes by review_count receive an affiliate_score;
    all others get NaN. All other columns (cart_ready, basket_value_est,
    review_velocity, revenue_proj_monthly) are computed for every row.
    """
    df = df.copy()
    if "review_count" not in df.columns:
        df["review_count"] = float("nan")

    # --- cart_ready (all rows) ---
    ic = df["ingredient_count"].fillna(0)
    df["cart_ready"] = pd.array(
        [bool(v) for v in ((ic >= 7) & (ic <= 11))], dtype=object
    )

    # --- basket_value_est and revenue_proj_monthly (all rows) ---
    df["basket_value_est"] = df["ingredient_count"].fillna(0) * 3.50
    df["revenue_proj_monthly"] = 10_000 * 0.02 * df["basket_value_est"] * 0.04

    # --- review_velocity (all rows) ---
    df["review_velocity"] = df["review_count"] / 73.0

    # --- affiliate_score (top-1000 by review_count only) ---
    df["affiliate_score"] = float("nan")

    top1k = df.nlargest(1000, "review_count").index

    # Median-fill cook time for normalisation; restore NaN afterwards
    cook = df.loc[top1k, "avg_cook_minutes"]
    cook_filled = cook.fillna(cook.median())
    max_cook = cook_filled.max()
    if max_cook == 0:
        max_cook = 1.0
    prep_norm = (cook_filled / max_cook).clip(lower=0.01)

    rc = df.loc[top1k, "review_count"].fillna(0)
    max_rc = rc.max()
    if max_rc == 0:
        max_rc = 1.0

    sweet = ((df.loc[top1k, "ingredient_count"].fillna(0) >= 7) &
             (df.loc[top1k, "ingredient_count"].fillna(0) <= 11)).astype(float)

    df.loc[top1k, "affiliate_score"] = (
        (rc / max_rc) * 0.40
        + (1.0 / prep_norm) * 0.30
        + sweet * 0.30
    ).clip(upper=1.0)

    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_recipes() -> pd.DataFrame:
    """
    Load recipe data. Tries PostgreSQL dim_recipe first; falls back to parquet.
    Returns a DataFrame with a normalised schema. Returns empty DataFrame on failure.
    """
    global _RECIPE_MEDIAN
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
            if "weighted_review_count" in df.columns:
                df["review_count"] = df["weighted_review_count"]
            elif "review_count" not in df.columns:
                df["review_count"] = float("nan")
            df = _compute_affiliate_columns(df)
            _RECIPE_MEDIAN = df[["protein", "fat", "carbs", "sugar", "sodium"]].median()
            return df
    except Exception:
        pass

    # Fallback: parquet staging files
    recipes_path   = STAGING_DIR / "recipes_clean.parquet"
    sentiment_path = STAGING_DIR / "recipe_sentiment_ratings.parquet"

    if not recipes_path.exists():
        return pd.DataFrame()

    import pyarrow.parquet as pq
    schema_names = pq.read_schema(recipes_path).names

    # Food.com PDV columns (always present; USDA-enriched columns are null until USDA pipeline runs)
    _pdv_map = {
        "calories_pdv":     "calories",
        "protein_pdv":      "protein",
        "total_fat_pdv":    "fat",
        "sugar_pdv":        "sugar",
        "sodium_pdv":       "sodium",
        "carbs_pdv":        "carbs",
        "saturated_fat_pdv": "saturated_fat",
    }
    keep = ["id", "name", "minutes", "tags", "ingredient_count",
            "ingredients_parsed"] + list(_pdv_map.keys())
    available = [c for c in keep if c in schema_names]

    df = pd.read_parquet(recipes_path, columns=available).rename(columns={
        "id": "recipe_id",
        "minutes": "avg_cook_minutes",
        **_pdv_map,
    })

    # Build top_ingredients from the parsed ingredient list column
    if "ingredients_parsed" in df.columns:
        def _to_pipe(x) -> str:
            # numpy.ndarray, list, tuple — anything iterable but not a bare string
            if hasattr(x, "__iter__") and not isinstance(x, str):
                return "|".join(str(i) for i in list(x)[:10])
            # Fallback: string that looks like a Python list repr
            try:
                parsed = ast.literal_eval(str(x))
                if isinstance(parsed, (list, tuple)):
                    return "|".join(str(i) for i in list(parsed)[:10])
            except Exception:
                pass
            return ""
        df["top_ingredients"] = df["ingredients_parsed"].apply(_to_pipe)
    else:
        df["top_ingredients"] = ""

    df["avg_rating"] = float("nan")

    if sentiment_path.exists():
        sent = pd.read_parquet(sentiment_path, columns=["recipe_id", "sentiment_rating"])
        df = df.merge(sent, on="recipe_id", how="left")
        # VADER compound scores are 0–1; scale to 1–5 to match the UI slider
        df["sentiment_rating"] = df["sentiment_rating"] * 5
    else:
        df["sentiment_rating"] = float("nan")

    df["display_rating"] = df["sentiment_rating"].fillna(df["avg_rating"])
    df = df.sort_values("display_rating", ascending=False, na_position="last")

    # Ensure all expected nutrition columns exist (fill missing with NaN)
    for col in ["calories", "protein", "fat", "sugar", "sodium", "carbs", "saturated_fat"]:
        if col not in df.columns:
            df[col] = float("nan")

    # Alias weighted_review_count → review_count for affiliate scoring
    if "weighted_review_count" in df.columns:
        df["review_count"] = df["weighted_review_count"]
    elif "review_count" not in df.columns:
        df["review_count"] = float("nan")

    df = _compute_affiliate_columns(df)

    _RECIPE_MEDIAN = df[["protein", "fat", "carbs", "sugar", "sodium"]].median()

    return df


@st.cache_data(ttl=300, max_entries=100)
def load_recipe_detail(recipe_id) -> dict:
    """Load steps for a single recipe from parquet. Returns {"steps": [...]}."""
    recipes_path = STAGING_DIR / "recipes_clean.parquet"
    if not recipes_path.exists():
        return {"steps": []}
    try:
        import pyarrow.parquet as pq
        schema_names = pq.read_schema(recipes_path).names
        if "steps" not in schema_names:
            return {"steps": []}
        df = pd.read_parquet(recipes_path, columns=["id", "steps"])
        match = df[df["id"] == recipe_id]
        if match.empty:
            return {"steps": []}
        raw = match.iloc[0]["steps"]
        if hasattr(raw, "__iter__") and not isinstance(raw, str):
            return {"steps": [str(s) for s in list(raw)]}
        try:
            parsed = ast.literal_eval(str(raw))
            if isinstance(parsed, list):
                return {"steps": [str(s) for s in parsed]}
        except Exception:
            pass
        return {"steps": []}
    except Exception:
        return {"steps": []}


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def _select_recipe(row: dict) -> None:
    """Switch to detail mode for the given recipe row dict."""
    st.session_state["_selected_recipe"] = row
    st.rerun()


def _back_to_list() -> None:
    """Return to list mode."""
    st.session_state["_selected_recipe"] = None
    st.rerun()


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
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=9)),
        ),
        showlegend=False,
        height=240,
        margin=dict(t=20, b=20, l=50, r=50),
    )
    return fig


def _render_compact_card(row: pd.Series, card_index: int = 0) -> None:
    """Full recipe card — clickable name navigates to detail view."""
    name          = str(row.get("name", "Unknown"))
    cook_min      = row.get("avg_cook_minutes")
    n_ingredients = row.get("ingredient_count")
    display_rating = row.get("display_rating")
    avg_rating    = row.get("avg_rating")
    top_ingr_str  = str(row.get("top_ingredients") or "")
    ingredients   = [i.strip() for i in top_ingr_str.split("|") if i.strip()]
    trend_index   = row.get("trend_index")
    recipe_id     = row.get("recipe_id", card_index)

    amazon_url    = build_amazon_url(ingredients)
    instacart_url = build_instacart_url(name)

    with st.container(border=True):
        col_info, col_shop = st.columns([3, 1])

        with col_info:
            st.markdown(f"### {html.escape(_restore_apostrophes(html.unescape(name)))}")

            subtitle_parts = []
            if display_rating is not None and not pd.isna(display_rating):
                subtitle_parts.append(f"⭐ {display_rating:.1f}")
            if cook_min is not None and not pd.isna(cook_min):
                subtitle_parts.append(f"⏱ {int(cook_min)} min")
            if n_ingredients is not None and not pd.isna(n_ingredients):
                subtitle_parts.append(f"{int(n_ingredients)} ingredients")
            if subtitle_parts:
                st.caption(" · ".join(subtitle_parts))
            if st.button("View full recipe →", key=f"select_{recipe_id}_{card_index}", type="primary"):
                _select_recipe(row.to_dict())

            badge_parts = []
            if avg_rating is not None and not pd.isna(avg_rating):
                badge_parts.append(
                    f'<span style="background:#f3f4f6;color:#374151;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;">{avg_rating:.1f} avg</span>'
                )
            if (display_rating is not None and not pd.isna(display_rating)
                    and avg_rating is not None and not pd.isna(avg_rating)):
                delta = display_rating - avg_rating
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

        with st.expander("📊 Nutrition & Ingredients", expanded=False):
            col_radar, col_bars, col_pills = st.columns([1.2, 1.5, 1])

            with col_radar:
                st.plotly_chart(
                    _nutrition_radar(row),
                    use_container_width=True,
                    key=f"radar_{recipe_id}_{card_index}",
                )

            with col_bars:
                st.markdown("**Nutrition per serving**")
                any_nutrition = False
                for nutrient, label in zip(_RADAR_NUTRIENTS, _RADAR_LABELS):
                    val = row.get(nutrient)
                    if val is None or pd.isna(val):
                        continue
                    any_nutrition = True
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
                if not any_nutrition:
                    st.caption("Nutrition data not available for this recipe.")

            with col_pills:
                st.markdown("**Ingredients**")
                if ingredients:
                    pills_html = "".join(
                        f'<span style="display:inline-block;background:#f3f4f6;border-radius:12px;'
                        f'padding:3px 10px;font-size:11px;color:#6b7280;margin:2px;">'
                        f'{html.escape(ing)}</span>'
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

            cal_val = row.get("calories")
            if cal_val is not None and not pd.isna(cal_val):
                st.caption(f"Calories: {cal_val:.0f} kcal per serving")


def _render_detail_view(row: dict) -> None:
    """Full-page detail view for a single recipe, including steps and nutrition."""
    recipe_id     = row.get("recipe_id")
    name          = str(row.get("name", "Unknown"))
    cook_min      = row.get("avg_cook_minutes")
    n_ingredients = row.get("ingredient_count")
    display_rating = row.get("display_rating")
    avg_rating    = row.get("avg_rating")
    top_ingr_str  = str(row.get("top_ingredients") or "")
    ingredients   = [i.strip() for i in top_ingr_str.split("|") if i.strip()]

    amazon_url    = build_amazon_url(ingredients)
    instacart_url = build_instacart_url(name)

    # --- Back button + shop buttons row ---
    col_back, col_shops = st.columns([3, 2])
    with col_back:
        if st.button("← Back to results", key="back_to_list", type="primary"):
            _back_to_list()
    with col_shops:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(
                f'<a href="{amazon_url}" target="_blank" style="display:block;background:#10b981;'
                f'color:white;text-align:center;border-radius:6px;padding:9px 12px;'
                f'font-weight:700;font-size:13px;text-decoration:none;">'
                f'🛒 Shop on Amazon Fresh</a>',
                unsafe_allow_html=True,
            )
        with sc2:
            st.markdown(
                f'<a href="{instacart_url}" target="_blank" style="display:block;background:white;'
                f'color:#374151;text-align:center;border-radius:6px;padding:8px 12px;'
                f'font-size:13px;text-decoration:none;border:1px solid #e5e7eb;">'
                f'🥬 Instacart</a>',
                unsafe_allow_html=True,
            )

    # --- Recipe title + metadata ---
    st.markdown(f"# {html.escape(_restore_apostrophes(html.unescape(name)))}")

    meta_parts = []
    if display_rating is not None and not pd.isna(display_rating):
        meta_parts.append(f"⭐ {display_rating:.1f}")
    if cook_min is not None and not pd.isna(cook_min):
        meta_parts.append(f"⏱ {int(cook_min)} min")
    if n_ingredients is not None and not pd.isna(n_ingredients):
        meta_parts.append(f"{int(n_ingredients)} ingredients")

    if (avg_rating is not None and not pd.isna(avg_rating)
            and display_rating is not None and not pd.isna(display_rating)
            and abs(display_rating - avg_rating) > 0.2):
        meta_parts.append(
            f'<span style="background:#f3f4f6;color:#374151;border-radius:4px;'
            f'padding:2px 8px;font-size:13px;">{avg_rating:.1f} avg</span>'
        )

    if meta_parts:
        st.markdown(" &nbsp;·&nbsp; ".join(meta_parts), unsafe_allow_html=True)

    st.divider()

    # --- Steps ---
    st.markdown("### Steps")
    detail = load_recipe_detail(recipe_id)
    steps = detail.get("steps", [])
    if steps:
        steps_html = "".join(
            f'<div style="display:flex;align-items:flex-start;margin-bottom:14px;">'
            f'<span style="background:#10b981;color:white;border-radius:50%;'
            f'min-width:26px;height:26px;display:flex;align-items:center;'
            f'justify-content:center;font-size:12px;font-weight:700;'
            f'flex-shrink:0;margin-right:14px;">{i}</span>'
            f'<span style="font-size:14px;line-height:1.6;color:#111827;">'
            f'{html.escape(str(step))}</span>'
            f'</div>'
            for i, step in enumerate(steps, 1)
        )
        st.markdown(steps_html, unsafe_allow_html=True)
    else:
        st.info("Recipe steps not available for this recipe.")

    st.divider()

    # --- Nutrition & Ingredients ---
    st.markdown("### Nutrition & Ingredients")
    row_series = pd.Series(row)
    col_radar, col_bars, col_pills = st.columns([1.2, 1.5, 1])

    with col_radar:
        st.plotly_chart(
            _nutrition_radar(row_series),
            use_container_width=True,
            key=f"detail_radar_{recipe_id}",
        )

    with col_bars:
        st.markdown("**Nutrition per serving**")
        any_nutrition = False
        for nutrient, label in zip(_RADAR_NUTRIENTS, _RADAR_LABELS):
            val = row.get(nutrient)
            if val is None or pd.isna(val):
                continue
            any_nutrition = True
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
        if not any_nutrition:
            st.caption("Nutrition data not available for this recipe.")

    with col_pills:
        st.markdown("**Ingredients**")
        if ingredients:
            pills_html = "".join(
                f'<span style="display:inline-block;background:#f3f4f6;border-radius:12px;'
                f'padding:3px 10px;font-size:11px;color:#6b7280;margin:2px;">'
                f'{html.escape(ing)}</span>'
                for ing in ingredients
            )
            st.markdown(pills_html, unsafe_allow_html=True)
        else:
            st.caption("—")

    cal_val = row.get("calories")
    if cal_val is not None and not pd.isna(cal_val):
        st.caption(f"Calories: {cal_val:.0f} kcal per serving")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def _render_list_mode(df: pd.DataFrame) -> None:
    """Render hero + search + paginated recipe cards."""

    # --- Hero ---
    st.markdown(
        '<div style="background:linear-gradient(135deg,#10b981,#059669);border-radius:10px;'
        'padding:24px 28px;margin-bottom:12px;">'
        '<h1 style="color:white;margin:0;font-size:26px;">🍳 Recipe Explorer</h1>'
        '<p style="color:#d1fae5;margin:6px 0 0;">Browse recipes — '
        'see ratings, nutrition, and shop ingredients in one click.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Search ---
    search = st.text_input(
        "",
        placeholder="🔍 Search recipes by name or ingredient",
        label_visibility="collapsed"
    )
    if search:
        prev_search = st.session_state.get("_recipe_search", "")
        if search != prev_search:
            st.session_state["_recipe_page"] = 0
            st.session_state["_recipe_search"] = search
        df = df[
            df["name"].str.contains(search, case=False, na=False)
            | df["top_ingredients"].str.contains(search, case=False, na=False)
        ]
    else:
        st.session_state.pop("_recipe_search", None)

    total = len(df)
    page = st.session_state.get("_recipe_page", 0)
    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, n_pages - 1))

    start = page * PAGE_SIZE
    end   = min(start + PAGE_SIZE, total)
    st.caption(f"{total:,} recipes · showing {start + 1}–{end}")

    # --- Cards ---
    page_df = df.iloc[start:end]
    for idx, (_, row) in enumerate(page_df.iterrows()):
        _render_compact_card(row, card_index=start + idx)

    # --- Pagination controls ---
    _, pc1, pc2, pc3, _ = st.columns([3, 1, 1, 1, 3])
    with pc1:
        if page > 0:
            if st.button("← Previous", key="prev_page", type="primary", use_container_width=True):
                st.session_state["_recipe_page"] = page - 1
                st.rerun()
        else:
            st.empty()
    with pc2:
        st.markdown(
            f'<div style="text-align:center;padding-top:6px;color:#6b7280;font-size:13px;">'
            f'Page {page + 1} of {n_pages}</div>',
            unsafe_allow_html=True,
        )
    with pc3:
        if page < n_pages - 1:
            if st.button("Next →", key="next_page", type="primary", use_container_width=True):
                st.session_state["_recipe_page"] = page + 1
                st.rerun()
        else:
            st.empty()


def render() -> None:
    df = load_recipes()
    if df.empty:
        st.warning(
            "No recipe data found. Run the batch pipeline first "
            "(or set `FOODCOM_STAGING_DIR` to your staging directory)."
        )
        return

    selected = st.session_state.get("_selected_recipe")
    if selected is not None:
        _render_detail_view(selected)
    else:
        _render_list_mode(df)
