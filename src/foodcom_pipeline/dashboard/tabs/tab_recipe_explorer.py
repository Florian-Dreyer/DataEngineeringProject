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

    # --- affiliate_score (all rows) ---
    # Weights: 40% popularity, 30% quickness (shorter = better), 30% ingredient sweet spot
    cook = df["avg_cook_minutes"]
    cook_filled = cook.fillna(cook.median())
    max_cook = cook_filled.max()
    if pd.isna(max_cook) or max_cook == 0:
        max_cook = 1.0
    prep_norm = (cook_filled / max_cook).clip(lower=0.0, upper=1.0)

    rc = df["review_count"].fillna(0)
    max_rc = rc.max()
    if max_rc == 0:
        max_rc = 1.0

    sweet = ((df["ingredient_count"].fillna(0) >= 7) &
             (df["ingredient_count"].fillna(0) <= 11)).astype(float)

    df["affiliate_score"] = (
        (rc / max_rc) * 0.40
        + (1.0 - prep_norm) * 0.30
        + sweet * 0.30
    )

    return df


def _affiliate_gauge(score: float) -> go.Figure:
    """Plotly indicator gauge for affiliate_score (0–1).

    Green ≥ 0.6, amber 0.3–0.59, red < 0.3.
    """
    if score >= 0.6:
        color = "#10b981"
    elif score >= 0.3:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"valueformat": ".2f", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 1], "tickfont": {"size": 9}},
            "bar": {"color": color},
            "bgcolor": "#f3f4f6",
            "borderwidth": 0,
        },
        title={"text": "Affiliate Score", "font": {"size": 12}},
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20))
    return fig


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
            _RECIPE_MEDIAN = df[["protein", "fat", "carbs", "sugar", "sodium"]].median().fillna(0.0)
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
                return "|".join(str(i) for i in list(x))
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

    # Compute raw review_count from interactions
    interactions_path = STAGING_DIR / "interactions_clean.parquet"
    if interactions_path.exists():
        counts = (
            pd.read_parquet(interactions_path, columns=["recipe_id"])
            .groupby("recipe_id")
            .size()
            .rename("review_count")
        )
        df = df.merge(counts, on="recipe_id", how="left")
        df["review_count"] = df["review_count"].fillna(0).astype(int)
    else:
        df["review_count"] = 0

    df = _compute_affiliate_columns(df)

    _RECIPE_MEDIAN = df[["protein", "fat", "carbs", "sugar", "sodium"]].median().fillna(0.0)

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

def _nutrition_radar(row: pd.Series, median_row: pd.Series | None = None) -> go.Figure:
    """Plotly polar chart for 5 nutrition axes (%DV values).

    If median_row is supplied, overlays a grey dashed trace for the category median.
    """
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

    if median_row is not None:
        med_values = [_safe_float(median_row.get(n)) for n in _RADAR_NUTRIENTS]
        fig.add_trace(go.Scatterpolar(
            r=med_values + [med_values[0]],
            theta=_RADAR_LABELS + [_RADAR_LABELS[0]],
            fill="toself",
            fillcolor="rgba(156,163,175,0.10)",
            line=dict(color="#9ca3af", width=1, dash="dash"),
            name="Category median",
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

            # --- Affiliate badges ---
            aff_score = row.get("affiliate_score")
            cart_rdy  = row.get("cart_ready")
            basket    = row.get("basket_value_est")
            rev_proj  = row.get("revenue_proj_monthly")

            aff_parts = []
            if aff_score is not None and not pd.isna(aff_score):
                aff_parts.append(
                    f'<span style="background:#fef3c7;color:#92400e;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;">🏷️ Affiliate Score: {aff_score:.2f}</span>'
                )
            if cart_rdy is True:
                aff_parts.append(
                    f'<span style="background:#d1fae5;color:#065f46;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;">✅ Cart-Ready</span>'
                )
            if aff_parts:
                st.markdown(" &nbsp; ".join(aff_parts), unsafe_allow_html=True)
            if basket is not None and not pd.isna(basket):
                st.caption(f"Est. basket value: ${basket:.2f}")
            if rev_proj is not None and not pd.isna(rev_proj):
                st.caption(f"Est. monthly revenue: ${rev_proj:.2f}")

        with col_shop:
            st.markdown(
                f'<a href="{instacart_url}" target="_blank" style="display:block;background:white;'
                f'color:#374151;text-align:center;border-radius:6px;padding:8px 12px;'
                f'font-size:13px;text-decoration:none;border:1px solid #e5e7eb;margin-bottom:6px;">'
                f'🥬 Instacart</a>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<a href="{amazon_url}" target="_blank" style="display:block;background:#10b981;'
                f'color:white;text-align:center;border-radius:6px;padding:9px 12px;'
                f'font-weight:700;font-size:13px;text-decoration:none;">'
                f'🛒 Shop on Amazon Fresh</a>',
                unsafe_allow_html=True,
            )

        with st.expander("📊 Nutrition & Ingredients", expanded=False):
            col_radar, col_bars, col_pills = st.columns([1.2, 1.5, 1])

            with col_radar:
                st.plotly_chart(
                    _nutrition_radar(row, median_row=None),
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
    col_back, col_shops = st.columns([3, 1])
    with col_back:
        if st.button("← Back to results", key="back_to_list", type="primary"):
            _back_to_list()
    with col_shops:
        st.markdown(
            f'<a href="{instacart_url}" target="_blank" style="display:block;background:white;'
            f'color:#374151;text-align:center;border-radius:6px;padding:8px 12px;'
            f'font-size:13px;text-decoration:none;border:1px solid #e5e7eb;margin-bottom:6px;">'
            f'🥬 Instacart</a>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<a href="{amazon_url}" target="_blank" style="display:block;background:#10b981;'
            f'color:white;text-align:center;border-radius:6px;padding:9px 12px;'
            f'font-weight:700;font-size:13px;text-decoration:none;">'
            f'🛒 Shop on Amazon Fresh</a>',
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

    # --- Affiliate Insights ---
    st.markdown("### 💰 Affiliate Insights")
    aff_score = row.get("affiliate_score")
    if aff_score is not None and not pd.isna(aff_score):
        ai1, ai2, ai3 = st.columns([1.2, 1.5, 1])

        with ai1:
            st.plotly_chart(
                _affiliate_gauge(float(aff_score)),
                use_container_width=True,
                key=f"affiliate_gauge_{recipe_id}",
            )

        with ai2:
            cart_rdy   = row.get("cart_ready")
            basket_est = row.get("basket_value_est")
            rev_proj   = row.get("revenue_proj_monthly")
            review_cnt = row.get("review_count")
            ic         = row.get("ingredient_count")

            basket  = basket_est
            rev     = rev_proj

            sweet      = "Yes" if (ic is not None and not pd.isna(ic) and 7 <= int(ic) <= 11) else "No"
            cart_str   = "Yes" if cart_rdy is True else "No"
            basket_str = f"${basket:.2f} (est.)" if basket is not None and not pd.isna(basket) else "—"
            rev_str    = f"${rev:.2f}" if rev is not None and not pd.isna(rev) else "—"
            rc_str     = f"{int(review_cnt):,}" if review_cnt is not None and not pd.isna(review_cnt) else "—"
            st.markdown(
                f"| Metric | Value |\n"
                f"|---|---|\n"
                f"| Review Count | {rc_str} |\n"
                f"| Cart Ready | {cart_str} |\n"
                f"| Basket Value | {basket_str} |\n"
                f"| Est. Monthly Revenue | {rev_str} |\n"
                f"| Ingredient Sweet Spot (7–11) | {sweet} |"
            )

        with ai3:
            basket_num = f"{basket:.2f}" if basket is not None and not pd.isna(basket) else "unknown"
            rev_num    = f"{rev:.2f}" if rev is not None and not pd.isna(rev) else "unknown"
            st.info(
                f"This recipe scores {aff_score:.2f} for affiliate potential. "
                f"With a basket of \\${basket_num} and 2% conversion at "
                f"10,000 monthly views, it could generate ~\\${rev_num}/month in commission."
            )
    else:
        st.caption("Affiliate data not available for this recipe.")

    st.divider()

    # --- Nutrition & Ingredients ---

    st.markdown("### Nutrition & Ingredients")
    row_series = pd.Series(row)
    col_radar, col_bars, col_pills = st.columns([1.2, 1.5, 1])

    with col_radar:
        st.plotly_chart(
            _nutrition_radar(row_series, median_row=None),
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
        '<p style="color:#d1fae5;margin:6px 0 0;">Discover high-potential recipes — '
        'explore affiliate scores, nutrition, basket value, and shop ingredients in one click.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Top 20 Affiliate Opportunities ---
    with st.expander("📈 Top 20 Affiliate Opportunities"):
        top20 = (
            df.dropna(subset=["affiliate_score"])
            .nlargest(20, "affiliate_score")
            [["name", "affiliate_score", "basket_value_est",
              "revenue_proj_monthly", "cart_ready",
              "avg_cook_minutes", "ingredient_count"]]
            .copy()
        )
        if not top20.empty:
            top20["name"] = top20["name"].apply(
                lambda n: _restore_apostrophes(html.unescape(str(n)))
            )
        if top20.empty:
            st.caption("No affiliate data yet — run the pipeline first.")
        else:
            st.dataframe(
                top20,
                use_container_width=True,
                column_config={
                    "name":                 st.column_config.TextColumn("Recipe"),
                    "affiliate_score":      st.column_config.NumberColumn("Affiliate Score", format="%.2f"),
                    "basket_value_est":     st.column_config.NumberColumn("Est. Basket", format="$%.2f"),
                    "revenue_proj_monthly": st.column_config.NumberColumn("Est. Monthly Rev.", format="$%.2f"),
                    "cart_ready":           st.column_config.CheckboxColumn("Cart Ready"),
                    "avg_cook_minutes":     st.column_config.NumberColumn("Cook (min)", format="%d"),
                    "ingredient_count":     st.column_config.NumberColumn("Ingredients", format="%d"),
                },
                hide_index=True,
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

    # --- Sidebar affiliate filters ---
    with st.sidebar:
        st.markdown(
            '<div style="background:linear-gradient(135deg,#10b981,#059669);'
            'border-radius:10px;padding:16px 18px;margin-bottom:4px;">'
            '<p style="color:white;font-weight:700;font-size:15px;margin:0;">🏷️ Affiliate Filters</p>'
            '<p style="color:#d1fae5;font-size:12px;margin:4px 0 0;">'
            'Narrow recipes by affiliate potential</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        st.markdown("**🛒 Availability**")
        cart_only = st.toggle("Cart-Ready only", key="aff_cart_only")
        st.caption(
            "Recipes with 7–11 ingredients hit the sweet spot for grocery cart conversion — "
            "enough variety to fill a basket, manageable enough not to overwhelm a shopper."
        )

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        st.markdown("**📊 Affiliate Score Range**")
        aff_range = st.slider(
            "Score range",
            min_value=0.0, max_value=1.0, value=(0.0, 1.0),
            step=0.05, key="aff_score_range",
            label_visibility="collapsed",
        )
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;margin-top:-8px;'>"
            f"<span style='font-size:12px;color:#6b7280;'>Min: <b>{aff_range[0]:.2f}</b></span>"
            f"<span style='font-size:12px;color:#6b7280;'>Max: <b>{aff_range[1]:.2f}</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.caption(
            "Affiliate score combines review popularity (40%), "
            "recipe simplicity (30%), and ingredient sweet spot (30%)."
        )

    if cart_only:
        df = df[df["cart_ready"] == True]  # noqa: E712
    aff_lo, aff_hi = aff_range
    df = df[
        (df["affiliate_score"].fillna(0) >= aff_lo) &
        (df["affiliate_score"].fillna(0) <= aff_hi)
    ]

    df = df.sort_values("affiliate_score", ascending=False, na_position="last")

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
