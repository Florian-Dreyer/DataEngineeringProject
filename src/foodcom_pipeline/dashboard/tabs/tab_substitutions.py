"""Smart Substitutions tab — substitution engine recommendations and swap impact."""

import ast
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st


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
_BAYESIAN_PSEUDO_COUNT = 10.0

_PG_USER = os.getenv("POSTGRES_USER", "user")
_PG_PASS = os.getenv("POSTGRES_PASSWORD", "password")
_PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
_PG_PORT = os.getenv("POSTGRES_PORT", "5432")
_PG_DB = os.getenv("POSTGRES_DB", "foodcom")
_DB_DSN = f"postgresql://{_PG_USER}:{_PG_PASS}@{_PG_HOST}:{_PG_PORT}/{_PG_DB}"


@st.cache_data(ttl=300)
def _load_substitution_pairs() -> pd.DataFrame:
    # Primary source: PostgreSQL serving view
    try:
        import psycopg2

        conn = psycopg2.connect(_DB_DSN, connect_timeout=3)
        try:
            df = pd.read_sql(
                """
                SELECT
                    candidate_ingredient,
                    substitute_ingredient,
                    recommendation_score,
                    rating_delta,
                    sentiment_delta,
                    protein_delta,
                    saturated_fat_delta,
                    sugar_delta,
                    sodium_delta,
                    calories_delta,
                    health_delta
                FROM serving_substitution_recommendations
                ORDER BY recommendation_score DESC NULLS LAST
                """,
                conn,
            )
        finally:
            conn.close()
        if not df.empty:
            return df
    except Exception:
        pass

    # Fallback: staged parquet (for local/dev before DB load has run)
    path = STAGING_DIR / "substitution_engine.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def _load_recipes_for_picker() -> pd.DataFrame:
    try:
        import psycopg2

        conn = psycopg2.connect(_DB_DSN, connect_timeout=3)
        try:
            df = pd.read_sql(
                """
                SELECT recipe_id, name, sentiment_rating, avg_rating, top_ingredients, weighted_review_count
                FROM dim_recipe
                ORDER BY COALESCE(sentiment_rating, avg_rating) DESC NULLS LAST
                """,
                conn,
            )
        finally:
            conn.close()
        if not df.empty:
            return df
    except Exception:
        pass

    recipes_path = STAGING_DIR / "recipes_clean.parquet"
    sentiment_path = STAGING_DIR / "recipe_sentiment_ratings.parquet"
    if not recipes_path.exists():
        return pd.DataFrame()

    recipes = pd.read_parquet(recipes_path, columns=["id", "name", "ingredients_canonical_normalized"])
    recipes = recipes.rename(columns={"id": "recipe_id", "ingredients_canonical_normalized": "top_ingredients"})
    recipes["avg_rating"] = None
    recipes["sentiment_rating"] = None
    recipes["weighted_review_count"] = None

    if sentiment_path.exists():
        sentiment = pd.read_parquet(
            sentiment_path, columns=["recipe_id", "sentiment_rating", "weighted_review_count"]
        )
        recipes = recipes.merge(sentiment, on="recipe_id", how="left", suffixes=("", "_new"))
        recipes["sentiment_rating"] = recipes["sentiment_rating_new"].fillna(recipes["sentiment_rating"])
        recipes["weighted_review_count"] = recipes["weighted_review_count_new"].fillna(
            recipes["weighted_review_count"]
        )
        recipes = recipes.drop(columns=["sentiment_rating_new"], errors="ignore")
        recipes = recipes.drop(columns=["weighted_review_count_new"], errors="ignore")

    return recipes.head(3000)


@st.cache_data(ttl=300)
def _load_global_sentiment_mean() -> float:
    try:
        import psycopg2

        conn = psycopg2.connect(_DB_DSN, connect_timeout=3)
        try:
            row = pd.read_sql(
                "SELECT AVG(sentiment_score) AS global_mean FROM fact_interactions WHERE sentiment_score IS NOT NULL",
                conn,
            )
        finally:
            conn.close()
        if not row.empty and pd.notna(row.iloc[0]["global_mean"]):
            return float(row.iloc[0]["global_mean"])
    except Exception:
        pass

    fallback = STAGING_DIR / "interactions_sentiment.parquet"
    if fallback.exists():
        df = pd.read_parquet(fallback, columns=["sentiment_score"])
        if not df.empty and df["sentiment_score"].notna().any():
            return float(df["sentiment_score"].mean())
    return 0.0


def _to_ingredients_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if "|" in value:
            return [x.strip().lower() for x in value.split("|") if x.strip()]
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip().lower() for x in parsed if str(x).strip()]
        except Exception:
            pass
    if hasattr(value, "__iter__") and not isinstance(value, str):
        return [str(x).strip().lower() for x in list(value) if str(x).strip()]
    return []


def _format_delta(value: float | None, higher_is_better: bool) -> str:
    if value is None or pd.isna(value):
        return "—"
    val = float(value)
    sign = "+" if val >= 0 else ""
    emoji = "✅" if (val >= 0 if higher_is_better else val <= 0) else "⚠️"
    return f"{emoji} {sign}{val:.3f}"


def _to_star_rating(sentiment_like: float | None) -> float | None:
    if sentiment_like is None or pd.isna(sentiment_like):
        return None
    value = float(sentiment_like)
    if -1.0 <= value <= 1.0:
        return max(1.0, min(5.0, 2.0 * value + 3.0))
    return max(0.0, min(5.0, value))


def _buy_link_for_ingredient(ingredient: str) -> str:
    query = quote_plus(ingredient.strip())
    return f"https://www.amazon.com/s?k={query}&i=grocery"


def _recompute_bayesian_sentiment(
    base_sentiment_rating: float | None,
    weighted_review_count: float | None,
    global_mean: float,
    swap_sentiment_deltas: list[float],
    pseudo_count: float = _BAYESIAN_PSEUDO_COUNT,
) -> float | None:
    """
    Recomputes Bayesian sentiment score in real time by adding synthetic swap evidence.
    """
    if base_sentiment_rating is None or pd.isna(base_sentiment_rating):
        return None

    base = float(base_sentiment_rating)
    w = float(weighted_review_count) if weighted_review_count is not None and pd.notna(weighted_review_count) else 0.0
    c = float(global_mean)
    m = float(pseudo_count)

    weighted_sum = base * (w + m) - m * c
    synthetic_values = [max(-1.0, min(1.0, base + float(delta))) for delta in swap_sentiment_deltas]
    synthetic_sum = sum(synthetic_values)
    synthetic_n = float(len(synthetic_values))

    return (weighted_sum + synthetic_sum + m * c) / (w + synthetic_n + m)


def render() -> None:
    st.header("🔄 Smart Substitutions")
    st.markdown(
        """
        <style>
        .subs-hero {
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfeff 100%);
            border: 1px solid #bbf7d0;
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }
        .subs-hero-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #14532d;
            margin-bottom: 2px;
        }
        .subs-hero-sub {
            color: #334155;
            font-size: 0.92rem;
            margin: 0;
        }
        .subs-section-title {
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="subs-hero">
          <div class="subs-hero-title">Smarter ingredient swaps, faster decisions</div>
          <p class="subs-hero-sub">
            Compare each substitute by expected quality lift, nutrition impact, and quick buy options.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    subs = _load_substitution_pairs()
    recipes = _load_recipes_for_picker()
    global_sentiment_mean = _load_global_sentiment_mean()

    if subs.empty:
        st.warning(
            "No substitution engine data found. Run the batch pipeline load step to populate "
            "`fact_substitution_recommendations` (or generate `substitution_engine.parquet`)."
        )
        return
    if recipes.empty:
        st.warning("No recipe data available to map substitutions.")
        return

    recipes = recipes.copy()
    recipes["ingredient_list"] = recipes["top_ingredients"].apply(_to_ingredients_list)
    recipes["display_rating"] = recipes["sentiment_rating"].fillna(recipes["avg_rating"])
    candidate_set = set(subs["candidate_ingredient"])
    recipes["has_substitutions"] = recipes["ingredient_list"].apply(
        lambda ing_list: any(ing in candidate_set for ing in ing_list)
    )

    recipes_with_subs = recipes[recipes["has_substitutions"]].copy()
    if recipes_with_subs.empty:
        st.info(
            "No recipes currently map to substitution candidates. "
            "Try rerunning features/load or selecting a broader recipe pool."
        )
        return

    option_map = {
        f"{row['recipe_id']} — {row['name']}": row["recipe_id"]
        for _, row in recipes_with_subs.iterrows()
        if pd.notna(row.get("recipe_id")) and str(row.get("name", "")).strip()
    }
    if not option_map:
        st.warning("Recipe picker could not be populated from current data.")
        return

    selected_label = st.selectbox("Select recipe", options=list(option_map.keys()))
    selected_recipe_id = option_map[selected_label]
    recipe_row = recipes_with_subs[recipes_with_subs["recipe_id"] == selected_recipe_id].iloc[0]
    recipe_ingredients = set(recipe_row["ingredient_list"])

    recs = subs[subs["candidate_ingredient"].isin(recipe_ingredients)].copy()
    if recs.empty:
        st.info(
            "No substitution candidates in this recipe. Pick a different recipe to see swap cards."
        )
        return

    recs = recs.sort_values("recommendation_score", ascending=False).reset_index(drop=True)

    st.info(
        "Recommendations prioritize underperforming ingredients and rank substitutes by rating lift, "
        "sentiment lift, culinary compatibility, and nutrition delta."
    )

    swap_key = f"_subs_swaps_{int(selected_recipe_id)}"
    if swap_key not in st.session_state:
        st.session_state[swap_key] = {}
    active_swaps: dict = st.session_state[swap_key]

    base_rating = recipe_row.get("display_rating")
    base_rating = float(base_rating) if pd.notna(base_rating) else None
    base_weight = recipe_row.get("weighted_review_count")
    base_weight = float(base_weight) if pd.notna(base_weight) else 0.0
    selected_rows = recs[
        recs.apply(
            lambda r: active_swaps.get(r["candidate_ingredient"]) == r["substitute_ingredient"],
            axis=1,
        )
    ]
    rating_uplift = float(selected_rows["rating_delta"].sum()) if not selected_rows.empty else 0.0
    swap_sentiment_deltas = (
        selected_rows["sentiment_delta"].astype(float).tolist() if not selected_rows.empty else []
    )
    adjusted_sentiment_bayes = _recompute_bayesian_sentiment(
        base_sentiment_rating=base_rating,
        weighted_review_count=base_weight,
        global_mean=global_sentiment_mean,
        swap_sentiment_deltas=swap_sentiment_deltas,
    )
    base_stars = _to_star_rating(base_rating)
    adjusted_stars = _to_star_rating(adjusted_sentiment_bayes)

    c1, c2, c3 = st.columns(3)
    c1.metric("Base Bayesian Sentiment", f"{base_rating:.3f}" if base_rating is not None else "—")
    c2.metric(
        "Estimated Rating Delta",
        f"+{rating_uplift:.2f}" if rating_uplift > 0 else f"{rating_uplift:.2f}",
    )
    c3.metric(
        "Adjusted Bayesian Sentiment",
        f"{adjusted_sentiment_bayes:.3f}" if adjusted_sentiment_bayes is not None else "—",
    )
    c4, c5 = st.columns(2)
    c4.metric("Base Star-Equivalent", f"{base_stars:.2f}★" if base_stars is not None else "—")
    c5.metric(
        "Adjusted Star-Equivalent",
        f"{adjusted_stars:.2f}★" if adjusted_stars is not None else "—",
    )
    st.caption(
        f"Bayesian recompute uses m={_BAYESIAN_PSEUDO_COUNT:.0f}, "
        f"global sentiment mean C={global_sentiment_mean:.3f}, "
        f"weighted review depth={base_weight:.2f}."
    )

    rendered_candidates: set[str] = set()
    for _, row in recs.iterrows():
        candidate = row["candidate_ingredient"]
        if candidate in rendered_candidates:
            continue
        rendered_candidates.add(candidate)

        candidate_rows = recs[recs["candidate_ingredient"] == candidate]
        st.markdown(f"### Replace `{candidate}`")
        for _, cand_row in candidate_rows.iterrows():
            sub = cand_row["substitute_ingredient"]
            with st.container(border=True):
                head_left, head_right = st.columns([4, 1])
                with head_left:
                    st.markdown(
                        f"**Suggested substitute:** `{sub}`  \n"
                        f"Recommendation score: `{cand_row['recommendation_score']:.3f}`"
                    )
                with head_right:
                    is_active = active_swaps.get(candidate) == sub
                    if st.button(
                        "Applied ✅" if is_active else "Apply swap",
                        key=f"swap_{selected_recipe_id}_{candidate}_{sub}",
                        type="primary" if is_active else "secondary",
                        use_container_width=True,
                    ):
                        if is_active:
                            active_swaps.pop(candidate, None)
                        else:
                            active_swaps[candidate] = sub
                        st.session_state[swap_key] = active_swaps
                        st.rerun()

                stat_a, stat_b, stat_c = st.columns(3)
                stat_a.metric("Rating delta", _format_delta(cand_row.get("rating_delta"), True))
                stat_b.metric("Sentiment delta", _format_delta(cand_row.get("sentiment_delta"), True))
                stat_c.metric("Health delta", _format_delta(cand_row.get("health_delta"), True))

                n1, n2, n3, n4 = st.columns(4)
                n1.metric("Protein", _format_delta(cand_row.get("protein_delta"), True))
                n2.metric("Sat fat", _format_delta(cand_row.get("saturated_fat_delta"), False))
                n3.metric("Sugar", _format_delta(cand_row.get("sugar_delta"), False))
                n4.metric("Sodium", _format_delta(cand_row.get("sodium_delta"), False))

                st.markdown('<div class="subs-section-title">Sponsored picks</div>', unsafe_allow_html=True)
                promo_cols = st.columns(2)
                with promo_cols[0]:
                    with st.container(border=True):
                        st.markdown(f"#### 🛒 Buy `{sub}`")
                        st.caption("Fast checkout for this recommended substitute.")
                        st.link_button(
                            "Shop substitute",
                            _buy_link_for_ingredient(sub),
                            use_container_width=True,
                            help="Opens an external page to buy this substitute ingredient.",
                        )
                with promo_cols[1]:
                    with st.container(border=True):
                        st.markdown(f"#### 🔍 Explore `{candidate}`")
                        st.caption("Compare brands and alternatives for the original ingredient.")
                        st.link_button(
                            "View alternatives",
                            _buy_link_for_ingredient(candidate),
                            use_container_width=True,
                            help="Opens an external page to buy the original ingredient.",
                        )
            st.markdown("")
        st.divider()
