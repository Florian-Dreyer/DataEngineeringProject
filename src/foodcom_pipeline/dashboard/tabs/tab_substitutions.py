"""Smart Substitutions tab — substitution engine recommendations and swap impact."""

import ast
import html
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st


def _get_fragment_decorator():
    decorator = getattr(st, "fragment", None)
    if decorator is not None:
        return decorator
    decorator = getattr(st, "experimental_fragment", None)
    if decorator is not None:
        return decorator
    return lambda fn: fn


_fragment = _get_fragment_decorator()


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
_RECIPE_PICKER_INITIAL_OPTIONS = 120
_RECIPE_PICKER_BATCH_SIZE = 250
_RECIPE_PICKER_MAX_INITIAL_BATCHES = 4

_RECIPE_PICKER_STATE_KEY = "_subs_recipe_picker_rows"
_RECIPE_PICKER_OFFSET_KEY = "_subs_recipe_picker_next_offset"
_RECIPE_PICKER_EXHAUSTED_KEY = "_subs_recipe_picker_exhausted"
_RECIPE_PICKER_CANDIDATE_COUNT_KEY = "_subs_recipe_picker_candidate_count"
_RECIPE_PICKER_LAST_SELECTION_KEY = "_subs_recipe_picker_last_selection"
_RECIPE_PICKER_LOAD_MORE_SENTINEL = "Load more recipes..."

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
                    substitute_similarity,
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
def _load_recipes_for_picker(limit: int = 3000, offset: int = 0) -> pd.DataFrame:
    try:
        import psycopg2

        conn = psycopg2.connect(_DB_DSN, connect_timeout=3)
        try:
            df = pd.read_sql(
                """
                SELECT
                    recipe_id,
                    name,
                    sentiment_rating,
                    avg_rating,
                    top_ingredients,
                    weighted_review_count,
                    protein,
                    saturated_fat,
                    sugar,
                    sodium,
                    calories
                FROM dim_recipe
                
                LIMIT %s
                OFFSET %s
                """,
                conn,
                params=(int(limit), int(offset)),
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
    recipes["protein"] = None
    recipes["saturated_fat"] = None
    recipes["sugar"] = None
    recipes["sodium"] = None
    recipes["calories"] = None

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

    recipes = recipes.sort_values(
        by=["sentiment_rating", "avg_rating"],
        ascending=False,
        na_position="last",
    )
    return recipes.iloc[offset : offset + limit]


def _fetch_recipes_with_substitutions_batch(
    candidate_set: set[str],
    offset: int,
    batch_size: int,
) -> tuple[pd.DataFrame, bool]:
    recipes_batch = _load_recipes_for_picker(limit=batch_size, offset=offset)
    if recipes_batch.empty:
        return recipes_batch, True

    recipes_batch = recipes_batch.copy()
    recipes_batch["ingredient_list"] = recipes_batch["top_ingredients"].apply(_to_ingredients_list)
    for col in ("sentiment_rating", "avg_rating", "weighted_review_count"):
        if col in recipes_batch.columns:
            recipes_batch[col] = pd.to_numeric(recipes_batch[col], errors="coerce")
    recipes_batch["display_rating"] = recipes_batch["sentiment_rating"].fillna(recipes_batch["avg_rating"])
    recipes_batch["has_substitutions"] = recipes_batch["ingredient_list"].apply(
        lambda ing_list: any(ing in candidate_set for ing in ing_list)
    )
    recipes_with_subs = recipes_batch[recipes_batch["has_substitutions"]].copy()
    exhausted = len(recipes_batch) < batch_size
    return recipes_with_subs, exhausted


def _prime_recipe_picker(candidate_set: set[str]) -> None:
    accumulated_batches: list[pd.DataFrame] = []
    offset = 0
    exhausted = False

    for _ in range(_RECIPE_PICKER_MAX_INITIAL_BATCHES):
        batch, exhausted = _fetch_recipes_with_substitutions_batch(
            candidate_set=candidate_set,
            offset=offset,
            batch_size=_RECIPE_PICKER_BATCH_SIZE,
        )
        if not batch.empty:
            accumulated_batches.append(batch)
        offset += _RECIPE_PICKER_BATCH_SIZE
        total_options = sum(len(df) for df in accumulated_batches)
        if total_options >= _RECIPE_PICKER_INITIAL_OPTIONS or exhausted:
            break

    if accumulated_batches:
        picker_df = pd.concat(accumulated_batches, ignore_index=True)
        picker_df = picker_df.drop_duplicates(subset=["recipe_id"], keep="first")
    else:
        picker_df = pd.DataFrame()

    st.session_state[_RECIPE_PICKER_STATE_KEY] = picker_df
    st.session_state[_RECIPE_PICKER_OFFSET_KEY] = offset
    st.session_state[_RECIPE_PICKER_EXHAUSTED_KEY] = exhausted


def _append_recipe_picker_batch(candidate_set: set[str]) -> None:
    if st.session_state.get(_RECIPE_PICKER_EXHAUSTED_KEY, False):
        return

    offset = int(st.session_state.get(_RECIPE_PICKER_OFFSET_KEY, 0))
    current = st.session_state.get(_RECIPE_PICKER_STATE_KEY)
    if not isinstance(current, pd.DataFrame):
        current = pd.DataFrame()

    next_batch, exhausted = _fetch_recipes_with_substitutions_batch(
        candidate_set=candidate_set,
        offset=offset,
        batch_size=_RECIPE_PICKER_BATCH_SIZE,
    )
    st.session_state[_RECIPE_PICKER_OFFSET_KEY] = offset + _RECIPE_PICKER_BATCH_SIZE
    st.session_state[_RECIPE_PICKER_EXHAUSTED_KEY] = exhausted

    if next_batch.empty:
        st.session_state[_RECIPE_PICKER_STATE_KEY] = current
        return

    combined = pd.concat([current, next_batch], ignore_index=True)
    combined = combined.drop_duplicates(subset=["recipe_id"], keep="first")
    st.session_state[_RECIPE_PICKER_STATE_KEY] = combined


def _trigger_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    experimental_rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun_fn):
        experimental_rerun_fn()


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


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg_rating_to_sentiment(avg: float) -> float:
    """Map typical 1–5 star mean into ~[-1, 1] to align with interaction sentiment scale."""
    return max(-1.0, min(1.0, (float(avg) - 3.0) / 2.0))


def _recipe_sentiment_score_for_bayesian(recipe_row: pd.Series) -> tuple[float | None, str]:
    """
    Returns (score_on_minus1_to_1, source) for Bayesian sentiment metrics.
    Prefers sentiment_rating; falls back to avg_rating / display_rating with scale detection.
    """
    s = _safe_float(recipe_row.get("sentiment_rating"))
    if s is not None:
        if -1.0 <= s <= 1.0:
            return s, "sentiment_rating"
        if 0.0 <= s <= 5.0:
            return _avg_rating_to_sentiment(s), "sentiment_rating"
        return max(-1.0, min(1.0, s)), "sentiment_rating"

    avg = _safe_float(recipe_row.get("avg_rating"))
    if avg is not None:
        return _avg_rating_to_sentiment(avg), "avg_rating"

    dr = _safe_float(recipe_row.get("display_rating"))
    if dr is None:
        return None, ""
    if -1.0 <= dr <= 1.0:
        return dr, "display_rating"
    if 0.0 <= dr <= 5.0:
        return _avg_rating_to_sentiment(dr), "display_rating"
    return max(-1.0, min(1.0, dr)), "display_rating"


def _recipe_star_equivalent(recipe_row: pd.Series) -> float | None:
    """Star display: from sentiment [-1,1] if available, else average rating on 0–5."""
    s = _safe_float(recipe_row.get("sentiment_rating"))
    if s is not None and -1.0 <= s <= 1.0:
        return _to_star_rating(s)
    if s is not None and 0.0 <= s <= 5.0:
        return max(0.0, min(5.0, s))

    avg = _safe_float(recipe_row.get("avg_rating"))
    if avg is not None:
        return max(0.0, min(5.0, avg))

    dr = _safe_float(recipe_row.get("display_rating"))
    if dr is None:
        return None
    if -1.0 <= dr <= 1.0:
        return _to_star_rating(dr)
    return max(0.0, min(5.0, dr))


def _format_metric_value(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:.2f}{suffix}"


def _format_base_adjusted(base: float | None, adjusted: float | None) -> str:
    if base is None or adjusted is None:
        return "—"
    return f"{base:.2f} -> {adjusted:.2f}"


def _buy_link_for_ingredient(ingredient: str) -> str:
    query = quote_plus(ingredient.strip())
    return f"https://www.amazon.com/s?k={query}&i=grocery"


def _update_swap_map(
    active_swaps: dict[str, str],
    candidate_ingredient: str,
    substitute_ingredient: str | None,
) -> dict[str, str]:
    updated_swaps = dict(active_swaps)
    if substitute_ingredient is None:
        updated_swaps.pop(candidate_ingredient, None)
    else:
        updated_swaps[candidate_ingredient] = substitute_ingredient
    return updated_swaps


def _set_recipe_swap(
    swap_key: str,
    candidate_ingredient: str,
    substitute_ingredient: str | None,
) -> None:
    active_swaps = dict(st.session_state.get(swap_key, {}))
    st.session_state[swap_key] = _update_swap_map(
        active_swaps,
        candidate_ingredient,
        substitute_ingredient,
    )


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


@_fragment
def _render_swap_workspace(
    recs: pd.DataFrame,
    recipe_row: pd.Series,
    global_sentiment_mean: float,
    selected_recipe_id: int,
) -> None:
    st.info(
        "Recommendations prioritize underperforming ingredients and rank substitutes by rating lift, "
        "sentiment lift, culinary compatibility, and nutrition delta."
    )

    swap_key = f"_subs_swaps_{int(selected_recipe_id)}"
    if swap_key not in st.session_state:
        st.session_state[swap_key] = {}
    active_swaps: dict = st.session_state[swap_key]

    if isinstance(recipe_row, pd.DataFrame):
        recipe_row = recipe_row.iloc[0]

    base_rating, base_rating_source = _recipe_sentiment_score_for_bayesian(recipe_row)
    base_stars = _recipe_star_equivalent(recipe_row)
    base_weight = recipe_row.get("weighted_review_count")
    base_weight = float(base_weight) if pd.notna(base_weight) else 0.0
    if active_swaps:
        active_df = pd.DataFrame(
            {
                "candidate_ingredient": list(active_swaps.keys()),
                "substitute_ingredient": list(active_swaps.values()),
            }
        )
        selected_rows = recs.merge(
            active_df,
            on=["candidate_ingredient", "substitute_ingredient"],
            how="inner",
        )
    else:
        selected_rows = recs.iloc[0:0]

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
    adjusted_stars = _to_star_rating(adjusted_sentiment_bayes)
    delta_totals = {
        "rating": float(selected_rows["rating_delta"].sum()) if not selected_rows.empty else 0.0,
        "sentiment": float(selected_rows["sentiment_delta"].sum()) if not selected_rows.empty else 0.0,
        "protein": float(selected_rows["protein_delta"].sum()) if not selected_rows.empty else 0.0,
        "saturated_fat": float(selected_rows["saturated_fat_delta"].sum()) if not selected_rows.empty else 0.0,
        "sugar": float(selected_rows["sugar_delta"].sum()) if not selected_rows.empty else 0.0,
        "sodium": float(selected_rows["sodium_delta"].sum()) if not selected_rows.empty else 0.0,
        "calories": float(selected_rows["calories_delta"].sum()) if not selected_rows.empty else 0.0,
        "health": float(selected_rows["health_delta"].sum()) if not selected_rows.empty else 0.0,
    }

    base_protein = _safe_float(recipe_row.get("protein"))
    base_saturated_fat = _safe_float(recipe_row.get("saturated_fat"))
    base_sugar = _safe_float(recipe_row.get("sugar"))
    base_sodium = _safe_float(recipe_row.get("sodium"))
    base_calories = _safe_float(recipe_row.get("calories"))
    base_health = (
        0.4 * base_protein
        - 0.25 * base_saturated_fat
        - 0.2 * base_sugar
        - 0.15 * base_sodium
        if None not in (base_protein, base_saturated_fat, base_sugar, base_sodium)
        else None
    )

    adjusted_protein = (
        base_protein + delta_totals["protein"] if base_protein is not None else None
    )
    adjusted_saturated_fat = (
        base_saturated_fat + delta_totals["saturated_fat"]
        if base_saturated_fat is not None
        else None
    )
    adjusted_sugar = base_sugar + delta_totals["sugar"] if base_sugar is not None else None
    adjusted_sodium = base_sodium + delta_totals["sodium"] if base_sodium is not None else None
    adjusted_calories = (
        base_calories + delta_totals["calories"] if base_calories is not None else None
    )
    adjusted_health = (
        base_health + delta_totals["health"] if base_health is not None else None
    )

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
    rating_note = ""
    if base_rating is not None and base_rating_source in ("avg_rating", "display_rating"):
        rating_note = (
            " Base sentiment uses average rating mapped to the sentiment scale when "
            "`sentiment_rating` is missing."
        )
    st.caption(
        f"Bayesian recompute uses m={_BAYESIAN_PSEUDO_COUNT:.0f}, "
        f"global sentiment mean C={global_sentiment_mean:.3f}, "
        f"weighted review depth={base_weight:.2f}.{rating_note}"
    )

    st.markdown("### Nutrition and Health (Base vs Adjusted)")
    n1, n2, n3 = st.columns(3)
    n1.metric(
        "Protein (base → adjusted)",
        _format_base_adjusted(base_protein, adjusted_protein),
        delta=(
            f"{delta_totals['protein']:+.2f}"
            if base_protein is not None
            else None
        ),
    )
    n2.metric(
        "Saturated fat (base → adjusted)",
        _format_base_adjusted(base_saturated_fat, adjusted_saturated_fat),
        delta=(
            f"{delta_totals['saturated_fat']:+.2f}"
            if base_saturated_fat is not None
            else None
        ),
        delta_color="inverse",
    )
    n3.metric(
        "Sugar (base → adjusted)",
        _format_base_adjusted(base_sugar, adjusted_sugar),
        delta=(
            f"{delta_totals['sugar']:+.2f}"
            if base_sugar is not None
            else None
        ),
        delta_color="inverse",
    )

    n4, n5, n6 = st.columns(3)
    n4.metric(
        "Sodium (base → adjusted)",
        _format_base_adjusted(base_sodium, adjusted_sodium),
        delta=(
            f"{delta_totals['sodium']:+.2f}"
            if base_sodium is not None
            else None
        ),
        delta_color="inverse",
    )
    n5.metric(
        "Calories (base → adjusted)",
        _format_base_adjusted(base_calories, adjusted_calories),
        delta=(
            f"{delta_totals['calories']:+.2f}"
            if base_calories is not None
            else None
        ),
        delta_color="inverse",
    )
    n6.metric(
        "Health score (base → adjusted)",
        _format_base_adjusted(base_health, adjusted_health),
        delta=(
            f"{delta_totals['health']:+.2f}"
            if base_health is not None
            else None
        ),
    )
    st.caption(
        "Adjusted values = base recipe nutrition from `dim_recipe` + sum of applied substitution deltas."
    )

    available_candidates = sorted(recs["candidate_ingredient"].dropna().astype(str).unique().tolist())
    st.markdown("### Ingredients in this recipe")
    ingredient_list = recipe_row.get("ingredient_list")
    ingredient_list = ingredient_list if isinstance(ingredient_list, list) else []
    ingredient_pills: list[str] = []
    for ingredient in ingredient_list:
        original = str(ingredient).strip().lower()
        swapped = active_swaps.get(original)
        if swapped:
            ingredient_pills.append(
                (
                    '<span class="subs-ing-pill subs-ing-pill-swapped">'
                    f'{html.escape(str(swapped))}'
                    '</span>'
                )
            )
        else:
            ingredient_pills.append(
                (
                    '<span class="subs-ing-pill subs-ing-pill-original">'
                    f'{html.escape(original)}'
                    '</span>'
                )
            )
    if ingredient_pills:
        st.markdown(
            f'<div class="subs-ing-list">{"".join(ingredient_pills)}</div>',
            unsafe_allow_html=True,
        )
    st.caption("Select an ingredient to view substitutes.")

    selected_candidate_key = f"_subs_selected_candidate_{int(selected_recipe_id)}"
    if selected_candidate_key not in st.session_state:
        st.session_state[selected_candidate_key] = available_candidates[0] if available_candidates else None
    if st.session_state[selected_candidate_key] not in available_candidates and available_candidates:
        st.session_state[selected_candidate_key] = available_candidates[0]

    selected_candidate = st.selectbox(
        "Pick ingredient",
        options=available_candidates,
        index=available_candidates.index(st.session_state[selected_candidate_key]),
        key=f"pick_ing_{selected_recipe_id}",
    )
    st.session_state[selected_candidate_key] = selected_candidate

    candidate_rows = (
        recs[recs["candidate_ingredient"] == selected_candidate]
        .sort_values("recommendation_score", ascending=False)
        .reset_index(drop=True)
    )
    st.divider()
    st.markdown(f"### Substitutes for `{selected_candidate}`")

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
                is_active = active_swaps.get(selected_candidate) == sub
                st.button(
                    "Applied ✅" if is_active else "Apply swap",
                    key=f"swap_{selected_recipe_id}_{selected_candidate}_{sub}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                    on_click=_set_recipe_swap,
                    args=(swap_key, selected_candidate, None if is_active else sub),
                )

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
                    st.markdown(f"#### 🔍 Explore `{selected_candidate}`")
                    st.caption("Compare brands and alternatives for the original ingredient.")
                    st.link_button(
                        "View alternatives",
                        _buy_link_for_ingredient(selected_candidate),
                        use_container_width=True,
                        help="Opens an external page to buy the original ingredient.",
                    )
        st.markdown("")


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
        .subs-ing-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 8px 0 10px 0;
        }
        .subs-ing-pill {
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 0.84rem;
            font-weight: 600;
            border: 1px solid transparent;
            line-height: 1.2;
        }
        .subs-ing-pill-original {
            color: #6b21a8;
            background: #f5e8ff;
            border-color: #d8b4fe;
        }
        .subs-ing-pill-swapped {
            color: #166534;
            background: #dcfce7;
            border-color: #86efac;
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
    global_sentiment_mean = _load_global_sentiment_mean()

    if subs.empty:
        st.warning(
            "No substitution engine data found. Run the batch pipeline load step to populate "
            "`fact_substitution_recommendations` (or generate `substitution_engine.parquet`)."
        )
        return
    candidate_set = set(subs["candidate_ingredient"])
    current_candidate_count = int(st.session_state.get(_RECIPE_PICKER_CANDIDATE_COUNT_KEY, -1))
    if current_candidate_count != len(candidate_set):
        st.session_state.pop(_RECIPE_PICKER_STATE_KEY, None)
        st.session_state.pop(_RECIPE_PICKER_OFFSET_KEY, None)
        st.session_state.pop(_RECIPE_PICKER_EXHAUSTED_KEY, None)
        st.session_state[_RECIPE_PICKER_CANDIDATE_COUNT_KEY] = len(candidate_set)

    if _RECIPE_PICKER_STATE_KEY not in st.session_state:
        _prime_recipe_picker(candidate_set)

    recipes_with_subs = st.session_state.get(_RECIPE_PICKER_STATE_KEY)
    if not isinstance(recipes_with_subs, pd.DataFrame):
        recipes_with_subs = pd.DataFrame()

    if recipes_with_subs.empty:
        for _ in range(3):
            if bool(st.session_state.get(_RECIPE_PICKER_EXHAUSTED_KEY, False)):
                break
            _append_recipe_picker_batch(candidate_set)
            recipes_with_subs = st.session_state.get(_RECIPE_PICKER_STATE_KEY, recipes_with_subs)
            if isinstance(recipes_with_subs, pd.DataFrame) and not recipes_with_subs.empty:
                break

    if recipes_with_subs.empty:
        st.info(
            "No recipes currently map to substitution candidates with available data. "
            "Try rerunning features/load for broader coverage."
        )
        return

    picker_total = len(recipes_with_subs)
    picker_exhausted = bool(st.session_state.get(_RECIPE_PICKER_EXHAUSTED_KEY, False))
    #st.caption(f"Recipe picker loaded {picker_total} recipes with substitutions.")

    option_map = {
        f"{row['recipe_id']} — {row['name']}": row["recipe_id"]
        for _, row in recipes_with_subs.iterrows()
        if pd.notna(row.get("recipe_id")) and str(row.get("name", "")).strip()
    }
    if not option_map:
        st.warning("Recipe picker could not be populated from current data.")
        return

    recipe_options = list(option_map.keys())
    if not picker_exhausted:
        recipe_options.append(_RECIPE_PICKER_LOAD_MORE_SENTINEL)

    previous_label = st.session_state.get(_RECIPE_PICKER_LAST_SELECTION_KEY)
    if previous_label not in option_map and option_map:
        previous_label = next(iter(option_map))

    default_index = 0
    if previous_label in recipe_options:
        default_index = recipe_options.index(previous_label)

    selected_label = st.selectbox(
        "Select recipe",
        options=recipe_options,
        index=default_index,
    )

    if selected_label == _RECIPE_PICKER_LOAD_MORE_SENTINEL:
        _append_recipe_picker_batch(candidate_set)
        fallback_label = previous_label if previous_label in option_map else next(iter(option_map))
        st.session_state[_RECIPE_PICKER_LAST_SELECTION_KEY] = fallback_label
        _trigger_rerun()
        return

    st.session_state[_RECIPE_PICKER_LAST_SELECTION_KEY] = selected_label
    selected_recipe_id = option_map[selected_label]
    recipe_row = recipes_with_subs.set_index("recipe_id").loc[selected_recipe_id]
    recipe_ingredients = set(recipe_row["ingredient_list"])

    recs = subs[subs["candidate_ingredient"].isin(recipe_ingredients)].copy()
    if recs.empty:
        st.info(
            "No substitution candidates in this recipe. Pick a different recipe to see swap cards."
        )
        return

    recs = recs.sort_values("recommendation_score", ascending=False).reset_index(drop=True)

    _render_swap_workspace(
        recs=recs,
        recipe_row=recipe_row,
        global_sentiment_mean=global_sentiment_mean,
        selected_recipe_id=int(selected_recipe_id),
    )
