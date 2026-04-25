import pandas as pd
import pytest
from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import (
    apply_filters,
    build_amazon_url,
    build_instacart_url,
    nutrition_bar_color,
    _is_useful_tag,
)


class TestBuildAmazonUrl:
    def test_basic(self):
        url = build_amazon_url(["chicken", "garlic", "tomato"])
        assert url.startswith("https://www.amazon.com/s?k=")
        assert "amazonfresh" in url
        assert "chicken" in url
        assert "garlic" in url

    def test_caps_at_eight_ingredients(self):
        ingredients = [f"ingredient{i}" for i in range(15)]
        url = build_amazon_url(ingredients)
        # Only first 8 should appear
        for i in range(8):
            assert f"ingredient{i}" in url
        assert "ingredient8" not in url

    def test_spaces_encoded(self):
        url = build_amazon_url(["soy sauce", "fish sauce"])
        assert " " not in url
        assert "soy" in url

    def test_empty_list_returns_base_url(self):
        url = build_amazon_url([])
        assert "amazon.com" in url


class TestBuildInstacartUrl:
    def test_basic(self):
        url = build_instacart_url("Chicken Tikka Masala")
        assert url.startswith("https://www.instacart.com/store/s?k=")
        assert "Chicken" in url or "chicken" in url.lower()

    def test_spaces_encoded(self):
        url = build_instacart_url("beef stir fry")
        assert " " not in url

    def test_ingredients_keyword_appended(self):
        url = build_instacart_url("pasta carbonara")
        assert "ingredient" in url.lower()


def _make_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"name": "Chicken Tikka", "top_ingredients": "chicken|tomato|cream",
         "avg_cook_minutes": 45.0, "display_rating": 4.8,
         "tags": "Indian, curry, spicy"},
        {"name": "Spaghetti",     "top_ingredients": "pasta|egg|bacon",
         "avg_cook_minutes": 20.0, "display_rating": 4.2,
         "tags": "Italian, quick"},
        {"name": "Veggie Stir Fry","top_ingredients": "broccoli|soy sauce|garlic",
         "avg_cook_minutes": 15.0, "display_rating": 3.9,
         "tags": "Asian, vegetarian"},
    ])


class TestApplyFilters:
    def test_search_by_name(self):
        result = apply_filters(_make_df(), "chicken", 180, 1.0, [])
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Chicken Tikka"

    def test_search_by_ingredient(self):
        result = apply_filters(_make_df(), "pasta", 180, 1.0, [])
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Spaghetti"

    def test_cook_time_filter(self):
        result = apply_filters(_make_df(), "", 20, 1.0, [])
        assert len(result) == 2
        assert all(result["avg_cook_minutes"] <= 20)

    def test_min_rating_filter(self):
        result = apply_filters(_make_df(), "", 180, 4.5, [])
        assert len(result) == 1
        assert result.iloc[0]["display_rating"] >= 4.5

    def test_tag_filter(self):
        result = apply_filters(_make_df(), "", 180, 1.0, ["Italian"])
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Spaghetti"

    def test_no_filters_returns_all(self):
        df = _make_df()
        result = apply_filters(df, "", 9999, 0.0, [])
        assert len(result) == len(df)

    def test_search_case_insensitive(self):
        result = apply_filters(_make_df(), "CHICKEN", 180, 1.0, [])
        assert len(result) == 1


class TestNutritionBarColor:
    def test_high_sugar_is_amber(self):
        assert nutrition_bar_color(35.0, "sugar") == "#f59e0b"

    def test_low_sugar_is_emerald(self):
        assert nutrition_bar_color(10.0, "sugar") == "#10b981"

    def test_high_sodium_is_red(self):
        assert nutrition_bar_color(60.0, "sodium") == "#ef4444"

    def test_low_sodium_is_emerald(self):
        assert nutrition_bar_color(20.0, "sodium") == "#10b981"

    def test_protein_always_emerald(self):
        assert nutrition_bar_color(80.0, "protein") == "#10b981"

    def test_fat_always_emerald(self):
        assert nutrition_bar_color(90.0, "fat") == "#10b981"


class TestIsUsefulTag:
    def test_blocklisted_tags_excluded(self):
        assert _is_useful_tag("time-to-make") is False
        assert _is_useful_tag("course") is False
        assert _is_useful_tag("main-ingredient") is False
        assert _is_useful_tag("preparation") is False
        assert _is_useful_tag("occasion") is False
        assert _is_useful_tag("equipment") is False
        assert _is_useful_tag("dietary") is False
        assert _is_useful_tag("technique") is False
        assert _is_useful_tag("number-of-servings") is False
        assert _is_useful_tag("meat") is False
        assert _is_useful_tag("vegetables") is False

    def test_for_prefix_excluded(self):
        assert _is_useful_tag("for-large-groups") is False
        assert _is_useful_tag("for-1-or-2-servings") is False

    def test_servings_suffix_excluded(self):
        assert _is_useful_tag("1-2-servings") is False
        assert _is_useful_tag("4-6-servings") is False

    def test_numeric_tags_excluded(self):
        assert _is_useful_tag("60") is False
        assert _is_useful_tag("30") is False

    def test_useful_tags_pass(self):
        assert _is_useful_tag("italian") is True
        assert _is_useful_tag("vegetarian") is True
        assert _is_useful_tag("desserts") is True
        assert _is_useful_tag("30-minutes-or-less") is True
        assert _is_useful_tag("asian") is True
        assert _is_useful_tag("low-fat") is True
        assert _is_useful_tag("affordable-for-groups") is True


import math
from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import (
    _compute_affiliate_columns,
    _nutrition_radar,
)


class TestNutritionRadar:
    def _make_row(self):
        return pd.Series({"protein": 20.0, "fat": 30.0, "carbs": 50.0,
                          "sugar": 10.0, "sodium": 40.0})

    def test_no_median_has_one_trace(self):
        fig = _nutrition_radar(self._make_row())
        assert len(fig.data) == 1

    def test_with_median_has_two_traces(self):
        median = pd.Series({"protein": 15.0, "fat": 25.0, "carbs": 40.0,
                            "sugar": 8.0, "sodium": 30.0})
        fig = _nutrition_radar(self._make_row(), median_row=median)
        assert len(fig.data) == 2

    def test_median_trace_is_dashed_grey(self):
        median = pd.Series({"protein": 15.0, "fat": 25.0, "carbs": 40.0,
                            "sugar": 8.0, "sodium": 30.0})
        fig = _nutrition_radar(self._make_row(), median_row=median)
        median_trace = fig.data[1]
        assert median_trace.line.dash == "dash"
        assert "9ca3af" in median_trace.line.color


def _make_affiliate_df(n=1200) -> pd.DataFrame:
    """1200-row df: recipes with varying review_count, cook time, ingredient_count."""
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "recipe_id":       range(n),
        "review_count":    rng.integers(1, 500, size=n).astype(float),
        "avg_cook_minutes": rng.integers(5, 120, size=n).astype(float),
        "ingredient_count": rng.integers(2, 20, size=n).astype(float),
    })


class TestComputeAffiliateColumns:
    def test_new_columns_exist(self):
        df = _compute_affiliate_columns(_make_affiliate_df())
        for col in ["affiliate_score", "cart_ready", "review_velocity",
                    "basket_value_est", "revenue_proj_monthly"]:
            assert col in df.columns, f"missing column: {col}"

    def test_affiliate_score_only_top1000(self):
        df = _compute_affiliate_columns(_make_affiliate_df(n=1200))
        non_nan = df["affiliate_score"].notna().sum()
        assert non_nan == 1000

    def test_affiliate_score_range(self):
        df = _compute_affiliate_columns(_make_affiliate_df())
        scores = df["affiliate_score"].dropna()
        assert (scores >= 0).all()
        assert (scores <= 1.01).all()  # small float tolerance

    def test_cart_ready_sweet_spot(self):
        df = pd.DataFrame({
            "recipe_id": [1, 2, 3],
            "review_count": [10.0, 10.0, 10.0],
            "avg_cook_minutes": [30.0, 30.0, 30.0],
            "ingredient_count": [7.0, 11.0, 12.0],
        })
        result = _compute_affiliate_columns(df)
        assert result.loc[result["recipe_id"] == 1, "cart_ready"].iloc[0] is True
        assert result.loc[result["recipe_id"] == 2, "cart_ready"].iloc[0] is True
        assert result.loc[result["recipe_id"] == 3, "cart_ready"].iloc[0] is False

    def test_basket_value_formula(self):
        df = pd.DataFrame({
            "recipe_id": [1],
            "review_count": [10.0],
            "avg_cook_minutes": [30.0],
            "ingredient_count": [8.0],
        })
        result = _compute_affiliate_columns(df)
        assert math.isclose(result.iloc[0]["basket_value_est"], 8 * 3.50)

    def test_revenue_proj_formula(self):
        df = pd.DataFrame({
            "recipe_id": [1],
            "review_count": [10.0],
            "avg_cook_minutes": [30.0],
            "ingredient_count": [8.0],
        })
        result = _compute_affiliate_columns(df)
        basket = 8 * 3.50
        expected_rev = 10_000 * 0.02 * basket * 0.04
        assert math.isclose(result.iloc[0]["revenue_proj_monthly"], expected_rev)

    def test_review_velocity_formula(self):
        df = pd.DataFrame({
            "recipe_id": [1],
            "review_count": [73.0],
            "avg_cook_minutes": [30.0],
            "ingredient_count": [8.0],
        })
        result = _compute_affiliate_columns(df)
        assert math.isclose(result.iloc[0]["review_velocity"], 1.0)

    def test_nan_safe_missing_cook_time(self):
        df = pd.DataFrame({
            "recipe_id": [1, 2],
            "review_count": [10.0, 20.0],
            "avg_cook_minutes": [float("nan"), 30.0],
            "ingredient_count": [8.0, 9.0],
        })
        result = _compute_affiliate_columns(df)
        # Should not raise; affiliate_score may be NaN for row 1 but no exception
        assert "affiliate_score" in result.columns
