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
