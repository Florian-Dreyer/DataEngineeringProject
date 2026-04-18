import pandas as pd
import pytest
from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import (
    apply_filters,
    build_amazon_url,
    build_instacart_url,
    nutrition_bar_color,
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
