"""Recipe Explorer tab — hero search, filters, recipe cards, Shop buttons, pagination."""

import os
import urllib.parse
from pathlib import Path

import pandas as pd
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


def render() -> None:
    st.info("Recipe Explorer — full UI coming in Task 9.")
