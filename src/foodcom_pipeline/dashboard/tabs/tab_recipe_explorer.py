"""Recipe Explorer tab — hero search, filters, recipe cards, Shop buttons, pagination."""

import os
import urllib.parse
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


def render() -> None:
    st.info("Recipe Explorer — full UI coming in Task 9.")
