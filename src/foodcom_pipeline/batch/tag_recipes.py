"""
tag_recipes.py
--------------
Generates food category tags for recipes using regex matching + LLM fallback.

Two-stage tagging:
  1. Regex matching: fast, deterministic, case-insensitive keyword matching
  2. LLM fallback: Gemini API for untagged recipes (capped at 20 per run)

Output: recipe_tags.parquet (staging) + PostgreSQL recipe_tags table
"""

import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────

from foodcom_pipeline.batch.extract import STAGING_DIR, POSTGRES_CONN

CLEANED_STAGING = STAGING_DIR / 'recipes_clean.parquet'
TAGS_STAGING = STAGING_DIR / 'recipe_tags.parquet'

MAX_LLM_RECIPES = 20
GEMINI_MODEL = 'gemini-1.5-flash'

# Tag dictionary: tag -> list of trigger keywords
TAG_DICT = {
    "pasta": ["pasta", "spaghetti", "fettuccine", "penne", "linguine", "rigatoni", "lasagna", "noodle"],
    "chicken": ["chicken", "poultry", "hen"],
    "beef": ["beef", "steak", "ground beef", "mince"],
    "seafood": ["salmon", "shrimp", "prawn", "cod", "tuna", "fish", "crab", "lobster", "scallop"],
    "vegetarian": ["tofu", "tempeh", "lentil", "chickpea", "veggie", "vegetable"],
    "baking": ["flour", "baking powder", "baking soda", "yeast", "cake", "cookie", "bread", "pastry"],
    "soup": ["soup", "stew", "broth", "chowder", "bisque"],
    "salad": ["salad", "slaw", "greens", "arugula", "spinach salad"],
    "breakfast": ["pancake", "waffle", "omelette", "omelet", "scrambled", "french toast", "breakfast"],
    "dessert": ["chocolate", "cake", "cookie", "brownie", "ice cream", "pudding", "tart", "pie"],
}

# ─────────────────────────────────────────────────────────────────────────
# Stage 1: Regex tagging
# ─────────────────────────────────────────────────────────────────────────


def tag_recipe_regex(name: str, ingredients: str) -> set[str]:
    """
    Apply regex matching to name and ingredients.
    Returns set of matched tags (lowercase).

    Args:
        name: recipe name
        ingredients: ingredients (typically pipe-separated string)

    Returns:
        set of tag strings that matched
    """
    text = f"{name} {ingredients}".lower()
    matched_tags = set()

    for tag, keywords in TAG_DICT.items():
        for keyword in keywords:
            # Word boundary matching to avoid partial matches
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text):
                matched_tags.add(tag)
                break  # Found this tag, move to next tag

    return matched_tags


# ─────────────────────────────────────────────────────────────────────────
# Stage 2: LLM fallback tagging (Gemini)
# ─────────────────────────────────────────────────────────────────────────


def _get_gemini_api_key() -> str | None:
    """
    Get Gemini API key from Airflow Variable or .env file.
    Returns None if not found.
    """
    try:
        from airflow.models import Variable
        return Variable.get('GEMINI_API_KEY')
    except (ImportError, KeyError):
        pass

    # Fallback to .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv('GEMINI_API_KEY')
    except Exception:
        pass

    return None


def tag_recipe_llm(name: str, ingredients: str, api_key: str) -> set[str]:
    """
    Use Gemini API to tag recipe via LLM.

    Args:
        name: recipe name
        ingredients: ingredients string
        api_key: Gemini API key

    Returns:
        set of tags returned by LLM (or empty set on failure)
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        tag_list = ", ".join(sorted(TAG_DICT.keys()))

        prompt = f"""Given this recipe name and ingredients, return up to 3 food category tags from this list only: {tag_list}. Recipe name: {name}. Ingredients: {ingredients}. Return only a JSON array of strings, no explanation."""

        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=f"You are a culinary expert. Assign up to 3 tags from this list: {tag_list}. Return ONLY a JSON array."
        )
        response = model.generate_content(prompt)

        # Parse JSON response
        response_text = response.text.strip()
        # Remove potential markdown formatting
        cleaned_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE)
        tags = json.loads(cleaned_text)

        if not isinstance(tags, list):
            logger.warning(f"Gemini returned non-list for {name}: {tags}")
            return set()

        # Normalize to lowercase, filter against TAG_DICT keys
        normalized_tags = set()
        for tag in tags:
            tag_lower = str(tag).strip().lower()
            if tag_lower in TAG_DICT:
                normalized_tags.add(tag_lower)

        return normalized_tags

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Gemini JSON response for recipe '{name}': {e}")
        return set()
    except Exception as e:
        logger.error(f"Gemini API call failed for recipe '{name}': {e}")
        return set()


# ─────────────────────────────────────────────────────────────────────────
# Main task
# ─────────────────────────────────────────────────────────────────────────


def run_tag_recipes(**context) -> None:
    """
    Tag recipes using regex matching + LLM fallback.
    Outputs to recipe_tags.parquet and PostgreSQL recipe_tags table.
    """
    logger.info("Starting recipe tagging task...")

    # Load cleaned recipes
    if not CLEANED_STAGING.exists():
        logger.warning(f"Cleaned staging file not found: {CLEANED_STAGING}")
        return

    cleaned_df = pd.read_parquet(CLEANED_STAGING)
    logger.info(f"Loaded {len(cleaned_df)} cleaned recipes")

    if cleaned_df.empty:
        logger.warning("No recipes to tag")
        return

    # Ensure required columns
    if 'id' not in cleaned_df.columns or 'name' not in cleaned_df.columns:
        logger.error("Missing 'id' or 'name' column in cleaned data")
        return

    # Stage 1: Regex tagging
    logger.info("Stage 1: Regex tagging...")
    tags_list = []
    regex_tag_count = 0

    for _, row in cleaned_df.iterrows():
        recipe_id = row['id']
        name = str(row.get('name', '')).strip()

        # Get ingredients string (multiple possible column names)
        ingredients = ''
        for col in ['ingredients_canonical_normalized', 'ingredients_normalized', 'ingredients']:
            if col in cleaned_df.columns and pd.notna(row[col]):
                ingredients = str(row[col]).strip()
                break

        matched_tags = tag_recipe_regex(name, ingredients)

        for tag in matched_tags:
            tags_list.append({
                'recipe_id': recipe_id,
                'tag': tag,
                'source': 'regex'
            })
            regex_tag_count += 1

    logger.info(f"Regex tagging found {regex_tag_count} tags across {len(cleaned_df)} recipes")

    # Stage 2: LLM tagging for untagged recipes
    logger.info("Stage 2: LLM tagging for untagged recipes...")

    # Find untagged recipes
    tagged_recipe_ids = set(row['recipe_id'] for row in tags_list)
    untagged_recipes = cleaned_df[~cleaned_df['id'].isin(tagged_recipe_ids)].copy()

    llm_tag_count = 0
    llm_skipped_count = 0

    if len(untagged_recipes) > 0:
        api_key = _get_gemini_api_key()

        if not api_key:
            logger.warning("Gemini API key not found (GEMINI_API_KEY env var or Airflow Variable)")
            llm_skipped_count = len(untagged_recipes)
        else:
            # Cap at MAX_LLM_RECIPES
            recipes_to_tag = untagged_recipes.head(MAX_LLM_RECIPES)
            llm_skipped_count = len(untagged_recipes) - len(recipes_to_tag)

            if llm_skipped_count > 0:
                logger.info(f"LLM tagging capped at {MAX_LLM_RECIPES} recipes. {llm_skipped_count} untagged recipes remain for future runs.")

            for _, row in recipes_to_tag.iterrows():
                recipe_id = row['id']
                name = str(row.get('name', '')).strip()

                # Get ingredients string
                ingredients = ''
                for col in ['ingredients_canonical_normalized', 'ingredients_normalized', 'ingredients']:
                    if col in cleaned_df.columns and pd.notna(row[col]):
                        ingredients = str(row[col]).strip()
                        break

                llm_tags = tag_recipe_llm(name, ingredients, api_key)

                for tag in llm_tags:
                    tags_list.append({
                        'recipe_id': recipe_id,
                        'tag': tag,
                        'source': 'llm'
                    })
                    llm_tag_count += 1

    logger.info(f"LLM tagging added {llm_tag_count} tags")

    # Create output DataFrame
    if not tags_list:
        logger.warning("No tags generated (all recipes untagged and LLM skipped or failed)")
        tags_df = pd.DataFrame(columns=['recipe_id', 'tag', 'source'])
    else:
        tags_df = pd.DataFrame(tags_list)

    logger.info(f"Total tags generated: {len(tags_df)}")

    # Write to parquet
    TAGS_STAGING.parent.mkdir(parents=True, exist_ok=True)
    tags_df.to_parquet(TAGS_STAGING, index=False)
    logger.info(f"Tags staged to {TAGS_STAGING}")

    # Write to PostgreSQL
    if not tags_df.empty:
        _write_tags_to_db(tags_df)

    # XCom push for monitoring
    context['ti'].xcom_push(key='regex_tag_count', value=regex_tag_count)
    context['ti'].xcom_push(key='llm_tag_count', value=llm_tag_count)
    context['ti'].xcom_push(key='llm_skipped_count', value=llm_skipped_count)
    context['ti'].xcom_push(key='total_tag_count', value=len(tags_df))


def _write_tags_to_db(tags_df: pd.DataFrame) -> None:
    """Write recipe tags to PostgreSQL."""
    try:
        engine = create_engine(POSTGRES_CONN)

        # Remove duplicates (recipe_id, tag, source combination)
        tags_df = tags_df.drop_duplicates(subset=['recipe_id', 'tag', 'source'])

        records = tags_df.to_dict(orient='records')

        upsert_sql = """
            INSERT INTO recipe_tags (recipe_id, tag, source)
            VALUES (:recipe_id, :tag, :source)
            ON CONFLICT (recipe_id, tag, source) DO NOTHING
        """

        # Batch upsert
        batch_size = 5000
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            with engine.begin() as conn:
                conn.execute(text(upsert_sql), batch)

        logger.info(f"Wrote {len(records)} recipe tags to PostgreSQL")

    except Exception as e:
        logger.error(f"Failed to write tags to PostgreSQL: {e}", exc_info=True)
