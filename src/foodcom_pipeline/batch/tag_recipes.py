"""
tag_recipes.py
--------------
Generates food category tags for recipes using regex matching + LLM fallback.

Two-stage tagging:
  1. Regex matching: fast, deterministic, case-insensitive keyword matching
  2. LLM fallback: Gemini API for weakly-tagged or untagged recipes (capped at 20 per run)

Output: recipe_tags.parquet (staging) + PostgreSQL recipe_tags table
"""

import json
import logging
import os
import re

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────

from foodcom_pipeline.batch.extract import STAGING_DIR, POSTGRES_CONN, atomic_parquet

CLEANED_STAGING = STAGING_DIR / "recipes_clean.parquet"
TAGS_STAGING = STAGING_DIR / "recipe_tags.parquet"

MAX_LLM_RECIPES = 20
GEMINI_MODEL = "gemini-1.5-flash"

# ---------------------------------------------------------------------
# Expanded tag dictionary
# Flat output is preserved for DB compatibility, but coverage is broader.
# ---------------------------------------------------------------------

TAG_DICT = {
    # Protein / base
    "chicken": [
        "chicken", "chicken breast", "chicken thigh", "chicken thighs",
        "drumstick", "drumsticks", "wings", "rotisserie", "poultry", "hen"
    ],
    "beef": [
        "beef", "steak", "ground beef", "mince", "minced beef", "burger",
        "burgers", "meatball", "meatballs", "brisket", "short rib", "pot roast"
    ],
    "pork": [
        "pork", "bacon", "ham", "sausage", "sausages", "prosciutto",
        "pork chop", "pork chops", "pulled pork", "ribs"
    ],
    "seafood": [
        "seafood", "salmon", "shrimp", "prawn", "prawns", "cod", "tuna",
        "fish", "crab", "lobster", "scallop", "scallops", "tilapia",
        "halibut", "mussels", "anchovy", "anchovies"
    ],
    "vegetarian": [
        "tofu", "tempeh", "lentil", "lentils", "chickpea", "chickpeas",
        "veggie", "vegetable", "vegetables", "mushroom", "mushrooms",
        "cauliflower", "eggplant", "zucchini", "beans", "black bean",
        "black beans", "white bean", "white beans"
    ],
    "egg": [
        "egg", "eggs", "omelette", "omelet", "scrambled eggs",
        "poached egg", "poached eggs", "frittata", "quiche", "shakshuka"
    ],

    # Starches / core formats
    "pasta": [
        "pasta", "spaghetti", "fettuccine", "penne", "linguine", "rigatoni",
        "lasagna", "mac and cheese", "macaroni", "ravioli", "tortellini",
        "gnocchi", "orzo", "noodle", "noodles", "ramen", "udon"
    ],
    "rice": [
        "rice", "fried rice", "risotto", "rice bowl", "grain bowl",
        "congee", "pilaf", "paella"
    ],
    "bread_baked": [
        "bread", "sourdough", "loaf", "roll", "rolls", "bun", "buns",
        "biscuit", "biscuits", "flatbread", "naan", "focaccia"
    ],

    # Dish formats
    "soup_stew": [
        "soup", "stew", "broth", "chowder", "bisque", "ramen", "pho"
    ],
    "salad": [
        "salad", "slaw", "greens", "arugula", "caesar", "cobb", "spinach salad"
    ],
    "sandwich_wrap": [
        "sandwich", "sandwiches", "wrap", "wraps", "burger", "burgers",
        "panini", "melt", "melts", "toastie", "toasties", "quesadilla", "quesadillas"
    ],
    "casserole_bake": [
        "casserole", "bake", "bakes", "gratin", "sheet pan", "sheet-pan",
        "traybake", "tray bake"
    ],
    "stir_fry": [
        "stir fry", "stir-fry", "fried noodles", "lo mein", "chow mein"
    ],
    "tacos_handheld": [
        "taco", "tacos", "burrito", "burritos", "enchilada", "enchiladas",
        "quesadilla", "quesadillas", "birria tacos"
    ],
    "bowl": [
        "bowl", "bowls", "grain bowl", "rice bowl", "poke bowl", "beef bowl"
    ],

    # Occasion / type
    "breakfast": [
        "pancake", "pancakes", "waffle", "waffles", "omelette", "omelet",
        "scrambled", "french toast", "breakfast", "granola", "muffin", "muffins",
        "brunch"
    ],
    "dessert": [
        "dessert", "desserts", "chocolate", "cake", "cakes", "cookie",
        "cookies", "brownie", "brownies", "ice cream", "pudding", "tart",
        "tarts", "pie", "pies", "cheesecake", "cupcake", "cupcakes",
        "cobbler", "donut", "donuts"
    ],
    "baking": [
        "flour", "baking powder", "baking soda", "yeast", "cake", "cookie",
        "cookies", "bread", "pastry", "pastries", "sourdough", "loaf"
    ],

    # Cuisine / flavor families
    "italian": [
        "italian", "marinara", "alfredo", "bolognese", "pesto",
        "parmesan", "parmesan pasta", "risotto"
    ],
    "mexican": [
        "mexican", "taco", "tacos", "burrito", "enchilada", "quesadilla",
        "birria", "salsa", "elote"
    ],
    "korean": [
        "korean", "bulgogi", "kimchi", "gochujang"
    ],
    "japanese": [
        "japanese", "teriyaki", "miso", "udon", "ramen"
    ],
    "thai": [
        "thai", "pad thai", "green curry", "red curry", "peanut noodles"
    ],
    "indian": [
        "indian", "curry", "masala", "tikka", "dal", "dahl"
    ],
    "mediterranean": [
        "mediterranean", "feta", "tzatziki", "hummus", "shawarma"
    ],

    # Intent / style
    "healthy": [
        "healthy", "high protein", "low carb", "low-calorie", "low calorie",
        "protein bowl", "meal prep", "clean eating"
    ],
    "comfort_food": [
        "comfort food", "creamy", "cheesy", "hearty", "garlic butter",
        "mac and cheese", "pot pie"
    ],
    "quick_easy": [
        "quick", "easy", "weeknight", "30-minute", "30 minute",
        "one-pot", "one pot", "sheet pan", "sheet-pan"
    ],
}

# Normalize expressive search / AI-mode modifiers down to more canonical text
PHRASE_NORMALIZATION_RULES = [
    (r"\bmarry me\b", ""),
    (r"\bviral\b", ""),
    (r"\bone[\s-]?pot\b", ""),
    (r"\bsheet[\s-]?pan\b", ""),
    (r"\b30[\s-]?minute\b", ""),
    (r"\bquick\b", ""),
    (r"\beasy\b", ""),
    (r"\bweeknight\b", ""),
    (r"\btrending\b", ""),
    (r"\brecipe\b", ""),
    (r"\brecipes\b", ""),
]

# Precompile patterns once
TAG_PATTERNS = {
    tag: [re.compile(r"\b" + re.escape(keyword.lower()) + r"\b") for keyword in keywords]
    for tag, keywords in TAG_DICT.items()
}


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    for pattern, replacement in PHRASE_NORMALIZATION_RULES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^\w\s/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_recipe_text(name: str, ingredients: str) -> str:
    return f"{name} {ingredients}".strip()


# ─────────────────────────────────────────────────────────────────────────
# Stage 1: Regex tagging
# ─────────────────────────────────────────────────────────────────────────

def tag_recipe_regex(name: str, ingredients: str) -> set[str]:
    """
    Apply regex matching to normalized recipe name + ingredients.
    Returns a set of matched tags.

    This version is more robust for:
    - Food.com recipe titles
    - Google AI dish phrases
    - Google Trends recipe-like queries
    """
    raw_text = _get_recipe_text(name, ingredients)
    normalized_text = _normalize_text(raw_text)

    matched_tags: set[str] = set()

    for tag, patterns in TAG_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(normalized_text):
                matched_tags.add(tag)
                break

    return matched_tags


# ─────────────────────────────────────────────────────────────────────────
# Stage 2: LLM fallback tagging (Gemini)
# ─────────────────────────────────────────────────────────────────────────

def _get_gemini_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY")


def tag_recipe_llm(name: str, ingredients: str, api_key: str) -> set[str]:
    """
    Use Gemini API to tag a recipe/query via LLM.
    Returns only tags present in TAG_DICT.
    """
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        tag_list = ", ".join(sorted(TAG_DICT.keys()))

        prompt = (
            f"Given this recipe or dish query, assign up to 5 tags from this list only: {tag_list}. "
            f"Recipe name: {name}. "
            f"Ingredients: {ingredients}. "
            "Choose tags that best capture protein/base, dish format, cuisine, and intent where relevant. "
            "Return only a JSON array of strings. No explanation."
        )

        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=(
                "You are a culinary taxonomy assistant. "
                f"Assign up to 5 tags from this exact allowed list only: {tag_list}. "
                "Return ONLY a JSON array."
            ),
        )
        response = model.generate_content(prompt)

        response_text = response.text.strip()
        cleaned_text = re.sub(r"^```json\s*|\s*```$", "", response_text, flags=re.MULTILINE)
        tags = json.loads(cleaned_text)

        if not isinstance(tags, list):
            logger.warning("Gemini returned non-list for %s: %s", name, tags)
            return set()

        normalized_tags = set()
        for tag in tags:
            tag_lower = str(tag).strip().lower()
            if tag_lower in TAG_DICT:
                normalized_tags.add(tag_lower)

        return normalized_tags

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse Gemini JSON response for recipe '%s': %s", name, e)
        return set()
    except Exception as e:
        logger.error("Gemini API call failed for recipe '%s': %s", name, e)
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

    if not CLEANED_STAGING.exists():
        logger.warning("Cleaned staging file not found: %s", CLEANED_STAGING)
        return

    cleaned_df = pd.read_parquet(CLEANED_STAGING)
    logger.info("Loaded %s cleaned recipes", len(cleaned_df))

    if cleaned_df.empty:
        logger.warning("No recipes to tag")
        return

    if "id" not in cleaned_df.columns or "name" not in cleaned_df.columns:
        logger.error("Missing 'id' or 'name' column in cleaned data")
        return

    logger.info("Stage 1: Regex tagging...")
    tags_list = []
    regex_tag_count = 0

    for _, row in cleaned_df.iterrows():
        recipe_id = row["id"]
        name = str(row.get("name", "")).strip()

        ingredients = ""
        for col in ["ingredients_canonical_normalized", "ingredients_normalized", "ingredients"]:
            if col in cleaned_df.columns and pd.notna(row[col]):
                ingredients = str(row[col]).strip()
                break

        matched_tags = tag_recipe_regex(name, ingredients)

        for tag in matched_tags:
            tags_list.append(
                {
                    "recipe_id": recipe_id,
                    "tag": tag,
                    "source": "regex",
                }
            )
            regex_tag_count += 1

    logger.info(
        "Regex tagging found %s tags across %s recipes",
        regex_tag_count,
        len(cleaned_df),
    )

    logger.info("Stage 2: LLM tagging for weakly-tagged / untagged recipes...")

    tag_counts_by_recipe: dict[int, int] = {}
    for row in tags_list:
        recipe_id = row["recipe_id"]
        tag_counts_by_recipe[recipe_id] = tag_counts_by_recipe.get(recipe_id, 0) + 1

    # send recipes with 0 or 1 tags to LLM fallback
    weakly_tagged_ids = {
        recipe_id for recipe_id in cleaned_df["id"].tolist()
        if tag_counts_by_recipe.get(recipe_id, 0) <= 1
    }

    untagged_recipes = cleaned_df[cleaned_df["id"].isin(weakly_tagged_ids)].copy()

    llm_tag_count = 0
    llm_skipped_count = 0

    if len(untagged_recipes) > 0:
        api_key = _get_gemini_api_key()

        if not api_key:
            logger.warning("Gemini API key not found (GEMINI_API_KEY env var)")
            llm_skipped_count = len(untagged_recipes)
        else:
            recipes_to_tag = untagged_recipes.head(MAX_LLM_RECIPES)
            llm_skipped_count = len(untagged_recipes) - len(recipes_to_tag)

            if llm_skipped_count > 0:
                logger.info(
                    "LLM tagging capped at %s recipes. %s recipes remain for future runs.",
                    MAX_LLM_RECIPES,
                    llm_skipped_count,
                )

            existing_pairs = {
                (row["recipe_id"], row["tag"])
                for row in tags_list
            }

            for _, row in recipes_to_tag.iterrows():
                recipe_id = row["id"]
                name = str(row.get("name", "")).strip()

                ingredients = ""
                for col in ["ingredients_canonical_normalized", "ingredients_normalized", "ingredients"]:
                    if col in cleaned_df.columns and pd.notna(row[col]):
                        ingredients = str(row[col]).strip()
                        break

                llm_tags = tag_recipe_llm(name, ingredients, api_key)

                for tag in llm_tags:
                    if (recipe_id, tag) in existing_pairs:
                        continue

                    tags_list.append(
                        {
                            "recipe_id": recipe_id,
                            "tag": tag,
                            "source": "llm",
                        }
                    )
                    existing_pairs.add((recipe_id, tag))
                    llm_tag_count += 1

    logger.info("LLM tagging added %s tags", llm_tag_count)

    if not tags_list:
        logger.warning("No tags generated")
        tags_df = pd.DataFrame(columns=["recipe_id", "tag", "source"])
    else:
        tags_df = pd.DataFrame(tags_list).drop_duplicates(
            subset=["recipe_id", "tag", "source"]
        )

    logger.info("Total tags generated: %s", len(tags_df))

    TAGS_STAGING.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(tags_df, TAGS_STAGING)
    logger.info("Tags staged to %s", TAGS_STAGING)

    if not tags_df.empty:
        _write_tags_to_db(tags_df)

    context["ti"].xcom_push(key="regex_tag_count", value=regex_tag_count)
    context["ti"].xcom_push(key="llm_tag_count", value=llm_tag_count)
    context["ti"].xcom_push(key="llm_skipped_count", value=llm_skipped_count)
    context["ti"].xcom_push(key="total_tag_count", value=len(tags_df))


def _write_tags_to_db(tags_df: pd.DataFrame) -> None:
    """Write recipe tags to PostgreSQL."""
    try:
        engine = create_engine(POSTGRES_CONN)

        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS recipe_tags (
                        recipe_id INTEGER NOT NULL,
                        tag       TEXT    NOT NULL,
                        source    TEXT    NOT NULL,
                        PRIMARY KEY (recipe_id, tag, source)
                    )
                    """
                )
            )

        tags_df = tags_df.drop_duplicates(subset=["recipe_id", "tag", "source"])
        records = tags_df.to_dict(orient="records")

        upsert_sql = """
            INSERT INTO recipe_tags (recipe_id, tag, source)
            VALUES (:recipe_id, :tag, :source)
            ON CONFLICT (recipe_id, tag, source) DO NOTHING
        """

        batch_size = 5000
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            with engine.begin() as conn:
                conn.execute(text(upsert_sql), batch)

        logger.info("Wrote %s recipe tags to PostgreSQL", len(records))

    except Exception as e:
        logger.error("Failed to write tags to PostgreSQL: %s", e, exc_info=True)