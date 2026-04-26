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

from foodcom_pipeline.batch.extract import (
    STAGING_DIR,
    atomic_parquet,
    AI_MODE_TERM_SCORES_STAGING,
    TRENDS_NORMALISED_STAGING,
)

from foodcom_pipeline.utils.recipe_terms import canonicalize_recipe_term

CLEANED_STAGING = STAGING_DIR / "recipes_clean.parquet"
TAGS_STAGING = STAGING_DIR / "recipe_tags.parquet"
SIGNAL_TAGS_STAGING = STAGING_DIR / "signal_tags.parquet"
RECIPE_TERM_INDEX_STAGING = STAGING_DIR / "recipe_term_index.parquet"
EXTERNAL_TERMS_STAGING = STAGING_DIR / "external_recipe_terms.parquet"
POSTGRES_CONN = os.getenv(
    "FOODCOM_POSTGRES_CONN", "postgresql://user:password@postgres:5432/foodcom"
)

MAX_LLM_RECIPES = 10
GEMINI_MODEL = "gemini-2.5-flash"

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

# Tag dimension mapping - categorizes tags into semantic groups
TAG_DIMENSION_MAP = {
    # Protein / base
    "chicken": "protein_base",
    "beef": "protein_base",
    "pork": "protein_base",
    "seafood": "protein_base",
    "vegetarian": "protein_base",
    "egg": "protein_base",
    
    # Starches / core formats
    "pasta": "dish_format",
    "rice": "dish_format",
    "bread_baked": "dish_format",
    
    # Dish formats
    "soup_stew": "dish_format",
    "salad": "dish_format",
    "sandwich_wrap": "dish_format",
    "casserole_bake": "dish_format",
    "stir_fry": "dish_format",
    "tacos_handheld": "dish_format",
    "bowl": "dish_format",
    
    # Occasion / type
    "breakfast": "occasion",
    "dessert": "occasion",
    "baking": "method",
    
    # Cuisine / flavor families
    "italian": "cuisine",
    "mexican": "cuisine",
    "korean": "cuisine",
    "japanese": "cuisine",
    "thai": "cuisine",
    "indian": "cuisine",
    "mediterranean": "cuisine",
    
    # Intent / style
    "healthy": "intent",
    "comfort_food": "intent",
    "quick_easy": "intent",
}

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


def _normalize_term(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r"\b(and|with|style)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -/")
    return text


def _canonicalize_recipe_term(name: str) -> str:
    """
    Normalize the recipe name into a comparable canonical dish phrase.
    Intended to align more closely with ai_mode dish terms.
    """
    text = _normalize_term(name)
    text = re.sub(r"\b(dinner|lunch|breakfast|meal|ideas|idea)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -/")
    return text


def _get_recipe_text(name: str, ingredients: str) -> str:
    return f"{name} {ingredients}".strip()


# ─────────────────────────────────────────────────────────────────────────
# Stage 1: Regex tagging
# ─────────────────────────────────────────────────────────────────────────

def tag_recipe_regex(name: str, ingredients: str) -> set[tuple[str, str, str]]:
    """
    Apply regex matching to normalized recipe name + ingredients.
    Returns a set of (tag, matched_term, tag_dimension) tuples.
    """
    raw_text = _get_recipe_text(name, ingredients)
    normalized_text = _normalize_text(raw_text)

    matched_pairs: set[tuple[str, str, str]] = set()

    for tag, patterns in TAG_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(normalized_text)
            if match:
                matched_term = _normalize_term(match.group())
                tag_dimension = TAG_DIMENSION_MAP.get(tag, "unknown")
                matched_pairs.add((tag, matched_term, tag_dimension))
                break

    return matched_pairs


# ─────────────────────────────────────────────────────────────────────────
# Stage 2: LLM tagging
# ─────────────────────────────────────────────────────────────────────────

_VALID_TAGS: frozenset[str] = frozenset(TAG_DICT.keys())

_LLM_PROMPT_TEMPLATE = """You are a food recipe classifier. Given a recipe name and its ingredients, return only the tags that apply from the list below. Output a comma-separated list of tag names only — no explanations, no extra text.

Valid tags:
{tags}

Recipe name: {name}
Ingredients: {ingredients}

Tags:"""


def tag_recipe_llm(name: str, ingredients: str, api_key: str) -> set[str]:
    """
    Use Gemini to assign tags from TAG_DICT to a recipe that regex under-tagged.
    Returns a set of valid tag strings (subset of TAG_DICT keys).
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    prompt = _LLM_PROMPT_TEMPLATE.format(
        tags=", ".join(sorted(_VALID_TAGS)),
        name=name,
        ingredients=ingredients[:500],
    )

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        raw = response.text.strip()
    except Exception as exc:
        logger.warning("Gemini call failed for %r: %s", name, exc)
        return set()

    tags: set[str] = set()
    for part in raw.split(","):
        token = part.strip().lower().replace(" ", "_")
        if token in _VALID_TAGS:
            tags.add(token)
        elif token.replace("_", " ") in _VALID_TAGS:
            tags.add(token.replace("_", " "))

    return tags


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
        canonical_term = _canonicalize_recipe_term(name)

        ingredients = ""
        for col in ["ingredients_canonical_normalized", "ingredients_normalized", "ingredients"]:
            if col in cleaned_df.columns and pd.notna(row[col]):
                ingredients = str(row[col]).strip()
                break

        matched_pairs = tag_recipe_regex(name, ingredients)

        for tag, matched_term, tag_dimension in matched_pairs:
            tags_list.append(
                {
                    "recipe_id": recipe_id,
                    "raw_term": name,
                    "canonical_term": canonicalize_recipe_term(name),
                    "tag": tag,
                    "tag_dimension": tag_dimension,
                    "matched_term": matched_term,
                    "tagging_method": "regex",
                    "source": "foodcom",
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

    weakly_tagged_ids = {
        recipe_id for recipe_id in cleaned_df["id"].tolist()
        if tag_counts_by_recipe.get(recipe_id, 0) <= 1
    }

    untagged_recipes = cleaned_df[cleaned_df["id"].isin(weakly_tagged_ids)].copy()

    llm_tag_count = 0
    llm_skipped_count = 0

    if len(untagged_recipes) > 0:
        api_key = os.environ.get("GEMINI_API_KEY", "")

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

            existing_rows = {
                (row["recipe_id"], row["tag"], row["matched_term"], row["canonical_term"])
                for row in tags_list
            }

            for _, row in recipes_to_tag.iterrows():
                recipe_id = row["id"]
                name = str(row.get("name", "")).strip()
                canonical_term = _canonicalize_recipe_term(name)

                ingredients = ""
                for col in ["ingredients_canonical_normalized", "ingredients_normalized", "ingredients"]:
                    if col in cleaned_df.columns and pd.notna(row[col]):
                        ingredients = str(row[col]).strip()
                        break

                llm_tags = tag_recipe_llm(name, ingredients, api_key)
                fallback_term = canonical_term

                for tag in llm_tags:
                    dedupe_key = (recipe_id, tag, fallback_term, canonical_term)
                    if dedupe_key in existing_rows:
                        continue

                    tags_list.append(
                        {
                            "recipe_id": recipe_id,
                            "raw_term": name,
                            "canonical_term": canonicalize_recipe_term(name),
                            "tag": tag,
                            "tag_dimension": TAG_DIMENSION_MAP.get(tag, "unknown"),
                            "matched_term": fallback_term,
                            "tagging_method": "llm",
                            "source": "foodcom",
                        }
                    )
                    existing_rows.add(dedupe_key)
                    llm_tag_count += 1

    logger.info("LLM tagging added %s tags", llm_tag_count)

    _TAG_COLS = [
        "recipe_id", "raw_term", "canonical_term", "tag",
        "tag_dimension", "matched_term", "tagging_method", "source",
    ]

    if not tags_list:
        logger.warning("No tags generated")
        tags_df = pd.DataFrame(columns=_TAG_COLS)
    else:
        tags_df = pd.DataFrame(tags_list).drop_duplicates(
            subset=["recipe_id", "tag", "matched_term", "canonical_term", "source"]
        )

    # Ensure exact schema: add missing columns as None, enforce column order.
    for col in _TAG_COLS:
        if col not in tags_df.columns:
            tags_df[col] = None
    tags_df = tags_df[_TAG_COLS]

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
                        recipe_id      INTEGER NOT NULL,
                        raw_term       TEXT,
                        canonical_term TEXT,
                        tag            TEXT    NOT NULL,
                        tag_dimension  TEXT,
                        matched_term   TEXT    NOT NULL,
                        tagging_method TEXT,
                        source         TEXT    NOT NULL
                    )
                    """
                )
            )

            # Add new columns if they don't exist (backward-compat for pre-existing tables)
            for col, col_type in [
                ("raw_term", "TEXT"),
                ("canonical_term", "TEXT"),
                ("tag_dimension", "TEXT"),
                ("tagging_method", "TEXT"),
            ]:
                conn.execute(
                    text(f"ALTER TABLE recipe_tags ADD COLUMN IF NOT EXISTS {col} {col_type}")
                )

        tags_df = tags_df.drop_duplicates(
            subset=["recipe_id", "tag", "matched_term", "canonical_term", "source"]
        )
        records = tags_df.to_dict(orient="records")

        upsert_sql = """
            INSERT INTO recipe_tags (recipe_id, raw_term, canonical_term, tag, tag_dimension, matched_term, tagging_method, source)
            VALUES (:recipe_id, :raw_term, :canonical_term, :tag, :tag_dimension, :matched_term, :tagging_method, :source)
            ON CONFLICT DO NOTHING
        """

        batch_size = 5000
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            with engine.begin() as conn:
                conn.execute(text(upsert_sql), batch)

        logger.info("Wrote %s recipe tags to PostgreSQL", len(records))

    except Exception as e:
        logger.error("Failed to write tags to PostgreSQL: %s", e, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────
# Signal tagging: tag AI Mode terms + Google Trends related queries
# ─────────────────────────────────────────────────────────────────────────

def tag_signals(**context) -> None:
    """
    Apply regex tagging to AI Mode term scores and Google Trends related queries.
    Writes results to signal_tags.parquet and PostgreSQL signal_tags table.
    """
    logger.info("Starting signal tagging task...")

    terms: list[dict] = []

    if AI_MODE_TERM_SCORES_STAGING.is_file():
        ai_df = pd.read_parquet(AI_MODE_TERM_SCORES_STAGING)
        for _, row in ai_df.iterrows():
            terms.append({
                "term": str(row["term"]).strip(),
                "data_source": "ai_mode",
            })
        logger.info("Loaded %d AI Mode terms", len(ai_df))
    else:
        logger.warning("AI Mode term scores not found: %s", AI_MODE_TERM_SCORES_STAGING)

    if TRENDS_NORMALISED_STAGING.is_file():
        trends_df = pd.read_parquet(TRENDS_NORMALISED_STAGING)
        for _, row in trends_df.iterrows():
            terms.append({
                "term": str(row["related_query"]).strip(),
                "data_source": "trends",
            })
        logger.info("Loaded %d Trends related queries", len(trends_df))
    else:
        logger.warning("Trends normalised staging not found: %s", TRENDS_NORMALISED_STAGING)

    if not terms:
        logger.warning("No signal terms to tag — skipping")
        return

    rows: list[dict] = []
    for item in terms:
        term = item["term"]
        canonical_term = _canonicalize_recipe_term(term)
        matched_pairs = tag_recipe_regex(term, "")
        for tag, matched_term, _tag_dimension in matched_pairs:
            rows.append({
                "term": term,
                "canonical_term": canonical_term,
                "data_source": item["data_source"],
                "tag": tag,
                "matched_term": matched_term,
            })

    if not rows:
        logger.warning("No tags produced for any signal terms")
        signal_df = pd.DataFrame(
            columns=["term", "canonical_term", "data_source", "tag", "matched_term"]
        )
    else:
        signal_df = pd.DataFrame(rows).drop_duplicates(
            subset=["term", "data_source", "tag"]
        )

    logger.info("Signal tagging produced %d rows", len(signal_df))

    SIGNAL_TAGS_STAGING.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(signal_df, SIGNAL_TAGS_STAGING)
    logger.info("Signal tags staged to %s", SIGNAL_TAGS_STAGING)

    if not signal_df.empty:
        _write_signal_tags_to_db(signal_df)

    context["ti"].xcom_push(key="signal_tag_count", value=len(signal_df))


def _write_signal_tags_to_db(signal_df: pd.DataFrame) -> None:
    """Write signal tags to PostgreSQL signal_tags table."""
    try:
        engine = create_engine(POSTGRES_CONN)

        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS signal_tags (
                    term           TEXT NOT NULL,
                    canonical_term TEXT,
                    data_source    TEXT NOT NULL,
                    tag            TEXT NOT NULL,
                    matched_term   TEXT NOT NULL
                )
            """))
            conn.execute(text("""
                ALTER TABLE signal_tags ADD COLUMN IF NOT EXISTS canonical_term TEXT
            """))

        records = signal_df.to_dict(orient="records")
        upsert_sql = """
            INSERT INTO signal_tags (term, canonical_term, data_source, tag, matched_term)
            VALUES (:term, :canonical_term, :data_source, :tag, :matched_term)
            ON CONFLICT DO NOTHING
        """

        batch_size = 5000
        for i in range(0, len(records), batch_size):
            with engine.begin() as conn:
                conn.execute(text(upsert_sql), records[i : i + batch_size])

        logger.info("Wrote %d signal tags to PostgreSQL", len(records))

    except Exception as exc:
        logger.error("Failed to write signal tags to PostgreSQL: %s", exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────
# Recipe term index: one row per recipe, tags aggregated to recipe level
# ─────────────────────────────────────────────────────────────────────────

_INDEX_COLS = [
    "recipe_id",
    "recipe_name",
    "raw_term",
    "canonical_term",
    "tags",
    "tag_dimensions",
    "ingredients_summary",
]


def build_recipe_term_index(**context) -> None:
    """
    Aggregate recipe_tags back to one row per recipe and join with cleaned
    recipe metadata to produce recipe_term_index.parquet.

    Columns: recipe_id, recipe_name, raw_term, canonical_term,
             tags, tag_dimensions, ingredients_summary
    """
    logger.info("Building recipe term index...")

    # ── 1. Load cleaned recipes ───────────────────────────────────────────
    if not CLEANED_STAGING.exists():
        logger.warning("Cleaned staging file not found: %s", CLEANED_STAGING)
        return

    recipes_df = pd.read_parquet(CLEANED_STAGING)
    logger.info("Loaded %d cleaned recipes", len(recipes_df))

    # ── 2. Load recipe tags (may not exist on first run) ──────────────────
    if TAGS_STAGING.exists():
        tags_df = pd.read_parquet(TAGS_STAGING)
        logger.info("Loaded %d recipe tag rows", len(tags_df))
    else:
        logger.warning("recipe_tags.parquet not found — index will have empty tags")
        tags_df = pd.DataFrame(
            columns=["recipe_id", "raw_term", "canonical_term", "tag", "tag_dimension"]
        )

    # ── 3. Aggregate tags per recipe ──────────────────────────────────────
    if not tags_df.empty:
        tag_agg = (
            tags_df
            .groupby("recipe_id", sort=False)
            .agg(
                raw_term=("raw_term", "first"),
                canonical_term=("canonical_term", "first"),
                tags=("tag", lambda s: "|".join(sorted(s.dropna().unique()))),
                tag_dimensions=("tag_dimension", lambda s: "|".join(sorted(s.dropna().unique()))),
            )
            .reset_index()
        )
    else:
        tag_agg = pd.DataFrame(
            columns=["recipe_id", "raw_term", "canonical_term", "tags", "tag_dimensions"]
        )

    # ── 4. Build ingredients_summary (top 10, pipe-separated) ─────────────
    ingr_col = (
        "ingredients_canonical_normalized"
        if "ingredients_canonical_normalized" in recipes_df.columns
        else "ingredients_normalized"
    )

    def _ingredients_summary(value) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        parts = [p.strip() for p in value.split("|") if p.strip()]
        return "|".join(parts[:10]) if parts else None

    recipe_base = recipes_df[["id", "name"]].copy()
    if ingr_col in recipes_df.columns:
        recipe_base["ingredients_summary"] = recipes_df[ingr_col].apply(_ingredients_summary)
    else:
        recipe_base["ingredients_summary"] = None

    recipe_base = recipe_base.rename(columns={"id": "recipe_id", "name": "recipe_name"})

    # ── 5. Join: left-join tags onto every recipe ──────────────────────────
    index_df = recipe_base.merge(tag_agg, on="recipe_id", how="left")

    # For recipes with no tag rows, derive raw_term / canonical_term from name
    no_raw = index_df["raw_term"].isna()
    index_df.loc[no_raw, "raw_term"] = index_df.loc[no_raw, "recipe_name"]
    index_df.loc[no_raw, "canonical_term"] = (
        index_df.loc[no_raw, "recipe_name"].apply(_canonicalize_recipe_term)
    )

    # ── 6. Enforce exact schema / column order ─────────────────────────────
    for col in _INDEX_COLS:
        if col not in index_df.columns:
            index_df[col] = None
    index_df = index_df[_INDEX_COLS]

    # ── 7. Write parquet ───────────────────────────────────────────────────
    RECIPE_TERM_INDEX_STAGING.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(index_df, RECIPE_TERM_INDEX_STAGING)
    logger.info(
        "Recipe term index written to %s (%d rows)", RECIPE_TERM_INDEX_STAGING, len(index_df)
    )

    # ── 8. Validation output ───────────────────────────────────────────────
    no_tags_mask = index_df["tags"].isna() | (index_df["tags"] == "")
    no_tags_count = int(no_tags_mask.sum())

    logger.info("=== recipe_term_index validation ===")
    logger.info("Row count          : %d", len(index_df))
    logger.info("Recipes with no tags: %d / %d", no_tags_count, len(index_df))

    sample = index_df.sample(min(10, len(index_df)), random_state=42)
    logger.info("--- sample 10 rows ---")
    for _, row in sample.iterrows():
        logger.info(
            "  id=%-8s  name=%-40s  canonical=%-30s  tags=%s",
            row["recipe_id"],
            str(row["recipe_name"])[:40],
            str(row["canonical_term"])[:30],
            row["tags"],
        )

    top_terms = (
        index_df["canonical_term"]
        .dropna()
        .loc[lambda s: s != ""]
        .value_counts()
        .head(20)
    )
    logger.info("--- top 20 canonical terms ---")
    for term, count in top_terms.items():
        logger.info("  %-40s  %d", term, count)
    logger.info("=== end validation ===")

    if context.get("ti"):
        context["ti"].xcom_push(key="recipe_term_index_count", value=len(index_df))
        context["ti"].xcom_push(key="recipes_without_tags", value=no_tags_count)


# ─────────────────────────────────────────────────────────────────────────
# External recipe terms: normalize Google AI Mode outputs
# ─────────────────────────────────────────────────────────────────────────

_EXT_COLS = [
    "source",
    "raw_term",
    "canonical_term",
    "source_score",
    "raw_frequency",
    "fetched_date",
    "tags",
    "tag_dimensions",
]

# Term-level patterns that indicate conversational AI residue, not dish names.
# Block-level filtering already runs in ai_mode.fetch_ai_mode_blocks, but
# individual terms can still carry these fragments after splitting.
_AI_RESIDUE = re.compile(
    r"would you like"
    r"|here are some"
    r"|let me know"
    r"|i hope"
    r"|feel free"
    r"|^for example\b"
    r"|^to (make|cook|prepare)\b"
    r"|^if you\b"
    r"|^you (can|might|could|should)\b"
    r"|^i (can|will|would|think)\b"
    r"|\?"
    r"|\bclick\b"
    r"|\bsee more\b"
    r"|\bread more\b"
    r"|\bsign up\b"
    r"|\bsubscribe\b",
    flags=re.IGNORECASE,
)

_MAX_TERM_CHARS = 60   # anything longer is almost certainly a sentence fragment
_MIN_ALPHA_RATIO = 0.6  # fraction of characters that must be alphabetic

# Non-food domain keywords — Trends queries matching these are always dropped.
_TRENDS_BLOCKLIST = re.compile(
    r"\bpython\b"
    r"|\binsurance\b"
    r"|\blaptops?\b"
    r"|\breal estate\b"
    r"|\bflights?\b"
    r"|\bweather\b"
    r"|\bstock market\b"
    r"|\bused cars?\b"
    r"|\bpc games?\b"
    r"|\bcoupons?\b"
    r"|\bnovels?\b"
    r"|\bgame reviews?\b"
    r"|\bcartoons?\b",
    flags=re.IGNORECASE,
)

# Recipe intent signal — a Trends query must pass this OR match a food taxonomy keyword.
_RECIPE_KEYWORDS = re.compile(r"\brecipes?\b", re.IGNORECASE)


def _has_food_intent(term: str) -> bool:
    """
    Return True if *term* is recipe/food-adjacent.
    Passes if the term contains "recipe"/"recipes" OR matches a food
    taxonomy keyword from TAG_DICT.
    """
    if _RECIPE_KEYWORDS.search(term):
        return True
    term_lower = term.lower()
    # Use compiled TAG_PATTERNS (already have word-boundary anchors)
    for patterns in TAG_PATTERNS.values():
        for pat in patterns:
            if pat.search(term_lower):
                return True
    return False


def _is_valid_external_term(term: str) -> bool:
    """
    Return True if *term* looks like a dish / recipe search phrase.
    Rejects conversational residue, prose fragments, and non-food junk.
    """
    if not term or len(term) < 3:
        return False
    if len(term) > _MAX_TERM_CHARS:
        return False
    if _AI_RESIDUE.search(term):
        return False
    alpha_chars = sum(c.isalpha() for c in term)
    if alpha_chars / len(term) < _MIN_ALPHA_RATIO:
        return False
    return True


def _aggregate_signal_tags(signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group signal_tags rows by term, producing pipe-joined tags and
    tag_dimensions (derived from TAG_DIMENSION_MAP, not stored in signal_tags).
    Returns a DataFrame with columns: term, tags, tag_dimensions.
    """
    if signal_df.empty:
        return pd.DataFrame(columns=["term", "tags", "tag_dimensions"])

    def _dims(tag_series) -> str:
        dims = sorted({
            TAG_DIMENSION_MAP.get(t, "unknown")
            for t in tag_series.dropna().unique()
        })
        return "|".join(dims)

    return (
        signal_df
        .groupby("term", sort=False)
        .agg(
            tags=("tag", lambda s: "|".join(sorted(s.dropna().unique()))),
            tag_dimensions=("tag", _dims),
        )
        .reset_index()
    )


def _enrich_with_tags(df: pd.DataFrame, tag_agg: pd.DataFrame) -> pd.DataFrame:
    """Left-join tag_agg onto df on raw_term, filling tags/tag_dimensions columns."""
    if tag_agg.empty or df.empty:
        return df
    merged = df.merge(tag_agg, left_on="raw_term", right_on="term", how="left", suffixes=("", "_sig"))
    for col in ("tags", "tag_dimensions"):
        sig_col = f"{col}_sig"
        if sig_col in merged.columns:
            merged[col] = merged[sig_col].where(merged[sig_col].notna(), merged[col])
            merged = merged.drop(columns=[sig_col])
    return merged.drop(columns=["term"], errors="ignore")


def _build_ai_mode_rows(scores_df: pd.DataFrame) -> tuple[list[dict], int, int]:
    """
    Convert ai_mode_term_scores rows into external term dicts.
    Returns (rows, dropped_residue, dropped_canonical).
    """
    rows: list[dict] = []
    dropped_residue = 0
    dropped_canonical = 0

    for _, row in scores_df.iterrows():
        raw_term = str(row["term"]).strip()
        if not _is_valid_external_term(raw_term):
            dropped_residue += 1
            logger.debug("AI Mode: dropping residue/junk %r", raw_term)
            continue
        canonical_term = canonicalize_recipe_term(raw_term)
        if not canonical_term:
            dropped_canonical += 1
            logger.debug("AI Mode: empty canonical for %r", raw_term)
            continue
        rows.append({
            "source": "google_ai",
            "raw_term": raw_term,
            "canonical_term": canonical_term,
            "source_score": row.get("normalised_score"),
            "raw_frequency": row.get("raw_frequency"),
            "fetched_date": row.get("fetched_date"),
            "tags": None,
            "tag_dimensions": None,
        })

    logger.info(
        "AI Mode: kept %d  |  dropped residue/junk %d  |  dropped empty canonical %d",
        len(rows), dropped_residue, dropped_canonical,
    )
    return rows, dropped_residue, dropped_canonical


def _build_trends_rows(trends_df: pd.DataFrame) -> tuple[list[dict], int, int, int]:
    """
    Convert google_trends_normalised rows into external term dicts.

    Filters:
      - Drop if matches _TRENDS_BLOCKLIST (non-food domain keywords)
      - Keep if contains recipe keyword OR matches food taxonomy keyword
      - Drop if canonical_term is empty after canonicalization

    Returns (rows, dropped_blocklist, dropped_intent, dropped_canonical).
    """
    rows: list[dict] = []
    dropped_blocklist = 0
    dropped_intent = 0
    dropped_canonical = 0

    # Deduplicate across seed/query_type: one row per related_query,
    # keeping the highest normalised_score.
    deduped = (
        trends_df
        .sort_values("normalised_score", ascending=False, na_position="last")
        .drop_duplicates(subset=["related_query"])
    )

    for _, row in deduped.iterrows():
        raw_term = str(row["related_query"]).strip()

        if _TRENDS_BLOCKLIST.search(raw_term):
            dropped_blocklist += 1
            logger.debug("Trends: blocklist drop %r", raw_term)
            continue

        if not _has_food_intent(raw_term):
            dropped_intent += 1
            logger.debug("Trends: no food intent %r", raw_term)
            continue

        canonical_term = canonicalize_recipe_term(raw_term)
        if not canonical_term:
            dropped_canonical += 1
            logger.debug("Trends: empty canonical for %r", raw_term)
            continue

        rows.append({
            "source": "google_trends",
            "raw_term": raw_term,
            "canonical_term": canonical_term,
            "source_score": row.get("normalised_score"),
            "raw_frequency": row.get("raw_value"),
            "fetched_date": row.get("fetched_date"),
            "tags": None,
            "tag_dimensions": None,
        })

    logger.info(
        "Trends: kept %d  |  dropped blocklist %d  |  dropped no-intent %d  |  dropped empty canonical %d",
        len(rows), dropped_blocklist, dropped_intent, dropped_canonical,
    )
    return rows, dropped_blocklist, dropped_intent, dropped_canonical


def build_external_recipe_terms(**context) -> None:
    """
    Normalize Google AI Mode and Google Trends outputs into
    external_recipe_terms.parquet.

    Columns: source, raw_term, canonical_term, source_score, raw_frequency,
             fetched_date, tags, tag_dimensions

    source is 'google_ai' or 'google_trends'.  raw_term is kept for
    explainability.  Tags are enriched from signal_tags.parquet when
    available; tag_dimensions are derived from TAG_DIMENSION_MAP.
    """
    logger.info("Building external recipe terms (AI Mode + Google Trends)...")

    all_rows: list[dict] = []
    total_dropped = 0

    # ── 1. AI Mode ────────────────────────────────────────────────────────
    if AI_MODE_TERM_SCORES_STAGING.is_file():
        scores_df = pd.read_parquet(AI_MODE_TERM_SCORES_STAGING)
        logger.info("Loaded %d AI Mode term scores", len(scores_df))
        ai_rows, dr, dc = _build_ai_mode_rows(scores_df)
        all_rows.extend(ai_rows)
        total_dropped += dr + dc
    else:
        logger.warning("AI Mode term scores not found: %s", AI_MODE_TERM_SCORES_STAGING)

    # ── 2. Google Trends ──────────────────────────────────────────────────
    if TRENDS_NORMALISED_STAGING.is_file():
        trends_df = pd.read_parquet(TRENDS_NORMALISED_STAGING)
        logger.info("Loaded %d Google Trends normalised rows", len(trends_df))
        tr_rows, db, di, dc = _build_trends_rows(trends_df)
        all_rows.extend(tr_rows)
        total_dropped += db + di + dc
    else:
        logger.warning("Trends normalised staging not found: %s", TRENDS_NORMALISED_STAGING)

    if not all_rows:
        logger.warning("No valid external recipe terms produced from any source")
        ext_df = pd.DataFrame(columns=_EXT_COLS)
    else:
        ext_df = pd.DataFrame(all_rows)

    # ── 3. Load signal_tags and enrich per source ─────────────────────────
    if SIGNAL_TAGS_STAGING.exists() and not ext_df.empty:
        signal_full = pd.read_parquet(SIGNAL_TAGS_STAGING)

        for source_val, data_source_val in (
            ("google_ai", "ai_mode"),
            ("google_trends", "trends"),
        ):
            sig_subset = signal_full[signal_full["data_source"] == data_source_val].copy()
            tag_agg = _aggregate_signal_tags(sig_subset)
            mask = ext_df["source"] == source_val
            if mask.any() and not tag_agg.empty:
                enriched = _enrich_with_tags(ext_df[mask].copy(), tag_agg)
                ext_df = pd.concat(
                    [enriched, ext_df[~mask]], ignore_index=True
                )
    else:
        if not SIGNAL_TAGS_STAGING.exists():
            logger.info("signal_tags.parquet not found — tags will be empty")

    # ── 4. Deduplicate within each source on canonical_term ───────────────
    ext_df = (
        ext_df
        .sort_values("source_score", ascending=False, na_position="last")
        .drop_duplicates(subset=["source", "canonical_term"])
        .reset_index(drop=True)
    )

    # ── 5. Enforce schema ─────────────────────────────────────────────────
    for col in _EXT_COLS:
        if col not in ext_df.columns:
            ext_df[col] = None
    ext_df = ext_df[_EXT_COLS]

    # ── 6. Write parquet ──────────────────────────────────────────────────
    EXTERNAL_TERMS_STAGING.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(ext_df, EXTERNAL_TERMS_STAGING)
    logger.info(
        "External recipe terms written to %s (%d rows, %d dropped total)",
        EXTERNAL_TERMS_STAGING, len(ext_df), total_dropped,
    )

    # ── 7. Validation ─────────────────────────────────────────────────────
    _validate_external_terms(ext_df)

    if context.get("ti"):
        context["ti"].xcom_push(key="external_terms_count", value=len(ext_df))
        context["ti"].xcom_push(key="external_terms_dropped", value=total_dropped)


def _validate_external_terms(df: pd.DataFrame) -> None:
    """
    Log validation checks for external_recipe_terms.
    Also runs the Trends filter against a fixed set of known-good and
    known-bad examples so the logic can be verified even when those
    exact strings are absent from the live data.
    """
    logger.info("=== external_recipe_terms validation ===")
    logger.info("Row count: %d", len(df))

    by_source = df.groupby("source").size().to_dict()
    for src, n in sorted(by_source.items()):
        logger.info("  source=%-15s  rows=%d", src, n)

    # Check: no "would you like" anywhere
    wyl = int(df["raw_term"].str.contains("would you like", case=False, na=False).sum())
    logger.info("Rows containing 'would you like': %d  [%s]", wyl, "OK" if wyl == 0 else "FAIL")

    # Check: all canonical_terms non-empty
    empty_ct = int((df["canonical_term"].isna() | (df["canonical_term"] == "")).sum())
    logger.info("Rows with empty canonical_term: %d  [%s]", empty_ct, "OK" if empty_ct == 0 else "FAIL")

    # Top 20 by source_score
    logger.info("--- top 20 rows by source_score ---")
    top = df.nlargest(20, "source_score")
    for _, row in top.iterrows():
        logger.info(
            "  [%-13s]  raw=%-35s  canonical=%-28s  score=%+.3f",
            row["source"],
            str(row["raw_term"])[:35],
            str(row["canonical_term"])[:28],
            float(row["source_score"]) if pd.notna(row["source_score"]) else 0.0,
        )

    # ── Trends filter unit checks ─────────────────────────────────────────
    _validate_trends_filter()

    logger.info("=== end validation ===")


def _validate_trends_filter() -> None:
    """
    Run the Trends blocklist + food-intent filter against a fixed set of
    known-drop and known-keep examples and log PASS/FAIL for each.
    """
    should_drop = [
        "how to learn python",
        "car insurance quotes",
        "cheap flights",
        "stock market news",
    ]
    should_keep = [
        "banana bread recipe",
        "ramen recipes",
        "chicken recipes",
        "gluten free pizza crust recipe",
    ]

    logger.info("--- trends filter unit checks ---")
    all_pass = True

    for term in should_drop:
        blocked = bool(_TRENDS_BLOCKLIST.search(term))
        food = _has_food_intent(term)
        # A term is dropped if blocklisted OR has no food intent
        dropped = blocked or not food
        ok = dropped
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        logger.info("  SHOULD DROP  [%s]  %r  (blocklist=%s, food_intent=%s)", status, term, blocked, food)

    for term in should_keep:
        blocked = bool(_TRENDS_BLOCKLIST.search(term))
        food = _has_food_intent(term)
        kept = not blocked and food
        ok = kept
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        logger.info("  SHOULD KEEP  [%s]  %r  (blocklist=%s, food_intent=%s)", status, term, blocked, food)

    logger.info("Trends filter unit checks: %s", "ALL PASS" if all_pass else "SOME FAILED")