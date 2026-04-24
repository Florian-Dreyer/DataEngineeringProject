"""Recipe term canonicalization helpers."""

import re


# Common recipe modifiers to strip
# Note: Order matters - longer/multi-word patterns first
RECIPE_MODIFIERS = re.compile(
    r"\b("
    # Multi-word modifiers (must come first to match first)
    r"one-pot|one-pan|one pan|sheet-pan|sheet pan|slow-cooker|slow cooker|"
    r"instant-pot|instant pot|air-fryer|air fryer|"
    r"low-carb|keto|paleo|gluten-free|dairy-free|sugar-free|low-fat|low-sugar|"
    r"gluten free|dairy free|sugar free|low fat|low sugar|"
    r"30-minute|30 minute|15-minute|15 minute|45-minute|45 minute|"
    r"1-hour|1 hour|under 30|over 60|"
    r"no-bake|no bake|make-ahead|make ahead|meal-prep|meal prep|"
    r"freezer-friendly|freezer friendly|"
    r"kid-friendly|kid friendly|family-friendly|family friendly|"
    r"child-friendly|child friendly|"
    r"homemade|homestyle|"
    # Single word modifiers
    r"best|easy|quick|simple|fast|healthy|delicious|tasty|yummy|perfect|"
    r"ultimate|amazing|awesome|great|good|classic|traditional|"
    r"authentic|fresh|real|original|"
    r"weeknight|weekend|holiday|party|christmas|thanksgiving|easter|"
    r"summer|winter|spring|fall|autumn|"
    r"low-carb|keto|paleo|gluten-free|dairy-free|vegan|vegetarian|"
    r"spicy|hot|mild|creamy|crispy|crunchy|soft|chewy|fluffy|light|rich|"
    r"mini|miniature|small|bite-sized|individual|"
    r"stuffed|filled|topped|covered|"
    r"no-bake|raw|cooked|grilled|baked|roasted|fried|steamed|boiled|"
    r"instant|ready|minute|hour|"
    r"budget|cheap|expensive|fancy|elegant|"
    r"kid-friendly|family-friendly|child-friendly|"
    r"make-ahead|meal-prep|freezer-friendly|"
    r"gluten free|dairy free|sugar free|low fat|low sugar|"
    r"30-minute|15-minute|45-minute|1-hour|under 30|over 60|"
    # Additional standalone words from compound modifiers
    r"one|pot|pan|sheet|with|for|in|on"
    r")\b",
    re.IGNORECASE,
)

# Recipe type keywords to keep
RECIPE_TYPES = re.compile(
    r"\b("
    r"recipe|recipes|method|instructions|directions|"
    r"how to make|how to cook|how to prepare"
    r")\b",
    re.IGNORECASE,
)

# Common cooking methods to keep as they're descriptive
COOKING_METHODS = re.compile(
    r"\b("
    r"grilled|roasted|baked|fried|steamed|boiled|braised|stewed|"
    r"sauteed|saute|pan-fried|deep-fried|poached|simmered|"
    r"slow-cooked|crock-pot|pressure-cooked|air-fried|"
    r"browned|caramelized|charred|smoked"
    r")\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """
    Normalize text for recipe term processing.
    
    - Lowercase
    - Remove extra whitespace
    - Remove common punctuation
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Replace common punctuation with spaces
    text = re.sub(r"[/-]", " ", text)
    
    # Remove punctuation except spaces
    text = re.sub(r"[^\w\s]", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def strip_recipe_modifiers(text: str) -> str:
    """
    Strip common recipe modifiers from text.
    
    Removes words like 'best', 'easy', 'quick', 'one-pot', etc.
    that don't add semantic meaning to the core recipe term.
    """
    if not text:
        return ""
    
    # First normalize the text
    text = normalize_text(text)
    
    # Remove recipe type words
    text = RECIPE_TYPES.sub("", text)
    
    # Remove modifiers (but keep cooking methods)
    # We need to be careful not to remove cooking methods
    words = text.split()
    filtered_words = []
    
    for word in words:
        # Check if it's a cooking method - keep it
        if COOKING_METHODS.match(word):
            filtered_words.append(word)
        # Check if it's a modifier - skip it
        elif RECIPE_MODIFIERS.match(word):
            continue
        else:
            filtered_words.append(word)
    
    return " ".join(filtered_words)


def canonicalize_recipe_term(text: str) -> str:
    """
    Canonicalize a recipe search term.
    
    Applies full canonicalization:
    1. Normalize text
    2. Strip modifiers
    3. Clean up result
    
    Examples:
    - "best garlic chicken recipe" -> "garlic chicken"
    - "easy weeknight pasta recipes" -> "pasta"
    - "one-pot creamy garlic chicken" -> "garlic chicken"
    - "sheet pan gnocchi with vegetables" -> "gnocchi vegetables"
    - "banana bread recipe" -> "banana bread"
    """
    if not text:
        return ""
    
    # Apply full pipeline
    text = normalize_text(text)
    text = strip_recipe_modifiers(text)
    
    # Final cleanup - remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def run_tests():
    """Self-test function for canonicalization helpers."""
    test_cases = [
        # (input, expected_output)
        ("best garlic chicken recipe", "garlic chicken"),
        ("easy weeknight pasta recipes", "pasta"),
        ("one-pot creamy garlic chicken", "garlic chicken"),
        ("sheet pan gnocchi with vegetables", "gnocchi vegetables"),
        ("banana bread recipe", "banana bread"),
    ]
    
    print("Running recipe term canonicalization tests...")
    print("-" * 60)
    
    all_passed = True
    for input_text, expected in test_cases:
        result = canonicalize_recipe_term(input_text)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"{status} Input:    '{input_text}'")
        print(f"  Expected: '{expected}'")
        print(f"  Got:      '{result}'")
        print()
    
    print("-" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    
    return all_passed


if __name__ == "__main__":
    run_tests()