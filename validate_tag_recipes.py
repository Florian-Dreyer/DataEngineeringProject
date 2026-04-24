#!/usr/bin/env python3
"""Validation script for tag_recipe_regex."""

import sys
sys.path.insert(0, 'src')

from foodcom_pipeline.batch.tag_recipes import tag_recipe_regex, TAG_DIMENSION_MAP
from foodcom_pipeline.utils.recipe_terms import canonicalize_recipe_term

test_cases = [
    'Creamy One-Pot Garlic Chicken Pasta',
    'Birria Tacos',
    'Banana Bread',
    'Sheet Pan Gnocchi with Vegetables',
]

print('Validation test for tag_recipe_regex:')
print('=' * 60)

for name in test_cases:
    print('')
    print('Recipe: {}'.format(name))
    print('  canonical_term: {}'.format(repr(canonicalize_recipe_term(name))))
    
    matches = tag_recipe_regex(name, '')
    print('  Tags found:')
    for tag, matched_term, tag_dimension in matches:
        print('    - tag={}, matched_term={}, tag_dimension={}'.format(
            repr(tag), repr(matched_term), repr(tag_dimension)))
    
    if not matches:
        print('    (no tags matched)')

print('')
print('=' * 60)
print('Validation complete!')