#!/usr/bin/env python3
"""
Standalone script to generate Google Trends keywords from recipe tags.

This script extracts unique recipe tags and saves them as keywords for the
Google Trends extraction pipeline. Can be run independently or integrated into
the Airflow DAG.

Usage:
    python generate_keywords_from_tags.py
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from foodcom_pipeline.batch.features import generate_recipe_tag_keywords


def main():
    print("Generating Google Trends keywords from recipe tags...")

    try:
        keywords_df = generate_recipe_tag_keywords()

        print(f"\n✓ Success!")
        print(f"  Generated {len(keywords_df)} unique recipe tags")
        print(f"  Sample: {list(keywords_df['keyword'].head(10).values)}")
        print(f"  Keywords saved to: config/trends_keywords.txt")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nThis script requires cleaned recipe data from the pipeline.")
        print("Run the Airflow DAG first to generate the staging files.")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
