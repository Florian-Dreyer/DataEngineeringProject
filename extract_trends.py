#!/usr/bin/env python3
"""
Main Entry Point for Google Trends Extraction

Calls: foodcom_pipeline.extraction.trends.extract_google_trends()
Purpose: User-facing script that extracts data and saves to CSV
Dependencies: Adds src to Python path for imports
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from foodcom_pipeline.extraction.trends import extract_google_trends

def main():
    print("Extracting Google Trends data...")

    try:
        # Extract the data
        df = extract_google_trends()

        # Save to CSV
        output_path = os.path.join("data", "trends_raw.csv")
        df.to_csv(output_path, index=False)

        print("Saved {} rows, {} columns to {}".format(df.shape[0], df.shape[1], output_path))
        print("Keywords extracted: {}".format(df['keyword'].nunique()))
        print("Date range: {} to {}".format(df['date'].min(), df['date'].max()))

        # Show sample
        print("\nSample data:")
        print(df.head())

    except Exception as e:
        print("Error: {}".format(e))
        sys.exit(1)

if __name__ == "__main__":
    main()