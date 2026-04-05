"""
Exports: extract_google_trends function
Imports from: .extraction module
Purpose: Defines the public API of the foodcom_pipeline package
"""

from .extraction import extract_google_trends

__all__ = ["extract_google_trends"]