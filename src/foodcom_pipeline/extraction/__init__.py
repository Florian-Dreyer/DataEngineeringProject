"""
Extraction Subpackage

Exports: extract_google_trends function
Imports from: .trends module
Purpose: Defines the public API of the extraction sub-package
"""

from .trends import extract_google_trends

__all__ = ["extract_google_trends"]