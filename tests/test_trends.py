"""Tests for Google Trends extraction."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from foodcom_pipeline.extraction.trends import extract_google_trends, load_keywords_from_file


class TestLoadKeywordsFromFile:
    """Test keyword loading functionality."""

    def test_load_valid_file(self):
        """Test loading keywords from a valid file."""
        content = """# Comment
pizza
pasta
# Another comment
sushi
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()

            keywords = load_keywords_from_file(f.name)

        Path(f.name).unlink()  # Cleanup

        assert keywords == ["pizza", "pasta", "sushi"]

    def test_load_empty_file(self):
        """Test loading from empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()

            keywords = load_keywords_from_file(f.name)

        Path(f.name).unlink()

        assert keywords == []

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_keywords_from_file("nonexistent.txt")


class TestExtractGoogleTrends:
    """Test Google Trends extraction."""

    @patch("foodcom_pipeline.extraction.trends.TrendReq")
    def test_extract_success(self, mock_trend_req):
        """Test successful extraction with mocked API."""
        """Test successful extraction with mocked API."""
        # Create mock data
        mock_pytrends = Mock()
        mock_trend_req.return_value = mock_pytrends

        # Mock interest_over_time
        interest_data = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=3, freq="ME"),  # Monthly end dates
            "pizza": [50, 60, 70],
            "pasta": [40, 50, 60],
            "sushi": [30, 40, 50],
            "tacos": [20, 30, 40],
            "curry": [25, 35, 45]
        })
        mock_pytrends.interest_over_time.return_value = interest_data

        # Mock related queries
        mock_pytrends.related_queries.return_value = {
            "pizza": {
                "rising": pd.DataFrame({
                    "query": ["margherita pizza", "pepperoni pizza", "pizza recipe"],
                    "value": [100, 80, 60]
                })
            },
            "pasta": {
                "rising": pd.DataFrame({
                    "query": ["pasta carbonara", "pasta salad"],
                    "value": [90, 70]
                })
            },
            "sushi": {
                "rising": pd.DataFrame({
                    "query": ["sushi near me", "sushi burrito"],
                    "value": [85, 65]
                })
            },
            "tacos": {
                "rising": pd.DataFrame({
                    "query": ["taco bell", "fish tacos"],
                    "value": [80, 60]
                })
            },
            "curry": {
                "rising": pd.DataFrame({
                    "query": ["curry recipe", "chicken curry"],
                    "value": [75, 55]
                })
            }
        }

        # Create temp keywords file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("pizza\npasta\nsushi\ntacos\ncurry\nsalad\n")
            f.flush()

            result = extract_google_trends(
                keywords_file=f.name,
                batch_size=4,  # Updated to match new default
                sleep_time=0  # No sleep for test
            )

        Path(f.name).unlink()

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["date", "interest_score", "keyword", "geo", "related_queries"]

        # Check data
        assert len(result) == 15  # 3 dates * 5 keywords
        assert result["keyword"].nunique() == 5
        assert all(result["geo"] == "global")

        # Check related queries
        pizza_rows = result[result["keyword"] == "pizza"]
        assert all(len(rq) == 3 for rq in pizza_rows["related_queries"])

    @patch("foodcom_pipeline.extraction.trends.TrendReq")
    def test_extract_no_keywords(self, mock_trend_req):
        """Test extraction with no keywords."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Only comments\n")
            f.flush()

            with pytest.raises(ValueError, match="No keywords found"):
                extract_google_trends(keywords_file=f.name)

        Path(f.name).unlink()

    @patch("foodcom_pipeline.extraction.trends.TrendReq")
    def test_extract_api_error(self, mock_trend_req):
        """Test handling of API errors."""
        mock_pytrends = Mock()
        mock_trend_req.return_value = mock_pytrends
        mock_pytrends.build_payload.side_effect = Exception("API Error")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("pizza\n")
            f.flush()

            with pytest.raises(RuntimeError, match="No data retrieved"):
                extract_google_trends(keywords_file=f.name, sleep_time=0)

        Path(f.name).unlink()

    def test_keywords_file_not_found(self):
        """Test error when keywords file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            extract_google_trends(keywords_file="nonexistent.txt")