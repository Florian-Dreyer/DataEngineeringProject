# 🍽️ Food.com Recipe Analytics Pipeline

> IS3107 Data Engineering — AY2025/2026 Semester 2

- **Google Trends integration** — Extract search interest data for recipe keywords to analyze popularity trends

## Requirements

- **Python 3.10+** (tested with Python 3.11)
- Dependencies: `pip install -e .` or install from `pyproject.toml`

## Google Trends Extraction

The pipeline includes Google Trends data extraction for analyzing search interest patterns of recipe-related keywords. This provides insights into recipe popularity trends over time.

### Features ✅
- **30 Cuisine Keywords**: Comprehensive coverage of global cuisines and food categories
- **Monthly Granularity**: Weekly Google Trends data aggregated to monthly summaries
- **Cross-Batch Normalization**: 'Pasta' anchor keyword ensures comparability across batches
- **Smart Batching**: Optimized API usage with anchor-based batching (5 keywords per batch)
- **Related Queries**: Top trending search queries for each keyword

### Data Structure
```python
# Output DataFrame columns:
# - keyword: str (one of 30 cuisine keywords)
# - date: datetime (monthly end dates)
# - interest_score: int (0-100, normalized across batches)
# - geo: str (geographic region)
# - related_queries: list[str] (rising search queries)
```

### Quick Start

```bash
# Install dependencies
pip install -e .

# Extract and save data to CSV
python extract_trends.py
```

**API Rate Limiting**: Google Trends API may return 429 errors during peak usage. The script includes exponential backoff to automatically retry failed batches. If you still encounter rate limiting:
  - Increase `sleep_time` parameter: `python3 extract_trends.py` (adjust in script if needed)
  - Reduce `batch_size`: Set to 2-3 instead of 4 in the script
  - Run during off-peak hours
  - The script will self-adjust sleep time when rate limits are detected

### Manual Usage

```bash
# Basic extraction (prints first 5 rows)
python -c "from foodcom_pipeline import extract_google_trends; df = extract_google_trends(); print(df.head())"

# Extract and save to CSV manually
python -c "
from foodcom_pipeline import extract_google_trends
df = extract_google_trends()
df.to_csv('data/trends_raw.csv', index=False)
print(f'Saved {df.shape[0]} rows to data/trends_raw.csv')
"
```

### Alternative Execution (if package install fails)

```bash
# Run directly from src directory
cd src
python -c "import sys; sys.path.insert(0, '.'); from foodcom_pipeline.extraction.trends import extract_google_trends; df = extract_google_trends(); print(df.head())"
```

### Configuration

- **Keywords**: Edit `config/trends_keywords.txt` to specify keywords (one per line). When Food.com recipe data becomes available, keywords should be automatically extracted from recipe ingredients and dish names to replace this manual file.
- **Parameters**: Modify batch size, sleep time, timeframe in the function call
- **Output**: DataFrame with columns `keyword`, `date`, `interest_score`, `geo`, `related_queries`

**Note**: Google Trends API has rate limits. The extraction batches keywords (5 at a time) with sleep intervals to avoid throttling.

### Pipeline Integration

When the full batch layer is implemented, the flow will be:
```
extract_google_trends() → clean() → transform() → load_db()
```

**Note**: The `load_db()` function for PostgreSQL insertion is planned but not yet implemented. Keywords for trends analysis should be automatically extracted from Food.com recipe data rather than manually maintained.