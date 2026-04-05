# Google Trends Extraction Design Log

## Overview
This log documents the design decisions, implementation choices, and development process for integrating Google Trends data extraction into the Food.com Lambda Architecture pipeline. The implementation prioritizes reliability and maintainability while handling the challenges of an unofficial API.

## Background & Requirements
The Food.com pipeline processes recipe reviews in real-time (VADER sentiment) and batch (DistilBERT sentiment, clustering, rating prediction). Google Trends extraction was added to provide external popularity signals for recipe keywords, enabling correlation analysis between search interest and recipe ratings/interactions.

**Key Requirements:**
- **Input**: List of keywords derived from recipes (ingredients, dish types, cuisines)
- **Processing**: Query Google Trends API with rate limiting and batching
- **Output**: Structured DataFrame with interest scores, timestamps, geography, and related queries
- **Integration**: Feed into batch layer cleaning and loading pipeline

## API Library Selection
**Decision**: Use pytrends (unofficial library) over direct HTTP requests or commercial scraping services.

**Alternatives Considered:**
- Direct HTTP requests to Google Trends API
- ScraperAPI or Oxylabs for proxy-based scraping
- Official Google Trends API (not publicly available)

**Rationale**: pytrends provides the most straightforward interface for interest-over-time, related queries, and regional data. Despite being unofficial and potentially unstable, it's the de facto standard for Google Trends automation. Direct HTTP would require reverse-engineering Google's authentication and request patterns, which proved complex and fragile in testing.

**Lesson Learned**: Unofficial APIs require robust error handling and monitoring. We implemented comprehensive exception handling to gracefully manage API failures without stopping the entire extraction process.

```python
# Key implementation for API resilience
try:
    pytrends.build_payload(kw_list=batch, timeframe=timeframe, geo=geo)
    interest_df = pytrends.interest_over_time()
except Exception as e:
    print(f"Error processing batch {batch}: {e}")
    continue  # Continue with next batch rather than failing completely
```

## Rate Limiting Strategy
**Decision**: Implement batch processing (5 keywords per batch) with configurable sleep intervals (default 5 seconds).

**Rationale**: Google Trends aggressively throttles requests. Batching reduces API calls while respecting limits. Sleep intervals prevent IP blocking. The batch size of 5 was determined through testing to be conservative yet effective.

**Lesson Learned**: Rate limiting is critical for API stability. The implementation includes configurable parameters to adapt to changing API limits without code changes.

```python
# Rate limiting implementation
for i in range(0, len(keywords), batch_size):
    batch = keywords[i:i + batch_size]
    # Process batch...
    if i + batch_size < len(keywords):
        time.sleep(sleep_time)  # Respect API limits
```

## Data Structure Design
**Decision**: Single DataFrame with columns: `keyword`, `date`, `interest_score`, `geo`, `related_queries`

**Rationale**:
- Normalized 0-100 interest scores (already provided by API)
- Weekly timestamps (Google's default granularity, suitable for trend analysis)
- Global geography initially (extensible to regional breakdown)
- Related queries as auxiliary features for trend analysis

**Lesson Learned**: Data structure design impacts downstream usability. The flat DataFrame format simplifies integration with pandas-based cleaning and transformation steps.

```python
# Data structure creation
kw_data = interest_df[[kw]].reset_index()
kw_data = kw_data.rename(columns={kw: "interest_score"})
kw_data["keyword"] = kw
kw_data["geo"] = geo if geo else "global"
# Add related queries...
```

## Configuration Management
**Decision**: External keywords file (`config/trends_keywords.txt`) with automatic path resolution.

**Rationale**: Keywords should be configurable without code changes. Path resolution works from any execution context, solving deployment issues encountered during development.

**Future Integration**: When Food.com recipe data is available, keywords should be automatically extracted from recipe ingredients and dish names to replace the manual `trends_keywords.txt` file. This ensures trends analysis is based on actual recipe content rather than manually curated lists.

**Implementation Plan**:
```python
# Future: Extract keywords from recipes
def extract_keywords_from_recipes(recipes_df: pd.DataFrame, top_n: int = 50) -> List[str]:
    # Parse ingredients and dish names
    # Count frequencies
    # Return top N keywords
    # Write to config/trends_keywords.txt
```

**Lesson Learned**: Configuration management is crucial for flexible deployment. The dynamic path resolution using `__file__` ensures the config file is found regardless of execution context.

```python
# Dynamic config path resolution
if keywords_file is None:
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "trends_keywords.txt"
    keywords_file = str(config_path)
```

## Error Handling Strategy
**Decision**: Continue processing on individual batch failures, raise exceptions only for complete failures.

**Rationale**: Network issues or API changes shouldn't stop the entire extraction. Partial data is better than no data for trend analysis.

**Lesson Learned**: Robust error handling prevents single failures from breaking the entire pipeline. This approach ensures maximum data collection even with intermittent API issues.

## Package Structure & Integration
**Decision**: `extraction/` module under `foodcom_pipeline/` with clean import structure.

**Rationale**: Follows the planned pipeline architecture. Extraction is separate from transformation and loading, enabling modular development.

**Lesson Learned**: Package structure impacts import reliability. The direct import in `__init__.py` ensures consistent access across different execution contexts.

```python
# Package integration
from .extraction.trends import extract_google_trends
__all__ = ["extract_google_trends"]
```

## Current Architecture & File Relationships

**Decision**: Clean separation between execution script, package structure, and core implementation.

**Rationale**: Maintains modularity while providing simple user interfaces. The layered approach allows different usage patterns (direct script, package import, or manual execution).

**File Call Hierarchy**:
```
extract_trends.py (main script)
    ↓ calls
foodcom_pipeline/__init__.py (package exports)
    ↓ imports from
extraction/__init__.py (sub-package exports)
    ↓ imports from
trends.py (core implementation)
    ↓ calls
Google Trends API (external)
```

**Detailed Relationships**:
- **extract_trends.py**: User-facing script that extracts data and saves to CSV
- **foodcom_pipeline/__init__.py**: Defines public API of the package
- **extraction/__init__.py**: Defines public API of extraction sub-package
- **trends.py**: Contains actual extraction logic with rate limiting and error handling

**Configuration Files**:
- **config/trends_keywords.txt**: Keyword list read by trends.py
- **data/trends_raw.csv**: Output file written by extract_trends.py

**Lesson Learned**: Clear file organization reduces complexity and makes the codebase more maintainable. The separation allows users to choose their preferred execution method while keeping the core logic isolated.

## Implementation Process & Challenges

### Phase 1: Core Extraction Logic
- Created `extraction/trends.py` with `extract_google_trends()` function
- Implemented keyword loading, API batching, response parsing
- **Challenge**: API rate limiting required iterative testing of batch sizes
- **Solution**: Configurable parameters with conservative defaults

### Phase 2: Integration & Packaging
- Updated package `__init__.py` for clean imports
- Added pytrends to `pyproject.toml` dependencies
- **Challenge**: Import errors when package not installed in editable mode
- **Solution**: Alternative execution method using `sys.path` manipulation

### Phase 3: Testing & Validation
- Unit tests with mocked API calls using pytest
- Error condition testing (missing files, API failures)
- **Challenge**: Testing external API dependencies
- **Solution**: Comprehensive mocking of pytrends responses

### Phase 4: Documentation & Deployment
- Updated README with multiple execution methods
- Added configuration and troubleshooting notes
- **Challenge**: Path resolution issues across different run contexts
- **Solution**: Dynamic path finding relative to module location

## Performance Considerations
- **Memory**: DataFrame scales with number of keywords × time periods (acceptable for weekly data)
- **Network**: API calls are the bottleneck; batching optimizes this
- **Storage**: Weekly granularity keeps data manageable
- **Update Frequency**: Daily/weekly extraction recommended based on trend analysis needs

## Data Flow Integration

### Current State
```python
# Extract and persist data
from foodcom_pipeline import extract_google_trends
df = extract_google_trends()
df.to_csv('data/trends_raw.csv', index=False)
```

### Planned Pipeline Flow
```
extract_recipes() → extract_google_trends() → clean_trends_data() → calculate_trend_metrics() → merge_trends_with_recipes() → load_db()
```

**Note**: The `load_db()` function for PostgreSQL insertion is planned but not yet implemented. Keywords for trends analysis should be automatically extracted from Food.com recipe data rather than manually maintained.

## Monitoring & Maintenance Plan
- **API Stability**: Monitor pytrends for breaking changes through regular testing
- **Rate Limits**: Track API usage and adjust batch sizes if needed
- **Data Quality**: Validate interest scores and handle missing data gracefully
- **Fallback Plan**: Prepare direct HTTP implementation if pytrends becomes unreliable

## Files Created/Modified
- `src/foodcom_pipeline/extraction/trends.py` (core extraction implementation)
- `src/foodcom_pipeline/extraction/__init__.py` (extraction module exports)
- `src/foodcom_pipeline/__init__.py` (main package exports)
- `config/trends_keywords.txt` (keyword configuration file)
- `extract_trends.py` (convenience script for data extraction)
- `data/trends_raw.csv` (output data file)
- `pyproject.toml` (updated with pytrends dependency)
- `README.md` (updated documentation and usage instructions)
- `design_log.md` (this comprehensive design documentation)

## Testing Results
- ✅ Unit tests pass with mocked API
- ✅ Manual execution successful with sample keywords
- ✅ Error handling verified for edge cases
- ✅ Output DataFrame structure validated
- ✅ CSV persistence working correctly
- ✅ Package imports working correctly after endpoints removal
- ✅ File call relationships properly structured

## Future Enhancements
1. Regional interest breakdown (currently global only)
2. Real-time trend monitoring integration
3. Correlation analysis with recipe ratings
4. Automated keyword discovery from recipe data
5. Trend prediction modeling using time series analysis

## Key Takeaways
1. **API Selection**: Unofficial APIs require robust error handling and monitoring
2. **Rate Limiting**: Critical for API stability; implement configurable batching
3. **Path Resolution**: Essential for flexible execution contexts
4. **Error Handling**: Continue processing on partial failures for maximum data collection
5. **Configuration**: External files with dynamic resolution enable flexible deployment
6. **Testing**: Comprehensive mocking prevents external dependency issues
7. **Documentation**: Multiple execution methods support different development workflows

## Conclusion
The Google Trends extraction module successfully provides external trend data for the Food.com pipeline. The implementation handles API limitations gracefully while providing clean, structured data for downstream analysis. The design prioritizes reliability, maintainability, and integration with the existing Lambda Architecture. After iterative refinement and scope reduction, the current implementation focuses on the core extraction functionality with a clean, modular architecture that supports multiple execution patterns.