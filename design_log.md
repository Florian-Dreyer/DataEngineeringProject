# Food.com Market Intelligence Pipeline Design Log

## 1. Executive Overview

This design log documents the construction of the Food.com market intelligence data pipeline. The pipeline began as a Google Trends extraction module and was expanded into a broader intelligence system that compares three sources of recipe demand and supply:

1. **Food.com recipe inventory** — what recipes and categories are already available in the Food.com corpus.
2. **Google Trends demand signals** — what users are actively searching for.
3. **Google AI Mode / AI Overview outputs** — what AI-mediated search experiences surface before or alongside traditional search results.

The core objective is to turn raw recipe, search, and AI-response data into dashboard-ready insights about **content gaps**, **emerging demand**, and **coverage opportunities**. The final system is designed not merely to extract data, but to explain how each insight was derived: from the raw query, to canonicalization, tagging, matching, scoring, and eventual dashboard interpretation.

The pipeline supports the following business questions:

- Which food terms are trending externally but weakly represented in Food.com?
- Which dishes are being surfaced by Google AI Mode but have no strong Food.com equivalent?
- Which Food.com categories are well-covered, under-covered, or over-matched?
- How was each match or gap determined?
- Which insights are reliable enough to support editorial, SEO, content strategy, or CPG adjacency decisions?

A key architectural shift made during development was moving from a **flat tag-only comparison** toward a **hybrid canonicalization + tagging + semantic/lexical matching architecture**. Tags remain useful for interpretation, but they are insufficient as the primary comparison mechanism because recipe demand terms often vary semantically rather than lexically. For example, `creamy one-pot garlic chicken`, `garlic chicken`, and `easy chicken dinner recipe` may point to the same broad demand cluster, even though simple keyword tags would fragment or flatten them.

## 2. Baseline Assumptions

The design is based on several explicit assumptions. These assumptions affect both the technical architecture and how dashboard outputs should be interpreted.

### 2.1 Demand is multi-source, not single-source

Google Trends captures search interest, while Google AI Mode captures AI-mediated surfacing of dish ideas. These are related but not identical signals:

- **Google Trends** reflects observed search behavior.
- **Google AI Mode** reflects what AI-generated search experiences recommend or summarize.
- **Food.com** reflects available recipe supply.

The dashboard should therefore avoid treating any one source as ground truth. Instead, it should show when multiple demand sources converge around the same term or cluster.

### 2.2 Search terms and recipe titles are not directly comparable without normalization

Raw terms differ substantially across systems:

- Google Trends may return `best banana bread recipe`.
- Google AI may return `One-Pot Creamy Garlic Chicken`.
- Food.com may contain `Garlic Chicken Pasta Bake`.

The pipeline therefore stores both:

- `raw_term` — original term for traceability.
- `canonical_term` — normalized term for comparison.

This ensures both reproducibility and analytic comparability.

### 2.3 Flat tags are interpretable but lossy

The initial Food.com tagging design grouped recipes using flat categories such as `pasta`, `chicken`, `beef`, `seafood`, `vegetarian`, `baking`, `soup`, `salad`, `breakfast`, and `dessert`. While useful, this structure is too shallow for comparing Food.com supply against external recipe demand.

For example:

- `birria tacos` is not just `beef`; it is also `mexican`, `tacos_handheld`, and possibly `comfort_food`.
- `sheet pan gnocchi with vegetables` is not just `vegetarian`; it also belongs to `pasta`, `casserole_bake`, and `quick_easy`.

The revised system keeps tags, but treats them as **explanatory metadata**, not the sole basis for matching.

### 2.4 Similarity thresholds are design choices, not objective truth

Match statuses such as `strong_match`, `weak_match`, and `gap` depend on thresholds. The current baseline uses:

- `strong_match`: similarity >= 0.80
- `weak_match`: similarity >= 0.60 and < 0.80
- `gap`: similarity < 0.60

These thresholds are intentionally visible in the UI so users understand that coverage is inferred, not absolute.

### 2.5 A zero-gap dashboard is suspicious, not automatically successful

If the system reports zero gaps despite external demand terms, this may indicate overly permissive matching rather than complete Food.com coverage. The dashboard therefore includes pipeline health warnings when all or nearly all external terms are classified as strong or weak matches.

## 3. System Architecture

The final architecture has five conceptual layers:

```text
External Demand Sources          Food.com Supply Source
-----------------------          ----------------------
Google Trends                    recipes_clean.parquet
Google AI Mode                   reviews / ingredients / metadata
        |                                  |
        v                                  v
External Term Normalization       Recipe Term Index
        |                                  |
        v                                  v
Tagging + Canonicalization        Tagging + Canonicalization
        |                                  |
        +--------------+-------------------+
                       v
              Matching + Gap Scoring
                       |
                       v
             Clustering + Explainability
                       |
                       v
              Streamlit Dashboard
```

The major outputs are:

| Output | Purpose |
|---|---|
| `recipe_tags.parquet` | Row-level tag assignments for Food.com recipes. |
| `recipe_term_index.parquet` | One row per Food.com recipe with canonical term and aggregated tags. |
| `external_recipe_terms.parquet` | Normalized Google Trends and Google AI terms. |
| `recipe_gap_analysis.parquet` | Main comparison table showing best Food.com match, similarity, gap score, and explanation. |
| `recipe_term_clusters.parquet` | Cluster-level summary of related external demand terms and Food.com coverage. |

## 4. Google Trends Extraction Design

### 4.1 Original requirement

Google Trends extraction was added to provide external popularity signals for recipe keywords. The original implementation extracted interest scores, timestamps, geography, and related queries using `pytrends`.

### 4.2 Key constraints

Google Trends does not provide a stable public API. The pipeline therefore faces several constraints:

1. **Unofficial API access** through `pytrends`.
2. **IP-based rate limiting**, often returning 429 errors.
3. **Batch-relative scoring**, where Google scores keywords relative to other terms in the same request.
4. **Partial failure risk**, where some batches may fail while others succeed.
5. **Noisy related queries**, where Google Trends may return non-recipe searches if seeds are too broad.

### 4.3 Extraction strategy

The extraction module uses:

- batched keyword requests,
- conservative rate limiting,
- exponential backoff,
- optional proxy rotation,
- successful batch tracking,
- anchor-based normalization.

The original anchor-based strategy used a repeated reference term such as `pasta` to calibrate cross-batch scores. This is necessary because a score of 80 in one Google Trends batch is not directly comparable to a score of 80 in another unless a shared anchor term is used.

### 4.4 Related-query filtering

A major issue discovered during testing was that Google Trends returned unrelated terms such as:

- `how to learn python`
- `car insurance quotes`
- `buy laptop online`
- `cheap flights`
- `stock market news`
- `weather forecast today`

This occurred because the seed set and related-query extraction were too broad. The revised pipeline adds a **recipe-intent filter**.

A Google Trends query is retained only if it satisfies at least one of the following:

1. It contains recipe language such as `recipe` or `recipes`.
2. It contains a known food, dish, cuisine, or ingredient keyword from the recipe taxonomy.
3. It can be mapped to a known canonical food term.

The pipeline also applies a blocklist for obvious non-recipe domains:

```python
BLOCKLIST = {
    "python", "insurance", "laptop", "real estate", "flights", "weather",
    "stock market", "used cars", "pc games", "coupons", "novels",
    "game reviews", "cartoon"
}
```

This filtering step is important because otherwise downstream gap analysis becomes corrupted by non-food demand signals.

## 5. Google AI Mode Extraction Design

### 5.1 Purpose

Google AI Mode extraction was added to understand what dishes AI-mediated search experiences surface when users ask broad food discovery questions such as:

- `dinner ideas`
- `easy weeknight meals`
- `healthy meal ideas`
- `what to cook tonight`

This is distinct from Google Trends. Trends captures what users search; AI Mode captures what AI systems recommend back to users.

### 5.2 SerpAPI client issue and resolution

During implementation, the pipeline encountered an import mismatch:

```python
from serpapi import GoogleSearch
```

This import failed because the installed `serpapi` package used the newer SDK interface. The resolution was to use the current client pattern:

```python
import serpapi
client = serpapi.Client(api_key=api_key)
results = client.search({"engine": "google_ai_mode", "q": prompted_query})
```

This change improved environment compatibility and avoided dependency confusion between the newer `serpapi` package and the legacy `google-search-results` package.

### 5.3 Prompt engineering for dish extraction

Initial AI Mode outputs contained prose, assistant-like follow-up text, and generic recommendations. For example, outputs included lines such as:

```text
would you like a custom shopping list or the macro breakdowns for any of these specific dishes?
```

This is not useful for structured data extraction. The prompt was therefore redesigned to force Google AI Mode into a structured-data behavior.

The prompt now instructs the AI system to:

- act as a structured data generator, not a conversational assistant;
- return dish names only;
- avoid explanations, questions, categories, and follow-up language;
- output a comma-separated list;
- remove descriptive modifiers such as `viral`, `one-pot`, `sheet-pan`, `easy`, and `quick` unless essential.

Example prompt pattern:

```python
def build_ai_query(seed: str) -> str:
    return (
        f"{seed}. "
        "You are a structured data generator, not a conversational assistant. "
        "Output a list of 20 dish names only. "
        "Each item must be a core dish name representing a real recipe or search term. "
        "Remove descriptive modifiers such as 'viral', 'one-pot', 'sheet-pan', 'easy', 'quick'. "
        "Do not include explanations, descriptions, questions, suggestions, or conversational text. "
        "Do not ask follow-up questions. "
        "Output format must be a single comma-separated list of dish names only. "
        "No numbering, no bullet points, no extra sentences. "
        "Examples: Tortellini, Garlic Chicken, Korean Beef Bowl, Feta Pasta, Birria Tacos."
    )
```

### 5.4 Raw AI Mode row duplication

The `ai_mode_raw` table may show repeated `seed_query` values. This is expected because the fetch function stores **one row per returned AI text block**, not one row per query. For example, if Google AI Mode returns four text blocks for `lunch ideas`, the table will contain four rows with the same `seed_query` but different `block_index` values.

This design preserves raw extraction fidelity. A grouped view can be created for user-facing display when necessary.

## 6. Canonicalization Design

Canonicalization is the central mechanism that allows Food.com recipes, Google Trends queries, and Google AI dish names to be compared.

### 6.1 Purpose

Canonicalization converts noisy raw text into a comparable dish-level phrase.

Examples:

| Raw term | Canonical term |
|---|---|
| `best garlic chicken recipe` | `garlic chicken` |
| `easy weeknight pasta recipes` | `pasta` |
| `one-pot creamy garlic chicken` | `garlic chicken` |
| `sheet pan gnocchi with vegetables` | `gnocchi vegetables` |
| `banana bread recipe` | `banana bread` |

### 6.2 Design choice

Canonicalization is deterministic and regex-based rather than model-based. This improves reproducibility and auditability.

The canonicalization layer removes:

- recipe boilerplate: `recipe`, `recipes`, `best`
- trend modifiers: `viral`, `trending`
- convenience modifiers: `easy`, `quick`, `weeknight`
- preparation modifiers: `one-pot`, `sheet-pan`, `30-minute`
- conversational residue from AI outputs

The pipeline stores both raw and canonical forms to satisfy two goals:

- **Traceability:** Users can see the original input.
- **Comparability:** The system can match similar terms across datasets.

## 7. Recipe Tagging and Taxonomy Design

### 7.1 Initial limitation

The original tag dictionary was a flat mapping from categories to keywords. It grouped Food.com recipes into categories such as:

- `pasta`
- `chicken`
- `beef`
- `seafood`
- `vegetarian`
- `baking`
- `soup`
- `salad`
- `breakfast`
- `dessert`

This was useful but insufficient for external demand comparison because it mixed proteins, formats, cuisines, and intents into one undifferentiated tag layer.

### 7.2 Revised taxonomy

The revised taxonomy is broader and more interpretable. It includes multiple dimensions:

| Dimension | Example tags |
|---|---|
| Protein/base | `chicken`, `beef`, `pork`, `seafood`, `egg`, `vegetarian` |
| Dish format | `pasta`, `rice`, `soup_stew`, `salad`, `sandwich_wrap`, `casserole_bake`, `tacos_handheld`, `bowl` |
| Cuisine/flavor | `italian`, `mexican`, `korean`, `japanese`, `thai`, `indian`, `mediterranean` |
| Intent/style | `healthy`, `comfort_food`, `quick_easy` |

### 7.3 Tag output design

The updated `recipe_tags` output stores:

| Column | Meaning |
|---|---|
| `recipe_id` | Food.com recipe ID. |
| `raw_term` | Original recipe name. |
| `canonical_term` | Normalized recipe/dish phrase. |
| `tag` | Assigned tag. |
| `tag_dimension` | Dimension such as `protein_base`, `dish_format`, `cuisine`, or `intent`. |
| `matched_term` | Keyword that triggered the tag. |
| `tagging_method` | `regex`, `dictionary`, or `llm`. |
| `source` | Usually `foodcom` for recipe inventory. |

This design improves explainability because users can see not only that a recipe was tagged as `mexican`, but also which term triggered the tag and how the classification was made.

### 7.4 LLM fallback

Gemini fallback is retained for recipes that are untagged or weakly tagged. However, it is capped and optional to preserve cost control and replicability.

The LLM is asked to assign only tags from the approved taxonomy. This prevents arbitrary uncontrolled labels from entering the database.

## 8. Recipe Term Index

The recipe term index provides one row per Food.com recipe. It is the main Food.com supply-side table used for matching external demand.

Expected columns:

| Column | Meaning |
|---|---|
| `recipe_id` | Food.com recipe ID. |
| `recipe_name` | Original recipe title. |
| `raw_term` | Same as recipe name, preserved for traceability. |
| `canonical_term` | Normalized comparable term. |
| `tags` | Aggregated tags assigned to recipe. |
| `tag_dimensions` | Aggregated dimensions represented by tags. |
| `ingredients_summary` | Concise ingredient representation for richer matching. |

This table decouples row-level tag assignments from recipe-level comparison.

## 9. External Recipe Terms

The external terms table unifies Google AI and Google Trends outputs.

Expected columns:

| Column | Meaning |
|---|---|
| `source` | `google_ai` or `google_trends`. |
| `raw_term` | Original AI dish phrase or Trends related query. |
| `canonical_term` | Normalized term. |
| `source_score` | Demand or normalized score from source. |
| `source_rank` or `raw_frequency` | Ranking/frequency signal where available. |
| `fetched_date` | Extraction date. |
| `tags` | Tags assigned using same taxonomy. |
| `tag_dimensions` | Dimensions represented by assigned tags. |

The goal is to make external demand terms comparable with Food.com recipes without losing their original source context.

## 10. Matching and Gap Analysis

### 10.1 Why matching is needed

The central analytical problem is not simply whether Food.com contains a keyword. It is whether Food.com contains a comparable recipe for an external demand term.

For example:

- External term: `birria tacos`
- Food.com closest match: `easy indian tacos`
- Similarity: 0.75

This is a partial lexical match, but semantically it may not be a satisfying content substitute. The UI must therefore expose both the score and the explanation.

### 10.2 Baseline matching method

The baseline matcher uses deterministic lexical matching:

1. Exact canonical term match.
2. Token overlap.
3. Fuzzy string similarity via RapidFuzz if available.
4. Fallback to Python `difflib` if RapidFuzz is unavailable.

This baseline is intentionally simple and inspectable. It supports reproducibility and makes debugging easier.

### 10.3 Optional embedding matching

Embeddings are recommended as the next improvement because they compare semantic similarity rather than exact keyword overlap. This is particularly useful for matching:

- `feta pasta` with `baked feta pasta with tomatoes`
- `garlic chicken` with `creamy garlic chicken pasta`
- `banana bread recipe` with `spicy banana bread`

The proposed embedding layer uses a local model such as `sentence-transformers/all-MiniLM-L6-v2`. It should be optional: if the dependency is unavailable, the pipeline falls back to lexical matching and logs a warning rather than failing.

### 10.4 Gap scoring

Gap score combines external demand with Food.com coverage.

Baseline formula:

```python
gap_score = source_score * (1 - best_foodcom_similarity)
```

Interpretation:

- High demand + poor Food.com match = high opportunity gap.
- High demand + strong Food.com match = low gap.
- Low demand + poor match = lower-priority gap.

### 10.5 Match status

The dashboard uses transparent thresholds:

| Status | Similarity threshold | Interpretation |
|---|---:|---|
| `strong_match` | >= 0.80 | Food.com likely has a comparable recipe. |
| `weak_match` | 0.60–0.79 | Food.com has a partial or imperfect match. |
| `gap` | < 0.60 | Food.com likely lacks a comparable recipe. |

These thresholds are exposed in dashboard tooltips because they shape the final interpretation.

### 10.6 Gap explanation

Every row in `recipe_gap_analysis` should contain a human-readable explanation.

Example:

```text
'birria tacos' appears in Google Trends but the closest Food.com match is 'easy indian tacos' with similarity 0.75. This is below the strong-match threshold of 0.80, so it is classified as a weak match. Tagged as mexican + tacos_handheld + beef.
```

This explanation makes the pipeline auditable and prevents the dashboard from becoming a black box.

## 11. Clustering Design

Clustering groups similar external demand terms into coherent opportunity areas.

Examples:

| Cluster | Terms |
|---|---|
| `banana bread` | `banana bread recipe`, `best banana bread recipe`, `healthy banana bread` |
| `garlic chicken` | `garlic chicken`, `creamy garlic chicken`, `one pot garlic chicken` |
| `birria tacos` | `birria tacos`, `birria taco recipe`, `beef birria tacos` |

The cluster output should include:

| Column | Meaning |
|---|---|
| `cluster_id` | Stable generated cluster ID. |
| `cluster_label` | Human-readable label. |
| `representative_term` | Primary term selected for display. |
| `source_terms` | Raw terms grouped into the cluster. |
| `sources_present` | Whether Google AI, Google Trends, or both contributed. |
| `dominant_tags` | Most common tags in cluster. |
| `avg_gap_score` | Average gap score across terms. |
| `max_gap_score` | Highest gap score in cluster. |
| `best_foodcom_match` | Best available Food.com match. |
| `foodcom_coverage_count` | Count of comparable Food.com recipes. |
| `explanation` | Plain-language cluster interpretation. |

Clustering should start with canonical-term grouping and fuzzy merging. Embedding clustering can be added later if stable and interpretable.

## 12. Streamlit Dashboard Design

### 12.1 Dashboard purpose

The dashboard should function as a decision-support interface, not a raw data viewer. It should answer:

- What is trending?
- Where does demand exceed Food.com supply?
- Why is something classified as a gap?
- What action should an editor, analyst, or product team take?

### 12.2 Existing UI issues

The current Streamlit dashboard shows the right underlying information, but several design issues reduce interpretability:

1. Dense tables dominate the page.
2. Key terms such as `gap_score`, `similarity`, and `weak_match` are not self-explanatory.
3. Explanations exist but are buried in long rows or logs.
4. Charts lack narrative framing and direct annotation.
5. Preset queries and active query configuration are not sufficiently visible.
6. Users cannot easily sort, filter, or inspect every table.

### 12.3 Revised UI principles

The dashboard should follow these principles:

1. **Insight first, data second** — show top opportunities before raw tables.
2. **Every metric needs a definition** — use tooltips and glossary sections.
3. **Every insight must be traceable** — raw term → canonical term → tags → match → score → explanation.
4. **All tables must be sortable and filterable** — users should be able to explore the data without editing code.
5. **Visual encodings should reduce cognitive load** — use progress bars, badges, and status colors.
6. **Warnings should flag suspicious pipeline outcomes** — for example, zero gaps may imply over-matching.

### 12.4 Query control panel

The Streamlit UI should include a persistent sidebar where users can:

- see preset Google Trends and Google AI queries,
- select which preset queries to use,
- add custom comma-separated queries,
- choose whether to include Google AI, Google Trends, or both,
- refresh or re-run the selected query set,
- view the active query set used for the current dashboard.

This improves replicability because users can see exactly which queries produced the displayed insights.

### 12.5 Global filters

The dashboard should include global filters for:

- source: `google_ai`, `google_trends`, or both;
- match status: `strong_match`, `weak_match`, `gap`;
- gap score range;
- similarity range;
- tags;
- tag dimensions;
- matching method: `lexical` or `embedding`.

All charts and tables should respect these filters.

### 12.6 Tooltip glossary

The dashboard should include tooltips for the following terms:

| Term | Tooltip definition |
|---|---|
| Gap score | Measures how much external demand exceeds Food.com recipe coverage. Higher means a larger opportunity gap. |
| Similarity | Similarity between an external demand term and the closest Food.com recipe, from 0 to 1. Higher means closer match. |
| Match status | Strong match >= 0.80, weak match 0.60–0.79, gap < 0.60. |
| Canonical term | Normalized form of a raw query or recipe title used for matching. |
| Source score | Demand signal from Google Trends or Google AI extraction. |
| Cluster | A group of similar search or AI terms representing the same recipe opportunity. |
| Matching method | Whether the match was computed using lexical similarity or embeddings. |

### 12.7 Recommended dashboard structure

Recommended tabs:

1. **Market Intelligence**
   - Top unmet recipe opportunities.
   - External demand overview.
   - Top Google AI and Google Trends terms.

2. **Gap Analysis**
   - Main sortable/filterable gap table.
   - Expandable row-level explanations.
   - Visual bar chart of top gap scores.

3. **Clusters**
   - Cluster-level opportunity table.
   - Cluster chart with hover tooltips.
   - Explanation of clustering method.

4. **Pipeline Health**
   - Counts of key output tables.
   - Strong/weak/gap breakdown.
   - Warnings for suspicious outputs.
   - Last updated timestamps.

5. **Segments / CPG Adjacency**
   - Audience and ingredient profiles.
   - CPG relevance mapping.
   - Clear caveat that these are strategic mappings, not causal claims.

### 12.8 Streamlit implementation choices

Use Streamlit-native components where possible:

- `st.sidebar` for query controls.
- `st.multiselect`, `st.text_area`, `st.checkbox`, `st.slider` for filters.
- `st.metric` for KPI cards.
- `st.expander` for explainability panels.
- `st.dataframe` with `st.column_config` for sortable tables and progress columns.
- Plotly charts for hoverable visualizations.

Example table configuration:

```python
st.dataframe(
    gap_df,
    column_config={
        "gap_score": st.column_config.ProgressColumn(
            "Gap Score",
            help="Measures how much consumer demand exceeds Food.com coverage. Higher = larger opportunity gap.",
            min_value=0,
            max_value=1,
        ),
        "best_foodcom_similarity": st.column_config.ProgressColumn(
            "Similarity",
            help="Similarity between external term and closest Food.com recipe.",
            min_value=0,
            max_value=1,
        ),
        "match_status": st.column_config.TextColumn(
            "Match Status",
            help="Strong >= 0.80, Weak 0.60–0.79, Gap < 0.60.",
        ),
    },
    use_container_width=True,
    hide_index=True,
)
```

## 13. Data Quality and Replicability

### 13.1 Replicability principles

The pipeline is designed to be replicable by preserving:

- raw terms,
- canonical terms,
- query sets,
- fetched dates,
- source labels,
- matching method,
- thresholds,
- gap reasons,
- tag triggers.

This ensures that a dashboard insight can be traced back to its originating data and transformation path.

### 13.2 Transparency principles

Each insight should answer:

1. What external term was observed?
2. Which source produced it?
3. How was it normalized?
4. What tags were assigned?
5. What Food.com recipe was closest?
6. What similarity score was computed?
7. What threshold determined its match status?
8. Why was it classified as a gap, weak match, or strong match?

### 13.3 Outcome implications

The pipeline supports strategic interpretation, but outputs should not be over-claimed.

For example:

- A high gap score indicates an opportunity signal, not guaranteed content success.
- A weak match may still be editorially acceptable if the Food.com recipe is close enough.
- AI Mode terms reflect AI-surfaced suggestions, not necessarily search volume.
- Google Trends scores are relative, normalized, and affected by query framing.
- Segment and CPG adjacency mappings are strategic heuristics, not causal attribution.

## 14. Error Handling and Operational Reliability

### 14.1 API failures

External APIs are treated as unstable dependencies. The pipeline therefore uses:

- retry logic,
- exponential backoff,
- partial failure handling,
- empty response checks,
- logging at each extraction stage.

### 14.2 File write safety

A file contention issue was encountered when writing parquet files in Airflow:

```text
OSError: [Errno 35] Resource deadlock avoided
```

The recommended fix is to write to a temporary unique file and then atomically replace the target:

```python
def safe_write_parquet(df, final_path):
    tmp_path = final_path.parent / f"{final_path.stem}.{uuid.uuid4().hex}.tmp.parquet"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, final_path)
```

This avoids partial writes and reduces contention during retries or overlapping task runs.

### 14.3 Airflow concurrency assumptions

Staging filenames should ideally be run-scoped or written atomically. Fixed staging filenames can cause collisions if:

- multiple manual runs overlap,
- retry attempts overlap with previous attempts,
- downstream tasks read while upstream tasks are still writing.

## 15. Monitoring and Pipeline Health

The dashboard should monitor:

- number of Food.com recipes indexed,
- number of recipe tags generated,
- number of external Google AI terms,
- number of external Google Trends terms,
- number of non-recipe Trends terms dropped,
- number of strong matches,
- number of weak matches,
- number of gaps,
- top 10 highest gap scores,
- last successful extraction time,
- matching method currently active.

A zero-gap result should trigger a warning:

```text
No gaps detected. This may indicate overly permissive matching rather than complete Food.com coverage.
```

## 16. Testing Strategy

### 16.1 Unit-level tests

Recommended tests:

1. Canonicalization examples:
   - `best garlic chicken recipe` -> `garlic chicken`
   - `banana bread recipe` -> `banana bread`
   - `sheet pan gnocchi with vegetables` -> `gnocchi vegetables`

2. Tagging examples:
   - `Birria Tacos` -> `mexican`, `tacos_handheld`, possibly `beef`
   - `Creamy Garlic Chicken Pasta` -> `chicken`, `pasta`, `comfort_food`
   - `Banana Bread` -> `baking`, `dessert` or `bread_baked`

3. Trends filter examples:
   - Drop `how to learn python`, `car insurance quotes`, `cheap flights`
   - Keep `banana bread recipe`, `ramen recipes`, `gluten free pizza crust recipe`

4. Matching examples:
   - `banana bread recipe` should strongly match a banana bread recipe if present.
   - `birria tacos` should be weak or gap unless a true birria recipe exists.

### 16.2 Integration tests

Integration tests should validate that each expected parquet output exists and contains required columns:

- `recipe_tags.parquet`
- `recipe_term_index.parquet`
- `external_recipe_terms.parquet`
- `recipe_gap_analysis.parquet`
- `recipe_term_clusters.parquet`

### 16.3 Dashboard tests

The Streamlit dashboard should be tested for:

- query control visibility,
- preset query display,
- custom query parsing,
- filter propagation across charts and tables,
- tooltip coverage,
- row-level explanation availability,
- empty/error states.

## 17. Implementation Phases

### Phase 1: Google Trends extraction

Implemented rate-limited Google Trends extraction using `pytrends`, batching, backoff, and normalization.

### Phase 2: Google AI Mode extraction

Integrated SerpAPI Google AI Mode extraction with the current SDK, prompt-engineered structured dish outputs, and filtering of conversational residue.

### Phase 3: Canonicalization and expanded taxonomy

Added deterministic canonicalization and expanded recipe taxonomy to preserve raw, canonical, matched, and tagged forms.

### Phase 4: Recipe term index

Created recipe-level supply index from Food.com recipes and tags.

### Phase 5: External term normalization

Unified Google Trends and Google AI outputs into one external demand table.

### Phase 6: Matching and gap scoring

Implemented lexical baseline matching, optional embedding strategy, similarity thresholds, gap scoring, and human-readable explanations.

### Phase 7: Clustering

Grouped related external terms into demand clusters for higher-level opportunity analysis.

### Phase 8: Streamlit dashboard redesign

Shifted the UI from raw tables toward insight cards, explainability expanders, sortable/filterable tables, query controls, tooltips, and pipeline transparency.

## 18. Advantages and Limitations

### Advantages

- Combines supply-side and demand-side data.
- Preserves raw and normalized data for auditability.
- Uses deterministic canonicalization for replicability.
- Supports explainable gap analysis.
- Allows fallback from embeddings to lexical matching.
- Makes dashboard insights traceable.
- Supports user-configurable query inputs.

### Limitations

- Google Trends extraction relies on an unofficial API.
- Google AI Mode outputs may vary between runs.
- Prompt engineering reduces but does not eliminate noisy AI responses.
- Lexical matching may overstate similarity for terms with overlapping words but different meanings.
- Embedding matching improves semantics but may reduce transparency unless explanations are carefully designed.
- Gap scores depend on chosen thresholds and source-score normalization.
- CPG segment mappings are heuristic and should be interpreted as strategic adjacency, not causal proof.

## 19. Future Enhancements

1. Add embedding-based matching as an optional production path.
2. Add manually reviewed benchmark pairs to calibrate similarity thresholds.
3. Add query-set versioning so dashboard runs are fully reproducible.
4. Add regional Google Trends extraction.
5. Add editorial feedback loops to mark false gaps and false matches.
6. Add embedding cluster labels generated from representative terms.
7. Add Streamlit controls for threshold adjustment.
8. Add dashboards for overrepresented Food.com categories.
9. Add model cards or data cards for each external source.
10. Add automated warnings when external terms are too broad or non-recipe-related.

## 20. Key Takeaways

1. **Canonicalization is essential** because recipe demand terms and recipe titles rarely match exactly.
2. **Tags explain the data but should not be the only comparison method.**
3. **Google Trends requires aggressive filtering** to avoid non-recipe contamination.
4. **Google AI Mode requires prompt engineering** to produce structured dish terms rather than conversational prose.
5. **Gap analysis must be explainable** because similarity thresholds are interpretive choices.
6. **A dashboard should communicate decisions, not merely display tables.**
7. **Replicability requires preserving raw terms, canonical terms, query sets, scores, thresholds, and explanations.**

## 21. Conclusion

The Food.com market intelligence pipeline evolved from a Google Trends extractor into a broader, explainable demand-supply comparison system. Its primary contribution is the ability to compare external recipe demand from Google Trends and Google AI Mode against Food.com recipe availability, while preserving enough transparency for users to understand how each insight was generated.

The final design prioritizes:

- reliability in the face of unstable external APIs,
- reproducibility through deterministic canonicalization,
- interpretability through tags and matched terms,
- semantic coverage through lexical and optional embedding matching,
- transparency through row-level gap explanations,
- usability through a Streamlit dashboard designed around decision-making.

The resulting architecture can support practical content strategy questions such as which recipes Food.com should add, update, or promote in response to emerging consumer demand. It also provides a reusable pattern for comparing internal content inventories against external search and AI-mediated demand signals.
