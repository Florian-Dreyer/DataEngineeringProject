# Recipe Explorer + Dashboard Restyle — Design Spec
**Date:** 2026-04-18  
**Scope:** Affiliate Commerce revenue stream, Recipe Explorer tab (Tab 1), and full dashboard visual restyle

---

## 1. Goals

1. Build the **Recipe Explorer** tab (currently an empty stub) as the primary consumer-facing feature, enabling the Affiliate Commerce revenue stream via "Shop Ingredients" buttons.
2. **Restyle all 4 tabs** to a consistent Clean & Modern theme (white/light-grey base, emerald-green accent `#10b981`).
3. **Refactor `app.py`** from a monolithic 772-line file into a thin orchestrator with one module per tab.

---

## 2. File Architecture

### New structure
```
src/foodcom_pipeline/dashboard/
├── app.py                          # Thin orchestrator: loads tabs, injects theme
├── theme.py                        # Shared CSS constants + colour palette
└── tabs/
    ├── tab_recipe_explorer.py      # NEW — Recipe Explorer + Affiliate Commerce
    ├── tab_substitutions.py        # Polished "Coming Soon" stub
    ├── tab_market_intelligence.py  # Extracted + restyled from app.py
    └── tab_pipeline_status.py      # Extracted + restyled from app.py

.streamlit/config.toml              # primaryColor, font, background colour
```

### Tab mapping
| Tab | Label | Old content | New content |
|-----|-------|-------------|-------------|
| 1 | 🍳 Recipe Explorer | Empty stub | Fully built |
| 2 | 🔄 Smart Substitutions | Empty stub | Polished "Coming Soon" card |
| 3 | 📊 Market Intelligence | Implemented (old style) | Same logic, restyled |
| 4 | ⚙️ Pipeline Status | Implemented (old style) | Same logic, restyled |

---

## 3. Theme

### Colours
| Token | Value | Usage |
|-------|-------|-------|
| `primary` | `#10b981` | Buttons, top borders on metric cards, active states |
| `primary-dark` | `#059669` | Hero gradient end, hover states |
| `primary-light` | `#d1fae5` | Tag backgrounds, Coming Soon card fill |
| `text-primary` | `#111827` | Headings, values |
| `text-secondary` | `#6b7280` | Labels, subtitles |
| `surface` | `#ffffff` | Cards |
| `bg` | `#f8fafc` | Page background |
| `border` | `#e5e7eb` | Card borders |

### Components (via `theme.py` CSS injected in `app.py`)
- **Metric card**: white, `border-top: 3px solid #10b981`, subtle `box-shadow`
- **Badge green**: `background: #10b981`, white text — Bayesian rating, status OK
- **Badge grey**: `background: #f3f4f6`, dark text — raw rating, secondary info
- **Badge amber**: `background: #fef3c7` — trend / warning signals
- **Shop button**: solid `#10b981`, white text, `border-radius: 6px`
- **Secondary button**: white, grey border — Instacart link

### Streamlit config (`.streamlit/config.toml`)
```toml
[theme]
primaryColor = "#10b981"
backgroundColor = "#f8fafc"
secondaryBackgroundColor = "#ffffff"
textColor = "#111827"
font = "sans serif"
```

### Plotly chart palette
All Plotly charts use `["#10b981", "#059669", "#34d399", "#6ee7b7", "#a7f3d0"]` as the colour sequence.

---

## 4. Tab 1 — Recipe Explorer

### Data sources
- **Primary**: `dim_recipe` table in PostgreSQL  
  Columns used: `name`, `avg_rating`, `sentiment_rating`, `avg_cook_minutes`, `top_ingredients` (pipe-separated), `tags`, `ingredient_count`, `calories`, `protein`, `fat`, `sugar`, `sodium`, `carbs`, `saturated_fat`
- **Fallback**: `staging/recipe_sentiment_ratings.parquet` joined with `staging/recipes_clean.parquet` if DB unavailable
- **Trend data**: `staging/ingredient_features.parquet` — `trend_index` column (currently NULL; badge hidden when NULL)

### Layout: Hero Search + Full-Width Cards

#### Hero header
- Emerald gradient (`#10b981` → `#059669`), `border-radius: 10px`
- Title: "Recipe Explorer"
- `st.text_input` search box (searches `name` + `top_ingredients`, case-insensitive)

#### Filter strip (below hero)
- Cook time slider: 0 – 180 min (cap at `avg_cook_minutes` max, capped at 180 in display)
- Minimum Bayesian rating: 1.0 – 5.0, step 0.5
- Cuisine tag multiselect: parsed from `tags` column (top 20 most frequent tags shown)
- Result count: "X recipes · showing Y–Z"

#### Recipe card (one per result)
Each card is a white rounded container with:

**Top row**
- Left: Recipe name (bold, 15px), cook time + ingredient count (grey subtitle)
- Rating row: `⭐ {sentiment_rating:.1f} Bayesian` (green badge) · `{avg_rating:.1f} raw` (grey badge) · `↑ +{delta:.1f} sentiment boost` (green text, only shown if delta > 0)
- Trend badge: `🔥 High / Medium / Low demand` (amber) — hidden if `trend_index` is NULL
- Right: **🛒 Shop on Amazon Fresh** button + **🥬 Instacart** button (stacked)

**Bottom section (3 columns)**
1. **Nutrition radar**: Plotly polar chart, 5 axes — Protein, Fat, Carbs, Sugar, Sodium — values are Food.com %DV columns (already 0–100 scale), used directly as radar values. Emerald fill, semi-transparent.
2. **Nutrition bars**: horizontal progress bars for the same 5 nutrients displaying `% Daily Value` (the raw column value from Food.com, already in 0–100 %DV range). Colour-coded: green (protein, carbs, fat), amber (sugar > 30%DV), red (sodium > 50%DV).
3. **Ingredient pills**: top ingredients displayed as grey rounded pills, truncated to 8 with `+N more`.

#### Shop button behaviour
**Amazon Fresh:**
```
https://www.amazon.com/s?k={ingredient1}+{ingredient2}+...&i=amazonfresh
```
Takes `top_ingredients` (pipe-separated), splits on `|`, URL-encodes each, joins with `+`. Maximum 8 ingredients to keep URL reasonable.

**Instacart:**
```
https://www.instacart.com/store/s?k={recipe_name}+ingredients
```
Recipe name URL-encoded, appended with `+ingredients`.

Both open in a new browser tab (`target="_blank"` via `st.markdown` link or `st.link_button`).

#### Pagination
- 10 results per page
- `st.session_state["recipe_page"]` tracks current page (reset to 0 on filter/search change)
- Previous / Next buttons rendered as columns below cards

### Caching
```python
@st.cache_data(ttl=300)
def load_recipes() -> pd.DataFrame: ...
```
Tries PostgreSQL first, falls back to parquet. Returns empty DataFrame on total failure (tab shows a warning banner, not a crash).

---

## 5. Tab 2 — Smart Substitutions (Stub)

Polished "Coming Soon" card:
- Emerald dashed border, light green gradient background
- Icon: 🔄
- Title: "Smart Substitutions"
- Description: "Ingredient swap cards with rating and nutritional deltas, real-time Bayesian recalculation, and native ad placement hooks. Detects underperforming ingredients and recommends trending alternatives."
- Pill badge: "Coming Soon" (solid emerald)

No data loading — purely presentational.

---

## 6. Tab 3 — Audience & Market Intelligence (Restyle Only)

All existing logic from `app.py` moves to `tab_market_intelligence.py` unchanged. Visual changes only:

- Segment filter, KPI row, radar chart, segment profiles table, CPG adjacency table, PDF export — all preserved
- KPI row replaced with metric cards (emerald top border)
- Section containers wrapped in white `border-radius: 8px` cards with `box-shadow`
- Plotly radar chart colour sequence updated to emerald palette
- CPG adjacency table: CPM range column highlighted in emerald green text

---

## 7. Tab 4 — Pipeline Status (Restyle Only)

All existing logic from `app.py` moves to `tab_pipeline_status.py` unchanged. Visual changes only:

- Data freshness, volume KPIs, quality signals, row counts table, clustering health chart, USDA coverage, Airflow runtimes — all preserved
- KPI row replaced with metric cards (emerald top border, delta row in green/red)
- Status badges: `✓ success` in green, `✗ failed` in red, `↻ running` in amber
- Pipeline stage table: striped rows, status badge column
- Plotly bar charts updated to emerald palette

---

## 8. `app.py` — Thin Orchestrator

```python
from foodcom_pipeline.dashboard.theme import inject_theme
from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import render as render_recipe_explorer
from foodcom_pipeline.dashboard.tabs.tab_substitutions import render as render_substitutions
from foodcom_pipeline.dashboard.tabs.tab_market_intelligence import render as render_market_intelligence
from foodcom_pipeline.dashboard.tabs.tab_pipeline_status import render as render_pipeline_status

st.set_page_config(page_title="Food.com Intelligence", page_icon="🍳", layout="wide")
inject_theme()

tabs = st.tabs(["🍳 Recipe Explorer", "🔄 Smart Substitutions", "📊 Market Intelligence", "⚙️ Pipeline Status"])
with tabs[0]: render_recipe_explorer()
with tabs[1]: render_substitutions()
with tabs[2]: render_market_intelligence()
with tabs[3]: render_pipeline_status()
```

`inject_theme()` in `theme.py` calls `st.markdown(CSS, unsafe_allow_html=True)`.

---

## 9. Out of Scope

- Smart Substitutions logic (Tab 2 is a stub only)
- Google Trends DAG wiring (trend_index remains NULL; badge hidden)
- Streaming layer / `recent_interactions`
- Any changes to ETL pipeline code

---

## 10. Data Assumptions

- `dim_recipe.top_ingredients` is pipe-separated (e.g. `"chicken|tomato|cream|garlic"`)
- `dim_recipe.sentiment_rating` may be NULL for recipes with no reviewed interactions — show `avg_rating` as fallback
- `ingredient_features.trend_index` is NULL for all ingredients until Google Trends is wired — trend badge is hidden, not shown as 0
- Nutrition columns (`calories`, `protein`, etc.) are in Food.com PDV units (% daily value per serving, 0–100 range) — displayed as-is with a `%DV` label on bars and radar axes. Calories is an exception (it's in kcal, not %DV in the raw data) — shown as a plain number with `kcal` label, excluded from the radar chart.
