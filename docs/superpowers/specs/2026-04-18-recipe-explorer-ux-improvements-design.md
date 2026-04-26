# Recipe Explorer — UX Improvements Design Spec
**Date:** 2026-04-18  
**Scope:** `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py` only  
**Parent spec:** `2026-04-18-recipe-explorer-dashboard-restyle-design.md`

---

## 1. Goals

1. Fix all rendering bugs in the Recipe Explorer tab (white boxes, clipped radar, bad rating range, meta-tags in dropdown).
2. Replace the per-card expander with a two-mode list/detail navigation pattern so users can click a recipe name to read the full recipe (including steps).
3. Improve filter UX so controls are visually grouped and legible on the beige background.
4. Eliminate the 10-Plotly-charts-per-page performance bottleneck.

---

## 2. Two-Mode Tab Structure

The tab renders in one of two modes, controlled by `st.session_state["_selected_recipe"]` (a dict of the selected row's data, or `None`).

```
_selected_recipe is None  →  LIST MODE   (search + filters + compact cards)
_selected_recipe is dict  →  DETAIL MODE (full recipe page for one recipe)
```

Switching modes is handled by two helpers:

```python
def _select_recipe(row: dict) -> None:
    st.session_state["_selected_recipe"] = row
    st.rerun()

def _back_to_list() -> None:
    st.session_state["_selected_recipe"] = None
    st.rerun()
```

---

## 3. List Mode

### 3.1 Hero header
- Emerald gradient div (unchanged from current).
- `st.text_input` moved **outside and below** the gradient div — sits between the hero and the filter strip.

### 3.2 Filter strip
Wrap all filter controls in `st.expander("🔍 Filters", expanded=True)`. Inside the expander, use `st.markdown` to inject a light-grey background container (`background:#f8fafc; border:1px solid #e5e7eb; border-radius:8px; padding:16px`). This makes the white input boxes visually distinct from the surrounding page.

Filter controls (unchanged layout — two rows of columns):

| Control | Change |
|---------|--------|
| Max cook time slider | No change |
| Min rating select_slider | Range changed: `0.0 – 5.0` step `0.5`, default `0.0` |
| Max ingredients slider | No change |
| Tags multiselect | Label → "Filter by tag"; blocklist applied (see §3.3) |
| Max calories, Min protein, Low sugar, Low sodium | No change |

### 3.3 Tag blocklist
When building `all_tags` from `tag_counts`, exclude any tag that:
- Is exactly one of: `time-to-make`, `course`, `main-ingredient`, `preparation`, `occasion`, `equipment`, `dietary`, `technique`, `number-of-servings`, `meat`, `vegetables`
- Contains the substring `for-` (e.g. `for-large-groups`)
- Contains the substring `-servings`
- Is purely numeric

Only the top 60 tags after blocklist filtering are shown in the multiselect.

### 3.4 Compact recipe cards
Each card renders as a single `st.markdown()` HTML shell (no `st.container()` wrapper, no orphaned closing `</div>`). Streamlit buttons are rendered after the HTML shell using columns.

Card layout:
```
┌─────────────────────────────────────────────────────────────────────┐
│  [Recipe Name — st.button styled as link]   ⭐ 4.3  4.1 avg         │
│  ⏱ 23 min · 9 ingredients                                           │
│  [🛒 Shop on Amazon Fresh]  [🥬 Instacart]  (right-aligned)         │
└─────────────────────────────────────────────────────────────────────┘
```

- Recipe name is `st.button(name, key=f"select_{recipe_id}", type="secondary", use_container_width=False)`. Global CSS in `theme.py` already gives secondary buttons a clean appearance with the emerald theme.
- Rating badges: `⭐ {display_rating:.1f}` green badge only. Raw avg shown as grey badge only if both values exist and differ by > 0.2. No "Bayesian" label anywhere.
- Trend badge hidden (trend_index is NULL — unchanged from current).
- **No Plotly chart in list mode.** This eliminates the 10-chart render on every page load.

### 3.5 Pagination
Unchanged from current implementation.

---

## 4. Detail Mode

Triggered when `st.session_state["_selected_recipe"]` is not None.

### 4.1 Layout

```
← Back to results                          [🛒 Shop on Amazon Fresh]  [🥬 Instacart]

# Recipe Name
⏱ 23 min  ·  9 ingredients  ·  ⭐ 4.3  ·  4.1 avg

─────────────────────────────────────────────────────
Steps
─────────────────────────────────────────────────────
① Mix flour and baking powder in a large bowl.
② Add eggs and buttermilk. Stir until just combined.
...

─────────────────────────────────────────────────────
Nutrition & Ingredients
─────────────────────────────────────────────────────
[ Radar chart ] [ Nutrition bars ] [ Ingredient pills ]

Calories: 353 kcal
```

### 4.2 Back button
```python
if st.button("← Back to results"):
    _back_to_list()
```
Placed at top-left using `st.columns([1, 4])`.

### 4.3 Steps section
Steps are loaded by `load_recipe_detail(recipe_id)` (see §5.2). Each step rendered as:
```html
<div style="display:flex;align-items:flex-start;margin-bottom:12px;">
  <span style="background:#10b981;color:white;border-radius:50%;
               width:24px;height:24px;display:flex;align-items:center;
               justify-content:center;font-size:12px;font-weight:700;
               flex-shrink:0;margin-right:12px;">{n}</span>
  <span style="font-size:14px;line-height:1.6;color:#111827;">{step_text}</span>
</div>
```
If steps unavailable (not in parquet or load error): show `st.info("Recipe steps not available for this recipe.")`.

### 4.4 Nutrition & Ingredients section
Same radar chart + bars + pills as current, but now only rendered for one recipe at a time (no performance concern). Radar margins: `l=50, r=50, t=20, b=20`.

---

## 5. Data Loading

### 5.1 `load_recipes()` — unchanged shape
No steps column added. Stays lean for the full 231k-row load.

### 5.2 `load_recipe_detail(recipe_id)` — new function
```python
@st.cache_data(ttl=300, max_entries=100)
def load_recipe_detail(recipe_id) -> dict:
    """Load steps for a single recipe from parquet. Returns dict with 'steps' key."""
```
Reads only `["id", "steps"]` from `recipes_clean.parquet` (if column exists), filters to the matching `recipe_id`, parses the steps list with `ast.literal_eval`. Returns `{"steps": [...]}` or `{"steps": []}` on failure.

Cached with `max_entries=100` so up to 100 recently-viewed recipes are instant.

---

## 6. Bug Fixes Summary

| Bug | Fix |
|-----|-----|
| White box between cards | Remove `st.container()` wrapper; use single HTML block per card |
| Radar labels clipped | `margin=dict(l=50, r=50, t=20, b=20)` + `angularaxis=dict(tickfont=dict(size=9))` |
| "Bayesian" text in badge | Removed — badge shows `⭐ 4.3` only |
| Rating filter min=1.0 | Changed to `0.0` (VADER×5 range starts at 0) |
| Meta-tags in dropdown | Tag blocklist applied before building multiselect options |
| 10 charts per page slow | Charts only rendered in detail mode (one recipe at a time) |
| Search input inside gradient div | Moved outside gradient div |

---

## 7. Out of Scope

- Changes to any other tab
- Loading steps from PostgreSQL (parquet fallback only for steps)
- Recipe images (not in dataset)
- Smart Substitutions tab
- Any ETL or pipeline changes
