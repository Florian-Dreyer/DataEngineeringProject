# Affiliate Commerce Scoring — Design Spec
Date: 2026-04-25  
File: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

---

## 1. Scope

Extend the Recipe Explorer tab with an affiliate commerce scoring layer. No new files, no new CSS classes, no renamed functions. All changes are in-place within `tab_recipe_explorer.py`.

---

## 2. Data Source Decisions

| Need | Source | Rationale |
|---|---|---|
| `review_count` | `weighted_review_count` from `recipe_sentiment_ratings.parquet` | Already loaded in `load_recipes()`; avoids a 1 M-row join on `interactions_clean.parquet` |
| `review_velocity` | `weighted_review_count / 73` | Dataset spans 2000–2018 ≈ 73 ninety-day periods; no submission date in parquet |
| Nutrition median | Computed inside `load_recipes()` from the full df | Stored as module-level `_RECIPE_MEDIAN` Series |

---

## 3. New Module-Level Variable

```python
_RECIPE_MEDIAN: pd.Series  # 5 radar nutrients, set inside load_recipes()
```

Initialised to `pd.Series({n: 0.0 for n in _RADAR_NUTRIENTS})` at module level so it is always defined even before `load_recipes()` is called.

---

## 4. New Columns — computed in `load_recipes()`

Computed **only for the top 1 000 recipes by `review_count`**; all other rows receive `NaN`.

Before computing, median-fill missing `ingredient_count` and `avg_cook_minutes`; restore NaN after.

| Column | Formula |
|---|---|
| `review_count` | alias of `weighted_review_count` (float) |
| `affiliate_score` | `(review_count / max_review_count) * 0.40 + (1 / prep_time_norm) * 0.30 + sweet_spot * 0.30` |
| `cart_ready` | `True` if `ingredient_count` in [7, 11] inclusive, else `False` (NaN rows → `False`) |
| `review_velocity` | `review_count / 73` |
| `basket_value_est` | `ingredient_count × 3.50` |
| `revenue_proj_monthly` | `10_000 × 0.02 × basket_value_est × 0.04` |

Where:
- `prep_time_norm` = `avg_cook_minutes / max(avg_cook_minutes)` (clamp min to 0.01 to avoid division-by-zero)
- `sweet_spot` = `1` if `ingredient_count` in [7, 11] else `0`
- `max_review_count` = max of `review_count` across the **top-1000 subset**

---

## 5. `_nutrition_radar()` Signature Change

```python
def _nutrition_radar(row: pd.Series, median_row: pd.Series | None = None) -> go.Figure
```

- If `median_row` is provided, add a second `Scatterpolar` trace:
  - `fillcolor="rgba(156,163,175,0.10)"`, `line=dict(color="#9ca3af", width=1, dash="dash")`
  - `name="Category median"`
- Pass `_RECIPE_MEDIAN` at all call sites in `_render_compact_card()` and `_render_detail_view()`.

---

## 6. `_render_compact_card()` Changes

Below the existing `badge_parts` markdown block, add (only when values are not NaN):

1. **Affiliate score badge** — amber span: `🏷️ Affiliate Score: X.XX`
2. **Cart-ready pill** — green span: `✅ Cart-Ready` (only when `cart_ready is True`)
3. **Basket value caption** — `st.caption("Est. basket value: $XX.XX")`
4. **Monthly revenue caption** — `st.caption("Est. monthly revenue: $XX.XX")`

All spans follow the existing inline-style pattern. No new CSS classes.

---

## 7. `_render_detail_view()` Changes

Add a new section **after** the Steps divider and **before** the Nutrition section:

```
### 💰 Affiliate Insights
```

Three-column layout `[1.2, 1.5, 1]`:

**Col 1 — Gauge chart**  
Plotly `go.Indicator` (gauge mode), value = `affiliate_score`, range [0, 1].  
- Green (`#10b981`) if ≥ 0.6  
- Amber (`#f59e0b`) if 0.3–0.59  
- Red (`#ef4444`) if < 0.3  
Key: `f"affiliate_gauge_{recipe_id}"`

**Col 2 — Markdown table**
```
| Metric | Value |
|---|---|
| Review Velocity | X.XX / 90 days |
| Cart Ready | Yes / No |
| Est. Basket Value | $XX.XX |
| Est. Monthly Revenue | $XX.XX |
| Ingredient Sweet Spot | Yes / No |
```

**Col 3 — Plain-language summary**  
`st.info()` with text:  
> "This recipe scores X.XX for affiliate potential. With an estimated basket of $XX and 2% conversion at 10,000 monthly views, it could generate ~$XX/month in commission."

When `affiliate_score` is NaN (recipe not in top 1 000), show `st.caption("Affiliate data not available for this recipe.")` instead of the three-column layout.

---

## 8. `_render_list_mode()` Changes

### 8a. Top-20 Expander (before search box)

```python
with st.expander("📈 Top 20 Affiliate Opportunities"):
    st.dataframe(top20_df, ...)
```

- `top20_df` = `df.dropna(subset=["affiliate_score"]).nlargest(20, "affiliate_score")`
- Columns shown: `name`, `affiliate_score`, `basket_value_est`, `revenue_proj_monthly`, `cart_ready`, `avg_cook_minutes`, `ingredient_count`
- Format: `affiliate_score` → 2 dp; `basket_value_est` and `revenue_proj_monthly` → `$XX.XX`
- Use `st.dataframe()` with `column_config` for formatting (no external libs)

### 8b. Sidebar Affiliate Filters (after search filter, before pagination)

```python
with st.sidebar:
    st.markdown("### 🏷️ Affiliate Filters")
    cart_only = st.toggle("Cart-Ready only (7–11 ingredients)")
    min_aff   = st.slider("Minimum Affiliate Score", 0.0, 1.0, 0.0, step=0.05)
```

Apply:
```python
if cart_only:
    df = df[df["cart_ready"] == True]
if min_aff > 0.0:
    df = df[df["affiliate_score"].fillna(0) >= min_aff]
```

---

## 9. What Does NOT Change

- Function signatures of `render()`, `_render_list_mode()`, `_back_to_list()`, `_select_recipe()`
- Existing badge row logic in `_render_compact_card()`
- Amazon Fresh and Instacart buttons
- Pagination controls
- All existing filter/search logic
- No new CSS classes or external stylesheets

---

## 10. Error / NaN Safety Rules

- All new UI elements guard with `if val is not None and not pd.isna(val)` before rendering
- `cart_ready` displayed as "Yes"/"No" strings in the detail table (never raw bool)
- If `affiliate_score` is NaN for a card, the new badge row is silently skipped
- `_RECIPE_MEDIAN` is always a valid Series (initialised at module level)
