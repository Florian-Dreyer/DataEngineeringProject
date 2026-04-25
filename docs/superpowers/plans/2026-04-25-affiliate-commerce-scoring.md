# Affiliate Commerce Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add affiliate commerce scoring columns, a gauge/insight UI section, a top-20 opportunities table, and sidebar filters to the Recipe Explorer tab — all inside one existing file.

**Architecture:** All changes are in-place in `tab_recipe_explorer.py`. A new pure helper `_compute_affiliate_columns()` isolates the scoring maths from Streamlit so it can be unit-tested. `_RECIPE_MEDIAN` is a module-level Series populated by `load_recipes()`. The radar chart gains an optional `median_row` parameter; all call sites pass `_RECIPE_MEDIAN`.

**Tech Stack:** Python 3.10+, pandas, plotly (go.Indicator for gauge), streamlit — all already imported in the file.

---

## File Map

| File | Change |
|---|---|
| `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py` | All implementation changes |
| `tests/dashboard/test_recipe_explorer.py` | New test classes for affiliate scoring |

> **Note:** The existing `TestBuildAmazonUrl` / `TestBuildInstacartUrl` tests assert URLs that do not match the current stub implementations — that is a pre-existing issue, not introduced by this plan. Do not modify those tests or functions.

---

## Task 1: Pure affiliate-column helper + tests

Extract all scoring maths into a testable pure function `_compute_affiliate_columns(df)`. No Streamlit involved.

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py` — add helper after `nutrition_bar_color`
- Modify: `tests/dashboard/test_recipe_explorer.py` — add `TestComputeAffiliateColumns`

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/dashboard/test_recipe_explorer.py`:

```python
import math
from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import (
    _compute_affiliate_columns,
)


def _make_affiliate_df(n=1200) -> pd.DataFrame:
    """1200-row df: recipes with varying review_count, cook time, ingredient_count."""
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "recipe_id":       range(n),
        "review_count":    rng.integers(1, 500, size=n).astype(float),
        "avg_cook_minutes": rng.integers(5, 120, size=n).astype(float),
        "ingredient_count": rng.integers(2, 20, size=n).astype(float),
    })


class TestComputeAffiliateColumns:
    def test_new_columns_exist(self):
        df = _compute_affiliate_columns(_make_affiliate_df())
        for col in ["affiliate_score", "cart_ready", "review_velocity",
                    "basket_value_est", "revenue_proj_monthly"]:
            assert col in df.columns, f"missing column: {col}"

    def test_affiliate_score_only_top1000(self):
        df = _compute_affiliate_columns(_make_affiliate_df(n=1200))
        non_nan = df["affiliate_score"].notna().sum()
        assert non_nan == 1000

    def test_affiliate_score_range(self):
        df = _compute_affiliate_columns(_make_affiliate_df())
        scores = df["affiliate_score"].dropna()
        assert (scores >= 0).all()
        assert (scores <= 1.01).all()  # small float tolerance

    def test_cart_ready_sweet_spot(self):
        df = pd.DataFrame({
            "recipe_id": [1, 2, 3],
            "review_count": [10.0, 10.0, 10.0],
            "avg_cook_minutes": [30.0, 30.0, 30.0],
            "ingredient_count": [7.0, 11.0, 12.0],
        })
        result = _compute_affiliate_columns(df)
        assert result.loc[result["recipe_id"] == 1, "cart_ready"].iloc[0] is True
        assert result.loc[result["recipe_id"] == 2, "cart_ready"].iloc[0] is True
        assert result.loc[result["recipe_id"] == 3, "cart_ready"].iloc[0] is False

    def test_basket_value_formula(self):
        df = pd.DataFrame({
            "recipe_id": [1],
            "review_count": [10.0],
            "avg_cook_minutes": [30.0],
            "ingredient_count": [8.0],
        })
        result = _compute_affiliate_columns(df)
        assert math.isclose(result.iloc[0]["basket_value_est"], 8 * 3.50)

    def test_revenue_proj_formula(self):
        df = pd.DataFrame({
            "recipe_id": [1],
            "review_count": [10.0],
            "avg_cook_minutes": [30.0],
            "ingredient_count": [8.0],
        })
        result = _compute_affiliate_columns(df)
        basket = 8 * 3.50
        expected_rev = 10_000 * 0.02 * basket * 0.04
        assert math.isclose(result.iloc[0]["revenue_proj_monthly"], expected_rev)

    def test_review_velocity_formula(self):
        df = pd.DataFrame({
            "recipe_id": [1],
            "review_count": [73.0],
            "avg_cook_minutes": [30.0],
            "ingredient_count": [8.0],
        })
        result = _compute_affiliate_columns(df)
        assert math.isclose(result.iloc[0]["review_velocity"], 1.0)

    def test_nan_safe_missing_cook_time(self):
        df = pd.DataFrame({
            "recipe_id": [1, 2],
            "review_count": [10.0, 20.0],
            "avg_cook_minutes": [float("nan"), 30.0],
            "ingredient_count": [8.0, 9.0],
        })
        result = _compute_affiliate_columns(df)
        # Should not raise; affiliate_score may be NaN for row 1 but no exception
        assert "affiliate_score" in result.columns
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/madhupolluru/Downloads/AY25:26 SEM 2/IS3107/Project/DataEngineeringProject"
python -m pytest tests/dashboard/test_recipe_explorer.py::TestComputeAffiliateColumns -v 2>&1 | head -30
```

Expected: `ImportError` — `_compute_affiliate_columns` does not exist yet.

- [ ] **Step 3: Add `_compute_affiliate_columns` to the source file**

In `tab_recipe_explorer.py`, add the following function **directly after `nutrition_bar_color`** (around line 121):

```python
def _compute_affiliate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute affiliate scoring columns in-place and return the DataFrame.

    Only the top 1 000 recipes by review_count receive an affiliate_score;
    all others get NaN.  All other columns (cart_ready, basket_value_est,
    review_velocity, revenue_proj_monthly) are computed for every row.
    """
    df = df.copy()

    # --- cart_ready (all rows) ---
    ic = df["ingredient_count"].fillna(0)
    df["cart_ready"] = ((ic >= 7) & (ic <= 11))

    # --- basket_value_est and revenue_proj_monthly (all rows) ---
    df["basket_value_est"] = df["ingredient_count"].fillna(0) * 3.50
    df["revenue_proj_monthly"] = 10_000 * 0.02 * df["basket_value_est"] * 0.04

    # --- review_velocity (all rows) ---
    df["review_velocity"] = df["review_count"] / 73.0

    # --- affiliate_score (top-1000 by review_count only) ---
    df["affiliate_score"] = float("nan")

    if "review_count" not in df.columns or df["review_count"].isna().all():
        return df

    top1k = df.nlargest(1000, "review_count").index

    # Median-fill cook time for normalisation; restore NaN afterwards
    cook = df.loc[top1k, "avg_cook_minutes"]
    cook_filled = cook.fillna(cook.median())
    max_cook = cook_filled.max()
    if max_cook == 0:
        max_cook = 1.0
    prep_norm = (cook_filled / max_cook).clip(lower=0.01)

    rc = df.loc[top1k, "review_count"].fillna(0)
    max_rc = rc.max()
    if max_rc == 0:
        max_rc = 1.0

    sweet = ((df.loc[top1k, "ingredient_count"].fillna(0) >= 7) &
             (df.loc[top1k, "ingredient_count"].fillna(0) <= 11)).astype(float)

    df.loc[top1k, "affiliate_score"] = (
        (rc / max_rc) * 0.40
        + (1.0 / prep_norm) * 0.30
        + sweet * 0.30
    ).clip(upper=1.0)

    return df
```

- [ ] **Step 4: Run tests — expect pass**

```bash
python -m pytest tests/dashboard/test_recipe_explorer.py::TestComputeAffiliateColumns -v
```

Expected: all 8 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py \
        tests/dashboard/test_recipe_explorer.py
git commit -m "feat: add _compute_affiliate_columns pure helper with tests"
```

---

## Task 2: Module-level `_RECIPE_MEDIAN` + wire into `load_recipes()`

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

- [ ] **Step 1: Add module-level initialisation**

Find the line `PAGE_SIZE = 10` (around line 45) and add directly below it:

```python
# Populated by load_recipes(); always a valid Series even before first load.
_RECIPE_MEDIAN: pd.Series = pd.Series(
    {n: 0.0 for n in ["protein", "fat", "carbs", "sugar", "sodium"]}
)
```

- [ ] **Step 2: Wire `_compute_affiliate_columns` and `_RECIPE_MEDIAN` into `load_recipes()`**

At the **very end** of `load_recipes()`, just before the `return df` statement (the final one in the function — it appears twice, once for the DB path and once for the parquet path), add these lines in **both** places:

For the DB path (after `df["display_rating"] = df["sentiment_rating"].fillna(df["avg_rating"])`):

```python
        if not df.empty:
            df["display_rating"] = df["sentiment_rating"].fillna(df["avg_rating"])
            # Alias weighted_review_count as review_count for affiliate scoring
            if "weighted_review_count" in df.columns:
                df["review_count"] = df["weighted_review_count"]
            elif "review_count" not in df.columns:
                df["review_count"] = float("nan")
            df = _compute_affiliate_columns(df)
            global _RECIPE_MEDIAN
            _RECIPE_MEDIAN = df[["protein", "fat", "carbs", "sugar", "sodium"]].median()
            return df
```

For the parquet path, find the two lines just before the final `return df`:

```python
    # Ensure all expected nutrition columns exist (fill missing with NaN)
    for col in ["calories", "protein", "fat", "sugar", "sodium", "carbs", "saturated_fat"]:
        if col not in df.columns:
            df[col] = float("nan")

    return df
```

Replace with:

```python
    # Ensure all expected nutrition columns exist (fill missing with NaN)
    for col in ["calories", "protein", "fat", "sugar", "sodium", "carbs", "saturated_fat"]:
        if col not in df.columns:
            df[col] = float("nan")

    # Alias weighted_review_count → review_count for affiliate scoring
    if "weighted_review_count" in df.columns:
        df["review_count"] = df["weighted_review_count"]
    elif "review_count" not in df.columns:
        df["review_count"] = float("nan")

    df = _compute_affiliate_columns(df)

    global _RECIPE_MEDIAN
    _RECIPE_MEDIAN = df[["protein", "fat", "carbs", "sugar", "sodium"]].median()

    return df
```

- [ ] **Step 3: Verify no import errors**

```bash
python -c "from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import _RECIPE_MEDIAN, _compute_affiliate_columns; print('OK', _RECIPE_MEDIAN)"
```

Expected output: `OK protein    0.0 ...` (the zero-initialised Series).

- [ ] **Step 4: Run full test suite to check no regressions**

```bash
python -m pytest tests/dashboard/test_recipe_explorer.py -v
```

Expected: all existing tests + `TestComputeAffiliateColumns` pass.

- [ ] **Step 5: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: populate _RECIPE_MEDIAN and affiliate columns in load_recipes()"
```

---

## Task 3: `_nutrition_radar()` median overlay

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

- [ ] **Step 1: Write failing test**

Add to `tests/dashboard/test_recipe_explorer.py`:

```python
from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import _nutrition_radar


class TestNutritionRadar:
    def _make_row(self):
        return pd.Series({"protein": 20.0, "fat": 30.0, "carbs": 50.0,
                          "sugar": 10.0, "sodium": 40.0})

    def test_no_median_has_one_trace(self):
        fig = _nutrition_radar(self._make_row())
        assert len(fig.data) == 1

    def test_with_median_has_two_traces(self):
        median = pd.Series({"protein": 15.0, "fat": 25.0, "carbs": 40.0,
                            "sugar": 8.0, "sodium": 30.0})
        fig = _nutrition_radar(self._make_row(), median_row=median)
        assert len(fig.data) == 2

    def test_median_trace_is_dashed_grey(self):
        median = pd.Series({"protein": 15.0, "fat": 25.0, "carbs": 40.0,
                            "sugar": 8.0, "sodium": 30.0})
        fig = _nutrition_radar(self._make_row(), median_row=median)
        median_trace = fig.data[1]
        assert median_trace.line.dash == "dash"
        assert "9ca3af" in median_trace.line.color
```

- [ ] **Step 2: Run test to confirm failure**

```bash
python -m pytest tests/dashboard/test_recipe_explorer.py::TestNutritionRadar -v
```

Expected: `TypeError` or `FAILED` — `_nutrition_radar` does not accept `median_row`.

- [ ] **Step 3: Update `_nutrition_radar` signature and body**

Replace the existing `_nutrition_radar` function with:

```python
def _nutrition_radar(row: pd.Series, median_row: pd.Series | None = None) -> go.Figure:
    """Plotly polar chart for 5 nutrition axes (%DV values).

    If median_row is supplied, overlays a grey dashed trace for the category median.
    """
    def _safe_float(v) -> float:
        return 0.0 if v is None or pd.isna(v) else float(v)

    values = [_safe_float(row.get(n)) for n in _RADAR_NUTRIENTS]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=_RADAR_LABELS + [_RADAR_LABELS[0]],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.15)",
        line=dict(color="#10b981", width=2),
        name="%DV",
    ))

    if median_row is not None:
        med_values = [_safe_float(median_row.get(n)) for n in _RADAR_NUTRIENTS]
        fig.add_trace(go.Scatterpolar(
            r=med_values + [med_values[0]],
            theta=_RADAR_LABELS + [_RADAR_LABELS[0]],
            fill="toself",
            fillcolor="rgba(156,163,175,0.10)",
            line=dict(color="#9ca3af", width=1, dash="dash"),
            name="Category median",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=9)),
        ),
        showlegend=False,
        height=240,
        margin=dict(t=20, b=20, l=50, r=50),
    )
    return fig
```

- [ ] **Step 4: Update all call sites to pass `_RECIPE_MEDIAN`**

There are three call sites:

1. In `_render_compact_card()`:
```python
# Before:
st.plotly_chart(
    _nutrition_radar(row),
    use_container_width=True,
    key=f"radar_{recipe_id}_{card_index}",
)
# After:
st.plotly_chart(
    _nutrition_radar(row, median_row=_RECIPE_MEDIAN),
    use_container_width=True,
    key=f"radar_{recipe_id}_{card_index}",
)
```

2. In `_render_detail_view()`:
```python
# Before:
st.plotly_chart(
    _nutrition_radar(row_series),
    use_container_width=True,
    key=f"detail_radar_{recipe_id}",
)
# After:
st.plotly_chart(
    _nutrition_radar(row_series, median_row=_RECIPE_MEDIAN),
    use_container_width=True,
    key=f"detail_radar_{recipe_id}",
)
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/dashboard/test_recipe_explorer.py::TestNutritionRadar -v
```

Expected: all 3 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py \
        tests/dashboard/test_recipe_explorer.py
git commit -m "feat: add category median overlay to nutrition radar chart"
```

---

## Task 4: Affiliate badge row in `_render_compact_card()`

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

- [ ] **Step 1: Locate insertion point**

In `_render_compact_card()`, find the closing block of the existing badge row:

```python
            if badge_parts:
                st.markdown(" &nbsp; ".join(badge_parts), unsafe_allow_html=True)
```

- [ ] **Step 2: Insert affiliate badge block immediately after that `if badge_parts` block**

```python
            # --- Affiliate badges ---
            aff_score = row.get("affiliate_score")
            cart_rdy  = row.get("cart_ready")
            basket    = row.get("basket_value_est")
            rev_proj  = row.get("revenue_proj_monthly")

            aff_parts = []
            if aff_score is not None and not pd.isna(aff_score):
                aff_parts.append(
                    f'<span style="background:#fef3c7;color:#92400e;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;">🏷️ Affiliate Score: {aff_score:.2f}</span>'
                )
            if cart_rdy is True:
                aff_parts.append(
                    f'<span style="background:#d1fae5;color:#065f46;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;">✅ Cart-Ready</span>'
                )
            if aff_parts:
                st.markdown(" &nbsp; ".join(aff_parts), unsafe_allow_html=True)
            if basket is not None and not pd.isna(basket):
                st.caption(f"Est. basket value: ${basket:.2f}")
            if rev_proj is not None and not pd.isna(rev_proj):
                st.caption(f"Est. monthly revenue: ${rev_proj:.2f}")
```

- [ ] **Step 3: Smoke-test import**

```bash
python -c "from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import _render_compact_card; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: add affiliate score and cart-ready badges to recipe cards"
```

---

## Task 5: Affiliate Insights section in `_render_detail_view()`

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

- [ ] **Step 1: Add `_affiliate_gauge` helper**

Add directly after `_compute_affiliate_columns` (before `_select_recipe`):

```python
def _affiliate_gauge(score: float, recipe_id) -> go.Figure:
    """Plotly indicator gauge for affiliate_score (0–1).

    Green ≥ 0.6, amber 0.3–0.59, red < 0.3.
    """
    if score >= 0.6:
        color = "#10b981"
    elif score >= 0.3:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"valueformat": ".2f", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 1], "tickfont": {"size": 9}},
            "bar": {"color": color},
            "bgcolor": "#f3f4f6",
            "borderwidth": 0,
        },
        title={"text": "Affiliate Score", "font": {"size": 12}},
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20))
    return fig
```

- [ ] **Step 2: Insert Affiliate Insights section into `_render_detail_view()`**

Find the block in `_render_detail_view()`:

```python
    st.divider()

    # --- Nutrition & Ingredients ---
    st.markdown("### Nutrition & Ingredients")
```

Insert the following **between** the second `st.divider()` and the Nutrition heading:

```python
    # --- Affiliate Insights ---
    st.markdown("### 💰 Affiliate Insights")
    aff_score = row.get("affiliate_score")
    if aff_score is not None and not pd.isna(aff_score):
        ai1, ai2, ai3 = st.columns([1.2, 1.5, 1])

        with ai1:
            st.plotly_chart(
                _affiliate_gauge(float(aff_score), recipe_id),
                use_container_width=True,
                key=f"affiliate_gauge_{recipe_id}",
            )

        with ai2:
            cart_rdy  = row.get("cart_ready")
            basket    = row.get("basket_value_est")
            rev_proj  = row.get("revenue_proj_monthly")
            velocity  = row.get("review_velocity")
            ic        = row.get("ingredient_count")
            sweet     = "Yes" if (ic is not None and not pd.isna(ic) and 7 <= int(ic) <= 11) else "No"
            cart_str  = "Yes" if cart_rdy is True else "No"
            basket_str = f"${basket:.2f}" if basket is not None and not pd.isna(basket) else "—"
            rev_str   = f"${rev_proj:.2f}" if rev_proj is not None and not pd.isna(rev_proj) else "—"
            vel_str   = f"{velocity:.2f}" if velocity is not None and not pd.isna(velocity) else "—"
            st.markdown(
                f"| Metric | Value |\n"
                f"|---|---|\n"
                f"| Review Velocity | {vel_str} / 90 days |\n"
                f"| Cart Ready | {cart_str} |\n"
                f"| Est. Basket Value | {basket_str} |\n"
                f"| Est. Monthly Revenue | {rev_str} |\n"
                f"| Ingredient Sweet Spot (7–11) | {sweet} |"
            )

        with ai3:
            basket_val = f"${basket:.2f}" if basket is not None and not pd.isna(basket) else "unknown"
            rev_val    = f"${rev_proj:.2f}" if rev_proj is not None and not pd.isna(rev_proj) else "unknown"
            st.info(
                f"This recipe scores {aff_score:.2f} for affiliate potential. "
                f"With an estimated basket of {basket_val} and 2% conversion at "
                f"10,000 monthly views, it could generate ~{rev_val}/month in commission."
            )
    else:
        st.caption("Affiliate data not available for this recipe.")

    st.divider()

```

> **Important:** Remove the original bare `st.divider()` that was between the Steps section and Nutrition — replace it with the block above which ends with its own `st.divider()`. The final file should have exactly one divider between Affiliate Insights and Nutrition.

- [ ] **Step 3: Smoke-test import**

```bash
python -c "from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import _affiliate_gauge, _render_detail_view; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: add Affiliate Insights section to recipe detail view"
```

---

## Task 6: Top-20 expander + sidebar affiliate filters in `_render_list_mode()`

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

- [ ] **Step 1: Add top-20 expander before the search box**

In `_render_list_mode()`, find the search input block:

```python
    # --- Search ---
    search = st.text_input(
```

Insert the following **immediately before** that block:

```python
    # --- Top 20 Affiliate Opportunities ---
    with st.expander("📈 Top 20 Affiliate Opportunities"):
        top20 = (
            df.dropna(subset=["affiliate_score"])
            .nlargest(20, "affiliate_score")
            [["name", "affiliate_score", "basket_value_est",
              "revenue_proj_monthly", "cart_ready",
              "avg_cook_minutes", "ingredient_count"]]
        )
        if top20.empty:
            st.caption("No affiliate data yet — run the pipeline first.")
        else:
            st.dataframe(
                top20,
                use_container_width=True,
                column_config={
                    "name":                 st.column_config.TextColumn("Recipe"),
                    "affiliate_score":      st.column_config.NumberColumn("Affiliate Score", format="%.2f"),
                    "basket_value_est":     st.column_config.NumberColumn("Est. Basket", format="$%.2f"),
                    "revenue_proj_monthly": st.column_config.NumberColumn("Est. Monthly Rev.", format="$%.2f"),
                    "cart_ready":           st.column_config.CheckboxColumn("Cart Ready"),
                    "avg_cook_minutes":     st.column_config.NumberColumn("Cook (min)", format="%d"),
                    "ingredient_count":     st.column_config.NumberColumn("Ingredients", format="%d"),
                },
                hide_index=True,
            )

```

- [ ] **Step 2: Add sidebar affiliate filters**

In `_render_list_mode()`, find the lines after the search filter block (just before `total = len(df)`):

```python
    else:
        st.session_state.pop("_recipe_search", None)

    total = len(df)
```

Insert the sidebar filter block **between** the `pop` line and `total = len(df)`:

```python
    else:
        st.session_state.pop("_recipe_search", None)

    # --- Sidebar affiliate filters ---
    with st.sidebar:
        st.markdown("### 🏷️ Affiliate Filters")
        cart_only = st.toggle("Cart-Ready only (7–11 ingredients)", key="aff_cart_only")
        min_aff   = st.slider("Minimum Affiliate Score", 0.0, 1.0, 0.0,
                              step=0.05, key="aff_min_score")
    if cart_only:
        df = df[df["cart_ready"] == True]  # noqa: E712
    if min_aff > 0.0:
        df = df[df["affiliate_score"].fillna(0) >= min_aff]

    total = len(df)
```

- [ ] **Step 3: Smoke-test import**

```bash
python -c "from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import _render_list_mode; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/dashboard/test_recipe_explorer.py -v
```

Expected: all tests pass, no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: add top-20 affiliate table and sidebar filters to recipe explorer"
```

---

## Self-Review Checklist

- [x] **Spec §1** (affiliate_score formula, top-1000 only) → Task 1 `_compute_affiliate_columns`
- [x] **Spec §2** (cart_ready boolean) → Task 1
- [x] **Spec §3** (review_velocity = review_count / 73) → Task 1
- [x] **Spec §4** (basket_value_est) → Task 1
- [x] **Spec §5** (revenue_proj_monthly) → Task 1
- [x] **Spec load_recipes changes** (NaN-safe, weighted_review_count alias) → Task 2
- [x] **Spec _nutrition_radar median trace** → Task 3 (all 3 call sites updated)
- [x] **Spec _render_compact_card badges** → Task 4
- [x] **Spec _render_detail_view Affiliate Insights** → Task 5 (gauge, table, info panel)
- [x] **Spec Top-20 expander** → Task 6
- [x] **Spec sidebar filters** → Task 6
- [x] **NaN safety** — all UI blocks guard with `is not None and not pd.isna()`
- [x] **No new CSS classes** — all spans use inline styles matching existing pattern
- [x] **No renamed functions** — existing signatures unchanged
