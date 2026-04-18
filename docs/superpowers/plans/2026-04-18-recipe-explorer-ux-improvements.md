# Recipe Explorer UX Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all rendering bugs, add list/detail two-mode navigation with recipe steps, and improve filter UX in `tab_recipe_explorer.py`.

**Architecture:** The tab switches between list mode (compact cards, no charts) and detail mode (full recipe with steps + nutrition) via `st.session_state["_selected_recipe"]`. Steps are loaded on-demand per recipe. All changes are confined to `tab_recipe_explorer.py` and its test file.

**Tech Stack:** Streamlit, Pandas, Plotly, PyArrow, Python 3.11+

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py` | Modify | All logic: data loading, helpers, renderers, main render() |
| `tests/dashboard/test_recipe_explorer.py` | Modify | Tests for pure helper functions |

---

## Task 1: Tag blocklist helper + fix rating slider minimum

**Files:**
- Modify: `tests/dashboard/test_recipe_explorer.py`
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

- [ ] **Step 1: Write the failing tests**

Add at the bottom of `tests/dashboard/test_recipe_explorer.py`, after the existing imports add `_is_useful_tag` to the import list, then add this test class:

```python
# Update the import at the top of the file:
from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import (
    apply_filters,
    build_amazon_url,
    build_instacart_url,
    nutrition_bar_color,
    _is_useful_tag,
)


class TestIsUsefulTag:
    def test_blocklisted_tags_excluded(self):
        assert _is_useful_tag("time-to-make") is False
        assert _is_useful_tag("course") is False
        assert _is_useful_tag("main-ingredient") is False
        assert _is_useful_tag("preparation") is False
        assert _is_useful_tag("occasion") is False
        assert _is_useful_tag("equipment") is False
        assert _is_useful_tag("dietary") is False
        assert _is_useful_tag("technique") is False
        assert _is_useful_tag("number-of-servings") is False
        assert _is_useful_tag("meat") is False
        assert _is_useful_tag("vegetables") is False

    def test_for_prefix_excluded(self):
        assert _is_useful_tag("for-large-groups") is False
        assert _is_useful_tag("for-1-or-2-servings") is False

    def test_servings_suffix_excluded(self):
        assert _is_useful_tag("1-2-servings") is False
        assert _is_useful_tag("4-6-servings") is False

    def test_numeric_tags_excluded(self):
        assert _is_useful_tag("60") is False
        assert _is_useful_tag("30") is False

    def test_useful_tags_pass(self):
        assert _is_useful_tag("italian") is True
        assert _is_useful_tag("vegetarian") is True
        assert _is_useful_tag("desserts") is True
        assert _is_useful_tag("30-minutes-or-less") is True
        assert _is_useful_tag("asian") is True
        assert _is_useful_tag("low-fat") is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/madhupolluru/Downloads/AY25:26 SEM 2/IS3107/Project/DataEngineeringProject"
python -m pytest tests/dashboard/test_recipe_explorer.py::TestIsUsefulTag -v
```

Expected: `ImportError` — `_is_useful_tag` not yet defined.

- [ ] **Step 3: Add `_is_useful_tag` and `_TAG_BLOCKLIST` to the tab module**

In `tab_recipe_explorer.py`, after the `_RADAR_LABELS` line, add:

```python
_TAG_BLOCKLIST = frozenset({
    "time-to-make", "course", "main-ingredient", "preparation",
    "occasion", "equipment", "dietary", "technique",
    "number-of-servings", "meat", "vegetables",
})


def _is_useful_tag(tag: str) -> bool:
    """Return True if a tag is meaningful for the cuisine/style filter."""
    if tag in _TAG_BLOCKLIST:
        return False
    if "for-" in tag:
        return False
    if "-servings" in tag:
        return False
    if tag.isdigit():
        return False
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/dashboard/test_recipe_explorer.py::TestIsUsefulTag -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/dashboard/test_recipe_explorer.py src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: add tag blocklist helper _is_useful_tag"
```

---

## Task 2: `load_recipe_detail()` — on-demand step loader

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

This function uses `@st.cache_data` so it cannot be unit-tested without a running Streamlit context. It is tested manually in Task 8.

- [ ] **Step 1: Add `load_recipe_detail` after `load_recipes()` in `tab_recipe_explorer.py`**

```python
@st.cache_data(ttl=300, max_entries=100)
def load_recipe_detail(recipe_id) -> dict:
    """Load steps for a single recipe from parquet. Returns {"steps": [...]}."""
    recipes_path = STAGING_DIR / "recipes_clean.parquet"
    if not recipes_path.exists():
        return {"steps": []}
    try:
        import pyarrow.parquet as pq
        schema_names = pq.read_schema(recipes_path).names
        if "steps" not in schema_names:
            return {"steps": []}
        df = pd.read_parquet(recipes_path, columns=["id", "steps"])
        match = df[df["id"] == recipe_id]
        if match.empty:
            return {"steps": []}
        raw = match.iloc[0]["steps"]
        if hasattr(raw, "__iter__") and not isinstance(raw, str):
            return {"steps": [str(s) for s in list(raw)]}
        try:
            parsed = ast.literal_eval(str(raw))
            if isinstance(parsed, list):
                return {"steps": [str(s) for s in parsed]}
        except Exception:
            pass
        return {"steps": []}
    except Exception:
        return {"steps": []}
```

- [ ] **Step 2: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: add load_recipe_detail() for on-demand step loading"
```

---

## Task 3: Session state navigation helpers

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

- [ ] **Step 1: Add navigation helpers after the data loading section, before the UI helpers section**

```python
# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def _select_recipe(row: dict) -> None:
    """Switch to detail mode for the given recipe row dict."""
    st.session_state["_selected_recipe"] = row
    st.rerun()


def _back_to_list() -> None:
    """Return to list mode."""
    st.session_state["_selected_recipe"] = None
    st.rerun()
```

- [ ] **Step 2: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: add _select_recipe and _back_to_list session state helpers"
```

---

## Task 4: Fix radar chart margins + remove "Bayesian" label

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

- [ ] **Step 1: Fix `_nutrition_radar` margins**

Find the `_nutrition_radar` function. Replace the `fig.update_layout(...)` call with:

```python
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=9)),
        ),
        showlegend=False,
        height=240,
        margin=dict(t=20, b=20, l=50, r=50),
    )
```

- [ ] **Step 2: Remove "Bayesian" from all badge strings in `_render_recipe_card`**

In `_render_recipe_card`, find the block that builds `badge_parts`. Replace the Bayesian badge line:

Old:
```python
            badge_parts.append(
                f'<span style="background:#10b981;color:white;border-radius:4px;'
                f'padding:2px 8px;font-size:12px;font-weight:700;">⭐ {bayesian:.1f} Bayesian</span>'
            )
```

New:
```python
            badge_parts.append(
                f'<span style="background:#10b981;color:white;border-radius:4px;'
                f'padding:2px 8px;font-size:12px;font-weight:700;">⭐ {bayesian:.1f}</span>'
            )
```

Also update the "raw" badge label from `{raw_rating:.1f} raw` to `{raw_rating:.1f} avg` for clarity.

- [ ] **Step 3: Run existing tests to confirm nothing broken**

```bash
python -m pytest tests/dashboard/test_recipe_explorer.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "fix: radar chart margins, remove Bayesian label from rating badge"
```

---

## Task 5: Replace `_render_recipe_card` with compact list card

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

The compact card has no Plotly chart (eliminating the 10-chart-per-page bottleneck) and uses `st.button` for the recipe name to trigger detail mode.

- [ ] **Step 1: Add `_render_compact_card` after the existing `_render_recipe_card` function**

```python
def _render_compact_card(row: pd.Series, card_index: int = 0) -> None:
    """Compact list-mode card — no chart. Recipe name triggers detail view."""
    name          = str(row.get("name", "Unknown"))
    cook_min      = row.get("avg_cook_minutes")
    n_ingredients = row.get("ingredient_count")
    display_rating = row.get("display_rating")
    avg_rating    = row.get("avg_rating")
    top_ingr_str  = str(row.get("top_ingredients") or "")
    ingredients   = [i.strip() for i in top_ingr_str.split("|") if i.strip()]
    recipe_id     = row.get("recipe_id", card_index)

    amazon_url    = build_amazon_url(ingredients)
    instacart_url = build_instacart_url(name)

    subtitle_parts = []
    if cook_min is not None and not pd.isna(cook_min):
        subtitle_parts.append(f"⏱ {int(cook_min)} min")
    if n_ingredients is not None and not pd.isna(n_ingredients):
        subtitle_parts.append(f"{int(n_ingredients)} ingredients")
    subtitle = " · ".join(subtitle_parts)

    badge_parts = []
    if display_rating is not None and not pd.isna(display_rating):
        badge_parts.append(
            f'<span style="background:#10b981;color:white;border-radius:4px;'
            f'padding:2px 8px;font-size:12px;font-weight:700;">⭐ {display_rating:.1f}</span>'
        )
    if (avg_rating is not None and not pd.isna(avg_rating)
            and display_rating is not None and not pd.isna(display_rating)
            and abs(display_rating - avg_rating) > 0.2):
        badge_parts.append(
            f'<span style="background:#f3f4f6;color:#374151;border-radius:4px;'
            f'padding:2px 8px;font-size:12px;">{avg_rating:.1f} avg</span>'
        )
    badge_html = " &nbsp; ".join(badge_parts) if badge_parts else ""

    st.markdown(
        '<div style="background:white;border-radius:10px;border:1px solid #e5e7eb;'
        'padding:16px 20px;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">',
        unsafe_allow_html=True,
    )
    col_info, col_shop = st.columns([3, 1])

    with col_info:
        if st.button(name, key=f"select_{recipe_id}_{card_index}", type="secondary"):
            _select_recipe(row.to_dict())
        if subtitle:
            st.caption(subtitle)
        if badge_html:
            st.markdown(badge_html, unsafe_allow_html=True)

    with col_shop:
        st.markdown(
            f'<a href="{amazon_url}" target="_blank" style="display:block;background:#10b981;'
            f'color:white;text-align:center;border-radius:6px;padding:9px 12px;'
            f'font-weight:700;font-size:13px;text-decoration:none;margin-bottom:6px;">'
            f'🛒 Shop on Amazon Fresh</a>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<a href="{instacart_url}" target="_blank" style="display:block;background:white;'
            f'color:#374151;text-align:center;border-radius:6px;padding:8px 12px;'
            f'font-size:13px;text-decoration:none;border:1px solid #e5e7eb;">'
            f'🥬 Instacart</a>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
```

- [ ] **Step 2: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: add _render_compact_card for list mode (no chart, clickable name)"
```

---

## Task 6: Add `_render_detail_view()` — full recipe page

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

- [ ] **Step 1: Add `_render_detail_view` after `_render_compact_card`**

```python
def _render_detail_view(row: dict) -> None:
    """Full-page detail view for a single recipe, including steps and nutrition."""
    recipe_id     = row.get("recipe_id")
    name          = str(row.get("name", "Unknown"))
    cook_min      = row.get("avg_cook_minutes")
    n_ingredients = row.get("ingredient_count")
    display_rating = row.get("display_rating")
    avg_rating    = row.get("avg_rating")
    top_ingr_str  = str(row.get("top_ingredients") or "")
    ingredients   = [i.strip() for i in top_ingr_str.split("|") if i.strip()]

    amazon_url    = build_amazon_url(ingredients)
    instacart_url = build_instacart_url(name)

    # --- Back button + shop buttons row ---
    col_back, col_shops = st.columns([3, 2])
    with col_back:
        if st.button("← Back to results", key="back_to_list"):
            _back_to_list()
    with col_shops:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(
                f'<a href="{amazon_url}" target="_blank" style="display:block;background:#10b981;'
                f'color:white;text-align:center;border-radius:6px;padding:9px 12px;'
                f'font-weight:700;font-size:13px;text-decoration:none;">'
                f'🛒 Shop on Amazon Fresh</a>',
                unsafe_allow_html=True,
            )
        with sc2:
            st.markdown(
                f'<a href="{instacart_url}" target="_blank" style="display:block;background:white;'
                f'color:#374151;text-align:center;border-radius:6px;padding:8px 12px;'
                f'font-size:13px;text-decoration:none;border:1px solid #e5e7eb;">'
                f'🥬 Instacart</a>',
                unsafe_allow_html=True,
            )

    # --- Recipe title + metadata ---
    st.markdown(f"# {html.escape(name)}")

    meta_parts = []
    if cook_min is not None and not pd.isna(cook_min):
        meta_parts.append(f"⏱ {int(cook_min)} min")
    if n_ingredients is not None and not pd.isna(n_ingredients):
        meta_parts.append(f"{int(n_ingredients)} ingredients")

    badge_parts = []
    if display_rating is not None and not pd.isna(display_rating):
        badge_parts.append(
            f'<span style="background:#10b981;color:white;border-radius:4px;'
            f'padding:2px 8px;font-size:13px;font-weight:700;">⭐ {display_rating:.1f}</span>'
        )
    if (avg_rating is not None and not pd.isna(avg_rating)
            and display_rating is not None and not pd.isna(display_rating)
            and abs(display_rating - avg_rating) > 0.2):
        badge_parts.append(
            f'<span style="background:#f3f4f6;color:#374151;border-radius:4px;'
            f'padding:2px 8px;font-size:13px;">{avg_rating:.1f} avg</span>'
        )

    meta_html = " &nbsp;·&nbsp; ".join(meta_parts)
    if badge_parts:
        if meta_html:
            meta_html += " &nbsp; "
        meta_html += " &nbsp; ".join(badge_parts)
    if meta_html:
        st.markdown(meta_html, unsafe_allow_html=True)

    st.divider()

    # --- Steps ---
    st.markdown("### Steps")
    detail = load_recipe_detail(recipe_id)
    steps = detail.get("steps", [])
    if steps:
        steps_html = "".join(
            f'<div style="display:flex;align-items:flex-start;margin-bottom:14px;">'
            f'<span style="background:#10b981;color:white;border-radius:50%;'
            f'min-width:26px;height:26px;display:flex;align-items:center;'
            f'justify-content:center;font-size:12px;font-weight:700;'
            f'flex-shrink:0;margin-right:14px;">{i}</span>'
            f'<span style="font-size:14px;line-height:1.6;color:#111827;">'
            f'{html.escape(str(step))}</span>'
            f'</div>'
            for i, step in enumerate(steps, 1)
        )
        st.markdown(steps_html, unsafe_allow_html=True)
    else:
        st.info("Recipe steps not available for this recipe.")

    st.divider()

    # --- Nutrition & Ingredients ---
    st.markdown("### Nutrition & Ingredients")
    row_series = pd.Series(row)
    col_radar, col_bars, col_pills = st.columns([1.2, 1.5, 1])

    with col_radar:
        st.plotly_chart(
            _nutrition_radar(row_series),
            use_container_width=True,
            key=f"detail_radar_{recipe_id}",
        )

    with col_bars:
        st.markdown("**Nutrition (% Daily Value)**")
        any_nutrition = False
        for nutrient, label in zip(_RADAR_NUTRIENTS, _RADAR_LABELS):
            val = row.get(nutrient)
            if val is None or pd.isna(val):
                continue
            any_nutrition = True
            pct = float(val)
            color = nutrition_bar_color(pct, nutrient)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:11px;color:#6b7280;margin-bottom:2px;">'
                f'<span>{label}</span><span>{pct:.0f}%</span></div>'
                f'<div style="height:6px;border-radius:3px;background:#f3f4f6;margin-bottom:6px;">'
                f'<div style="height:6px;border-radius:3px;background:{color};'
                f'width:{min(pct, 100):.0f}%;"></div></div>',
                unsafe_allow_html=True,
            )
        if not any_nutrition:
            st.caption("Nutrition data not available for this recipe.")

    with col_pills:
        st.markdown("**Ingredients**")
        if ingredients:
            pills_html = "".join(
                f'<span style="display:inline-block;background:#f3f4f6;border-radius:12px;'
                f'padding:3px 10px;font-size:11px;color:#6b7280;margin:2px;">'
                f'{html.escape(ing)}</span>'
                for ing in ingredients[:8]
            )
            if len(ingredients) > 8:
                pills_html += (
                    f'<span style="display:inline-block;background:#e5e7eb;border-radius:12px;'
                    f'padding:3px 10px;font-size:11px;color:#9ca3af;margin:2px;">'
                    f'+{len(ingredients)-8} more</span>'
                )
            st.markdown(pills_html, unsafe_allow_html=True)
        else:
            st.caption("—")

    cal_val = row.get("calories")
    if cal_val is not None and not pd.isna(cal_val):
        st.caption(f"Calories: {cal_val:.0f} kcal")
```

- [ ] **Step 2: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: add _render_detail_view with steps, nutrition, and shop buttons"
```

---

## Task 7: Refactor `render()` — two-mode structure + filter UX fixes

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

This task replaces the entire `render()` function. All filter, pagination, and card-rendering logic moves into a `_render_list_mode()` helper so that `render()` stays clean.

- [ ] **Step 1: Add `_render_list_mode()` helper — add this before `render()`**

```python
def _render_list_mode(df: pd.DataFrame) -> None:
    """Render the search + filter strip + paginated compact cards."""

    # --- Hero ---
    st.markdown(
        '<div style="background:linear-gradient(135deg,#10b981,#059669);border-radius:10px;'
        'padding:24px 28px;margin-bottom:12px;">'
        '<h1 style="color:white;margin:0;font-size:26px;">🍳 Recipe Explorer</h1>'
        '<p style="color:#d1fae5;margin:6px 0 0;">Search 231,637 recipes — '
        'see ratings, nutrition, and shop ingredients in one click.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    search = st.text_input(
        "", placeholder="🔍 Search recipes or ingredients...", label_visibility="collapsed"
    )

    # --- Filter strip ---
    with st.expander("🔍 Filters", expanded=True):
        st.markdown(
            '<div style="background:#f8fafc;border:1px solid #e5e7eb;'
            'border-radius:8px;padding:16px;">',
            unsafe_allow_html=True,
        )
        fcol1, fcol2, fcol3, fcol4 = st.columns([1.5, 1, 1, 1.5])
        with fcol1:
            max_cook = st.slider("Max cook time (min)", 5, 180, 180, step=5)
        with fcol2:
            min_rating = st.select_slider(
                "Min rating",
                options=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                value=0.0,
            )
        with fcol3:
            max_ingredients = st.slider("Max ingredients", 1, 30, 30, step=1)
        with fcol4:
            tag_counts: Counter = Counter()
            for tags_str in df["tags"].dropna():
                for t in _parse_tag_list(str(tags_str)):
                    if t and _is_useful_tag(t):
                        tag_counts[t] += 1
            all_tags = [tag for tag, _ in tag_counts.most_common(60)]
            selected_tags = st.multiselect("Filter by tag", options=all_tags, default=[])

        qcol1, qcol2, qcol3, qcol4, _ = st.columns([1, 1, 1, 1, 2])
        with qcol1:
            max_calories = st.number_input("Max calories (%DV)", min_value=0, max_value=500, value=500, step=10)
        with qcol2:
            min_protein = st.number_input("Min protein (%DV)", min_value=0, max_value=100, value=0, step=5)
        with qcol3:
            low_sugar = st.checkbox("Low sugar (<30% DV)")
        with qcol4:
            low_sodium = st.checkbox("Low sodium (<50% DV)")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Apply filters ---
    filtered = apply_filters(df, search, max_cook, min_rating, selected_tags)
    if "ingredient_count" in filtered.columns:
        filtered = filtered[filtered["ingredient_count"].fillna(999) <= max_ingredients]
    if "calories" in filtered.columns:
        filtered = filtered[filtered["calories"].fillna(0) <= max_calories]
    if "protein" in filtered.columns:
        filtered = filtered[filtered["protein"].fillna(0) >= min_protein]
    if low_sugar and "sugar" in filtered.columns:
        filtered = filtered[filtered["sugar"].fillna(100) < 30]
    if low_sodium and "sodium" in filtered.columns:
        filtered = filtered[filtered["sodium"].fillna(100) < 50]

    total = len(filtered)

    # --- Pagination state ---
    filter_hash = hash((
        search, max_cook, min_rating, max_ingredients,
        max_calories, min_protein, low_sugar, low_sodium,
        tuple(sorted(selected_tags)),
    ))
    if st.session_state.get("_recipe_filter_hash") != filter_hash:
        st.session_state["_recipe_page"] = 0
        st.session_state["_recipe_filter_hash"] = filter_hash
    page = st.session_state.get("_recipe_page", 0)
    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, n_pages - 1))

    start = page * PAGE_SIZE
    end   = min(start + PAGE_SIZE, total)
    st.caption(f"{total:,} recipes · showing {start + 1}–{end}" if total else "No recipes match your filters.")

    # --- Cards ---
    page_df = filtered.iloc[start:end]
    for idx, (_, row) in enumerate(page_df.iterrows()):
        _render_compact_card(row, card_index=start + idx)

    # --- Pagination controls ---
    pc1, pc2, pc3 = st.columns([1, 2, 1])
    with pc1:
        if page > 0 and st.button("← Previous", key="prev_page"):
            st.session_state["_recipe_page"] = page - 1
            st.rerun()
    with pc2:
        st.caption(f"Page {page + 1} of {n_pages}")
    with pc3:
        if page < n_pages - 1 and st.button("Next →", key="next_page"):
            st.session_state["_recipe_page"] = page + 1
            st.rerun()
```

- [ ] **Step 2: Replace the existing `render()` function body with the two-mode dispatcher**

Find the `def render() -> None:` function and replace its entire body with:

```python
def render() -> None:
    df = load_recipes()
    if df.empty:
        st.warning(
            "No recipe data found. Run the batch pipeline first "
            "(or set `FOODCOM_STAGING_DIR` to your staging directory)."
        )
        return

    selected = st.session_state.get("_selected_recipe")
    if selected is not None:
        _render_detail_view(selected)
    else:
        _render_list_mode(df)
```

- [ ] **Step 3: Run all tests**

```bash
python -m pytest tests/dashboard/test_recipe_explorer.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: two-mode list/detail render, improved filter strip, tag blocklist"
```

---

## Task 8: Manual smoke test in the browser

- [ ] **Step 1: Start the dashboard**

```bash
cd "/Users/madhupolluru/Downloads/AY25:26 SEM 2/IS3107/Project/DataEngineeringProject"
streamlit run src/foodcom_pipeline/dashboard/app.py
```

- [ ] **Step 2: Verify list mode**

Open the Recipe Explorer tab and check:
- [ ] Recipe cards show: name button, cook time, ingredient count, rating badge (no "Bayesian" text), shop buttons
- [ ] No white box artefacts between cards
- [ ] Filter strip is inside a styled expander
- [ ] Min rating slider starts at 0.0
- [ ] Tag multiselect does NOT contain: `time-to-make`, `course`, `main-ingredient`, `preparation`, `occasion`, `equipment`
- [ ] Tag multiselect DOES contain meaningful tags like `italian`, `vegetarian`, `desserts`
- [ ] Page loads noticeably faster than before (no Plotly charts in list)

- [ ] **Step 3: Verify detail mode**

Click any recipe name and check:
- [ ] Tab switches to detail view with recipe title, metadata, shop buttons
- [ ] Steps section shows numbered steps (or "not available" message if steps not in parquet)
- [ ] Nutrition radar chart is fully visible with no clipped labels
- [ ] Nutrition bars and ingredient pills render correctly
- [ ] "← Back to results" returns to same page + filters

- [ ] **Step 4: Final commit**

```bash
git add -p  # stage any manual fixes from smoke test
git commit -m "fix: smoke test corrections for recipe explorer UX improvements"
```
