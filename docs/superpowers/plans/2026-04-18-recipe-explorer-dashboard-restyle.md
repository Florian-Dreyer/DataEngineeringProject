# Recipe Explorer + Dashboard Restyle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Recipe Explorer tab (with Affiliate Commerce Shop buttons) and restyle all 4 dashboard tabs to a Clean & Modern emerald theme, while refactoring `app.py` into a modular tab architecture.

**Architecture:** `app.py` becomes a thin orchestrator that imports a `render()` function from each of 4 tab modules under `tabs/`. A shared `theme.py` injects global CSS. The new `tab_recipe_explorer.py` reads from `dim_recipe` (PostgreSQL primary, parquet fallback) and exposes pure helper functions that are independently testable.

**Tech Stack:** Streamlit, Plotly, pandas, psycopg2, urllib.parse (stdlib), pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `.streamlit/config.toml` | Streamlit theme tokens (primaryColor, bg, font) |
| Create | `src/foodcom_pipeline/dashboard/theme.py` | `inject_theme()` — injects global CSS via `st.markdown` |
| Create | `src/foodcom_pipeline/dashboard/tabs/__init__.py` | Empty — makes `tabs` a package |
| **Create** | `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py` | Hero search, filters, recipe cards, Shop buttons, pagination |
| Create | `src/foodcom_pipeline/dashboard/tabs/tab_substitutions.py` | Polished "Coming Soon" stub |
| Create | `src/foodcom_pipeline/dashboard/tabs/tab_market_intelligence.py` | Extracted + restyled Audience & Market tab |
| Create | `src/foodcom_pipeline/dashboard/tabs/tab_pipeline_status.py` | Extracted + restyled Pipeline Status tab |
| Modify | `src/foodcom_pipeline/dashboard/app.py` | Thin orchestrator — remove old tab functions, import tab modules |
| Create | `tests/dashboard/__init__.py` | Empty — makes `tests/dashboard` a package |
| Create | `tests/dashboard/test_recipe_explorer.py` | Unit tests for pure helper functions |

---

## Task 1: Streamlit theme config + theme.py

**Files:**
- Create: `.streamlit/config.toml`
- Create: `src/foodcom_pipeline/dashboard/theme.py`

- [ ] **Step 1: Create `.streamlit/config.toml`**

```toml
[theme]
primaryColor = "#10b981"
backgroundColor = "#f8fafc"
secondaryBackgroundColor = "#ffffff"
textColor = "#111827"
font = "sans serif"
```

- [ ] **Step 2: Create `src/foodcom_pipeline/dashboard/theme.py`**

```python
import streamlit as st

_CSS = """
<style>
/* Page background */
.stApp { background-color: #f8fafc; }

/* Metric cards — emerald top border */
[data-testid="stMetric"] {
    background: white;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    border-top: 3px solid #10b981;
    padding: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Tab bar */
.stTabs [data-baseweb="tab"] {
    background-color: white;
    border-radius: 6px 6px 0 0;
    border: 1px solid #e5e7eb;
    border-bottom: none;
    padding: 8px 18px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    border-top: 2px solid #10b981 !important;
    color: #10b981 !important;
}

/* Primary buttons */
.stButton > button[kind="primary"],
.stButton > button {
    background-color: #10b981;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #059669;
    border: none;
}

/* DataFrames */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Dividers */
hr { border-color: #e5e7eb; }

/* Section card helper — use via st.container() + custom class not possible in
   Streamlit, so we use st.markdown with inline styles where needed */
</style>
"""


def inject_theme() -> None:
    """Inject global Clean & Modern CSS into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)
```

- [ ] **Step 3: Commit**

```bash
git add .streamlit/config.toml src/foodcom_pipeline/dashboard/theme.py
git commit -m "feat: add Clean & Modern theme (emerald palette, metric card styles)"
```

---

## Task 2: Create tab module skeletons + refactor app.py

**Files:**
- Create: `src/foodcom_pipeline/dashboard/tabs/__init__.py`
- Create: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py` (stub)
- Create: `src/foodcom_pipeline/dashboard/tabs/tab_substitutions.py` (stub)
- Create: `src/foodcom_pipeline/dashboard/tabs/tab_market_intelligence.py` (stub)
- Create: `src/foodcom_pipeline/dashboard/tabs/tab_pipeline_status.py` (stub)
- Modify: `src/foodcom_pipeline/dashboard/app.py`

- [ ] **Step 1: Create `src/foodcom_pipeline/dashboard/tabs/__init__.py`**

Empty file. Just create it:
```python
```

- [ ] **Step 2: Create stub render() in each tab file**

`src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`:
```python
import streamlit as st


def render() -> None:
    st.info("Recipe Explorer — coming in next task.")
```

`src/foodcom_pipeline/dashboard/tabs/tab_substitutions.py`:
```python
import streamlit as st


def render() -> None:
    st.info("Smart Substitutions — coming in next task.")
```

`src/foodcom_pipeline/dashboard/tabs/tab_market_intelligence.py`:
```python
import streamlit as st


def render() -> None:
    st.info("Market Intelligence — coming in next task.")
```

`src/foodcom_pipeline/dashboard/tabs/tab_pipeline_status.py`:
```python
import streamlit as st


def render() -> None:
    st.info("Pipeline Status — coming in next task.")
```

- [ ] **Step 3: Replace `src/foodcom_pipeline/dashboard/app.py` with the thin orchestrator**

Replace the entire file content with:

```python
"""
app.py — thin orchestrator for the Food.com Analytics dashboard.

Run locally (from project root):
  FOODCOM_STAGING_DIR=./staging streamlit run src/foodcom_pipeline/dashboard/app.py
"""

import streamlit as st

from foodcom_pipeline.dashboard.theme import inject_theme
from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import render as render_recipe_explorer
from foodcom_pipeline.dashboard.tabs.tab_substitutions import render as render_substitutions
from foodcom_pipeline.dashboard.tabs.tab_market_intelligence import render as render_market_intelligence
from foodcom_pipeline.dashboard.tabs.tab_pipeline_status import render as render_pipeline_status


def main() -> None:
    st.set_page_config(
        page_title="Food.com Intelligence",
        page_icon="🍳",
        layout="wide",
    )
    inject_theme()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🍳 Recipe Explorer",
        "🔄 Smart Substitutions",
        "📊 Market Intelligence",
        "⚙️ Pipeline Status",
    ])

    with tab1:
        render_recipe_explorer()
    with tab2:
        render_substitutions()
    with tab3:
        render_market_intelligence()
    with tab4:
        render_pipeline_status()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify the app starts without import errors**

```bash
cd "/Users/madhupolluru/Downloads/AY25:26 SEM 2/IS3107/Project/DataEngineeringProject"
FOODCOM_STAGING_DIR=./staging python -c "
import streamlit.testing.v1
# Just check imports resolve
from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import render
from foodcom_pipeline.dashboard.tabs.tab_substitutions import render
from foodcom_pipeline.dashboard.tabs.tab_market_intelligence import render
from foodcom_pipeline.dashboard.tabs.tab_pipeline_status import render
from foodcom_pipeline.dashboard.theme import inject_theme
print('All imports OK')
"
```
Expected output: `All imports OK`

- [ ] **Step 5: Commit**

```bash
git add src/foodcom_pipeline/dashboard/app.py src/foodcom_pipeline/dashboard/tabs/
git commit -m "refactor: split app.py into tab modules, app.py becomes thin orchestrator"
```

---

## Task 3: Extract + restyle tab_pipeline_status.py

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_pipeline_status.py`

All logic is moved verbatim from `app.py`. Visual changes: emerald palette for charts, status badge colours updated to emerald.

- [ ] **Step 1: Replace `tab_pipeline_status.py` with the full extracted + restyled implementation**

```python
"""Pipeline Status tab — data health monitoring."""

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config (duplicated from app.py constants — each tab is self-contained)
# ---------------------------------------------------------------------------

def _resolve_staging_dir() -> Path:
    if env_val := os.getenv("FOODCOM_STAGING_DIR"):
        return Path(env_val)
    project_root = Path(__file__).resolve().parents[4]
    candidate = project_root / "staging"
    if candidate.exists():
        return candidate
    return Path("/opt/airflow/staging")


STAGING_DIR = _resolve_staging_dir()
ELBOW_STATS_PATH = STAGING_DIR / "elbow_stats.json"
STALE_THRESHOLD_HOURS = 25
DAG_ID = "foodcom_batch_pipeline"

_PG_USER = os.getenv("POSTGRES_USER", "user")
_PG_PASS = os.getenv("POSTGRES_PASSWORD", "password")
_PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
_PG_PORT = os.getenv("POSTGRES_PORT", "5432")
_PG_DB   = os.getenv("POSTGRES_DB", "foodcom")
AIRFLOW_DB_DSN = f"postgresql://{_PG_USER}:{_PG_PASS}@{_PG_HOST}:{_PG_PORT}/{_PG_DB}"

PIPELINE_STAGES: list[dict] = [
    {"stage": "Extract",   "task": "extract_recipes",        "file": "recipes_extracted.parquet",      "label": "Raw recipes"},
    {"stage": "Extract",   "task": "extract_interactions",   "file": "interactions_extracted.parquet", "label": "Raw interactions"},
    {"stage": "Extract",   "task": "extract_usda_nutrients", "file": "usda_nutrients.parquet",         "label": "USDA ingredient rows"},
    {"stage": "Clean",     "task": "clean",                  "file": "recipes_clean.parquet",          "label": "Clean recipes"},
    {"stage": "Clean",     "task": "clean",                  "file": "interactions_clean.parquet",     "label": "Clean interactions"},
    {"stage": "Sentiment", "task": "run_vader_sentiment",    "file": "interactions_sentiment.parquet", "label": "Sentiment-scored interactions"},
    {"stage": "Features",  "task": "features",               "file": "user_stats.parquet",             "label": "Users with features"},
    {"stage": "Features",  "task": "features",               "file": "ingredient_features.parquet",    "label": "Ingredient features"},
    {"stage": "Cluster",   "task": "run_kmeans_clustering",  "file": "user_clusters.parquet",          "label": "Clustered users"},
]

USDA_NUTRIENT_COLS = [
    "calories_per_100g", "protein_g_per_100g", "fat_g_per_100g",
    "saturated_fat_g_per_100g", "sugar_g_per_100g", "sodium_g_per_100g", "carbs_g_per_100g",
]

# Emerald-first state colours
_STATE_COLORS = {
    "success": "#10b981",
    "failed":  "#ef4444",
    "running": "#f59e0b",
    "skipped": "#a7f3d0",
}

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _load_staging_stats() -> list[dict]:
    rows = []
    for entry in PIPELINE_STAGES:
        path = STAGING_DIR / entry["file"]
        if path.exists():
            import pyarrow.parquet as pq
            count = pq.read_metadata(path).num_rows
        else:
            count = None
        rows.append({"Stage": entry["stage"], "Task": entry["task"],
                      "Description": entry["label"], "Rows": count})
    return rows


@st.cache_data(ttl=300)
def _load_usda_coverage() -> dict | None:
    path = STAGING_DIR / "usda_nutrients.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    total = len(df)
    if total == 0:
        return {"total_ingredients": 0, "matched": 0, "coverage_rate": 0.0, "per_nutrient": {}}
    available = [c for c in USDA_NUTRIENT_COLS if c in df.columns]
    matched = int(df[available].notna().any(axis=1).sum()) if available else 0
    per_nutrient = {col: int(df[col].notna().sum()) for col in available}
    return {"total_ingredients": total, "matched": matched,
            "coverage_rate": matched / total, "per_nutrient": per_nutrient}


@st.cache_data(ttl=60)
def _load_airflow_runtimes() -> pd.DataFrame | None:
    try:
        import psycopg2
        conn = psycopg2.connect(AIRFLOW_DB_DSN, connect_timeout=3)
        query = """
            SELECT task_id, duration, state, start_date
            FROM task_instance
            WHERE dag_id = %(dag_id)s
              AND run_id = (
                  SELECT run_id FROM task_instance
                  WHERE dag_id = %(dag_id)s
                  ORDER BY start_date DESC LIMIT 1
              )
            ORDER BY start_date;
        """
        df = pd.read_sql(query, conn, params={"dag_id": DAG_ID})
        conn.close()
        df = df.rename(columns={"duration": "duration_s"})
        df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
        return df
    except Exception:
        return None


@st.cache_data(ttl=60)
def _load_data_freshness() -> dict | None:
    from datetime import datetime
    mtimes = [
        (STAGING_DIR / e["file"]).stat().st_mtime
        for e in PIPELINE_STAGES
        if (STAGING_DIR / e["file"]).exists()
    ]
    if not mtimes:
        return None
    last_run = datetime.fromtimestamp(max(mtimes))
    return {"last_run": last_run, "hours_ago": (datetime.now() - last_run).total_seconds() / 3600}


@st.cache_data(ttl=300)
def _load_sentiment_coverage() -> dict | None:
    path = STAGING_DIR / "interactions_sentiment.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["sentiment_score"])
    total = len(df)
    scored = int(df["sentiment_score"].notna().sum())
    return {"total": total, "scored": scored, "coverage": scored / total if total > 0 else 0.0}


@st.cache_data(ttl=300)
def _load_clustering_health() -> dict | None:
    import json
    if not ELBOW_STATS_PATH.exists():
        return None
    with open(ELBOW_STATS_PATH) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def _load_substitution_stats() -> dict | None:
    path = STAGING_DIR / "ingredient_features.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["is_substitution_candidate"])
    total = len(df)
    candidates = int(df["is_substitution_candidate"].sum())
    return {"total_ingredients": total, "candidates": candidates,
            "rate": candidates / total if total > 0 else 0.0}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render() -> None:
    st.header("⚙️ Pipeline Status")

    freshness   = _load_data_freshness()
    stats       = _load_staging_stats()
    usda        = _load_usda_coverage()
    runtimes    = _load_airflow_runtimes()
    sentiment   = _load_sentiment_coverage()
    clustering  = _load_clustering_health()
    sub_stats   = _load_substitution_stats()

    # Freshness banner
    if freshness is None:
        st.warning("No staging files found. Run the batch pipeline first.")
    else:
        h = freshness["hours_ago"]
        ts = freshness["last_run"].strftime("%Y-%m-%d %H:%M:%S")
        if h > STALE_THRESHOLD_HOURS:
            st.warning(f"Pipeline data is stale — last run was **{h:.1f} hours ago** ({ts}).")
        else:
            st.success(f"Pipeline up to date — last run {h:.1f} h ago ({ts}).")

    st.divider()

    # KPI row 1: volume & loss
    def _rows(label: str) -> int | None:
        return next((r["Rows"] for r in stats if r["Description"].startswith(label)), None)

    raw_r, clean_r = _rows("Raw recipes"), _rows("Clean recipes")
    raw_i, clean_i = _rows("Raw interactions"), _rows("Clean interactions")
    recipe_loss = (raw_r - clean_r) / raw_r if raw_r and clean_r and raw_r > 0 else None
    int_loss    = (raw_i - clean_i) / raw_i if raw_i and clean_i and raw_i > 0 else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clean Recipes",      f"{clean_r:,}"       if clean_r else "—")
    c2.metric("Clean Interactions", f"{clean_i:,}"       if clean_i else "—")
    c3.metric("Recipe Loss",        f"{recipe_loss:.2%}" if recipe_loss is not None else "—",
              delta=f"-{raw_r - clean_r:,} rows" if recipe_loss is not None else None, delta_color="inverse")
    c4.metric("Interaction Loss",   f"{int_loss:.2%}"    if int_loss is not None else "—",
              delta=f"-{raw_i - clean_i:,} rows" if int_loss is not None else None, delta_color="inverse")

    # KPI row 2: quality signals
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("USDA Coverage",          f"{usda['coverage_rate']:.1%}"  if usda        else "—")
    c6.metric("Sentiment Coverage",     f"{sentiment['coverage']:.1%}"  if sentiment   else "—")
    c7.metric("Substitution Candidates",f"{sub_stats['candidates']:,}"  if sub_stats   else "—",
              delta=f"{sub_stats['rate']:.1%} of ingredients" if sub_stats else None, delta_color="off")
    c8.metric("Optimal k (clusters)",   str(clustering["chosen_k"])     if clustering  else "—")

    st.divider()

    # Row counts table
    st.subheader("Row Counts by Stage")
    df_stats = pd.DataFrame(stats)
    df_stats["Rows"] = df_stats["Rows"].apply(lambda x: f"{x:,}" if x is not None else "— (not yet produced)")
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    st.divider()

    # Clustering health
    st.subheader("Clustering Health")
    if clustering is None:
        st.info("`elbow_stats.json` not found — run the clustering step first.")
    else:
        chosen_k = clustering["chosen_k"]
        colors = ["#10b981" if k == chosen_k else "#6ee7b7" for k in clustering["k_range"]]
        fig = go.Figure(go.Bar(
            x=[str(k) for k in clustering["k_range"]],
            y=clustering["silhouettes"],
            marker_color=colors,
            text=[f"{s:.4f}" for s in clustering["silhouettes"]],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="k (number of clusters)", yaxis_title="Silhouette Score",
            title=f"Silhouette Scores — chosen k={chosen_k} (dark green)",
            height=300, margin=dict(t=40, b=40),
            yaxis=dict(range=[0, max(clustering["silhouettes"]) * 1.2]),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # USDA coverage breakdown
    st.subheader("USDA Nutrient Coverage")
    if usda is None:
        st.warning("`usda_nutrients.parquet` not found.")
    else:
        st.caption(f"{usda['matched']:,} of {usda['total_ingredients']:,} ingredients matched ({usda['coverage_rate']:.1%}).")
        if usda["per_nutrient"]:
            nutrient_df = pd.DataFrame([
                {"Nutrient": col.replace("_per_100g", "").replace("_", " ").title(),
                 "Matched": count, "Coverage": f"{count / usda['total_ingredients']:.1%}"}
                for col, count in usda["per_nutrient"].items()
            ])
            st.dataframe(nutrient_df, use_container_width=True, hide_index=True)

    st.divider()

    # Airflow runtimes
    st.subheader("Airflow Task Runtimes (most recent run)")
    if runtimes is None:
        st.warning(f"Could not connect to Airflow metadata DB at `{_PG_HOST}:{_PG_PORT}`.")
    elif runtimes.empty:
        st.info(f"No task runs found for DAG `{DAG_ID}`.")
    else:
        bar_colors = [_STATE_COLORS.get(str(s), "#6ee7b7") for s in runtimes["state"]]
        fig = go.Figure(go.Bar(
            x=runtimes["duration_s"].fillna(0).round(1),
            y=runtimes["task_id"],
            orientation="h",
            marker_color=bar_colors,
            text=runtimes["duration_s"].apply(lambda s: f"{s:.0f}s" if pd.notna(s) else "—"),
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Duration (seconds)", yaxis=dict(autorange="reversed"),
            height=max(300, 50 * len(runtimes)), margin=dict(l=10, r=60, t=20, b=40),
        )
        legend_cols = st.columns(len(_STATE_COLORS))
        for col, (state, color) in zip(legend_cols, _STATE_COLORS.items()):
            col.markdown(f'<span style="color:{color}">■</span> {state}', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

        summary = runtimes[["task_id", "state", "duration_s", "start_date"]].copy()
        summary["duration_s"] = summary["duration_s"].apply(lambda s: f"{s:.1f}s" if pd.notna(s) else "—")
        summary.columns = ["Task", "State", "Duration", "Started"]
        st.dataframe(summary, use_container_width=True, hide_index=True)
```

- [ ] **Step 2: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_pipeline_status.py
git commit -m "feat: extract pipeline status tab to module, apply emerald theme"
```

---

## Task 4: Extract + restyle tab_market_intelligence.py

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_market_intelligence.py`

- [ ] **Step 1: Replace `tab_market_intelligence.py` with the full extracted + restyled implementation**

```python
"""Audience & Market Intelligence tab — CPG segment advertising tool."""

import io
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _resolve_staging_dir() -> Path:
    if env_val := os.getenv("FOODCOM_STAGING_DIR"):
        return Path(env_val)
    project_root = Path(__file__).resolve().parents[4]
    candidate = project_root / "staging"
    if candidate.exists():
        return candidate
    return Path("/opt/airflow/staging")


STAGING_DIR = _resolve_staging_dir()
CLUSTER_PROFILE_PATH = STAGING_DIR / "cluster_profiles.json"

RADAR_FEATURES = ["avg_rating_dairy", "avg_rating_protein", "avg_rating_vegetable",
                  "avg_rating_baking", "avg_rating_international"]
RADAR_LABELS   = ["Dairy", "Protein", "Vegetable", "Baking", "International"]

# Emerald palette for segment colours
SEGMENT_COLORS = ["#10b981", "#059669", "#34d399", "#6ee7b7", "#a7f3d0"]

CPG_ADJACENCY: dict[str, dict] = {
    "Indulgent Baker":       {"top_categories": ["baking", "dairy"],
                              "brands": ["King Arthur Baking", "Ghirardelli", "Domino Sugar", "Land O'Lakes"],
                              "cpm_range": "$4 – $8"},
    "International Explorer":{"top_categories": ["international", "protein"],
                              "brands": ["Kikkoman", "Blue Dragon", "Goya Foods", "McCormick"],
                              "cpm_range": "$5 – $9"},
    "Protein-Forward Cook":  {"top_categories": ["protein", "dairy"],
                              "brands": ["Tyson Foods", "Beyond Meat", "Kirkland Signature", "Applegate"],
                              "cpm_range": "$6 – $10"},
    "Health-Conscious Cook": {"top_categories": ["vegetable", "international"],
                              "brands": ["Amy's Kitchen", "Whole Foods 365", "Green Giant", "Earthbound Farm"],
                              "cpm_range": "$7 – $12"},
    "General Cook":          {"top_categories": ["dairy", "vegetable"],
                              "brands": ["Heinz", "Kraft", "Campbell's", "Birds Eye"],
                              "cpm_range": "$3 – $6"},
}


@st.cache_data(ttl=300)
def _load_cluster_profiles() -> dict | None:
    if not CLUSTER_PROFILE_PATH.exists():
        return None
    with open(CLUSTER_PROFILE_PATH) as f:
        return json.load(f)


def _radar_chart(profiles: dict, selected_labels: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, (_, profile) in enumerate(profiles.items()):
        label = profile["cluster_label"]
        if label not in selected_labels:
            continue
        stats = profile.get("feature_stats", {})
        values = [stats.get(f, {}).get("mean", 0.0) for f in RADAR_FEATURES]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=RADAR_LABELS + [RADAR_LABELS[0]],
            fill="toself",
            name=label,
            line_color=SEGMENT_COLORS[i % len(SEGMENT_COLORS)],
            opacity=0.65,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[3.5, 5.0])),
        showlegend=True,
        title="Taste Profile Radar — Avg Rating per Ingredient Category",
        height=480,
    )
    return fig


def _segment_profile_table(profiles: dict, selected_labels: list[str]) -> pd.DataFrame:
    rows = []
    for profile in profiles.values():
        if profile["cluster_label"] not in selected_labels:
            continue
        rows.append({
            "Segment": profile["cluster_label"],
            "Users": f"{profile['n_users']:,}",
            "% of Total": f"{profile['pct_users']:.1f}%",
            "Peak Month": profile.get("peak_month") or "—",
            "Top Ingredients": ", ".join(profile.get("top_ingredients") or []) or "—",
            "Avg ! per Review": (f"{profile['avg_exclamation_count']:.2f}"
                                  if profile.get("avg_exclamation_count") is not None else "—"),
            "Sub. Exposure": (f"{profile['substitute_exposure_rate']:.1%}"
                               if profile.get("substitute_exposure_rate") is not None else "—"),
        })
    return pd.DataFrame(rows)


def _brand_adjacency_table(selected_labels: list[str]) -> pd.DataFrame:
    rows = []
    for label, data in CPG_ADJACENCY.items():
        if label not in selected_labels:
            continue
        rows.append({
            "Segment": label,
            "Key Categories": ", ".join(data["top_categories"]),
            "Recommended CPG Brands": ", ".join(data["brands"]),
            "Est. CPM Range": data["cpm_range"],
        })
    return pd.DataFrame(rows)


def _generate_pdf(profiles: dict, selected_labels: list[str]) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError:
        return b""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Food.com — Audience & Market Intelligence", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "CPG Advertising Segment Report", ln=True, align="C")
    pdf.ln(6)
    for profile in profiles.values():
        label = profile["cluster_label"]
        if label not in selected_labels:
            continue
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"{label}  ({profile['n_users']:,} users, {profile['pct_users']:.1f}%)", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  Top ingredients : {', '.join(profile.get('top_ingredients') or []) or '—'}", ln=True)
        pdf.cell(0, 6, f"  Peak month      : {profile.get('peak_month') or '—'}", ln=True)
        sub = profile.get("substitute_exposure_rate")
        pdf.cell(0, 6, f"  Sub. exposure   : {sub:.1%}" if sub is not None else "  Sub. exposure   : —", ln=True)
        adj = CPG_ADJACENCY.get(label, {})
        if adj:
            pdf.cell(0, 6, f"  CPG brands      : {', '.join(adj.get('brands', []))}", ln=True)
            pdf.cell(0, 6, f"  Est. CPM        : {adj.get('cpm_range', '—')}", ln=True)
        pdf.ln(4)
    return bytes(pdf.output())


def render() -> None:
    st.header("📊 Audience & Market Intelligence")

    profiles_data = _load_cluster_profiles()
    if profiles_data is None:
        st.warning(f"No cluster profiles found at `{CLUSTER_PROFILE_PATH}`. Run the batch pipeline first.")
        return

    profiles = profiles_data.get("clusters", {})
    all_labels = [p["cluster_label"] for p in profiles.values()]

    st.subheader("Segment Filter")
    selected_labels = st.multiselect("Select segments to display", options=all_labels, default=all_labels)
    if not selected_labels:
        st.warning("Select at least one segment.")
        return

    # KPI row — metric cards styled by theme.py
    total_users = profiles_data.get("total_users", 0)
    n_clusters  = profiles_data.get("n_clusters", len(profiles))
    c1, c2 = st.columns(2)
    c1.metric("Total Segmented Users", f"{total_users:,}")
    c2.metric("Number of Segments", n_clusters)

    st.divider()

    st.subheader("Taste Profile Radar")
    st.plotly_chart(_radar_chart(profiles, selected_labels), use_container_width=True)

    st.divider()

    st.subheader("Segment Profiles")
    st.dataframe(_segment_profile_table(profiles, selected_labels), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("CPG Brand Adjacency")
    st.caption("Static mapping of segments to recommended CPG brands and estimated programmatic CPM ranges.")
    brand_df = _brand_adjacency_table(selected_labels)
    # Highlight CPM column in green via styled HTML
    st.dataframe(brand_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Export")
    pdf_bytes = _generate_pdf(profiles, selected_labels)
    if pdf_bytes:
        st.download_button("Download PDF Report", data=pdf_bytes,
                            file_name="foodcom_audience_report.pdf", mime="application/pdf")
    else:
        st.info("PDF export requires `fpdf2`. Install with `uv add fpdf2`.")
```

- [ ] **Step 2: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_market_intelligence.py
git commit -m "feat: extract market intelligence tab to module, apply emerald palette"
```

---

## Task 5: Build tab_substitutions.py (Coming Soon stub)

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_substitutions.py`

- [ ] **Step 1: Replace `tab_substitutions.py` with polished Coming Soon card**

```python
"""Smart Substitutions tab — polished Coming Soon stub."""

import streamlit as st


def render() -> None:
    st.header("🔄 Smart Substitutions")

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #f0fdf4, #dcfce7);
            border: 2px dashed #10b981;
            border-radius: 12px;
            padding: 48px 32px;
            text-align: center;
            margin-top: 32px;
        ">
            <div style="font-size: 48px; margin-bottom: 16px;">🔄</div>
            <h2 style="color: #065f46; margin-bottom: 12px;">Smart Substitutions</h2>
            <p style="color: #374151; max-width: 480px; margin: 0 auto; line-height: 1.8; font-size: 15px;">
                Ingredient swap cards with rating and nutritional deltas, real-time
                Bayesian recalculation, and native ad placement hooks.
                Detects underperforming ingredients and recommends trending alternatives.
            </p>
            <div style="
                display: inline-block;
                margin-top: 24px;
                background: #10b981;
                color: white;
                border-radius: 8px;
                padding: 10px 24px;
                font-weight: 700;
                font-size: 14px;
            ">Coming Soon</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_substitutions.py
git commit -m "feat: add polished Coming Soon stub for Smart Substitutions tab"
```

---

## Task 6: Recipe Explorer — URL builders (TDD)

**Files:**
- Create: `tests/dashboard/__init__.py`
- Create: `tests/dashboard/test_recipe_explorer.py`
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

- [ ] **Step 1: Create `tests/dashboard/__init__.py`**

Empty file.

- [ ] **Step 2: Write failing tests for `build_amazon_url` and `build_instacart_url`**

`tests/dashboard/test_recipe_explorer.py` — write the full file including all imports needed for both Task 6 and Task 7 tests:
```python
import pandas as pd
import pytest
from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import (
    apply_filters,
    build_amazon_url,
    build_instacart_url,
    nutrition_bar_color,
)


class TestBuildAmazonUrl:
    def test_basic(self):
        url = build_amazon_url(["chicken", "garlic", "tomato"])
        assert url.startswith("https://www.amazon.com/s?k=")
        assert "amazonfresh" in url
        assert "chicken" in url
        assert "garlic" in url

    def test_caps_at_eight_ingredients(self):
        ingredients = [f"ingredient{i}" for i in range(15)]
        url = build_amazon_url(ingredients)
        # Only first 8 should appear
        for i in range(8):
            assert f"ingredient{i}" in url
        assert "ingredient8" not in url

    def test_spaces_encoded(self):
        url = build_amazon_url(["soy sauce", "fish sauce"])
        assert " " not in url
        assert "soy" in url

    def test_empty_list_returns_base_url(self):
        url = build_amazon_url([])
        assert "amazon.com" in url


class TestBuildInstacartUrl:
    def test_basic(self):
        url = build_instacart_url("Chicken Tikka Masala")
        assert url.startswith("https://www.instacart.com/store/s?k=")
        assert "Chicken" in url or "chicken" in url.lower()

    def test_spaces_encoded(self):
        url = build_instacart_url("beef stir fry")
        assert " " not in url

    def test_ingredients_keyword_appended(self):
        url = build_instacart_url("pasta carbonara")
        assert "ingredient" in url.lower()
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
cd "/Users/madhupolluru/Downloads/AY25:26 SEM 2/IS3107/Project/DataEngineeringProject"
python -m pytest tests/dashboard/test_recipe_explorer.py -v 2>&1 | head -30
```
Expected: `ImportError` or `AttributeError` — functions don't exist yet.

- [ ] **Step 4: Implement `build_amazon_url` and `build_instacart_url` in `tab_recipe_explorer.py`**

Replace the stub content of `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py` with:

```python
"""Recipe Explorer tab — hero search, recipe cards, Affiliate Commerce Shop buttons."""

import os
import urllib.parse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _resolve_staging_dir() -> Path:
    if env_val := os.getenv("FOODCOM_STAGING_DIR"):
        return Path(env_val)
    project_root = Path(__file__).resolve().parents[4]
    candidate = project_root / "staging"
    if candidate.exists():
        return candidate
    return Path("/opt/airflow/staging")


STAGING_DIR = _resolve_staging_dir()

_PG_USER = os.getenv("POSTGRES_USER", "user")
_PG_PASS = os.getenv("POSTGRES_PASSWORD", "password")
_PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
_PG_PORT = os.getenv("POSTGRES_PORT", "5432")
_PG_DB   = os.getenv("POSTGRES_DB", "foodcom")
_DB_DSN  = f"postgresql://{_PG_USER}:{_PG_PASS}@{_PG_HOST}:{_PG_PORT}/{_PG_DB}"

PAGE_SIZE = 10

# Nutrition axes shown on the radar (all Food.com %DV columns)
_RADAR_NUTRIENTS = ["protein", "fat", "carbs", "sugar", "sodium"]
_RADAR_LABELS    = ["Protein %DV", "Fat %DV", "Carbs %DV", "Sugar %DV", "Sodium %DV"]


# ---------------------------------------------------------------------------
# Pure helpers (independently testable — no Streamlit dependency)
# ---------------------------------------------------------------------------

def build_amazon_url(ingredients: list[str]) -> str:
    """Build an Amazon Fresh search URL for the first 8 ingredients."""
    top = ingredients[:8]
    if not top:
        return "https://www.amazon.com/s?i=amazonfresh"
    query = "+".join(urllib.parse.quote_plus(ing) for ing in top)
    return f"https://www.amazon.com/s?k={query}&i=amazonfresh"


def build_instacart_url(recipe_name: str) -> str:
    """Build an Instacart search URL for a recipe name."""
    query = urllib.parse.quote_plus(f"{recipe_name} ingredients")
    return f"https://www.instacart.com/store/s?k={query}"


def apply_filters(
    df: pd.DataFrame,
    search: str,
    max_cook_time: int,
    min_rating: float,
    selected_tags: list[str],
) -> pd.DataFrame:
    """Filter the recipe DataFrame by search text, cook time, rating, and tags."""
    if search:
        mask = (
            df["name"].str.contains(search, case=False, na=False)
            | df["top_ingredients"].str.contains(search, case=False, na=False)
        )
        df = df[mask]
    df = df[df["avg_cook_minutes"].fillna(0) <= max_cook_time]
    df = df[df["display_rating"].fillna(0) >= min_rating]
    if selected_tags:
        def _has_tag(tag_str: str) -> bool:
            return any(t.lower() in str(tag_str).lower() for t in selected_tags)
        df = df[df["tags"].apply(_has_tag)]
    return df


def nutrition_bar_color(pct_dv: float, nutrient: str) -> str:
    """Return an emerald/amber/red hex colour based on nutrient and %DV level."""
    if nutrient == "sugar" and pct_dv > 30:
        return "#f59e0b"  # amber — high sugar
    if nutrient == "sodium" and pct_dv > 50:
        return "#ef4444"  # red — high sodium
    return "#10b981"      # emerald — default
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
python -m pytest tests/dashboard/test_recipe_explorer.py::TestBuildAmazonUrl tests/dashboard/test_recipe_explorer.py::TestBuildInstacartUrl -v
```
Expected: All 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/dashboard/ src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: add build_amazon_url and build_instacart_url helpers with tests"
```

---

## Task 7: Recipe Explorer — filter logic + nutrition colour (TDD)

**Files:**
- Modify: `tests/dashboard/test_recipe_explorer.py`
- (tab_recipe_explorer.py already has the implementations from Task 6)

- [ ] **Step 1: Add tests for `apply_filters` and `nutrition_bar_color`**

Append the following classes to `tests/dashboard/test_recipe_explorer.py` (imports are already at the top from Task 6 — do not add them again):

```python
def _make_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"name": "Chicken Tikka", "top_ingredients": "chicken|tomato|cream",
         "avg_cook_minutes": 45.0, "display_rating": 4.8,
         "tags": "Indian, curry, spicy"},
        {"name": "Spaghetti",     "top_ingredients": "pasta|egg|bacon",
         "avg_cook_minutes": 20.0, "display_rating": 4.2,
         "tags": "Italian, quick"},
        {"name": "Veggie Stir Fry","top_ingredients": "broccoli|soy sauce|garlic",
         "avg_cook_minutes": 15.0, "display_rating": 3.9,
         "tags": "Asian, vegetarian"},
    ])


class TestApplyFilters:
    def test_search_by_name(self):
        result = apply_filters(_make_df(), "chicken", 180, 1.0, [])
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Chicken Tikka"

    def test_search_by_ingredient(self):
        result = apply_filters(_make_df(), "pasta", 180, 1.0, [])
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Spaghetti"

    def test_cook_time_filter(self):
        result = apply_filters(_make_df(), "", 20, 1.0, [])
        assert len(result) == 2
        assert all(result["avg_cook_minutes"] <= 20)

    def test_min_rating_filter(self):
        result = apply_filters(_make_df(), "", 180, 4.5, [])
        assert len(result) == 1
        assert result.iloc[0]["display_rating"] >= 4.5

    def test_tag_filter(self):
        result = apply_filters(_make_df(), "", 180, 1.0, ["Italian"])
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Spaghetti"

    def test_no_filters_returns_all(self):
        df = _make_df()
        result = apply_filters(df, "", 9999, 0.0, [])
        assert len(result) == len(df)

    def test_search_case_insensitive(self):
        result = apply_filters(_make_df(), "CHICKEN", 180, 1.0, [])
        assert len(result) == 1


class TestNutritionBarColor:
    def test_high_sugar_is_amber(self):
        assert nutrition_bar_color(35.0, "sugar") == "#f59e0b"

    def test_low_sugar_is_emerald(self):
        assert nutrition_bar_color(10.0, "sugar") == "#10b981"

    def test_high_sodium_is_red(self):
        assert nutrition_bar_color(60.0, "sodium") == "#ef4444"

    def test_low_sodium_is_emerald(self):
        assert nutrition_bar_color(20.0, "sodium") == "#10b981"

    def test_protein_always_emerald(self):
        assert nutrition_bar_color(80.0, "protein") == "#10b981"

    def test_fat_always_emerald(self):
        assert nutrition_bar_color(90.0, "fat") == "#10b981"
```

- [ ] **Step 2: Run tests — verify they pass**

```bash
python -m pytest tests/dashboard/test_recipe_explorer.py -v
```
Expected: All 20 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/dashboard/test_recipe_explorer.py
git commit -m "test: add apply_filters and nutrition_bar_color tests"
```

---

## Task 8: Recipe Explorer — data loader

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

- [ ] **Step 1: Add `load_recipes()` to `tab_recipe_explorer.py`**

Append this function after the pure helpers section (before the closing of the file):

```python
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_recipes() -> pd.DataFrame:
    """
    Load recipe data. Tries PostgreSQL dim_recipe first; falls back to parquet.
    Returns a DataFrame with a normalised schema. Returns empty DataFrame on failure.
    """
    # Primary: PostgreSQL dim_recipe
    try:
        import psycopg2
        conn = psycopg2.connect(_DB_DSN, connect_timeout=3)
        df = pd.read_sql("""
            SELECT recipe_id, name,
                   COALESCE(avg_rating, 0)      AS avg_rating,
                   sentiment_rating,
                   COALESCE(avg_cook_minutes, 0) AS avg_cook_minutes,
                   COALESCE(top_ingredients, '') AS top_ingredients,
                   COALESCE(tags, '')            AS tags,
                   COALESCE(ingredient_count, 0) AS ingredient_count,
                   calories, protein, fat, sugar, sodium, carbs, saturated_fat
            FROM dim_recipe
            ORDER BY COALESCE(sentiment_rating, avg_rating) DESC NULLS LAST
        """, conn)
        conn.close()
        if not df.empty:
            df["display_rating"] = df["sentiment_rating"].fillna(df["avg_rating"])
            return df
    except Exception:
        pass

    # Fallback: parquet staging files
    recipes_path   = STAGING_DIR / "recipes_clean.parquet"
    sentiment_path = STAGING_DIR / "recipe_sentiment_ratings.parquet"

    if not recipes_path.exists():
        return pd.DataFrame()

    nutrition_cols = ["calories", "protein", "fat", "sugar", "sodium", "carbs", "saturated_fat"]
    keep = ["id", "name", "minutes", "tags", "ingredients", "n_ingredients"] + nutrition_cols
    available = [c for c in keep if c in pd.read_parquet(recipes_path, columns=[]).columns]

    df = pd.read_parquet(recipes_path, columns=available).rename(columns={
        "id": "recipe_id",
        "minutes": "avg_cook_minutes",
        "n_ingredients": "ingredient_count",
    })

    # Build top_ingredients from the ingredients list column
    if "ingredients" in df.columns:
        df["top_ingredients"] = df["ingredients"].apply(
            lambda x: "|".join(str(i) for i in x[:10]) if isinstance(x, list) else ""
        )
    else:
        df["top_ingredients"] = ""

    df["avg_rating"] = float("nan")

    if sentiment_path.exists():
        sent = pd.read_parquet(sentiment_path, columns=["recipe_id", "sentiment_rating"])
        df = df.merge(sent, on="recipe_id", how="left")
    else:
        df["sentiment_rating"] = float("nan")

    df["display_rating"] = df["sentiment_rating"].fillna(df["avg_rating"])
    df = df.sort_values("display_rating", ascending=False, na_position="last")

    # Ensure all expected nutrition columns exist (fill missing with NaN)
    for col in nutrition_cols:
        if col not in df.columns:
            df[col] = float("nan")

    return df
```

- [ ] **Step 2: Verify the loader imports without error**

```bash
python -c "
from foodcom_pipeline.dashboard.tabs.tab_recipe_explorer import load_recipes
print('load_recipes importable OK')
"
```
Expected: `load_recipes importable OK`

- [ ] **Step 3: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: add load_recipes() with PostgreSQL primary and parquet fallback"
```

---

## Task 9: Recipe Explorer — full render() UI

**Files:**
- Modify: `src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py`

- [ ] **Step 1: Add `_nutrition_radar()` and `_render_recipe_card()` helpers to `tab_recipe_explorer.py`**

Append these functions after `load_recipes()`:

```python
# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _nutrition_radar(row: pd.Series) -> go.Figure:
    """Plotly polar chart for 5 nutrition axes (%DV values)."""
    values = [float(row.get(n) or 0) for n in _RADAR_NUTRIENTS]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=_RADAR_LABELS + [_RADAR_LABELS[0]],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.15)",
        line=dict(color="#10b981", width=2),
        name="%DV",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9))),
        showlegend=False,
        height=220,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    return fig


def _render_recipe_card(row: pd.Series) -> None:
    """Render one full-width recipe card inside a styled container."""
    name          = str(row.get("name", "Unknown"))
    cook_min      = row.get("avg_cook_minutes")
    n_ingredients = row.get("ingredient_count")
    bayesian      = row.get("sentiment_rating")
    raw_rating    = row.get("avg_rating")
    display       = row.get("display_rating")
    top_ingr_str  = str(row.get("top_ingredients") or "")
    ingredients   = [i.strip() for i in top_ingr_str.split("|") if i.strip()]
    trend_index   = row.get("trend_index")  # None/NaN when Google Trends not yet wired

    amazon_url    = build_amazon_url(ingredients)
    instacart_url = build_instacart_url(name)

    with st.container():
        st.markdown(
            '<div style="background:white;border-radius:10px;border:1px solid #e5e7eb;'
            'padding:20px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">',
            unsafe_allow_html=True,
        )

        # --- Top row: name + shop buttons ---
        col_info, col_shop = st.columns([3, 1])

        with col_info:
            subtitle_parts = []
            if cook_min is not None and not pd.isna(cook_min):
                subtitle_parts.append(f"⏱ {int(cook_min)} min")
            if n_ingredients is not None and not pd.isna(n_ingredients):
                subtitle_parts.append(f"{int(n_ingredients)} ingredients")
            st.markdown(f"### {name}")
            if subtitle_parts:
                st.caption(" · ".join(subtitle_parts))

            # Rating badges
            badge_parts = []
            if bayesian is not None and not pd.isna(bayesian):
                badge_parts.append(
                    f'<span style="background:#10b981;color:white;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;font-weight:700;">⭐ {bayesian:.1f} Bayesian</span>'
                )
            if raw_rating is not None and not pd.isna(raw_rating):
                badge_parts.append(
                    f'<span style="background:#f3f4f6;color:#374151;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;">{raw_rating:.1f} raw</span>'
                )
            if (bayesian is not None and not pd.isna(bayesian)
                    and raw_rating is not None and not pd.isna(raw_rating)):
                delta = bayesian - raw_rating
                if delta > 0:
                    badge_parts.append(
                        f'<span style="color:#059669;font-size:12px;font-weight:600;">'
                        f'↑ +{delta:.1f} sentiment boost</span>'
                    )
            if trend_index is not None and not pd.isna(trend_index):
                level = "🔥 High" if trend_index > 66 else ("📈 Medium" if trend_index > 33 else "📉 Low")
                badge_parts.append(
                    f'<span style="background:#fef3c7;color:#92400e;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;">{level} trend demand</span>'
                )
            if badge_parts:
                st.markdown(" &nbsp; ".join(badge_parts), unsafe_allow_html=True)

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

        # --- Bottom row: radar | nutrition bars | ingredient pills ---
        col_radar, col_bars, col_pills = st.columns([1, 1.5, 1])

        with col_radar:
            st.plotly_chart(_nutrition_radar(row), use_container_width=True, key=f"radar_{name[:20]}")

        with col_bars:
            st.markdown("**Nutrition (% Daily Value)**")
            for nutrient, label in zip(_RADAR_NUTRIENTS, _RADAR_LABELS):
                val = row.get(nutrient)
                if val is None or pd.isna(val):
                    continue
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

        with col_pills:
            st.markdown("**Ingredients**")
            if ingredients:
                pills_html = "".join(
                    f'<span style="display:inline-block;background:#f3f4f6;border-radius:12px;'
                    f'padding:3px 10px;font-size:11px;color:#6b7280;margin:2px;">{ing}</span>'
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

        st.markdown("</div>", unsafe_allow_html=True)
```

- [ ] **Step 2: Add the `render()` function to `tab_recipe_explorer.py`**

Append after `_render_recipe_card()`:

```python
# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    # Hero header
    st.markdown(
        '<div style="background:linear-gradient(135deg,#10b981,#059669);border-radius:10px;'
        'padding:24px 28px;margin-bottom:16px;">'
        '<h1 style="color:white;margin:0;font-size:26px;">🍳 Recipe Explorer</h1>'
        '<p style="color:#d1fae5;margin:6px 0 14px;">Search 231,637 recipes — '
        'see Bayesian ratings, nutrition, and shop ingredients in one click.</p>',
        unsafe_allow_html=True,
    )
    search = st.text_input("", placeholder="Search recipes or ingredients...", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    df = load_recipes()
    if df.empty:
        st.warning(
            "No recipe data found. Run the batch pipeline first "
            "(or set `FOODCOM_STAGING_DIR` to your staging directory)."
        )
        return

    # Filter strip
    filter_col1, filter_col2, filter_col3 = st.columns([1.5, 1, 1.5])
    with filter_col1:
        max_cook = st.slider("Max cook time (min)", 5, 180, 60, step=5)
    with filter_col2:
        min_rating = st.select_slider("Min Bayesian rating", options=[1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0], value=3.0)
    with filter_col3:
        all_tags = sorted({
            tag.strip()
            for tags_str in df["tags"].dropna()
            for tag in str(tags_str).split(",")
            if tag.strip()
        })[:40]
        selected_tags = st.multiselect("Cuisine / tags", options=all_tags, default=[])

    filtered = apply_filters(df, search, max_cook, min_rating, selected_tags)
    total = len(filtered)

    # Pagination state
    filter_hash = hash((search, max_cook, min_rating, tuple(sorted(selected_tags))))
    if st.session_state.get("_recipe_filter_hash") != filter_hash:
        st.session_state["_recipe_page"] = 0
        st.session_state["_recipe_filter_hash"] = filter_hash
    page = st.session_state.get("_recipe_page", 0)
    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, n_pages - 1))

    # Result count
    start = page * PAGE_SIZE
    end   = min(start + PAGE_SIZE, total)
    st.caption(f"{total:,} recipes · showing {start + 1}–{end}" if total else "No recipes match your filters.")

    # Recipe cards
    page_df = filtered.iloc[start:end]
    for _, row in page_df.iterrows():
        _render_recipe_card(row)

    # Pagination controls
    pc1, pc2, pc3 = st.columns([1, 2, 1])
    with pc1:
        if page > 0:
            if st.button("← Previous", key="prev_page"):
                st.session_state["_recipe_page"] = page - 1
                st.rerun()
    with pc2:
        st.caption(f"Page {page + 1} of {n_pages}")
    with pc3:
        if page < n_pages - 1:
            if st.button("Next →", key="next_page"):
                st.session_state["_recipe_page"] = page + 1
                st.rerun()
```

- [ ] **Step 3: Run the full test suite to confirm nothing broke**

```bash
python -m pytest tests/dashboard/test_recipe_explorer.py -v
```
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/foodcom_pipeline/dashboard/tabs/tab_recipe_explorer.py
git commit -m "feat: build Recipe Explorer tab with hero search, nutrition radar, and Shop buttons"
```

---

## Task 10: Smoke test + final cleanup

**Files:**
- Verify: `src/foodcom_pipeline/dashboard/app.py` (confirm no old tab functions remain)

- [ ] **Step 1: Confirm app.py has no leftover code from old tabs**

The refactored `app.py` written in Task 2 already only contains the thin orchestrator. Verify:

```bash
python -c "
import ast, pathlib
tree = ast.parse(pathlib.Path(
    'src/foodcom_pipeline/dashboard/app.py'
).read_text())
funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
print('Functions in app.py:', funcs)
assert 'tab_audience_market' not in funcs, 'old tab function still present'
assert 'tab_pipeline_status' not in funcs, 'old tab function still present'
print('OK — app.py is clean')
"
```
Expected: `Functions in app.py: ['main']` and `OK — app.py is clean`

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: All tests PASS (includes `tests/test_trends.py` + `tests/dashboard/test_recipe_explorer.py`).

- [ ] **Step 3: Start the dashboard and do a visual smoke test**

```bash
FOODCOM_STAGING_DIR=./staging streamlit run src/foodcom_pipeline/dashboard/app.py
```

Open `http://localhost:8501` and verify:
- Tab bar shows: 🍳 Recipe Explorer · 🔄 Smart Substitutions · 📊 Market Intelligence · ⚙️ Pipeline Status
- Tab 1: Hero search bar renders, filter strip visible, recipe cards appear (or "Run pipeline first" warning if no data)
- Tab 2: Emerald "Coming Soon" card visible
- Tab 3: Radar chart and CPG table visible with emerald colours
- Tab 4: Metric cards have emerald top border, charts use emerald palette

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "feat: Recipe Explorer + full dashboard restyle complete"
```
