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
