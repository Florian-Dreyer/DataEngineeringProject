"""
app.py
------
Streamlit dashboard for the Food.com Lambda Architecture pipeline.

Tabs:
  1. Overview            — pipeline run summary and key metrics  [stub]
  2. Recipe Analytics    — top recipes, sentiment ratings        [stub]
  3. Audience & Market Intelligence — CPG advertising segments   [IMPLEMENTED]
  4. Pipeline Status     — row counts, data loss, USDA coverage, Airflow runtimes [IMPLEMENTED]

Run locally (from project root):
  FOODCOM_STAGING_DIR=./staging streamlit run src/foodcom_pipeline/dashboard/app.py

The dashboard reads from the staging parquet/JSON files written by the batch pipeline.
Tab 4 additionally queries the Airflow metadata PostgreSQL for task runtimes.
"""

import io
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _resolve_staging_dir() -> Path:
    # 1. Explicit env var always wins (useful inside Docker or CI)
    if env_val := os.getenv('FOODCOM_STAGING_DIR'):
        return Path(env_val)
    # 2. Auto-detect: app.py is at src/foodcom_pipeline/dashboard/app.py
    #    → project root is 3 levels up → staging/ lives there after a pipeline run
    project_root = Path(__file__).resolve().parents[3]
    candidate = project_root / 'staging'
    if candidate.exists():
        return candidate
    # 3. Docker-internal fallback
    return Path('/opt/airflow/staging')

STAGING_DIR = _resolve_staging_dir()
CLUSTER_PROFILE_PATH = STAGING_DIR / 'cluster_profiles.json'
CLUSTER_STAGING      = STAGING_DIR / 'user_clusters.parquet'
ELBOW_STATS_PATH     = STAGING_DIR / 'elbow_stats.json'

STALE_THRESHOLD_HOURS = 25  # pipeline runs hourly; warn if last run > 25 h ago

# Staging files in pipeline order (for the data-flow table)
PIPELINE_STAGES: list[dict] = [
    {'stage': 'Extract',   'task': 'extract_recipes',        'file': 'recipes_extracted.parquet',       'label': 'Raw recipes'},
    {'stage': 'Extract',   'task': 'extract_interactions',   'file': 'interactions_extracted.parquet',  'label': 'Raw interactions'},
    {'stage': 'Extract',   'task': 'extract_usda_nutrients', 'file': 'usda_nutrients.parquet',          'label': 'USDA ingredient rows'},
    {'stage': 'Clean',     'task': 'clean',                  'file': 'recipes_clean.parquet',           'label': 'Clean recipes'},
    {'stage': 'Clean',     'task': 'clean',                  'file': 'interactions_clean.parquet',      'label': 'Clean interactions'},
    {'stage': 'Sentiment', 'task': 'run_vader_sentiment',    'file': 'interactions_sentiment.parquet',  'label': 'Sentiment-scored interactions'},
    {'stage': 'Features',  'task': 'features',               'file': 'user_stats.parquet',              'label': 'Users with features'},
    {'stage': 'Features',  'task': 'features',               'file': 'ingredient_features.parquet',     'label': 'Ingredient features'},
    {'stage': 'Cluster',   'task': 'run_kmeans_clustering',  'file': 'user_clusters.parquet',           'label': 'Clustered users'},
]

# USDA nutrient value columns (None = not matched in USDA)
USDA_NUTRIENT_COLS = [
    'calories_per_100g',
    'protein_g_per_100g',
    'fat_g_per_100g',
    'saturated_fat_g_per_100g',
    'sugar_g_per_100g',
    'sodium_g_per_100g',
    'carbs_g_per_100g',
]

# Airflow metadata DB connection — same Postgres instance, read from env vars
_PG_USER = os.getenv('POSTGRES_USER', 'user')
_PG_PASS = os.getenv('POSTGRES_PASSWORD', 'password')
_PG_HOST = os.getenv('POSTGRES_HOST', 'localhost')
_PG_PORT = os.getenv('POSTGRES_PORT', '5432')
_PG_DB   = os.getenv('POSTGRES_DB', 'foodcom')
AIRFLOW_DB_DSN = f'postgresql://{_PG_USER}:{_PG_PASS}@{_PG_HOST}:{_PG_PORT}/{_PG_DB}'
DAG_ID = 'foodcom_batch_pipeline'

# Static CPG brand adjacency mapping (one entry per segment label)
CPG_ADJACENCY: dict[str, dict] = {
    'Indulgent Baker': {
        'top_categories': ['baking', 'dairy'],
        'brands': ['King Arthur Baking', 'Ghirardelli', 'Domino Sugar', "Land O'Lakes"],
        'cpm_range': '$4 – $8',
    },
    'International Explorer': {
        'top_categories': ['international', 'protein'],
        'brands': ['Kikkoman', 'Blue Dragon', 'Goya Foods', 'McCormick'],
        'cpm_range': '$5 – $9',
    },
    'Protein-Forward Cook': {
        'top_categories': ['protein', 'dairy'],
        'brands': ['Tyson Foods', 'Beyond Meat', 'Kirkland Signature', 'Applegate'],
        'cpm_range': '$6 – $10',
    },
    'Health-Conscious Cook': {
        'top_categories': ['vegetable', 'international'],
        'brands': ["Amy's Kitchen", 'Whole Foods 365', 'Green Giant', 'Earthbound Farm'],
        'cpm_range': '$7 – $12',
    },
    'General Cook': {
        'top_categories': ['dairy', 'vegetable'],
        'brands': ['Heinz', 'Kraft', "Campbell's", "Birds Eye"],
        'cpm_range': '$3 – $6',
    },
}

# Radar axes correspond to the 5 ingredient category features
RADAR_FEATURES = [
    'avg_rating_dairy',
    'avg_rating_protein',
    'avg_rating_vegetable',
    'avg_rating_baking',
    'avg_rating_international',
]
RADAR_LABELS = ['Dairy', 'Protein', 'Vegetable', 'Baking', 'International']

SEGMENT_COLORS = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_cluster_profiles() -> dict | None:
    if not CLUSTER_PROFILE_PATH.exists():
        return None
    with open(CLUSTER_PROFILE_PATH) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_user_clusters() -> pd.DataFrame | None:
    if not CLUSTER_STAGING.exists():
        return None
    return pd.read_parquet(CLUSTER_STAGING)


@st.cache_data(ttl=300)
def load_staging_stats() -> list[dict]:
    """
    For each pipeline stage entry, count rows in the corresponding parquet file.
    Returns a list of dicts with keys: stage, task, label, rows.
    """
    rows = []
    for entry in PIPELINE_STAGES:
        path = STAGING_DIR / entry['file']
        if path.exists():
            import pyarrow.parquet as pq
            count = pq.read_metadata(path).num_rows
        else:
            count = None
        rows.append({
            'Stage': entry['stage'],
            'Task': entry['task'],
            'Description': entry['label'],
            'Rows': count,
        })
    return rows


@st.cache_data(ttl=300)
def load_usda_coverage() -> dict | None:
    """
    Reads usda_nutrients.parquet and returns coverage stats:
      - total_ingredients: total rows
      - matched: rows with at least one non-null nutrient value
      - coverage_rate: matched / total
      - per_nutrient: {col: matched_count} for each nutrient column
    """
    path = STAGING_DIR / 'usda_nutrients.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    total = len(df)
    if total == 0:
        return {'total_ingredients': 0, 'matched': 0, 'coverage_rate': 0.0, 'per_nutrient': {}}

    available_cols = [c for c in USDA_NUTRIENT_COLS if c in df.columns]
    if not available_cols:
        return {'total_ingredients': total, 'matched': 0, 'coverage_rate': 0.0, 'per_nutrient': {}}

    matched = int(df[available_cols].notna().any(axis=1).sum())
    per_nutrient = {col: int(df[col].notna().sum()) for col in available_cols}
    return {
        'total_ingredients': total,
        'matched': matched,
        'coverage_rate': matched / total,
        'per_nutrient': per_nutrient,
    }


@st.cache_data(ttl=60)
def load_airflow_runtimes() -> pd.DataFrame | None:
    """
    Queries the Airflow metadata PostgreSQL for task durations in the most recent
    DAG run of foodcom_batch_pipeline.
    Returns a DataFrame with columns: task_id, duration_s, state, start_date.
    Returns None if the DB is unreachable.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(AIRFLOW_DB_DSN, connect_timeout=3)
        query = """
            SELECT task_id, duration, state, start_date
            FROM task_instance
            WHERE dag_id = %(dag_id)s
              AND run_id = (
                  SELECT run_id
                  FROM task_instance
                  WHERE dag_id = %(dag_id)s
                  ORDER BY start_date DESC
                  LIMIT 1
              )
            ORDER BY start_date;
        """
        df = pd.read_sql(query, conn, params={'dag_id': DAG_ID})
        conn.close()
        df = df.rename(columns={'duration': 'duration_s'})
        df['duration_s'] = pd.to_numeric(df['duration_s'], errors='coerce')
        return df
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_data_freshness() -> dict | None:
    """
    Returns the modification time of the most recently written staging file
    and how many hours ago that was. Used to detect stale pipeline runs.
    """
    from datetime import datetime
    mtimes = [
        (STAGING_DIR / entry['file']).stat().st_mtime
        for entry in PIPELINE_STAGES
        if (STAGING_DIR / entry['file']).exists()
    ]
    if not mtimes:
        return None
    last_run = datetime.fromtimestamp(max(mtimes))
    hours_ago = (datetime.now() - last_run).total_seconds() / 3600
    return {'last_run': last_run, 'hours_ago': hours_ago}


@st.cache_data(ttl=300)
def load_sentiment_coverage() -> dict | None:
    """
    Reads interactions_sentiment.parquet and returns the fraction of interactions
    that had scoreable review text (non-null sentiment_score).
    """
    path = STAGING_DIR / 'interactions_sentiment.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=['sentiment_score'])
    total = len(df)
    scored = int(df['sentiment_score'].notna().sum())
    return {
        'total': total,
        'scored': scored,
        'coverage': scored / total if total > 0 else 0.0,
    }


@st.cache_data(ttl=300)
def load_clustering_health() -> dict | None:
    """
    Reads elbow_stats.json written by cluster.py after each run.
    Returns k_range, silhouette scores, and chosen_k.
    """
    if not ELBOW_STATS_PATH.exists():
        return None
    with open(ELBOW_STATS_PATH) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_substitution_stats() -> dict | None:
    """
    Reads ingredient_features.parquet and returns substitution candidate counts.
    """
    path = STAGING_DIR / 'ingredient_features.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=['is_substitution_candidate'])
    total = len(df)
    candidates = int(df['is_substitution_candidate'].sum())
    return {
        'total_ingredients': total,
        'candidates': candidates,
        'rate': candidates / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Tab 3 helpers
# ---------------------------------------------------------------------------


def _radar_chart(profiles: dict, selected_labels: list[str]) -> go.Figure:
    fig = go.Figure()
    colors = SEGMENT_COLORS

    for i, (cid, profile) in enumerate(profiles.items()):
        label = profile['cluster_label']
        if label not in selected_labels:
            continue
        feature_stats = profile.get('feature_stats', {})
        values = [
            feature_stats.get(feat, {}).get('mean', 0.0) for feat in RADAR_FEATURES
        ]
        # Close the polygon
        values_closed = values + [values[0]]
        labels_closed = RADAR_LABELS + [RADAR_LABELS[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                fill='toself',
                name=label,
                line_color=colors[i % len(colors)],
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[3.5, 5.0])),
        showlegend=True,
        title='Taste Profile Radar — Avg Rating per Ingredient Category',
        height=480,
    )
    return fig


def _segment_profile_table(profiles: dict, selected_labels: list[str]) -> pd.DataFrame:
    rows = []
    for profile in profiles.values():
        if profile['cluster_label'] not in selected_labels:
            continue
        rows.append(
            {
                'Segment': profile['cluster_label'],
                'Users': f"{profile['n_users']:,}",
                '% of Total': f"{profile['pct_users']:.1f}%",
                'Peak Month': profile.get('peak_month') or '—',
                'Top Ingredients': ', '.join(profile.get('top_ingredients') or []) or '—',
                'Avg ! per Review': (
                    f"{profile['avg_exclamation_count']:.2f}"
                    if profile.get('avg_exclamation_count') is not None
                    else '—'
                ),
                'Sub. Exposure': (
                    f"{profile['substitute_exposure_rate']:.1%}"
                    if profile.get('substitute_exposure_rate') is not None
                    else '—'
                ),
            }
        )
    return pd.DataFrame(rows)


def _brand_adjacency_table(selected_labels: list[str]) -> pd.DataFrame:
    rows = []
    for label, data in CPG_ADJACENCY.items():
        if label not in selected_labels:
            continue
        rows.append(
            {
                'Segment': label,
                'Key Categories': ', '.join(data['top_categories']),
                'Recommended CPG Brands': ', '.join(data['brands']),
                'Est. CPM Range': data['cpm_range'],
            }
        )
    return pd.DataFrame(rows)


def _generate_pdf(profiles: dict, selected_labels: list[str]) -> bytes:
    """Generate a simple PDF report using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        return b''

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Food.com — Audience & Market Intelligence', ln=True, align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, 'CPG Advertising Segment Report', ln=True, align='C')
    pdf.ln(6)

    for profile in profiles.values():
        label = profile['cluster_label']
        if label not in selected_labels:
            continue
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, f"{label}  ({profile['n_users']:,} users, {profile['pct_users']:.1f}%)", ln=True)
        pdf.set_font('Helvetica', '', 10)

        top_ing = ', '.join(profile.get('top_ingredients') or []) or '—'
        pdf.cell(0, 6, f"  Top ingredients : {top_ing}", ln=True)
        pdf.cell(0, 6, f"  Peak month      : {profile.get('peak_month') or '—'}", ln=True)

        sub_exp = profile.get('substitute_exposure_rate')
        pdf.cell(
            0, 6,
            f"  Sub. exposure   : {sub_exp:.1%}" if sub_exp is not None else '  Sub. exposure   : —',
            ln=True,
        )

        adj = CPG_ADJACENCY.get(label, {})
        if adj:
            pdf.cell(0, 6, f"  CPG brands      : {', '.join(adj.get('brands', []))}", ln=True)
            pdf.cell(0, 6, f"  Est. CPM        : {adj.get('cpm_range', '—')}", ln=True)

        pdf.ln(4)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Tab implementations
# ---------------------------------------------------------------------------


def tab_overview() -> None:
    st.header('Pipeline Overview')
    st.info('TODO: Add pipeline run summary, last DAG execution time, row counts per stage.')


def tab_recipe_analytics() -> None:
    st.header('Recipe Analytics')
    st.info('TODO: Add top-rated recipes, sentiment rating distribution, cook-time analysis.')


def tab_audience_market() -> None:
    st.header('Audience & Market Intelligence')

    profiles_data = load_cluster_profiles()

    if profiles_data is None:
        st.warning(
            f'No cluster profiles found at `{CLUSTER_PROFILE_PATH}`. '
            'Run the batch pipeline first to generate segment data.'
        )
        return

    profiles: dict = profiles_data.get('clusters', {})
    all_labels = [p['cluster_label'] for p in profiles.values()]

    # --- Segment filter ---
    st.subheader('Segment Filter')
    selected_labels = st.multiselect(
        'Select segments to display',
        options=all_labels,
        default=all_labels,
    )

    if not selected_labels:
        st.warning('Select at least one segment.')
        return

    # --- KPI summary row ---
    total_users = profiles_data.get('total_users', 0)
    n_clusters = profiles_data.get('n_clusters', len(profiles))
    col1, col2 = st.columns(2)
    col1.metric('Total Segmented Users', f'{total_users:,}')
    col2.metric('Number of Segments', n_clusters)

    st.divider()

    # --- Radar chart ---
    st.subheader('Taste Profile Radar')
    fig = _radar_chart(profiles, selected_labels)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Segment profile table ---
    st.subheader('Segment Profiles')
    profile_df = _segment_profile_table(profiles, selected_labels)
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    st.divider()

    # --- Brand adjacency table ---
    st.subheader('CPG Brand Adjacency')
    st.caption(
        'Static mapping of audience segments to recommended CPG brand partners '
        'and estimated CPM ranges for programmatic ad buys.'
    )
    brand_df = _brand_adjacency_table(selected_labels)
    st.dataframe(brand_df, use_container_width=True, hide_index=True)

    st.divider()

    # --- PDF export ---
    st.subheader('Export')
    pdf_bytes = _generate_pdf(profiles, selected_labels)
    if pdf_bytes:
        st.download_button(
            label='Download PDF Report',
            data=pdf_bytes,
            file_name='foodcom_audience_report.pdf',
            mime='application/pdf',
        )
    else:
        st.info(
            'PDF export requires `fpdf2`. '
            'Install it with `uv add fpdf2` and restart the dashboard.'
        )


def tab_pipeline_status() -> None:
    st.header('Pipeline Status')

    staging_stats  = load_staging_stats()
    usda           = load_usda_coverage()
    runtimes       = load_airflow_runtimes()
    freshness      = load_data_freshness()
    sentiment_cov  = load_sentiment_coverage()
    clustering     = load_clustering_health()
    sub_stats      = load_substitution_stats()

    # ── Data freshness banner ─────────────────────────────────────────────────
    if freshness is None:
        st.warning('No staging files found. Run the batch pipeline first.')
    else:
        hours = freshness['hours_ago']
        last_run_str = freshness['last_run'].strftime('%Y-%m-%d %H:%M:%S')
        if hours > STALE_THRESHOLD_HOURS:
            st.warning(
                f'Pipeline data is stale — last run was **{hours:.1f} hours ago** '
                f'({last_run_str}). Expected cadence: hourly.'
            )
        else:
            st.success(f'Pipeline up to date — last run {hours:.1f} h ago ({last_run_str}).')

    st.divider()

    # ── KPI row 1: volume and loss ────────────────────────────────────────────
    def _find_rows(label_prefix: str) -> int | None:
        for r in staging_stats:
            if r['Description'].startswith(label_prefix):
                return r['Rows']
        return None

    raw_recipes   = _find_rows('Raw recipes')
    clean_recipes = _find_rows('Clean recipes')
    raw_ints      = _find_rows('Raw interactions')
    clean_ints    = _find_rows('Clean interactions')

    recipe_loss = (
        (raw_recipes - clean_recipes) / raw_recipes
        if raw_recipes and clean_recipes and raw_recipes > 0
        else None
    )
    int_loss = (
        (raw_ints - clean_ints) / raw_ints
        if raw_ints and clean_ints and raw_ints > 0
        else None
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Clean Recipes',      f"{clean_recipes:,}"     if clean_recipes else '—')
    col2.metric('Clean Interactions', f"{clean_ints:,}"        if clean_ints    else '—')
    col3.metric('Recipe Loss',        f"{recipe_loss:.2%}"     if recipe_loss is not None else '—',
                delta=f"-{raw_recipes - clean_recipes:,} rows" if recipe_loss is not None else None,
                delta_color='inverse')
    col4.metric('Interaction Loss',   f"{int_loss:.2%}"        if int_loss is not None else '—',
                delta=f"-{raw_ints - clean_ints:,} rows"       if int_loss is not None else None,
                delta_color='inverse')

    # ── KPI row 2: quality signals ────────────────────────────────────────────
    col5, col6, col7, col8 = st.columns(4)
    col5.metric(
        'USDA Coverage',
        f"{usda['coverage_rate']:.1%}" if usda else '—',
        help='Fraction of canonical ingredients matched to at least one USDA nutrient value.',
    )
    col6.metric(
        'Sentiment Coverage',
        f"{sentiment_cov['coverage']:.1%}" if sentiment_cov else '—',
        help='Fraction of clean interactions that had scoreable review text.',
    )
    col7.metric(
        'Substitution Candidates',
        f"{sub_stats['candidates']:,}" if sub_stats else '—',
        delta=f"{sub_stats['rate']:.1%} of ingredients" if sub_stats else None,
        delta_color='off',
        help='Ingredients with avg_rating < 3.5 in ≥5 recipes — flagged for substitution.',
    )
    col8.metric(
        'Optimal k (clusters)',
        str(clustering['chosen_k']) if clustering else '—',
        help='K chosen by highest silhouette score on the most recent run.',
    )

    st.divider()

    # ── Data flow table ───────────────────────────────────────────────────────
    st.subheader('Row Counts by Stage')
    df_stats = pd.DataFrame(staging_stats)
    df_stats['Rows'] = df_stats['Rows'].apply(
        lambda x: f'{x:,}' if x is not None else '— (not yet produced)'
    )
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    st.divider()

    # ── Clustering health ─────────────────────────────────────────────────────
    st.subheader('Clustering Health')
    if clustering is None:
        st.info('`elbow_stats.json` not found — run the clustering step first.')
    else:
        chosen_k = clustering['chosen_k']
        k_range  = clustering['k_range']
        silhouettes = clustering['silhouettes']

        colors = [
            '#00CC96' if k == chosen_k else '#636EFA'
            for k in k_range
        ]
        fig_sil = go.Figure(
            go.Bar(
                x=[str(k) for k in k_range],
                y=silhouettes,
                marker_color=colors,
                text=[f'{s:.4f}' for s in silhouettes],
                textposition='outside',
                hovertemplate='k=%{x}: silhouette=%{y:.4f}<extra></extra>',
            )
        )
        fig_sil.update_layout(
            xaxis_title='k (number of clusters)',
            yaxis_title='Silhouette Score',
            title=f'Silhouette Scores — chosen k={chosen_k} (green)',
            height=300,
            margin=dict(t=40, b=40),
            yaxis=dict(range=[0, max(silhouettes) * 1.2]),
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.divider()

    # ── USDA coverage breakdown ───────────────────────────────────────────────
    st.subheader('USDA Nutrient Coverage')
    if usda is None:
        st.warning('`usda_nutrients.parquet` not found. Run the pipeline first.')
    else:
        st.caption(
            f"{usda['matched']:,} of {usda['total_ingredients']:,} canonical ingredients "
            f"matched at least one USDA nutrient value ({usda['coverage_rate']:.1%})."
        )
        if usda['per_nutrient']:
            nutrient_df = pd.DataFrame(
                [
                    {
                        'Nutrient': col.replace('_per_100g', '').replace('_', ' ').title(),
                        'Matched Ingredients': count,
                        'Coverage': f"{count / usda['total_ingredients']:.1%}",
                    }
                    for col, count in usda['per_nutrient'].items()
                ]
            )
            st.dataframe(nutrient_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Airflow task runtimes ─────────────────────────────────────────────────
    st.subheader('Airflow Task Runtimes (most recent run)')
    if runtimes is None:
        st.warning(
            'Could not connect to the Airflow metadata database. '
            f'Ensure Postgres is running and accessible at `{_PG_HOST}:{_PG_PORT}`. '
            'Set `POSTGRES_HOST`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` '
            'environment variables if needed.'
        )
    elif runtimes.empty:
        st.info(f'No task runs found for DAG `{DAG_ID}`.')
    else:
        state_colors = {
            'success': '#00CC96',
            'failed':  '#EF553B',
            'running': '#FFA15A',
            'skipped': '#AB63FA',
        }
        bar_colors = [
            state_colors.get(str(s), '#636EFA')
            for s in runtimes['state']
        ]

        fig = go.Figure(
            go.Bar(
                x=runtimes['duration_s'].fillna(0).round(1),
                y=runtimes['task_id'],
                orientation='h',
                marker_color=bar_colors,
                text=runtimes['duration_s'].apply(
                    lambda s: f'{s:.0f}s' if pd.notna(s) else '—'
                ),
                textposition='outside',
                hovertemplate='%{y}: %{x:.1f}s<extra></extra>',
            )
        )
        fig.update_layout(
            xaxis_title='Duration (seconds)',
            yaxis_title='',
            yaxis=dict(autorange='reversed'),
            height=max(300, 50 * len(runtimes)),
            margin=dict(l=10, r=60, t=20, b=40),
        )

        # Legend explanation
        legend_cols = st.columns(len(state_colors))
        for col, (state, color) in zip(legend_cols, state_colors.items()):
            col.markdown(
                f'<span style="color:{color}">■</span> {state}',
                unsafe_allow_html=True,
            )

        st.plotly_chart(fig, use_container_width=True)

        # Summary table below chart
        summary = runtimes[['task_id', 'state', 'duration_s', 'start_date']].copy()
        summary['duration_s'] = summary['duration_s'].apply(
            lambda s: f'{s:.1f}s' if pd.notna(s) else '—'
        )
        summary.columns = ['Task', 'State', 'Duration', 'Started']
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title='Food.com Analytics',
        page_icon='🍽️',
        layout='wide',
    )
    st.title('Food.com — Analytics Dashboard')

    tab1, tab2, tab3, tab4 = st.tabs(
        ['Overview', 'Recipe Analytics', 'Audience & Market Intelligence', 'Pipeline Status']
    )

    with tab1:
        tab_overview()
    with tab2:
        tab_recipe_analytics()
    with tab3:
        tab_audience_market()
    with tab4:
        tab_pipeline_status()


if __name__ == '__main__':
    main()
