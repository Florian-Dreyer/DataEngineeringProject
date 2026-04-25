"""Global CSS theme — inject_theme() writes enhanced palette styles for decision-focused dashboard."""

_CSS = """
<style>
/* ============================================
   BASE STYLES
   ============================================ */
.stApp { background-color: #f8fafc; }

/* ============================================
   COLOR SYSTEM
   ============================================ */
:root {
  /* Status Colors */
  --color-gap: #ef4444;           /* Red - gap / high priority */
  --color-weak: #f59e0b;          /* Amber - weak match / medium */
  --color-strong: #10b981;        /* Green - strong match / good */
  --color-neutral: #9ca3af;       /* Gray - low priority */
  
  /* Brand Colors */
  --color-primary: #10b981;       /* Emerald */
  --color-primary-dark: #059669;
  
  /* Backgrounds */
  --color-bg-card: #ffffff;
  --color-bg-muted: #f8fafc;
  
  /* Text */
  --color-text-primary: #0f172a;
  --color-text-secondary: #64748b;
  --color-text-muted: #94a3b8;
  
  /* Borders */
  --color-border: #e2e8f0;
  --color-border-light: #f1f5f9;
}

/* ============================================
   TYPOGRAPHY
   ============================================ */
h1, h2, h3 {
  color: var(--color-text-primary);
  font-weight: 600;
}

/* Page header */
.stHeader h1, .stHeader h2 {
  font-size: 28px;
  font-weight: 700;
  color: var(--color-text-primary);
}

/* Section headers */
h2, h3, .stSubheader {
  font-size: 20px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: 16px;
}

/* Card titles */
.card-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--color-text-primary);
}

/* Body text */
p, .stMarkdown {
  color: #334155;
  font-size: 14px;
}

/* Caption / muted */
.caption, .stCaption {
  color: var(--color-text-muted);
  font-size: 12px;
}

/* ============================================
   METRIC CARDS
   ============================================ */
[data-testid="stMetric"] {
  background: white;
  border-radius: 8px;
  border: 1px solid var(--color-border);
  border-top: 3px solid var(--color-primary);
  padding: 16px !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* ============================================
   CUSTOM BADGES
   ============================================ */
.badge {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.badge-gap {
  background-color: #fef2f2;
  color: var(--color-gap);
  border: 1px solid #fecaca;
}

.badge-weak {
  background-color: #fffbeb;
  color: var(--color-weak);
  border: 1px solid #fde68a;
}

.badge-strong {
  background-color: #ecfdf5;
  color: var(--color-strong);
  border: 1px solid #a7f3d0;
}

.badge-high {
  background-color: #fef2f2;
  color: var(--color-gap);
  border: 1px solid #fecaca;
}

.badge-medium {
  background-color: #fffbeb;
  color: var(--color-weak);
  border: 1px solid #fde68a;
}

.badge-low {
  background-color: #f3f4f6;
  color: var(--color-neutral);
  border: 1px solid #d1d5db;
}

/* ============================================
   INSIGHT CARDS (Hero Panel)
   ============================================ */
.insight-card {
  background: var(--color-bg-card);
  border-radius: 8px;
  border: 1px solid var(--color-border);
  padding: 16px;
  margin-bottom: 12px;
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.insight-card:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  transform: translateY(-1px);
}

.insight-card.gap {
  border-left: 4px solid var(--color-gap);
}

.insight-card.weak {
  border-left: 4px solid var(--color-weak);
}

.insight-card.strong {
  border-left: 4px solid var(--color-strong);
}

/* ============================================
   PROGRESS BARS (inline)
   ============================================ */
.progress-bar-container {
  display: flex;
  align-items: center;
  gap: 8px;
}

.progress-bar {
  height: 8px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
  flex: 1;
}

.progress-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s ease;
}

.progress-bar-fill.gap { background: var(--color-gap); }
.progress-bar-fill.weak { background: var(--color-weak); }
.progress-bar-fill.strong { background: var(--color-strong); }
.progress-bar-fill.demand { background: linear-gradient(90deg, #3b82f6, #1d4ed8); }

/* ============================================
   TOOLTIP STYLES
   ============================================ */
.tooltip-container {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.tooltip-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #e2e8f0;
  color: #64748b;
  font-size: 10px;
  font-weight: 700;
  cursor: help;
}

.tooltip-text {
  visibility: hidden;
  opacity: 0;
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: #1e293b;
  color: white;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 12px;
  white-space: nowrap;
  z-index: 1000;
  margin-bottom: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.tooltip-container:hover .tooltip-text {
  visibility: visible;
  opacity: 1;
}

/* ============================================
   EXPANDABLE DETAIL PANELS
   ============================================ */
.detail-panel {
  background: var(--color-bg-muted);
  border-radius: 8px;
  border: 1px solid var(--color-border);
  padding: 16px;
  margin-top: 8px;
}

.detail-section {
  margin-bottom: 16px;
}

.detail-section:last-child {
  margin-bottom: 0;
}

.detail-section-title {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--color-text-secondary);
  margin-bottom: 8px;
}

/* ============================================
   TAB BAR
   ============================================ */
.stTabs [data-baseweb="tab"] {
  background-color: white;
  border-radius: 6px 6px 0 0;
  border: 1px solid var(--color-border);
  border-bottom: none;
  padding: 8px 18px;
  font-weight: 500;
}
.stTabs [aria-selected="true"] {
  border-top: 2px solid var(--color-primary) !important;
  color: var(--color-primary) !important;
}

/* ============================================
   PRIMARY BUTTONS
   ============================================ */
.stButton > button[kind="primary"],
.stButton > button {
  background-color: var(--color-primary);
  color: white;
  border: none;
  border-radius: 6px;
  font-weight: 600;
}
.stButton > button:hover {
  background-color: var(--color-primary-dark);
  border: none;
}

/* ============================================
   DATA FRAMES
   ============================================ */
[data-testid="stDataFrame"] { 
  border-radius: 8px; 
  overflow: hidden; 
}

/* ============================================
   DIVIDERS
   ============================================ */
hr { border-color: var(--color-border); }

/* ============================================
   EMPTY STATE / ERROR STATE
   ============================================ */
.empty-state {
  background: var(--color-bg-card);
  border-radius: 8px;
  border: 1px dashed var(--color-border);
  padding: 32px;
  text-align: center;
}

.empty-state-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.empty-state-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: 8px;
}

.empty-state-description {
  color: var(--color-text-secondary);
  margin-bottom: 16px;
}

/* ============================================
   HEALTH CARD (Pipeline Status)
   ============================================ */
.health-card {
  background: var(--color-bg-card);
  border-radius: 8px;
  border: 1px solid var(--color-border);
  padding: 16px;
  text-align: center;
}

.health-card .metric-value {
  font-size: 24px;
  font-weight: 700;
  color: var(--color-text-primary);
}

.health-card .metric-label {
  font-size: 12px;
  color: var(--color-text-secondary);
  margin-top: 4px;
}

.health-card .metric-status {
  font-size: 11px;
  margin-top: 8px;
  padding: 4px 8px;
  border-radius: 4px;
  display: inline-block;
}

.health-card .metric-status.ok {
  background: #ecfdf5;
  color: var(--color-strong);
}

.health-card .metric-status.warning {
  background: #fffbeb;
  color: var(--color-weak);
}

.health-card .metric-status.critical {
  background: #fef2f2;
  color: var(--color-gap);
}

/* ============================================
   SPACING UTILITIES
   ============================================ */
.section-spacing {
  margin-bottom: 32px;
}

.card-gap {
  gap: 16px;
}

/* ============================================
   INFO BOXES
   ============================================ */
.stAlert {
  border-radius: 8px;
}
</style>
"""


def inject_theme() -> None:
    """Inject global Enhanced CSS into the Streamlit page."""
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)
