# Market Intelligence Dashboard Redesign Specification

**Date:** April 25, 2026  
**Goal:** Transform the Market Intelligence tab from a data dump into a decision-making tool

---

## 1. Visual Hierarchy

### 1.1 Hero Insight Panel (Top of Page)

**Location:** Top of Market Intelligence tab, before any other content

**Content:** Top 3-5 unmet recipe opportunities as cards

**Card Structure:**
```
┌─────────────────────────────────────────────────────────────┐
│ 🔴 GAP    "Birria tacos"                                    │
│                                                             │
│   Gap Score: ████████░░ 0.82                                │
│   Best Match: "easy indian tacos" (similarity: 0.75)        │
│                                                             │
│   📈 High demand on Google Trends — no strong Food.com match│
└─────────────────────────────────────────────────────────────┘
```

**Color Coding:**
- 🔴 Red border + badge = Gap (similarity < 0.60)
- 🟡 Yellow border + badge = Weak Match (similarity 0.60-0.79)
- 🟢 Green border + badge = Strong Match (similarity ≥ 0.80)

**Icons:**
- Gap: 🔴 (red circle)
- Weak Match: 🟡 (yellow circle)
- Strong Match: 🟢 (green circle)

---

## 2. Metric Tooltips

### 2.1 Gap Score
**Tooltip Text:**
> "Measures how much consumer demand exceeds available Food.com recipes. Higher = larger opportunity gap. Calculated as: (external demand score - Food.com coverage) normalized to 0-1."

### 2.2 Similarity Score
**Tooltip Text:**
> "Semantic similarity between search term and closest Food.com recipe (0 to 1). Higher = closer match. Uses embedding-based similarity with fallback to lexical matching."

### 2.3 Match Status
**Tooltip Text:**
> - **Strong match:** similarity ≥ 0.80 — Good Food.com coverage exists
> - **Weak match:** similarity 0.60–0.79 — Partial coverage, room for improvement
> - **Gap:** similarity < 0.60 — Demand exceeds supply, content opportunity

### 2.4 Source Score
**Tooltip Text:**
> "Represents demand signal from Google Trends or AI Mode. Higher = more searched or surfaced. Normalized 0-1 scale across all terms."

### 2.5 Opportunity Label
**Tooltip Text:**
> - **High:** Gap score ≥ 0.70 — Priority content opportunity
> - **Medium:** Gap score 0.40–0.69 — Monitor and consider
> - **Low:** Gap score < 0.40 — Lower priority

---

## 3. Table Redesign

### 3.1 Gap Analysis Decision Table

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| Term | Text | Raw search term (truncated to 30 chars) |
| Demand | Visual Bar | Horizontal bar showing source score |
| Best Match | Text | Food.com recipe name (truncated) |
| Similarity | Progress Bar | 0-1 with color gradient |
| Status | Badge | Gap/Weak/Strong with color |
| Insight | Text | One-line explanation (expandable) |

**Visual Encodings:**
- Demand bar: Blue gradient (#3b82f6 to #1d4ed8)
- Similarity bar: Red (#ef4444) → Yellow (#f59e0b) → Green (#10b981)
- Status badge: Colored pill (red/yellow/green)

**Interactivity:**
- Click row → Expand to show full explanation panel
- Hover on any cell → Show tooltip with metric definition
- High gap score rows → Subtle red left border

---

## 4. Explainability Layer

### 4.1 Expandable Row Panel

**Trigger:** Click on any row in the gap analysis table

**Expanded Content:**
```
┌─────────────────────────────────────────────────────────────┐
│ 📋 TERM DETAILS                                             │
├─────────────────────────────────────────────────────────────┤
│ Raw Term:        "birria tacos"                             │
│ Canonical Term:  "birria tacos"                             │
│ Tags:            [mexican] [tacos] [handheld] [protein]     │
│                                                             │
│ 🍳 FOOD.COM MATCH                                           │
│ Best Match:     "easy indian tacos"                         │
│ Similarity:     0.75 (weak match)                           │
│ Method:         embedding                                   │
│                                                             │
│ 💡 WHY THIS CLASSIFICATION                                  │
│ This term was classified as a gap because its similarity   │
│ to the closest Food.com recipe (0.75) is below the strong  │
│ match threshold (0.80). Tagged as mexican + tacos_handheld.│
│ High demand signal from Google Trends (score: 0.89).       │
└─────────────────────────────────────────────────────────────┘
```

**Styling:**
- Card with light background (#f8fafc)
- Distinct sections with subheadings
- Monospace for technical values

---

## 5. Chart Improvements

### 5.1 Gap Bar Chart

**Current Issues:**
- Unclear what values mean
- Lacks narrative

**Improvements:**
- **Title:** "Where Demand Exceeds Supply"
- **Subtitle:** "Higher bars = bigger opportunity gaps. Red = high priority, Orange = medium, Gray = low."
- **Annotations:** Top 3 bars annotated directly with gap score
- **Hover:** Show term, gap score, opportunity label, best match

**Example Annotation:**
```
Birria tacos
Gap: 0.82 | High Priority
```

### 5.2 Cluster Chart

**Renamed to:** "Demand Clusters (Grouped by Similar Dishes)"

**Tooltip:**
> "Clusters group similar search terms using semantic similarity. Each cluster represents a distinct cuisine or dish category with common demand patterns."

**On Hover Show:**
- Cluster label
- Number of terms in cluster
- Dominant tags (top 3)
- Average gap score
- Best Food.com coverage

---

## 6. Pipeline Health

### 6.1 Current Display
```
recipe_tags: 489,209 ✓ present
```

### 6.2 Enhanced Display
```
┌──────────────────────────────┐
│ 🏷️ Recipe Tags               │
│ 489,209 generated            │
│                              │
│ ✅ High coverage across      │
│   Food.com corpus            │
└──────────────────────────────┘
```

### 6.3 Interpretation Rules

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| recipe_tags | > 100,000 | 50,000-100,000 | < 50,000 |
| gap_count | > 0 | = 0 | N/A |
| weak_match_pct | < 30% | 30-50% | > 50% |
| strong_match_pct | > 50% | 30-50% | < 30% |

### 6.4 Warning States

**Warning Example:**
```
⚠️ Gap count is 0 — this may indicate an over-matching issue.
Review similarity thresholds.
```

**Critical Example:**
```
🔴 Weak match rate is 65% — threshold may be too aggressive.
Consider lowering strong match threshold from 0.80 to 0.75.
```

---

## 7. Empty/Error States

### 7.1 Current
```
⚠️ AI Mode data not yet available — run the pipeline first.
```

### 7.2 Improved
```
┌─────────────────────────────────────────────────────────────┐
│ 📊 AI-Powered Search Intelligence                           │
│                                                             │
│                    🤖 AI Mode Data                          │
│                                                             │
│   This section shows real-time consumer demand from        │
│   Google AI Mode — the AI answers users see before         │
│   any search results.                                       │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │           🚀 Run Pipeline to Populate               │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   Last updated: Never                                      │
└─────────────────────────────────────────────────────────────┘
```

**CTA Button:** "Run Pipeline" → Triggers Airflow DAG or shows instructions

---

## 8. Design System

### 8.1 Color Palette

| Purpose | Color | Hex |
|---------|-------|-----|
| Primary (Emerald) | Green | #10b981 |
| Gap / High Priority | Red | #ef4444 |
| Weak Match / Medium | Amber | #f59e0b |
| Strong Match / Low | Gray | #9ca3af |
| Strong Match Good | Green | #10b981 |
| Background | Slate 50 | #f8fafc |
| Card Background | White | #ffffff |
| Text Primary | Slate 900 | #0f172a |
| Text Secondary | Slate 500 | #64748b |
| Border | Slate 200 | #e2e8f0 |

### 8.2 Typography

| Element | Weight | Size | Color |
|---------|--------|------|-------|
| Page Header | Bold | 28px | #0f172a |
| Section Header | SemiBold | 20px | #0f172a |
| Card Title | SemiBold | 16px | #0f172a |
| Body Text | Regular | 14px | #334155 |
| Caption/Muted | Regular | 12px | #64748b |
| Metric Value | Bold | 24px | #0f172a |
| Badge Text | SemiBold | 12px | white |

### 8.3 Spacing

- Section padding: 24px
- Card padding: 16px
- Card gap: 16px
- Element gap: 8px
- Border radius: 8px (cards), 4px (badges)

---

## 9. Narrative Flow

### 9.1 Section Headers (Story-Driven)

1. **"What Consumers Are Searching For"**
   - AI Mode + Google Trends data
   - Shows external demand signals

2. **"Where Demand Exceeds Supply"**
   - Gap analysis chart
   - Hero insight cards

3. **"Top Opportunities to Create Content"**
   - Decision table with top gaps
   - Expandable explanations

4. **"How These Opportunities Were Identified"**
   - Cluster chart with methodology
   - Explanation of matching process

5. **"Pipeline Health"**
   - Data freshness indicators
   - Warning states

---

## 10. Implementation Notes

### 10.1 Component Hierarchy

```
tab_market_intelligence/
├── HeroInsightPanel (new)
│   └── InsightCard[] (3-5 cards)
├── DemandSignalSection
│   ├── AI Mode Table (enhanced)
│   └── Trends Table (enhanced)
├── GapAnalysisSection
│   ├── GapBarChart (enhanced)
│   ├── InsightCards (existing, improved)
│   └── GapDecisionTable (redesigned)
│       └── RowExpandPanel (new)
├── ClusterSection
│   ├── ClusterChart (enhanced)
│   └── ClusterTable (enhanced)
└── PipelineHealthSection
    └── HealthCard[] (enhanced)
```

### 10.2 Streamlit Components Needed

1. **st.help()** - For native tooltips (limited)
2. **st.expander()** - For expandable row details
3. **st.container()** - For custom card layouts
4. **Custom HTML/CSS** - For progress bars and badges in tables
5. **plotly annotations** - For chart improvements

### 10.3 Data Requirements

Ensure these columns exist in gap analysis:
- `raw_term`
- `canonical_term`
- `gap_score`
- `source_score`
- `match_status` (gap/weak_match/strong_match)
- `similarity` (or `best_foodcom_similarity`)
- `best_foodcom_recipe_name`
- `matching_method` (lexical/embedding)
- `tags` (list or string)
- `insight_summary`
- `opportunity_label` (High/Medium/Low)

---

## 11. Example Redesigned Row

### Input Data
```json
{
  "raw_term": "birria tacos",
  "canonical_term": "birria tacos",
  "gap_score": 0.82,
  "source_score": 0.89,
  "match_status": "gap",
  "similarity": 0.54,
  "best_foodcom_recipe_name": "easy indian tacos",
  "matching_method": "embedding",
  "tags": ["mexican", "tacos", "handheld", "protein"],
  "insight_summary": "High demand on Google Trends — no strong Food.com match"
}
```

### Visual Output
```
┌────────────────────────────────────────────────────────────────┐
│ 🔴 GAP  birria tacos                                           │
│                                                                    │
│  Demand: ████████████████████░░░░░ 0.89                         │
│  Best Match: easy indian tacos                                   │
│  Similarity: ████████████░░░░░░░░░░░ 0.54                       │
│                                                                    │
│  📈 High demand on Google Trends — no strong Food.com match     │
│                                                                    │
│  [▼ Show Details]                                                │
└────────────────────────────────────────────────────────────────┘
```

### Expanded Details
```
┌────────────────────────────────────────────────────────────────┐
│ 📋 TERM DETAILS                                                 │
├────────────────────────────────────────────────────────────────┤
│ Raw Term:        birria tacos                                   │
│ Canonical Term:  birria tacos                                   │
│ Tags:            [mexican] [tacos] [handheld] [protein]        │
│                                                                  │
│ 🍳 FOOD.COM MATCH                                               │
│ Best Match:      easy indian tacos                              │
│ Similarity:      0.54 (gap)                                     │
│ Method:          embedding                                      │
│                                                                  │
│ 💡 WHY THIS CLASSIFICATION                                      │
│ This term was classified as a gap because its similarity to    │
│ the closest Food.com recipe (0.54) is below the weak match     │
│ threshold (0.60). Tagged as mexican + tacos_handheld.          │
│ High demand signal from Google Trends (score: 0.89).           │
└────────────────────────────────────────────────────────────────┘
```

---

## 12. Success Metrics

After implementation, the dashboard should achieve:

1. **Time to Insight:** < 30 seconds to understand top opportunities
2. **Tooltip Coverage:** 100% of metrics have explanations
3. **Visual Scanning:** Key info scannable in < 5 seconds
4. **Decision Clarity:** User knows what action to take
5. **Error Recovery:** Empty states guide user to next steps