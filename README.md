# SERSA Product Demand Relationship Analysis

Satellite project of [SERSA Industrial Bajío — Vending Machine Sales Analysis](https://github.com/Cesar423-png/SERSA_Industrial_Bajio_Sales_Analysis).  
It applies time series correlation, detrending, lead-lag analysis, and network graph methods to 4 years of industrial vending machine transaction data (April 2022 – May 2026) across 92 unique SKUs and 3 automotive industry clients in the Bajío region of Mexico.

> **Central question:** Do products whose demand moves together reveal hidden relationships in industrial consumption patterns — and do any products anticipate the behavior of others?

---

## Methodology

The project is structured as a four-layer analytical pipeline, where each layer refines or extends the previous one.

### Layer 1 — Raw Contemporaneous Correlation
**Notebook:** `02_correlation_analysis.ipynb`

Monthly transaction counts per SKU were aggregated from 639,568 individual dispense events and pivoted into a 50-month × 86-SKU matrix (6 near-zero SKUs excluded). Pearson correlation was computed across all 3,655 unique pairs.

- **298 pairs** with r ≥ 0.75 identified.
- Several pairs reached r = 1.0, notably size variants of the same base product (e.g. `SF1340L8`, `SF1340M7`, `SF1340S6`, `SF1340XS5`). This is expected behavior: operators dispense all sizes together, making variants behave as a single demand unit. Documented as a finding, not treated as noise.
- Raw correlation at this stage is inflated by a shared business growth trend (294% revenue increase 2022–2025), making it impossible to distinguish genuine demand co-movement from coincidental parallel growth.

### Layer 2 — Detrended Growth-Rate Correlation
**Notebook:** `03_growth_correlation.ipynb`

Instead of asking whether two SKUs have similar volume levels, this layer asks: *when SKU A grows faster than usual in a given month, does SKU B also grow faster than usual?*

Month-over-month percentage change (`pct_change`) removes the common upward trend. Zero-to-nonzero transitions produce `inf` values, replaced with `NaN` before computing correlation; `pandas corr()` handles remaining NaN pairwise.

- **104 of 298 pairs survive detrending** — 65% of raw correlations were spurious trend artifacts.
- **7 negative pairs** emerge that were invisible in raw counts, suggesting potential substitution dynamics (e.g. `AN23-201T9 ↔ ME6632L`, r = −0.9999).
- The raw vs. growth-rate scatter plot shows the majority of pairs falling well below the diagonal, confirming the common trend was the dominant driver of raw correlations.

### Layer 3 — Lead-Lag Analysis
**Notebook:** `04_lead_lag_analysis.ipynb`

For each of the 104 genuine pairs, Pearson cross-correlation was computed at lags −6 to +6 months on growth rates. The dominant lag — the lag at which absolute cross-correlation is highest — was identified per pair.

- **96 of 104 pairs are purely contemporaneous** (dominant lag = 0): demand co-movement occurs within the same month, with no predictive structure.
- **8 pairs show lag ≠ 0**; 4 are analytically robust (r > 0.87 at the dominant lag):

| Relationship | Dominant lag | r at lag | Interpretation |
|---|---|---|---|
| `UVSVP401` → `ME6632XL` | −2 months | 0.934 | UVSVP401 leads ME6632XL by ~2 months |
| `PIP33-BLK125BPM` → `ME6632XL` | −2 months | 0.923 | PIP33-BLK125BPM leads ME6632XL by ~2 months |
| `UVS4040` → `PIP33-BLK125BPM` | +2 months | 0.881 | UVS4040 leads PIP33-BLK125BPM by ~2 months |
| `UVSVP401` → `UVS4040` | −2 months | 0.878 | UVSVP401 leads UVS4040 by ~2 months |

The remaining 4 lead-lag candidates have lower r values at their dominant lag or show irregular cross-correlation profiles inconsistent with a stable predictive signal.

### Layer 4 — Product Demand Network
**Notebook:** `05_product_network.ipynb`

A graph was constructed from the 104 genuine edges (growth-rate r ≥ 0.75), with SKUs as nodes and edge weights proportional to correlation strength. Lead-lag edges are visually distinguished from contemporaneous ones. Community detection used the greedy modularity algorithm.

- **63 SKUs** appear in the network (29 SKUs have no strong relationship with any other).
- **12 communities** detected, ranging from 2 to 13 SKUs.
- **9 connected components** — the network is not fully connected, meaning several demand communities operate independently.
- **Average clustering coefficient: 0.313** — moderate internal density within communities.
- Most connected SKUs (highest degree): `PIP33-BLK125BPM`, `ME6632L`, `CRST1130` (degree = 9 each).

---

## Key Methodological Decisions

**Why monthly aggregation?**  
Daily granularity is too noisy for correlation analysis. Monthly smooths operational variance while preserving seasonal and trend signals — standard practice for industrial demand analysis.

**Why `pct_change` for detrending?**  
The business grew 294% over the analysis period. Without detrending, nearly any two growing SKUs appear correlated. Growth rates isolate genuine co-movement from shared trend.

**Why not remove size-variant SKUs?**  
Size variants (SF1340 family, AN11840 family, SFPUG111 family) show r = 1.0 in both raw and growth-rate correlations. This is a real business phenomenon — not a data artifact — and removing them would suppress a meaningful finding.

**Why restrict lead-lag analysis to the 104 genuine pairs?**  
Running cross-correlation on all 3,655 pairs would produce many false positives at non-zero lags by chance. Restricting to pairs with confirmed contemporaneous co-movement gives a principled candidate set.

---

## Project Structure

```
SERSA_Product_Demand_Relationship_Analysis/
├── data/
│   └── processed/          ← master_sales.csv (from main project)
├── notebooks/
│   ├── 01_time_series_preparation.ipynb
│   ├── 02_correlation_analysis.ipynb
│   ├── 03_growth_correlation.ipynb
│   ├── 04_lead_lag_analysis.ipynb
│   └── 05_product_network.ipynb
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   └── ts_utils.py
├── README.md
└── requirements.txt
```

---

## Notebook Overview

| # | Notebook | Input | Output |
|---|----------|-------|--------|
| 01 | `01_time_series_preparation` | `master_sales.csv` | `monthly_sales_pivot.csv` — 50 months × 92 SKUs |
| 02 | `02_correlation_analysis` | `monthly_sales_pivot.csv` | Pearson correlation matrix, 298 raw pairs |
| 03 | `03_growth_correlation` | `02_pivot_filtered.csv` | Growth-rate correlation matrix, 104 genuine pairs |
| 04 | `04_lead_lag_analysis` | `03_top_positive_pairs_growth.csv` | Cross-correlations at lags −6 to +6, 8 lead-lag candidates |
| 05 | `05_product_network` | Correlation matrices + lag data | Network graph, 12 communities, edge and node exports |

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Cesar423-png/SERSA_Product_Demand_Relationship_Analysis.git
   cd SERSA_Product_Demand_Relationship_Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Copy `master_sales.csv`** from the main project into `data/processed/`.

4. **Run the notebooks in order** from within the `notebooks/` directory:
   ```
   01 → 02 → 03 → 04 → 05
   ```
   All paths are relative — no configuration needed as long as notebooks are opened from within `notebooks/`.

---

## Tech Stack

- **Python 3.14** — pandas, numpy, matplotlib, seaborn, networkx
- **Jupyter Notebook**
- **Power BI** — dashboard in progress

---

## Related Project

This is a satellite project. The main project — [SERSA Industrial Bajío — Vending Machine Sales Analysis](https://github.com/Cesar423-png/SERSA_Industrial_Bajio_Sales_Analysis) — covers business KPIs, revenue trends, profitability analysis, and Pareto concentration across the same dataset.

---

## Author

Cesar Enrique Banda Martinez — [LinkedIn](https://www.linkedin.com/in/c%C3%A9sar-banda-79b6b3262/?locale=en_US) · [GitHub](https://github.com/Cesar423-png)
