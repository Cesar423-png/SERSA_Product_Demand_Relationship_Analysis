"""
prepare_powerbi_data.py
=======================
Generates clean CSV files optimized for Power BI import.

Run this script from the project root:
    python src/prepare_powerbi_data.py

Outputs are saved to outputs/tables/powerbi/
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = os.path.join(os.path.dirname(__file__), "..")
TABLES_DIR    = os.path.join(ROOT, "outputs", "tables")
POWERBI_DIR   = os.path.join(TABLES_DIR, "powerbi")
os.makedirs(POWERBI_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Network edges  (Page 1 — Product Network)
# ---------------------------------------------------------------------------
edges = pd.read_csv(os.path.join(TABLES_DIR, "05_network_edges.csv"), decimal=",")

# Add readable lag label
def lag_label(lag):
    if lag == 0:
        return "Contemporaneous"
    elif lag > 0:
        return f"SKU_B leads SKU_A by {lag}m"
    else:
        return f"SKU_A leads SKU_B by {abs(lag)}m"

edges["lag_label"]        = edges["dominant_lag"].apply(lag_label)
edges["community_A_label"] = "Community " + (edges["community_A"] + 1).astype(str)
edges["community_B_label"] = "Community " + (edges["community_B"] + 1).astype(str)
edges["r_pct"]            = (edges["Pearson_r"] * 100).round(1)

edges.to_csv(os.path.join(POWERBI_DIR, "pb_network_edges.csv"), index=False)
print(f"pb_network_edges.csv       -> {len(edges)} rows")

# ---------------------------------------------------------------------------
# 2. Network nodes  (Page 1 — Product Network)
# ---------------------------------------------------------------------------
nodes = pd.read_csv(os.path.join(TABLES_DIR, "05_network_nodes.csv"), decimal=",")

nodes["community_label"] = "Community " + (nodes["community"] + 1).astype(str)

nodes.to_csv(os.path.join(POWERBI_DIR, "pb_network_nodes.csv"), index=False)
print(f"pb_network_nodes.csv       -> {len(nodes)} rows")

# ---------------------------------------------------------------------------
# 3. Raw vs Growth-Rate scatter data  (Page 2)
# ---------------------------------------------------------------------------
corr_raw = pd.read_csv(
    os.path.join(TABLES_DIR, "02_correlation_matrix.csv"),
    index_col=0, decimal=","
)
corr_growth = pd.read_csv(
    os.path.join(TABLES_DIR, "03_correlation_matrix_growth.csv"),
    index_col=0, decimal=","
)

# Align SKUs
shared_skus = corr_raw.index.intersection(corr_growth.index)
corr_raw    = corr_raw.loc[shared_skus, shared_skus]
corr_growth = corr_growth.loc[shared_skus, shared_skus]

# Extract upper triangle pairs
mask = np.triu(np.ones(corr_raw.shape, dtype=bool), k=1)

raw_vals    = corr_raw.values[mask]
growth_vals = corr_growth.values[mask]

skus = corr_raw.index.tolist()
pairs_a, pairs_b = [], []
for i in range(len(skus)):
    for j in range(i + 1, len(skus)):
        pairs_a.append(skus[i])
        pairs_b.append(skus[j])

scatter_df = pd.DataFrame({
    "SKU_A"         : pairs_a,
    "SKU_B"         : pairs_b,
    "raw_r"         : raw_vals.round(4),
    "growth_r"      : growth_vals.round(4),
    "pair"          : [f"{a} ↔ {b}" for a, b in zip(pairs_a, pairs_b)],
})

# Drop rows where growth_r is NaN
scatter_df = scatter_df.dropna(subset=["growth_r"])

# Quadrant classification
def quadrant(raw, growth, thr=0.75):
    if raw >= thr and growth >= thr:
        return "Genuine (strong in both)"
    elif raw >= thr and growth < thr:
        return "Spurious (trend only)"
    elif raw < thr and growth >= thr:
        return "Hidden (only in growth)"
    else:
        return "Weak"

scatter_df["quadrant"] = scatter_df.apply(
    lambda r: quadrant(r["raw_r"], r["growth_r"]), axis=1
)

scatter_df.to_csv(os.path.join(POWERBI_DIR, "pb_raw_vs_growth_scatter.csv"), index=False)
print(f"pb_raw_vs_growth_scatter.csv -> {len(scatter_df)} rows")

# ---------------------------------------------------------------------------
# 4. Cross-correlation by lag  (Page 3 — Lead-Lag)
# ---------------------------------------------------------------------------
cc_all = pd.read_csv(os.path.join(TABLES_DIR, "04_crosscorr_all_pairs.csv"), decimal=",")
cc_all["pair"] = cc_all["SKU_A"] + " ↔ " + cc_all["SKU_B"]
cc_all["Pearson_r"] = cc_all["Pearson_r"].round(4)

cc_all.to_csv(os.path.join(POWERBI_DIR, "pb_crosscorr_by_lag.csv"), index=False)
print(f"pb_crosscorr_by_lag.csv    -> {len(cc_all)} rows")

# ---------------------------------------------------------------------------
# 5. Lead-lag candidates summary  (Page 3 — Lead-Lag)
# ---------------------------------------------------------------------------
candidates = pd.read_csv(os.path.join(TABLES_DIR, "04_lead_lag_candidates.csv"), decimal=",")

def lead_direction(row):
    if row["dominant_lag"] < 0:
        return f"{row['SKU_A']} leads {row['SKU_B']} by {abs(row['dominant_lag'])}m"
    else:
        return f"{row['SKU_B']} leads {row['SKU_A']} by {row['dominant_lag']}m"

candidates["direction"]   = candidates.apply(lead_direction, axis=1)
candidates["pair"]        = candidates["SKU_A"] + " ↔ " + candidates["SKU_B"]
candidates["max_abs_r"]   = candidates["max_abs_r"].round(4)

candidates.to_csv(os.path.join(POWERBI_DIR, "pb_lead_lag_candidates.csv"), index=False)
print(f"pb_lead_lag_candidates.csv -> {len(candidates)} rows")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print()
print(f"All files saved to: {os.path.normpath(POWERBI_DIR)}")