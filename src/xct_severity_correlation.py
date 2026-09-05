"""
XCT-Severity Correlation -- the actual external validation result
========================================================================
Aggregates per-slice XCT porosity into per-layer porosity (matching
your 40um layer thickness), then merges against B8's actual TAM/SCR
severity predictions for the overlapping layer range (0-25).

This is the real answer to: does your thermal-severity model's
prediction actually correlate with independently measured, physical
porosity? Report the honest Spearman correlation, whatever it is.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

BASE = '/content/drive/MyDrive/DC-CPT-Project'
XCT_PATH = f"{BASE}/results/tables/xct_full_porosity_profile.csv"
SEVERITY_PATH = f"{BASE}/Data/processed/gate1_labeled_dataset.csv"
RESULTS_TABLES = f"{BASE}/results/tables"
RESULTS_FIGURES = f"{BASE}/results/figures"

LAYER_THICKNESS_UM = 40.0

# ============================================================================
# STEP 1: Aggregate per-slice XCT porosity into per-layer porosity
# ============================================================================
xct_df = pd.read_csv(XCT_PATH)
xct_df['layer'] = np.round(xct_df['approx_layer']).astype(int)

# only keep AM-deposited region (layer >= 0), and only layers with a
# real specimen present in this slice (avoid diluting with empty/background
# slices at the very start of the scan)
xct_am = xct_df[(xct_df['layer'] >= 0) & (xct_df['specimen_area'] > 0)]

xct_per_layer = xct_am.groupby('layer').agg(
    mean_porosity=('porosity_fraction', 'mean'),
    max_porosity=('porosity_fraction', 'max'),
    n_slices=('porosity_fraction', 'count'),
).reset_index()

print("="*70)
print("Per-layer XCT porosity (aggregated from slices)")
print("="*70)
print(xct_per_layer.to_string(index=False))

# ============================================================================
# STEP 2: Load B8's severity data for the same layer range
# ============================================================================
severity_df = pd.read_csv(SEVERITY_PATH)
b8_severity = severity_df[severity_df['build'] == 'B8'][
    ['layer', 'severity', 'risk_score', 'tam_p90', 'tam_mean', 'scr_std']
]

# ============================================================================
# STEP 3: Merge on layer number -- this is the actual validation dataset
# ============================================================================
merged = xct_per_layer.merge(b8_severity, on='layer', how='inner')
print(f"\n{'='*70}")
print(f"Merged dataset: {len(merged)} overlapping layers")
print(f"{'='*70}")
print(merged.to_string(index=False))

out_path = f"{RESULTS_TABLES}/xct_severity_merged.csv"
merged.to_csv(out_path, index=False)
print(f"\n[SAVED] {out_path}")

if len(merged) < 5:
    print("\nWARNING: fewer than 5 overlapping layers -- correlation results")
    print("below will be statistically very weak regardless of the numbers.")
    print("Report this as a limited case-study observation, not a robust")
    print("statistical correlation.")

# ============================================================================
# STEP 4: The actual validation correlations
# ============================================================================
print(f"\n{'='*70}")
print("CORRELATION: measured XCT porosity vs. predicted severity signals")
print(f"{'='*70}")

correlation_results = []
for col in ['severity', 'risk_score', 'tam_p90', 'tam_mean', 'scr_std']:
    if merged[col].nunique() < 2 or merged['mean_porosity'].nunique() < 2:
        print(f"  {col}: insufficient variance to compute correlation")
        continue
    rho, p_spearman = spearmanr(merged['mean_porosity'], merged[col])
    r, p_pearson = pearsonr(merged['mean_porosity'], merged[col])
    print(f"  {col:12s} vs mean_porosity: Spearman rho={rho:+.3f} (p={p_spearman:.3f}), "
          f"Pearson r={r:+.3f} (p={p_pearson:.3f})")
    correlation_results.append(dict(feature=col, spearman_rho=rho, spearman_p=p_spearman,
                                     pearson_r=r, pearson_p=p_pearson, n=len(merged)))

save_corr_path = f"{RESULTS_TABLES}/xct_severity_correlation_results.csv"
pd.DataFrame(correlation_results).to_csv(save_corr_path, index=False)
print(f"\n[SAVED] {save_corr_path}")

# ============================================================================
# STEP 5: Figure -- side by side comparison, layer by layer
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].bar(merged['layer'], merged['mean_porosity'] * 100, color='firebrick', alpha=0.7)
axes[0].set_ylabel('Measured XCT Porosity (%)')
axes[0].set_title('Layer-by-Layer: Measured Porosity vs. Predicted Severity Signal (B8)')

axes[1].bar(merged['layer'], merged['tam_p90'], color='steelblue', alpha=0.7)
axes[1].set_xlabel('Layer Number')
axes[1].set_ylabel('tam_p90 (predicted severity signal)')

plt.tight_layout()
fig_path = f"{RESULTS_FIGURES}/xct_severity_layer_comparison.png"
fig.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"[SAVED] {fig_path}")

print(f"\n{'='*70}")
print("XCT-SEVERITY CORRELATION COMPLETE.")
print("Report these correlations EXACTLY as computed -- do not round up")
print("borderline p-values, and explicitly state n (number of overlapping")
print("layers) alongside every correlation, since this is a small-sample")
print("case-study validation, not a large-scale statistical claim.")
print(f"{'='*70}")
