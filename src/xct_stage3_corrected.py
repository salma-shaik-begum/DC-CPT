"""
XCT Stage 3 CORRECTED: True exact registration using the actual
Xgrid_v/Ygrid_v coordinate arrays (found under Calibration/Registration/,
missed in the first pass), plus fixed NaN handling in correlations.

Reuses saved outputs from the previous run (pore descriptors) -- only
re-does Steps 2, 4, 5 with the corrected registration.
"""

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = '/content/drive/MyDrive/DC-CPT-Project'
TAM_PATH = f"{BASE}/Data/Raw/AMB2022-01-718-AMMT-B8-StaringCamera_TAM.h5"
SCR_PATH = f"{BASE}/Data/Raw/AMB2022-01-718-AMMT-B8-StaringCamera_SCR.h5"
RESULTS_TABLES = f"{BASE}/results/tables"

# from the previous run's printed output -- XCT specimen build-coordinate footprint
XCT_X_MIN, XCT_X_MAX = -162.6, 1249.2   # um
XCT_Y_MIN, XCT_Y_MAX = 27916.0, 29530.0  # um

descriptors_df = pd.read_csv(f"{RESULTS_TABLES}/xct_pore_descriptors_per_layer.csv")

# ============================================================================
# STEP 2 CORRECTED: use the ACTUAL grid coordinate vectors
# ============================================================================
print("="*70)
print("STEP 2 (corrected): exact registration using Xgrid_v / Ygrid_v")
print("="*70)

with h5py.File(TAM_PATH, 'r') as f:
    xgrid_v = f['Calibration/Registration/Xgrid_v'][:]
    ygrid_v = f['Calibration/Registration/Ygrid_v'][:]
    tam_ds = f['ThermalData/TAM']
    n_layers, dim1, dim2 = tam_ds.shape

print(f"  Xgrid_v: shape={xgrid_v.shape}, range=[{xgrid_v.min():.2f}, {xgrid_v.max():.2f}]")
print(f"  Ygrid_v: shape={ygrid_v.shape}, range=[{ygrid_v.min():.2f}, {ygrid_v.max():.2f}]")
print(f"  TAM shape (Layer, dim1, dim2): {tam_ds.shape}")
print(f"  Ygrid_v length ({len(ygrid_v)}) matches dim1 ({dim1}) -- dim1 indexes Y")
print(f"  Xgrid_v length ({len(xgrid_v)}) matches dim2 ({dim2}) -- dim2 indexes X")

# unit check: are these grids in mm or um? Compare range to expected ~100mm plate
grid_range_x = xgrid_v.max() - xgrid_v.min()
print(f"\n  Xgrid_v total range: {grid_range_x:.2f} (units unknown -- inferring from magnitude)")
if grid_range_x < 1000:
    print("  Range suggests MILLIMETERS (typical build plate ~100mm) -- converting to um")
    unit_scale = 1000.0
else:
    print("  Range suggests already in MICROMETERS")
    unit_scale = 1.0

xgrid_v_um = xgrid_v * unit_scale
ygrid_v_um = ygrid_v * unit_scale
print(f"  Xgrid_v converted range: [{xgrid_v_um.min():.1f}, {xgrid_v_um.max():.1f}] um")
print(f"  Ygrid_v converted range: [{ygrid_v_um.min():.1f}, {ygrid_v_um.max():.1f}] um")

# find EXACT pixel indices matching the XCT footprint, using real coordinates
dim2_matches = np.where((xgrid_v_um >= XCT_X_MIN) & (xgrid_v_um <= XCT_X_MAX))[0]
dim1_matches = np.where((ygrid_v_um >= XCT_Y_MIN) & (ygrid_v_um <= XCT_Y_MAX))[0]

print(f"\n  Exact matched pixels: dim1 (Y) -> {len(dim1_matches)} pixels, "
      f"dim2 (X) -> {len(dim2_matches)} pixels")

MIN_WINDOW = 8  # sensible minimum regardless of exact match count, to avoid
                 # degenerate 1-3 pixel windows that mostly hit NaN regions
if len(dim1_matches) < MIN_WINDOW:
    center = int(np.argmin(np.abs(ygrid_v_um - (XCT_Y_MIN + XCT_Y_MAX) / 2)))
    half = MIN_WINDOW // 2
    dim1_matches = np.arange(max(0, center - half), min(dim1, center + half))
    print(f"  Widened dim1 window to minimum size: {len(dim1_matches)} pixels around index {center}")
if len(dim2_matches) < MIN_WINDOW:
    center = int(np.argmin(np.abs(xgrid_v_um - (XCT_X_MIN + XCT_X_MAX) / 2)))
    half = MIN_WINDOW // 2
    dim2_matches = np.arange(max(0, center - half), min(dim2, center + half))
    print(f"  Widened dim2 window to minimum size: {len(dim2_matches)} pixels around index {center}")

d1_start, d1_end = dim1_matches.min(), dim1_matches.max() + 1
d2_start, d2_end = dim2_matches.min(), dim2_matches.max() + 1
print(f"  Final window: dim1[{d1_start}:{d1_end}], dim2[{d2_start}:{d2_end}]")

# ============================================================================
# STEP 4 CORRECTED: local thermal features from the EXACT window
# ============================================================================
print(f"\n{'='*70}")
print("STEP 4 (corrected): local thermal features")
print(f"{'='*70}")

with h5py.File(TAM_PATH, 'r') as f_tam, h5py.File(SCR_PATH, 'r') as f_scr:
    tam_ds = f_tam['ThermalData/TAM']
    scr_ds = f_scr['ThermalData/SCR']

    local_rows = []
    for layer in range(n_layers):
        tam_layer = tam_ds[layer, d1_start:d1_end, d2_start:d2_end]
        scr_layer = scr_ds[layer, d1_start:d1_end, d2_start:d2_end]
        tam_valid = tam_layer[~np.isnan(tam_layer)]
        scr_valid = scr_layer[~np.isnan(scr_layer)]

        local_rows.append(dict(
            layer=layer,
            local_tam_p90=float(np.percentile(tam_valid, 90)) if tam_valid.size else np.nan,
            local_tam_mean=float(tam_valid.mean()) if tam_valid.size else np.nan,
            local_scr_mean=float(scr_valid.mean()) if scr_valid.size else np.nan,
            local_scr_std_spatial=float(scr_valid.std()) if scr_valid.size else np.nan,
            n_valid_pixels=int(tam_valid.size),
        ))

local_features_df = pd.DataFrame(local_rows)
save_local = f"{RESULTS_TABLES}/xct_local_thermal_features_corrected.csv"
local_features_df.to_csv(save_local, index=False)
print(local_features_df.head(26).to_string(index=False))
print(f"\n[SAVED] {save_local}")

# ============================================================================
# STEP 5 CORRECTED: re-test with proper NaN handling
# ============================================================================
print(f"\n{'='*70}")
print("STEP 5 (corrected): re-test, with NaN rows properly dropped before correlation")
print(f"{'='*70}")

final_merged = descriptors_df.merge(local_features_df, on='layer', how='inner')
print(f"Merged: {len(final_merged)} layers before NaN filtering")

pore_targets = ['pore_count', 'diameter_p90_um', 'max_diameter_um',
                 'total_pore_volume_um3', 'volume_fraction']
thermal_predictors = ['local_tam_p90', 'local_tam_mean', 'local_scr_mean', 'local_scr_std_spatial']

final_corr_rows = []
for target in pore_targets:
    for predictor in thermal_predictors:
        pair = final_merged[[predictor, target]].dropna()
        if len(pair) < 4 or pair[predictor].nunique() < 2 or pair[target].nunique() < 2:
            print(f"  {predictor:22s} vs {target:22s}: insufficient valid data (n={len(pair)})")
            continue
        rho, p = spearmanr(pair[predictor], pair[target])
        print(f"  {predictor:22s} vs {target:22s}: rho={rho:+.3f} (p={p:.3f}), n={len(pair)}")
        final_corr_rows.append(dict(predictor=predictor, target=target,
                                     spearman_rho=rho, p_value=p, n=len(pair)))

save_final_corr = f"{RESULTS_TABLES}/xct_final_local_correlation_results_corrected.csv"
pd.DataFrame(final_corr_rows).to_csv(save_final_corr, index=False)
print(f"\n[SAVED] {save_final_corr}")

print(f"\n{'='*70}")
print("STAGE 3 CORRECTION COMPLETE.")
print(f"{'='*70}")
