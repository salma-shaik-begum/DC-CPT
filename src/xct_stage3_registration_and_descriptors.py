"""
XCT Stage 3: Threshold Sensitivity, Exact Spatial Registration,
Richer Pore Descriptors, Local Thermal Features, Multi-Target Re-Test
========================================================================
Explicit constraints respected:
  - No new classifiers added.
  - MIRI untouched.
  - The XCT segmentation threshold is NOT changed/cherry-picked -- this
    script SWEEPS multiple thresholds to test robustness (that's the
    point of Step 1), it does not select a new "final" threshold to
    improve results.

STEP 1: Threshold sensitivity -- same segmentation pipeline run at
        several threshold values on the SAME raw crop, to prove results
        aren't an artifact of one arbitrary cutoff.
STEP 2: Exact XCT<->TAM spatial registration -- instead of comparing
        porosity in a sub-mm specimen against TAM/SCR averaged over the
        ENTIRE build plate (the current, diluted approach), find the
        exact TAM/SCR pixel region matching the XCT specimen's X/Y
        footprint, using both datasets' shared build coordinate system.
STEP 3: Richer pore descriptors per layer -- count, P90 diameter, max
        diameter, total volume, volume fraction (not just one porosity
        number per layer).
STEP 4: Local thermal features -- tam_p90, scr_std, etc. computed ONLY
        within the matched spatial region, not the whole build plate.
STEP 5: Re-test -- correlate local thermal features against ALL pore
        descriptors (count, size, max diameter, volume), not just
        porosity alone.
"""

import os
import gc
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import spearmanr

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

BASE = '/content/drive/MyDrive/DC-CPT-Project'
DREAM3D_PATH = f"{BASE}/Data/Raw/XCT_reconstruction/2 - XRCT_stack_BuildCoordSystem.dream3d"
DATASET_PATH = 'DataContainers/ImageDataContainer/CellData/ImageData'
GEOMETRY_GROUP = 'DataContainers/ImageDataContainer/_SIMPL_GEOMETRY'
TAM_PATH = f"{BASE}/Data/Raw/AMB2022-01-718-AMMT-B8-StaringCamera_TAM.h5"
SCR_PATH = f"{BASE}/Data/Raw/AMB2022-01-718-AMMT-B8-StaringCamera_SCR.h5"
RESULTS_TABLES = f"{BASE}/results/tables"
RESULTS_FIGURES = f"{BASE}/results/figures"

LAYER_THICKNESS_UM = 40.0
PRIMARY_THRESHOLD = 29562  # the Otsu threshold already established -- used as
                            # THE analysis threshold throughout; other values
                            # in Step 1 are for sensitivity testing ONLY

# ============================================================================
# Load XCT geometry + a generously-padded raw crop (reused across all steps)
# ============================================================================
print("="*70)
print("Loading XCT geometry and raw intensity crop")
print("="*70)

with h5py.File(DREAM3D_PATH, 'r') as f:
    origin = f[f'{GEOMETRY_GROUP}/ORIGIN'][:]       # X, Y, Z in um
    spacing = f[f'{GEOMETRY_GROUP}/SPACING'][:]      # um/voxel
    dims = f[f'{GEOMETRY_GROUP}/DIMENSIONS'][:]      # voxel counts X, Y, Z
    ds = f[DATASET_PATH]

    # reuse the previously-established specimen Z-range and a generously
    # padded XY box (wider than before, so lower sensitivity thresholds
    # aren't accidentally clipped)
    z_start, z_end = 26, 862
    y_start, y_end = 66, 872    # padded wider than Stage 2's 96:842
    x_start, x_end = 44, 749    # padded wider than Stage 2's 74:719

    print(f"  Loading RAW intensity crop: Z[{z_start}:{z_end}], "
          f"Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
    raw_crop = ds[z_start:z_end, y_start:y_end, x_start:x_end, 0]
    print(f"  Crop shape: {raw_crop.shape}, {raw_crop.nbytes/1e9:.2f} GB")

voxel_volume_um3 = spacing[0] * spacing[1] * spacing[2]

# build-coordinate origin of THIS crop (needed for Step 2 registration)
crop_origin_x_um = origin[0] + x_start * spacing[0]
crop_origin_y_um = origin[1] + y_start * spacing[1]
crop_origin_z_um = origin[2] + z_start * spacing[2]
print(f"  Crop origin in build coordinates (X,Y,Z um): "
      f"({crop_origin_x_um:.1f}, {crop_origin_y_um:.1f}, {crop_origin_z_um:.1f})")

# ============================================================================
# STEP 1: Threshold sensitivity sweep -- proves results aren't threshold-cherry-picked
# ============================================================================
print(f"\n{'='*70}")
print("STEP 1: Threshold sensitivity sweep")
print(f"{'='*70}")

SENSITIVITY_THRESHOLDS = [25000, 27500, PRIMARY_THRESHOLD, 31500, 33765]

sensitivity_results = []
for thresh in SENSITIVITY_THRESHOLDS:
    is_metal = raw_crop >= thresh
    labeled, n_comp = ndimage.label(is_metal)
    if n_comp == 0:
        continue
    sizes = ndimage.sum(is_metal, labeled, range(1, n_comp + 1))
    largest = (labeled == (np.argmax(sizes) + 1))
    filled = ndimage.binary_fill_holes(largest)
    pores = filled & ~largest
    n_pore_voxels = pores.sum()
    specimen_voxels = filled.sum()
    poros = n_pore_voxels / specimen_voxels if specimen_voxels > 0 else np.nan

    labeled_pores, n_pores = ndimage.label(pores)
    tag = " <-- PRIMARY (Otsu)" if thresh == PRIMARY_THRESHOLD else ""
    print(f"  threshold={thresh:6d}: specimen_voxels={specimen_voxels:>12,d}, "
          f"porosity={poros:.4%}, n_pores={n_pores}{tag}")
    sensitivity_results.append(dict(threshold=thresh, specimen_voxels=int(specimen_voxels),
                                     porosity=poros, n_pores=n_pores))
    del labeled, largest, filled, pores, labeled_pores
    gc.collect()

sens_df = pd.DataFrame(sensitivity_results)
save_sens = f"{RESULTS_TABLES}/xct_threshold_sensitivity.csv"
sens_df.to_csv(save_sens, index=False)
print(f"\n  [SAVED] {save_sens}")

poros_range = sens_df['porosity'].max() - sens_df['porosity'].min()
poros_relative_spread = poros_range / sens_df['porosity'].mean() if sens_df['porosity'].mean() > 0 else np.nan
print(f"\n  Porosity range across all tested thresholds: "
      f"{sens_df['porosity'].min():.4%} to {sens_df['porosity'].max():.4%}")
print(f"  Relative spread: {poros_relative_spread:.1%} of mean")
if poros_relative_spread < 1.0:
    print("  Result is reasonably ROBUST to threshold choice within this range.")
else:
    print("  Result is SENSITIVE to threshold choice -- report this honestly,")
    print("  it means absolute porosity values should be treated as approximate.")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(sens_df['threshold'], sens_df['porosity'] * 100, 'o-', color='firebrick')
ax.axvline(PRIMARY_THRESHOLD, color='gray', linestyle='--', alpha=0.6, label='Primary (Otsu)')
ax.set_xlabel('Segmentation Threshold')
ax.set_ylabel('Porosity (%)')
ax.set_title('Threshold Sensitivity: Porosity vs. Segmentation Cutoff')
ax.legend()
save_fig1 = f"{RESULTS_FIGURES}/xct_threshold_sensitivity.png"
fig.savefig(save_fig1, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  [SAVED] {save_fig1}")

# ============================================================================
# Run the PRIMARY (Otsu) segmentation once, keep results for Steps 3-5
# ============================================================================
print(f"\n{'='*70}")
print(f"Running primary segmentation at threshold={PRIMARY_THRESHOLD} for Steps 3-5")
print(f"{'='*70}")

is_metal = raw_crop >= PRIMARY_THRESHOLD
del raw_crop
gc.collect()

labeled_metal, n_comp = ndimage.label(is_metal)
sizes = ndimage.sum(is_metal, labeled_metal, range(1, n_comp + 1))
largest_label = np.argmax(sizes) + 1
specimen_metal = (labeled_metal == largest_label)
del labeled_metal, is_metal
gc.collect()

specimen_filled = ndimage.binary_fill_holes(specimen_metal)
pores_3d = specimen_filled & ~specimen_metal
del specimen_metal
gc.collect()

labeled_pores, n_pores_total = ndimage.label(pores_3d)
print(f"  Total pores (primary threshold): {n_pores_total}")

# get each pore's centroid Z (in CROP-LOCAL voxel index) and volume
pore_ids = np.arange(1, n_pores_total + 1)
centroids = ndimage.center_of_mass(pores_3d, labeled_pores, pore_ids)
pore_voxel_counts = ndimage.sum(pores_3d, labeled_pores, pore_ids)
pore_volumes_um3 = pore_voxel_counts * voxel_volume_um3
pore_diameters_um = (6 * pore_volumes_um3 / np.pi) ** (1/3)
pore_z_local = np.array([c[0] for c in centroids])  # local Z index within crop

# convert to build_z_um then to layer number (floor-based, as established)
pore_build_z_um = crop_origin_z_um + pore_z_local * spacing[2]
pore_layer = np.floor(pore_build_z_um / LAYER_THICKNESS_UM).astype(int)

specimen_voxels_total = specimen_filled.sum()
print(f"  Specimen total voxels: {specimen_voxels_total:,}")

# per-layer specimen voxel counts (needed for volume fraction denominators)
z_local_range = np.arange(specimen_filled.shape[0])
build_z_per_local_z = crop_origin_z_um + z_local_range * spacing[2]
layer_per_local_z = np.floor(build_z_per_local_z / LAYER_THICKNESS_UM).astype(int)

per_layer_specimen_voxels = {}
for layer in np.unique(layer_per_local_z):
    mask = (layer_per_local_z == layer)
    per_layer_specimen_voxels[layer] = specimen_filled[mask].sum()

del specimen_filled, pores_3d, labeled_pores
gc.collect()

# ============================================================================
# STEP 3: Richer pore descriptors per layer
# ============================================================================
print(f"\n{'='*70}")
print("STEP 3: Richer per-layer pore descriptors")
print(f"{'='*70}")

pore_df = pd.DataFrame(dict(pore_id=pore_ids, layer=pore_layer,
                             volume_um3=pore_volumes_um3, diameter_um=pore_diameters_um))

descriptor_rows = []
for layer in sorted(pore_df['layer'].unique()):
    if layer < 0:
        continue
    layer_pores = pore_df[pore_df['layer'] == layer]
    spec_vox = per_layer_specimen_voxels.get(layer, 0)
    total_pore_vol = layer_pores['volume_um3'].sum()
    total_pore_vox = total_pore_vol / voxel_volume_um3
    volume_fraction = total_pore_vox / spec_vox if spec_vox > 0 else np.nan

    descriptor_rows.append(dict(
        layer=int(layer),
        pore_count=len(layer_pores),
        diameter_p90_um=layer_pores['diameter_um'].quantile(0.9) if len(layer_pores) > 0 else 0.0,
        max_diameter_um=layer_pores['diameter_um'].max() if len(layer_pores) > 0 else 0.0,
        total_pore_volume_um3=total_pore_vol,
        volume_fraction=volume_fraction,
        specimen_voxels=int(spec_vox),
    ))

descriptors_df = pd.DataFrame(descriptor_rows)
save_desc = f"{RESULTS_TABLES}/xct_pore_descriptors_per_layer.csv"
descriptors_df.to_csv(save_desc, index=False)
print(descriptors_df.to_string(index=False))
print(f"\n  [SAVED] {save_desc}")

# ============================================================================
# STEP 2 + 4: Exact spatial registration + LOCAL thermal features
# ============================================================================
print(f"\n{'='*70}")
print("STEP 2 & 4: Exact XCT<->TAM spatial registration + local thermal features")
print(f"{'='*70}")

with h5py.File(TAM_PATH, 'r') as f:
    print("  TAM.h5 top-level structure (verifying grid arrays before assuming names):")
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"    [DATASET] {name}  shape={obj.shape}")
    f.visititems(show)

    tam_ds = f['ThermalData']['TAM']
    grid_group = f.get('DataProcessing', None)

    # look for spatial grid arrays under common names -- print whatever is
    # actually found rather than assuming
    xgrid = ygrid = None
    for candidate_path in ['Xgrid_3D', 'DataProcessing/Xgrid_3D', 'Grids/Xgrid_3D']:
        if candidate_path in f:
            xgrid = f[candidate_path][:]
            print(f"  Found X grid at: {candidate_path}, shape={xgrid.shape}")
            break
    for candidate_path in ['Ygrid_3D', 'DataProcessing/Ygrid_3D', 'Grids/Ygrid_3D']:
        if candidate_path in f:
            ygrid = f[candidate_path][:]
            print(f"  Found Y grid at: {candidate_path}, shape={ygrid.shape}")
            break

    if xgrid is None or ygrid is None:
        print("\n  NOTE: explicit X/Y coordinate grid arrays not found under the")
        print("  expected names in this file. Falling back to build-attribute")
        print("  based pixel spacing (hatch_spacing-derived) to estimate the")
        print("  TAM/SCR pixel-to-build-coordinate mapping instead of assuming")
        print("  a grid array that may not exist in this specific file release.")

        build_key = [k for k in f.keys() if k.startswith('AMB2022')][0]
        attrs = f[build_key].attrs
        print(f"  Available build attributes: {list(attrs.keys())}")

with h5py.File(SCR_PATH, 'r') as f:
    scr_ds = f['ThermalData']['SCR']

# ----------------------------------------------------------------------
# Registration approach: TAM/SCR arrays are (Layer, X_pixels, Y_pixels)
# covering the full build plate. Without confirmed absolute X/Y grid
# coordinate arrays in this file, we register using the KNOWN specimen
# build coordinates from the CONTEXT_2767.pdf documentation (Leg 9
# location) combined with the XCT crop's own build-coordinate origin
# computed above -- both are on the same documented build coordinate
# system (verified in Stage 1/2 via the Y-origin match to the EBSD
# reconstruction range).
# ----------------------------------------------------------------------
print(f"\n  XCT specimen build-coordinate footprint (X,Y): "
      f"X=[{crop_origin_x_um:.1f}, {crop_origin_x_um + raw_crop.shape[2]*spacing[0] if False else 'see below'}]")

# NOTE: raw_crop was already deleted for memory -- recompute extent from
# stored shape info instead
crop_x_extent_um = (x_end - x_start) * spacing[0]
crop_y_extent_um = (y_end - y_start) * spacing[1]
print(f"  XCT footprint: X=[{crop_origin_x_um:.1f}, {crop_origin_x_um + crop_x_extent_um:.1f}] um, "
      f"Y=[{crop_origin_y_um:.1f}, {crop_origin_y_um + crop_y_extent_um:.1f}] um")

with h5py.File(TAM_PATH, 'r') as f:
    build_key = [k for k in f.keys() if k.startswith('AMB2022')][0]
    tam_shape = f['ThermalData']['TAM'].shape
    print(f"  TAM array shape (Layer, dim1, dim2): {tam_shape}")

print("\n  IMPORTANT LIMITATION: without a confirmed absolute X/Y coordinate")
print("  grid array in the TAM/SCR HDF5 files (checked above, not found under")
print("  standard names), exact pixel-level registration to the XCT footprint")
print("  cannot be completed with full certainty in this pass. Proceeding with")
print("  the best available registration: assuming TAM/SCR pixel grids are")
print("  uniformly distributed across the documented build plate extent, and")
print("  extracting the pixel sub-region proportionally corresponding to the")
print("  XCT specimen's build-coordinate footprint. This is an IMPROVEMENT over")
print("  whole-plate averaging but should be reported as approximate registration,")
print("  not exact, pending confirmation of TAM/SCR's true coordinate grid.")

print(f"\n{'='*70}")
print("STAGE 3 PARTIAL COMPLETE -- see limitation note above on Step 2.")
print("Proceeding to extract a LOCAL sub-region using best-available registration.")
print(f"{'='*70}")

# ============================================================================
# Fallback registration: proportional mapping using documented build plate
# extent (100mm x 100mm, per AMB2022-01 documentation). This is an
# APPROXIMATION pending confirmation of TAM/SCR's true coordinate grid --
# stated explicitly, not hidden.
# ============================================================================
ASSUMED_BUILD_PLATE_EXTENT_UM = 100000.0  # 100mm x 100mm, per AMB2022-01 docs

with h5py.File(TAM_PATH, 'r') as f_tam, h5py.File(SCR_PATH, 'r') as f_scr:
    tam_ds = f_tam['ThermalData']['TAM']
    scr_ds = f_scr['ThermalData']['SCR']
    n_layers_tam, dim1, dim2 = tam_ds.shape

    # proportional pixel bounds corresponding to the XCT footprint
    px1_start = int((crop_origin_x_um / ASSUMED_BUILD_PLATE_EXTENT_UM) * dim1)
    px1_end = int(((crop_origin_x_um + crop_x_extent_um) / ASSUMED_BUILD_PLATE_EXTENT_UM) * dim1)
    px2_start = int((crop_origin_y_um / ASSUMED_BUILD_PLATE_EXTENT_UM) * dim2)
    px2_end = int(((crop_origin_y_um + crop_y_extent_um) / ASSUMED_BUILD_PLATE_EXTENT_UM) * dim2)

    # guard against degenerate (too-small or out-of-range) pixel windows
    px1_start, px1_end = max(0, px1_start), min(dim1, max(px1_start + 1, px1_end))
    px2_start, px2_end = max(0, px2_start), min(dim2, max(px2_start + 1, px2_end))

    print(f"\n  Mapped XCT footprint to TAM/SCR pixel window: "
          f"dim1[{px1_start}:{px1_end}], dim2[{px2_start}:{px2_end}] "
          f"(full array is {dim1}x{dim2})")

    if (px1_end - px1_start) < 3 or (px2_end - px2_start) < 3:
        print("  WARNING: mapped pixel window is very small (<3 px per side) --")
        print("  proportional registration may be too coarse to trust. Padding")
        print("  window by 5px per side as a practical compromise.")
        px1_start, px1_end = max(0, px1_start - 5), min(dim1, px1_end + 5)
        px2_start, px2_end = max(0, px2_start - 5), min(dim2, px2_end + 5)

    local_feature_rows = []
    for layer in range(n_layers_tam):
        tam_layer = tam_ds[layer, px1_start:px1_end, px2_start:px2_end]
        scr_layer = scr_ds[layer, px1_start:px1_end, px2_start:px2_end]
        tam_valid = tam_layer[~np.isnan(tam_layer)]
        scr_valid = scr_layer[~np.isnan(scr_layer)]

        if tam_valid.size == 0:
            local_feature_rows.append(dict(layer=layer, local_tam_p90=np.nan,
                                             local_tam_mean=np.nan, local_scr_mean=np.nan,
                                             local_scr_std_spatial=np.nan))
            continue

        local_feature_rows.append(dict(
            layer=layer,
            local_tam_p90=float(np.percentile(tam_valid, 90)),
            local_tam_mean=float(tam_valid.mean()),
            local_scr_mean=float(scr_valid.mean()) if scr_valid.size else np.nan,
            local_scr_std_spatial=float(scr_valid.std()) if scr_valid.size else np.nan,
        ))

local_features_df = pd.DataFrame(local_feature_rows)
save_local = f"{RESULTS_TABLES}/xct_local_thermal_features.csv"
local_features_df.to_csv(save_local, index=False)
print(f"\n  [SAVED] {save_local}")
print(local_features_df.head(26).to_string(index=False))

# ============================================================================
# STEP 5: Re-test -- local thermal features vs ALL pore descriptors
# ============================================================================
print(f"\n{'='*70}")
print("STEP 5: Re-test -- local thermal features vs. all pore descriptors")
print(f"{'='*70}")

final_merged = descriptors_df.merge(local_features_df, on='layer', how='inner')
save_final = f"{RESULTS_TABLES}/xct_final_local_correlation_data.csv"
final_merged.to_csv(save_final, index=False)
print(f"Merged dataset: {len(final_merged)} layers\n")

pore_targets = ['pore_count', 'diameter_p90_um', 'max_diameter_um',
                 'total_pore_volume_um3', 'volume_fraction']
thermal_predictors = ['local_tam_p90', 'local_tam_mean', 'local_scr_mean', 'local_scr_std_spatial']

final_corr_rows = []
for target in pore_targets:
    for predictor in thermal_predictors:
        if final_merged[target].nunique() < 2 or final_merged[predictor].nunique() < 2:
            continue
        rho, p = spearmanr(final_merged[predictor], final_merged[target])
        print(f"  {predictor:22s} vs {target:22s}: rho={rho:+.3f} (p={p:.3f}), n={len(final_merged)}")
        final_corr_rows.append(dict(predictor=predictor, target=target,
                                     spearman_rho=rho, p_value=p, n=len(final_merged)))

save_final_corr = f"{RESULTS_TABLES}/xct_final_local_correlation_results.csv"
pd.DataFrame(final_corr_rows).to_csv(save_final_corr, index=False)
print(f"\n[SAVED] {save_final_corr}")

print(f"\n{'='*70}")
print("XCT STAGE 3 COMPLETE.")
print("Report Step 2's registration as APPROXIMATE (proportional, pending exact")
print("grid confirmation) -- do not overstate it as exact pixel-level registration.")
print(f"{'='*70}")
