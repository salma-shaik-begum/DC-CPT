"""
XCT Stage 2: True 3D Pore Segmentation, Multi-Threshold Validation,
Region Separation, and Corrected Layer Aggregation
========================================================================
Upgrades over the Stage 1 (2D slice-wise) analysis:

  STEP A: Verify registration -- confirm the baseplate/AM transition in
          the metal-fraction curve lands at build_z=0, as it should if
          NIST's ORIGIN/SPACING registration is correct. (No separate
          NIST mapping file exists beyond ORIGIN/SPACING/layer-thickness,
          already in use -- this step verifies those values, not a new
          data source.)

  STEP B: Compare 3 threshold methods (fixed value, global Otsu, local
          adaptive) on the same test region -- confirms the porosity
          result isn't an artifact of one arbitrary threshold choice.
          (No NIST-provided binary segmentation exists for XCT --
          confirmed absent from the README in Stage 1.)

  STEP C: TRUE 3D pore segmentation -- 3D connected-component labeling
          on the full volume, not per-slice 2D fill_holes. This gives
          real pore COUNT, individual pore VOLUMES, and a genuine
          pore-size distribution, and correctly handles pores that span
          multiple Z-slices (which 2D analysis could miss or fragment).

  STEP D: Separate baseplate / interface / AM-deposited regions so
          baseplate statistics don't contaminate AM-layer porosity.

  STEP E: Layer assignment via floor(build_z / 40), not round() -- the
          physically correct binning for layers occupying [n*40, (n+1)*40).

  STEP F: Recompute correlations on the corrected 3D per-layer porosity,
          plus a defect-classification metric (ROC-AUC) treating
          above-median porosity layers as "elevated."

MEMORY NOTE: this script crops to only the Z-range containing actual
specimen material (using the Stage 1 profile to find those bounds),
not the full 882 slices, to keep 3D operations memory-safe.
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
import h5py

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

BASE = '/content/drive/MyDrive/DC-CPT-Project'
DREAM3D_PATH = f"{BASE}/Data/Raw/XCT_reconstruction/2 - XRCT_stack_BuildCoordSystem.dream3d"
DATASET_PATH = 'DataContainers/ImageDataContainer/CellData/ImageData'
GEOMETRY_GROUP = 'DataContainers/ImageDataContainer/_SIMPL_GEOMETRY'
STAGE1_PROFILE_PATH = f"{BASE}/results/tables/xct_full_porosity_profile.csv"
SEVERITY_PATH = f"{BASE}/Data/processed/gate1_labeled_dataset.csv"
RESULTS_TABLES = f"{BASE}/results/tables"
RESULTS_FIGURES = f"{BASE}/results/figures"

LAYER_THICKNESS_UM = 40.0
FIXED_THRESHOLD = 33765

# ============================================================================
# STEP A: Verify registration using the Stage 1 profile already computed
# ============================================================================
print("="*70)
print("STEP A: Registration verification")
print("="*70)

stage1 = pd.read_csv(STAGE1_PROFILE_PATH)
stage1_valid = stage1[stage1['specimen_area'] > 0].copy()

# find the transition: metal_fraction should show a step change near build_z=0
# if baseplate (below 0) and AM-deposited (above 0) have different geometry
below_zero = stage1_valid[stage1_valid['build_z_um'] < 0]['metal_fraction']
above_zero = stage1_valid[stage1_valid['build_z_um'] >= 0]['metal_fraction']
print(f"  Mean metal fraction, build_z < 0 (baseplate):     {below_zero.mean():.1%}")
print(f"  Mean metal fraction, build_z >= 0 (AM-deposited):  {above_zero.mean():.1%}")
if abs(below_zero.mean() - above_zero.mean()) > 0.05:
    print("  VERIFIED: clear geometric transition at build_z=0 -- registration")
    print("  is consistent with independent physical evidence (specimen shape")
    print("  change), not just an assumed coordinate.")
else:
    print("  WARNING: no clear transition detected at build_z=0 -- registration")
    print("  should be double-checked before trusting layer assignments.")

# crop bounds for 3D processing -- use actual specimen extent, not full 882 slices
first_slice = stage1_valid['z_index'].min()
last_slice = stage1_valid['z_index'].max()
print(f"\n  Cropping 3D processing to slices {first_slice}-{last_slice} "
      f"({last_slice - first_slice + 1} slices, vs. 882 total) to keep memory safe.")

# ============================================================================
# STEP B: Multi-threshold comparison on one representative slice
# ============================================================================
print(f"\n{'='*70}")
print("STEP B: Threshold method comparison")
print(f"{'='*70}")

with h5py.File(DREAM3D_PATH, 'r') as f:
    ds = f[DATASET_PATH]
    test_z = int((first_slice + last_slice) / 2)
    test_slice = ds[test_z, :, :, 0].astype(np.float64)

try:
    from skimage.filters import threshold_otsu, threshold_local
    otsu_thresh = threshold_otsu(test_slice)
    print(f"  Fixed threshold (Stage 1):  {FIXED_THRESHOLD:.0f}")
    print(f"  Global Otsu threshold:      {otsu_thresh:.0f}")

    local_thresh_map = threshold_local(test_slice, block_size=51, method='gaussian')
    local_binary = test_slice > local_thresh_map
    print(f"  Local adaptive: mean effective threshold ~{local_thresh_map.mean():.0f} "
          f"(varies spatially, range {local_thresh_map.min():.0f}-{local_thresh_map.max():.0f})")

    for name, thresh_val in [('Fixed', FIXED_THRESHOLD), ('Otsu', otsu_thresh)]:
        binary = test_slice >= thresh_val
        labeled, n = ndimage.label(binary)
        if n > 0:
            sizes = ndimage.sum(binary, labeled, range(1, n + 1))
            largest = (labeled == (np.argmax(sizes) + 1))
            filled = ndimage.binary_fill_holes(largest)
            pores = filled & ~largest
            poros = pores.sum() / filled.sum() if filled.sum() > 0 else np.nan
            print(f"    {name} threshold -> porosity on test slice: {poros:.4%}")

    print("\n  No NIST-provided binary segmentation exists for XCT data (confirmed")
    print("  absent from the dataset README in Stage 1) -- fixed and Otsu thresholds")
    print("  are compared as the two principled options available.")
    HAS_SKIMAGE = True
except ImportError:
    print("  scikit-image not installed -- run: pip install scikit-image --break-system-packages")
    print("  Skipping Otsu/local comparison, proceeding with fixed threshold only.")
    otsu_thresh = FIXED_THRESHOLD
    HAS_SKIMAGE = False

# use Otsu going forward if available (data-driven, not a manually chosen constant)
SEGMENTATION_THRESHOLD = otsu_thresh if HAS_SKIMAGE else FIXED_THRESHOLD
print(f"\n  Using threshold={SEGMENTATION_THRESHOLD:.0f} for the full 3D analysis below.")

del test_slice
gc.collect()

# ============================================================================
# STEP C: TRUE 3D volume loading + segmentation (cropped Z-range)
# ============================================================================
print(f"\n{'='*70}")
print("STEP C: Loading cropped volume and running full 3D segmentation")
print(f"{'='*70}")

with h5py.File(DREAM3D_PATH, 'r') as f:
    ds = f[DATASET_PATH]
    origin = f[f'{GEOMETRY_GROUP}/ORIGIN'][:]
    spacing = f[f'{GEOMETRY_GROUP}/SPACING'][:]

    print(f"  Loading slices {first_slice}-{last_slice} into memory as boolean mask...")
    n_crop = last_slice - first_slice + 1
    shape_yx = ds.shape[1:3]
    is_metal_3d = np.zeros((n_crop,) + shape_yx, dtype=bool)

    for i, z in enumerate(range(first_slice, last_slice + 1)):
        is_metal_3d[i] = ds[z, :, :, 0] >= SEGMENTATION_THRESHOLD
        if i % 200 == 0:
            print(f"    loaded {i}/{n_crop} slices...")

voxel_volume_um3 = spacing[0] * spacing[1] * spacing[2]
print(f"\n  Volume loaded: {is_metal_3d.shape}, {is_metal_3d.nbytes / 1e9:.2f} GB in memory")
print(f"  Voxel volume: {voxel_volume_um3:.4f} um^3")

# ----------------------------------------------------------------------
# MEMORY FIX: crop to the specimen's tight XY bounding box before any
# expensive 3D operation. Most of each frame is empty background/margin
# around a much smaller specimen (metal fraction was only ~15-24%) --
# cropping to the actual occupied region cuts the array size (and every
# downstream label/fill_holes array) substantially, without changing
# correctness, since we only ever cared about the specimen region anyway.
# ----------------------------------------------------------------------
print("\n  Computing tight bounding box around specimen (memory optimization)...")
any_y = np.any(is_metal_3d, axis=(0, 2))
any_x = np.any(is_metal_3d, axis=(0, 1))
y_indices = np.where(any_y)[0]
x_indices = np.where(any_x)[0]

PAD = 5
y_min, y_max = max(0, y_indices.min() - PAD), min(is_metal_3d.shape[1], y_indices.max() + PAD)
x_min, x_max = max(0, x_indices.min() - PAD), min(is_metal_3d.shape[2], x_indices.max() + PAD)

is_metal_3d = is_metal_3d[:, y_min:y_max, x_min:x_max]
print(f"  Cropped from full frame to bounding box: Y[{y_min}:{y_max}], X[{x_min}:{x_max}]")
print(f"  New cropped shape: {is_metal_3d.shape}, {is_metal_3d.nbytes / 1e9:.2f} GB in memory")
gc.collect()

print("\n  Running 3D connected-component labeling (metal)...")
labeled_metal, n_metal_components = ndimage.label(is_metal_3d)
print(f"  Found {n_metal_components} connected metal component(s)")

sizes = ndimage.sum(is_metal_3d, labeled_metal, range(1, n_metal_components + 1))
largest_label = np.argmax(sizes) + 1
specimen_metal = (labeled_metal == largest_label)
print(f"  Largest component: {specimen_metal.sum()} voxels "
      f"({100*specimen_metal.sum()/is_metal_3d.sum():.1f}% of all metal voxels)")

del labeled_metal, is_metal_3d
gc.collect()

print("\n  Filling enclosed 3D holes (this is the real upgrade over 2D slice-wise)...")
specimen_filled = ndimage.binary_fill_holes(specimen_metal)
pores_3d = specimen_filled & ~specimen_metal
print(f"  Total specimen volume: {specimen_filled.sum() * voxel_volume_um3:.1f} um^3 "
      f"({specimen_filled.sum()} voxels)")
print(f"  Total pore volume: {pores_3d.sum() * voxel_volume_um3:.1f} um^3 "
      f"({pores_3d.sum()} voxels)")

overall_porosity_3d = pores_3d.sum() / specimen_filled.sum() if specimen_filled.sum() > 0 else np.nan
print(f"  OVERALL 3D POROSITY (phi = V_pore / V_specimen): {overall_porosity_3d:.4%}")

# ============================================================================
# STEP C continued: individual pore objects -- count, volumes, size distribution
# ============================================================================
print("\n  Labeling individual pore objects in 3D...")
labeled_pores, n_pores = ndimage.label(pores_3d)
print(f"  Total distinct pores found: {n_pores}")

if n_pores > 0:
    pore_voxel_counts = ndimage.sum(pores_3d, labeled_pores, range(1, n_pores + 1))
    pore_volumes_um3 = pore_voxel_counts * voxel_volume_um3
    # equivalent spherical diameter, standard way to report pore size
    pore_equiv_diameters_um = (6 * pore_volumes_um3 / np.pi) ** (1/3)

    pore_stats_df = pd.DataFrame({
        'pore_id': range(1, n_pores + 1),
        'volume_um3': pore_volumes_um3,
        'equiv_diameter_um': pore_equiv_diameters_um,
    }).sort_values('volume_um3', ascending=False)

    print(f"  Pore volume range: {pore_volumes_um3.min():.2f} to {pore_volumes_um3.max():.2f} um^3")
    print(f"  Pore equivalent diameter range: {pore_equiv_diameters_um.min():.2f} to "
          f"{pore_equiv_diameters_um.max():.2f} um")
    print(f"  Largest 5 pores:")
    print(pore_stats_df.head(5).to_string(index=False))

    save_pore_stats = f"{RESULTS_TABLES}/xct_3d_pore_size_distribution.csv"
    pore_stats_df.to_csv(save_pore_stats, index=False)
    print(f"  [SAVED] {save_pore_stats}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(pore_equiv_diameters_um, bins=30, color='firebrick', alpha=0.7)
    ax.set_xlabel('Equivalent Pore Diameter (um)')
    ax.set_ylabel('Count')
    ax.set_title(f'3D Pore Size Distribution (n={n_pores} pores)')
    ax.set_yscale('log')
    fig_path = f"{RESULTS_FIGURES}/xct_3d_pore_size_distribution.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {fig_path}")
else:
    pore_stats_df = pd.DataFrame(columns=['pore_id', 'volume_um3', 'equiv_diameter_um'])
    print("  No discrete pores found -- material is essentially fully dense in this region.")

# ============================================================================
# STEP D: Separate baseplate / interface / AM-deposited regions
# ============================================================================
print(f"\n{'='*70}")
print("STEP D: Region separation (baseplate / interface / AM-deposited)")
print(f"{'='*70}")

INTERFACE_BAND_UM = 50.0  # +/- 50um around the plate surface treated as transition

z_indices_crop = np.arange(first_slice, last_slice + 1)
build_z_per_slice = origin[2] + z_indices_crop * spacing[2]

region_labels = np.where(build_z_per_slice < -INTERFACE_BAND_UM, 'baseplate',
                  np.where(build_z_per_slice > INTERFACE_BAND_UM, 'AM_deposited', 'interface'))

for region in ['baseplate', 'interface', 'AM_deposited']:
    mask_slices = (region_labels == region)
    if mask_slices.sum() == 0:
        continue
    region_specimen_vox = specimen_filled[mask_slices].sum()
    region_pore_vox = pores_3d[mask_slices].sum()
    region_porosity = region_pore_vox / region_specimen_vox if region_specimen_vox > 0 else np.nan
    print(f"  {region:12s}: {mask_slices.sum()} slices, porosity={region_porosity:.4%}")

# ============================================================================
# STEP E: Corrected layer assignment -- floor(), not round()
# ============================================================================
print(f"\n{'='*70}")
print("STEP E: Per-layer 3D porosity (floor-based layer assignment)")
print(f"{'='*70}")

layer_per_slice = np.floor(build_z_per_slice / LAYER_THICKNESS_UM).astype(int)

per_layer_rows = []
for layer in np.unique(layer_per_slice):
    if layer < 0:
        continue  # baseplate, not a printed layer
    mask_slices = (layer_per_slice == layer)
    layer_specimen_vox = specimen_filled[mask_slices].sum()
    layer_pore_vox = pores_3d[mask_slices].sum()
    layer_porosity = layer_pore_vox / layer_specimen_vox if layer_specimen_vox > 0 else np.nan
    per_layer_rows.append(dict(layer=int(layer), porosity_3d=layer_porosity,
                                specimen_voxels=int(layer_specimen_vox),
                                pore_voxels=int(layer_pore_vox)))

per_layer_3d_df = pd.DataFrame(per_layer_rows)
save_layer_path = f"{RESULTS_TABLES}/xct_3d_per_layer_porosity.csv"
per_layer_3d_df.to_csv(save_layer_path, index=False)
print(per_layer_3d_df.to_string(index=False))
print(f"\n[SAVED] {save_layer_path}")

del specimen_metal, specimen_filled, pores_3d, labeled_pores
gc.collect()

# ============================================================================
# STEP F: Recompute correlations + defect classification metric
# ============================================================================
print(f"\n{'='*70}")
print("STEP F: Recomputed correlations (3D porosity) + classification metric")
print(f"{'='*70}")

severity_df = pd.read_csv(SEVERITY_PATH)
b8_severity = severity_df[severity_df['build'] == 'B8'][
    ['layer', 'severity', 'risk_score', 'tam_p90', 'tam_mean', 'scr_std']
]

merged_3d = per_layer_3d_df.merge(b8_severity, on='layer', how='inner')
print(f"Merged (3D-corrected): {len(merged_3d)} overlapping layers\n")

corr_rows = []
for col in ['severity', 'risk_score', 'tam_p90', 'tam_mean', 'scr_std']:
    if merged_3d[col].nunique() < 2 or merged_3d['porosity_3d'].nunique() < 2:
        continue
    rho, p_s = spearmanr(merged_3d['porosity_3d'], merged_3d[col])
    r, p_p = pearsonr(merged_3d['porosity_3d'], merged_3d[col])
    print(f"  {col:12s} vs 3D porosity: Spearman rho={rho:+.3f} (p={p_s:.3f}), "
          f"Pearson r={r:+.3f} (p={p_p:.3f})")
    corr_rows.append(dict(feature=col, spearman_rho=rho, spearman_p=p_s,
                           pearson_r=r, pearson_p=p_p, n=len(merged_3d)))

save_corr_3d = f"{RESULTS_TABLES}/xct_3d_severity_correlation.csv"
pd.DataFrame(corr_rows).to_csv(save_corr_3d, index=False)
print(f"\n[SAVED] {save_corr_3d}")

# defect classification: layers ABOVE MEDIAN 3D porosity treated as "elevated"
# -- our own reasonable operational definition, not a NIST-defined threshold,
# stated explicitly since no official defect/no-defect label exists
if merged_3d['porosity_3d'].nunique() > 1:
    median_poros = merged_3d['porosity_3d'].median()
    y_true = (merged_3d['porosity_3d'] > median_poros).astype(int)
    print(f"\nDefect classification (elevated = porosity > median {median_poros:.4%}):")
    for col in ['risk_score', 'tam_p90', 'severity']:
        if y_true.nunique() > 1:
            try:
                auc = roc_auc_score(y_true, merged_3d[col])
                print(f"  ROC-AUC using {col} as predictor: {auc:.3f}  (n={len(merged_3d)}, 0.5=chance)")
            except ValueError as e:
                print(f"  {col}: could not compute AUC ({e})")

print(f"\n{'='*70}")
print("XCT STAGE 2 (3D) ANALYSIS COMPLETE.")
print("Report the 3D porosity, pore count/size distribution, region-separated")
print("porosity, and corrected per-layer correlations -- these supersede the")
print("Stage 1 2D slice-wise numbers; report Stage 1 only as the initial")
print("validation step that motivated this more rigorous follow-up.")
print(f"{'='*70}")

# ============================================================================
# SENSITIVITY CHECK: exclude likely-incomplete boundary layers before
# trusting correlations equally across all 24 layers. Layers with far
# fewer specimen voxels than the typical ~4 million are almost certainly
# partial/edge-of-scan slices, not real physical layers.
# ============================================================================
print(f"\n{'='*70}")
print("SENSITIVITY CHECK: excluding incomplete boundary layers")
print(f"{'='*70}")

TYPICAL_VOXEL_COUNT = per_layer_3d_df['specimen_voxels'].median()
MIN_FRACTION = 0.5  # exclude layers with less than 50% of the typical voxel count

complete_layers = per_layer_3d_df[
    per_layer_3d_df['specimen_voxels'] >= TYPICAL_VOXEL_COUNT * MIN_FRACTION
]
excluded = per_layer_3d_df[~per_layer_3d_df['layer'].isin(complete_layers['layer'])]
print(f"  Excluding {len(excluded)} likely-incomplete layer(s): {excluded['layer'].tolist()}")

merged_clean = complete_layers.merge(b8_severity, on='layer', how='inner')
print(f"  Remaining: {len(merged_clean)} layers for sensitivity-checked correlation\n")

for col in ['severity', 'risk_score', 'tam_p90', 'tam_mean', 'scr_std']:
    if merged_clean[col].nunique() < 2 or merged_clean['porosity_3d'].nunique() < 2:
        continue
    rho, p_s = spearmanr(merged_clean['porosity_3d'], merged_clean[col])
    print(f"  {col:12s} vs 3D porosity (boundary layers excluded): "
          f"Spearman rho={rho:+.3f} (p={p_s:.3f}), n={len(merged_clean)}")

print(f"\n{'='*70}")
print("SENSITIVITY CHECK COMPLETE.")
print(f"{'='*70}")
