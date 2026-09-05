"""
XCT Full Porosity Profile -- all 882 slices, converted to build Z-coordinates
================================================================================
Streams through every slice (never loads the full 3D volume into memory
at once), computes porosity per slice using the validated method, then
converts each slice's Z-index into build-coordinate micrometers and
approximate layer number -- ready to align against TAM/SCR data.

IMPORTANT SCOPE NOTE: this XCT volume covers ~1.77mm in Z, starting
below the baseplate (ORIGIN_Z is negative) and extending only a short
way into the AM-deposited region. At 40um/layer, this covers roughly
the first ~15-20 build layers, NOT the full 312-layer build. This is a
genuine, honest partial-region validation, not full-build validation --
report it as such.
"""

import os
import time
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

BASE = '/content/drive/MyDrive/DC-CPT-Project'
DREAM3D_PATH = f"{BASE}/Data/Raw/XCT_reconstruction/2 - XRCT_stack_BuildCoordSystem.dream3d"
DATASET_PATH = 'DataContainers/ImageDataContainer/CellData/ImageData'
GEOMETRY_GROUP = 'DataContainers/ImageDataContainer/_SIMPL_GEOMETRY'
RESULTS_TABLES = f"{BASE}/results/tables"
RESULTS_FIGURES = f"{BASE}/results/figures"
os.makedirs(RESULTS_TABLES, exist_ok=True)
os.makedirs(RESULTS_FIGURES, exist_ok=True)

METAL_THRESHOLD = 33765
LAYER_THICKNESS_UM = 40.0


def compute_slice_porosity(slice_2d, threshold=METAL_THRESHOLD):
    is_metal = slice_2d >= threshold
    labeled, n_components = ndimage.label(is_metal)
    if n_components == 0:
        return dict(metal_fraction=0.0, porosity_fraction=np.nan,
                     n_components=0, specimen_area=0)
    sizes = ndimage.sum(is_metal, labeled, range(1, n_components + 1))
    largest_label = np.argmax(sizes) + 1
    specimen_metal = (labeled == largest_label)
    specimen_filled = ndimage.binary_fill_holes(specimen_metal)
    pores = specimen_filled & ~specimen_metal
    specimen_area = specimen_filled.sum()
    porosity_fraction = pores.sum() / specimen_area if specimen_area > 0 else np.nan
    return dict(metal_fraction=is_metal.mean(), porosity_fraction=porosity_fraction,
                n_components=n_components, specimen_area=specimen_area)


with h5py.File(DREAM3D_PATH, 'r') as f:
    origin = f[f'{GEOMETRY_GROUP}/ORIGIN'][:]
    spacing = f[f'{GEOMETRY_GROUP}/SPACING'][:]
    ds = f[DATASET_PATH]
    n_slices = ds.shape[0]

    origin_z, spacing_z = origin[2], spacing[2]

    print(f"Processing all {n_slices} slices (streaming, one at a time)...")
    print(f"  Origin Z: {origin_z:.1f} um, Spacing Z: {spacing_z:.4f} um/voxel")
    print("  (This may take a few minutes -- progress printed every 100 slices)\n")

    results = []
    t0 = time.time()
    for z in range(n_slices):
        slice_2d = ds[z, :, :, 0].astype(np.float64)
        stats = compute_slice_porosity(slice_2d)

        build_z_um = origin_z + z * spacing_z
        approx_layer = build_z_um / LAYER_THICKNESS_UM  # negative = below baseplate top

        stats['z_index'] = z
        stats['build_z_um'] = build_z_um
        stats['approx_layer'] = approx_layer
        results.append(stats)

        if z % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Slice {z}/{n_slices}  (build_z={build_z_um:.0f}um, "
                  f"~layer {approx_layer:.1f})  [{elapsed:.0f}s elapsed]")

results_df = pd.DataFrame(results)
out_path = f"{RESULTS_TABLES}/xct_full_porosity_profile.csv"
results_df.to_csv(out_path, index=False)
print(f"\n[SAVED] {out_path}  ({len(results_df)} rows)")

# ============================================================================
# Figure: porosity profile vs build Z / approximate layer
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

axes[0].plot(results_df['build_z_um'], results_df['metal_fraction'] * 100, color='gray')
axes[0].axvline(0, color='red', linestyle='--', alpha=0.6, label='Build plate surface (Z=0)')
axes[0].set_ylabel('Metal Fraction (%)')
axes[0].legend()
axes[0].set_title('XCT Metal Fraction and Porosity vs. Build Z-Coordinate')

axes[1].plot(results_df['build_z_um'], results_df['porosity_fraction'] * 100, color='firebrick')
axes[1].axvline(0, color='red', linestyle='--', alpha=0.6)
axes[1].set_xlabel('Build Z-Coordinate (micrometers, 0 = plate surface)')
axes[1].set_ylabel('Porosity Fraction (%)')

plt.tight_layout()
fig_path = f"{RESULTS_FIGURES}/xct_porosity_profile.png"
fig.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"[SAVED] {fig_path}")

# ============================================================================
# Identify which slices fall within the actual AM-deposited region
# (build_z_um > 0) -- this is the only part comparable to your TAM/SCR
# layer data, since Z<0 is baseplate material, not printed layers
# ============================================================================
am_region = results_df[results_df['build_z_um'] > 0]
print(f"\n{'='*70}")
print(f"AM-deposited region only (build_z_um > 0): {len(am_region)} slices")
print(f"  Covers approximately layers 0 to {am_region['approx_layer'].max():.1f}")
print(f"  Mean porosity in this region: {am_region['porosity_fraction'].mean():.4%}")
print(f"  Max porosity in this region: {am_region['porosity_fraction'].max():.4%} "
      f"(at build_z={am_region.loc[am_region['porosity_fraction'].idxmax(), 'build_z_um']:.0f}um)")
print(f"{'='*70}")

print(f"\n{'='*70}")
print("FULL POROSITY PROFILE COMPLETE.")
print("Next step: aggregate this to per-layer porosity and merge against")
print("your TAM/SCR severity data for the overlapping layer range.")
print(f"{'='*70}")
