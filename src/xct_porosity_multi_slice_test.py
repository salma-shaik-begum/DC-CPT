"""
XCT Porosity -- test across MULTIPLE slices, not just one
========================================================================
A single near-zero porosity reading could mean either (a) this is
genuinely a dense, healthy region -- plausible and common in LPBF --
or (b) the method is missing small pores. Testing 20 slices spread
across the full Z range tells us which: if porosity varies meaningfully
across slices, the method is sensitive and working; if it's uniformly
zero everywhere, that itself is informative (very dense build) but
worth flagging as a real, honest finding either way.
"""

import os
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

BASE = '/content/drive/MyDrive/DC-CPT-Project'
DREAM3D_PATH = f"{BASE}/Data/Raw/XCT_reconstruction/2 - XRCT_stack_BuildCoordSystem.dream3d"
DATASET_PATH = 'DataContainers/ImageDataContainer/CellData/ImageData'
RESULTS_TABLES = f"{BASE}/results/tables"
os.makedirs(RESULTS_TABLES, exist_ok=True)

METAL_THRESHOLD = 33765
N_TEST_SLICES = 20


def compute_slice_porosity(slice_2d, threshold=METAL_THRESHOLD):
    is_metal = slice_2d >= threshold
    border_touch = (is_metal[0, :].any() or is_metal[-1, :].any() or
                     is_metal[:, 0].any() or is_metal[:, -1].any())

    labeled, n_components = ndimage.label(is_metal)
    if n_components == 0:
        return dict(metal_fraction=0.0, porosity_fraction=np.nan,
                     n_components=0, border_touch=border_touch, specimen_area=0)

    sizes = ndimage.sum(is_metal, labeled, range(1, n_components + 1))
    largest_label = np.argmax(sizes) + 1
    specimen_metal = (labeled == largest_label)

    specimen_filled = ndimage.binary_fill_holes(specimen_metal)
    pores = specimen_filled & ~specimen_metal

    specimen_area = specimen_filled.sum()
    porosity_fraction = pores.sum() / specimen_area if specimen_area > 0 else np.nan

    return dict(
        metal_fraction=is_metal.mean(),
        porosity_fraction=porosity_fraction,
        n_components=n_components,
        border_touch=border_touch,
        specimen_area=specimen_area,
    )


with h5py.File(DREAM3D_PATH, 'r') as f:
    ds = f[DATASET_PATH]
    n_slices_total = ds.shape[0]
    test_z_indices = np.linspace(0, n_slices_total - 1, N_TEST_SLICES).astype(int)

    print(f"Testing {N_TEST_SLICES} slices spread across Z=0 to Z={n_slices_total-1}...\n")

    results = []
    for z in test_z_indices:
        slice_2d = ds[z, :, :, 0].astype(np.float64)
        stats = compute_slice_porosity(slice_2d)
        stats['z_index'] = z
        results.append(stats)
        print(f"  Z={z:4d}: metal_frac={stats['metal_fraction']:.1%}, "
              f"porosity={stats['porosity_fraction']:.4%}, "
              f"n_components={stats['n_components']}, "
              f"border_touch={stats['border_touch']}")

results_df = pd.DataFrame(results)
out_path = f"{RESULTS_TABLES}/xct_multi_slice_porosity_test.csv"
results_df.to_csv(out_path, index=False)
print(f"\n[SAVED] {out_path}")

print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")
valid = results_df.dropna(subset=['porosity_fraction'])
print(f"  Porosity range across tested slices: {valid['porosity_fraction'].min():.4%} to {valid['porosity_fraction'].max():.4%}")
print(f"  Mean: {valid['porosity_fraction'].mean():.4%}, Std: {valid['porosity_fraction'].std():.4%}")
print(f"  Slices with any detected metal: {(results_df['metal_fraction'] > 0).sum()} of {N_TEST_SLICES}")
print(f"  Slices where specimen touches border: {results_df['border_touch'].sum()} of {N_TEST_SLICES}")

if valid['porosity_fraction'].std() > 0.0001:
    print("\n  Porosity VARIES across slices -- method appears sensitive, not just")
    print("  uniformly returning zero. This is a good sign the pipeline works.")
else:
    print("\n  Porosity is uniformly near-zero across all tested slices. This could")
    print("  be a genuinely very dense build (plausible for well-parametrized LPBF),")
    print("  or indicate the threshold/method needs refinement. Worth reporting as")
    print("  an honest finding either way, with this caveat noted.")

print(f"\n{'='*70}")
print("MULTI-SLICE TEST COMPLETE.")
print(f"{'='*70}")
