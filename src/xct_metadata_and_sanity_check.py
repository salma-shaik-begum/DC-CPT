"""
XCT Step 2: Read coordinate metadata, sanity-check ONE slice
========================================================================
Reads ORIGIN/SPACING/DIMENSIONS (tiny, instant) to establish the
voxel-to-micrometer coordinate mapping, then loads and visualizes ONE
middle Z-slice (a 2D read, not the full 3D volume) to confirm the data
looks like a real CT scan before committing to any full-volume work.

h5py can read a slice directly from disk without loading the whole
1.1-billion-voxel array into RAM -- this is safe to run even on modest
Colab memory.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

BASE = '/content/drive/MyDrive/DC-CPT-Project'
DREAM3D_PATH = f"{BASE}/Data/Raw/XCT_reconstruction/2 - XRCT_stack_BuildCoordSystem.dream3d"
FIGURES_DIR = f"{BASE}/results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASET_PATH = 'DataContainers/ImageDataContainer/CellData/ImageData'
GEOMETRY_GROUP = 'DataContainers/ImageDataContainer/_SIMPL_GEOMETRY'

with h5py.File(DREAM3D_PATH, 'r') as f:
    dims = f[f'{GEOMETRY_GROUP}/DIMENSIONS'][:]
    origin = f[f'{GEOMETRY_GROUP}/ORIGIN'][:]
    spacing = f[f'{GEOMETRY_GROUP}/SPACING'][:]

    print("="*70)
    print("Coordinate metadata")
    print("="*70)
    print(f"  DIMENSIONS (voxel counts, X/Y/Z): {dims}")
    print(f"  ORIGIN (micrometers, X/Y/Z):      {origin}")
    print(f"  SPACING (micrometers/voxel):      {spacing}")

    extent_um = dims * spacing
    print(f"\n  Physical extent (micrometers, X/Y/Z): {extent_um}")
    print(f"  Physical extent (millimeters, X/Y/Z): {extent_um / 1000}")

    # dataset shape is (Z, Y, X, 1) in numpy/h5py read order -- DREAM3D
    # stores arrays with the FASTEST-varying dimension last, which for a
    # 3D image array typically means shape[0]=Z, shape[1]=Y, shape[2]=X
    ds = f[DATASET_PATH]
    print(f"\n  Dataset shape (as stored, likely Z,Y,X,1): {ds.shape}")

    mid_z = ds.shape[0] // 2
    print(f"\n  Reading middle slice at Z-index={mid_z} (single 2D read, not full volume)...")
    mid_slice = ds[mid_z, :, :, 0]  # single slice read -- does not load full 3D array

print(f"\n  Slice shape: {mid_slice.shape}, dtype: {mid_slice.dtype}")
print(f"  Intensity range in this slice: min={mid_slice.min()}, max={mid_slice.max()}")

# ============================================================================
# Visualize the slice + its intensity histogram -- confirms real CT data
# and gives a first look at whether metal/void are bimodally separable
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

axes[0].imshow(mid_slice, cmap='gray')
axes[0].set_title(f'XCT Middle Slice (Z-index={mid_z})')
axes[0].axis('off')

axes[1].hist(mid_slice.flatten(), bins=100, color='steelblue')
axes[1].set_xlabel('Voxel Intensity (16-bit)')
axes[1].set_ylabel('Count')
axes[1].set_title('Intensity Histogram\n(look for bimodal peaks: void vs. metal)')
axes[1].set_yscale('log')

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'xct_slice_sanity_check.png')
fig.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\n  [SAVED] {fig_path}")

print(f"\n{'='*70}")
print("SANITY CHECK COMPLETE.")
print("Check the saved figure: does the slice look like a real cross-section")
print("(specimen shape visible, not noise)? Does the histogram show two")
print("separable peaks (dark=void/background, bright=metal)? Paste the")
print("printed metadata + your read of the figure back, and the next step")
print("extracts just your ~500x500x750um region of interest for full")
print("3D porosity analysis.")
print(f"{'='*70}")
