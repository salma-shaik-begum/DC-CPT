"""
XCT Porosity Extraction -- 2D test on one slice first
========================================================================
Correctly distinguishes INTERNAL pores (real defects, enclosed by metal)
from EXTERNAL background/mounting material (not defects), using the
standard approach: threshold to find solid metal, fill enclosed holes to
get the specimen's solid silhouette, then anything dark trapped INSIDE
that silhouette is a genuine pore.

Tested on a single 2D slice first, numerically (no image viewing needed),
before committing to the full 882-slice 3D volume.
"""

import os
import h5py
import numpy as np
from scipy import ndimage

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

BASE = '/content/drive/MyDrive/DC-CPT-Project'
DREAM3D_PATH = f"{BASE}/Data/Raw/XCT_reconstruction/2 - XRCT_stack_BuildCoordSystem.dream3d"
DATASET_PATH = 'DataContainers/ImageDataContainer/CellData/ImageData'

# threshold separating METAL from everything else (mounting + air + pores).
# Set between the mid population (~15748, likely mounting resin) and the
# high population (~51783, likely metal) -- refine this once we confirm
# what the mid population actually is.
METAL_THRESHOLD = 33765

with h5py.File(DREAM3D_PATH, 'r') as f:
    ds = f[DATASET_PATH]
    mid_z = ds.shape[0] // 2
    mid_slice = ds[mid_z, :, :, 0].astype(np.float64)

print("="*70)
print(f"Testing pore segmentation on slice Z={mid_z}")
print("="*70)

# Step 1: binary mask of solid metal
is_metal = mid_slice >= METAL_THRESHOLD
print(f"  Metal fraction (raw threshold): {is_metal.mean():.1%}")

# Step 2: does the metal region touch the image border? If so, fill_holes
# won't work correctly (background could "leak" through the touching edge
# and get counted as enclosed). Check this explicitly, don't assume.
border_touch = (is_metal[0, :].any() or is_metal[-1, :].any() or
                 is_metal[:, 0].any() or is_metal[:, -1].any())
print(f"  Metal region touches image border: {border_touch}")

# Step 3: keep only the LARGEST connected metal component (the specimen
# itself, not small unconnected bright noise elsewhere in the frame)
labeled, n_components = ndimage.label(is_metal)
if n_components > 0:
    sizes = ndimage.sum(is_metal, labeled, range(1, n_components + 1))
    largest_label = np.argmax(sizes) + 1
    specimen_metal = (labeled == largest_label)
    print(f"  Found {n_components} connected metal component(s); "
          f"largest covers {specimen_metal.sum()} pixels "
          f"({100*specimen_metal.sum()/is_metal.sum():.1f}% of all metal pixels)")
else:
    specimen_metal = is_metal
    print("  WARNING: no connected components found -- check threshold")

# Step 4: fill enclosed holes -- this gives the SOLID SILHOUETTE of the
# specimen, including any internal pores as if they were filled in
specimen_filled = ndimage.binary_fill_holes(specimen_metal)

# Step 5: pores = filled silhouette MINUS actual metal = holes that were
# enclosed by metal (real internal defects), not background outside the
# specimen boundary (which fill_holes correctly excludes)
pores = specimen_filled & ~specimen_metal

specimen_area = specimen_filled.sum()
pore_area = pores.sum()
porosity_fraction = pore_area / specimen_area if specimen_area > 0 else np.nan

print(f"\n  Specimen silhouette area: {specimen_area} pixels")
print(f"  Detected pore area: {pore_area} pixels")
print(f"  POROSITY FRACTION (this slice): {porosity_fraction:.4%}")

# Step 6: sanity range check -- real LPBF porosity is typically well under
# 5%, often under 1% for good process parameters. A wildly high number
# here (e.g. >20%) would indicate the threshold or method needs adjustment,
# not that the material is genuinely 20% void.
if porosity_fraction > 0.20:
    print("\n  FLAG: porosity fraction is unusually high (>20%) -- this likely")
    print("  indicates a segmentation issue (threshold too high, specimen")
    print("  touching border, or wrong slice), not genuine material porosity.")
    print("  Do not trust this number yet -- diagnose before proceeding to 3D.")
elif porosity_fraction < 0.0001:
    print("\n  Porosity fraction is near zero -- plausible for a healthy region,")
    print("  but also check this isn't a false negative (verify on other slices).")
else:
    print("\n  Porosity fraction is in a physically plausible range for LPBF IN718.")
    print("  Reasonable to proceed to testing a few more slices, then the full volume.")

print(f"\n{'='*70}")
print("2D TEST COMPLETE.")
print(f"{'='*70}")
