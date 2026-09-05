"""
XCT Bimodality Check -- numeric, no image viewing needed
========================================================================
Instead of visually inspecting the histogram, this finds the actual
peaks in the intensity distribution programmatically using scipy, and
reports whether the data cleanly separates into void/metal populations.
This tells us everything we need to proceed, without any image upload.
"""

import os
import h5py
import numpy as np
from scipy.signal import find_peaks
from scipy import ndimage

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

BASE = '/content/drive/MyDrive/DC-CPT-Project'
DREAM3D_PATH = f"{BASE}/Data/Raw/XCT_reconstruction/2 - XRCT_stack_BuildCoordSystem.dream3d"
DATASET_PATH = 'DataContainers/ImageDataContainer/CellData/ImageData'

with h5py.File(DREAM3D_PATH, 'r') as f:
    ds = f[DATASET_PATH]
    mid_z = ds.shape[0] // 2
    mid_slice = ds[mid_z, :, :, 0].astype(np.float64)

print("="*70)
print("Basic statistics")
print("="*70)
print(f"  Shape: {mid_slice.shape}")
print(f"  Min: {mid_slice.min():.0f}, Max: {mid_slice.max():.0f}")
print(f"  Mean: {mid_slice.mean():.1f}, Std: {mid_slice.std():.1f}")
print(f"  Median: {np.median(mid_slice):.1f}")

# fraction of near-zero pixels (likely background/void/air outside specimen)
near_zero_frac = (mid_slice < 500).mean()
print(f"  Fraction of pixels below intensity 500: {near_zero_frac:.1%}")

# ============================================================================
# Numeric bimodality check: find peaks in the smoothed histogram
# ============================================================================
print(f"\n{'='*70}")
print("Histogram peak detection")
print(f"{'='*70}")

hist, bin_edges = np.histogram(mid_slice.flatten(), bins=200)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# smooth the histogram slightly to avoid noise being counted as fake peaks
hist_smooth = ndimage.gaussian_filter1d(hist.astype(float), sigma=2)

# find peaks with a minimum prominence relative to the tallest peak, so we
# only count REAL separated populations, not minor bumps
peaks, properties = find_peaks(hist_smooth, prominence=hist_smooth.max() * 0.02)

print(f"  Found {len(peaks)} significant peak(s) in the intensity histogram:")
for p in peaks:
    print(f"    Intensity ~{bin_centers[p]:.0f}  (count={hist_smooth[p]:.0f})")

if len(peaks) >= 2:
    print("\n  RESULT: Bimodal (or multi-modal) -- distinct populations exist.")
    print("  Simple global thresholding (e.g. Otsu) between the two main peaks")
    print("  should work well for metal/void segmentation.")
    # Otsu-style threshold: midpoint between the two most prominent peaks
    top_two = sorted(peaks, key=lambda p: hist_smooth[p], reverse=True)[:2]
    threshold_estimate = np.mean([bin_centers[p] for p in top_two])
    print(f"  Estimated threshold (midpoint between top 2 peaks): {threshold_estimate:.0f}")
else:
    print("\n  RESULT: Not clearly bimodal in this slice -- single dominant population.")
    print("  This could mean: (a) this slice is mostly one material (e.g. deep")
    print("  in solid, low porosity region), or (b) void/background wasn't")
    print("  captured in this particular slice. Check a slice near the top or")
    print("  edge of the volume next, not just the middle.")

# ============================================================================
# Quick test: what fraction of THIS slice would be classified void vs metal
# at a reasonable threshold, as a sanity check
# ============================================================================
if len(peaks) >= 2:
    below = (mid_slice < threshold_estimate).mean()
    above = (mid_slice >= threshold_estimate).mean()
    print(f"\n  At threshold {threshold_estimate:.0f}: {below:.1%} below (void/background), "
          f"{above:.1%} above (metal)")
    print("  NOTE: this single mid-slice includes mounting material and background")
    print("  around the specimen, not just the specimen itself -- this fraction is")
    print("  NOT yet a porosity estimate, just a sanity check that thresholding works.")

print(f"\n{'='*70}")
print("BIMODALITY CHECK COMPLETE.")
print(f"{'='*70}")
