"""
Inspect the unzipped XCT reconstruction -- lists what's actually there,
then opens the .dream3d (HDF5) file structure WITHOUT loading the full
volume into memory. Same careful pattern as every dataset so far: look
before loading.
"""

import os
import h5py

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

RAW_DIR = '/content/drive/MyDrive/DC-CPT-Project/Data/Raw'

# ============================================================================
# STEP 1: List everything under Data/Raw/ with sizes, so we know exactly
# what we're working with before opening anything
# ============================================================================
print("="*70)
print("Files under Data/Raw/ (recursive):")
print("="*70)

dream3d_files = []
xdmf_files = []

for root, dirs, files in os.walk(RAW_DIR):
    for f in files:
        full_path = os.path.join(root, f)
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        print(f"  {full_path}  ({size_mb:.1f} MB)")
        if f.endswith('.dream3d'):
            dream3d_files.append(full_path)
        elif f.endswith('.xdmf'):
            xdmf_files.append(full_path)

print(f"\nFound {len(dream3d_files)} .dream3d file(s), {len(xdmf_files)} .xdmf file(s)")

# ============================================================================
# STEP 2: Open each .dream3d file's STRUCTURE only (HDF5 group/dataset
# tree, shapes, dtypes) -- does NOT load actual voxel data into memory
# ============================================================================
for path in dream3d_files:
    print(f"\n{'='*70}")
    print(f"Structure of: {os.path.basename(path)}")
    print(f"{'='*70}")

    with h5py.File(path, 'r') as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  [DATASET] {name}  shape={obj.shape}  dtype={obj.dtype}")
            else:
                print(f"  [GROUP]   {name}")
        f.visititems(show)

        print("\n  Top-level attributes:")
        for k, v in f.attrs.items():
            print(f"    {k}: {v}")

print(f"\n{'='*70}")
print("INSPECTION COMPLETE -- no voxel data loaded yet, structure only.")
print("Paste this output back and the next script will target the exact")
print("dataset path (e.g. .../CellData/GrayValue or similar) for the")
print("actual intensity data, without loading the whole volume blindly.")
print(f"{'='*70}")
