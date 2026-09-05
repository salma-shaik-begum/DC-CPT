"""
Check existing Drive folder structure, then create ONLY what's missing
============================================================================
Run this first, every session. It:
  1. Prints the ACTUAL current folder tree in Drive (so you see what's
     really there, not what you assume is there -- this is exactly the
     kind of check that would have caught the earlier Data/Raw vs
     data/raw casing mismatch immediately).
  2. Checks each required folder CASE-INSENSITIVELY against what already
     exists, so it won't create a duplicate "results" next to an
     existing "Results".
  3. Creates only the folders that are genuinely missing.
  4. Prints a final confirmed tree so you can visually verify everything
     is where it should be before running any other script.
"""

import os
from google.colab import drive

drive.mount('/content/drive')

BASE = '/content/drive/MyDrive/DC-CPT-Project'

REQUIRED_FOLDERS = [
    'Data/Raw',
    'Data/processed',
    'notebooks',
    'src',
    'results/tables',
    'results/figures',
    'docs',
]


def print_current_tree(base):
    print(f"Current folder tree under {base}:\n")
    if not os.path.isdir(base):
        print("  (base project folder does not exist yet)")
        return
    for root, dirs, files in os.walk(base):
        level = root.replace(base, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root) or base}/")
        subindent = '  ' * (level + 1)
        for f in files:
            print(f"{subindent}{f}")


def existing_dirs_lowercase(base):
    """Map of lowercase relative path -> actual existing path, so we can
    check case-insensitively without creating duplicates like
    'Data/Raw' AND 'data/raw' side by side."""
    existing = {}
    if not os.path.isdir(base):
        return existing
    for root, dirs, _ in os.walk(base):
        for d in dirs:
            full = os.path.join(root, d)
            rel = os.path.relpath(full, base)
            existing[rel.lower()] = rel
    return existing


def create_missing_folders(base, required):
    os.makedirs(base, exist_ok=True)
    existing = existing_dirs_lowercase(base)

    print(f"\n{'='*70}")
    print("Checking required folders (case-insensitive):")
    print(f"{'='*70}")

    for folder in required:
        key = folder.lower()
        if key in existing:
            print(f"  [EXISTS]  {existing[key]}  (matches required '{folder}')")
        else:
            full_path = os.path.join(base, folder)
            os.makedirs(full_path, exist_ok=True)
            confirmed = os.path.isdir(full_path)
            print(f"  [{'CREATED' if confirmed else 'FAILED'}]  {folder}")


if __name__ == '__main__':
    print("BEFORE:")
    print_current_tree(BASE)

    create_missing_folders(BASE, REQUIRED_FOLDERS)

    print(f"\n{'='*70}")
    print("AFTER:")
    print(f"{'='*70}")
    print_current_tree(BASE)
