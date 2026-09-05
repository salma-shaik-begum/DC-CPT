"""
Project Setup - run this FIRST, in every fresh Colab session
=================================================================
Creates the full folder structure in Drive (safe to rerun -- exist_ok=True
means it never errors or duplicates), and defines the standard paths every
other script should import instead of hardcoding strings. This is the fix
for "nothing is getting saved" -- every script from here on imports PATHS
from this file rather than writing its own path logic.
"""

import os

BASE = '/content/drive/MyDrive/DC-CPT-Project'

PATHS = {
    'base': BASE,
    'data_raw': f'{BASE}/Data/Raw',
    'data_processed': f'{BASE}/Data/processed',
    'notebooks': f'{BASE}/notebooks',
    'src': f'{BASE}/src',
    'results': f'{BASE}/results',
    'results_tables': f'{BASE}/results/tables',
    'results_figures': f'{BASE}/results/figures',
    'docs': f'{BASE}/docs',
}


def setup_project():
    for name, path in PATHS.items():
        os.makedirs(path, exist_ok=True)
        exists = os.path.isdir(path)
        print(f"  [{'OK' if exists else 'FAILED'}] {name:<18} -> {path}")
    return PATHS


if __name__ == '__main__':
    from google.colab import drive
    drive.mount('/content/drive')
    print("Setting up DC-CPT-Project folder structure in Drive:\n")
    setup_project()
    print("\nDone. Every future script should start with:")
    print("  from project_setup import PATHS")
    print("  # then use PATHS['results_tables'], PATHS['results_figures'], etc.")
