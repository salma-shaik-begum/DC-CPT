"""
Assemble Final Zenodo Package
========================================================================
Run this in Colab AFTER uploading and extracting the DC-CPT-Package.zip
into your Drive. This script copies your ACTUAL results (which only
exist in your Drive's results/tables, results/figures, Data/processed)
into the matching folders inside the package, so the final zip is fully
self-contained: code + results + docs, ready for Zenodo.

Usage: adjust PACKAGE_ROOT and EXISTING_PROJECT_ROOT below to match
your actual Drive paths, then run.
"""

import os
import shutil

PACKAGE_ROOT = '/content/drive/MyDrive/DC-CPT-Package'          # the extracted zip
EXISTING_PROJECT_ROOT = '/content/drive/MyDrive/DC-CPT-Project'  # your original working project

def copy_with_report(src_dir, dst_dir, extension):
    if not os.path.isdir(src_dir):
        print(f"  [SKIP] {src_dir} does not exist")
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for f in os.listdir(src_dir):
        if f.endswith(extension):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            count += 1
    print(f"  [COPIED] {count} {extension} file(s): {src_dir} -> {dst_dir}")
    return count

print("="*70)
print("Assembling final Zenodo-ready package")
print("="*70)

print("\nCopying result tables (CSV)...")
copy_with_report(f"{EXISTING_PROJECT_ROOT}/results/tables",
                  f"{PACKAGE_ROOT}/results/tables", '.csv')

print("\nCopying result figures (PNG)...")
copy_with_report(f"{EXISTING_PROJECT_ROOT}/results/figures",
                  f"{PACKAGE_ROOT}/results/figures", '.png')

print("\nCopying processed intermediate datasets (CSV)...")
copy_with_report(f"{EXISTING_PROJECT_ROOT}/Data/processed",
                  f"{PACKAGE_ROOT}/data/processed", '.csv')

# ============================================================================
# Verification: does every script in src/ have a plausible corresponding
# output in results/? Flags gaps rather than assuming completeness.
# ============================================================================
print(f"\n{'='*70}")
print("Verification: checking for scripts with no matching output")
print(f"{'='*70}")

all_tables = set(os.listdir(f"{PACKAGE_ROOT}/results/tables")) if os.path.isdir(f"{PACKAGE_ROOT}/results/tables") else set()
all_figures = set(os.listdir(f"{PACKAGE_ROOT}/results/figures")) if os.path.isdir(f"{PACKAGE_ROOT}/results/figures") else set()
print(f"  Total table files present: {len(all_tables)}")
print(f"  Total figure files present: {len(all_figures)}")

expected_min_tables = [
    'gate1_fold_results.csv', 'gate2_fold_results.csv', 'gate3_summary_by_build.csv',
    'gate4_intent_comparison_tiers.csv', 'baseline_comparison_summary.csv',
    'ablation_results_pooled.csv', 'bootstrap_confidence_intervals.csv',
    'temporal_vs_independent_comparison.csv',
]
missing = [t for t in expected_min_tables if t not in all_tables]
if missing:
    print(f"\n  WARNING -- expected key tables not found: {missing}")
    print("  Check that all pipeline scripts were actually run and saved before")
    print("  finalizing the package -- this package is not yet complete.")
else:
    print("\n  All core expected tables present.")

print(f"\n{'='*70}")
print("ASSEMBLY COMPLETE.")
print("Next: zip the PACKAGE_ROOT folder for Zenodo upload (see final message).")
print(f"{'='*70}")
