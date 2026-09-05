"""
Save Utilities - import this in every script to actually persist outputs
=============================================================================
The recurring problem: scripts print results to Colab's cell output, which
looks like it worked, but nothing is written to disk/Drive unless you
explicitly call a save function. These wrappers make saving a single line,
and always print the saved path so you get a visible confirmation instead
of silent nothing.

Usage in any script:
    from save_utils import save_table, save_figure, save_text
    save_table(results_df, 'ablation_results')       -> results/tables/ablation_results.csv
    save_figure(fig, 'risk_coverage_curve')            -> results/figures/risk_coverage_curve.png
    save_text(summary_string, 'gate3_notes')           -> results/tables/gate3_notes.txt
"""

import os
import pandas as pd

try:
    from project_setup import PATHS, setup_project
    PATHS = setup_project()  # ensures folders exist even if setup wasn't run first
except ImportError:
    # fallback if project_setup.py isn't in the same session -- still works,
    # just defines paths directly rather than importing
    BASE = '/content/drive/MyDrive/DC-CPT-Project'
    PATHS = {
        'results_tables': f'{BASE}/results/tables',
        'results_figures': f'{BASE}/results/figures',
    }
    for p in PATHS.values():
        os.makedirs(p, exist_ok=True)


def save_table(df: pd.DataFrame, name: str):
    """Saves a DataFrame as CSV into results/tables/. Always prints the
    saved path AND confirms the file actually exists on disk afterward --
    don't trust a silent success, verify it."""
    if not name.endswith('.csv'):
        name += '.csv'
    path = os.path.join(PATHS['results_tables'], name)
    df.to_csv(path, index=False)
    confirmed = os.path.isfile(path)
    size_kb = os.path.getsize(path) / 1024 if confirmed else 0
    print(f"  [{'SAVED' if confirmed else 'FAILED'}] {path}  ({size_kb:.1f} KB)")
    return path


def save_figure(fig, name: str, dpi=200):
    """Saves a matplotlib figure into results/figures/."""
    if not name.endswith('.png'):
        name += '.png'
    path = os.path.join(PATHS['results_figures'], name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    confirmed = os.path.isfile(path)
    size_kb = os.path.getsize(path) / 1024 if confirmed else 0
    print(f"  [{'SAVED' if confirmed else 'FAILED'}] {path}  ({size_kb:.1f} KB)")
    return path


def save_text(text: str, name: str):
    """Saves any text output (summaries, logs, notes) into results/tables/."""
    if not name.endswith('.txt'):
        name += '.txt'
    path = os.path.join(PATHS['results_tables'], name)
    with open(path, 'w') as f:
        f.write(text)
    confirmed = os.path.isfile(path)
    print(f"  [{'SAVED' if confirmed else 'FAILED'}] {path}")
    return path
