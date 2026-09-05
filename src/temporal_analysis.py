"""
TEMPORAL FEATURES + MODEL -- ONE SCRIPT, FULLY AUTOMATIC
========================================================================
Tests whether layer-sequence context improves severity prediction beyond
treating each layer independently, as the critique suggested.

Adds temporal features (built from data you already have, no new
download needed):
  - Lag features: TAM/SCR from the previous 1-3 layers
  - Rolling features: rolling mean/std of TAM/SCR over a 5-layer window
  - Delta features: layer-to-layer change in TAM/SCR
  - Cumulative thermal exposure: running sum of TAM up to the current layer

Then compares: independent-layer model (baseline, what you already have)
vs. temporal-feature-augmented model, same leave-one-build-out splits,
same metrics (accuracy, MACE) -- answers "does temporal context help"
with a number, not an assumption.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import mord
from sklearn.metrics import accuracy_score

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

BASE = '/content/drive/MyDrive/DC-CPT-Project'
PATHS = {
    'data_processed': f'{BASE}/Data/processed',
    'results_tables': f'{BASE}/results/tables',
    'results_figures': f'{BASE}/results/figures',
}
for p in PATHS.values():
    os.makedirs(p, exist_ok=True)


def save_table(df, name):
    if not name.endswith('.csv'):
        name += '.csv'
    path = os.path.join(PATHS['results_tables'], name)
    df.to_csv(path, index=False)
    print(f"  [{'SAVED' if os.path.isfile(path) else 'FAILED'}] {path}  ({os.path.getsize(path)/1024:.1f} KB)")
    return path


def save_figure(fig, name):
    if not name.endswith('.png'):
        name += '.png'
    path = os.path.join(PATHS['results_figures'], name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [{'SAVED' if os.path.isfile(path) else 'FAILED'}] {path}  ({os.path.getsize(path)/1024:.1f} KB)")
    return path


DATA_PATH = f"{PATHS['data_processed']}/gate1_labeled_dataset.csv"
STATE_NAMES = ['Stable', 'Degrading', 'Recoverable', 'Critical', 'Irrecoverable']
BASE_FEATURES = [
    'tam_mean', 'tam_max', 'tam_p90',
    'scr_mean', 'scr_std', 'scr_min', 'scr_max',
    'scan_speed', 'hatch_spacing',
]

df = pd.read_csv(DATA_PATH)

# ============================================================================
# BUILD TEMPORAL FEATURES -- must be done PER BUILD, sorted by layer, so
# lag/rolling windows never leak across build boundaries
# ============================================================================
print(f"\n{'='*70}\nBuilding temporal features (per-build, layer-ordered)\n{'='*70}")


def add_temporal_features(build_df):
    build_df = build_df.sort_values('layer').copy()

    for lag in [1, 2, 3]:
        build_df[f'tam_p90_lag{lag}'] = build_df['tam_p90'].shift(lag)
        build_df[f'scr_std_lag{lag}'] = build_df['scr_std'].shift(lag)

    build_df['tam_p90_roll5_mean'] = build_df['tam_p90'].rolling(5, min_periods=1).mean()
    build_df['tam_p90_roll5_std'] = build_df['tam_p90'].rolling(5, min_periods=1).std()
    build_df['scr_std_roll5_mean'] = build_df['scr_std'].rolling(5, min_periods=1).mean()

    build_df['tam_p90_delta'] = build_df['tam_p90'].diff()
    build_df['scr_std_delta'] = build_df['scr_std'].diff()

    build_df['cumulative_tam_exposure'] = build_df['tam_mean'].cumsum()

    return build_df


df_temporal = pd.concat([add_temporal_features(df[df['build'] == b]) for b in ['B6', 'B7', 'B8']], ignore_index=True)

TEMPORAL_FEATURES = [
    'tam_p90_lag1', 'tam_p90_lag2', 'tam_p90_lag3',
    'scr_std_lag1', 'scr_std_lag2', 'scr_std_lag3',
    'tam_p90_roll5_mean', 'tam_p90_roll5_std', 'scr_std_roll5_mean',
    'tam_p90_delta', 'scr_std_delta', 'cumulative_tam_exposure',
]
COMBINED_FEATURES = BASE_FEATURES + TEMPORAL_FEATURES

# early layers in each build won't have full lag history -- drop rows with
# any NaN in the temporal features rather than silently imputing, so we're
# only comparing on layers where BOTH models have complete information
df_temporal_clean = df_temporal.dropna(subset=COMBINED_FEATURES + BASE_FEATURES)
print(f"  {len(df) - len(df_temporal_clean)} early-layer rows dropped (insufficient lag history)")
print(f"  {len(df_temporal_clean)} rows remain for fair comparison")

# ============================================================================
# COMPARE: independent-layer model vs temporal-augmented model
# ============================================================================
print(f"\n{'='*70}\nComparing independent-layer vs temporal-augmented model\n{'='*70}")

results = []
for held_out in ['B6', 'B7', 'B8']:
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
    train_df = df_temporal_clean[df_temporal_clean['build'].isin(train_builds)]
    test_df = df_temporal_clean[df_temporal_clean['build'] == held_out]
    y_train, y_test = train_df['severity'].values, test_df['severity'].values

    for label, feat_cols in [('Independent-layer (baseline)', BASE_FEATURES),
                               ('Temporal-augmented', COMBINED_FEATURES)]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feat_cols])
        X_test = scaler.transform(test_df[feat_cols])

        model = mord.LogisticAT(alpha=0.01)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mace = np.mean(np.abs(y_test - y_pred))
        print(f"  {held_out} / {label}: accuracy={acc:.3f}, MACE={mace:.3f}")
        results.append(dict(held_out=held_out, model=label, accuracy=acc, mace=mace, n_features=len(feat_cols)))

results_df = pd.DataFrame(results)
save_table(results_df, 'temporal_vs_independent_comparison')

fig, ax = plt.subplots(figsize=(8, 5))
summary = results_df.groupby('model')['mace'].mean().reindex(['Independent-layer (baseline)', 'Temporal-augmented'])
bars = ax.bar(summary.index, summary.values, color=['#4C72B0', '#55A868'])
ax.set_ylabel('Mean MACE (lower = better)')
ax.set_title('Does Temporal Context Improve Severity Prediction?')
for bar, val in zip(bars, summary.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', ha='center')
save_figure(fig, 'temporal_vs_independent_chart')

improvement = summary['Independent-layer (baseline)'] - summary['Temporal-augmented']
pct_improvement = 100 * improvement / summary['Independent-layer (baseline)']
print(f"\n  MACE change from adding temporal features: {improvement:+.3f} ({pct_improvement:+.1f}%)")
if improvement > 0:
    print("  Temporal context IMPROVED prediction -- report this as a positive finding.")
else:
    print("  Temporal context did NOT improve prediction on this dataset -- report honestly;")
    print("  this itself is informative (severity may be dominated by instantaneous")
    print("  thermal state rather than trajectory, at least at this per-layer granularity).")

print(f"\n{'='*70}")
print("TEMPORAL ANALYSIS COMPLETE.")
print(f"{'='*70}")
