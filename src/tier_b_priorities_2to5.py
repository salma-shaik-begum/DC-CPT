"""
TIER B, PRIORITIES 2-5 -- ONE SCRIPT, FULLY AUTOMATIC
========================================================================
Priority 2: Calibration -- reliability diagram, ECE, Brier score
Priority 3: SHAP dependence plots -- directionality for TAM-p90, SCR-std, SCR-mean
Priority 4: SHAP cross-build stability -- B6 vs B7 vs B8 importance ranking
Priority 5: Thermal-only vs process-only vs combined feature ablation

Priorities 1 (CatBoost fix), 6 (XCT), and 7 (temporal) are intentionally
NOT in this script -- 1 is already fixed and confirmed, 6 remains the
capped stretch goal, 7 is correctly deferred until after this set.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mord
import shap

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
FEATURE_COLS = [
    'tam_mean', 'tam_max', 'tam_p90',
    'scr_mean', 'scr_std', 'scr_min', 'scr_max',
    'scan_speed', 'hatch_spacing',
]
THERMAL_ONLY_COLS = ['tam_mean', 'tam_max', 'tam_p90', 'scr_mean', 'scr_std', 'scr_min', 'scr_max']
PROCESS_ONLY_COLS = ['scan_speed', 'hatch_spacing']

df = pd.read_csv(DATA_PATH)

# ============================================================================
# PRIORITY 2: CALIBRATION -- reliability diagram, ECE, Brier score
# ============================================================================
print(f"\n{'='*70}\nPRIORITY 2: Calibration (Reliability Diagram, ECE, Brier Score)\n{'='*70}")


def compute_calibration(y_true, y_proba, n_bins=10):
    """Top-label calibration: for each sample, take the predicted class's
    own probability (confidence), bin by confidence, compare bin-mean
    confidence to bin-mean accuracy. ECE = weighted average gap between
    confidence and accuracy across bins -- the standard multiclass
    calibration metric when a single reliability curve is needed."""
    y_pred = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    correct = (y_pred == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc, bin_count = [], [], []
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() > 0:
            bin_conf.append(confidences[mask].mean())
            bin_acc.append(correct[mask].mean())
            bin_count.append(mask.sum())
            ece += (mask.sum() / len(y_true)) * abs(confidences[mask].mean() - correct[mask].mean())
        else:
            bin_conf.append(np.nan)
            bin_acc.append(np.nan)
            bin_count.append(0)

    # multiclass Brier score: mean squared error between one-hot true and
    # full predicted probability vector, averaged over classes and samples
    n_classes = y_proba.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    brier = np.mean(np.sum((y_proba - one_hot) ** 2, axis=1))

    return dict(bin_conf=bin_conf, bin_acc=bin_acc, bin_count=bin_count, ece=ece, brier=brier)


calibration_rows = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, held_out in zip(axes, ['B6', 'B7', 'B8']):
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
    train_df = df[df['build'].isin(train_builds)]
    test_df = df[df['build'] == held_out]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])
    y_train, y_test = train_df['severity'].values, test_df['severity'].values

    model = mord.LogisticAT(alpha=0.01)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)

    cal = compute_calibration(y_test, y_proba)
    print(f"  {held_out}: ECE={cal['ece']:.4f}, Brier={cal['brier']:.4f}")
    calibration_rows.append(dict(held_out=held_out, ece=cal['ece'], brier_score=cal['brier']))

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    valid = ~np.isnan(cal['bin_conf'])
    ax.plot(np.array(cal['bin_conf'])[valid], np.array(cal['bin_acc'])[valid], 'o-', label='Ordinal model')
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title(f'{held_out}  (ECE={cal["ece"]:.3f})')
    ax.legend(fontsize=8)

fig.suptitle('Reliability Diagrams -- Ordinal Model, Leave-One-Build-Out')
save_figure(fig, 'calibration_reliability_diagrams')
save_table(pd.DataFrame(calibration_rows), 'calibration_ece_brier')

# ============================================================================
# PRIORITY 3: SHAP DEPENDENCE PLOTS -- directionality
# ============================================================================
print(f"\n{'='*70}\nPRIORITY 3: SHAP Dependence Plots (TAM-p90, SCR-std, SCR-mean)\n{'='*70}")

# use B8 held out as the representative fold, consistent with prior SHAP run
train_builds = ['B6', 'B7']
train_df = df[df['build'].isin(train_builds)]
test_df = df[df['build'] == 'B8']
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[FEATURE_COLS])
X_test = scaler.transform(test_df[FEATURE_COLS])
y_train = train_df['severity'].values

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# average SHAP contribution across classes for a single "overall severity
# direction" signal per sample (sum of class-weighted SHAP, using predicted
# class as the reference is the simplest interpretable choice here)
if isinstance(shap_values, list):
    shap_avg = np.mean([sv for sv in shap_values], axis=0)  # shape (n, n_features)
else:
    shap_avg = shap_values.mean(axis=2) if shap_values.ndim == 3 else shap_values

dependence_features = ['tam_p90', 'scr_std', 'scr_mean']
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, feat in zip(axes, dependence_features):
    idx = FEATURE_COLS.index(feat)
    raw_values = test_df[feat].values
    ax.scatter(raw_values, shap_avg[:, idx], alpha=0.5, s=15)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xlabel(feat)
    ax.set_ylabel('SHAP value (mean over classes)')
    ax.set_title(f'SHAP Dependence: {feat}')
fig.suptitle('SHAP Dependence Plots -- Directionality Check (held out: B8)')
save_figure(fig, 'shap_dependence_plots')

print("  Interpretation guide: a rising scatter (SHAP value increases with")
print("  feature value) confirms the expected physical direction -- higher")
print("  thermal exposure / cooling instability should push toward higher")
print("  predicted severity, not lower.")

# ============================================================================
# PRIORITY 4: SHAP CROSS-BUILD STABILITY -- B6 vs B7 vs B8
# ============================================================================
print(f"\n{'='*70}\nPRIORITY 4: SHAP Cross-Build Stability\n{'='*70}")

shap_stability_rows = []
fig, ax = plt.subplots(figsize=(9, 5.5))
width = 0.25
x = np.arange(len(FEATURE_COLS))

for i, held_out in enumerate(['B6', 'B7', 'B8']):
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
    train_df = df[df['build'].isin(train_builds)]
    test_df = df[df['build'] == held_out]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])
    y_train = train_df['severity'].values

    rf_fold = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_fold.fit(X_train, y_train)
    explainer_fold = shap.TreeExplainer(rf_fold)
    shap_values_fold = explainer_fold.shap_values(X_test)

    if isinstance(shap_values_fold, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values_fold], axis=0)
    else:
        mean_abs_shap = (np.abs(shap_values_fold).mean(axis=(0, 2)) if shap_values_fold.ndim == 3
                          else np.abs(shap_values_fold).mean(axis=0))

    ax.bar(x + (i - 1) * width, mean_abs_shap, width, label=f'Held out: {held_out}')
    for feat, val in zip(FEATURE_COLS, mean_abs_shap):
        shap_stability_rows.append(dict(held_out=held_out, feature=feat, mean_abs_shap=val))

ax.set_xticks(x)
ax.set_xticklabels(FEATURE_COLS, rotation=45, ha='right')
ax.set_ylabel('Mean |SHAP value|')
ax.set_title('SHAP Feature Importance Stability Across Held-Out Builds')
ax.legend()
save_figure(fig, 'shap_cross_build_stability')

shap_stability_df = pd.DataFrame(shap_stability_rows)
save_table(shap_stability_df, 'shap_cross_build_stability')

# check rank stability: is the top feature the same across all 3 folds?
top_features = shap_stability_df.loc[shap_stability_df.groupby('held_out')['mean_abs_shap'].idxmax()]
print("  Top SHAP feature per held-out build:")
print(top_features[['held_out', 'feature', 'mean_abs_shap']].to_string(index=False))

# ============================================================================
# PRIORITY 5: THERMAL-ONLY vs PROCESS-ONLY vs COMBINED
# ============================================================================
print(f"\n{'='*70}\nPRIORITY 5: Thermal-Only vs Process-Only vs Combined Features\n{'='*70}")

FEATURE_SETS = {
    'Thermal-only (TAM/SCR)': THERMAL_ONLY_COLS,
    'Process-only (speed/hatch)': PROCESS_ONLY_COLS,
    'Combined (all features)': FEATURE_COLS,
}

feature_set_results = []
for held_out in ['B6', 'B7', 'B8']:
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
    train_df = df[df['build'].isin(train_builds)]
    test_df = df[df['build'] == held_out]
    y_train, y_test = train_df['severity'].values, test_df['severity'].values

    for set_name, cols in FEATURE_SETS.items():
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[cols])
        X_test = scaler.transform(test_df[cols])

        model = mord.LogisticAT(alpha=0.01)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mace = np.mean(np.abs(y_test - y_pred))
        print(f"  {held_out} / {set_name}: accuracy={acc:.3f}, MACE={mace:.3f}")
        feature_set_results.append(dict(held_out=held_out, feature_set=set_name, accuracy=acc, mace=mace))

feature_set_df = pd.DataFrame(feature_set_results)
save_table(feature_set_df, 'feature_set_ablation')

fig, ax = plt.subplots(figsize=(8, 5))
summary = feature_set_df.groupby('feature_set')['mace'].mean().reindex(FEATURE_SETS.keys())
ax.bar(summary.index, summary.values, color=['#4C72B0', '#DD8452', '#55A868'])
ax.set_ylabel('Mean MACE (lower = better)')
ax.set_title('Feature Group Ablation: Thermal vs Process vs Combined')
plt.xticks(rotation=15, ha='right')
save_figure(fig, 'feature_set_ablation_chart')

print(f"\n{'='*70}")
print("PRIORITIES 2-5 COMPLETE.")
print(f"{'='*70}")
