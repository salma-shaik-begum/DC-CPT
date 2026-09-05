"""
TIER B ANALYSIS -- ONE SCRIPT, FULLY AUTOMATIC
========================================================================
Adds four things reviewers will expect that Tier A didn't cover:
  1. Risk-coverage curve -- answers "is the system just refusing to act
     on everything?" with a curve, not a single 12.2% number.
  2. Modern tabular baselines (LightGBM, CatBoost, HistGradientBoosting,
     ExtraTrees) added to the model comparison.
  3. Wilcoxon signed-rank test -- paired statistical significance across
     the 3 held-out builds, not just eyeballing whether numbers differ.
  4. SHAP feature importance -- run on Random Forest (the best raw
     predictor), giving physical interpretation: does higher thermal
     exposure / cooling instability actually drive the severity score
     the way physics would predict?

Same automatic pattern as run_full_pipeline.py: mounts Drive, saves
every table and figure with a confirmed [SAVED] line, no manual steps
beyond the one pip install and one paste.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import wilcoxon
from mapie.classification import SplitConformalClassifier
import mord

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("lightgbm not installed -- run: pip install lightgbm --break-system-packages")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("catboost not installed -- run: pip install catboost --break-system-packages")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("shap not installed -- run: pip install shap --break-system-packages")

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
    confirmed = os.path.isfile(path)
    print(f"  [{'SAVED' if confirmed else 'FAILED'}] {path}  ({os.path.getsize(path)/1024:.1f} KB)")
    return path


def save_figure(fig, name):
    if not name.endswith('.png'):
        name += '.png'
    path = os.path.join(PATHS['results_figures'], name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    confirmed = os.path.isfile(path)
    print(f"  [{'SAVED' if confirmed else 'FAILED'}] {path}  ({os.path.getsize(path)/1024:.1f} KB)")
    return path


DATA_PATH = f"{PATHS['data_processed']}/gate1_labeled_dataset.csv"
STATE_NAMES = ['Stable', 'Degrading', 'Recoverable', 'Critical', 'Irrecoverable']
FEATURE_COLS = [
    'tam_mean', 'tam_max', 'tam_p90',
    'scr_mean', 'scr_std', 'scr_min', 'scr_max',
    'scan_speed', 'hatch_spacing',
]

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found -- run the main pipeline first.")

df = pd.read_csv(DATA_PATH)

# ============================================================================
# PART 1: RISK-COVERAGE CURVE
# ============================================================================
print(f"\n{'='*70}\nPART 1: Risk-Coverage Curve\n{'='*70}")

CONFIDENCE_LEVELS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]


def get_risk_coverage_point(df, held_out_build, confidence_level, calib_frac=0.3, seed=42):
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out_build]
    train_pool = df[df['build'].isin(train_builds)].sample(frac=1, random_state=seed)
    n_calib = int(len(train_pool) * calib_frac)
    calib_df = train_pool.iloc[:n_calib]
    fit_df = train_pool.iloc[n_calib:]
    test_df = df[df['build'] == held_out_build]

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(fit_df[FEATURE_COLS])
    X_calib = scaler.transform(calib_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])

    base_model = mord.LogisticAT(alpha=0.01)
    base_model.fit(X_fit, fit_df['severity'].values)

    mapie_model = SplitConformalClassifier(
        estimator=base_model, confidence_level=confidence_level, prefit=True, conformity_score='aps',
    )
    mapie_model.conformalize(X_calib, calib_df['severity'].values)
    y_pred, y_pred_sets = mapie_model.predict_set(X_test)

    set_masks = y_pred_sets[:, :, 0]
    set_sizes = set_masks.sum(axis=1)
    y_true = test_df['severity'].values
    is_singleton = (set_sizes == 1)

    autonomy_rate = is_singleton.mean()
    if is_singleton.sum() > 0:
        singleton_acc = (y_pred[is_singleton] == y_true[is_singleton]).mean()
        error_rate = 1 - singleton_acc
    else:
        error_rate = np.nan

    return autonomy_rate, error_rate


risk_coverage_rows = []
for conf_level in CONFIDENCE_LEVELS:
    autonomy_vals, error_vals = [], []
    for held_out in ['B6', 'B7', 'B8']:
        ar, er = get_risk_coverage_point(df, held_out, conf_level)
        autonomy_vals.append(ar)
        error_vals.append(er)
    mean_autonomy = np.nanmean(autonomy_vals)
    mean_error = np.nanmean(error_vals)
    print(f"  target_confidence={conf_level:.2f}  ->  mean_autonomy={mean_autonomy:.3f}, mean_error_when_autonomous={mean_error:.3f}")
    risk_coverage_rows.append(dict(target_confidence=conf_level, mean_autonomy_rate=mean_autonomy, mean_error_when_autonomous=mean_error))

risk_coverage_df = pd.DataFrame(risk_coverage_rows)
save_table(risk_coverage_df, 'risk_coverage_curve_data')

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(risk_coverage_df['mean_error_when_autonomous'], risk_coverage_df['mean_autonomy_rate'],
        marker='o', linewidth=2)
for _, row in risk_coverage_df.iterrows():
    ax.annotate(f"{row['target_confidence']:.2f}",
                (row['mean_error_when_autonomous'], row['mean_autonomy_rate']),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel('Error Rate When Autonomous')
ax.set_ylabel('Autonomy Rate')
ax.set_title('Risk-Coverage Curve: Autonomy vs. Error as Confidence Threshold Varies\n(labels show target confidence level)')
ax.grid(alpha=0.3)
save_figure(fig, 'risk_coverage_curve')

# ============================================================================
# PART 2: MODERN TABULAR BASELINES (LightGBM, CatBoost, HistGB, ExtraTrees)
# ============================================================================
print(f"\n{'='*70}\nPART 2: Modern Tabular Baselines\n{'='*70}")


def get_modern_models():
    models = {
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200, random_state=42),
    }
    if HAS_LGBM:
        models['LightGBM'] = LGBMClassifier(n_estimators=200, max_depth=4, random_state=42, verbosity=-1)
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostClassifier(
            iterations=200, depth=4, random_state=42, verbose=False,
            loss_function='MultiClass',  # explicit -- avoids CatBoost silently
                                          # picking an unexpected objective/label
                                          # encoding that broke MACE last run
        )
    return models


modern_results = []
rf_predictions_for_shap = []  # collect for SHAP later
for held_out in ['B6', 'B7', 'B8']:
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
    train_df = df[df['build'].isin(train_builds)]
    test_df = df[df['build'] == held_out]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])
    y_train, y_test = train_df['severity'].values, test_df['severity'].values

    for name, model in get_modern_models().items():
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test)).flatten().astype(int)  # CatBoost
                                                                 # returns shape (n,1)
                                                                 # float predictions --
                                                                 # flatten + cast to int
                                                                 # to match sklearn's
                                                                 # (n,) integer output,
                                                                 # or MACE silently
                                                                 # computes against a
                                                                 # broadcast (n,n) array
        acc = accuracy_score(y_test, y_pred)
        mace = np.mean(np.abs(y_test - y_pred))
        print(f"  {held_out} / {name}: accuracy={acc:.3f}, MACE={mace:.3f}")
        modern_results.append(dict(held_out=held_out, model=name, accuracy=acc, mace=mace))

    # also refit RF here for the Wilcoxon test and SHAP step below
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_predictions_for_shap.append((held_out, rf, X_test, test_df, scaler))
    modern_results.append(dict(held_out=held_out, model='Random Forest (recomputed)',
                                accuracy=accuracy_score(y_test, y_pred_rf),
                                mace=np.mean(np.abs(y_test - y_pred_rf))))

    ordinal_model = mord.LogisticAT(alpha=0.01)
    ordinal_model.fit(X_train, y_train)
    y_pred_ord = ordinal_model.predict(X_test)
    modern_results.append(dict(held_out=held_out, model='Ordinal Logistic (ours)',
                                accuracy=accuracy_score(y_test, y_pred_ord),
                                mace=np.mean(np.abs(y_test - y_pred_ord))))

modern_df = pd.DataFrame(modern_results)
save_table(modern_df, 'modern_baseline_comparison')

# ============================================================================
# PART 3: WILCOXON SIGNED-RANK TEST -- paired, across the 3 held-out builds
# ============================================================================
print(f"\n{'='*70}\nPART 3: Wilcoxon Signed-Rank Test (Ordinal vs Random Forest, paired by build)\n{'='*70}")

ordinal_mace = modern_df[modern_df['model'] == 'Ordinal Logistic (ours)'].sort_values('held_out')['mace'].values
rf_mace = modern_df[modern_df['model'] == 'Random Forest (recomputed)'].sort_values('held_out')['mace'].values

print(f"  Ordinal MACE per build: {ordinal_mace}")
print(f"  RF MACE per build:      {rf_mace}")

if len(ordinal_mace) == len(rf_mace) and len(ordinal_mace) >= 3:
    try:
        stat, pval = wilcoxon(ordinal_mace, rf_mace)
        print(f"  Wilcoxon signed-rank: statistic={stat:.3f}, p-value={pval:.4f}")
        note = ("NOTE: with only n=3 paired builds, this test has very low power -- "
                 "report the p-value honestly but do not overstate significance from n=3.")
        print(f"  {note}")
        wilcoxon_result = pd.DataFrame([dict(comparison='Ordinal vs RF (MACE, paired by build)',
                                              statistic=stat, p_value=pval, n_pairs=len(ordinal_mace), note=note)])
    except ValueError as e:
        print(f"  Wilcoxon test could not be computed: {e}")
        wilcoxon_result = pd.DataFrame([dict(comparison='Ordinal vs RF (MACE, paired by build)',
                                              statistic=np.nan, p_value=np.nan, n_pairs=len(ordinal_mace),
                                              note=f'Could not compute: {e}')])
else:
    wilcoxon_result = pd.DataFrame([dict(comparison='Ordinal vs RF', statistic=np.nan, p_value=np.nan,
                                          n_pairs=0, note='Insufficient paired data')])

save_table(wilcoxon_result, 'wilcoxon_significance_test')

# ============================================================================
# PART 4: SHAP FEATURE IMPORTANCE (on Random Forest -- best raw predictor)
# ============================================================================
print(f"\n{'='*70}\nPART 4: SHAP Feature Importance\n{'='*70}")

if HAS_SHAP:
    # Use the last fold's fitted RF (B8 held out) as the representative example
    held_out_name, rf_model, X_test_shap, test_df_shap, scaler_shap = rf_predictions_for_shap[-1]
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_shap)

    # shap_values for multiclass RF: list of arrays, one per class -- average
    # absolute SHAP value across classes to get overall feature importance
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 else np.abs(shap_values).mean(axis=0)

    shap_importance_df = pd.DataFrame({
        'feature': FEATURE_COLS, 'mean_abs_shap_value': mean_abs_shap
    }).sort_values('mean_abs_shap_value', ascending=False)
    print(shap_importance_df.to_string(index=False))
    save_table(shap_importance_df, 'shap_feature_importance')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(shap_importance_df['feature'], shap_importance_df['mean_abs_shap_value'])
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'SHAP Feature Importance -- Random Forest (held out: {held_out_name})')
    ax.invert_yaxis()
    save_figure(fig, 'shap_feature_importance')

    print("\nPhysical interpretation check -- does thermal exposure/cooling")
    print("instability rank as expected, ahead of process parameters alone?")
    top_feature = shap_importance_df.iloc[0]['feature']
    print(f"  Top feature: {top_feature}")
else:
    print("  Skipped -- shap not installed. Run: pip install shap --break-system-packages")

print(f"\n{'='*70}")
print("TIER B ANALYSIS COMPLETE.")
print(f"{'='*70}")
