"""
Bootstrap Confidence Intervals
==================================
Wraps the key reported metrics with bootstrap resampling to get 95% CIs,
instead of reporting bare point estimates. Resamples at the LAYER level
(with replacement) from the pooled out-of-fold predictions across all
three held-out builds, recomputing each metric per resample.

Metrics covered:
  - Gate 1 severity model: accuracy, MACE
  - Gate 2 conformal: empirical coverage, autonomy rate, singleton accuracy
  - Full pipeline (ablation condition 1): accuracy_when_autonomous,
    unsafe_actuation_rate, autonomy_rate

Note: resampling the WHOLE pooled dataset each iteration (not just the
autonomous subset) correctly propagates uncertainty in WHICH layers
become autonomous into the CI, not just uncertainty in accuracy given a
fixed subset -- this is the more honest bootstrap for a conditional metric.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mapie.classification import SplitConformalClassifier
import mord
import sys
sys.path.insert(0, '/content/drive/MyDrive/DC-CPT-Project/src')
from save_utils import save_table

DATA_PATH = '/content/drive/MyDrive/DC-CPT-Project/Data/processed/gate3_labeled_dataset.csv'
FEATURE_COLS = [
    'tam_mean', 'tam_max', 'tam_p90',
    'scr_mean', 'scr_std', 'scr_min', 'scr_max',
    'scan_speed', 'hatch_spacing',
]
TARGET_CONFIDENCE = 0.90
N_BOOTSTRAP = 2000
SEED = 42


def get_predictions_for_build(df, held_out_build, calib_frac=0.3, seed=42):
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out_build]
    train_pool = df[df['build'].isin(train_builds)].sample(frac=1, random_state=seed)

    n_calib = int(len(train_pool) * calib_frac)
    calib_df = train_pool.iloc[:n_calib]
    fit_df = train_pool.iloc[n_calib:]
    test_df = df[df['build'] == held_out_build].copy()

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(fit_df[FEATURE_COLS])
    X_calib = scaler.transform(calib_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])

    base_model = mord.LogisticAT(alpha=0.01)
    base_model.fit(X_fit, fit_df['severity'].values)

    mapie_model = SplitConformalClassifier(
        estimator=base_model, confidence_level=TARGET_CONFIDENCE, prefit=True, conformity_score='aps',
    )
    mapie_model.conformalize(X_calib, calib_df['severity'].values)
    y_pred, y_pred_sets = mapie_model.predict_set(X_test)

    set_masks = y_pred_sets[:, :, 0]
    set_sizes = set_masks.sum(axis=1)
    in_set = np.array([set_masks[i, test_df['severity'].values[i]] for i in range(len(test_df))])

    test_df['predicted_severity'] = y_pred
    test_df['is_confident'] = (set_sizes == 1)
    test_df['true_in_set'] = in_set
    return test_df


def bootstrap_ci(values_fn, df, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """values_fn(sampled_df) -> dict of metric_name -> value.
    Returns dict of metric_name -> (point_estimate, ci_low, ci_high)."""
    rng = np.random.RandomState(seed)
    n = len(df)

    point = values_fn(df)
    boot_results = {k: [] for k in point.keys()}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sample = df.iloc[idx]
        vals = values_fn(sample)
        for k, v in vals.items():
            boot_results[k].append(v)

    summary = {}
    for k, v in point.items():
        arr = np.array([x for x in boot_results[k] if not np.isnan(x)])
        if len(arr) == 0:
            summary[k] = (v, np.nan, np.nan)
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            summary[k] = (v, lo, hi)
    return summary


def compute_metrics(df):
    """All metrics of interest, computed on a (possibly resampled) df."""
    acc = (df['predicted_severity'] == df['severity']).mean()
    mace = np.mean(np.abs(df['predicted_severity'] - df['severity']))
    coverage = df['true_in_set'].mean()

    is_conf = df['is_confident']
    autonomy_rate = is_conf.mean()
    if is_conf.sum() > 0:
        singleton_acc = (df.loc[is_conf, 'predicted_severity'] == df.loc[is_conf, 'severity']).mean()
    else:
        singleton_acc = np.nan

    would_act = df['is_confident'] & df['physics_admissible']
    n_act = would_act.sum()
    if n_act > 0:
        acted = df[would_act]
        acc_when_auto = (acted['predicted_severity'] == acted['severity']).mean()
        unsafe = ((acted['predicted_severity'] != acted['severity']) & (acted['severity'] >= 3)).mean()
    else:
        acc_when_auto, unsafe = np.nan, np.nan
    full_pipeline_autonomy_rate = would_act.mean()

    return dict(
        accuracy=acc, mace=mace, coverage=coverage,
        autonomy_rate=autonomy_rate, singleton_accuracy=singleton_acc,
        full_pipeline_accuracy_when_autonomous=acc_when_auto,
        full_pipeline_unsafe_actuation_rate=unsafe,
        full_pipeline_autonomy_rate=full_pipeline_autonomy_rate,
    )


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)

    all_predictions = []
    for held_out in ['B6', 'B7', 'B8']:
        pred_df = get_predictions_for_build(df, held_out)
        all_predictions.append(pred_df)
    full_df = pd.concat(all_predictions, ignore_index=True)

    print(f"Running {N_BOOTSTRAP} bootstrap resamples over {len(full_df)} pooled layers...\n")
    ci_results = bootstrap_ci(compute_metrics, full_df)

    print(f"{'Metric':<45} {'Point':>8} {'95% CI Low':>12} {'95% CI High':>12}")
    print("-" * 80)
    ci_rows = []
    for metric, (point, lo, hi) in ci_results.items():
        print(f"{metric:<45} {point:>8.3f} {lo:>12.3f} {hi:>12.3f}")
        ci_rows.append(dict(metric=metric, point_estimate=point, ci_low=lo, ci_high=hi))

    save_table(pd.DataFrame(ci_rows), 'bootstrap_confidence_intervals')
