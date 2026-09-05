"""
Ablation Study: Gate 2 and Gate 3 Contribution
==================================================
Four conditions, same underlying severity predictions, same held-out
builds -- only the AUTHORIZATION LOGIC changes:

  1. Full pipeline      : act autonomously only if confident AND physics-admissible
  2. No Gate 2 (no conf.): act autonomously if physics-admissible, ignore confidence
  3. No Gate 3 (no phys.): act autonomously if confident, ignore physics
  4. No gating (raw)     : act autonomously on every single prediction (baseline --
                            what a plain classifier deployed without governance does)

Key metric: ACCURACY WHEN AUTONOMOUS. This directly quantifies each
gate's safety contribution -- removing a gate should increase the
autonomous action rate (more layers acted on) while decreasing accuracy
when autonomous (more of those actions are wrong). That trade-off IS
the ablation result.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mapie.classification import SplitConformalClassifier
import mord
import sys
sys.path.insert(0, '/content/drive/MyDrive/DC-CPT-Project/src')
from save_utils import save_table  # writes to results/tables/, confirms on disk

DATA_PATH = '/content/drive/MyDrive/DC-CPT-Project/Data/processed/gate3_labeled_dataset.csv'
FEATURE_COLS = [
    'tam_mean', 'tam_max', 'tam_p90',
    'scr_mean', 'scr_std', 'scr_min', 'scr_max',
    'scan_speed', 'hatch_spacing',
]
TARGET_CONFIDENCE = 0.90


def get_predictions_for_build(df, held_out_build, calib_frac=0.3, seed=42):
    """Fit + conformalize + predict for one held-out build. Returns the
    full held-out build's data with predicted severity, confidence flag,
    and (already-present) physics_admissible column attached."""
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

    base_model = mord.LogisticAT(alpha=0.01)  # tuned value from baseline comparison
    base_model.fit(X_fit, fit_df['severity'].values)

    mapie_model = SplitConformalClassifier(
        estimator=base_model, confidence_level=TARGET_CONFIDENCE, prefit=True, conformity_score='aps',
    )
    mapie_model.conformalize(X_calib, calib_df['severity'].values)
    y_pred, y_pred_sets = mapie_model.predict_set(X_test)

    set_masks = y_pred_sets[:, :, 0]
    set_sizes = set_masks.sum(axis=1)

    test_df['predicted_severity'] = y_pred
    test_df['is_confident'] = (set_sizes == 1)
    # physics_admissible already present in the loaded CSV from Gate 3
    return test_df


def evaluate_condition(df, condition_name, use_confidence, use_physics):
    df = df.copy()

    if use_confidence and use_physics:
        would_act = df['is_confident'] & df['physics_admissible']
    elif use_confidence and not use_physics:
        would_act = df['is_confident']
    elif not use_confidence and use_physics:
        would_act = df['physics_admissible']
    else:
        would_act = pd.Series(True, index=df.index)

    n_total = len(df)
    n_act = would_act.sum()
    autonomy_rate = n_act / n_total

    if n_act > 0:
        acted = df[would_act]
        correct = (acted['predicted_severity'] == acted['severity']).sum()
        accuracy_when_autonomous = correct / n_act
        # unsafe actuation: acted AND wrong AND the true state was actually
        # severe (Critical=3 or Irrecoverable=4) -- the costly error type
        unsafe = ((acted['predicted_severity'] != acted['severity']) & (acted['severity'] >= 3)).sum()
        unsafe_rate = unsafe / n_act
    else:
        accuracy_when_autonomous = np.nan
        unsafe_rate = np.nan

    return dict(
        condition=condition_name,
        autonomy_rate=autonomy_rate,
        accuracy_when_autonomous=accuracy_when_autonomous,
        unsafe_actuation_rate=unsafe_rate,
        n_autonomous=n_act,
        n_total=n_total,
    )


CONDITIONS = [
    ('1. Full pipeline (Gate2 + Gate3)', True, True),
    ('2. No Gate 2 (confidence removed)', False, True),
    ('3. No Gate 3 (physics removed)', True, False),
    ('4. No gating (raw predictions, baseline)', False, False),
]


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)

    all_predictions = []
    for held_out in ['B6', 'B7', 'B8']:
        pred_df = get_predictions_for_build(df, held_out)
        all_predictions.append(pred_df)
    full_df = pd.concat(all_predictions, ignore_index=True)

    print("Ablation results (pooled across all 3 held-out builds, 936 total layers):\n")
    results = []
    for name, use_conf, use_phys in CONDITIONS:
        res = evaluate_condition(full_df, name, use_conf, use_phys)
        results.append(res)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    save_table(results_df, 'ablation_results_pooled')

    print("\nPer-build breakdown:")
    per_build_all = []
    for held_out in ['B6', 'B7', 'B8']:
        build_df = full_df[full_df['build'] == held_out]
        print(f"\n--- {held_out} ---")
        build_results = [evaluate_condition(build_df, name, uc, up) for name, uc, up in CONDITIONS]
        build_results_df = pd.DataFrame(build_results)
        build_results_df['held_out_build'] = held_out
        print(build_results_df.to_string(index=False))
        per_build_all.append(build_results_df)

    per_build_combined = pd.concat(per_build_all, ignore_index=True)
    save_table(per_build_combined, 'ablation_results_per_build')
