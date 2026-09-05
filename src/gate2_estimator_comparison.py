"""
Gate 2 Base Estimator Comparison: Ordinal Logistic vs Random Forest
=======================================================================
Random Forest beat Ordinal Logistic on raw accuracy/MACE (see baseline
comparison). This tests whether that advantage carries through to the
part that actually matters for the governance framework: does it
produce BETTER CALIBRATED, ORDINALLY SENSIBLE confidence sets when
wrapped in conformal prediction?

Three things compared, for each base estimator:
  1. Coverage / autonomy rate / singleton accuracy (same as original Gate 2)
  2. Set adjacency (%) among escalated cases -- THIS is the key test.
     RF's predict_proba doesn't know severity states are ORDERED. It
     could easily produce a confidence set like {Stable, Critical} for
     an ambiguous case, since nothing in RF's training enforces that
     adjacent classes should be favored over distant ones when uncertain.
     Ordinal logistic structurally can't do this by construction.
  3. Whether losing adjacency (if it happens) is a real practical cost --
     i.e. does RF's better point-accuracy outweigh worse-behaved
     uncertainty, or not.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from mapie.classification import SplitConformalClassifier
import mord

DATA_PATH = '/content/drive/MyDrive/DC-CPT-Project/Data/processed/gate1_labeled_dataset.csv'
STATE_NAMES = ['Stable', 'Degrading', 'Recoverable', 'Critical', 'Irrecoverable']
FEATURE_COLS = [
    'tam_mean', 'tam_max', 'tam_p90',
    'scr_mean', 'scr_std', 'scr_min', 'scr_max',
    'scan_speed', 'hatch_spacing',
]
TARGET_CONFIDENCE = 0.90


def get_base_model(name):
    if name == 'Ordinal Logistic':
        return mord.LogisticAT(alpha=0.01)  # tuned value from baseline comparison
    elif name == 'Random Forest':
        return RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        raise ValueError(name)


def run_fold(df, held_out_build, model_name, calib_frac=0.3, seed=42):
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

    base_model = get_base_model(model_name)
    base_model.fit(X_fit, fit_df['severity'].values)

    mapie_model = SplitConformalClassifier(
        estimator=base_model, confidence_level=TARGET_CONFIDENCE, prefit=True, conformity_score='aps',
    )
    mapie_model.conformalize(X_calib, calib_df['severity'].values)
    y_pred, y_pred_sets = mapie_model.predict_set(X_test)

    set_masks = y_pred_sets[:, :, 0]
    set_sizes = set_masks.sum(axis=1)
    y_test = test_df['severity'].values

    in_set = np.array([set_masks[i, y_test[i]] for i in range(len(y_test))])
    empirical_coverage = in_set.mean()

    is_singleton = (set_sizes == 1)
    autonomy_rate = is_singleton.mean()
    singleton_acc = (y_pred[is_singleton] == y_test[is_singleton]).mean() if is_singleton.sum() > 0 else np.nan

    # adjacency check among escalated (non-singleton) cases
    is_multi = (set_sizes > 1)
    adjacent_count, nonadjacent_count = 0, 0
    for i in np.where(is_multi)[0]:
        states_in_set = sorted(np.where(set_masks[i])[0])
        span = states_in_set[-1] - states_in_set[0]
        if span == len(states_in_set) - 1:
            adjacent_count += 1
        else:
            nonadjacent_count += 1
    total_multi = adjacent_count + nonadjacent_count
    adjacency_pct = adjacent_count / total_multi if total_multi > 0 else np.nan

    return dict(
        held_out=held_out_build, model=model_name,
        empirical_coverage=empirical_coverage,
        autonomy_rate=autonomy_rate,
        singleton_accuracy=singleton_acc,
        mean_set_size=set_sizes.mean(),
        adjacency_pct=adjacency_pct,
        n_escalated=total_multi,
    )


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)

    results = []
    for held_out in ['B6', 'B7', 'B8']:
        for model_name in ['Ordinal Logistic', 'Random Forest']:
            results.append(run_fold(df, held_out, model_name))

    results_df = pd.DataFrame(results)
    print("Per-fold comparison:")
    print(results_df.to_string(index=False))

    print(f"\n{'='*70}")
    print("SUMMARY: mean across the 3 held-out builds, per base estimator")
    print(f"{'='*70}")
    summary = results_df.groupby('model').agg(
        coverage=('empirical_coverage', 'mean'),
        autonomy_rate=('autonomy_rate', 'mean'),
        singleton_acc=('singleton_accuracy', 'mean'),
        mean_set_size=('mean_set_size', 'mean'),
        adjacency_pct=('adjacency_pct', 'mean'),
    )
    print(summary.round(3))
