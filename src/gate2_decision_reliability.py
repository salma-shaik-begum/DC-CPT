"""
DC-CPT Gate 2: Statistical Decision Reliability (Conformal Confidence Gate)
=============================================================================
Wraps the Gate 1 ordinal severity model with conformal prediction (MAPIE).

Instead of one predicted state per layer, this produces a CONFIDENCE SET:
  - Singleton set (e.g. just {Critical})       -> model is statistically
                                                    unambiguous -> eligible
                                                    for autonomous action
  - Multi-state set (e.g. {Critical, Irrecov.}) -> model is genuinely
                                                    unsure -> escalate /
                                                    default to conservative
                                                    action, per your MDGL
                                                    Gate 4 policy design

This is the mechanism that turns "we have a classifier" into "we have a
governance layer that knows when not to trust itself" -- the actual novel
claim of the paper.

Install once: pip install mapie mord --break-system-packages
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mapie.classification import SplitConformalClassifier
import mord

DATA_PATH = '/content/drive/MyDrive/DC-CPT-Project/Data/processed/gate1_labeled_dataset.csv'

STATE_NAMES = ['Stable', 'Degrading', 'Recoverable', 'Critical', 'Irrecoverable']
FEATURE_COLS = [
    'tam_mean', 'tam_max', 'tam_p90',
    'scr_mean', 'scr_std', 'scr_min', 'scr_max',
    'scan_speed', 'hatch_spacing',
]

ALPHA_LEVELS = [0.05, 0.10, 0.20]  # target miscoverage rates -> 95%, 90%, 80% coverage


def split_calib_test(df, held_out_build, calib_frac=0.5, seed=42):
    """Split the held-out build's layers into a calibration set (used to
    fit the conformal wrapper) and a true test set (used to evaluate it).
    Splitting BY LAYER within the build, not across builds -- this build
    was never seen during Gate 1 training, so both halves are honestly
    out-of-sample for the underlying classifier."""
    held_df = df[df['build'] == held_out_build].sample(frac=1, random_state=seed)
    n_calib = int(len(held_df) * calib_frac)
    return held_df.iloc[:n_calib], held_df.iloc[n_calib:]


def run_gate2(df, train_builds, held_out_build):
    train_df = df[df['build'].isin(train_builds)]
    calib_df, test_df = split_calib_test(df, held_out_build)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    X_calib = scaler.transform(calib_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])

    y_train = train_df['severity'].values
    y_calib = calib_df['severity'].values
    y_test = test_df['severity'].values

    # Gate 1 model, same as before -- fit once on the two training builds
    base_model = mord.LogisticAT(alpha=1.0)
    base_model.fit(X_train, y_train)

    print(f"\n{'='*70}")
    print(f"Held out build: {held_out_build}  (calib n={len(calib_df)}, test n={len(test_df)})")
    print(f"{'='*70}")

    results = {}
    for alpha in ALPHA_LEVELS:
        target_coverage = 1 - alpha

        # MAPIE v1 API: SplitConformalClassifier + conformalize() + predict_set()
        # (older tutorials online still show MapieClassifier -- that class was
        # renamed/restructured in a recent MAPIE major release)
        mapie_model = SplitConformalClassifier(
            estimator=base_model,
            confidence_level=target_coverage,
            prefit=True,
            conformity_score='aps',
        )
        mapie_model.conformalize(X_calib, y_calib)
        y_pred, y_pred_sets = mapie_model.predict_set(X_test)
        # y_pred_sets shape: (n_samples, n_classes, 1) boolean mask of which
        # classes are in each sample's confidence set
        set_masks = y_pred_sets[:, :, 0]
        set_sizes = set_masks.sum(axis=1)

        # empirical coverage: fraction of test layers where the TRUE state
        # was actually inside the predicted confidence set
        in_set = np.array([set_masks[i, y_test[i]] for i in range(len(y_test))])
        empirical_coverage = in_set.mean()

        # singleton = model is confident enough to authorize autonomous action
        is_singleton = (set_sizes == 1)
        autonomy_rate = is_singleton.mean()

        # of the layers where we WOULD act autonomously, how often is the
        # single predicted state actually correct?
        if is_singleton.sum() > 0:
            singleton_correct = (y_pred[is_singleton] == y_test[is_singleton]).mean()
        else:
            singleton_correct = np.nan

        print(f"\n--- Target coverage: {target_coverage:.0%} (alpha={alpha}) ---")
        print(f"  Empirical coverage:        {empirical_coverage:.1%}  (should be >= target)")
        print(f"  Mean confidence set size:  {set_sizes.mean():.2f}  (1=confident, 5=totally unsure)")
        print(f"  Autonomy rate (singleton): {autonomy_rate:.1%}  of layers eligible for autonomous action")
        print(f"  Accuracy WHEN autonomous:  {singleton_correct:.1%}  (this is your safety number)")

        results[alpha] = dict(
            target_coverage=target_coverage,
            empirical_coverage=empirical_coverage,
            mean_set_size=set_sizes.mean(),
            autonomy_rate=autonomy_rate,
            singleton_accuracy=singleton_correct,
        )

    return results


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)

    all_results = {}
    for held_out in ['B6', 'B7', 'B8']:
        train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
        all_results[held_out] = run_gate2(df, train_builds, held_out)

    print(f"\n{'='*70}")
    print("SUMMARY: autonomy rate vs. accuracy-when-autonomous, at 90% target coverage")
    print(f"{'='*70}")
    for build, res in all_results.items():
        r = res[0.10]
        print(f"{build}: autonomy_rate={r['autonomy_rate']:.1%}, "
              f"accuracy_when_autonomous={r['singleton_accuracy']:.1%}, "
              f"empirical_coverage={r['empirical_coverage']:.1%}")
