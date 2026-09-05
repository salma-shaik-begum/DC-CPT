"""
DC-CPT Gate 1: Manufacturing State Assessment
================================================
Builds the five-state ordinal severity hierarchy:
  0 Stable -> 1 Degrading -> 2 Recoverable -> 3 Critical -> 4 Irrecoverable

IMPORTANT HONESTY NOTE (keep this in your methods section):
We do not have pixel/layer-level defect ground truth (see prior discussion
on XCT/porosity data availability). Severity labels here are PROXY labels,
constructed from physically-motivated deviation rules (how far a layer's
thermal behavior and commanded process parameters sit from nominal/expected
values). This is a legitimate way to test the *mechanics* of the ordinal
model and calibration pipeline, but it is NOT a defect-detection claim.
When/if real defect labels become available (XCT segmentation, NIST
challenge updates, etc.), swap `build_proxy_severity_label()` for the real
target and re-run everything downstream unchanged.

Install once: pip install mord scikit-learn pandas numpy --break-system-packages
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import mord
import sys
sys.path.insert(0, '/content/drive/MyDrive/DC-CPT-Project/src')
from save_utils import save_table

DATA_PATH = '/content/drive/MyDrive/DC-CPT-Project/Data/processed/all_builds_gate_dataset.csv'
OUT_DIR = '/content/drive/MyDrive/DC-CPT-Project/Data/processed'

NOMINAL_POWER = 285.0  # W, from build metadata

STATE_NAMES = ['Stable', 'Degrading', 'Recoverable', 'Critical', 'Irrecoverable']


def build_proxy_severity_label(df):
    """
    Composite risk score from four physically-motivated signals:
      - tam_p90        : how hot the hottest 10% of the layer got
      - tam_frac_above_global_threshold : how much of the layer ran hot
      - scr_std         : cooling-rate instability across the layer
      - power_deviation : how far commanded power drifted from nominal

    Each is z-scored (so they're on comparable scales) and averaged.
    Binned into 5 ordinal states by quantile, so classes are balanced
    by construction -- revisit bin edges once real defect data anchors
    the thresholds to physically meaningful cut points.
    """
    df = df.copy()
    df['power_deviation'] = (df['commanded_power_mean'] - NOMINAL_POWER).abs()

    risk_features = ['tam_p90', 'tam_frac_above_global_threshold', 'scr_std', 'power_deviation']
    scaler = StandardScaler()
    z = scaler.fit_transform(df[risk_features].fillna(df[risk_features].median()))
    df['risk_score'] = z.mean(axis=1)

    # 5 ordinal bins via quantiles -> 0=Stable ... 4=Irrecoverable
    df['severity'] = pd.qcut(df['risk_score'], q=5, labels=False, duplicates='drop')
    return df


FEATURE_COLS = [
    'tam_mean', 'tam_max', 'tam_p90',
    'scr_mean', 'scr_std', 'scr_min', 'scr_max',
    'scan_speed', 'hatch_spacing',
]


def prepare(df):
    df = df.dropna(subset=FEATURE_COLS + ['commanded_power_mean']).copy()
    df = build_proxy_severity_label(df)
    return df


def train_and_evaluate(df, train_builds, test_build):
    train_df = df[df['build'].isin(train_builds)]
    test_df = df[df['build'] == test_build]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])
    y_train = train_df['severity'].values
    y_test = test_df['severity'].values

    # LogisticAT = ordinal "all-thresholds" logistic regression (mord):
    # respects class ORDER, unlike plain multiclass softmax, which is
    # the whole point vs treating severity as unordered categories.
    model = mord.LogisticAT(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n=== Held out: {test_build} (trained on {train_builds}) ===")
    print(classification_report(y_test, y_pred, target_names=STATE_NAMES,
                                 labels=list(range(5)), zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=list(range(5)))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(cm, index=STATE_NAMES, columns=STATE_NAMES))

    # Ordinal-aware error: how many classes off, not just right/wrong.
    # A Stable->Degrading miss is very different from Stable->Irrecoverable.
    mean_abs_class_error = np.mean(np.abs(y_test - y_pred))
    print(f"\nMean absolute class error (0=perfect, ordinal distance): {mean_abs_class_error:.3f}")

    accuracy = (y_test == y_pred).mean()
    return dict(held_out=test_build, accuracy=accuracy, mace=mean_abs_class_error)


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    df = prepare(df)

    print("Severity label distribution (proxy labels, all builds):")
    print(df['severity'].value_counts().sort_index().rename(index=dict(enumerate(STATE_NAMES))))

    # Leave-one-build-out: train on two builds, test on the third.
    # This is the right validation split -- random row-level splits would
    # leak information across layers of the same build and overstate performance.
    fold_results = []
    for held_out in ['B6', 'B7', 'B8']:
        train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
        result = train_and_evaluate(df, train_builds, held_out)
        fold_results.append(result)

    fold_results_df = pd.DataFrame(fold_results)
    save_table(fold_results_df, 'gate1_fold_results')

    df.to_csv(f"{OUT_DIR}/gate1_labeled_dataset.csv", index=False)
    print(f"\nSaved labeled dataset -> {OUT_DIR}/gate1_labeled_dataset.csv")
