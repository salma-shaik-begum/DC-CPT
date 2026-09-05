"""
Baseline Model Comparison
============================
Compares Gate 1's ordinal model (mord.LogisticAT) against 7 standard
classifiers, on IDENTICAL leave-one-build-out splits, same features,
same proxy severity labels.

Two metrics matter here, for different reasons:
  - accuracy / macro F1 : standard classification performance
  - mean absolute class error (MACE) : ORDINAL-AWARE error. This is the
    metric that should show the ordinal model's advantage -- a model
    that confuses Stable with Irrecoverable should be penalized far more
    than one that confuses Stable with Degrading, and only MACE captures
    that. Plain accuracy/F1 treat all misclassifications as equally bad,
    which is exactly the wrong lens for a severity hierarchy.

Install once:
  pip install mord scikit-learn xgboost pandas numpy --break-system-packages
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
import mord
import sys
sys.path.insert(0, '/content/drive/MyDrive/DC-CPT-Project/src')
from save_utils import save_table

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed -- skipping XGBoost baseline. "
          "Run: pip install xgboost --break-system-packages")

DATA_PATH = '/content/drive/MyDrive/DC-CPT-Project/Data/processed/gate1_labeled_dataset.csv'
OUT_DIR = '/content/drive/MyDrive/DC-CPT-Project/Data/processed'

STATE_NAMES = ['Stable', 'Degrading', 'Recoverable', 'Critical', 'Irrecoverable']
FEATURE_COLS = [
    'tam_mean', 'tam_max', 'tam_p90',
    'scr_mean', 'scr_std', 'scr_min', 'scr_max',
    'scan_speed', 'hatch_spacing',
]


def spc_control_chart_predict(train_df, test_df, feature='tam_p90', k_sigma=(1, 2, 3)):
    """The actual manufacturing-standard baseline: a Shewhart-style
    control chart on a single monitored variable. Control limits are
    fit on TRAINING data's Stable-labeled layers only (the 'in-control'
    reference population, exactly how SPC is used in practice), then
    every layer is bucketed into a severity state by how many standard
    deviations it falls from that in-control mean.

    This is NOT a learned model -- no fitting beyond computing mean/std
    -- deliberately, because this represents current shop-floor practice,
    which this paper argues is insufficient without decision intelligence
    layered on top. It should be the WEAKEST baseline, and showing that
    clearly is the point."""
    stable_train = train_df[train_df['severity'] == 0][feature].dropna()
    mu, sigma = stable_train.mean(), stable_train.std()

    def to_severity(x):
        if pd.isna(x):
            return 0
        z = abs(x - mu) / sigma if sigma > 0 else 0
        if z < k_sigma[0]:
            return 0  # Stable
        elif z < k_sigma[1]:
            return 1  # Degrading
        elif z < k_sigma[2]:
            return 2  # Recoverable
        elif z < k_sigma[2] * 1.5:
            return 3  # Critical
        else:
            return 4  # Irrecoverable

    return test_df[feature].apply(to_severity).values


def physics_only_ved_predict(train_df, test_df):
    """Second manufacturing-domain baseline: pure physics rule using
    Volumetric Energy Density deviation from the training set's nominal
    VED, no thermal camera data, no ML at all -- represents physics-only
    process control, the dominant paradigm in AM before ML-based
    monitoring entered the picture."""
    layer_thickness_mm = 0.04
    hatch_mm_train = train_df['hatch_spacing'] / 1000.0
    ved_train = train_df['commanded_power_mean'] / (train_df['scan_speed'] * hatch_mm_train * layer_thickness_mm)
    mu, sigma = ved_train.mean(), ved_train.std()

    hatch_mm_test = test_df['hatch_spacing'] / 1000.0
    ved_test = test_df['commanded_power_mean'] / (test_df['scan_speed'] * hatch_mm_test * layer_thickness_mm)

    def to_severity(v):
        if pd.isna(v):
            return 0
        z = abs(v - mu) / sigma if sigma > 0 else 0
        if z < 0.5:
            return 0
        elif z < 1.0:
            return 1
        elif z < 1.5:
            return 2
        elif z < 2.0:
            return 3
        else:
            return 4

    return ved_test.apply(to_severity).values


def get_tuned_ordinal_model(X_train, y_train):
    """Grid-search the ordinal model's regularization strength instead of
    using an arbitrary fixed value -- needed for a fair comparison against
    the ensemble baselines, which use reasonable defaults/tuned settings."""
    n = len(X_train)
    split = int(n * 0.8)
    best_alpha, best_mace = 1.0, np.inf
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        model = mord.LogisticAT(alpha=alpha)
        model.fit(X_train[:split], y_train[:split])
        preds = model.predict(X_train[split:])
        mace = np.mean(np.abs(y_train[split:] - preds))
        if mace < best_mace:
            best_mace, best_alpha = mace, alpha
    final_model = mord.LogisticAT(alpha=best_alpha)
    final_model.fit(X_train, y_train)
    return final_model, best_alpha


def get_models():
    models = {
        'Multinomial Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'k-Nearest Neighbors': KNeighborsClassifier(n_neighbors=15),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, max_depth=4, eval_metric='mlogloss', random_state=42
        )
    return models


def run_comparison(df):
    results = []

    for held_out in ['B6', 'B7', 'B8']:
        train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
        train_df = df[df['build'].isin(train_builds)]
        test_df = df[df['build'] == held_out]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[FEATURE_COLS])
        X_test = scaler.transform(test_df[FEATURE_COLS])
        y_train = train_df['severity'].values
        y_test = test_df['severity'].values

        # Ordinal model, properly tuned (not a fixed arbitrary alpha) --
        # fair comparison requires this get the same tuning effort as
        # the ensemble baselines' reasonable defaults
        ordinal_model, chosen_alpha = get_tuned_ordinal_model(X_train, y_train)
        y_pred_ord = ordinal_model.predict(X_test)
        results.append(dict(
            held_out=held_out, model=f'Ordinal Logistic (ours, alpha={chosen_alpha})',
            accuracy=accuracy_score(y_test, y_pred_ord),
            macro_f1=f1_score(y_test, y_pred_ord, average='macro', zero_division=0),
            mace=np.mean(np.abs(y_test - y_pred_ord)),
        ))

        for model_name, model in get_models().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            mace = np.mean(np.abs(y_test - y_pred))  # mean absolute CLASS error (ordinal distance)

            results.append(dict(
                held_out=held_out, model=model_name,
                accuracy=acc, macro_f1=macro_f1, mace=mace,
            ))

        # Two MANUFACTURING-DOMAIN baselines, not just generic ML --
        # these are the comparisons a manufacturing-venue reviewer
        # actually cares about: current shop-floor practice, not just
        # "how does this compare to other classifiers"
        y_pred_spc = spc_control_chart_predict(train_df, test_df)
        results.append(dict(
            held_out=held_out, model='SPC Control Chart (manufacturing baseline)',
            accuracy=accuracy_score(y_test, y_pred_spc),
            macro_f1=f1_score(y_test, y_pred_spc, average='macro', zero_division=0),
            mace=np.mean(np.abs(y_test - y_pred_spc)),
        ))

        y_pred_ved = physics_only_ved_predict(train_df, test_df)
        results.append(dict(
            held_out=held_out, model='Physics-Only VED Rule (manufacturing baseline)',
            accuracy=accuracy_score(y_test, y_pred_ved),
            macro_f1=f1_score(y_test, y_pred_ved, average='macro', zero_division=0),
            mace=np.mean(np.abs(y_test - y_pred_ved)),
        ))

    return pd.DataFrame(results)


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    results_df = run_comparison(df)

    print("Per-fold results:")
    print(results_df.to_string(index=False))

    print(f"\n{'='*70}")
    print("SUMMARY: mean +/- std across the 3 held-out builds, per model")
    print(f"{'='*70}")
    summary = results_df.groupby('model').agg(
        accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
        mace_mean=('mace', 'mean'), mace_std=('mace', 'std'),
    ).sort_values('mace_mean')  # lower MACE = better ordinal-aware performance
    print(summary.round(3))

    save_table(results_df, 'baseline_comparison_per_fold')
    save_table(summary.reset_index(), 'baseline_comparison_summary')
