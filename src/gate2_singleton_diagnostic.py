"""
Standalone diagnostic: which true severity states make up the
autonomous (singleton) predictions at 90% target coverage?
Paste and run as its own cell -- rebuilds everything needed from scratch.
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

df = pd.read_csv(DATA_PATH)

for held_out in ['B6', 'B7', 'B8']:
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
    train_df = df[df['build'].isin(train_builds)]

    held_df = df[df['build'] == held_out].sample(frac=1, random_state=42)
    n_calib = int(len(held_df) * 0.5)
    calib_df, test_df = held_df.iloc[:n_calib], held_df.iloc[n_calib:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    X_calib = scaler.transform(calib_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])
    y_train, y_calib, y_test = train_df['severity'].values, calib_df['severity'].values, test_df['severity'].values

    base_model = mord.LogisticAT(alpha=1.0)
    base_model.fit(X_train, y_train)

    mapie_model = SplitConformalClassifier(
        estimator=base_model, confidence_level=0.90, prefit=True, conformity_score='aps',
    )
    mapie_model.conformalize(X_calib, y_calib)
    y_pred, y_pred_sets = mapie_model.predict_set(X_test)

    set_masks = y_pred_sets[:, :, 0]
    set_sizes = set_masks.sum(axis=1)
    is_singleton = (set_sizes == 1)

    print(f"\n=== {held_out} (90% target coverage) ===")
    print(f"Total test layers: {len(y_test)}, autonomous (singleton): {is_singleton.sum()}")

    singleton_true_states = y_test[is_singleton]
    print("True state distribution among AUTONOMOUS (singleton) layers:")
    vals, counts = np.unique(singleton_true_states, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"  {STATE_NAMES[v]}: {c}")

    # also show what's in the FULL test set for comparison
    print("For comparison, true state distribution in FULL test set:")
    vals_all, counts_all = np.unique(y_test, return_counts=True)
    for v, c in zip(vals_all, counts_all):
        print(f"  {STATE_NAMES[v]}: {c}")
