"""
Gate 2, corrected labeling pass
=================================
The original Gate 2 script split each HELD-OUT build in half (calib/test)
to demonstrate the coverage/autonomy tradeoff -- that analysis stands as
reported. But it means only half of each build ever got scored, which
isn't good enough input for Gate 4 (which needs a confidence label on
EVERY layer to make a policy decision).

Fix: calibrate using a held-out slice of the TRAINING builds instead of
splitting the held-out build. This lets the full held-out build (all 312
layers) get scored, and doesn't cost us anything -- the two training
builds have plenty of data to spare a calibration slice.

Output: one CSV with 'is_confident' (singleton) attached to every one of
the 936 layers, ready to feed directly into Gate 4.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mapie.classification import SplitConformalClassifier
import mord

DATA_PATH = '/content/drive/MyDrive/DC-CPT-Project/Data/processed/gate3_labeled_dataset.csv'
OUT_DIR = '/content/drive/MyDrive/DC-CPT-Project/Data/processed'

FEATURE_COLS = [
    'tam_mean', 'tam_max', 'tam_p90',
    'scr_mean', 'scr_std', 'scr_min', 'scr_max',
    'scan_speed', 'hatch_spacing',
]
TARGET_CONFIDENCE = 0.90  # matches the 90% target coverage level used before


def label_full_build(df, held_out_build, calib_frac=0.3, seed=42):
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out_build]
    train_pool = df[df['build'].isin(train_builds)].sample(frac=1, random_state=seed)

    n_calib = int(len(train_pool) * calib_frac)
    calib_df = train_pool.iloc[:n_calib]
    fit_df = train_pool.iloc[n_calib:]

    test_df = df[df['build'] == held_out_build]  # FULL build, not split

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(fit_df[FEATURE_COLS])
    X_calib = scaler.transform(calib_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])

    base_model = mord.LogisticAT(alpha=1.0)
    base_model.fit(X_fit, fit_df['severity'].values)

    mapie_model = SplitConformalClassifier(
        estimator=base_model, confidence_level=TARGET_CONFIDENCE, prefit=True, conformity_score='aps',
    )
    mapie_model.conformalize(X_calib, calib_df['severity'].values)
    y_pred, y_pred_sets = mapie_model.predict_set(X_test)

    set_masks = y_pred_sets[:, :, 0]
    set_sizes = set_masks.sum(axis=1)

    result = test_df.copy()
    result['is_confident'] = (set_sizes == 1)
    result['confidence_set_size'] = set_sizes
    return result


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)

    labeled = pd.concat(
        [label_full_build(df, b) for b in ['B6', 'B7', 'B8']],
        ignore_index=True
    )

    print("is_confident coverage check:")
    print(labeled.groupby('build')['is_confident'].agg(['sum', 'count', 'mean']))

    out_path = f"{OUT_DIR}/gate2_full_labeled_dataset.csv"
    labeled.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
