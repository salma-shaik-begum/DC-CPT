"""
MIRI: Manufacturing Intervention Readiness Index
====================================================
A continuous composite score, with LEARNED (not hand-tuned) weights,
validated against INTERNAL PIPELINE CONSISTENCY -- not external ground
truth, which is not currently available for this dataset (see prior
discussion on XCT/porosity data limitations).

HONESTY NOTE FOR YOUR METHODS SECTION (keep this precise, don't inflate):
MIRI is learned to approximate Gate 4's discrete action_tier, using
CONTINUOUS underlying signals rather than the discrete/thresholded
versions Gate 1-3 use internally. This means MIRI is not a trivial
re-statement of the rule table -- it adds resolution WITHIN a given
action tier (e.g. distinguishing a "solidly Stable" layer from a
"borderline Stable" layer that got the same discrete action) and can
reveal near-boundary cases the discrete gates treat identically.

What this validates:  MIRI behaves coherently with the governance
pipeline's own internal logic (monotonic separation across severity
states, meaningful correlation with realized actions).
What this does NOT validate: whether MIRI (or the governance pipeline
generally) correctly identifies REAL manufacturing defects. That
requires external ground truth (e.g. segmented XCT porosity data),
which is identified as necessary future work.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import sys
sys.path.insert(0, '/content/drive/MyDrive/DC-CPT-Project/src')
from save_utils import save_table

DATA_PATH = '/content/drive/MyDrive/DC-CPT-Project/Data/processed/gate4_quality_dataset.csv'
OUT_DIR = '/content/drive/MyDrive/DC-CPT-Project/Data/processed'

STATE_NAMES = ['Stable', 'Degrading', 'Recoverable', 'Critical', 'Irrecoverable']

# Continuous input signals -- deliberately NOT the discrete severity/
# confidence/physics_admissible booleans themselves, so MIRI is a genuine
# regression onto raw signals rather than a re-encoding of the rule table.
MIRI_FEATURES = [
    'tam_p90',                 # peak thermal severity
    'scr_std',                 # cooling-rate instability
    'confidence_set_size',     # from Gate 2 -- continuous ambiguity signal
    'scr_residual',            # from Gate 3 -- physics-consistency deviation
]


def prepare_features(df):
    df = df.copy()
    df['scr_residual'] = df['scr_residual'].abs()  # magnitude of physics deviation, not signed
    return df


def fit_miri(df, train_builds):
    """Learn MIRI weights via linear regression onto Gate 4's action_tier,
    fit on training builds only. Coefficients are the 'learned weights'
    -- report these directly in the paper as the MIRI formula."""
    train_df = df[df['build'].isin(train_builds)].dropna(subset=MIRI_FEATURES + ['action_tier'])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[MIRI_FEATURES])
    y_train = train_df['action_tier'].values

    reg = LinearRegression().fit(X_train, y_train)
    return reg, scaler


def evaluate_miri(df, reg, scaler, held_out_build):
    test_df = df[df['build'] == held_out_build].dropna(subset=MIRI_FEATURES + ['action_tier']).copy()
    X_test = scaler.transform(test_df[MIRI_FEATURES])
    test_df['MIRI'] = reg.predict(X_test)

    print(f"\n{'='*70}")
    print(f"MIRI evaluation, held out: {held_out_build}")
    print(f"{'='*70}")

    # 1. Correlation with realized action tier -- expect strong but not
    #    perfect (perfect would suggest MIRI is just re-deriving the rule
    #    table with no added resolution)
    rho, pval = spearmanr(test_df['MIRI'], test_df['action_tier'])
    print(f"Spearman correlation with action_tier: rho={rho:.3f} (p={pval:.2e})")

    # 2. Monotonic separation across true severity states -- this is the
    #    key internal-consistency check
    print("\nMIRI distribution by true severity state:")
    summary = test_df.groupby(test_df['severity'].map(dict(enumerate(STATE_NAMES))))['MIRI'].agg(['mean', 'std', 'count'])
    summary = summary.reindex(STATE_NAMES)
    print(summary)

    is_monotonic = summary['mean'].is_monotonic_increasing
    print(f"\nMonotonic increase Stable -> Irrecoverable: {is_monotonic}")

    # 3. Within-tier resolution -- does MIRI vary meaningfully WITHIN a
    #    single action_tier, i.e. is it adding information beyond the
    #    discrete decision?
    print("\nMIRI variance WITHIN each action tier (checks added resolution beyond discrete gates):")
    within_tier = test_df.groupby('action')['MIRI'].agg(['mean', 'std', 'count'])
    print(within_tier)

    return test_df


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    df = prepare_features(df)

    all_results = {}
    weight_rows = []
    for held_out in ['B6', 'B7', 'B8']:
        train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
        reg, scaler = fit_miri(df, train_builds)

        print(f"\nLearned MIRI weights (held out {held_out}, standardized features):")
        for feat, coef in zip(MIRI_FEATURES, reg.coef_):
            print(f"  {feat}: {coef:+.3f}")
            weight_rows.append(dict(held_out=held_out, feature=feat, weight=coef))
        print(f"  intercept: {reg.intercept_:.3f}")
        weight_rows.append(dict(held_out=held_out, feature='intercept', weight=reg.intercept_))

        result = evaluate_miri(df, reg, scaler, held_out)
        all_results[held_out] = result

    save_table(pd.DataFrame(weight_rows), 'miri_learned_weights')

    combined = pd.concat(all_results.values(), ignore_index=True)
    save_table(combined, 'miri_labeled_dataset')
