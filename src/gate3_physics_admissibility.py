"""
DC-CPT Gate 3: Physics Admissibility
======================================
Two independent physics checks, each layer must pass BOTH to be
"physics admissible":

  (A) Volumetric Energy Density (VED) Envelope
      VED = P / (v * h * t)   [J/mm^3]
      P = laser power (W), v = scan speed (mm/s),
      h = hatch spacing (mm), t = layer thickness (mm)
      This is the standard LPBF process-energy indicator. We define the
      admissible envelope statistically from the TRAINING builds' own
      distribution (mean +/- k*std), i.e. a statistical-process-control
      style bound. This is a placeholder for a literature/domain-expert-
      verified VED window (e.g. published IN718 process maps) -- swap
      compute_ved_envelope() for literature bounds once you have a
      citable source; treat the current version as a self-consistency
      check, not an absolute physical limit.

  (B) TAM-SCR Physical Consistency
      Physically, time-above-melt and cooling rate should be related:
      layers that stayed hot longer should generally show a
      characteristic cooling response. We fit this relationship on
      layers Gate 1 already labeled 'Stable' (i.e. presumed physically
      normal), then flag any layer whose actual TAM/SCR relationship
      deviates strongly from that fitted baseline -- this catches
      layers where the *pattern* between two physically linked
      measurements breaks down, not just where one measurement alone
      looks extreme.

A layer that fails either check is flagged NOT physics-admissible,
meaning Gate 4 (Policy) should not authorize the action Gate 1/2 would
otherwise recommend -- it escalates one severity level instead.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
sys.path.insert(0, '/content/drive/MyDrive/DC-CPT-Project/src')
from save_utils import save_table

DATA_PATH = '/content/drive/MyDrive/DC-CPT-Project/Data/processed/gate1_labeled_dataset.csv'
OUT_DIR = '/content/drive/MyDrive/DC-CPT-Project/Data/processed'

LAYER_THICKNESS_MM = 0.04  # 40 um, from build metadata
STATE_NAMES = ['Stable', 'Degrading', 'Recoverable', 'Critical', 'Irrecoverable']

VED_STD_MULTIPLIER = 2.0       # how many std devs define the admissible VED band
RESIDUAL_STD_MULTIPLIER = 3.0  # how many std devs define admissible TAM-SCR residual
                                 # (raised from 2.0 -- that setting flagged 61-73% of
                                 # ALL layers as inadmissible, which is too aggressive
                                 # to function as a meaningful veto; re-tune this against
                                 # domain expectations once you have real defect labels)


def compute_ved(power_w, speed_mm_s, hatch_mm, layer_thickness_mm=LAYER_THICKNESS_MM):
    """Volumetric energy density in J/mm^3."""
    return power_w / (speed_mm_s * hatch_mm * layer_thickness_mm)


def fit_ved_envelope(df, train_builds):
    """Statistically-derived admissible VED band from training builds only
    -- this must be fit on training data alone to avoid leaking test-build
    information into the admissibility threshold."""
    train_df = df[df['build'].isin(train_builds)].copy()
    hatch_mm = train_df['hatch_spacing'] / 1000.0  # stored in um, convert to mm
    ved = compute_ved(train_df['commanded_power_mean'], train_df['scan_speed'], hatch_mm)
    ved = ved.dropna()
    mean, std = ved.mean(), ved.std()
    lower = mean - VED_STD_MULTIPLIER * std
    upper = mean + VED_STD_MULTIPLIER * std
    return lower, upper, mean, std


def fit_tam_scr_baseline(df, train_builds):
    """Fit expected SCR ~ f(TAM) relationship using only layers Gate 1
    already labeled Stable, on the training builds. This is our
    'physically normal' reference relationship."""
    train_df = df[df['build'].isin(train_builds)]
    stable_df = train_df[train_df['severity'] == 0].dropna(subset=['tam_p90', 'scr_mean'])

    X = stable_df[['tam_p90']].values
    y = stable_df['scr_mean'].values
    reg = LinearRegression().fit(X, y)

    # residual std on the SAME stable/training data -- defines what counts
    # as a "normal" deviation from the fitted relationship
    residuals = y - reg.predict(X)
    residual_std = residuals.std()

    return reg, residual_std


def apply_gate3(df, train_builds):
    df = df.copy()
    hatch_mm = df['hatch_spacing'] / 1000.0
    df['ved'] = compute_ved(df['commanded_power_mean'], df['scan_speed'], hatch_mm)

    ved_lower, ved_upper, ved_mean, ved_std = fit_ved_envelope(df, train_builds)
    df['ved_admissible'] = df['ved'].between(ved_lower, ved_upper)

    reg, residual_std = fit_tam_scr_baseline(df, train_builds)
    valid = df['tam_p90'].notna() & df['scr_mean'].notna()
    df.loc[valid, 'scr_predicted'] = reg.predict(df.loc[valid, ['tam_p90']].values)
    df['scr_residual'] = df['scr_mean'] - df['scr_predicted']
    residual_threshold = RESIDUAL_STD_MULTIPLIER * residual_std
    df['tam_scr_admissible'] = df['scr_residual'].abs() <= residual_threshold

    df['physics_admissible'] = df['ved_admissible'] & df['tam_scr_admissible'].fillna(False)

    return df, dict(
        ved_band=(ved_lower, ved_upper), ved_mean=ved_mean, ved_std=ved_std,
        residual_std=residual_std, residual_threshold=residual_threshold,
    )


def evaluate_gate3(df, held_out_build, params):
    test_df = df[df['build'] == held_out_build]

    print(f"\n{'='*70}")
    print(f"Gate 3 evaluation, held out: {held_out_build}")
    print(f"{'='*70}")
    print(f"VED admissible band: [{params['ved_band'][0]:.2f}, {params['ved_band'][1]:.2f}] J/mm^3 "
          f"(train mean={params['ved_mean']:.2f}, std={params['ved_std']:.2f})")
    print(f"TAM-SCR residual threshold: +/- {params['residual_threshold']:.0f}")

    n_total = len(test_df)
    n_ved_fail = (~test_df['ved_admissible']).sum()
    n_tam_scr_fail = (~test_df['tam_scr_admissible']).sum()
    n_either_fail = (~test_df['physics_admissible']).sum()

    print(f"\nOut of {n_total} test layers:")
    print(f"  Failed VED envelope check:        {n_ved_fail} ({100*n_ved_fail/n_total:.1f}%)")
    print(f"  Failed TAM-SCR consistency check:  {n_tam_scr_fail} ({100*n_tam_scr_fail/n_total:.1f}%)")
    print(f"  Failed EITHER (physics-inadmissible overall): {n_either_fail} ({100*n_either_fail/n_total:.1f}%)")

    # Does physics-inadmissibility concentrate at higher severity states?
    # This is the key validation: if the physics gate is meaningful, it
    # should disproportionately flag Critical/Irrecoverable layers, not
    # flag randomly across all severity levels.
    print(f"\nPhysics-inadmissible rate by severity state (does it concentrate at high severity?):")
    by_state_rows = []
    for state_idx, state_name in enumerate(STATE_NAMES):
        state_df = test_df[test_df['severity'] == state_idx]
        if len(state_df) == 0:
            continue
        fail_rate = (~state_df['physics_admissible']).mean()
        print(f"  {state_name}: {fail_rate:.1%} inadmissible (n={len(state_df)})")
        by_state_rows.append(dict(held_out=held_out_build, state=state_name,
                                   inadmissible_rate=fail_rate, n=len(state_df)))

    summary_row = dict(
        held_out=held_out_build, n_total=n_total,
        ved_fail_rate=n_ved_fail / n_total, tam_scr_fail_rate=n_tam_scr_fail / n_total,
        either_fail_rate=n_either_fail / n_total,
    )
    return summary_row, by_state_rows


if __name__ == '__main__':
    df_raw = pd.read_csv(DATA_PATH)

    all_results = {}
    summary_rows, by_state_all = [], []
    for held_out in ['B6', 'B7', 'B8']:
        train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
        df_gated, params = apply_gate3(df_raw, train_builds)
        summary_row, by_state_rows = evaluate_gate3(df_gated, held_out, params)
        summary_rows.append(summary_row)
        by_state_all.extend(by_state_rows)
        all_results[held_out] = df_gated[df_gated['build'] == held_out]

    save_table(pd.DataFrame(summary_rows), 'gate3_summary_by_build')
    save_table(pd.DataFrame(by_state_all), 'gate3_inadmissibility_by_severity')

    combined = pd.concat(all_results.values(), ignore_index=True)
    out_path = f"{OUT_DIR}/gate3_labeled_dataset.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
