# Decision-Centric Cyber-Physical Twin (DC-CPT): Reproducibility Package

This repository contains the complete, reproducible codebase, results, and
documentation for the DC-CPT manufacturing decision-governance framework,
developed and validated on NIST AM Bench 2022 (AMB2022-01) laser powder bed
fusion (LPBF) data.

## What this package contains

- **`src/`** — every experiment script, organized in the order experiments
  were run and should be reproduced. Each numbered folder corresponds to
  one stage of the pipeline.
- **`results/tables/`** — every CSV output referenced in the paper.
- **`results/figures/`** — every figure (PNG) referenced in the paper.
- **`data/processed/`** — intermediate labeled datasets (small, included).
  Raw NIST source data is NOT bundled here due to size (see Data Sources
  below) — scripts expect it at `data/raw/`, downloaded separately.
- **`docs/`** — terminology glossary (critical — see below) and results
  section draft.

## IMPORTANT: terminology discipline

Before reading or citing any result in this package, read
`docs/terminology_glossary.md`. This project uses proxy severity labels
derived from thermal/process data, not confirmed defect ground truth
(except where explicitly noted as XCT-audited). The glossary defines the
precise, non-overclaiming language used throughout — e.g. "thermal-process
severity states" not "defect labels," "independent physical validation
audit" not "XCT validates the model." Follow this discipline in any paper
text drawn from this package.

## Data Sources (must be downloaded separately)

This project uses public NIST AM Bench data, not redistributed here due to
size (multi-GB HDF5/CT files). Download from:

| Dataset | DOI | Used for |
|---|---|---|
| AMB2022-01 3D Build Thermography (B6/B7/B8 TAM/SCR) | https://doi.org/10.18434/mds2-2715 | Gates 1-4, all core experiments |
| AMB2022-01 Scan Strategy + Thermocouples | https://doi.org/10.18434/mds2-2607 | Gate 3 physics features |
| AM Bench 2022 IN718 Serial Sectioning + XCT | https://doi.org/10.18434/mds2-2767 | XCT independent validation audit (src/11) |

Place downloaded files under `data/raw/` matching the paths referenced at
the top of each script (`DATA_PATH`, `TAM_PATH`, `DREAM3D_PATH`, etc.).

## Reproduction order

Run in this order — later stages depend on earlier stages' saved outputs:

```
00_data_loader/build_gate_dataset.py
    -> joins TAM/SCR/scan-strategy/thermocouples into all_builds_gate_dataset.csv

01_gate1_state_assessment/gate1_state_assessment.py
    -> ordinal severity labeling + calibrated model, leave-one-build-out
    -> produces gate1_labeled_dataset.csv, gate1_fold_results.csv

02_gate2_decision_reliability/
    gate2_full_labeling.py       -> conformal confidence labels, full builds
    gate2_decision_reliability.py -> coverage/autonomy rate at multiple alphas
    gate2_singleton_diagnostic.py -> which severity states are autonomous
    gate2_escalation_quality.py   -> adjacency check on escalated predictions
    gate2_estimator_comparison.py -> ordinal vs Random Forest conformal comparison

03_gate3_physics_admissibility/gate3_physics_admissibility.py
    -> VED envelope + TAM-SCR consistency physics gate

04_gate4_policy_engine/gate4_policy_engine.py
    -> intent-parameterized (quality/productivity) action policy

05_miri/miri_readiness_index.py
    -> composite readiness index, internal-consistency validated

06_baseline_comparison/
    baseline_comparison.py -> 8-model + manufacturing-domain baselines
    tier_b_analysis.py     -> risk-coverage curve, modern baselines (LightGBM/
                               CatBoost/HistGB/ExtraTrees), Wilcoxon test, SHAP

07_calibration_and_shap/tier_b_priorities_2to5.py
    -> reliability diagrams, ECE, Brier score, SHAP dependence + cross-build stability,
       thermal-vs-process feature ablation

08_ablation_study/ablation_study.py
    -> Gate 2 / Gate 3 removal ablation (4 conditions)

09_bootstrap_ci/bootstrap_confidence_intervals.py
    -> 95% CIs on every headline metric

10_temporal_analysis/temporal_analysis.py
    -> lag/rolling/cumulative features vs. independent-layer baseline

11_xct_validation/ (run in this exact sub-order)
    inspect_xct_reconstruction.py              -> structure check
    xct_metadata_and_sanity_check.py           -> coordinate metadata + slice sanity check
    xct_bimodality_check.py                    -> numeric histogram peak detection
    xct_porosity_2d_test.py                    -> single-slice segmentation test
    xct_porosity_multi_slice_test.py           -> 20-slice validation sweep
    xct_full_porosity_profile.py               -> all 882 slices, build-Z conversion
    xct_severity_correlation.py                -> Stage 1 (2D) correlation vs. severity
    xct_3d_pore_analysis.py                    -> TRUE 3D pore segmentation (Stage 2)
    xct_stage3_registration_and_descriptors.py -> threshold sensitivity + pore descriptors
    xct_stage3_corrected.py                    -> EXACT registration fix (use this, not
                                                    the approximate version in the prior script)
    xct_stage4_final_checks.py                 -> unit verification, lagged/cumulative
                                                    features, pore-event AUC, final summary
```

`utils/` contains `project_setup.py` (folder structure) and `save_utils.py`
(the `save_table`/`save_figure` helpers every script imports) — run
`project_setup.py` once per fresh environment before anything else.

## Environment

See `requirements.txt` for exact package versions. A `Dockerfile` is
provided for full environment reproducibility — see Docker section below.

## Key results at a glance

| Experiment | Headline result | Table/Figure |
|---|---|---|
| Gate 1 (severity) | Leave-one-build-out accuracy 64-74%, MACE 0.28-0.42 | `gate1_fold_results.csv` |
| Gate 2 (conformal) | 90% target coverage achieved (94.6-97.6% empirical); 100% adjacency in escalated cases | `gate2_fold_results.csv`, `xct...` N/A |
| Gate 3 (physics) | Physics-inadmissibility rises monotonically with severity (3-7% at Stable to ~99% at Critical) | `gate3_summary_by_build.csv` |
| Gate 4 (policy) | Intent measurably shifts action conservativeness at low severity, converges at high severity | `gate4_intent_comparison_tiers.csv` |
| Baseline comparison | Ordinal model: 100% adjacent conformal sets vs. Random Forest's ~92% | `baseline_comparison_summary.csv` |
| Ablation | Full pipeline: 0% unsafe actuation vs. 15.4% with no governance | `ablation_results_pooled.csv` |
| Temporal features | 28.2% MACE reduction with lag/rolling/cumulative features | `temporal_vs_independent_comparison.csv` |
| XCT independent audit | No statistically significant correlation found (p>0.10, n=18-24); registration verified to 4.1µm | `xct_final_local_correlation_results_corrected.csv` |

## Known corrections applied (documented for transparency)

- **CatBoost MACE bug (fixed):** original run showed MACE≈1.5-1.65 due to a
  shape mismatch (`(n,1)` float predictions vs. `(n,)` integer labels).
  Fixed via explicit `.flatten().astype(int)` in `tier_b_analysis.py`.
  Corrected values: MACE 0.228-0.296, consistent with other ensemble models.
- **Gate 4 action-tier ordering bug (fixed):** an earlier version allowed
  physics-driven downgrades to convert human-escalation decisions into
  autonomous actions. Fixed by reordering `ACTION_TIERS` so `Escalate_Human`
  is more conservative than any autonomous action.
- **Layer assignment (fixed):** XCT-to-build-layer mapping corrected from
  `round(Z/40)` to `floor(Z/40)`, the physically correct binning for
  layers occupying `[n*40, (n+1)*40)` µm bands.
- **XCT spatial registration (corrected):** an initial pass used
  approximate proportional registration; `xct_stage3_corrected.py`
  implements exact registration using NIST's native `Xgrid_v`/`Ygrid_v`
  coordinate arrays, verified to a 0.2-4.1µm centroid offset.

## Citation

If you use this code or data, please cite both this repository (Zenodo DOI
to be assigned on deposit) and the underlying NIST AM Bench datasets listed
above.
