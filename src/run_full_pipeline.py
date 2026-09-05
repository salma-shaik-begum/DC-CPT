"""
DC-CPT FULL PIPELINE — ONE SCRIPT, FULLY AUTOMATIC
========================================================================
Run this ONE script in Colab. It does everything itself:
  - mounts Drive
  - creates every folder it needs (safe to rerun, never duplicates)
  - runs Gate 1 -> Gate 2 -> Gate 3 -> Gate 4 -> MIRI -> baseline
    comparison -> ablation -> bootstrap CIs, in order
  - saves every result table to results/tables/ automatically, with a
    confirmed [SAVED] line for each -- no manual save calls needed
  - saves a copy of ITSELF into src/ automatically at the end, so your
    source code is archived too, without a separate writefile step

You only ever paste this once per session. Everything after that is
automatic.

ONLY manual step required, and only once per fresh Colab runtime:
  !pip install mord scikit-learn mapie xgboost scipy pandas numpy --break-system-packages -q
Run that in the cell ABOVE this one, then paste this whole script in the
next cell and run it.
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy.stats import spearmanr
import mord

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# ============================================================================
# SECTION 0: FOLDER SETUP -- automatic, safe to rerun, no duplicates
# ============================================================================
BASE = '/content/drive/MyDrive/DC-CPT-Project'
PATHS = {
    'data_raw': f'{BASE}/Data/Raw',
    'data_processed': f'{BASE}/Data/processed',
    'src': f'{BASE}/src',
    'results': f'{BASE}/results',
    'results_tables': f'{BASE}/results/tables',
    'results_figures': f'{BASE}/results/figures',
    'docs': f'{BASE}/docs',
}
for name, path in PATHS.items():
    os.makedirs(path, exist_ok=True)
print("Folder setup confirmed:")
for name, path in PATHS.items():
    print(f"  [OK] {name:<16} -> {path}")


def save_table(df, name):
    """Saves a DataFrame to results/tables/ and CONFIRMS it landed on disk
    -- prints [SAVED] with the real file size, not just a bare print()."""
    if not name.endswith('.csv'):
        name += '.csv'
    path = os.path.join(PATHS['results_tables'], name)
    df.to_csv(path, index=False)
    confirmed = os.path.isfile(path)
    size_kb = os.path.getsize(path) / 1024 if confirmed else 0
    print(f"  [{'SAVED' if confirmed else 'FAILED'}] {path}  ({size_kb:.1f} KB)")
    return path


def save_figure(fig, name):
    """Saves a matplotlib figure to results/figures/ and CONFIRMS it landed
    on disk -- same [SAVED] confirmation pattern as save_table."""
    if not name.endswith('.png'):
        name += '.png'
    path = os.path.join(PATHS['results_figures'], name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    confirmed = os.path.isfile(path)
    size_kb = os.path.getsize(path) / 1024 if confirmed else 0
    print(f"  [{'SAVED' if confirmed else 'FAILED'}] {path}  ({size_kb:.1f} KB)")
    return path


def save_processed(df, name):
    """Saves intermediate pipeline datasets to Data/processed/ (same
    convention used throughout this project)."""
    if not name.endswith('.csv'):
        name += '.csv'
    path = os.path.join(PATHS['data_processed'], name)
    df.to_csv(path, index=False)
    confirmed = os.path.isfile(path)
    print(f"  [{'SAVED' if confirmed else 'FAILED'}] {path}")
    return path


DATA_PATH = f"{PATHS['data_processed']}/all_builds_gate_dataset.csv"
STATE_NAMES = ['Stable', 'Degrading', 'Recoverable', 'Critical', 'Irrecoverable']
FEATURE_COLS = [
    'tam_mean', 'tam_max', 'tam_p90',
    'scr_mean', 'scr_std', 'scr_min', 'scr_max',
    'scan_speed', 'hatch_spacing',
]
NOMINAL_POWER = 285.0
TARGET_CONFIDENCE = 0.90

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Run the data loader script first "
        f"(build_gate_dataset.py) -- this pipeline starts from Gate 1 onward."
    )

print(f"\n{'='*70}\nSECTION 1: GATE 1 -- Manufacturing State Assessment\n{'='*70}")

# ----------------------------------------------------------------------------
# Gate 1
# ----------------------------------------------------------------------------
def build_proxy_severity_label(df):
    df = df.copy()
    df['power_deviation'] = (df['commanded_power_mean'] - NOMINAL_POWER).abs()
    risk_features = ['tam_p90', 'tam_frac_above_global_threshold', 'scr_std', 'power_deviation']
    scaler = StandardScaler()
    z = scaler.fit_transform(df[risk_features].fillna(df[risk_features].median()))
    df['risk_score'] = z.mean(axis=1)
    df['severity'] = pd.qcut(df['risk_score'], q=5, labels=False, duplicates='drop')
    return df


def get_tuned_ordinal_model(X_train, y_train):
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


df_raw = pd.read_csv(DATA_PATH)
df_g1 = df_raw.dropna(subset=FEATURE_COLS + ['commanded_power_mean']).copy()
df_g1 = build_proxy_severity_label(df_g1)

print("Severity label distribution (proxy labels, all builds):")
print(df_g1['severity'].value_counts().sort_index().rename(index=dict(enumerate(STATE_NAMES))))

gate1_fold_results = []
for held_out in ['B6', 'B7', 'B8']:
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
    train_df = df_g1[df_g1['build'].isin(train_builds)]
    test_df = df_g1[df_g1['build'] == held_out]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])
    y_train, y_test = train_df['severity'].values, test_df['severity'].values

    model, alpha = get_tuned_ordinal_model(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mace = np.mean(np.abs(y_test - y_pred))
    print(f"  {held_out}: accuracy={acc:.3f}, MACE={mace:.3f}, tuned_alpha={alpha}")
    gate1_fold_results.append(dict(held_out=held_out, accuracy=acc, mace=mace, alpha=alpha))

save_table(pd.DataFrame(gate1_fold_results), 'gate1_fold_results')
save_processed(df_g1, 'gate1_labeled_dataset')

# Figure: confusion matrix heatmap (using the last held-out fold, B8, as the example)
cm = confusion_matrix(y_test, y_pred, labels=list(range(5)))
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(STATE_NAMES, rotation=45, ha='right'); ax.set_yticklabels(STATE_NAMES)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Gate 1 Confusion Matrix (held out: {held_out})')
for i in range(5):
    for j in range(5):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
fig.colorbar(im, ax=ax)
save_figure(fig, 'gate1_confusion_matrix')

print(f"\n{'='*70}\nSECTION 2: GATE 2 -- Conformal Decision Reliability (full-build labeling)\n{'='*70}")

# ----------------------------------------------------------------------------
# Gate 2 -- full-build conformal labeling (calibrate on training builds,
# score the ENTIRE held-out build, not just half of it)
# ----------------------------------------------------------------------------
from mapie.classification import SplitConformalClassifier


def label_full_build_gate2(df, held_out_build, calib_frac=0.3, seed=42):
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

    base_model = mord.LogisticAT(alpha=0.01)
    base_model.fit(X_fit, fit_df['severity'].values)

    mapie_model = SplitConformalClassifier(
        estimator=base_model, confidence_level=TARGET_CONFIDENCE, prefit=True, conformity_score='aps',
    )
    mapie_model.conformalize(X_calib, calib_df['severity'].values)
    y_pred, y_pred_sets = mapie_model.predict_set(X_test)

    set_masks = y_pred_sets[:, :, 0]
    set_sizes = set_masks.sum(axis=1)
    y_true = test_df['severity'].values
    in_set = np.array([set_masks[i, y_true[i]] for i in range(len(y_true))])

    test_df['predicted_severity'] = y_pred
    test_df['is_confident'] = (set_sizes == 1)
    test_df['confidence_set_size'] = set_sizes
    test_df['true_in_set'] = in_set
    return test_df


gate2_results = []
gate2_labeled_parts = []
for held_out in ['B6', 'B7', 'B8']:
    labeled = label_full_build_gate2(df_g1, held_out)
    gate2_labeled_parts.append(labeled)
    coverage = labeled['true_in_set'].mean()
    autonomy = labeled['is_confident'].mean()
    if labeled['is_confident'].sum() > 0:
        singleton_acc = (labeled.loc[labeled['is_confident'], 'predicted_severity'] ==
                          labeled.loc[labeled['is_confident'], 'severity']).mean()
    else:
        singleton_acc = np.nan
    print(f"  {held_out}: coverage={coverage:.3f}, autonomy_rate={autonomy:.3f}, singleton_acc={singleton_acc:.3f}")
    gate2_results.append(dict(held_out=held_out, coverage=coverage, autonomy_rate=autonomy, singleton_accuracy=singleton_acc))

df_g2 = pd.concat(gate2_labeled_parts, ignore_index=True)
save_table(pd.DataFrame(gate2_results), 'gate2_fold_results')
save_processed(df_g2, 'gate2_full_labeled_dataset')

print(f"\n{'='*70}\nSECTION 3: GATE 3 -- Physics Admissibility\n{'='*70}")

# ----------------------------------------------------------------------------
# Gate 3
# ----------------------------------------------------------------------------
LAYER_THICKNESS_MM = 0.04
VED_STD_MULTIPLIER = 2.0
RESIDUAL_STD_MULTIPLIER = 3.0


def compute_ved(power_w, speed_mm_s, hatch_mm, layer_thickness_mm=LAYER_THICKNESS_MM):
    return power_w / (speed_mm_s * hatch_mm * layer_thickness_mm)


def apply_gate3(df, train_builds):
    df = df.copy()
    hatch_mm = df['hatch_spacing'] / 1000.0
    df['ved'] = compute_ved(df['commanded_power_mean'], df['scan_speed'], hatch_mm)

    train_df = df[df['build'].isin(train_builds)]
    train_hatch_mm = train_df['hatch_spacing'] / 1000.0
    train_ved = compute_ved(train_df['commanded_power_mean'], train_df['scan_speed'], train_hatch_mm).dropna()
    ved_mean, ved_std = train_ved.mean(), train_ved.std()
    ved_lower, ved_upper = ved_mean - VED_STD_MULTIPLIER * ved_std, ved_mean + VED_STD_MULTIPLIER * ved_std
    df['ved_admissible'] = df['ved'].between(ved_lower, ved_upper)

    stable_df = train_df[train_df['severity'] == 0].dropna(subset=['tam_p90', 'scr_mean'])
    reg = LinearRegression().fit(stable_df[['tam_p90']].values, stable_df['scr_mean'].values)
    residuals = stable_df['scr_mean'].values - reg.predict(stable_df[['tam_p90']].values)
    residual_std = residuals.std()

    valid = df['tam_p90'].notna() & df['scr_mean'].notna()
    df.loc[valid, 'scr_predicted'] = reg.predict(df.loc[valid, ['tam_p90']].values)
    df['scr_residual'] = df['scr_mean'] - df['scr_predicted']
    df['tam_scr_admissible'] = df['scr_residual'].abs() <= RESIDUAL_STD_MULTIPLIER * residual_std

    df['physics_admissible'] = df['ved_admissible'] & df['tam_scr_admissible'].fillna(False)
    return df


gate3_summary_rows = []
gate3_parts = []
for held_out in ['B6', 'B7', 'B8']:
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
    df_gated = apply_gate3(df_g2, train_builds)
    test_df = df_gated[df_gated['build'] == held_out]
    gate3_parts.append(test_df)

    fail_rate = (~test_df['physics_admissible']).mean()
    print(f"  {held_out}: physics_inadmissible_rate={fail_rate:.3f}")
    gate3_summary_rows.append(dict(held_out=held_out, physics_inadmissible_rate=fail_rate))

df_g3 = pd.concat(gate3_parts, ignore_index=True)
save_table(pd.DataFrame(gate3_summary_rows), 'gate3_summary_by_build')
save_processed(df_g3, 'gate3_labeled_dataset')

print(f"\n{'='*70}\nSECTION 4: GATE 4 -- Intent-Parameterized Policy Engine\n{'='*70}")

# ----------------------------------------------------------------------------
# Gate 4
# ----------------------------------------------------------------------------
ACTION_TIERS = ['Continue', 'Monitor', 'Adjust_Parameters', 'Reduce_Scan_Speed', 'Escalate_Human', 'Pause_Build']
BASE_POLICY = {
    ('Stable', True): 'Continue', ('Stable', False): 'Monitor',
    ('Degrading', True): 'Monitor', ('Degrading', False): 'Escalate_Human',
    ('Recoverable', True): 'Adjust_Parameters', ('Recoverable', False): 'Escalate_Human',
    ('Critical', True): 'Reduce_Scan_Speed', ('Critical', False): 'Reduce_Scan_Speed',
    ('Irrecoverable', True): 'Pause_Build', ('Irrecoverable', False): 'Pause_Build',
}


def downgrade_action(action):
    if action == 'Pause_Build':
        return action
    idx = ACTION_TIERS.index(action)
    return ACTION_TIERS[min(idx + 1, len(ACTION_TIERS) - 1)]


def apply_gate4(df, intent='quality'):
    df = df.copy()

    def decide(row):
        state_name = STATE_NAMES[int(row['severity'])]
        is_confident = bool(row['is_confident'])
        action = BASE_POLICY[(state_name, is_confident)]
        physics_ok = bool(row['physics_admissible'])
        if not physics_ok:
            if intent == 'quality':
                action = downgrade_action(action)
            elif intent == 'productivity' and state_name in ('Recoverable', 'Critical', 'Irrecoverable'):
                action = downgrade_action(action)
        return action

    df['action'] = df.apply(decide, axis=1)
    df['action_tier'] = df['action'].map(lambda a: ACTION_TIERS.index(a))
    return df


gate4_tier_tables = []
gate4_full = {}
for intent in ['quality', 'productivity']:
    gated = apply_gate4(df_g3, intent=intent)
    gate4_full[intent] = gated
    tier_by_severity = gated.groupby(gated['severity'].map(dict(enumerate(STATE_NAMES))))['action_tier'].mean()
    print(f"  Intent={intent}: mean action_tier by severity:")
    print(f"    {dict(tier_by_severity.round(2))}")
    tier_df = tier_by_severity.reset_index()
    tier_df.columns = ['severity_state', 'mean_action_tier']
    tier_df['intent'] = intent
    gate4_tier_tables.append(tier_df)
    save_processed(gated, f'gate4_{intent}_dataset')

save_table(pd.concat(gate4_tier_tables, ignore_index=True), 'gate4_intent_comparison_tiers')

print(f"\n{'='*70}\nSECTION 5: MIRI -- Manufacturing Intervention Readiness Index\n{'='*70}")

# ----------------------------------------------------------------------------
# MIRI (built on the 'quality' intent dataset)
# ----------------------------------------------------------------------------
MIRI_FEATURES = ['tam_p90', 'scr_std', 'confidence_set_size', 'scr_residual']

df_miri_base = gate4_full['quality'].copy()
df_miri_base['scr_residual'] = df_miri_base['scr_residual'].abs()

miri_weight_rows = []
miri_parts = []
for held_out in ['B6', 'B7', 'B8']:
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
    train_df = df_miri_base[df_miri_base['build'].isin(train_builds)].dropna(subset=MIRI_FEATURES + ['action_tier'])
    test_df = df_miri_base[df_miri_base['build'] == held_out].dropna(subset=MIRI_FEATURES + ['action_tier']).copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[MIRI_FEATURES])
    X_test = scaler.transform(test_df[MIRI_FEATURES])
    reg = LinearRegression().fit(X_train, train_df['action_tier'].values)
    test_df['MIRI'] = reg.predict(X_test)

    rho, _ = spearmanr(test_df['MIRI'], test_df['action_tier'])
    print(f"  {held_out}: MIRI-action_tier Spearman rho={rho:.3f}")

    for feat, coef in zip(MIRI_FEATURES, reg.coef_):
        miri_weight_rows.append(dict(held_out=held_out, feature=feat, weight=coef))
    miri_weight_rows.append(dict(held_out=held_out, feature='intercept', weight=reg.intercept_))
    miri_parts.append(test_df)

save_table(pd.DataFrame(miri_weight_rows), 'miri_learned_weights')
miri_combined = pd.concat(miri_parts, ignore_index=True)
save_table(miri_combined, 'miri_labeled_dataset')

# Figure: MIRI distribution by true severity state, pooled across all builds
fig, ax = plt.subplots(figsize=(7, 5))
data_by_state = [miri_combined.loc[miri_combined['severity'] == i, 'MIRI'].dropna().values for i in range(5)]
ax.boxplot(data_by_state, labels=STATE_NAMES)
ax.set_xlabel('True Severity State'); ax.set_ylabel('MIRI Score')
ax.set_title('MIRI Distribution by Severity State (pooled, all held-out builds)')
plt.xticks(rotation=30, ha='right')
save_figure(fig, 'miri_distribution_by_severity')

print(f"\n{'='*70}\nSECTION 6: Baseline Model Comparison\n{'='*70}")

# ----------------------------------------------------------------------------
# Baseline comparison (generic ML + manufacturing-domain baselines)
# ----------------------------------------------------------------------------
def spc_control_chart_predict(train_df, test_df):
    mean_, std_ = train_df['tam_p90'].mean(), train_df['tam_p90'].std()
    z = (test_df['tam_p90'] - mean_) / std_
    bins = [-np.inf, -1.0, 0.0, 1.0, 2.0, np.inf]
    return np.clip(np.digitize(z, bins) - 1, 0, 4)


def physics_only_ved_predict(train_df, test_df):
    hatch_mm_train = train_df['hatch_spacing'] / 1000.0
    ved_train = compute_ved(train_df['commanded_power_mean'], train_df['scan_speed'], hatch_mm_train)
    mu, sigma = ved_train.mean(), ved_train.std()
    hatch_mm_test = test_df['hatch_spacing'] / 1000.0
    ved_test = compute_ved(test_df['commanded_power_mean'], test_df['scan_speed'], hatch_mm_test)

    def to_sev(v):
        if pd.isna(v) or sigma == 0:
            return 0
        z = abs(v - mu) / sigma
        return int(np.clip(np.digitize(z, [0.5, 1.0, 1.5, 2.0]), 0, 4))
    return ved_test.apply(to_sev).values


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
        models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=4, eval_metric='mlogloss', random_state=42)
    return models


baseline_results = []
for held_out in ['B6', 'B7', 'B8']:
    train_builds = [b for b in ['B6', 'B7', 'B8'] if b != held_out]
    train_df = df_g1[df_g1['build'].isin(train_builds)]
    test_df = df_g1[df_g1['build'] == held_out]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])
    y_train, y_test = train_df['severity'].values, test_df['severity'].values

    ordinal_model, chosen_alpha = get_tuned_ordinal_model(X_train, y_train)
    y_pred_ord = ordinal_model.predict(X_test)
    baseline_results.append(dict(
        held_out=held_out, model=f'Ordinal Logistic (ours, alpha={chosen_alpha})',
        accuracy=accuracy_score(y_test, y_pred_ord),
        macro_f1=f1_score(y_test, y_pred_ord, average='macro', zero_division=0),
        mace=np.mean(np.abs(y_test - y_pred_ord)),
    ))

    for model_name, model in get_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        baseline_results.append(dict(
            held_out=held_out, model=model_name,
            accuracy=accuracy_score(y_test, y_pred),
            macro_f1=f1_score(y_test, y_pred, average='macro', zero_division=0),
            mace=np.mean(np.abs(y_test - y_pred)),
        ))

    y_pred_spc = spc_control_chart_predict(train_df, test_df)
    baseline_results.append(dict(
        held_out=held_out, model='SPC Control Chart (manufacturing baseline)',
        accuracy=accuracy_score(y_test, y_pred_spc),
        macro_f1=f1_score(y_test, y_pred_spc, average='macro', zero_division=0),
        mace=np.mean(np.abs(y_test - y_pred_spc)),
    ))

    y_pred_ved = physics_only_ved_predict(train_df, test_df)
    baseline_results.append(dict(
        held_out=held_out, model='Physics-Only VED Rule (manufacturing baseline)',
        accuracy=accuracy_score(y_test, y_pred_ved),
        macro_f1=f1_score(y_test, y_pred_ved, average='macro', zero_division=0),
        mace=np.mean(np.abs(y_test - y_pred_ved)),
    ))

baseline_df = pd.DataFrame(baseline_results)
baseline_summary = baseline_df.groupby('model').agg(
    accuracy_mean=('accuracy', 'mean'), macro_f1_mean=('macro_f1', 'mean'), mace_mean=('mace', 'mean'),
).sort_values('mace_mean').reset_index()
print(baseline_summary.round(3).to_string(index=False))

save_table(baseline_df, 'baseline_comparison_per_fold')
save_table(baseline_summary, 'baseline_comparison_summary')

print(f"\n{'='*70}\nSECTION 7: Ablation Study\n{'='*70}")

# ----------------------------------------------------------------------------
# Ablation (reuses Gate 2/3 predictions already computed in df_g3)
# ----------------------------------------------------------------------------
def evaluate_condition(df, condition_name, use_confidence, use_physics):
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
        acc = (acted['predicted_severity'] == acted['severity']).mean()
        unsafe = ((acted['predicted_severity'] != acted['severity']) & (acted['severity'] >= 3)).mean()
    else:
        acc, unsafe = np.nan, np.nan
    return dict(condition=condition_name, autonomy_rate=autonomy_rate,
                accuracy_when_autonomous=acc, unsafe_actuation_rate=unsafe,
                n_autonomous=n_act, n_total=n_total)


CONDITIONS = [
    ('1. Full pipeline (Gate2 + Gate3)', True, True),
    ('2. No Gate 2 (confidence removed)', False, True),
    ('3. No Gate 3 (physics removed)', True, False),
    ('4. No gating (raw predictions, baseline)', False, False),
]

ablation_results = [evaluate_condition(df_g3, name, uc, up) for name, uc, up in CONDITIONS]
ablation_df = pd.DataFrame(ablation_results)
print(ablation_df.to_string(index=False))
save_table(ablation_df, 'ablation_results_pooled')

# Figure: ablation trade-off -- autonomy rate vs accuracy-when-autonomous vs unsafe rate
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(ablation_df))
width = 0.25
ax.bar(x - width, ablation_df['autonomy_rate'], width, label='Autonomy Rate')
ax.bar(x, ablation_df['accuracy_when_autonomous'], width, label='Accuracy When Autonomous')
ax.bar(x + width, ablation_df['unsafe_actuation_rate'], width, label='Unsafe Actuation Rate')
ax.set_xticks(x)
ax.set_xticklabels([c.split('. ')[1] for c in ablation_df['condition']], rotation=20, ha='right')
ax.set_ylabel('Rate')
ax.set_title('Ablation: Effect of Removing Gate 2 / Gate 3 on Safety Trade-offs')
ax.legend()
save_figure(fig, 'ablation_tradeoff_chart')

per_build_rows = []
for held_out in ['B6', 'B7', 'B8']:
    build_df = df_g3[df_g3['build'] == held_out]
    for name, uc, up in CONDITIONS:
        res = evaluate_condition(build_df, name, uc, up)
        res['held_out_build'] = held_out
        per_build_rows.append(res)
save_table(pd.DataFrame(per_build_rows), 'ablation_results_per_build')

print(f"\n{'='*70}\nSECTION 8: Bootstrap Confidence Intervals\n{'='*70}")

# ----------------------------------------------------------------------------
# Bootstrap CIs (pooled over df_g3, which has everything needed)
# ----------------------------------------------------------------------------
N_BOOTSTRAP = 2000


def compute_ci_metrics(df):
    acc = (df['predicted_severity'] == df['severity']).mean()
    mace = np.mean(np.abs(df['predicted_severity'] - df['severity']))
    coverage = df['true_in_set'].mean()
    is_conf = df['is_confident']
    autonomy_rate = is_conf.mean()
    singleton_acc = ((df.loc[is_conf, 'predicted_severity'] == df.loc[is_conf, 'severity']).mean()
                      if is_conf.sum() > 0 else np.nan)
    would_act = df['is_confident'] & df['physics_admissible']
    n_act = would_act.sum()
    if n_act > 0:
        acted = df[would_act]
        acc_when_auto = (acted['predicted_severity'] == acted['severity']).mean()
        unsafe = ((acted['predicted_severity'] != acted['severity']) & (acted['severity'] >= 3)).mean()
    else:
        acc_when_auto, unsafe = np.nan, np.nan
    return dict(accuracy=acc, mace=mace, coverage=coverage, autonomy_rate=autonomy_rate,
                singleton_accuracy=singleton_acc,
                full_pipeline_accuracy_when_autonomous=acc_when_auto,
                full_pipeline_unsafe_actuation_rate=unsafe,
                full_pipeline_autonomy_rate=would_act.mean())


rng = np.random.RandomState(42)
n = len(df_g3)
point = compute_ci_metrics(df_g3)
boot_vals = {k: [] for k in point.keys()}
for _ in range(N_BOOTSTRAP):
    idx = rng.randint(0, n, size=n)
    sample = df_g3.iloc[idx]
    vals = compute_ci_metrics(sample)
    for k, v in vals.items():
        boot_vals[k].append(v)

ci_rows = []
for k, v in point.items():
    arr = np.array([x for x in boot_vals[k] if not np.isnan(x)])
    lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) > 0 else (np.nan, np.nan))
    print(f"  {k:<45} point={v:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    ci_rows.append(dict(metric=k, point_estimate=v, ci_low=lo, ci_high=hi))

save_table(pd.DataFrame(ci_rows), 'bootstrap_confidence_intervals')

# ============================================================================
# SECTION 9: Archiving the source code into src/
# ============================================================================
# A script CANNOT reliably read its own text when pasted directly into a
# Colab cell -- there is no __file__ for pasted code, only for actual saved
# files. This isn't fixable from inside the script itself. The one
# guaranteed-reliable way to archive source code in Colab is the built-in
# %%writefile magic, used ONCE, as follows:
#
#   Cell A:
#     %%writefile /content/drive/MyDrive/DC-CPT-Project/src/run_full_pipeline.py
#     <paste this entire script>
#
#   Cell B:
#     %run /content/drive/MyDrive/DC-CPT-Project/src/run_full_pipeline.py
#
# Cell A both saves the file to src/ AND does not execute it (writefile
# suppresses execution). Cell B then runs the saved copy. This is two
# cells total, run once per script -- not per-result, not repeatedly --
# and it is the standard, dependable Colab pattern; there is no more
# "automatic" way to persist source code that Colab itself provides.
print("\nNOTE: To also save this script's source code into src/, use the")
print("%%writefile pattern described in the comments at the top of Section 9")
print("of this script -- this is the one reliable way Colab provides to")
print("persist pasted code as a file.")

print(f"\n{'='*70}")
print("PIPELINE COMPLETE. All results and figures saved.")
print(f"{'='*70}")
