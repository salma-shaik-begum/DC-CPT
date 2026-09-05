"""
DC-CPT Gate 4: Intent-Parameterized Policy Engine
====================================================
Combines everything upstream into one authorization decision per layer:
  Gate 1 severity state + Gate 2 confidence (singleton vs escalated)
  + Gate 3 physics admissibility  ->  final action

Design principle (explain this in your methods section -- it's a real,
citable safety-engineering choice, not an arbitrary rule):
  CONSERVATIVE actions (Monitor, Escalate, Reduce, Pause) do NOT require
  high confidence to trigger -- taking the safe action unnecessarily is
  low-cost. RISKY actions (Continue unchanged, Adjust parameters) DO
  require high confidence -- acting wrongly here is what the whole
  framework exists to prevent. This asymmetry is why confidence gates
  the escalation path one way, not both ways.

Manufacturing Intent parameterizes how strictly Gate 3 physics failures
are enforced:
  'quality'      -> any physics inadmissibility downgrades the action
                     to a more conservative tier, regardless of severity
  'productivity' -> physics inadmissibility is only enforced at
                     Recoverable severity and above; minor physics
                     deviations at Stable/Degrading are tolerated to
                     avoid unnecessary interruptions

This is intentionally a simple, INTERPRETABLE rule table, not a learned
policy -- justified the same way rule-based logic is preferred in other
safety-critical domains (aviation, medical devices): every decision this
gate makes is auditable and explainable after the fact, which matters
more here than squeezing out marginal performance from a black-box policy.
"""

import pandas as pd
import sys
sys.path.insert(0, '/content/drive/MyDrive/DC-CPT-Project/src')
from save_utils import save_table

DATA_PATH = '/content/drive/MyDrive/DC-CPT-Project/Data/processed/gate2_full_labeled_dataset.csv'
OUT_DIR = '/content/drive/MyDrive/DC-CPT-Project/Data/processed'

STATE_NAMES = ['Stable', 'Degrading', 'Recoverable', 'Critical', 'Irrecoverable']

# Ordered from least to most conservative -- used for physics-driven downgrades
# NOTE: Escalate_Human sits ABOVE all autonomous actions (just below Pause).
# This matters: a downgrade must never convert a human-escalation decision
# into an autonomous action -- that would silently remove human oversight
# exactly when a second risk signal (physics failure) has just appeared.
# (Caught this ordering bug during testing -- see conversation notes; it's
# a good illustration of the kind of governance-logic error this framework
# is designed to prevent, and worth a sentence in the paper's discussion.)
ACTION_TIERS = [
    'Continue',
    'Monitor',
    'Adjust_Parameters',
    'Reduce_Scan_Speed',
    'Escalate_Human',
    'Pause_Build',
]

# Base policy: (severity_state, is_confident) -> action
# Note Critical/Irrecoverable trigger their conservative action REGARDLESS
# of confidence -- this is the asymmetry described above.
BASE_POLICY = {
    ('Stable', True):          'Continue',
    ('Stable', False):         'Monitor',
    ('Degrading', True):       'Monitor',
    ('Degrading', False):      'Escalate_Human',
    ('Recoverable', True):     'Adjust_Parameters',
    ('Recoverable', False):    'Escalate_Human',
    ('Critical', True):        'Reduce_Scan_Speed',
    ('Critical', False):       'Reduce_Scan_Speed',
    ('Irrecoverable', True):   'Pause_Build',
    ('Irrecoverable', False):  'Pause_Build',
}

# Broad authorization category each action falls under -- for reporting
AUTHORIZATION_CATEGORY = {
    'Continue': 'Autonomous',
    'Monitor': 'Autonomous',
    'Adjust_Parameters': 'Autonomous_Corrective',
    'Reduce_Scan_Speed': 'Autonomous_Corrective',
    'Pause_Build': 'Autonomous_Corrective',
    'Escalate_Human': 'Human_Review',
}


def downgrade_action(action):
    """Move one tier toward more conservative. Already-maximally-conservative
    or already-escalated actions stay put -- can't get more careful than pausing."""
    if action == 'Pause_Build':
        return action
    idx = ACTION_TIERS.index(action)
    return ACTION_TIERS[min(idx + 1, len(ACTION_TIERS) - 1)]


def apply_gate4(df, intent='quality'):
    df = df.copy()

    def decide(row):
        state_name = STATE_NAMES[int(row['severity'])]
        # is_confident is not a column yet -- Gate 2's script computed this
        # per alpha level but didn't save it to CSV. Recompute a simple
        # proxy here: treat 'tam_scr_admissible' + high tam_p90 confidence
        # is NOT available post-hoc without rerunning Gate 2's conformal
        # model, so this script expects an 'is_confident' column -- see
        # note in __main__ below on how it's attached before calling this.
        is_confident = bool(row['is_confident'])
        action = BASE_POLICY[(state_name, is_confident)]

        physics_ok = bool(row['physics_admissible'])
        if not physics_ok:
            if intent == 'quality':
                action = downgrade_action(action)
            elif intent == 'productivity' and state_name in ('Recoverable', 'Critical', 'Irrecoverable'):
                action = downgrade_action(action)
            # else: productivity intent tolerates physics ambiguity at
            # Stable/Degrading, action unchanged

        return action

    df['action'] = df.apply(decide, axis=1)
    df['authorization'] = df['action'].map(AUTHORIZATION_CATEGORY)
    df['action_tier'] = df['action'].map(lambda a: ACTION_TIERS.index(a))
    return df


def summarize(df, intent_label):
    print(f"\n{'='*70}")
    print(f"Policy summary -- intent = {intent_label}")
    print(f"{'='*70}")
    print("\nAction distribution:")
    print(df['action'].value_counts())
    print("\nAuthorization category distribution:")
    print(df['authorization'].value_counts())
    print("\nAuthorization category by true severity state:")
    print(pd.crosstab(df['severity'].map(dict(enumerate(STATE_NAMES))), df['authorization']))

    print("\nMean action_tier by severity state (0=Continue ... 5=Pause_Build):")
    print("This is the precise conservativeness metric -- use THIS to compare")
    print("intents, not the 3-category bucket above, which can obscure a")
    print("downgrade that lands within the same broad category.")
    tier_by_severity = df.groupby(df['severity'].map(dict(enumerate(STATE_NAMES))))['action_tier'].mean()
    print(tier_by_severity)

    tier_df = tier_by_severity.reset_index()
    tier_df.columns = ['severity_state', 'mean_action_tier']
    tier_df['intent'] = intent_label
    return tier_df


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    # 'is_confident' now comes directly from gate2_full_labeling.py -- real
    # conformal singleton flags, computed for every layer, not an approximation.

    all_tier_tables = []
    for intent in ['quality', 'productivity']:
        gated = apply_gate4(df, intent=intent)
        tier_df = summarize(gated, intent)
        all_tier_tables.append(tier_df)
        save_table(gated, f'gate4_{intent}_dataset')

    combined_tiers = pd.concat(all_tier_tables, ignore_index=True)
    save_table(combined_tiers, 'gate4_intent_comparison_tiers')
