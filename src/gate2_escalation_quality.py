"""
Escalation-quality diagnostic: for layers where the model did NOT produce
a singleton (i.e. escalated), what does the confidence set actually
contain? Adjacent-state sets (e.g. {Recoverable, Critical}) indicate
well-calibrated, sensible ordinal ambiguity. Scattered/non-adjacent sets
(e.g. {Stable, Irrecoverable}) would indicate the model is just noisy,
not meaningfully uncertain -- an important distinction for the paper.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mapie.classification import SplitConformalClassifier
import mord
from collections import Counter

DATA_PATH = '/content/drive/MyDrive/DC-CPT-Project/Data/processed/gate1_labeled_dataset.csv'
STATE_NAMES = ['Stable', 'Degrading', 'Recoverable', 'Critical', 'Irrecoverable']
FEATURE_COLS = [
    'tam_mean', 'tam_max', 'tam_p90',
    'scr_mean', 'scr_std', 'scr_min', 'scr_max',
    'scan_speed', 'hatch_spacing',
]

df = pd.read_csv(DATA_PATH)

overall_adjacent = 0
overall_nonadjacent = 0
nonadjacent_examples = []

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
    is_multi = (set_sizes > 1)

    print(f"\n=== {held_out}: escalated (non-singleton) layers = {is_multi.sum()} of {len(y_test)} ===")

    set_composition_counter = Counter()
    adjacent_count = 0
    nonadjacent_count = 0
    true_state_in_set_count = 0

    for i in np.where(is_multi)[0]:
        states_in_set = sorted(np.where(set_masks[i])[0])
        names_in_set = tuple(STATE_NAMES[s] for s in states_in_set)
        set_composition_counter[names_in_set] += 1

        # "adjacent" = every state in the set is within 1 ordinal step of
        # its neighbor in the set (i.e. no gaps like Stable+Irrecoverable
        # without the states in between also being included)
        span = states_in_set[-1] - states_in_set[0]
        is_adjacent = (span == len(states_in_set) - 1)  # consecutive integers, no gaps
        if is_adjacent:
            adjacent_count += 1
        else:
            nonadjacent_count += 1
            nonadjacent_examples.append((held_out, names_in_set, STATE_NAMES[y_test[i]]))

        if y_test[i] in states_in_set:
            true_state_in_set_count += 1

    print(f"  Adjacent (sensible ordinal ambiguity) sets: {adjacent_count} ({100*adjacent_count/max(is_multi.sum(),1):.0f}%)")
    print(f"  Non-adjacent (scattered) sets:               {nonadjacent_count} ({100*nonadjacent_count/max(is_multi.sum(),1):.0f}%)")
    print(f"  True state actually contained in set:         {true_state_in_set_count} ({100*true_state_in_set_count/max(is_multi.sum(),1):.0f}%)")
    print(f"  Most common confidence set compositions:")
    for combo, count in set_composition_counter.most_common(5):
        print(f"    {combo}: {count}")

    overall_adjacent += adjacent_count
    overall_nonadjacent += nonadjacent_count

print(f"\n{'='*70}")
print(f"OVERALL: {overall_adjacent} adjacent vs {overall_nonadjacent} non-adjacent escalated sets "
      f"({100*overall_adjacent/(overall_adjacent+overall_nonadjacent):.0f}% adjacent)")
print(f"{'='*70}")

if nonadjacent_examples:
    print("\nNon-adjacent (scattered) examples, for inspection:")
    for build, combo, true_state in nonadjacent_examples[:10]:
        print(f"  {build}: set={combo}, true state was {true_state}")
