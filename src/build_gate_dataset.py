"""
DC-CPT Data Loader
Joins TAM (severity), SCR (physics), thermocouples (ground-truth), and
scan-strategy (process parameters) into one aligned per-layer table.

Run this in Colab with Drive mounted. Output: one CSV per build in
Data/processed/, ready to feed Gate 1 (state) and Gate 3 (physics) models.
"""

import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

BASE = '/content/drive/MyDrive/DC-CPT-Project/Data/Raw'
OUT_DIR = '/content/drive/MyDrive/DC-CPT-Project/Data/processed'
os.makedirs(OUT_DIR, exist_ok=True)

BUILDS = ['B6', 'B7', 'B8']

# TAM threshold above which a pixel counts as "overheating" for severity scoring.
# T_melt used by NIST was 1298C; TAM values here are fractional time-above-melt,
# so treat the 90th percentile of each layer's own distribution as a starting
# severity cut. Revisit this once you're calibrating Gate 1 for real.
TAM_SEVERITY_PCTL = 90


def compute_global_tam_threshold(tam_ds, n_sample_layers=30):
    """Fixed severity threshold, computed once across a sample of layers
    from this build, so 'fraction above threshold' is comparable layer
    to layer instead of trivially ~10% by construction."""
    n_layers = tam_ds.shape[0]
    sample_idx = np.linspace(0, n_layers - 1, min(n_sample_layers, n_layers)).astype(int)
    pooled = []
    for i in sample_idx:
        layer = tam_ds[i, :, :]
        valid = layer[~np.isnan(layer)]
        if valid.size:
            pooled.append(valid)
    pooled = np.concatenate(pooled) if pooled else np.array([0.0])
    return float(np.percentile(pooled, TAM_SEVERITY_PCTL))


def load_thermal_summary(build):
    """Extract per-layer summary stats from TAM and SCR files."""
    tam_path = f"{BASE}/AMB2022-01-718-AMMT-{build}-StaringCamera_TAM.h5"
    scr_path = f"{BASE}/AMB2022-01-718-AMMT-{build}-StaringCamera_SCR.h5"

    rows = []
    with h5py.File(tam_path, 'r') as ftam, h5py.File(scr_path, 'r') as fscr:
        tam_ds = ftam['ThermalData']['TAM']
        scr_ds = fscr['ThermalData']['SCR']
        n_layers = tam_ds.shape[0]
        global_tam_threshold = compute_global_tam_threshold(tam_ds)

        # pull build-level process parameters once
        build_key = [k for k in ftam.keys() if k.startswith('AMB2022')][0]
        attrs = ftam[build_key].attrs
        laser_power = float(np.ravel(attrs.get('laser_power', [np.nan]))[0])
        scan_speed = float(np.ravel(attrs.get('scan_speed', [np.nan]))[0])
        hatch_spacing = float(np.ravel(attrs.get('hatch_spacing', [np.nan]))[0])

        for layer in range(n_layers):
            tam_layer = tam_ds[layer, :, :]
            scr_layer = scr_ds[layer, :, :]

            tam_valid = tam_layer[~np.isnan(tam_layer)]
            scr_valid = scr_layer[~np.isnan(scr_layer)]

            if tam_valid.size == 0:
                # empty layer (e.g. before build starts / after it ends)
                rows.append({
                    'build': build, 'layer': layer,
                    'tam_mean': np.nan, 'tam_max': np.nan, 'tam_p90': np.nan,
                    'tam_frac_above_global_threshold': np.nan,
                    'scr_mean': np.nan, 'scr_std': np.nan,
                    'scr_min': np.nan, 'scr_max': np.nan,
                    'laser_power': laser_power, 'scan_speed': scan_speed,
                    'hatch_spacing': hatch_spacing,
                })
                continue

            tam_p90 = np.percentile(tam_valid, TAM_SEVERITY_PCTL)  # descriptive, per-layer

            rows.append({
                'build': build,
                'layer': layer,
                'tam_mean': float(tam_valid.mean()),
                'tam_max': float(tam_valid.max()),
                'tam_p90': float(tam_p90),
                # this now genuinely varies layer-to-layer, since the threshold
                # is fixed across the build rather than redefined every layer
                'tam_frac_above_global_threshold': float((tam_valid > global_tam_threshold).mean()),
                'scr_mean': float(scr_valid.mean()) if scr_valid.size else np.nan,
                'scr_std': float(scr_valid.std()) if scr_valid.size else np.nan,
                'scr_min': float(scr_valid.min()) if scr_valid.size else np.nan,
                'scr_max': float(scr_valid.max()) if scr_valid.size else np.nan,
                'laser_power': laser_power,
                'scan_speed': scan_speed,
                'hatch_spacing': hatch_spacing,
            })

    return pd.DataFrame(rows)


def load_scan_strategy_summary(n_layers_needed):
    """Extract per-layer commanded power and estimated layer duration
    from the shared scan-strategy file (same file covers all builds)."""
    xypt_path = f"{BASE}/AMB2022-01-AMMT-XYPT_v1.h5"
    rows = []
    with h5py.File(xypt_path, 'r') as f:
        group = f['XYPT']
        digital_rate = float(np.ravel(group.attrs.get('digital_rate', [np.nan]))[0])

        layer_keys = sorted(
            [k for k in group.keys() if k.isdigit()],
            key=lambda x: int(x)
        )

        for lk in layer_keys:
            layer_idx = int(lk)
            if layer_idx >= n_layers_needed:
                continue
            g = group[lk]
            p = g['P'][:]
            n_points = p.size  # use total element count, not shape[0] --
                                 # MATLAB-exported vectors often come out as (1, N)
            rows.append({
                'layer': layer_idx,
                'commanded_power_mean': float(np.mean(p)),
                'commanded_power_max': float(np.max(p)),
                'n_scan_points': int(n_points),
                'est_layer_duration_s': float(n_points / digital_rate) if digital_rate else np.nan,
            })
    return pd.DataFrame(rows)


def load_thermocouple(build):
    """Load thermocouple CSV, handling missing P3 column (e.g. build B8)."""
    tc_path = f"{BASE}/AMB2022-01-AMMT-{build}-Thermocouple.csv"
    df = pd.read_csv(tc_path)
    if 'P3' not in df.columns:
        df['P3'] = np.nan  # keep schema consistent across builds
    df = df[['Time', 'P2', 'P3', 'Chamber']]
    return df


def estimate_layer_times(build_datetime_str, scan_summary):
    """Approximate wall-clock time at the start of each layer by
    accumulating estimated layer durations from the build start time.
    This is an approximation -- good enough for coarse alignment to
    1Hz thermocouple data, not for frame-level sync."""
    build_start = datetime.strptime(build_datetime_str, '%d-%b-%Y %H:%M:%S')
    times = []
    cursor = build_start
    for _, row in scan_summary.sort_values('layer').iterrows():
        times.append({'layer': row['layer'], 'est_time': cursor})
        dur = row['est_layer_duration_s']
        cursor = cursor + timedelta(seconds=dur if not np.isnan(dur) else 0)
    return pd.DataFrame(times)


def nearest_thermocouple_reading(est_time, tc_df, build_date):
    """Find the thermocouple row closest in time to est_time."""
    target = est_time.time()
    tc_df = tc_df.copy()
    tc_df['_t'] = pd.to_datetime(tc_df['Time'], format='%H:%M:%S').dt.time
    tc_df['_diff'] = tc_df['_t'].apply(
        lambda t: abs(datetime.combine(build_date, t) - datetime.combine(build_date, target)).total_seconds()
    )
    nearest = tc_df.loc[tc_df['_diff'].idxmin()]
    return nearest['P2'], nearest['P3'], nearest['Chamber']


def build_dataset_for(build):
    print(f"Processing {build}...")

    thermal = load_thermal_summary(build)
    n_layers = thermal['layer'].nunique()

    scan = load_scan_strategy_summary(n_layers)
    merged = thermal.merge(scan, on='layer', how='left')

    # get build_datetime from TAM file attrs for time estimation
    tam_path = f"{BASE}/AMB2022-01-718-AMMT-{build}-StaringCamera_TAM.h5"
    with h5py.File(tam_path, 'r') as f:
        build_key = [k for k in f.keys() if k.startswith('AMB2022')][0]
        build_dt_str = f[build_key].attrs.get('Build_datetime', None)
        if isinstance(build_dt_str, bytes):
            build_dt_str = build_dt_str.decode()

    tc_df = load_thermocouple(build)

    if build_dt_str:
        layer_times = estimate_layer_times(build_dt_str, scan)
        merged = merged.merge(layer_times, on='layer', how='left')

        build_date = datetime.strptime(build_dt_str, '%d-%b-%Y %H:%M:%S').date()
        p2_list, p3_list, chamber_list = [], [], []
        for est_time in merged['est_time']:
            if pd.isna(est_time):
                p2_list.append(np.nan); p3_list.append(np.nan); chamber_list.append(np.nan)
                continue
            p2, p3, ch = nearest_thermocouple_reading(est_time, tc_df, build_date)
            p2_list.append(p2); p3_list.append(p3); chamber_list.append(ch)
        merged['thermocouple_P2'] = p2_list
        merged['thermocouple_P3'] = p3_list
        merged['thermocouple_Chamber'] = chamber_list
    else:
        print(f"  Warning: no Build_datetime found for {build}, skipping thermocouple alignment")

    out_path = f"{OUT_DIR}/{build}_gate_dataset.csv"
    merged.to_csv(out_path, index=False)
    print(f"  Saved {len(merged)} layers -> {out_path}")
    return merged


if __name__ == '__main__':
    all_builds = {}
    for b in BUILDS:
        all_builds[b] = build_dataset_for(b)

    combined = pd.concat(all_builds.values(), ignore_index=True)
    combined_path = f"{OUT_DIR}/all_builds_gate_dataset.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined dataset: {len(combined)} rows -> {combined_path}")
    print(combined.head())
