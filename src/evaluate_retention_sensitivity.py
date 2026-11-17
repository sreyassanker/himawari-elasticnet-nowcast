#!/usr/bin/env python3
"""
Sensitivity analysis of Elastic Net sparsity and performance across OLS retention thresholds (90%, 95%, 99%).
Uses Gram-matrix coordinate descent from train_nowcast.py for efficient convergence.

Outputs:
  - results/retention_sensitivity.json: Sparsity and R2 metrics across retention thresholds
"""
import sys, json, time
from pathlib import Path
import numpy as np

# Import exact Gram-matrix coordinate descent solver
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nowcast import solve_enet_gram, L1_RATIO

BANDS = ['B08','B09','B10','B11','B12','B13','B14','B15','B16']
LEADS = [20, 30, 40, 50, 60, 70, 80, 90]
LEAD_FRAMES = [l // 10 for l in LEADS]
N_BANDS = len(BANDS)
N_LEADS = len(LEADS)

train_dates = ['20230602','20230611','20230617','20230618','20230625','20230626','20230627']
data_dir = Path('data/training')
out_dir = Path('results')
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading training data...", flush=True)
t0 = time.time()
all_data = {b: [] for b in BANDS}
for d in train_dates:
    arr = np.load(data_dir / f'{d}.npz')
    for b in BANDS:
        all_data[b].append(arr[b])
for b in BANDS:
    all_data[b] = np.concatenate(all_data[b], axis=0)
X_cube = np.stack([all_data[b] for b in BANDS], axis=-1).astype(np.float32)
del all_data

print(f"Loaded {len(train_dates)} days in {time.time()-t0:.2f}s | Shape: {X_cube.shape}", flush=True)

alphas = [10.0, 5.0, 3.0, 1.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001]
retention_thresholds = [0.90, 0.95, 0.99]
results_sens = {th: {'nz': [], 'r2': []} for th in retention_thresholds}

t_start = time.time()
for li, (lead, nf) in enumerate(zip(LEADS, LEAD_FRAMES)):
    t_lead = time.time()
    X_flat = X_cube[:-nf].reshape(-1, N_BANDS)
    y_all = X_cube[nf:].reshape(-1, N_BANDS)
    valid = ~(np.isnan(X_flat).any(axis=1) | np.isnan(y_all).any(axis=1))
    X_v = X_flat[valid].astype(np.float64)
    y_v = y_all[valid].astype(np.float64)
    N = len(X_v)

    # Standardize predictors and targets
    x_mean = np.mean(X_v, axis=0)
    x_std = np.std(X_v, axis=0)
    y_mean = np.mean(y_v, axis=0)
    y_std = np.std(y_v, axis=0)

    X_s = (X_v - x_mean) / x_std
    Y_s = (y_v - y_mean) / y_std

    # Exact Gram matrix G = X_s^T X_s and cross-products Xy_mat = X_s^T Y_s
    G = np.dot(X_s.T, X_s)
    Xy_mat = np.dot(X_s.T, Y_s)

    beta_ols = np.linalg.solve(G, Xy_mat)

    for bi, b in enumerate(BANDS):
        b_ols = beta_ols[:, bi]
        res_ols = N - 2.0 * np.dot(b_ols, Xy_mat[:, bi]) + np.dot(b_ols, np.dot(G, b_ols))
        r2_ols = float(max(0.0, 1.0 - res_ols / N))

        path_fits = []
        beta_warm = None
        for alpha in alphas:
            b_enet = solve_enet_gram(G, Xy_mat[:, bi], N, alpha=alpha, l1_ratio=L1_RATIO, beta_init=beta_warm)
            beta_warm = b_enet.copy()
            res_enet = N - 2.0 * np.dot(b_enet, Xy_mat[:, bi]) + np.dot(b_enet, np.dot(G, b_enet))
            r2 = float(max(0.0, 1.0 - res_enet / N))
            nz = int(np.sum(np.abs(b_enet) > 1e-5))
            path_fits.append((alpha, nz, r2))

        for th in retention_thresholds:
            cutoff = r2_ols * th
            qualifying = [p for p in path_fits if p[1] > 0 and p[2] >= cutoff]
            if qualifying:
                min_nz = min(p[1] for p in qualifying)
                best_fit = max([p for p in qualifying if p[1] == min_nz], key=lambda x: x[2])
                results_sens[th]['nz'].append(best_fit[1])
                results_sens[th]['r2'].append(best_fit[2])
            else:
                results_sens[th]['nz'].append(9)
                results_sens[th]['r2'].append(r2_ols)

    print(f"  Lead {lead:2d} min done in {time.time()-t_lead:.2f}s", flush=True)

print(f"\nAll 8 leads completed in {time.time()-t_start:.2f}s!")

# Save to JSON
out_file = out_dir / 'retention_sensitivity.json'
json_out = {
    str(int(th*100)): {
        'mean_nz': float(np.mean(results_sens[th]['nz'])),
        'mean_r2': float(np.mean(results_sens[th]['r2'])),
        'nz_counts': {
            '1': int((np.array(results_sens[th]['nz']) == 1).sum()),
            '2': int((np.array(results_sens[th]['nz']) == 2).sum()),
            '3': int((np.array(results_sens[th]['nz']) == 3).sum()),
            '4+': int((np.array(results_sens[th]['nz']) >= 4).sum()),
        }
    } for th in retention_thresholds
}

with open(out_file, 'w') as f:
    json.dump(json_out, f, indent=2)

print("\n=== SENSITIVITY TESTING OF RETENTION THRESHOLD ===")
for th in retention_thresholds:
    nz_arr = np.array(results_sens[th]['nz'])
    r2_arr = np.array(results_sens[th]['r2'])
    print(f"Retention Threshold {th*100:.0f}% of OLS R2:")
    print(f"  Mean nonzero predictors: {np.mean(nz_arr):.2f} (min: {np.min(nz_arr)}, max: {np.max(nz_arr)})")
    print(f"  Nonzero counts: 1 pred: {(nz_arr==1).sum()}, 2 pred: {(nz_arr==2).sum()}, 3 pred: {(nz_arr==3).sum()}, 4+ pred: {(nz_arr>=4).sum()}")
    print(f"  Mean R2 across all 72 equations: {np.mean(r2_arr):.4f}")
print(f"\nSaved results to {out_file}")
