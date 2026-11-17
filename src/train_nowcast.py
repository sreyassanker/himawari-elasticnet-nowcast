#!/usr/bin/env python3
"""
Multi-Year Elastic Net Nowcasting Model Training
Trained across 373 days (2023-2026 multi-year archive, ~50.3M valid observations)
Predicts 9 Himawari-9 infrared brightness temperature bands across 8 forecast lead times (20–90 min).

Uses Elastic Net (l1 + l2) regularization to resolve collinear channel-swapping instability
while enforcing the 95% OLS retention sparsity threshold.

Outputs:
  - results/model_coefficients.json: 72 sparse linear equations in physical Kelvin units
  - results/train_diagnostics.json: Fit diagnostics, sparsity counts, and R2 retention metrics
"""

import os
import sys
import glob
import json
import time
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

BANDS = ['B08', 'B09', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16']
LEADS = [20, 30, 40, 50, 60, 70, 80, 90]
LEAD_FRAMES = [l // 10 for l in LEADS]
N_BANDS = len(BANDS)
N_LEADS = len(LEADS)

DATA_DIR = Path('data/training')
OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Decreasing regularization path for warm-start
ALPHAS = [10.0, 5.0, 3.0, 1.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001, 0.0005]
L1_RATIO = 0.85  # Elastic Net mixing: 85% L1 (sparsity) + 15% L2 (collinear grouping)
R2_RETENTION = 0.95  # 95% OLS retention threshold

def solve_enet_gram(G, Xy, N, alpha, l1_ratio, beta_init=None, max_iter=2000, tol=1e-7):
    """
    Exact coordinate descent for Elastic Net using precomputed Gram matrix G = X^T X and Xy = X^T y.
    Computes global minimum in < 0.1 ms per fit.
    """
    p = len(Xy)
    beta = np.zeros(p, dtype=np.float64) if beta_init is None else beta_init.copy()
    l1 = alpha * l1_ratio * N
    l2 = alpha * (1.0 - l1_ratio) * N
    
    for it in range(max_iter):
        max_change = 0.0
        for j in range(p):
            # Compute partial residual correlation: rho_j = x_j^T (y - X_{-j} beta_{-j})
            rho_j = Xy[j] - (np.dot(G[j], beta) - G[j, j] * beta[j])
            
            # Soft-thresholding operator with L2 curvature scaling
            if rho_j > l1:
                new_beta = (rho_j - l1) / (G[j, j] + l2)
            elif rho_j < -l1:
                new_beta = (rho_j + l1) / (G[j, j] + l2)
            else:
                new_beta = 0.0
                
            change = abs(new_beta - beta[j])
            if change > max_change:
                max_change = change
            beta[j] = new_beta
            
        if max_change < tol:
            break
            
    return beta

def main():
    print("=" * 80)
    print(" HIMAWARI-9 MULTI-YEAR ELASTIC NET NOWCASTING MODEL TRAINING")
    print("=" * 80)
    
    # 1. Identify training days (complete South Asia archive: 2023-2026, all 373 days)
    all_files = sorted(glob.glob(str(DATA_DIR / '*.npz')))
    train_files = all_files
    train_dates = [os.path.basename(f).replace('.npz', '') for f in train_files]
    
    print(f"Loading complete South Asia archive: {len(train_dates)} training days (2023-2026)...", flush=True)
    t0 = time.time()
    cube_list = []
    for f in train_files:
        d = np.load(f)
        # Stack 9 bands as float32: shape (144, 1000, 9)
        day_bands = np.stack([d[b] for b in BANDS], axis=-1).astype(np.float32)
        cube_list.append(day_bands)
        
    X_cube = np.concatenate(cube_list, axis=0)
    del cube_list
    
    N_TIMES, N_PIXELS, _ = X_cube.shape
    print(f"Successfully loaded {len(train_dates)} days in {time.time()-t0:.2f}s.")
    print(f"Cube Shape: ({N_TIMES} timestamps, {N_PIXELS} pixels, {N_BANDS} bands)", flush=True)
    
    results = {}
    diagnostics = {
        "model_type": "ElasticNet",
        "l1_ratio": L1_RATIO,
        "r2_retention_threshold": R2_RETENTION,
        "train_years": [2023, 2024, 2025, 2026],
        "n_days": len(train_dates),
        "n_times": int(N_TIMES),
        "n_pixels": int(N_PIXELS),
        "train_dates_sample": [train_dates[0], train_dates[len(train_dates)//2], train_dates[-1]],
        "leads": {}
    }
    
    r2_ols = np.zeros((N_LEADS, N_BANDS))
    r2_enet = np.zeros((N_LEADS, N_BANDS))
    nz_counts = np.zeros((N_LEADS, N_BANDS), dtype=int)
    
    print(f"\nTraining 9 bands x 8 leads = 72 Elastic Net equations (L1 ratio={L1_RATIO})...", flush=True)
    
    for li, (lead, nf) in enumerate(zip(LEADS, LEAD_FRAMES)):
        t_lead = time.time()
        # Predictors at time t, Targets at time t + nf
        X_flat = X_cube[:-nf].reshape(-1, N_BANDS)
        y_all = X_cube[nf:].reshape(-1, N_BANDS)
        n_rows = len(X_flat)
        chunk_size = 2000000

        # Pass 1: Streaming mean across valid observations
        sum_x = np.zeros(N_BANDS, dtype=np.float64)
        sum_y = np.zeros(N_BANDS, dtype=np.float64)
        N = 0
        for s in range(0, n_rows, chunk_size):
            e = min(s + chunk_size, n_rows)
            xc = X_flat[s:e]
            yc = y_all[s:e]
            v = ~(np.isnan(xc).any(axis=1) | np.isnan(yc).any(axis=1))
            if not np.any(v):
                continue
            xc_v = xc[v].astype(np.float64)
            yc_v = yc[v].astype(np.float64)
            sum_x += np.sum(xc_v, axis=0)
            sum_y += np.sum(yc_v, axis=0)
            N += len(xc_v)

        x_mean = sum_x / N
        y_mean = sum_y / N

        # Pass 2: Streaming centered Gram matrix G and cross-product Xy
        cov_XX = np.zeros((N_BANDS, N_BANDS), dtype=np.float64)
        cov_XY = np.zeros((N_BANDS, N_BANDS), dtype=np.float64)
        sum_sq_x = np.zeros(N_BANDS, dtype=np.float64)
        sum_sq_y = np.zeros(N_BANDS, dtype=np.float64)

        for s in range(0, n_rows, chunk_size):
            e = min(s + chunk_size, n_rows)
            xc = X_flat[s:e]
            yc = y_all[s:e]
            v = ~(np.isnan(xc).any(axis=1) | np.isnan(yc).any(axis=1))
            if not np.any(v):
                continue
            xc_v = xc[v].astype(np.float64) - x_mean
            yc_v = yc[v].astype(np.float64) - y_mean
            sum_sq_x += np.sum(xc_v**2, axis=0)
            sum_sq_y += np.sum(yc_v**2, axis=0)
            cov_XX += np.dot(xc_v.T, xc_v)
            cov_XY += np.dot(xc_v.T, yc_v)

        x_std = np.sqrt(sum_sq_x / N)
        y_std = np.sqrt(sum_sq_y / N)

        # Standardized Gram matrix G = Xs^T Xs and cross-product Xy_mat = Xs^T Ys
        G = cov_XX / np.outer(x_std, x_std)
        Xy_mat = cov_XY / np.outer(x_std, y_std)
        
        n_boundary_pairs = int((len(train_dates) - 1) * (nf - 1))
        diagnostics["leads"][str(lead)] = {
            "n_frames": nf,
            "candidate_pairs": int((N_TIMES - nf) * N_PIXELS),
            "valid_pairs": int(N),
            "masked_pairs": int((N_TIMES - nf) * N_PIXELS - N),
            "cross_midnight_pairs": n_boundary_pairs,
            "cross_midnight_fraction": float(n_boundary_pairs / N if N else 0.0),
        }
        
        # OLS reference solution: G * beta = Xy_mat
        beta_ols = np.linalg.solve(G, Xy_mat)
        
        for bi, b in enumerate(BANDS):
            b_ols = beta_ols[:, bi]
            
            # OLS R2 in standardized space: R2 = 1 - ||y_s - Xs*beta||^2 / ||y_s||^2
            # Notice ||y_s||^2 = N, and ||y_s - Xs*beta||^2 = N - 2*beta^T*Xy + beta^T*G*beta
            res_ols = N - 2.0 * np.dot(b_ols, Xy_mat[:, bi]) + np.dot(b_ols, np.dot(G, b_ols))
            r2_olsv = float(max(0.0, 1.0 - res_ols / N))
            r2_ols[li, bi] = r2_olsv
            threshold = r2_olsv * R2_RETENTION
            
            best_r2 = None
            best_coefs_std = None
            best_nz = None
            best_alpha = None
            
            # Regularization path search
            beta_warm = None
            for alpha in ALPHAS:
                b_enet = solve_enet_gram(G, Xy_mat[:, bi], N, alpha=alpha, l1_ratio=L1_RATIO, beta_init=beta_warm)
                beta_warm = b_enet.copy()
                
                # Compute R2 from Gram matrix in O(1) time
                res_enet = N - 2.0 * np.dot(b_enet, Xy_mat[:, bi]) + np.dot(b_enet, np.dot(G, b_enet))
                r2 = float(max(0.0, 1.0 - res_enet / N))
                nz = int(np.sum(np.abs(b_enet) > 1e-5))
                
                if nz > 0 and r2 >= threshold:
                    if (best_nz is None) or (nz < best_nz) or (nz == best_nz and r2 > best_r2):
                        best_r2 = r2
                        best_coefs_std = b_enet.copy()
                        best_nz = nz
                        best_alpha = alpha
                        
            if best_nz is None:
                # Fallback to OLS if no sparse fit meets 95% retention
                best_r2 = r2_olsv
                best_coefs_std = b_ols.copy()
                best_nz = int(np.sum(np.abs(b_ols) > 1e-5))
                best_alpha = None
                
            r2_enet[li, bi] = best_r2
            nz_counts[li, bi] = best_nz
            
            # Transform coefficients back to physical raw Kelvin units:
            # y = y_mean + y_std * sum_j [ beta_std_j * (x_j - x_mean_j) / x_std_j ]
            # raw_coef_j = beta_std_j * y_std / x_std_j
            # raw_intercept = y_mean - sum_j [ raw_coef_j * x_mean_j ]
            raw_coefs = best_coefs_std * (y_std[bi] / x_std)
            raw_intercept = y_mean[bi] - np.sum(raw_coefs * x_mean)
            
            results.setdefault(str(lead), {})[b] = {
                'intercept': float(raw_intercept),
                'coefs': [float(c) for c in raw_coefs],
                'nz': int(best_nz),
                'alpha': float(best_alpha) if best_alpha is not None else None,
                'l1_ratio': float(L1_RATIO),
                'r2': float(best_r2),
                'ols_r2': float(r2_olsv),
                'intercept_std': float(0.0),
                'coefs_std': [float(c) for c in best_coefs_std],
                'predictor_mean': [float(m) for m in x_mean],
                'predictor_sigma': [float(s) for s in x_std],
                'target_mean': float(y_mean[bi]),
                'target_sigma': float(y_std[bi]),
            }
            
        print(f"  -> Lead {lead:2d} min ({N:,} valid pairs) completed in {time.time()-t_lead:.2f}s | Mean R2={np.mean(r2_enet[li]):.4f}, Mean NZ={np.mean(nz_counts[li]):.1f}/9", flush=True)

    # Save outputs
    with open(OUT_DIR / 'model_coefficients.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / 'train_diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2)
        
    print(f"\nSaved updated coefficients to {OUT_DIR / 'model_coefficients.json'}")
    print(f"Saved fit diagnostics to {OUT_DIR / 'train_diagnostics.json'}")
    
    # Print summary of all 72 equations
    print("\n" + "=" * 80)
    print("ALL 72 SPARSE EQUATIONS (Physical Kelvin Scale: BT_k(t+dt) = ...)")
    print("=" * 80)
    for lead in LEADS:
        print(f"\n--- FORECAST LEAD = {lead} MINUTES ---")
        for b in BANDS:
            res = results[str(lead)][b]
            terms = [f"{res['intercept']:.4f}"]
            for ci, c in enumerate(res['coefs']):
                if abs(c) > 1e-4:
                    sign = ' + ' if c >= 0 else ' - '
                    terms.append(f"{sign}{abs(c):.4f}*{BANDS[ci]}")
            eq_str = "".join(terms)
            print(f"  {b}(t+{lead}) = {eq_str}  [R2={res['r2']:.4f}, NZ={res['nz']}/9]")
            
    print("\nTraining complete!")

if __name__ == '__main__':
    main()
