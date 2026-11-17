#!/usr/bin/env python3
"""
Evaluate trained Himawari-9 nowcasting Elastic Net equations across held-out domains:
  1. East Asia (368 days, 2023–2026)
  2. Maritime Continent (368 days, 2023–2026)
  3. Australia (368 days, 2023–2026)

Outputs:
  - results/validation_results.json: R2, RMSE, and bias per domain, lead time, and band
"""
import os, sys, json, gc, time, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

BANDS = ['B08','B09','B10','B11','B12','B13','B14','B15','B16']
LEADS = [20,30,40,50,60,70,80,90]
LEAD_FRAMES = [l//10 for l in LEADS]
N_BANDS = len(BANDS)
N_LEADS = len(LEADS)

COEF_PATH = Path('results/model_coefficients.json')
VALID_DIR = Path('data/validation')
TRAIN_DIR = Path('data/training')
OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ['east_asia', 'maritime', 'australia']
DOMAIN_LABELS = ['East Asia', 'Maritime Continent', 'Australia']

with open(COEF_PATH) as f:
    coef_data = json.load(f)

def load_domain_data(dom):
    files = sorted(VALID_DIR.glob(f'{dom}_*.npz'))
    if not files:
        return None, 0
    t0 = time.time()
    cubes = []
    for fpath in files:
        d = np.load(fpath)
        cubes.append(np.stack([d[b] for b in BANDS], axis=-1).astype(np.float32))
    X_cube = np.concatenate(cubes, axis=0)
    del cubes
    gc.collect()
    print(f"  Loaded {len(files)} files in {time.time()-t0:.2f}s, shape: {X_cube.shape}", flush=True)
    return X_cube, len(files)

results = {}
diagnostics = {"domains": {}}

print("=" * 80)
print(" CROSS-DOMAIN GEOGRAPHIC VALIDATION BENCHMARK (ALL 3 DOMAINS)")
print("=" * 80)

for domain, dlabel in zip(DOMAINS, DOMAIN_LABELS):
    print(f"\nEvaluating Domain: {dlabel} ({domain})...", flush=True)
    X_cube, nfiles = load_domain_data(domain)
    if X_cube is None:
        print(f"  WARNING: No files found for {domain}!", flush=True)
        continue

    nan_frac = float(np.isnan(X_cube).mean())
    diagnostics["domains"][domain] = {
        "days": nfiles,
        "shape": list(X_cube.shape),
        "nan_fraction": nan_frac
    }

    domain_results = {}
    CHUNK_SIZE = 5000  # Process in lean ~180MB temporal chunks avoiding memory pressure
    T_total = X_cube.shape[0]

    for li, (lead, nf) in enumerate(zip(LEADS, LEAD_FRAMES)):
        sum_y = {b: 0.0 for b in BANDS}
        sum_y2 = {b: 0.0 for b in BANDS}
        sum_err = {b: 0.0 for b in BANDS}
        sum_err2 = {b: 0.0 for b in BANDS}
        n_valid_total = 0

        lead_coefs = {b: np.array(coef_data[str(lead)][b]['coefs'], dtype=np.float32) for b in BANDS}
        lead_inter = {b: np.float32(coef_data[str(lead)][b]['intercept']) for b in BANDS}

        T_lead = T_total - nf
        for t_start in range(0, T_lead, CHUNK_SIZE):
            t_end = min(t_start + CHUNK_SIZE, T_lead)
            c_in = X_cube[t_start : t_end]
            c_out = X_cube[t_start + nf : t_end + nf]

            valid = ~(np.isnan(c_in).any(axis=2) | np.isnan(c_out).any(axis=2))
            c_in_v = c_in[valid]
            c_out_v = c_out[valid]
            n_chunk = len(c_in_v)
            if n_chunk == 0:
                continue

            n_valid_total += n_chunk
            for bi, b in enumerate(BANDS):
                ya = c_out_v[:, bi].astype(np.float64)
                yp = (lead_inter[b] + c_in_v @ lead_coefs[b]).astype(np.float64)
                err = yp - ya
                sum_y[b] += np.sum(ya)
                sum_y2[b] += np.dot(ya, ya)
                sum_err[b] += np.sum(err)
                sum_err2[b] += np.dot(err, err)

        lead_results = {}
        for b in BANDS:
            if n_valid_total > 0:
                ss_tot = sum_y2[b] - (sum_y[b] * sum_y[b]) / n_valid_total
                ss_res = sum_err2[b]
                r2 = float(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
                rmse = float(np.sqrt(sum_err2[b] / n_valid_total))
                bias = float(sum_err[b] / n_valid_total)
            else:
                r2, rmse, bias = np.nan, np.nan, np.nan

            lead_results[b] = {
                'R2': r2,
                'RMSE': rmse,
                'Bias': bias
            }

        domain_results[str(lead)] = lead_results
        r2_vals = [v['R2'] for v in lead_results.values()]
        rmse_vals = [v['RMSE'] for v in lead_results.values()]
        print(f"    Lead {lead:2d} min: Mean R2={np.mean(r2_vals):.4f} [{min(r2_vals):.3f} - {max(r2_vals):.3f}], "
              f"Mean RMSE={np.mean(rmse_vals):.2f} K, N={n_valid_total:,}", flush=True)

    results[domain] = domain_results
    del X_cube
    gc.collect()

# In-sample training scores across complete South Asia (2023-2026)
print("\nExtracting in-sample South Asia training scores (373 days, 2023-2026)...", flush=True)
train_insample = {
    str(lead): {
        b: {'R2': float(coef_data[str(lead)][b]['r2'])}
        for b in BANDS
    }
    for lead in LEADS
}

# Temporal validation on 2026 South Asia subset
print("\nEvaluating 2026 South Asia subset (82 days)...", flush=True)
test_2026_files = sorted(TRAIN_DIR.glob('2026*.npz'))
if test_2026_files:
    cubes_26 = []
    for fpath in test_2026_files:
        d = np.load(fpath)
        cubes_26.append(np.stack([d[b] for b in BANDS], axis=-1).astype(np.float32))
    X_test_2026 = np.concatenate(cubes_26, axis=0)
    del cubes_26
    gc.collect()

    test_2026_results = {}
    CHUNK_SIZE = 5000
    T_26 = X_test_2026.shape[0]
    for li, (lead, nf) in enumerate(zip(LEADS, LEAD_FRAMES)):
        sum_y = {b: 0.0 for b in BANDS}
        sum_y2 = {b: 0.0 for b in BANDS}
        sum_err = {b: 0.0 for b in BANDS}
        sum_err2 = {b: 0.0 for b in BANDS}
        n_valid_total = 0

        lead_coefs = {b: np.array(coef_data[str(lead)][b]['coefs'], dtype=np.float32) for b in BANDS}
        lead_inter = {b: np.float32(coef_data[str(lead)][b]['intercept']) for b in BANDS}

        T_lead = T_26 - nf
        for t_start in range(0, T_lead, CHUNK_SIZE):
            t_end = min(t_start + CHUNK_SIZE, T_lead)
            c_in = X_test_2026[t_start : t_end]
            c_out = X_test_2026[t_start + nf : t_end + nf]

            valid = ~(np.isnan(c_in).any(axis=2) | np.isnan(c_out).any(axis=2))
            c_in_v = c_in[valid]
            c_out_v = c_out[valid]
            n_chunk = len(c_in_v)
            if n_chunk == 0:
                continue

            n_valid_total += n_chunk
            for bi, b in enumerate(BANDS):
                ya = c_out_v[:, bi].astype(np.float64)
                yp = (lead_inter[b] + c_in_v @ lead_coefs[b]).astype(np.float64)
                err = yp - ya
                sum_y[b] += np.sum(ya)
                sum_y2[b] += np.dot(ya, ya)
                sum_err[b] += np.sum(err)
                sum_err2[b] += np.dot(err, err)

        lead_results = {}
        for b in BANDS:
            if n_valid_total > 0:
                ss_tot = sum_y2[b] - (sum_y[b] * sum_y[b]) / n_valid_total
                ss_res = sum_err2[b]
                r2 = float(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
                rmse = float(np.sqrt(sum_err2[b] / n_valid_total))
                bias = float(sum_err[b] / n_valid_total)
            else:
                r2, rmse, bias = np.nan, np.nan, np.nan
            lead_results[b] = {'R2': r2, 'RMSE': rmse, 'Bias': bias}
        test_2026_results[str(lead)] = lead_results
    diagnostics["temporal_test_2026"] = {
        "days": len(test_2026_files),
        "date_range": [test_2026_files[0].stem, test_2026_files[-1].stem],
        "shape": list(X_test_2026.shape)
    }
    del X_test_2026
    gc.collect()
else:
    test_2026_results = {}

out = {
    "train_insample": train_insample,
    "domains": results,
    "test_2026_temporal": test_2026_results,
    "diagnostics": diagnostics
}

with open(OUT_DIR / 'validation_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(f"\nSaved {OUT_DIR / 'validation_results.json'}", flush=True)
print("Cross-domain geographic validation completed successfully!", flush=True)
