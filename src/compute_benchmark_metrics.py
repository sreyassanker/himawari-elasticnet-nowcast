#!/usr/bin/env python3
"""
Compute summary metrics and key verification statistics from model results.

Generates results/benchmark_metrics.json for manuscript text and tables.
"""
import json, numpy as np
from pathlib import Path

BANDS = ['B08','B09','B10','B11','B12','B13','B14','B15','B16']
LEADS = [20,30,40,50,60,70,80,90]
cj = json.load(open('results/model_coefficients.json'))
vj = json.load(open('results/validation_results.json'))
tj = json.load(open('results/train_diagnostics.json'))

r2 = np.array([[cj[str(L)][b]['r2'] for b in BANDS] for L in LEADS])
nz = np.array([[cj[str(L)][b]['nz'] for b in BANDS] for L in LEADS])
ols_r2 = np.array([[cj[str(L)][b]['ols_r2'] for b in BANDS] for L in LEADS])

# Fully standardized dimensionless coefficients: c* = raw * sigma_j / sigma_target
def std_coefs(L, b):
    e = cj[str(L)][b]
    return np.array(e['coefs']) * np.array(e['predictor_sigma']) / e['target_sigma']

pers, cross, viol_not_largest, zero_self = [], [], [], []
freq = {b: 0 for b in BANDS}
for L in LEADS:
    for b in BANDS:
        bi = BANDS.index(b)
        c = std_coefs(L, b)
        sel = [i for i in range(9) if abs(c[i]) > 1e-8]
        if abs(c[bi]) <= 1e-8:
            zero_self.append(f"{b}@{L}")
        else:
            if len(sel) > 1 and abs(c[bi]) < max(abs(c[i]) for i in sel):
                viol_not_largest.append(f"{b}@{L}")
            pers.append(c[bi])
        cross += [abs(c[i]) for i in sel if i != bi]
        for i in sel:
            if i != bi: freq[BANDS[i]] += 1

p = np.array(pers); cr = np.array(cross)
train_mean = r2.mean(1)
slope = (r2[-1] - r2[0]) / 7

VAL_DOMAINS = ['east_asia', 'maritime', 'australia']
val = {}
for d in VAL_DOMAINS:
    val[d] = np.array([[vj['domains'][d][str(L)][b]['R2'] for b in BANDS] for L in LEADS])
    val[d + '_rmse'] = np.array([[vj['domains'][d][str(L)][b]['RMSE'] for b in BANDS] for L in LEADS])
train_isc = np.array([[vj['train_insample'][str(L)][b]['R2'] for b in BANDS] for L in LEADS])
# consistency check: shipped vs re-applied in-sample
assert np.max(np.abs(train_isc - r2)) < 1e-4, "train r2 mismatch between json and re-application"

stats = {
  "table1_r2": {str(L): {b: round(float(r2[i, j]), 3) for j, b in enumerate(BANDS)} for i, L in enumerate(LEADS)},
  "table2_nz": {str(L): {b: int(nz[i, j]) for j, b in enumerate(BANDS)} for i, L in enumerate(LEADS)},
  "table3_r2_20min": {b: {"train": round(float(r2[0, i]), 3),
                          "east_asia": round(float(val['east_asia'][0, i]), 3),
                          "maritime": round(float(val['maritime'][0, i]), 3),
                          "australia": round(float(val['australia'][0, i]), 3)}
                      for i, b in enumerate(BANDS)},
  "train_mean_per_lead": [round(float(v), 3) for v in train_mean],
  "val_mean_per_lead": {d: [round(float(v), 3) for v in val[d].mean(1)] for d in VAL_DOMAINS},
  "drop_per_lead": {d: [round(float(train_mean[i] - val[d].mean(1)[i]), 3) for i in range(8)] for d in VAL_DOMAINS},
  "mean_nz": round(float(nz.mean()), 2),
  "nz_min": int(nz.min()), "nz_max": int(nz.max()),
  "self_persistence": {
      "n_zero_self": len(zero_self), "zero_self_eqs": zero_self,
      "n_self_not_largest": len(viol_not_largest), "not_largest_eqs": viol_not_largest,
      "in_0.6_0.9_pct": round(float(((p>=0.6)&(p<=0.9)).mean()*100), 1),
      "min": round(float(p.min()), 3), "median": round(float(np.median(p)), 3), "max": round(float(p.max()), 3),
  },
  "cross_coefs": {
      "in_0.1_0.4_pct": round(float(((cr>=0.1)&(cr<=0.4)).mean()*100), 1),
      "min": round(float(cr.min()), 3), "median": round(float(np.median(cr)), 3),
      "p95": round(float(np.percentile(cr, 95)), 3), "max": round(float(cr.max()), 3),
  },
  "predictor_freq": dict(sorted(freq.items(), key=lambda x: -x[1])),
  "b08_pure_persistence_all_leads": bool(np.all(nz[:, 0] == 1)),
  "best_band_at_20": BANDS[int(np.argmax(r2[0]))],
  "worst_band_at_20": BANDS[int(np.argmin(r2[0]))],
  "best_band_at_90": BANDS[int(np.argmax(r2[-1]))],
  "worst_band_at_90": BANDS[int(np.argmin(r2[-1]))],
  "min_band_per_lead": [BANDS[int(np.argmin(r2[i]))] for i in range(8)],
  "max_band_per_lead": [BANDS[int(np.argmax(r2[i]))] for i in range(8)],
  "slopes": {b: round(float(slope[i]), 4) for i, b in enumerate(BANDS)},
  "slowest_decay_band": BANDS[int(np.argmax(slope))],
  "fastest_decay_band": BANDS[int(np.argmin(slope))],
  "band_order_preserved_validation": {
      d: {"B08_highest_all_leads": bool(np.all(np.argmax(val[d], axis=1) == 0)),
          "B12_lowest_all_leads": bool(np.all(np.argmin(val[d], axis=1) == 4)),
          "B08_20": round(float(val[d][0, 0]), 3),
          "best_20": BANDS[int(np.argmax(val[d][0]))],
          "worst_90": BANDS[int(np.argmin(val[d][-1]))]}
      for d in VAL_DOMAINS},
  "samples": {str(L): tj['leads'][str(L)] for L in LEADS},
  "retention_rule_min_ratio": round(float(np.min(r2 / ols_r2)), 4),
}

if 'test_2026_temporal' in vj:
    t26_r2 = np.array([[vj['test_2026_temporal'][str(L)][b]['R2'] for b in BANDS] for L in LEADS])
    t26_rmse = np.array([[vj['test_2026_temporal'][str(L)][b]['RMSE'] for b in BANDS] for L in LEADS])
    stats["test_2026_temporal"] = {
        "r2_mean_per_lead": [round(float(v), 3) for v in t26_r2.mean(1)],
        "rmse_mean_per_lead": [round(float(v), 3) for v in t26_rmse.mean(1)],
        "r2_20min": {b: round(float(t26_r2[0, i]), 3) for i, b in enumerate(BANDS)},
        "r2_90min": {b: round(float(t26_r2[-1, i]), 3) for i, b in enumerate(BANDS)},
        "rmse_20min": {b: round(float(t26_rmse[0, i]), 3) for i, b in enumerate(BANDS)},
        "rmse_90min": {b: round(float(t26_rmse[-1, i]), 3) for i, b in enumerate(BANDS)},
    }

with open('results/benchmark_metrics.json', 'w') as f:
    json.dump(stats, f, indent=2)
print(json.dumps({k: stats[k] for k in ['train_mean_per_lead','val_mean_per_lead','drop_per_lead',
    'mean_nz','self_persistence','cross_coefs','predictor_freq','slowest_decay_band',
    'fastest_decay_band','band_order_preserved_validation','retention_rule_min_ratio']}, indent=1))
print("Saved results/benchmark_metrics.json")
