#!/usr/bin/env python3
"""Figure 7: Cross-domain validation — mean R² decay across held-out domains."""
import json, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_PNG = 'figures/Fig7.png'
OUTPUT_PDF = 'figures/Fig7.pdf'

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

BANDS = ['B08','B09','B10','B11','B12','B13','B14','B15','B16']
LEADS = [20, 30, 40, 50, 60, 70, 80, 90]

# Load validation results directly from results/validation_results.json
with open('results/validation_results.json') as f:
    vj = json.load(f)

# Training R² from in-sample South Asia (373 days)
train_r2 = np.array([np.mean([vj['train_insample'][str(l)][b]['R2'] for b in BANDS]) for l in LEADS])

# Validation R² for each domain (368 days each)
domains = {
    'East Asia': 'east_asia',
    'Australia': 'australia',
    'Maritime':  'maritime',
}

val_r2 = {}
for dname, dkey in domains.items():
    val_r2[dname] = np.array([np.mean([vj['domains'][dkey][str(l)][b]['R2'] for b in BANDS]) for l in LEADS])

# ── Plot ──
fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.6))

ax.set_title('Cross-Domain Validation: Mean R² Decay', fontsize=11, fontweight='bold', pad=8)

domain_styles = {
    'East Asia': ('#377eb8', '--'),
    'Australia': ('#e41a1c', ':'),
    'Maritime':  ('#4daf4a', '-.'),
}

ax.plot(LEADS, train_r2, 'ko-', linewidth=2.5, label='Training (South Asia)', markersize=5)
for dname in domains:
    c, ls = domain_styles[dname]
    ax.plot(LEADS, val_r2[dname], color=c, linestyle=ls, linewidth=2,
            label=f'{dname} (368 days)', markersize=4)

ax.set_xlabel('Lead time (min)', fontsize=10)
ax.set_ylabel('Mean R² (over 9 bands)', fontsize=10)
ax.set_xlim(15, 95)
ax.set_xticks(LEADS)
ax.set_ylim(0.50, 1.00)
ax.grid(True, alpha=0.25, linestyle=':')
ax.tick_params(labelsize=9)
ax.legend(fontsize=8, loc='lower left', framealpha=0.85)

# Annotation with R² drop
drop_20 = train_r2[0] - min(val_r2[d][0] for d in domains)
ax.annotate(f'Max R² drop at 20 min: {drop_20:.3f}',
            xy=(0.98, 0.03), xycoords='axes fraction', ha='right', va='bottom',
            fontsize=8, color='#444444', fontstyle='italic')

# Note
ax.annotate('Training: 373 days (2023–2026, >50.2M pairs)  |  Held-out Spatial Domains: 368 days each (1,104 cubes)',
            xy=(0.02, 0.97), xycoords='axes fraction', ha='left', va='top',
            fontsize=7, color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8',
                      edgecolor='#cccccc', linewidth=0.5))

plt.tight_layout(pad=0.8)
plt.savefig(OUTPUT_PNG, dpi=600, facecolor='white')
plt.savefig(OUTPUT_PDF, dpi=600, facecolor='white')
print(f'Saved {OUTPUT_PNG} and {OUTPUT_PDF}')
plt.close()
