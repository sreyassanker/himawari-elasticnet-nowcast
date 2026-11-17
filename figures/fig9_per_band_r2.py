#!/usr/bin/env python3
"""Figure 9: Per-band R² decay curves comparing training and validation domains."""
import json, numpy as np
import matplotlib.pyplot as plt

OUTPUT_PNG = 'figures/Fig9.png'
OUTPUT_PDF = 'figures/Fig9.pdf'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

BANDS = ['B08','B09','B10','B11','B12','B13','B14','B15','B16']
LEADS = [20, 30, 40, 50, 60, 70, 80, 90]
DOMAINS = ['east_asia', 'australia', 'maritime']
domain_labels = {'east_asia': 'East Asia (368 d)', 'australia': 'Australia (368 d)', 'maritime': 'Maritime (368 d)'}
domain_colors = {'east_asia': '#377eb8', 'australia': '#ff7f00', 'maritime': '#4daf4a'}
domain_styles = {'east_asia': '--', 'australia': ':', 'maritime': '-.'}

with open('results/validation_results.json') as f:
    vj = json.load(f)

results = vj['domains']
train_insample = vj['train_insample']

fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for bi, b in enumerate(BANDS):
    ax = axes[bi // 3, bi % 3]
    train_vals = [train_insample[str(l)][b]['R2'] for l in LEADS]
    ax.plot(LEADS, train_vals, 'ko-', label='Training: South Asia (373 d)', linewidth=1.8, markersize=5)
    for domain in DOMAINS:
        vals = [results[domain][str(l)][b]['R2'] for l in LEADS]
        ax.plot(LEADS, vals, color=domain_colors[domain], linestyle=domain_styles[domain],
                label=domain_labels[domain], linewidth=1.4, markersize=3.5)
    ax.set_xlabel('Lead Time (min)')
    ax.set_ylabel('R²')
    ax.set_title(b, fontweight='bold')
    ax.set_xticks(LEADS)
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.25, linestyle=':')
    if bi == 0:
        ax.legend(fontsize=7.5, loc='lower left')

plt.tight_layout(pad=1.0)
plt.savefig(OUTPUT_PNG, dpi=600, facecolor='white')
plt.savefig(OUTPUT_PDF, dpi=600, facecolor='white')
print(f'Saved {OUTPUT_PNG} and {OUTPUT_PDF}')
plt.close()
