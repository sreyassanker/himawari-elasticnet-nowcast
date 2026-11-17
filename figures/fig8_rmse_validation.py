#!/usr/bin/env python3
"""Figure 8: Root-mean-square error (RMSE) versus lead time across validation domains."""
import json, numpy as np
import matplotlib.pyplot as plt

OUTPUT_PNG = 'figures/Fig8.png'
OUTPUT_PDF = 'figures/Fig8.pdf'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

BANDS = ['B08','B09','B10','B11','B12','B13','B14','B15','B16']
LEADS = [20, 30, 40, 50, 60, 70, 80, 90]
DOMAINS = ['east_asia', 'australia', 'maritime']
domain_colors = {'east_asia': '#377eb8', 'australia': '#ff7f00', 'maritime': '#4daf4a'}
domain_styles = {'east_asia': '--', 'australia': '-.', 'maritime': ':'}

with open('results/validation_results.json') as f:
    vj = json.load(f)

results = vj['domains']

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
for bi, b in enumerate(BANDS):
    ax = axes[bi // 3, bi % 3]
    for domain in DOMAINS:
        vals = [results[domain][str(l)][b]['RMSE'] for l in LEADS]
        ax.plot(LEADS, vals, color=domain_colors[domain], linestyle=domain_styles[domain],
                label=domain.replace('_', ' ').title(), linewidth=1.5, markersize=4)
    ax.set_xlabel('Lead Time (min)')
    ax.set_ylabel('RMSE (K)')
    ax.set_title(b, fontweight='bold')
    ax.set_xticks(LEADS)
    ax.grid(True, alpha=0.2, linestyle=':')
    if bi == 0:
        ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Elastic Net RMSE by Validation Domain (Multi-Year Archive: 368 Days Each)', fontweight='bold', fontsize=16, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUTPUT_PNG)
fig.savefig(OUTPUT_PDF)
print(f"Saved {OUTPUT_PNG} and {OUTPUT_PDF}")
plt.close()
