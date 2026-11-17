#!/usr/bin/env python3
"""Figure 3: Elastic Net R² decay across lead times (training performance)."""
import json, numpy as np
import matplotlib.pyplot as plt

OUTPUT_PNG = 'figures/Fig3.png'
OUTPUT_PDF = 'figures/Fig3.pdf'

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.sans-serif'] = ['Helvetica']

with open('results/model_coefficients.json') as f:
    coef_data = json.load(f)

leads = sorted(int(k) for k in coef_data.keys())
bands = ['B08','B09','B10','B11','B12','B13','B14','B15','B16']
band_colors = plt.cm.Set1(np.linspace(0, 1, 9))

r2_all = np.array([[coef_data[str(l)][b]['r2'] for l in leads] for b in bands])
mean_r2 = r2_all.mean(axis=0)

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

# Shaded range
ax.fill_between(leads, r2_all.min(axis=0), r2_all.max(axis=0),
                alpha=0.10, color='black', label='Range (9 bands)')

# Individual bands
for i, b in enumerate(bands):
    ax.plot(leads, r2_all[i], color=band_colors[i], linewidth=0.8, alpha=0.6)

# Mean
ax.plot(leads, mean_r2, 'k-', linewidth=2.5, label='Mean')

# Title
ax.set_title('Elastic Net R² Decay vs Lead Time', fontsize=11, fontweight='bold', pad=8)

# Labels
ax.set_xlabel('Lead time (min)', fontsize=10)
ax.set_ylabel('R²', fontsize=10)
ax.set_xlim(15, 95)
ax.set_xticks(leads)
ax.set_ylim(0.55, 1.00)
ax.grid(True, alpha=0.25, linestyle=':')
ax.tick_params(labelsize=9)

# Legend
legend_texts = bands + ['Mean', 'Range']
legend_lines = [plt.Line2D([0],[0], color=band_colors[i], lw=1.2) for i in range(9)] \
             + [plt.Line2D([0],[0], color='k', lw=2.5),
                plt.Line2D([0],[0], color='gray', lw=4, alpha=0.15)]
leg = ax.legend(legend_lines, legend_texts, fontsize=7, loc='upper right',
                ncol=2, framealpha=0.85)
leg.set_title('Band', prop={'size': 7, 'weight': 'bold'})

ax.annotate(f'Mean R² = {mean_r2[0]:.2f} (20 min)  ->  {mean_r2[-1]:.2f} (90 min)',
            xy=(0.98, 0.03), xycoords='axes fraction', ha='right', va='bottom',
            fontsize=8, color='#444444', fontstyle='italic')

plt.tight_layout(pad=0.8)
plt.savefig(OUTPUT_PNG, dpi=600, facecolor='white')
plt.savefig(OUTPUT_PDF, dpi=600, facecolor='white')
print(f'Saved {OUTPUT_PNG} and {OUTPUT_PDF}')
plt.close()
