#!/usr/bin/env python3
"""Figure 5: Per-band R² comparison at 20-min and 90-min lead times."""
import json, numpy as np
import matplotlib.pyplot as plt

OUTPUT_PNG = 'figures/Fig5.png'
OUTPUT_PDF = 'figures/Fig5.pdf'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

with open('results/model_coefficients.json') as f:
    data = json.load(f)

bands = ['B08','B09','B10','B11','B12','B13','B14','B15','B16']
leads = [20, 90]

r2 = {l: [data[str(l)][b]['r2'] for b in bands] for l in leads}
labels_short = [f'B08\n6.2$\\mu$m', f'B09\n6.9$\\mu$m', f'B10\n7.3$\\mu$m', f'B11\n8.6$\\mu$m',
                f'B12\n9.6$\\mu$m', f'B13\n10.4$\\mu$m', f'B14\n11.2$\\mu$m', f'B15\n12.4$\\mu$m', f'B16\n13.3$\\mu$m']

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

x = np.arange(len(bands))
w = 0.35

bars_20 = ax.bar(x - w/2, r2[20], w, label='20-min lead', color='#377eb8', edgecolor='white', linewidth=0.3)
bars_90 = ax.bar(x + w/2, r2[90], w, label='90-min lead', color='#e41a1c', edgecolor='white', linewidth=0.3)

# Value labels inside the top of bars in bold white (vertical orientation)
for bar in bars_20:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.015,
            f'{bar.get_height():.3f}', ha='center', va='top', fontsize=7.8, color='white', fontweight='bold', rotation=90)
for bar in bars_90:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.015,
            f'{bar.get_height():.3f}', ha='center', va='top', fontsize=7.8, color='white', fontweight='bold', rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(labels_short, fontsize=7.5)
ax.set_xlabel('Band (wavelength)', fontsize=10)
ax.set_ylabel('R²', fontsize=10)
ax.set_title('Per-Band Training R² (In-Sample Fit)', fontsize=11, fontweight='bold', pad=8)
ax.set_ylim(0.50, 1.02)
ax.grid(True, axis='y', alpha=0.25, linestyle=':')
ax.legend(fontsize=9, loc='lower right')
ax.tick_params(labelsize=9)

plt.tight_layout(pad=0.8)
plt.savefig(OUTPUT_PNG, dpi=600, facecolor='white')
plt.savefig(OUTPUT_PDF, dpi=600, facecolor='white')
print(f'Saved {OUTPUT_PNG} and {OUTPUT_PDF}')
plt.close()
