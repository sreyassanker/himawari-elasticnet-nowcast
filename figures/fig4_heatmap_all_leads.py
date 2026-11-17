#!/usr/bin/env python3
"""Figure 4: Per-band R² across lead times for training and validation domains."""
import json, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

OUTPUT_PNG = 'figures/Fig4.png'
OUTPUT_PDF = 'figures/Fig4.pdf'

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

BANDS = ['B08','B09','B10','B11','B12','B13','B14','B15','B16']
LEADS = [20, 30, 40, 50, 60, 70, 80, 90]
N_BANDS = len(BANDS)
N_LEADS = len(LEADS)

with open('results/model_coefficients.json') as f:
    coef_data = json.load(f)

with open('results/validation_results.json') as f:
    vj = json.load(f)

print('Reading per-band R² across all lead times from validation results...')

# Training matrix (South Asia, 373 days)
train_mat = np.array([[coef_data[str(l)][b]['r2'] for b in BANDS] for l in LEADS])

# Validation matrices (368 days each across 2023–2026)
ea_mat = np.array([[vj['domains']['east_asia'][str(l)][b]['R2'] for b in BANDS] for l in LEADS])
au_mat = np.array([[vj['domains']['australia'][str(l)][b]['R2'] for b in BANDS] for l in LEADS])
mc_mat = np.array([[vj['domains']['maritime'][str(l)][b]['R2'] for b in BANDS] for l in LEADS])

# ── Plot 2×2 ──
panels = [
    ('(a) Training: South Asia (373 days)', train_mat),
    ('(b) Validation: East Asia (368 days)', ea_mat),
    ('(c) Validation: Australia (368 days)', au_mat),
    ('(d) Validation: Maritime Continent (368 days)', mc_mat),
]

fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
vmin, vmax = 0.50, 1.00
norm = Normalize(vmin=vmin, vmax=vmax)

for ax, (title, mat) in zip(axes.flat, panels):
    im = ax.imshow(mat, cmap='YlOrRd', norm=norm, aspect='auto')

    ax.set_xticks(range(N_BANDS))
    ax.set_xticklabels(BANDS, fontsize=7.5)
    ax.set_yticks(range(N_LEADS))
    ax.set_yticklabels([str(l) for l in LEADS], fontsize=7.5)
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=8)
    ax.set_xlabel('Infrared Band', fontsize=8)
    ax.set_ylabel('Lead Time (min)', fontsize=8)

    # Annotate cells
    for li in range(N_LEADS):
        for bi in range(N_BANDS):
            val = mat[li, bi]
            c = 'white' if val > 0.75 else '#222222'
            ax.text(bi, li, f'{val:.2f}', ha='center', va='center',
                    fontsize=6.5, color=c, fontweight='bold')

fig.suptitle('Multi-Domain Per-Band R² Across Lead Times (2023–2026)', fontsize=11, fontweight='bold', y=0.98)

cbar_ax = fig.add_axes([0.92, 0.10, 0.015, 0.80])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('R²', fontsize=9)
cbar.ax.tick_params(labelsize=8)

plt.subplots_adjust(left=0.07, right=0.90, bottom=0.07, top=0.91, wspace=0.25, hspace=0.32)
plt.savefig(OUTPUT_PNG, dpi=600, facecolor='white')
plt.savefig(OUTPUT_PDF, dpi=600, facecolor='white')
print(f'Saved {OUTPUT_PNG} and {OUTPUT_PDF}')
plt.close()
