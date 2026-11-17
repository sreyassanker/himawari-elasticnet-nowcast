#!/usr/bin/env python3
"""Figure 6: Cross-band influence matrix of standardized Elastic Net coefficients at 20-min lead time."""
import json, numpy as np
import matplotlib.pyplot as plt

OUTPUT_PNG = 'figures/Fig6.png'
OUTPUT_PDF = 'figures/Fig6.pdf'

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
N_BANDS = len(BANDS)

with open('results/model_coefficients.json') as f:
    results = json.load(f)

# Extract standardized coefficients at 20-min lead
mat = np.array([results['20'][b]['coefs_std'] for b in BANDS])

fig, ax = plt.subplots(figsize=(10, 8))
vmax = max(abs(mat).max(), 1.0)
im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
ax.set_xticks(range(N_BANDS))
ax.set_yticks(range(N_BANDS))
ax.set_xticklabels(BANDS, rotation=45)
ax.set_yticklabels(BANDS)
ax.set_xlabel('Predictor Band')
ax.set_ylabel('Target Band')
ax.set_title('Cross-Band Influence Matrix (20-min Lead, std. coefs)', fontweight='bold')
plt.colorbar(im, ax=ax, label='Elastic Net Coefficient', shrink=0.8)

fig.tight_layout()
fig.savefig(OUTPUT_PNG)
fig.savefig(OUTPUT_PDF)
print(f"Saved {OUTPUT_PNG} and {OUTPUT_PDF}")
plt.close()
