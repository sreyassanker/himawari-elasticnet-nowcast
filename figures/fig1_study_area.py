#!/usr/bin/env python3
"""Figure 1: Study area — Himawari-9 full disk with training and validation domains."""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

OUTPUT_PNG = 'figures/Fig1.png'
OUTPUT_PDF = 'figures/Fig1.pdf'

HIMAWARI_LON = 140.7

DOMAINS = {
    'South Asia (train + val)': {'lat': (5, 35), 'lon': (65, 95), 'color': '#e41a1c'},
    'East Asia (validation)':   {'lat': (20, 45), 'lon': (110, 145), 'color': '#377eb8'},
    'Maritime (validation)':    {'lat': (-10, 10), 'lon': (95, 140), 'color': '#4daf4a'},
}

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.set_extent([50, 170, -20, 50], crs=ccrs.PlateCarree())
ax.set_title('Himawari-9 coverage with training and validation domains',
             fontsize=12, fontweight='bold', pad=10)

ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.6)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.08)

# Full disk coverage circle (~81.3° great circle arc)
lon_grid = np.linspace(-180, 180, 361)
lat_grid = np.linspace(-90, 90, 181)
lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
dist = np.arccos(np.sin(np.deg2rad(lat_2d)) * np.sin(np.deg2rad(0)) +
                 np.cos(np.deg2rad(lat_2d)) * np.cos(np.deg2rad(0)) *
                 np.cos(np.deg2rad(lon_2d - HIMAWARI_LON)))
disk_mask = dist <= np.deg2rad(81.3)
ax.contour(lon_2d, lat_2d, disk_mask.astype(float),
           levels=[0.5], colors=['gray'], linewidths=0.8, linestyles=':', alpha=0.5,
           transform=ccrs.PlateCarree())

# Sub-Satellite Point (SSP)
ax.plot(HIMAWARI_LON, 0, 'k*', markersize=12, transform=ccrs.PlateCarree(), zorder=5)
ax.text(HIMAWARI_LON + 2, -4, r'$\mathbf{SSP}\ 140.7^{\circ}\mathrm{E}$',
        fontsize=7, ha='left', transform=ccrs.PlateCarree(), color='gray')

# Domain boxes
for name, d in DOMAINS.items():
    rect = Rectangle(
        (d['lon'][0], d['lat'][0]),
        d['lon'][1] - d['lon'][0],
        d['lat'][1] - d['lat'][0],
        linewidth=2, edgecolor=d['color'], facecolor=d['color'], alpha=0.12, zorder=4,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(rect)
    cx = (d['lon'][0] + d['lon'][1]) / 2
    cy = (d['lat'][0] + d['lat'][1]) / 2
    ax.text(cx, cy, name, fontsize=7, ha='center', va='center', fontweight='bold',
            color=d['color'], transform=ccrs.PlateCarree(),
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))

# Lat/Lon gridlines
gl = ax.gridlines(draw_labels=True, linestyle=':', alpha=0.4, linewidth=0.3, color='gray')
gl.top_labels = True
gl.right_labels = True
gl.bottom_labels = True
gl.left_labels = True
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

# North arrow (N below the arrow)
ax.annotate('', xy=(0.96, 0.94), xycoords='axes fraction',
            xytext=(0.96, 0.85), fontsize=14,
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(0.96, 0.82, 'N', fontsize=14, fontweight='bold', ha='center', va='center',
        transform=ax.transAxes, color='black')

# Scale bar (~1000 km at equator)
import cartopy.geodesic as cgeo
from matplotlib.patches import FancyBboxPatch

scale_lon, scale_lat = 55, 44
scale_length_deg = 12  # ~1000 km at this latitude
ax.plot([scale_lon, scale_lon + scale_length_deg], [scale_lat, scale_lat],
        color='black', linewidth=2, transform=ccrs.PlateCarree())
ax.plot([scale_lon, scale_lon], [scale_lat - 0.5, scale_lat + 0.5],
        color='black', linewidth=2, transform=ccrs.PlateCarree())
ax.plot([scale_lon + scale_length_deg, scale_lon + scale_length_deg],
        [scale_lat - 0.5, scale_lat + 0.5],
        color='black', linewidth=2, transform=ccrs.PlateCarree())
ax.text(scale_lon + scale_length_deg / 2, scale_lat + 1.5, '1000 km',
        fontsize=8, ha='center', transform=ccrs.PlateCarree())

# Legend
legend_elements = [
    Line2D([0], [0], color='gray', linestyle=':', label=r'Full disk coverage (~81$^{\circ}$)'),
    Line2D([0], [0], marker='*', color='gray', label=r'SSP 140.7$^{\circ}$E',
           markerfacecolor='gray', markersize=8, linestyle='None'),
]
for name, d in DOMAINS.items():
    legend_elements.append(Line2D([0], [0], color=d['color'], linewidth=2, label=name))
ax.legend(handles=legend_elements, loc='lower left', fontsize=7.5, framealpha=0.85,
          edgecolor='gray')

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=600, bbox_inches='tight')
plt.savefig(OUTPUT_PDF, dpi=600, bbox_inches='tight')
print(f'Saved {OUTPUT_PNG} and {OUTPUT_PDF}')
plt.close()
