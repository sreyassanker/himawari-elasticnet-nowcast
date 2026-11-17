#!/usr/bin/env python3
"""
Pre-compute balanced pixel locations across Himawari-9 AHI segments for the South Asia domain.

Outputs:
  - data/pixel_locations.npz: Pixel indices and segment coordinates
"""
import numpy as np
from pyresample.geometry import AreaDefinition

LAT_MIN, LAT_MAX = 5, 35
LON_MIN, LON_MAX = 65, 95

area = AreaDefinition("full", "", "geos",
    "+proj=geos +lon_0=140.7 +h=35785863 +a=6378137 +b=6356752.3 +no_defs",
    5500, 5500, [-5500000, -5500000, 5500000, 5500000])

RES = 0.05
lons = np.arange(LON_MIN, LON_MAX + RES, RES)
lats = np.arange(LAT_MIN, LAT_MAX + RES, RES)
lon_g, lat_g = np.meshgrid(lons, lats)
pts = np.column_stack([lon_g.ravel(), lat_g.ravel()])

cols, rows = area.get_array_indices_from_lonlat(pts[:, 0], pts[:, 1])
valid = ~rows.mask
rows_v = rows[valid].data.astype(int)
cols_v = cols[valid].data.astype(int)
lats_v = lat_g.ravel()[valid]
lons_v = lon_g.ravel()[valid]
seg = rows_v // 550 + 1
seg_row = rows_v % 550

# Sample 1000 pixels balanced across segments
N_TOTAL = 1000
rng = np.random.RandomState(42)
unique_segs = sorted(set(seg))
n_per_seg = N_TOTAL // len(unique_segs)

pixel_info = []
for s in unique_segs:
    idx = np.where(seg == s)[0]
    chosen = rng.choice(idx, min(n_per_seg, len(idx)), replace=False)
    for i in chosen:
        pixel_info.append((float(lats_v[i]), float(lons_v[i]),
                           int(seg[i]), int(seg_row[i]), int(cols_v[i])))

pixel_info = np.array(pixel_info, dtype=[
    ('lat', 'f4'), ('lon', 'f4'), ('seg', 'i4'), ('row', 'i4'), ('col', 'i4')])

# Add remaining if undersampled
if len(pixel_info) < N_TOTAL:
    extra = N_TOTAL - len(pixel_info)
    remaining_idx = np.setdiff1d(np.arange(valid.sum()),
                                  np.concatenate([np.where(seg == s)[0][:n_per_seg] for s in unique_segs]))
    extra_chosen = rng.choice(remaining_idx, extra, replace=False)
    for i in extra_chosen:
        pixel_info = np.append(pixel_info, np.array([
            (float(lats_v[i]), float(lons_v[i]),
             int(seg[i]), int(seg_row[i]), int(cols_v[i]))],
            dtype=pixel_info.dtype))

print(f"Total: {len(pixel_info)} pixels")
for s in sorted(set(pixel_info['seg'])):
    mask = pixel_info['seg'] == s
    print(f"  S{s:04d}: {mask.sum()} px, lat=[{pixel_info['lat'][mask].min():.1f}, {pixel_info['lat'][mask].max():.1f}]°N")

np.savez_compressed('data/pixel_locations.npz', info=pixel_info)
print("\nSaved data/pixel_locations.npz")
