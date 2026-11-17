#!/usr/bin/env python3
"""
Screen candidate dates by mean Band 13 brightness temperature to select diverse meteorological regimes.
"""
import sys, os, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsd_reader import _download_and_extract_one

PIXEL_FILE = 'data/pixel_locations.npz'
pix = np.load(PIXEL_FILE)['info']

# Sub-sample to 100 pixels for fast scan
rng = np.random.RandomState(42)
idx = rng.choice(len(pix), 100, replace=False)

seg_pixels = {}
seg_idx_map = {}
for i in idx:
    p = pix[i]
    seg = f'S{int(p["seg"]):02d}10'
    seg_pixels.setdefault(seg, ([], []))[0].append(int(p['row']))
    seg_pixels[seg][1].append(int(p['col']))
    seg_idx_map.setdefault(seg, []).append(len(seg_idx_map.get(seg, [])))
seg_pixels = {s: (np.array(r), np.array(c)) for s, (r, c) in seg_pixels.items()}

print(f"Scanning B13 over 100 South Asia pixels, segments: {sorted(seg_pixels.keys())}")

def scan_one_day(date_str):
    """Get mean B13 BT at 12:00 UTC for a given day."""
    time_str = '1200'
    results = {}
    for seg, (rows, cols) in seg_pixels.items():
        key = f"AHI-L1b-FLDK/{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}/{time_str}/HS_H09_{date_str}_{time_str}_B13_FLDK_R20_{seg}.DAT.bz2"
        band, bt = _download_and_extract_one(key, rows, cols, 13)
        results[seg] = bt
    # Merge
    all_bt = np.concatenate([results[s] for s in sorted(results.keys())])
    return np.nanmean(all_bt), np.nanstd(all_bt)

# Scan June 2023
import calendar
year, month = 2023, 6
days_in_month = calendar.monthrange(year, month)[1]
dates = [f"{year}{month:02d}{d:02d}" for d in range(1, days_in_month + 1)]

print(f"Scanning {len(dates)} days in {year}-{month:02d}...")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(scan_one_day, d): d for d in dates}
    for fut in as_completed(futs):
        d = futs[fut]
        mean_bt, std_bt = fut.result()
        results.append((d, mean_bt, std_bt))
        print(f"  {d}: mean_BT={mean_bt:.1f}K, std={std_bt:.1f}K", flush=True)

results.sort(key=lambda x: x[1])  # sort by mean BT

print(f"\n{'='*60}")
print(f"Scan complete in {time.time()-t0:.1f}s")
print(f"\n4 COLDEST days (lowest mean B13 BT):")
for d, m, s in results[:4]:
    print(f"  {d}: {m:.1f}K ±{s:.1f}K")

print(f"\n3 WARMEST days (highest mean B13 BT):")
for d, m, s in results[-3:]:
    print(f"  {d}: {m:.1f}K ±{s:.1f}K")

print(f"\nFull sorted list:")
for d, m, s in results:
    print(f"  {d}: {m:.1f}K ±{s:.1f}K")

# Save
with open('data/training_days.txt', 'w') as f:
    f.write(f"# Training days from {dates[0]}-{dates[-1]}\n")
    f.write(f"# COLDEST:\n")
    for d, m, s in results[:4]:
        f.write(f"{d}  {m:.1f} {s:.1f}\n")
    f.write(f"# WARMEST:\n")
    for d, m, s in results[-3:]:
        f.write(f"{d}  {m:.1f} {s:.1f}\n")
print(f"\nSaved training_days.txt")
