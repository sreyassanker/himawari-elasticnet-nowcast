#!/usr/bin/env python3
"""
Extract Himawari-9 HSD data for cross-domain validation cubes (East Asia, Maritime Continent, Australia).

Samples 1,000 random pixels per day deterministically seeded by date and domain.

Outputs:
  - data/validation/{domain}_{date}.npz: Calibrated brightness temperature records
"""
import sys, os, time, gc, random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsd_reader import _download_and_extract_one

OUTPUT_DIR = 'data/validation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4 validation domains
DOMAINS = {
    'east_asia':   {'lat': (20, 45), 'lon': (110, 145)},
    'south_asia':  {'lat': (5, 30), 'lon': (65, 90)},
    'maritime':    {'lat': (-10, 10), 'lon': (95, 140)},
    'australia':   {'lat': (-25, -10), 'lon': (125, 150)},
}

WEEKS = {
    'east_asia': ['20230729','20230730','20230731','20230801','20230802','20230803','20230804'],
    'south_asia': ['20230707','20230708','20230709','20230710','20230711','20230712','20230713'],
    'maritime': ['20230226','20230227','20230228','20230301','20230302','20230303','20230304'],
    'australia': ['20220224','20220225','20220226','20220227','20220228','20220301','20220302'],
}

# AHI full-disk area for pixel location
from pyresample.geometry import AreaDefinition
AREA = AreaDefinition("full", "", "geos",
    "+proj=geos +lon_0=140.7 +h=35785863 +a=6378137 +b=6356752.3 +no_defs",
    5500, 5500, [-5500000, -5500000, 5500000, 5500000])

N_PIXELS = 1000


def get_pixels_for_day(date_str, domain):
    """Generate 1,000 random domain pixels deterministically seeded by date and domain."""
    bounds = DOMAINS[domain]
    
    # Grid to find valid pixels within domain bounds
    rng = random.Random(date_str + domain)
    lats_r = np.array([rng.uniform(bounds['lat'][0], bounds['lat'][1]) for _ in range(10000)])
    lons_r = np.array([rng.uniform(bounds['lon'][0], bounds['lon'][1]) for _ in range(10000)])
    
    cols, rows = AREA.get_array_indices_from_lonlat(lons_r, lats_r)
    valid = ~rows.mask
    rows_v = rows[valid].data.astype(int)
    cols_v = cols[valid].data.astype(int)
    lats_v = lats_r[valid]
    lons_v = lons_r[valid]
    
    if len(rows_v) < N_PIXELS:
        print(f"  WARNING: only {len(rows_v)} valid pixels for {domain}, padding")
        rows_v = np.pad(rows_v, (0, N_PIXELS - len(rows_v)), 'wrap')
        cols_v = np.pad(cols_v, (0, N_PIXELS - len(cols_v)), 'wrap')
        lats_v = np.pad(lats_v, (0, N_PIXELS - len(lats_v)), 'wrap')
        lons_v = np.pad(lons_v, (0, N_PIXELS - len(lons_v)), 'wrap')
    
    idx = rng.sample(range(len(rows_v)), N_PIXELS)
    
    seg = rows_v[idx] // 550 + 1
    seg_row = rows_v[idx] % 550
    seg_col = cols_v[idx]
    
    pixel_info = np.zeros(N_PIXELS, dtype=[('lat','f4'),('lon','f4'),('seg','i4'),('row','i4'),('col','i4')])
    pixel_info['lat'] = lats_v[idx]
    pixel_info['lon'] = lons_v[idx]
    pixel_info['seg'] = seg
    pixel_info['row'] = seg_row
    pixel_info['col'] = seg_col
    
    return pixel_info


def process_one_validation_day(date_str, domain):
    """Download one validation day with per-day random pixels."""
    print(f"\n{'='*60}")
    print(f"Processing {date_str} ({domain})")
    t0 = time.time()
    
    pix = get_pixels_for_day(date_str, domain)
    
    # Group by segment
    seg_pixels = {}
    seg_idx_map = {}
    for i, p in enumerate(pix):
        seg = f'S{int(p["seg"]):02d}10'
        seg_pixels.setdefault(seg, ([], []))[0].append(int(p['row']))
        seg_pixels[seg][1].append(int(p['col']))
        seg_idx_map.setdefault(seg, []).append(i)
    seg_pixels = {s: (np.array(r), np.array(c)) for s, (r, c) in seg_pixels.items()}
    
    n_ts = 144
    all_bands = {}
    
    for hour in range(24):
        for minute in [0, 10, 20, 30, 40, 50]:
            ts = hour * 6 + minute // 10
            time_str = f"{hour:02d}{minute:02d}"
            
            t1 = time.time()
            
            # Submit 36 files (4 segs × 9 bands) in parallel
            futs = []
            seg_results = {}
            with ThreadPoolExecutor(max_workers=8) as ex:
                for seg, (rows, cols) in seg_pixels.items():
                    for band in range(8, 17):
                        key = f"AHI-L1b-FLDK/{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}/{time_str}/HS_H09_{date_str}_{time_str}_B{band:02d}_FLDK_R20_{seg}.DAT.bz2"
                        seg_results[(seg, band)] = None
                        fut = ex.submit(_download_and_extract_one, key, rows, cols, band)
                        futs.append((fut, seg, band))
                for fut, seg, band in futs:
                    _, bt = fut.result()
                    seg_results[(seg, band)] = bt
            
            # Merge per-segment results
            for band in range(8, 17):
                if band not in all_bands:
                    all_bands[band] = np.full((n_ts, N_PIXELS), np.nan)
                arr = np.full(N_PIXELS, np.nan)
                for seg, idxs in seg_idx_map.items():
                    arr[idxs] = seg_results[(seg, band)]
                all_bands[band][ts] = arr
            
            done = ts + 1
            eta = (n_ts - done) * (time.time() - t0) / done
            print(f"  {date_str}_{time_str}: {time.time()-t1:.1f}s [{done}/{n_ts}, ETA {eta/60:.1f}min]", flush=True)
            gc.collect()
    
    total = time.time() - t0
    print(f"\n{date_str} ({domain}): {done}/{n_ts} in {total/60:.1f}min")
    
    if done > 10:
        outfile = os.path.join(OUTPUT_DIR, f"{domain}_{date_str}.npz")
        save_dict = {f'B{b:02d}': all_bands[b] for b in sorted(all_bands.keys())}
        save_dict['pixel_lat'] = pix['lat']
        save_dict['pixel_lon'] = pix['lon']
        save_dict['pixel_seg'] = pix['seg']
        np.savez_compressed(outfile, **save_dict)
        print(f"Saved {outfile} ({os.path.getsize(outfile)/1024:.0f} KB)")
    
    return done


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Extract cross-domain validation cubes from Himawari-9 HSD archives.")
    parser.add_argument('--domains', nargs='+', default=['east_asia', 'maritime', 'australia'],
                        choices=['east_asia', 'maritime', 'australia'],
                        help='Validation domains to extract')
    parser.add_argument('--days', nargs='+', default=None, help='Specific dates to extract (YYYYMMDD).')
    args = parser.parse_args()

    total_ts = 0
    for domain in args.domains:
        day_list = args.days if args.days else WEEKS.get(domain, [])
        print(f"\n>>> Domain: {domain} ({len(day_list)} days)")
        for day in day_list:
            total_ts += process_one_validation_day(day, domain)

    print(f"\n{'='*60}")
    print(f"Done! Processed {total_ts} validation timestamps")
