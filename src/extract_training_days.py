#!/usr/bin/env python3
"""
Download Himawari-9 HSD segments for designated training days and extract regional pixel time series.

Outputs:
  - data/training/{date}.npz: Calibrated brightness temperatures for 9 infrared bands
"""
import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsd_reader import process_timestamp

OUTPUT_DIR = 'data/training'
PIXEL_FILE = 'data/pixel_locations.npz'
os.makedirs(OUTPUT_DIR, exist_ok=True)

pix = np.load(PIXEL_FILE)['info']
print(f"Loaded {len(pix)} pixel locations")

# Build seg_pixels and seg_idx_map
seg_pixels = {}
seg_idx_map = {}
for i, p in enumerate(pix):
    seg = f'S{int(p["seg"]):02d}10'
    seg_pixels.setdefault(seg, ([], []))[0].append(int(p['row']))
    seg_pixels[seg][1].append(int(p['col']))
    seg_idx_map.setdefault(seg, []).append(i)

seg_pixels = {s: (np.array(r), np.array(c)) for s, (r, c) in seg_pixels.items()}
print(f"Segments: {sorted(seg_pixels.keys())}")


def process_one_timestamp(date_str, time_str):
    return process_timestamp(date_str, time_str, seg_pixels, seg_idx_map, max_workers=8)


def process_one_day(date_str):
    print(f"\n{'='*60}")
    print(f"Processing {date_str}")
    t0 = time.time()

    all_bands = {}
    n_ts = 144

    for hour in range(24):
        for minute in [0, 10, 20, 30, 40, 50]:
            ts = hour * 6 + minute // 10
            time_str = f"{hour:02d}{minute:02d}"

            t1 = time.time()
            try:
                results = process_one_timestamp(date_str, time_str)
            except Exception as e:
                print(f"  FAILED {date_str}_{time_str}: {e}")
                continue

            for band, bt in results.items():
                if band not in all_bands:
                    all_bands[band] = np.full((n_ts, len(pix)), np.nan)
                all_bands[band][ts] = bt

            elapsed = time.time() - t1
            done = ts + 1
            eta = (n_ts - done) * (time.time() - t0) / done
            print(f"  {date_str}_{time_str}: {elapsed:.1f}s [{done}/{n_ts}, ETA {eta/60:.1f}min]", flush=True)
            gc.collect()

    total = time.time() - t0
    print(f"\nDay {date_str}: {done}/{n_ts} timestamps in {total/60:.1f}min")

    if done > 10:
        outfile = os.path.join(OUTPUT_DIR, f"{date_str}.npz")
        save_dict = {f'B{b:02d}': all_bands[b] for b in sorted(all_bands.keys())}
        save_dict['pixel_lat'] = pix['lat']
        save_dict['pixel_lon'] = pix['lon']
        save_dict['pixel_seg'] = pix['seg']
        np.savez_compressed(outfile, **save_dict)
        print(f"Saved {outfile} ({os.path.getsize(outfile)/1024:.0f} KB)")
    else:
        print("Too few timestamps, skipping save")

    return done


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('days', nargs='+', help='Days: YYYYMMDD')
    args = parser.parse_args()

    total_ts = 0
    for day in args.days:
        total_ts += process_one_day(day)

    print(f"\nDone! Processed {total_ts} timestamps across {len(args.days)} days")
