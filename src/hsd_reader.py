#!/usr/bin/env python3
"""
Himawari Standard Data (HSD) decompression, header parsing, and brightness temperature calibration.
"""
import io, bz2, struct, time, os, gc, pickle, threading
import numpy as np
import boto3
from botocore.handlers import disable_signing
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── HSD header dtypes ──────────────────────────────────────────────
_BASIC = np.dtype([("hblock_number", "u1"), ("blocklength", "<u2"),
    ("total_number_of_hblocks", "<u2"), ("byte_order", "u1"),
    ("satellite", "S16"), ("proc_center_name", "S16"),
    ("observation_area", "S4"), ("other_observation_info", "S2"),
    ("observation_timeline", "<u2"), ("observation_start_time", "f8"),
    ("observation_end_time", "f8"), ("file_creation_time", "f8"),
    ("total_header_length", "<u4"), ("total_data_length", "<u4"),
    ("quality_flag1", "u1"), ("quality_flag2", "u1"),
    ("quality_flag3", "u1"), ("quality_flag4", "u1"),
    ("file_format_version", "S32"), ("file_name", "S128"), ("spare", "S40")])

_DATA = np.dtype([("hblock_number", "u1"), ("blocklength", "<u2"),
    ("number_of_bits_per_pixel", "<u2"), ("number_of_columns", "<u2"),
    ("number_of_lines", "<u2"), ("compression_flag_for_data", "u1"),
    ("spare", "S40")])

_CAL = np.dtype([("hblock_number", "u1"), ("blocklength", "<u2"),
    ("band_number", "<u2"), ("central_wave_length", "f8"),
    ("valid_number_of_bits_per_pixel", "<u2"),
    ("count_value_error_pixels", "<u2"),
    ("count_value_outside_scan_pixels", "<u2"),
    ("gain_count2rad_conversion", "f8"),
    ("offset_count2rad_conversion", "f8")])

_IRCAL = np.dtype([("c0_rad2tb_conversion", "f8"), ("c1_rad2tb_conversion", "f8"),
    ("c2_rad2tb_conversion", "f8"), ("c0_tb2rad_conversion", "f8"),
    ("c1_tb2rad_conversion", "f8"), ("c2_tb2rad_conversion", "f8"),
    ("speed_of_light", "f8"), ("planck_constant", "f8"),
    ("boltzmann_constant", "f8"), ("spare", "S40")])

BUCKET = 'noaa-himawari9'


def _make_s3():
    s3 = boto3.client('s3', config=Config(
        max_pool_connections=64,
        retries={'max_attempts': 3},
        read_timeout=30,
        connect_timeout=10))
    s3.meta.events.register('choose-signer.s3.*', disable_signing)
    return s3

_S3 = None
_HEADER_CACHE = {}


_thread_local = threading.local()

def _get_s3():
    global _S3
    if _S3 is None:
        _S3 = _make_s3()
    return _S3

def _get_s3_thread():
    """Get a per-thread S3 client (thread-safe)."""
    if not hasattr(_thread_local, 's3'):
        s3 = boto3.client('s3', config=Config(
            max_pool_connections=32, retries={'max_attempts': 2},
            read_timeout=30, connect_timeout=10))
        s3.meta.events.register('choose-signer.s3.*', disable_signing)
        _thread_local.s3 = s3
    return _thread_local.s3


def parse_headers(raw):
    """Parse HSD headers, return header dict and data_offset."""
    pos = 0
    blk1 = np.frombuffer(raw[pos:pos+_BASIC.itemsize], dtype=_BASIC, count=1)
    total_hdr = int(blk1["total_header_length"][0])
    pos += int(blk1["blocklength"][0])

    blk2 = np.frombuffer(raw[pos:pos+_DATA.itemsize], dtype=_DATA, count=1)
    ncol = int(blk2["number_of_columns"][0])
    nrow = int(blk2["number_of_lines"][0])
    pos += int(blk2["blocklength"][0])

    # Skip block3 (proj), block4 (nav)
    blk3_len = struct.unpack('<H', raw[pos+1:pos+3])[0]
    pos += blk3_len
    blk4_len = struct.unpack('<H', raw[pos+1:pos+3])[0]
    pos += blk4_len

    blk5 = np.frombuffer(raw[pos:pos+_CAL.itemsize], dtype=_CAL, count=1)
    cal_pos = pos + _CAL.itemsize
    ircal = np.frombuffer(raw[cal_pos:cal_pos+_IRCAL.itemsize], dtype=_IRCAL, count=1)
    pos += int(blk5["blocklength"][0])

    # Skip blocks 6-11
    for _ in range(6):
        bl = struct.unpack('<H', raw[pos+1:pos+3])[0]
        if bl == 0:
            break
        pos += bl

    hdr = {
        'ncol': ncol, 'nrow': nrow,
        'gain': float(blk5["gain_count2rad_conversion"][0]),
        'offset': float(blk5["offset_count2rad_conversion"][0]),
        'err_val': int(blk5["count_value_error_pixels"][0]),
        'outside_val': int(blk5["count_value_outside_scan_pixels"][0]),
        'cwl': float(blk5["central_wave_length"][0]),
        'c0': float(ircal["c0_rad2tb_conversion"][0]),
        'c1': float(ircal["c1_rad2tb_conversion"][0]),
        'c2': float(ircal["c2_rad2tb_conversion"][0]),
        'data_offset': total_hdr,
    }
    return hdr


def get_cached_header(s3_key, s3=None):
    """Parse and cache headers from S3 key."""
    if s3_key in _HEADER_CACHE:
        return _HEADER_CACHE[s3_key]
    if s3 is None:
        s3 = _get_s3()
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, s3_key, buf)
    raw = bz2.decompress(buf.getvalue())
    hdr = parse_headers(raw)
    _HEADER_CACHE[s3_key] = hdr
    return hdr


def counts_to_bt(counts, hdr):
    """Convert uint16 counts to brightness temperature."""
    data = np.asarray(counts, dtype=np.float32)
    invalid = (data == hdr['err_val']) | (data == hdr['outside_val'])
    if invalid.any():
        data = data.copy()
        data[invalid] = np.nan
    cwl_m = hdr['cwl'] * 1e-6
    rad = data * np.float32(hdr['gain']) + np.float32(hdr['offset'])
    a = 6.62606957e-34 * 299792458.0 / (1.3806488e-23 * cwl_m)
    rad_safe = np.where(np.isfinite(rad) & (rad > 0), rad, np.nan)
    b = (2 * 6.62606957e-34 * 299792458.0**2) / (rad_safe * 1e6 * cwl_m**5) + 1
    bt = np.where(np.isfinite(b) & (b > 1), a / np.log(b), np.nan)
    bt = hdr['c0'] + hdr['c1'] * bt + hdr['c2'] * bt**2
    return np.clip(bt, 0, None)


def _download_and_extract_one(key, seg_rows, seg_cols, band):
    """Single file: download, decompress, parse, extract pixels."""
    try:
        buf = io.BytesIO()
        _get_s3_thread().download_fileobj(BUCKET, key, buf)
        raw = bz2.decompress(buf.getvalue())
        hdr = parse_headers(raw)
        d = hdr['data_offset']
        raw_data = raw[d:d + hdr['nrow'] * hdr['ncol'] * 2]
        counts = np.frombuffer(raw_data, dtype='<u2').reshape(hdr['nrow'], hdr['ncol'])
        pixel_counts = counts[seg_rows, seg_cols]
        bt = counts_to_bt(pixel_counts, hdr)
        return band, bt
    except Exception as e:
        print(f"  FAIL B{band:02d}: {e}")
        return band, np.full(len(seg_rows), np.nan)


def fetch_full_segment_bt(date_str, time_str, seg, band):
    """Download and return full BT 2D array for one band/segment."""
    key = f"AHI-L1b-FLDK/{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}/{time_str}/HS_H09_{date_str}_{time_str}_B{band:02d}_FLDK_R20_{seg}.DAT.bz2"
    buf = io.BytesIO()
    _get_s3().download_fileobj(BUCKET, key, buf)
    raw = bz2.decompress(buf.getvalue())
    hdr = parse_headers(raw)
    d = hdr['data_offset']
    raw_data = raw[d:d + hdr['nrow'] * hdr['ncol'] * 2]
    counts = np.frombuffer(raw_data, dtype='<u2').reshape(hdr['nrow'], hdr['ncol'])
    return counts_to_bt(counts, hdr)

def process_timestamp(date_str, time_str, seg_pixels, seg_idx_map, max_workers=8):
    """Process one timestamps: all 9 IR bands across given segments.

    seg_pixels: {seg_name: (rows_array, cols_array)}
    seg_idx_map: {seg_name: indices_in_final_array}
    Returns dict {band_num: bt_array_of_N_pixels}
    """
    futs = []
    # per-segment storage: results[(seg, band)] = bt_array
    seg_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for seg, (rows, cols) in seg_pixels.items():
            for band in range(8, 17):
                key = f"AHI-L1b-FLDK/{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}/{time_str}/HS_H09_{date_str}_{time_str}_B{band:02d}_FLDK_R20_{seg}.DAT.bz2"
                seg_results[(seg, band)] = rows  # placeholder
                fut = ex.submit(_download_and_extract_one, key, rows, cols, band)
                futs.append((fut, seg, band))
        for fut, seg, band in futs:
            _, bt = fut.result()
            seg_results[(seg, band)] = bt

    # Merge per-segment results into full-pixel arrays
    all_indices = sorted(set(idx for idxs in seg_idx_map.values() for idx in idxs))
    n_pixels = len(all_indices)
    results = {}
    for band in range(8, 17):
        arr = np.full(n_pixels, np.nan)
        for seg, idxs in seg_idx_map.items():
            arr[idxs] = seg_results[(seg, band)]
        results[band] = arr
    return results
