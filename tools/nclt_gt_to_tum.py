#!/usr/bin/env python3
"""
Convert NCLT ground truth CSV to TUM trajectory format.

The NCLT ground truth (groundtruth_YYYY-MM-DD.csv) is RTK post-processed
lat/lon/alt. This script converts it to the same local ENU frame that
FusionCore uses (anchored at the first GPS fix), so ATE/RPE comparisons
are in the same coordinate system.

Usage:
  python3 tools/nclt_gt_to_tum.py \
    --gt    /path/to/nclt/2012-01-08/groundtruth_2012-01-08.csv \
    --gps   /path/to/nclt/2012-01-08/gps.csv \
    --out   ground_truth.tum

The ENU origin is the first valid GPS fix from gps.csv — the same anchor
FusionCore sets on its first fix.

Output format (TUM):  timestamp tx ty tz qx qy qz qw
Orientation is identity (1 0 0 0) — NCLT ground truth has no heading.
evo_ape uses only position for ATE; orientation doesn't affect the metric.
"""

import argparse
import csv
import math
import sys


def latlon_to_ecef(lat_deg: float, lon_deg: float, alt_m: float):
    """WGS84 lat/lon/alt → ECEF XYZ (meters)."""
    a = 6378137.0
    e2 = 6.6943799901414e-3
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    x = (N + alt_m) * math.cos(lat) * math.cos(lon)
    y = (N + alt_m) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - e2) + alt_m) * math.sin(lat)
    return x, y, z


def ecef_to_enu(px, py, pz, ref_lat_deg, ref_lon_deg, ref_alt_m):
    """ECEF point → ENU relative to reference origin."""
    rx, ry, rz = latlon_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    dx, dy, dz = px - rx, py - ry, pz - rz
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    east  = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up    =  cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east, north, up


def load_first_gps_fix(gps_csv: str):
    """Return (lat, lon, alt) of first valid GPS fix from gps.csv."""
    with open(gps_csv) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('#'):
                continue
            try:
                lat, lon, alt, mode = float(row[1]), float(row[2]), float(row[3]), int(float(row[4]))
                if mode >= 2:
                    return lat, lon, alt
            except (ValueError, IndexError):
                continue
    raise RuntimeError(f'No valid GPS fix found in {gps_csv}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gt',  required=True, help='NCLT groundtruth CSV file')
    parser.add_argument('--gps', required=True, help='NCLT gps.csv (to get ENU origin)')
    parser.add_argument('--out', required=True, help='Output TUM file')
    args = parser.parse_args()

    print(f'Finding ENU origin from {args.gps}...')
    ref_lat, ref_lon, ref_alt = load_first_gps_fix(args.gps)
    print(f'  ENU origin: lat={ref_lat:.6f} lon={ref_lon:.6f} alt={ref_alt:.1f}m')

    count = 0
    with open(args.gt) as fin, open(args.out, 'w') as fout:
        for row in csv.reader(fin):
            if not row or row[0].startswith('#'):
                continue
            try:
                utime = int(row[0])
                lat, lon, alt = float(row[1]), float(row[2]), float(row[3])
            except (ValueError, IndexError):
                continue

            # Convert to ENU
            ex, ey, ez = ecef_to_enu(
                *latlon_to_ecef(lat, lon, alt),
                ref_lat, ref_lon, ref_alt
            )

            # TUM format: timestamp tx ty tz qx qy qz qw
            ts = utime / 1e6
            fout.write(f'{ts:.6f} {ex:.6f} {ey:.6f} {ez:.6f} '
                       f'0.000000 0.000000 0.000000 1.000000\n')
            count += 1

    print(f'Written {count} ground truth poses to {args.out}')


if __name__ == '__main__':
    main()
