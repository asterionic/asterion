#!/usr/bin/env python
import argparse
import cmath
import collections
import csv
import math
import os
import sys
from random import random, shuffle
from typing import Tuple

import numpy as np
import swisseph as swe

import roc_sd

swe.set_ephe_path("/home/dmc/astrolog/astephem")

# Statistics which should be independent of long-term effects:
# 1) Su-Mo-Me-Vn-Ma, singly and all pairs [0-4]
# 2) Ce-Pa-Jn-Vs singly (not pairs) [17-20]
# 3) Pairs between 1) and 3) except Ma-Vs [4-20] which is close to factor of 2

PLANET_STR = """
0 Sun 1.0
1 Moon 0.074
2 Mercury 0.241
3 Vnus 0.699
4 Mars 1.88
5 Jpiter 11.83
6 Saturn 29.46
7 Uranus 84
8 Neptune 165 
9 Pluto 248
11 Node 18.6
15 Chiron 50
16 Pholus 92
17 Ceres 4.6
18 Pallas 4.62
19 Jno 4.36
20 Vsta 3.63
"""

PLANETS = []
ORBITAL_PERIOD = {}
ABBREV = {}
for line in PLANET_STR.split("\n"):
    toks = line.split()
    if len(toks) >= 2:
        PLANETS.append(int(toks[0]))
    if len(toks) >= 3:
        ORBITAL_PERIOD[int(toks[0])] = float(toks[2])
        ABBREV[int(toks[0])] = toks[1][:2]

ROC_SD = {
    (192052, 208507): 0.0009,
    (12277, 23601): 0.002984,
    (12382, 23601): 0.003465,
    (12968, 23601): 0.003099,
    (12968, 25907): 0.003469,
    (13139, 23601): 0.003463,
    (13139, 25907): 0.003227,
    (13139, 26031): 0.002989,
    (13139, 26057): 0.002989,
    (13802, 23601): 0.002970,
    (13802, 25907): 0.003022,
    (13802, 26031): 0.003125,
    (13802, 26057): 0.003568,
    (15184, 23601): 0.002597,
    (15184, 25907): 0.002804,
    (15184, 26031): 0.003020,
    (15184, 26057): 0.003079,
    (15184, 28205): 0.003013,
    (15184, 29454): 0.003254,
    (16089, 23601): 0.002922,
    (16089, 25907): 0.002805,
    (16089, 26031): 0.002640,
    (16089, 26057): 0.002674,
    (16089, 28205): 0.002615,
    (16089, 29454): 0.003047,
    (16175, 23601): 0.003365,
    (16175, 25907): 0.002948,
    (16175, 26031): 0.002855,
    (16175, 26057): 0.002811,
    (16175, 28205): 0.002966,
    (16175, 29454): 0.002484,
    (192052, 208507): 0.000900,
    (20456, 23601): 0.003033,
    (20456, 25907): 0.002539,
    (20456, 26031): 0.002928,
    (20456, 26057): 0.002497,
    (20456, 28205): 0.002666,
    (20456, 29454): 0.002602,
    (20456, 37785): 0.002458,
    (20456, 38231): 0.002581,
    (21654, 23399): 0.002624,
    (21654, 23601): 0.002828,
    (21654, 25907): 0.002493,
    (21654, 26031): 0.002555,
    (21654, 26057): 0.002307,
    (21654, 28205): 0.002526,
    (21654, 29454): 0.002684,
    (21654, 37785): 0.002432,
    (21654, 38231): 0.002326,
    (23227, 23399): 0.002747,
    (23227, 23601): 0.002683,
    (23227, 25907): 0.002863,
    (23227, 26031): 0.002613,
    (23227, 26057): 0.002530,
    (23227, 28205): 0.002635,
    (23227, 29454): 0.002556,
    (23227, 37785): 0.002266,
    (23227, 38231): 0.002267,
    (23399, 23601): 0.002812,
    (23399, 25907): 0.002780,
    (23399, 26031): 0.002638,
    (23399, 26057): 0.002442,
    (23399, 28205): 0.002612,
    (23399, 29454): 0.002659,
    (23399, 37785): 0.002573,
    (23399, 38231): 0.002474,
    (23601, 25907): 0.002914,
    (23601, 26031): 0.002405,
    (23601, 26057): 0.002190,
    (23601, 28205): 0.002309,
    (23601, 29454): 0.002471,
    (23601, 37785): 0.002134,
    (23601, 38231): 0.002514,
    (25907, 26031): 0.002567,
    (25907, 26057): 0.002594,
    (25907, 28205): 0.002385,
    (25907, 29454): 0.002448,
    (25907, 37785): 0.002238,
    (25907, 38231): 0.002005,
    (26031, 26057): 0.002642,
    (26031, 28205): 0.002640,
    (26031, 29454): 0.002563,
    (26031, 37785): 0.002002,
    (26031, 38231): 0.002245,
    (26057, 28205): 0.002468,
    (26057, 29454): 0.002274,
    (26057, 37785): 0.002362,
    (26057, 38231): 0.002447,
    (28205, 29454): 0.002264,
    (28205, 37785): 0.002303,
    (28205, 38231): 0.002210,
    (29454, 37785): 0.002363,
    (29454, 38231): 0.002375,
    (37785, 38231): 0.002106,
    (37785, 61516): 0.001736,
    (38231, 61516): 0.002005,
}


def get_dates(f):
    dates = []
    with open(f) as fh:
        for _, day in csv.reader(fh):
            try:
                y, m, d = day.split("-")
                if int(y) > 1800 and int(m) > 0 and int(d) > 0:
                    dates.append(swe.julday(int(y), int(m), int(d), 12.0))
            except:
                pass
    return dates


def year_buckets(dates, bucket_size_in_years):
    days_in_year = 365.2422
    zero_date = swe.julday(1900, 1, 1)
    buckets = collections.defaultdict(list)
    for date in dates:
        diff = (date - zero_date) / (bucket_size_in_years * days_in_year)
        if diff < 0:
            bucket = -(int(-diff) + 1)
        else:
            bucket = int(diff)
        buckets[bucket].append(date)
    return buckets


def month_buckets(dates, bucket_size_in_months):
    days_in_year = 365.2422
    zero_date = swe.julday(1900, 1, 1)
    buckets = collections.defaultdict(list)
    for date in dates:
        year_fraction = (date - zero_date) / days_in_year + 1000
        year_fraction -= int(year_fraction)
        bucket = int(year_fraction * 12 / bucket_size_in_months)
        buckets[bucket].append(date)
    return buckets


def match_by_time_buckets(dates1: np.ndarray, dates2: np.ndarray,
                          bucket_years: float, bucket_months: int) -> Tuple[np.ndarray, np.ndarray]:
    if bucket_years > 0:
        len1, len2 = len(dates1), len(dates2)
        buckets1 = year_buckets(dates1, bucket_years)
        buckets2 = year_buckets(dates2, bucket_years)
        dates1, dates2 = apply_buckets(buckets1, buckets2, len1, len2)
        print(f"# With bucket_years  = {bucket_years:2d}, "
              f"reduced counts from {len1:6d} and {len2:6d} to {len(dates1):6d} and {len(dates2):6d}")
    if bucket_months > 0:
        len1, len2 = len(dates1), len(dates2)
        buckets1 = month_buckets(dates1, bucket_months)
        buckets2 = month_buckets(dates2, bucket_months)
        dates1, dates2 = apply_buckets(buckets1, buckets2, len1, len2)
        print(f"# With bucket_months = {bucket_months:2d}, " 
              f"reduced counts from {len1:6d} and {len2:6d} to {len(dates1):6d} and {len(dates2):6d}")
    return dates1, dates2

def apply_buckets(buckets1, buckets2, n1, n2):
    new_dates1 = []
    new_dates2 = []
    for bucket in buckets1:
        bucket1 = buckets1[bucket]
        bucket2 = buckets2[bucket]
        len1 = len(bucket1)
        prop1 = len1 / n1
        len2 = len(bucket2)
        prop2 = len2 / n2
        if prop1 < prop2:
            shuffle(bucket2)
            new_len2 = int(0.5 + len2 * prop1 / prop2)
            if new_len2 < len2:
                bucket2 = bucket2[:new_len2]
        elif prop2 < prop1:
            shuffle(bucket1)
            new_len1 = int(0.5 + len1 * prop2 / prop1)
            if new_len1 < len1:
                bucket1 = bucket1[:new_len1]
        new_dates1.extend(bucket1)
        new_dates2.extend(bucket2)
    return np.array(new_dates1), np.array(new_dates2)


def calculate_mean_positions(rads: np.ndarray) -> np.ndarray:
    return np.array([float(np.mean(np.cos(rads))), float(np.mean(np.sin(rads)))])


def effective_orbital_period(p1, p2):
    # In time op1, p1 will go round once and p2 will go round op1/op2 times, so
    # difference goes round (op1-op2)/op2 times. So OP of difference is op1*op2/(op1-op2).
    op1 = ORBITAL_PERIOD[p1]
    op2 = ORBITAL_PERIOD[p2]
    if p1 in [2, 3] and p2 == 0:
        return op1
    if p2 in [2, 3] and p1 == 0:
        return op2
    if p1 in [2, 3] and p2 not in [0, 2, 3]:
        op1 = 1  # p1 is Me or Vn; goes roughly like Su
    elif p2 in [2, 3] and p1 not in [0, 2, 3]:
        op2 = 1  # p2 is Me or Vn; goes roughly like Su
    return op1 * op2 / abs(op1 - op2)


def compare_files(args: argparse.Namespace, f1, dates1, f2, dates2):
    if not (dates1 and dates2):
        return
    max_planet = args.max_planet
    max_orbit = args.max_orbit
    degree_step = args.degree_step
    max_harmonic = args.max_harmonic
    min_harmonic = args.min_harmonic
    rads1_dct = {}
    rads2_dct = {}
    planets_to_use = [p for p in PLANETS if p <= max_planet]
    dates1, dates2 = match_by_time_buckets(dates1, dates2, args.match_by_years, args.match_by_months)
    if args.shuffle:
        dates12 = np.concatenate([dates1, dates2])
        shuffle(dates12)
        dates1, dates2 = dates12[:len(dates1)], dates12[len(dates1):]
    if min(len(dates1), len(dates2)) < args.min_dataset_size:
        return
    for h in range(min_harmonic, max_harmonic + 1):
        means1_dct = {}
        means2_dct = {}
        for p in planets_to_use:
            if p not in rads1_dct:
                rads1_dct[p] = radian_positions(dates1, p)
                rads2_dct[p] = radian_positions(dates2, p)
            if p not in means1_dct:
                means1_dct[p] = calculate_mean_positions(rads1_dct[p] * h)
                means2_dct[p] = calculate_mean_positions(rads2_dct[p] * h)
            rads1 = rads1_dct[p]
            rads2 = rads2_dct[p]
            len_tuple = tuple(sorted([len(rads1), len(rads2)]))
            if len_tuple not in ROC_SD:
                ROC_SD[len_tuple] = roc_sd.roc_sd(*len_tuple, extra_samples=1000000)[0]
                print(f"# null_sd{len_tuple} = {ROC_SD[len_tuple]:8.6f}")
            null_sd = ROC_SD[len_tuple]
            sys.stdout.flush()
            denom = len(rads1) * len(rads2)
            for p2 in planets_to_use:
                if p2 <= p:
                    continue
                eop = effective_orbital_period(p, p2)
                if eop > h * args.max_orbit or abs(eop - 1) < 1 / (h * args.max_orbit):
                    continue
                if p2 not in rads1_dct:
                    rads1_dct[p2] = radian_positions(dates1, p2)
                    rads2_dct[p2] = radian_positions(dates2, p2)
                rads1_diff = rads1 - rads1_dct[p2]
                rads2_diff = rads2 - rads2_dct[p2]
                # p_values = scipy.stats.norm.sf(abs(z_scores)) #one-sided
                # p_values = scipy.stats.norm.sf(abs(z_scores))*2 #twosided
                roc = calculate_roc(denom, 0.0, h, rads1_diff, rads2_diff)
                z = (roc - 0.5) / null_sd
                print(
                    f"{z:9.5f} {h:2d} {ABBREV[p]:2s}   {ABBREV[p2]:2s} {roc:8.6f} {len(dates1):6d} {f1:14s} {len(dates2):6d} {f2:14s} {eop / h:7.3f}"
                )
                sys.stdout.flush()
            eop = ORBITAL_PERIOD[0 if p in [2, 3] else p]
            if eop > args.max_orbit * h or abs(eop - 1) < 1 / (h * args.max_orbit):
                continue
            vector = means2_dct[p] - means1_dct[p]
            vcomp = complex(vector[0], vector[1])
            magnitude, angle = cmath.polar(vcomp)
            data = np.concatenate([rads1_dct[p], rads2_dct[p]])
            diff_sd = np.sqrt((np.var(np.cos(data)) + np.var(np.sin(data))) * (1 / len(dates1) + 1 / len(dates2)))
            diff_z = magnitude / diff_sd
            print(f"{diff_z:9.5f} {h:2d} {ABBREV[p]:2s} {int(0.5+angle*180/math.pi):4d} {magnitude:8.6f} {len(dates1):6d} {f1:14s} {len(dates2):6d} {f2:14s} {eop / h:7.3f}")
            if degree_step > 0:
                for d in range(0, 180, degree_step):  # don't need full circle because cos is antisymmetric
                    dr = d * math.pi / 180
                    roc = calculate_roc(denom, dr, h, rads1, rads2)
                    z = (roc - 0.5) / null_sd
                    print(
                        f"{z:9.5f} {h:2d} {ABBREV[p]:2s} {d:4d} {roc:8.6f} {len(dates1):6d} {f1:14s} {len(dates2):6d} {f2:14s} {eop / h:7.3f}"
                    )
                    sys.stdout.flush()


def calculate_roc(denom, dr, h, rads1, rads2):
    pairs1 = cosine_pairs(dr, h, rads1, 1)
    pairs2 = cosine_pairs(dr, h, rads2, 2)
    n1 = 0
    n12 = 0
    for _, c in sorted(pairs1 + pairs2):
        if c == 1:
            n1 += 1
        else:
            n12 += n1
    roc1 = n12 / denom
    roc = roc1
    return roc


def cosine_pairs(dr, h, rads, idx):
    return [(math.cos(h * rad + dr) + 0.00001 * random(), idx) for rad in rads]


def radian_positions(dates1, p):
    return np.array([swe.calc(date, p)[0][0] * math.pi / 180 for date in dates1])


def size_ratio_ok(sz1: int, sz2: int, max_size_ratio: float) -> bool:
    return sz1 * max_size_ratio >= sz2 and sz2 * max_size_ratio >= sz1


def main():
    args = parse_arguments()
    files = args.files
    dates = {}
    for f in files:
        dates[f] = get_dates(f)
    files = sorted(files, key=lambda f: -len(dates[f]))
    for delta1 in range(1, len(files)):
        if args.pairs_first and delta1 < 3:
            delta = 3 - delta1
        else:
            delta = delta1
        for i, f1 in enumerate(files):
            #print(f"# delta = {delta}, i = {i}")
            if i + delta >= len(files):
                break
            f2 = files[i + delta]
            if size_ratio_ok(len(dates[f1]), len(dates[f2]), args.max_size_ratio):
                compare_files(
                    args,
                    os.path.basename(f1),
                    dates[f1],
                    os.path.basename(f2),
                    dates[f2],
                )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_size_ratio", type=float, default=5.0)
    parser.add_argument("--max_planet", type=int, default=20)
    parser.add_argument("--max_harmonic", type=int, default=11)
    parser.add_argument("--degree_step", type=int, default=0)
    parser.add_argument("--max_orbit", type=float, default=5)
    parser.add_argument("--min_harmonic", type=int, default=1)
    parser.add_argument("--match_by_years", type=int, default=0)
    parser.add_argument("--match_by_months", type=int, default=0)
    parser.add_argument("--min_dataset_size", type=int, default=100)
    parser.add_argument("--pairs_first", action="store_true", default=False)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    for a in sorted(dir(args)):
        if a != "files" and not a.startswith("_"):
            print(f"# --{a} {getattr(args, a)}")
    sys.stdout.flush()
    return args


if __name__ == "__main__":
    main()
