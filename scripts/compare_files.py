#!/usr/bin/env python
import argparse
import csv
import math
import os
import sys
from random import random

import numpy as np
import swisseph as swe

import roc_sd

swe.set_ephe_path("/home/dmc/astrolog/astephem")

PLANET_STR = """
0 Sun 1.0
1 Moon 0.074
2 Mercury 1.0
3 Vnus 1.0
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


def compare_files(args: argparse.Namespace, f1, dates1, f2, dates2):
    if not (dates1 and dates2):
        return
    max_planet = args.max_planet
    max_orbit = args.max_orbit
    degree_step = args.degree_step
    max_harmonic = args.max_harmonic
    min_sun_harmonic = args.min_sun_harmonic
    rads1_dct = {}
    rads2_dct = {}
    planets_to_use = [
        p for p in PLANETS if p <= max_planet and ORBITAL_PERIOD[p] <= max_orbit
    ]
    planets_to_use2 = [p for p in PLANETS if p <= max_planet]
    for h in range(1, max_harmonic + 1):
        for p in planets_to_use:
            if p not in rads1_dct:
                rads1_dct[p] = radian_positions(dates1, p)
                rads2_dct[p] = radian_positions(dates2, p)
                # print(f"# Calculated positions for planet {p}")
            rads1 = rads1_dct[p]
            rads2 = rads2_dct[p]
            len_tuple = tuple(sorted([len(rads1), len(rads2)]))
            if len_tuple not in ROC_SD:
                ROC_SD[len_tuple] = roc_sd.roc_sd(*len_tuple, extra_samples=1000000)[0]
                print(f"# null_sd{len_tuple} = {ROC_SD[len_tuple]:8.6f}")
            null_sd = ROC_SD[len_tuple]

            sys.stdout.flush()
            denom = len(rads1) * len(rads2)
            for p2 in planets_to_use2:
                if p2 <= p:
                    continue
                if p2 not in rads1_dct:
                    rads1_dct[p2] = radian_positions(dates1, p2)
                    rads2_dct[p2] = radian_positions(dates2, p2)
                    # print(f"# Calculated positions for planet {p2}")
                rads1_diff = rads1 - rads1_dct[p2]
                rads2_diff = rads2 - rads2_dct[p2]
                roc = calculate_roc(denom, 0.0, h, rads1_diff, rads2_diff)
                z = (roc - 0.5) / null_sd
                print(
                    f"{z:9.5f} {h:2d} {ABBREV[p]:2s}   {ABBREV[p2]:2s} {roc:8.6f} {len(dates1):6d} {f1:14s} {len(dates2):6d} {f2:14s}"
                )
                sys.stdout.flush()
            if h < min_sun_harmonic and p in [0, 2, 3]:
                continue  # avoid hemisphere effect for Sun and inner planets
            for d in range(
                0, 180, degree_step
            ):  # don't need full circle because cos is antisymmetric
                dr = d * math.pi / 180
                roc = calculate_roc(denom, dr, h, rads1, rads2)
                z = (roc - 0.5) / null_sd
                print(
                    f"{z:9.5f} {h:2d} {ABBREV[p]:2s} {d:4d} {roc:8.6f} {len(dates1):6d} {f1:14s} {len(dates2):6d} {f2:14s}"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_size_ratio", type=float, default=5.0)
    parser.add_argument("--max_planet", type=int, default=20)
    parser.add_argument("--max_harmonic", type=int, default=6)
    parser.add_argument("--degree_step", type=int, default=15)
    parser.add_argument("--max_orbit", type=float, default=10)
    parser.add_argument("--min_sun_harmonic", type=int, default=3)
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    files = args.files
    dates = {}
    # print("# Getting dates")
    # sys.stdout.flush()
    for f in files:
        dates[f] = get_dates(f)
    # print("# Finished getting dates")
    # sys.stdout.flush()
    files = sorted(files, key=lambda f: -len(dates[f]))
    for i, f1 in enumerate(files):
        for f2 in files[i + 1 :]:
            if size_ratio_ok(len(dates[f1]), len(dates[f2]), args.max_size_ratio):
                compare_files(
                    args,
                    os.path.basename(f1),
                    dates[f1],
                    os.path.basename(f2),
                    dates[f2],
                )


if __name__ == "__main__":
    main()
