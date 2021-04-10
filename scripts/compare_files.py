#!/usr/bin/env python
import argparse
import csv
import glob
import math
import sys
from random import random
import numpy as np
import swisseph as swe
import roc_sd

swe.set_ephe_path("/home/dmc/astrolog/astephem")

PLANET_STR = """
0 Sun
1 Moon
2 Mercury
3 Venus
4 Mars
5 Jupiter
6 Saturn
7 Uranus
8 Neptune
9 Pluto
11 True Node
15 Chiron
16 Pholus
17 Ceres
18 Pallas
19 Juno
20 Vesta
"""

PLANETS = []
for line in PLANET_STR.split("\n"):
    toks = line.split()
    if len(toks) >= 2:
        PLANETS.append(int(toks[0]))

ROC_SD = {}

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


def compare_files(f1, dates1, f2, dates2, max_planet: int, degree_step: int, max_harmonic: int):
    if not (dates1 and dates2):
        return
    for p in PLANETS:
        if p > max_planet:
            break
        rads1 = radian_positions(dates1, p)
        rads2 = radian_positions(dates2, p)
        len_tuple = tuple(sorted([len(rads1), len(rads2)]))
        if len_tuple not in ROC_SD:
            ROC_SD[len_tuple] = roc_sd.roc_sd(*len_tuple, extra_samples=1000000)[0]
        null_sd = ROC_SD[len_tuple]
        denom = len(rads1) * len(rads2)
        for h in range(1, max_harmonic + 1):
            for d in range(0, 180, degree_step):  # don't need full circle because cos is antisymmetric
                dr = d * math.pi / 180
                pairs1 = cosine_pairs(dr, h, rads1, 1)
                pairs2 = cosine_pairs(dr, h, rads2, 2)
                n1 = 0
                n12 = 0
                for _, c in sorted(pairs1 + pairs2):
                    if c == 1:
                        n1 += 1
                    else:
                        n12 += n1
                roc = n12 / denom
                z = (roc - 0.5) / null_sd
                print(
                    f"{z:9.6f} {h:2d} {p:2d} {d:3d} {roc:8.6f} {len(dates1):6d} {f1:14s} {len(dates2):6d} {f2:14s}"
                )
                sys.stdout.flush()
    return True


def cosine_pairs(dr, h, rads, idx):
    return [(math.cos(h * rad + dr) + 0.00001 * random(), idx) for rad in rads]


def radian_positions(dates1, p):
    return [swe.calc(date, p)[0][0] / (2 * math.pi) for date in dates1]


def size_ratio_ok(sz1: int, sz2: int, max_size_ratio: float) -> bool:
    return sz1 * max_size_ratio >= sz2 and sz2 * max_size_ratio >= sz1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_size_ratio", type=float, default=2.0)
    parser.add_argument("--max_planet", type=int, default=9)
    parser.add_argument("--max_harmonic", type=int, default=6)
    parser.add_argument("--degree_step", type=int, default=15)
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    files = args.files
    dates = {}
    for f in files:
        dates[f] = get_dates(f)
    for i, f1 in enumerate(files):
        for f2 in files[i + 1 :]:
            if size_ratio_ok(len(dates[f1]), len(dates[f2]), args.max_size_ratio):
                compare_files(f1, dates[f1], f2, dates[f2], args.max_planet, args.degree_step, args.max_harmonic)


if __name__ == "__main__":
    main()
