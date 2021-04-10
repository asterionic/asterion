#!/usr/bin/env python

import csv
import glob
import math
import sys
from random import random
import numpy as np
import swisseph as swe

swe.set_ephe_path("/home/dmc/astrolog/astephem")

"""
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

PLANETS = [0, 1, 2, 3, 4, 5, 6]  # ,7,8,9,11,15,16,17,18,19,20]


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


def compare_files(f1, f2):
    dates1 = get_dates(f1)
    dates2 = get_dates(f2)
    if not (dates1 and dates2):
        return True
    if len(dates1) > 2 * len(dates2) or len(dates2) > 2 * len(dates1):
        return False
    for p in PLANETS:
        rads1 = [swe.calc(date, p)[0][0] / (2 * math.pi) for date in dates1]
        rads2 = [swe.calc(date, p)[0][0] / (2 * math.pi) for date in dates2]
        for h in range(1, 7):
            for d in range(
                0, 180, 15
            ):  # don't need full circle because cos is antisymmetric
                dr = d / (2 * math.pi)
                pairs1 = [
                    (math.cos(h * rad + dr) + 0.00001 * random(), 1) for rad in rads1
                ]
                pairs2 = [
                    (math.cos(h * rad + dr) + 0.00001 * random(), 2) for rad in rads2
                ]
                order = [c for (_, c) in sorted(pairs1 + pairs2)]
                n1 = 0
                n12 = 0
                for c in order:
                    if c == 1:
                        n1 += 1
                    else:
                        n12 += n1
                roc = n12 / (n1 * (len(order) - n1))
                print(
                    f"{h} {p:2d} {d:3d} {roc:8.6f} {len(dates1):6d} {f1:14s} {len(dates2):6d} {f2:14s} "
                )
                sys.stdout.flush()
    return True


def main():
    files = sys.argv[1:]  # assume longest first or sorted(glob.glob("*.csv"))
    is_first = True
    for i, f1 in enumerate(files):
        for f2 in files[i + 1 :]:
            if is_first:
                is_first = False  # don't repeat
            elif not compare_files(f1, f2):
                break


if __name__ == "__main__":
    main()
