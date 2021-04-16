#!/usr/bin/env python
import argparse
import cmath
import collections
import csv
import math
import os
import sys
from enum import Enum
from random import random, shuffle
from typing import Tuple, List, Dict
from pathlib import Path

import numpy as np
import swisseph as swe

from roc_estimates import ROC_SD
import roc_sd

swe.set_ephe_path("/home/dmc/astrolog/astephem")

SUN = 0
MERCURY = 2
VENUS = 3

"""
For each "planet" we're going to consider, we record its index in SwissEph, its full name, a unique two-letter
identifier, and its orbital period in years. (Astrologers use the term "planet" to include not only true astronomical
planets but all bodies of interest in the solar system, including the Sun, Moon, asteroids and other things too small
to be planets).
"""
PLANET_DATA = [
    (0, "Sun", "Su", 1.0),
    (1, "Moon", "Mo", 0.074),
    (2, "Mercury", "Me", 0.241),
    (3, "Venus", "Ve", 0.699),
    (4, "Mars", "Ma", 1.88),
    (5, "Jupiter", "Jp", 11.83),
    (6, "Saturn", "Sa", 29.46),
    (7, "Uranus", "Ur", 84),
    (8, "Neptune", "Ne", 165),
    (9, "Pluto", "Pl", 248),
    (11, "Node", "Nd", 18.6),
    (15, "Chiron", "Ch", 50),
    (16, "Pholus", "Ph", 92),
    (17, "Ceres", "Ce", 4.6),
    (18, "Pallas", "Pa", 4.62),
    (19, "Juno", "Jn", 4.36),
    (20, "Vesta", "Vs", 3.63),
]

PLANETS = [t[0] for t in PLANET_DATA]
PLANET_NAME = dict((t[0], t[1]) for t in PLANET_DATA)
ORBITAL_PERIOD = dict((t[0], t[3]) for t in PLANET_DATA)
ABBREV = dict((t[0], t[2]) for t in PLANET_DATA)


def get_paths_and_dates(files: List[str]) -> Tuple[List[Path], Dict[Path, np.ndarray]]:
    """
    :param files: a list of file paths; each should be a csv file containing data expected by get_dates
    :return: paths: a list of Path objects, one for each file, ordered by decreasing number of successfully
    extracted dates in the file; and dates: a dictionary from paths to arrays of day values.
    """
    paths = [Path(f) for f in files]
    dates = dict((p, get_dates(p)) for p in paths)
    paths.sort(key=lambda p: len(dates[p]), reverse=True)
    return paths, dates


def get_dates(path: Path):
    """
    :param path: a csv file, assumed to contain a date in the last column of each row.
    The date should be in the form yyyy-mm-dd. We keep dates in 1800 or later, as earlier
    dates may be inaccurate or use a different calendar. We assume there is no time information,
    so we set the time to a random number of hours between 0 and 24.
    :return: an array of "Julian" (actually Gregorian) day values, one for each date in the file.
    """
    dates = []
    with path.open() as inp:
        for row in csv.reader(inp):
            try:
                day = row[-1]
                y, m, d = day.split("-")
                if int(y) > 1800 and int(m) > 0 and int(d) > 0:
                    dates.append(swe.julday(int(y), int(m), int(d), 24.0 * random()))
            except: # not all rows may be of the expected format
                pass
    return np.array(dates)


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


def effective_orbital_period(p1: int, p2: int) -> float:
    """
    :param p1: number for one planet
    :param p2: number for another planet, different from p1
    :return: the effective period for the angle between the two planets, averaged over a long time,
    as seen from Earth. For example if p1 orbits in 5 years and p2 in 2 years, then over 1000 years,
    p1 will orbit 200 times and p2 500 times, and the angle (difference) between them will cycle around
    500-200=300 times, i.e. 10/3 years per cycle on average.
    """
    assert p1 != p2
    # In time op1, p1 will go round once and p2 will go round op1/op2 times, so
    # difference goes round (op1-op2)/op2 times. So OP of difference is op1*op2/(op1-op2).
    op1 = ORBITAL_PERIOD[p1]
    op2 = ORBITAL_PERIOD[p2]
    inner_planets = [MERCURY, VENUS]
    # Aspect between Mercury or Venus and Sun has period of Mercury or Venus respectively.
    if p1 in inner_planets and p2 == SUN:
        return op1
    if p2 in inner_planets and p1 == SUN:
        return op2
    # For an aspect an inner planet and an outer planet, the inner planet's motion is
    # Sun plus an oscillation, so its period is effectively that of the Sun.
    if p1 in inner_planets and p2 not in inner_planets + [SUN]:
        op1 = ORBITAL_PERIOD[SUN]
    elif p2 in inner_planets and p1 not in inner_planets + [SUN]:
        op2 = ORBITAL_PERIOD[SUN]
    # In time op1, p1 will go round once and p2 will go round op1/op2 times, so the difference between them goes
    # round (op1-op2)/op2 times. Therefore in time 1, the difference goes round (op1-op2)/(op1*op2) times, and
    # the effective orbital period is the inverse of that.
    return op1 * op2 / abs(op1 - op2)


def compare_files(args: argparse.Namespace, f1: str, dates1: np.ndarray, f2: str, dates2: np.ndarray):
    if min(len(dates1), len(dates2)) < args.min_dataset_size:
        return
    # Dictionary from planet numbers to an array of planet positions in radians, one for each date in dates1.
    rads1_dct = {}
    # Ditto for dates2.
    rads2_dct = {}
    # List of planets we're going to consider.
    planets_to_use = [p for p in PLANETS if p <= args.max_planet]
    # If time buckets (by numbers of years and/or time of year) were specified, we prune the data accordingly.
    dates1, dates2 = match_by_time_buckets(dates1, dates2, args.match_by_years, args.match_by_months)
    # We check sizes again, in case pruning has reduced them below the minimum.
    if min(len(dates1), len(dates2)) < args.min_dataset_size:
        return
    if args.shuffle:
        dates1, dates2 = shuffle_dates_between_categories(dates1, dates2)
    null_sd = null_hypothesis_roc_standard_deviation(len(dates1), len(dates2))
    for h in range(args.min_harmonic, args.max_harmonic + 1):
        # Dictionary from planet numbers to the mean (centroid) of the positions of that planet
        # on dates in dates1, treating the zodiac as a unit circle.
        means1_dct = {}
        # Ditto for dates2.
        means2_dct = {}
        for ia, pa in enumerate(planets_to_use):
            if pa not in rads1_dct:
                # Fill rads1_dct and rads2_dct for planet pa.
                calculate_planet_radians(dates1, pa, rads1_dct)
                calculate_planet_radians(dates2, pa, rads2_dct)
            if pa not in means1_dct:
                # Fill means1_dct and means2_dct for planet pa.
                means1_dct[pa] = calculate_mean_positions(rads1_dct[pa] * h)
                means2_dct[pa] = calculate_mean_positions(rads2_dct[pa] * h)
            # Calculate statistics line for difference between position of pa and every later planet.
            for pb in planets_to_use[ia + 1:]:
                # Effective period for the angle between planets pa and pb.
                eop = effective_orbital_period(pa, pb)
                if too_slow_or_too_seasonal(eop, h * args.max_orbit):
                    # The angle moves slowly enough that generational effects may be the cause of any
                    # non-randomness, or its period is close enough to 1 year that seasonality may be the cause.
                    continue
                if pb not in rads1_dct:
                    # Fill rads1_dct and rads2_dct for planet pb.
                    calculate_planet_radians(dates1, pb, rads1_dct)
                    calculate_planet_radians(dates2, pb, rads2_dct)
                # Differences between positions of planets pa and pb, over all entries in rads1_dct (first dataset).
                rads1_diff = rads1_dct[pa] - rads1_dct[pb]
                # Ditto for rads2_dct (second dataset).
                rads2_diff = rads2_dct[pa] - rads2_dct[pb]
                # p_values = scipy.stats.norm.sf(abs(z_scores)) #one-sided
                # p_values = scipy.stats.norm.sf(abs(z_scores))*2 #twosided
                roc = calculate_roc(0.0, h, rads1_diff, rads2_diff)
                # z value is number of standard deviations by which roc deviates from 0.5, given an estimated
                # standard deviation null_sd.
                z = (roc - 0.5) / null_sd
                # Fields in this line: z value, harmonic number, two-letter abbreviations for the two planets,
                # roc value, number of entries and name of first dataset, number of entries and name of second
                # dataset, and effective orbital period of the difference angle when multiplied by the harmonic.
                # If this last value is big (enough years for generational effects to be relevant) or close to 1.0
                # (so seasonal effects may be relevant) then treat the z value with scepticism.
                print(f"{z:9.5f} {h:2d} {ABBREV[pa]:2s}   {ABBREV[pb]:2s} {roc:8.6f} " 
                      f"{len(dates1):6d} {f1:14s} {len(dates2):6d} {f2:14s} {eop / h:7.3f}")
                sys.stdout.flush()
            # Calculate statistics line(s) for position of pa itself.
            eop = ORBITAL_PERIOD[SUN if pa in [MERCURY, VENUS] else pa]
            if too_slow_or_too_seasonal(eop, h * args.max_orbit):
                continue
            # Vector (in (x,y) coordinate space) for difference between means of the two sets. This will be somewhere
            # inside the unit circle, usually very near the centre.
            vector = means2_dct[pa] - means1_dct[pa]
            # Convert (x,y) to polar coordinates.
            vcomp = complex(vector[0], vector[1])
            magnitude, angle = cmath.polar(vcomp)  # type: ignore
            # Estimate diff_sd, the standard deviation for vector, under the null hypothesis that the two datasets are
            # from the same distribution. The expected value is (0,0).
            data = np.concatenate([rads1_dct[pa], rads2_dct[pa]])
            diff_sd = np.sqrt((np.var(np.cos(data)) + np.var(np.sin(data))) * (1 / len(dates1) + 1 / len(dates2)))
            diff_z = magnitude / diff_sd
            # Statistics line: z value, harmonic, two-letter abbreviation for first planet, angle (in degrees) and
            # magnitude of centroid, size and name of first dataset, size and name of second dataset, effective
            # orbital period of the harmonic of the planet.
            print(f"{diff_z:9.5f} {h:2d} {ABBREV[pa]:2s} {int(0.5+angle*180/math.pi):4d} {magnitude:8.6f} " 
                  f"{len(dates1):6d} {f1:14s} {len(dates2):6d} {f2:14s} {eop / h:7.3f}")
            # If degree_step is positive, we calculate the roc value for the cosines of all the positions of pa over the
            # two datasets, taking 0, degree_step, 2 * degree_step ... 180 as the origin for the cosines. This is
            # a (probably inferior) alternative to mean differences. Evaluating z is extra tricky in this case!
            if args.degree_step > 0:
                for d in range(0, 180, args.degree_step):  # don't need full circle because cos is antisymmetric
                    dr = d * math.pi / 180
                    roc = calculate_roc(dr, h, rads1_dct[pa], rads2_dct[pa])
                    z = (roc - 0.5) / null_sd
                    # Statistics line: z value, harmonic, two-letter abbreviation for the planet, degree value,
                    # ROC value, size and name of first dataset, size and name of second dataset, effective
                    # orbital period of the harmonic of the planet.
                    print(f"{z:9.5f} {h:2d} {ABBREV[pa]:2s} {d:4d} {roc:8.6f}" 
                          f"{len(dates1):6d} {f1:14s} {len(dates2):6d} {f2:14s} {eop / h:7.3f}")
                    sys.stdout.flush()


def too_slow_or_too_seasonal(eop: float, max_orbit: float) -> bool:
    """
    :param eop: effective orbital period of some quantity
    :param max_orbit: maximum orbital period we will tolerate
    :return: true if eop is greater than max_orbit, or is closer to 1 than 1/max_orbit.
    Thus higher values of max_orbit mean more tolerance.
    Example: max_orbit = 10 means eop = 9 is OK but 11 is not; and
    eop = 1.11 is OK but eop = 1.09 is not.
    """
    return eop > max_orbit or abs(eop - 1) < 1 / max_orbit


def calculate_planet_radians(dates1, p, rads1_dct):
    rads1_dct[p] = radian_positions(dates1, p)


def null_hypothesis_roc_standard_deviation(n1: int, n2: int) -> float:
    len_tuple = tuple(sorted([n1, n2]))
    if len_tuple not in ROC_SD:
        ROC_SD[len_tuple] = roc_sd.roc_sd(*len_tuple, extra_samples=1000000)[0]
        print(f"# null_sd{len_tuple} = {ROC_SD[len_tuple]:8.6f}")
    null_sd = ROC_SD[len_tuple]
    sys.stdout.flush()
    return null_sd


def shuffle_dates_between_categories(dates1, dates2):
    dates12 = np.concatenate([dates1, dates2])
    np.shuffle(dates12)
    dates1, dates2 = dates12[:len(dates1)], dates12[len(dates1):]
    return dates1, dates2


def calculate_roc(dr, h, rads1, rads2):
    """
    Returns the ROC (Receiver Operating Characteristic) or AUC (Area Under Curve) value for the given datasets.
    :param dr: value to add to each angle before taking cosine.
    :param h: harmonic: value to multiply each angle by (before adding dr)
    :param rads1: values in set 1, in radians.
    :param rads2: values in set 2, in radians.
    :return: ROC value for set 2 vs set 1 (will be > 0.5 if set 2 mostly has higher values).
    """
    # Cosine of each angle in rads1 multiplied by h, paired with constant 1.
    pairs1 = cosine_pairs(dr, h, rads1, 1)
    # Cosine of each angle in rads2 multiplied by h, paired with constant 2.
    pairs2 = cosine_pairs(dr, h, rads2, 2)
    # Pairs in order, from lowest (most negative) cosine to highest.
    sorted_pairs = sorted(pairs1 + pairs2)
    # Number of pairs we've seen so far with 1 in second position. At the end of sorted_pairs, this will be len(rads1).
    n1 = 0
    # Sum of values of n1 when we encounter a pair with 2 in second position. At the end of sorted_pairs, this will
    # be the number of pairs with one member from each of rads1 and rads2, such that the cosine value of the rads1
    # item is less than that of the rads2 item.
    n12 = 0
    for _, c in sorted_pairs:
        if c == 1:
            n1 += 1
        else:
            n12 += n1
    # n1 * len(rads2) is the total number of pairs we can make from rads1 and rads2. So n12 over that value is the
    # proportion of pairs with the rads1 value less than the rads2 value. If the two sets are from the same distribution
    # the expected value of this is 0.5.
    return n12 / (n1 * len(rads2))


def cosine_pairs(dr, h, rads, idx):
    return [(math.cos(h * rad + dr), idx) for rad in rads]


def radian_positions(dates1, p):
    return np.array([swe.calc(date, p)[0][0] * math.pi / 180 for date in dates1])


def size_ratio_ok(sz1: int, sz2: int, max_size_ratio: float) -> bool:
    return sz1 * max_size_ratio >= sz2 and sz2 * max_size_ratio >= sz1


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
    log_arguments(args)
    return args


def log_arguments(args):
    for a in sorted(dir(args)):
        if a != "files" and not a.startswith("_"):
            print(f"# --{a} {getattr(args, a)}")
    sys.stdout.flush()


def main():
    args = parse_arguments()
    paths, dates = get_paths_and_dates(args.files)
    for delta1 in range(1, len(paths)):
        delta = 3 - delta1 if args.pairs_first and delta1 < 3 else delta1
        for i, path1 in enumerate(paths):
            if i + delta >= len(paths):
                break
            path2 = paths[i + delta]
            if not size_ratio_ok(len(dates[path1]), len(dates[path2]), args.max_size_ratio):
                continue
            compare_files(args, path1.name, dates[path1], path2.name, dates[path2])


if __name__ == "__main__":
    main()
