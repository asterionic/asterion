#!/usr/bin/env python
import argparse
import cmath
import collections
import csv
import math
import sys
from pathlib import Path
from random import random, shuffle
from typing import Tuple, List, Dict, Optional

import numpy as np
import swisseph as swe

import roc_sd
from roc_estimates import ROC_SD

DAYS_IN_YEAR = 365.2422

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

PLANETS: List[int] = [t[0] for t in PLANET_DATA]
PLANET_NAME: Dict[int, str] = dict((t[0], t[1]) for t in PLANET_DATA)
ORBITAL_PERIOD: Dict[int, float] = dict((t[0], t[3]) for t in PLANET_DATA)
ABBREV: Dict[int, str] = dict((t[0], t[2]) for t in PLANET_DATA)


def get_paths_and_dates(args: argparse.Namespace) -> Tuple[List[Path], Dict[Path, np.ndarray]]:
    """
    :param args: see create_parser
    :return: paths: a list of Path objects, one for each file, ordered by decreasing number of successfully
    extracted dates in the file; and dates: a dictionary from paths to arrays of day values.
    """
    paths = [Path(f) for f in args.files]
    dates = dict((p, get_dates(p, args)) for p in paths)
    paths.sort(key=lambda p: len(dates[p]), reverse=True)
    return paths, dates


def is_leap_year(y):
    if y % 100 == 0:
        y = int(y / 100)
    return y % 4 == 0


def get_dates(path: Path, args: argparse.Namespace):
    """
    :param path: a csv file, assumed to contain a date in the last column of each row.
    The date should be in the form yyyy-mm-dd. We assume there is no time information,
    so we set the time to a random number of hours between 0 and 24.
    :return: an array of "Julian" (actually Gregorian) day values, one for each date in the file.
    """
    dates = []
    keep_first_days = args.keep_first_days
    min_year = args.min_year
    max_year = args.max_year
    flatten = args.flatten_by_day
    default_to_noon = args.default_to_noon
    date_years = collections.defaultdict(list)
    max_day = [None, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # reject all February 29
    with path.open() as inp:
        for row in csv.reader(inp):
            try:
                day = row[-1]
                y, m, d = [int(n) for n in day.split("-")]
                if y < min_year or y > max_year:
                    continue  # year outside the range we want
                if m <= 0 or d <= 0 or m > 12 or d > max_day[m]:
                    continue  # invalid month or day value
                if (m, d) == (2, 29) and (flatten or not is_leap_year(y)):
                    continue  # don't want Feb 29 if flattening, or if it's not a leap year
                if d == 1 and keep_first_days < 2 and (m == 1 or keep_first_days == 0):
                    continue  # first day of month; discard if January (if keep_first_days == 1) or always (if 2)
                date_years[(m, d)].append(y)
            except IndexError:  # not all rows may be of the expected format
                pass
    if flatten:
        counts = [len(lst) for lst in date_years.values()]
        min_count = min(counts)
    else:
        min_count = 0
    for (m, d), y_list in date_years.items():
        if flatten:
            y_list = y_list[:min_count]
        for y in y_list:
            hour = 12.0 if default_to_noon else 24.0 * random()
            dates.append(swe.julday(y, m, d, hour))
    if flatten:
        print(f"# Equalizing by day of year reduces count from {sum(counts)} to " 
              f"{len(date_years)}*{min_count}={len(dates)} for {path}")
    return np.array(dates)


class Comparison:

    def __init__(self, args: argparse.Namespace, dates1: np.ndarray, dates2: np.ndarray, identifier: str):
        self.args = args
        self.dates1 = dates1
        self.dates2 = dates2
        self.identifier = identifier
        # Dictionary from planet numbers to an array of planet positions in radians, one for each date in dates1.
        self.rads1_dct = {}
        # Ditto for dates2.
        self.rads2_dct = {}
        # List of planets we're going to consider.
        self.planets_to_use = [p for p in PLANETS if p <= self.args.max_planet]

    def run(self):
        # If time buckets (by numbers of years and/or time of year) were specified, we prune the data accordingly.
        self.match_by_time_buckets()
        # We check sizes again, in case pruning has reduced them below the minimum.
        if min(len(self.dates1), len(self.dates2)) < self.args.min_dataset_size:
            return
        if self.args.shuffle:
            self.shuffle_dates_between_categories()
        null_sd = self.null_hypothesis_roc_standard_deviation(len(self.dates1), len(self.dates2))
        self.calculate_linear_roc(null_sd)
        self.look_up_planet_positions()
        for h in range(self.args.min_harmonic, self.args.max_harmonic + 1):
            for pa in self.planets_to_use:
                # Calculate statistics line for difference between position of pa and every later planet.
                for pb, eop_ab in self.planets_to_pair_with(pa, h):
                    self.compare_planet_pair(h, pa, pb, eop_ab, null_sd)
                # Calculate statistics line(s) for position of pa itself.
                eop_a = ORBITAL_PERIOD[SUN if pa in [MERCURY, VENUS] else pa]
                self.calculate_and_print_difference_vector(h, pa, eop_a)
                self.calculate_rocs_for_angles(h, pa, eop_a, null_sd)

    def calculate_linear_roc(self, null_sd: float):
        diff = np.mean(self.dates2) - np.mean(self.dates1)
        std1 = np.std(self.dates1)
        std2 = np.std(self.dates2)
        print(f"# Difference in means, set 2 - set 1: {diff / DAYS_IN_YEAR:5.3f} years")
        print(f"# Standard deviations: {std1 / DAYS_IN_YEAR:5.3f} years (set 1) "
              f"and {std2 / DAYS_IN_YEAR:5.3f} years (set 2)")
        roc = Comparison.calculate_roc(self.dates1, self.dates2)
        z = (roc - 0.5) / null_sd
        # Statistics line: z value, harmonic, two-letter abbreviation for the planet, degree value,
        # ROC value, size and name of first dataset, size and name of second dataset, effective
        # orbital period of the harmonic of the planet.
        self.print_line("L", z, 1, -1, "Ln", roc, 0.0)
        sys.stdout.flush()

    def year_buckets(self, dates):
        zero_date = swe.julday(1900, 1, 1)
        buckets = collections.defaultdict(list)
        for date in dates:
            diff = (date - zero_date) / (self.args.match_by_years * DAYS_IN_YEAR)
            if diff < 0:
                bucket = -(int(-diff) + 1)
            else:
                bucket = int(diff)
            buckets[bucket].append(date)
        return buckets

    def day_buckets(self, dates):
        zero_date = swe.julday(1900, 1, 1)
        buckets = collections.defaultdict(list)
        for date in dates:
            year_fraction = (date - zero_date) / DAYS_IN_YEAR + 1000
            year_fraction -= int(year_fraction)
            bucket = int(year_fraction * DAYS_IN_YEAR / self.args.match_by_days)
            buckets[bucket].append(date)
        return buckets

    def match_by_time_buckets(self) -> None:
        if self.args.match_by_years > 0:
            len1, len2 = len(self.dates1), len(self.dates2)
            buckets1 = self.year_buckets(self.dates1)
            buckets2 = self.year_buckets(self.dates2)
            self.dates1, self.dates2 = self.apply_buckets(buckets1, buckets2, len1, len2)
            print(f"# With year bucket size = {self.args.match_by_years:2d}, "
                  f"reduced counts from {len1:6d} and {len2:6d} to {len(self.dates1):6d} and {len(self.dates2):6d}")
        if self.args.match_by_days > 0:
            len1, len2 = len(self.dates1), len(self.dates2)
            buckets1 = self.day_buckets(self.dates1)
            buckets2 = self.day_buckets(self.dates2)
            self.dates1, self.dates2 = self.apply_buckets(buckets1, buckets2, len1, len2)
            print(f"# With  day bucket size = {self.args.match_by_days:2d}, " 
                  f"reduced counts from {len1:6d} and {len2:6d} to {len(self.dates1):6d} and {len(self.dates2):6d}")

    @staticmethod
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

    @staticmethod
    def calculate_mean_positions(rads: np.ndarray) -> np.ndarray:
        return np.array([float(np.mean(np.cos(rads))), float(np.mean(np.sin(rads)))])

    @staticmethod
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

    def print_line(self, line_type, z, h, pa, b_str, value, eop):
        ast = "*" if self.too_slow_or_too_seasonal(eop, h) else " "
        print(f"{line_type:s}{ast} {abs(z):9.5f} {h:2d} {ABBREV.get(pa, 'Xx'):2s} {b_str:4s} {value:8.6f}"
              f"{eop / h:10.3f} {self.identifier}")
        sys.stdout.flush()

    def planets_to_pair_with(self, p1: int, h: int) -> List[Tuple[int, float]]:
        result = []
        for p2 in self.planets_to_use:
            if p2 > p1:
                eop = self.effective_orbital_period(p1, p2)
                result.append((p2, eop))
        return result

    def calculate_and_print_difference_vector(self, h, pa, eop_a):
        magnitude, angle = self.calculate_mean_difference(h, pa)
        # Estimate diff_sd, the standard deviation for vector, under the null hypothesis that the two datasets
        # are from the same distribution. The expected value is (0,0).
        data = np.concatenate([self.rads1_dct[pa], self.rads2_dct[pa]])
        diff_var = (np.var(np.cos(data)) + np.var(np.sin(data))) * (1 / len(self.dates1) + 1 / len(self.dates2))
        diff_z = magnitude / np.sqrt(diff_var)
        # Statistics line: z value, harmonic, two-letter abbreviation for first planet, angle (in degrees) and
        # magnitude of centroid, size and name of first dataset, size and name of second dataset, effective
        # orbital period of the harmonic of the planet.
        self.print_line("V", diff_z, h, pa, str(int(0.5 + angle * 180 / math.pi)), magnitude, eop_a)

    def calculate_rocs_for_angles(self, h, pa, eop_a, null_sd):
        # If degree_step is positive, we calculate the roc value for the cosines of all the positions of pa
        # over the two datasets, taking 0, degree_step, 2 * degree_step ... 180 as the origin for the cosines.
        # This is a (probably inferior) alternative to mean differences. Evaluating z is extra tricky in this
        # case!
        if self.args.degree_step <= 0:
            return
        # don't need full circle because cos is antisymmetric:
        for d in range(0, 180, self.args.degree_step):
            dr = d * math.pi / 180
            roc = self.calculate_roc_from_cosines(dr, h, self.rads1_dct[pa], self.rads2_dct[pa])
            z = (roc - 0.5) / null_sd
            # Statistics line: z value, harmonic, two-letter abbreviation for the planet, degree value,
            # ROC value, size and name of first dataset, size and name of second dataset, effective
            # orbital period of the harmonic of the planet.
            self.print_line("D", z, h, pa, str(d), roc, eop_a)
            sys.stdout.flush()

    def calculate_mean_difference(self, h, p):
        # Vector (in (x,y) coordinate space) for difference between the mean (centroid) of the positions of
        # that planet on dates in dates2, minus the same on dates1, treating the zodiac as a unit circle.
        mean1 = self.calculate_mean_positions(self.rads1_dct[p] * h)
        mean2 = self.calculate_mean_positions(self.rads2_dct[p] * h)
        vector: np.ndarray = mean2 - mean1
        # Convert to polar coordinates.
        magnitude, angle = cmath.polar(complex(vector[0], vector[1]))  # type: ignore
        return magnitude, angle

    def compare_planet_pair(self, h, pa, pb, eop_ab, null_sd):
        # Differences between positions of planets pa and pb, over all entries in rads1_dct (first dataset).
        rads1_diff = self.rads1_dct[pa] - self.rads1_dct[pb]
        # Ditto for rads2_dct (second dataset).
        rads2_diff = self.rads2_dct[pa] - self.rads2_dct[pb]
        # p_values = scipy.stats.norm.sf(abs(z_scores)) #one-sided
        # p_values = scipy.stats.norm.sf(abs(z_scores))*2 #twosided
        roc = self.calculate_roc_from_cosines(0.0, h, rads1_diff, rads2_diff)
        # z value is number of standard deviations by which roc deviates from 0.5, given an estimated
        # standard deviation null_sd.
        z = (roc - 0.5) / null_sd
        # Fields in this line: z value, harmonic number, two-letter abbreviations for the two planets,
        # roc value, number of entries and name of first dataset, number of entries and name of second
        # dataset, and effective orbital period of the difference angle when multiplied by the harmonic.
        # If this last value is big (enough years for generational effects to be relevant) or close to 1.0
        # (so seasonal effects may be relevant) then treat the z value with scepticism.
        self.print_line("A", z, h, pa, ABBREV[pb], roc, eop_ab)
        sys.stdout.flush()

    def look_up_planet_positions(self):
        for p in self.planets_to_use:
            # Fill rads1_dct and rads2_dct for planet p.
            self.calculate_planet_radians(self.dates1, p, self.rads1_dct)
            self.calculate_planet_radians(self.dates2, p, self.rads2_dct)

    def too_slow_or_too_seasonal(self, eop: float, h: int) -> bool:
        """
        :param eop: effective orbital period of some quantity (without considering harmonic)
        :param h: harmonic
        :return: True in two cases:
          (1) too slow: EOP of harmonic (eop_h) is basic eop divided by harmonic number.
              This has to be at most max_orbit.
          (2) too seasonal: eop_h cannot be too close to 1. If its integer part is closer to 1 than 1 / max_orbit,
              then it will take more than max_orbit years for the position at any fixed time of year to cycle
              around the zodiac.
        """
        max_orbit = self.args.max_orbit
        eop_h = eop / h
        return eop_h > max_orbit or abs(eop_h - 1) < 1 / max_orbit

    @staticmethod
    def calculate_planet_radians(dates1, p, rads1_dct):
        rads1_dct[p] = Comparison.radian_positions(dates1, p)

    @staticmethod
    def null_hypothesis_roc_standard_deviation(n1: int, n2: int) -> float:
        len_tuple = tuple(sorted([n1, n2]))
        if len_tuple not in ROC_SD:
            ROC_SD[len_tuple] = roc_sd.roc_sd(*len_tuple, extra_samples=1000000)[0]
            print(f"# null_sd{len_tuple} = {ROC_SD[len_tuple]:8.6f}")
        null_sd = ROC_SD[len_tuple]
        sys.stdout.flush()
        return null_sd

    def shuffle_dates_between_categories(self) -> None:
        dates12 = np.concatenate([self.dates1, self.dates2])
        shuffle(dates12)
        self.dates1, self.dates2 = dates12[:len(self.dates1)], dates12[len(self.dates1):]

    @staticmethod
    def calculate_roc_from_cosines(dr, h, rads1, rads2) -> float:
        """
        Returns the ROC (Receiver Operating Characteristic) or AUC (Area Under Curve) value for the given datasets.
        :param dr: value to add to each angle before taking cosine.
        :param h: harmonic: value to multiply each angle by (before adding dr)
        :param rads1: values in set 1, in radians.
        :param rads2: values in set 2, in radians.
        :return: ROC value for set 2 vs set 1 (will be > 0.5 if set 2 mostly has higher values).
        """
        # Cosine of each angle in rads1 multiplied by h
        values1 = Comparison.cosine_values(dr, h, rads1)
        # Cosine of each angle in rads2 multiplied by h
        values2 = Comparison.cosine_values(dr, h, rads2)
        return Comparison.calculate_roc(values1, values2)

    @staticmethod
    def calculate_roc(values1, values2):
        # Pairs in order, from lowest (most negative) value to highest.
        sorted_pairs = sorted([(v, 1) for v in values1] + [(v, 2) for v in values2])
        # Number of pairs we've seen so far with 1 in second position. At the end of sorted_pairs, this will be
        # len(values1).
        n1 = 0
        # Sum of values of n1 when we encounter a pair with 2 in second position. At the end of sorted_pairs, this will
        # be the number of pairs with one member from each of values1 and values2, such that the cosine value of the
        # values1 item is less than that of the values2 item.
        n12 = 0
        for _, c in sorted_pairs:
            if c == 1:
                n1 += 1
            else:
                n12 += n1
        # n1 * len(values2) is the total number of pairs we can make from values1 and values2. So n12 over that value
        # is the proportion of pairs with the values1 value less than the values2 value. If the two sets are from the
        # same distribution the expected value of this is 0.5.
        return n12 / (n1 * len(values2))

    @staticmethod
    def cosine_values(dr, h, rads):
        return [math.cos(h * rad + dr) for rad in rads]

    @staticmethod
    def radian_positions(dates1, p):
        return np.array([swe.calc(date, p)[0][0] * math.pi / 180 for date in dates1])


def sizes_are_ok(sz1: int, sz2: int, args: argparse.Namespace) -> bool:
    if min(sz1, sz2) < args.min_dataset_size:
        return False
    return sz1 * args.max_size_ratio >= sz2 and sz2 * args.max_size_ratio >= sz1


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(args)
    parser.add_argument("--max_size_ratio", type=float, default=5.0)
    parser.add_argument("--max_planet", type=int, default=20)
    parser.add_argument("--max_harmonic", type=int, default=11)
    parser.add_argument("--degree_step", type=int, default=0)
    parser.add_argument("--max_orbit", type=float, default=5)
    parser.add_argument("--min_harmonic", type=int, default=1)
    parser.add_argument("--match_by_years", type=int, default=0)
    parser.add_argument("--match_by_days", type=int, default=0)
    parser.add_argument("--min_dataset_size", type=int, default=100)
    parser.add_argument("--pairs_first", action="store_true", default=False)
    parser.add_argument("--shuffle", action="store_true", default=False)
    # 0: discard all first-of-month dates. 1: discard January 1st only. 2: discard none.
    parser.add_argument("--keep_first_days", type=int, default=0)
    # By default, we only keep dates of 1800 or later, as earlier dates may be inaccurate or use a different calendar.
    parser.add_argument("--min_year", type=int, default=1800)
    parser.add_argument("--max_year", type=int, default=2200)
    # Whether to set the time for each day to noon, as opposed to choosing a uniform random time.
    parser.add_argument("--default_to_noon", action="store_true", default=False)
    # Whether to discard instances so we have exactly the same number of instances for each day of the year
    parser.add_argument("--flatten_by_day", action="store_true", default=False)
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
    paths, dates = get_paths_and_dates(args)
    for delta1 in range(1, len(paths)):
        delta = 3 - delta1 if args.pairs_first and delta1 < 3 else delta1
        for i, path1 in enumerate(paths):
            if i + delta >= len(paths):
                break
            path2 = paths[i + delta]
            if sizes_are_ok(len(dates[path1]), len(dates[path2]), args):
                identifier = f"{i},{i + delta}"
                print(f"# Comparison {identifier} = {path1}({len(dates[path1])}) {path2}({len(dates[path2])})")
                comp = Comparison(args, dates[path1], dates[path2], identifier)
                comp.run()


if __name__ == "__main__":
    main()
