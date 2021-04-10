#!/usr/bin/python

import csv
import argparse
import re
from pathlib import Path
from typing import Optional, List

DOB_CODE = "P569"
OCCUPATION_CODE = "P106"


def join_with_dob(args: argparse.Namespace):
    data_dir = Path(args.data_dir)
    for f in sorted(data_dir.glob("x??.csv")):
        join_with_dob_file(data_dir, f, args)
    if args.min_members > 1:
        rel_dir = data_dir / args.rel_code
        for f in rel_dir.glob("Q*.csv"):
            with f.open() as fh:
                for i, row in enumerate(csv.reader(fh), 1):
                    if i == args.min_members:
                        continue
            f.unlink()


class UniqueDict(dict):
    def __init__(self):
        super().__init__()
        self._set = set()

    def __setitem__(self, key, value):
        if key in self:
            del self._dict[key]
        elif key not in self._set:
            self._set.add(key)
            super().__setitem__(key, value)


def join_with_dob_file(data_dir: Path, f: Path, args: argparse.Namespace) -> None:
    dob_dct = UniqueDict()
    rel_dct = UniqueDict()
    rel_dir = data_dir / args.rel_code
    rel_dir.mkdir(exist_ok=True)
    with open(f) as fh:
        for row in csv.reader(fh):
            subj, rel, val = row[:3]
            if rel == DOB_CODE:
                val = shorten_date(val, args.min_year, args.max_year)
                if val is not None:
                    dob_dct[subj] = val
            elif rel == args.rel_code:
                rel_dct[subj] = val
    for subj in sorted(dob_dct):
        val = rel_dct.get(subj, None)
        if val is None:
            continue
        with (rel_dir / f"{val}.csv").open("a") as gh:
            gh.write(f"{subj},{dob_dct[subj]}\n")
    print(f"Done {f}")


def shorten_date(val: str, min_year: Optional[int], max_year: Optional[int]) -> str:
    if val.startswith("+"):
        val = val[1:]
    if val.endswith("T00:00:00Z"):
        val = val[: -len("T00:00:00Z")]
    m = re.match("([0-9][0-9][0-9][0-9])-[0-1][0-9]-[0-3][0-9]")
    if m:
        year = m.group(1)
        if (min_year is None or year >= min_year) and (
            max_year is None or year <= max_year
        ):
            return val
    return None


def main(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--relation", type=str, default=OCCUPATION_CODE)
    parser.add_argument("--min_year", type=int, default=None)
    parser.add_argument("--max_year", type=int, default=None)
    parser.add_argument("--min_members", type=int, default=100)
    args = parser.parse_args(args)
    join_with_dob(args)


if __name__ == "__main__":
    main()
