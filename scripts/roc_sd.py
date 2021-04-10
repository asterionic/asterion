#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import os
import sys
from statistics import pstdev


MIN_TRIALS = 100
EXTRA_SAMPLES = 100000


def roc_sd(n1: int, n2: int):
    n_trials = MIN_TRIALS + int(EXTRA_SAMPLES / (n1 + n2))
    rocs = [sample_roc(n1, n2) for _ in range(n_trials)]
    sd = pstdev(rocs)
    return sd, n1, n2, n_trials


def sample_roc(n1, n2):
    k1, k2 = n1, n2
    sum = 0
    data = np.concatenate([np.ones(n1, dtype=np.int), np.full(n2, 2, dtype=np.int)])
    np.random.shuffle(data)
    for i in data:
        if i == 1:
            sum += k2
        else:
            k2 -= 1
    roc = sum / (n1 * n2)
    return roc


def main():
    assert len(sys.argv) == 2
    in_file = Path(sys.argv[1])
    assert in_file.exists()
    sigma = -1
    sn1, sn2 = -1, -1
    with open(sys.argv[1]) as fh:
        for line in fh:
            toks = line.strip().split()
            n1, n2 = int(toks[4]), int(toks[6])
            if (n1, n2) != (sn1, sn2):
                sn1, sn2 = n1, n2
                sigma = roc_sd(sn1, sn2)[0]
                sys.stdout.flush()
            z = abs(float(toks[3]) - 0.5) / sigma
            print("%8.6f %s" % (z, line.strip()))


if __name__ == '__main__':
    main()