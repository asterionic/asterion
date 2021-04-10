#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import os
import sys
from statistics import pstdev


MIN_TRIALS = 100
EXTRA_SAMPLES = 100000


def roc_sd(n1: int, n2: int, extra_samples: int = EXTRA_SAMPLES):
    n_trials = MIN_TRIALS + int(extra_samples / (n1 + n2))
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
