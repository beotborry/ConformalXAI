import torch
import numpy as np
import random

from numpy import ndarray

def set_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def normalize_scale(attr: ndarray, scale_factor: float):
    assert scale_factor != 0
    assert scale_factor >= 1e-5

    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)

def cumulative_sum_threshold(values: ndarray, percentile):
    assert percentile >= 0 and percentile <= 100

    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]

def entropy(probs):
    log_probs = np.log(probs)
    return (-probs * log_probs).sum()
