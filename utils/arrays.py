import numpy as np


def normalize(arr):
    return (arr - np.mean(arr, 0)) / np.std(arr, 0)
