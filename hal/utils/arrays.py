import numpy as np


def normalize(arr):
    return (arr - np.mean(arr, 0)) / np.std(arr, 0)


def multi_class_to_matrix(multi_class):
    unique_outputs = np.unique(multi_class)
    multi_outputs = [multi_class == output_value for output_value in unique_outputs]
    return np.column_stack(multi_outputs).astype(int)