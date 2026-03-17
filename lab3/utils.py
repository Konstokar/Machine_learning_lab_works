import numpy as np


def add_noise(pattern, noise_level=0.3):
    noisy = pattern.copy()
    mask = np.random.rand(*pattern.shape) < noise_level
    noisy[mask] = 1 - noisy[mask]
    return noisy


def print_pattern(p):
    for row in p:
        print("".join(["0" if x else "." for x in row]))
    print()