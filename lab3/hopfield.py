import numpy as np


class HopfieldNetwork:

    def __init__(self):
        self.weights = None

    def train(self, patterns):
        n = patterns[0].size
        self.weights = np.zeros((n, n))

        for p in patterns:
            p = p.flatten()
            p = np.where(p == 0, -1, 1)

            self.weights += np.outer(p, p)

        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def predict(self, pattern, steps=5):
        state = pattern.flatten()
        state = np.where(state == 0, -1, 1)

        for _ in range(steps):
            state = np.sign(self.weights @ state)

        return np.where(state == -1, 0, 1).reshape(pattern.shape)