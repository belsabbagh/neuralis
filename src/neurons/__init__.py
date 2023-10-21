from typing import Any
import numpy as np


def _sop(x, w):
    return np.dot(x, w.T[1:].T) - w.T[0]


class Neuron:
    weights = None

    def __init__(self, activation, weights) -> None:
        weights = np.array(weights)
        self.activation = activation
        self.weights = weights

    def calc(self, x):
        return self.activation(_sop(x, self.weights))

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.calc(x)

    def back_calc(self, y, delta, weights):
        return y * (1 - y) * np.dot(weights, delta)
