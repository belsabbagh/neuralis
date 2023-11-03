from typing import Any
import numpy as np


def _sop(x, w):
    return np.dot(x, w.T[1:].T) - w.T[0]


class Neuron:
    weights = None

    def __init__(self, activation, weights) -> None:
        weights = np.array(weights)
        self.activation, self.derv = activation
        self.weights = weights

    def calc(self, x):
        """$$ y = \\sigma(\\sum_{i=1}^{n} w_i x_i) $$
        where:
        - $y$ is the output of the neuron
        - $\\sigma$ is the activation function
        - $n$ is the number of inputs
        - $w_i$ is the $i$th weight
        - $x_i$ is the $i$th input
        """
        return self.activation(_sop(x, self.weights))

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.calc(x)

    def back_calc(self, y, delta, weights):
        return y * (1 - y) * np.dot(weights, delta)
