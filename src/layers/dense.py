import numpy as np
from src.layers.base import Layer
from typing import Optional


def _sop(x, w):
    return np.dot(x, w.T[1:].T) - w.T[0]


def update(w, y, d, alpha=0.1):
    """$$ w_{i+1} = w_i + \\alpha * d * y $$
    where:
    - $w_{i+1}$ is the new weight
    - $w_i$ is the old weight
    - $\\alpha$ is the learning rate
    - $d$ is the delta
    - $y$ is the output of the neuron
    """
    return np.array([w[0] + alpha * d, *(w[1:] - alpha * d * y)])


class Dense(Layer):
    """Each layer has $n$ neurons, a weight matrix $W$ of size $n \\times m + 1$ where $m$ is the number of inputs, and an activation function $\\sigma$."""

    weights = None
    bias = None
    input_shape = None

    def __init__(
        self,
        units,
        activation,
        weights: Optional[list[list[float]]],
        input_shape: Optional[int] = None,
    ) -> None:
        if weights is not None and len(weights) != units:
            raise ValueError(
                f"Units ({units}) and weights ({len(weights)}) count mismatch!"
            )
        if input_shape is None and weights is not None:
            input_shape = len(weights[0]) - 1
        if weights is None:
            weights = np.random.rand(units, input_shape + 1)
        self.activation, self.derv = activation
        self.units = units
        self.weights = np.array(weights)

    def forward(self, x, verbose=1):
        if verbose:
            print(f"Input: {x}\nWeights: {self.get_weights()}")
        y = [self.activation(_sop(x, w)) for w in self.weights]
        if verbose:
            print(f"Output: {y}")
            print(f"--------------------------------------")
        return y

    def get_weights(self):
        return self.weights

    def update(self, y, d, lr=0.01):
        self.weights = update(self.weights, y, d, alpha=lr)

    def backward(self, y, deltas, weights):
        res = [yi * (1 - yi) * np.dot(w, deltas) for yi, w in zip(y, weights)]
        return np.array(res)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        activation_name = self.activation.__name__
        return f"Layer({self.units}, {activation_name})"

    def __dict__(self):
        return {"activation": self.activation, "units": self.units}
