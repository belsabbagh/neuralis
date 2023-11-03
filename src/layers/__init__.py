import numpy as np
from src.neurons import Neuron


class InputLayer:
    units = None

    def __init__(self, units) -> None:
        self.units = units

    def calc(self, x):
        if len(x) != self.units:
            raise ValueError(f"Input Mismatch. Expected {self.units}, got {len(x)}")
        return x


class Layer:
    """Each layer has $n$ neurons, a weight matrix $W$ of size $n \\times m + 1$ where $m$ is the number of inputs, and an activation function $\\sigma$."""

    neurons = None
    weights = None
    bias = None
    input_shape = None

    def __init__(
        self, units, activation, weights: list[list[float]] = None, input_shape=None
    ) -> None:
        if weights is not None and len(weights) != units:
            raise ValueError(
                f"Units ({units}) and weights ({len(weights)}) count mismatch!"
            )
        self.neurons = [Neuron(activation, w) for w in weights]

    def calc(self, x, verbose=1):
        if verbose:
            print(f"Input: {x}\nWeights: {self.get_weights()}")
        y = [i(x) for i in self.neurons]
        if verbose:
            print(f"Output: {y}")
            print(f"--------------------------------------")
        return y

    def get_weights(self):
        return np.array([i.weights for i in self.neurons])

    def backpropagate(self, y, deltas, weights):
        res = [i.back_calc(yi, deltas, w) for i, yi, w in zip(self.neurons, y, weights)]
        return np.array(res)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        activation_name = self.neurons[0].activation.__name__
        return f"Layer({len(self.neurons)}, {activation_name})"

    def __dict__(self):
        return {"activation": self.activation, "units": len(self.neurons)}
