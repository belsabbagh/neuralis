import numpy as np
from src.neurons import Neuron


class InputLayer:
    units = None

    def __init__(self, units) -> None:
        self.units = units

    def calc(self, x):
        if len(x) != self.units:
            raise ValueError(
                f"Input Mismatch. Expected {self.units}, got {len(x)}")
        return x


class Layer:
    neurons = None
    activation = None
    weights = None
    bias = None

    def __init__(
        self,
        units,
        activation,
        weights: list[list[float]] = None,
        bias: list[float] = None,
    ) -> None:
        if weights is not None and len(weights) != units:
            raise ValueError(
                f"Units ({units}) and weights ({len(weights)}) count mismatch!"
            )
        if bias is not None and len(bias) != units:
            raise ValueError(
                f"Units ({units}) and weights ({len(bias)}) count mismatch!"
            )
        self.bias = bias
        self.activation = activation
        self.neurons = [Neuron(activation, w, b)
                        for w, b in zip(weights, bias)]

    def calc(self, x, verbose=1):
        bias = [0] * len(self.neurons) if self.bias is None else self.bias
        if verbose:
            print(f"Input: {x}\nWeights: {self.get_weights()}\nBias: {bias}")
            print(f"--------------------------------------")
        return [i.calc(x) for i in self.neurons]

    def get_weights(self):
        return np.array([i.weights for i in self.neurons])

    def backpropagate(self, y, deltas, weights):
        res = []
        for i, yi, w in zip(self.neurons, y, weights):
            d = i.back_calc(yi, deltas, w)
            res.append(d)
        return res

    def __repr__(self) -> str:
        return f"Layer({len(self.neurons)}, {self.activation.__name__})"

    def __str__(self) -> str:
        return f"Layer({len(self.neurons)}, {self.activation.__name__})"

    def __dict__(self):
        return {"activation": self.activation, "units": len(self.neurons)}
