import numpy as np
from src.layers import Layer


class FeedForward:
    layers = None

    def __init__(self, layers: list[Layer] = None) -> None:
        self.layers = [] if layers is None else layers

    def add(self, layer: Layer):
        self.layers.append(layer)

    def calc(self, x):
        res = []
        x_in = x
        for layer in self.layers:
            x_in = np.array(layer.calc(x_in))
            res.append(x_in)
        return res

    def fit(self, x, d):
        y = self.calc(x)
        return y, self.backpropagate(y, d)

    def backpropagate(self, y, d):
        out = y[-1]
        deltas = out * (1 - out) * (d - out)
        res = [deltas]
        layers = list(enumerate(self.layers))
        hidden_layers = layers[:-1][1:]
        prev_weights = layers[-1][1].get_weights()
        for i, layer in reversed(hidden_layers):
            deltas = layer.backpropagate(y[i], deltas, prev_weights.T)
            prev_weights = layer.get_weights()
            res.append(deltas)
        return list(reversed(res))

    def __repr__(self):
        return (
            "FeedForward(\n" + "\n".join("  " + str(i)
                                         for i in self.layers) + "\n)\n"
        )

    def __dict__(self):
        return {"layers": [dict(i) for i in self.layers]}
