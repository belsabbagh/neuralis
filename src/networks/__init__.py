import numpy as np
from src.layers import Layer


def update(w, d):
    return w


class FeedForward:
    layers = None

    def __init__(self, layers: list[Layer] = None) -> None:
        self.layers = [] if layers is None else layers

    def add(self, layer: Layer):
        self.layers.append(layer)

    def __calc(self, x):
        res = []
        x_in = x
        for layer in self.layers:
            x_in = np.array(layer.calc(x_in))
            res.append(x_in)
        return res

    def update_weights(self, deltas, update_fn=None):
        if update_fn is None:
            update_fn = update
        for layer, delta in zip(self.layers[1:], deltas):
            for neuron, d in zip(layer.neurons, delta):
                neuron.weights = update_fn(neuron.weights, d)

    def fit(self, x, d, epochs=10):
        for i in range(epochs):
            _, deltas = self.calc(x, d)
            self.update_weights(deltas)
        return self

    def calc(self, x, d):
        y = self.__calc(x)
        deltas = self.backpropagate(y, d)
        return y, deltas

    def backpropagate(self, y, d):
        out = y[-1]
        deltas = out * (1 - out) * (d - out)
        res = [deltas]
        layers = list(enumerate(self.layers))
        hidden_layers = layers[:-1]
        prev_weights = layers[-1][1].get_weights()
        for i, layer in reversed(hidden_layers):
            deltas = layer.backpropagate(y[i], deltas, prev_weights.T)
            prev_weights = layer.get_weights()
            res.append(deltas)
        return np.array(list(reversed(res)), dtype=object)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n" + "\n".join("  " + str(i)
                                                        for i in self.layers) + "\n)\n"
        )

    def __dict__(self):
        return {"layers": [dict(i) for i in self.layers]}
