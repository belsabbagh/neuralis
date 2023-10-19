import numpy as np
from src.layers import Layer


def update(w, y, d, learning_rate=0.1):
    bias, weights = w[0], w[1:]
    return np.array([bias + learning_rate * d, *(weights + learning_rate * d * y)])


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

    def update_weights(self, y_pred, deltas, update_fn=None, learning_rate=0.1):
        if update_fn is None:
            update_fn = update
        for layer, delta in zip(self.layers[1:], deltas):
            for neuron, y, d in zip(layer.neurons, y_pred, delta):
                neuron.weights = update_fn(neuron.weights, y, d, learning_rate)

    def fit(self, x, y, epochs=10, learning_rate=0.1):
        for i in range(epochs):
            yhat, deltas = self.calc(x, y)
            self.update_weights(yhat, deltas, learning_rate=learning_rate)
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
        input_shape = self.layers[0].get_weights().shape[1] - 1
        return (
            f"{self.__class__.__name__}(input_shape={input_shape}, layers=[\n"
            + ",\n".join("  " + str(i) for i in self.layers)
            + "\n])\n"
        )

    def __dict__(self):
        return {"layers": [dict(i) for i in self.layers]}
