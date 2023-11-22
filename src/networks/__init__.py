import numpy as np
from src.layers import Dense


def _backpropagate(layers, y, d):
    y, out = y[:-1], y[-1]
    deltas = np.subtract(out, d)
    res = [deltas]
    hidden_layers, out_layer = layers[:-1], layers[-1]
    prev_weights = out_layer.get_weights()
    for yhat, layer in zip(y[::-1], hidden_layers[::-1]):
        deltas = layer.backward(yhat, deltas, prev_weights.T)
        prev_weights = layer.get_weights()
        res.append(deltas)
    return np.array(res[::-1], dtype=object)


class FeedForward:
    layers = None

    def __init__(self, layers: list[Dense] = None) -> None:
        self.layers = [] if layers is None else layers

    def add(self, layer: Dense):
        self.layers.append(layer)

    def __calc(self, x):
        """Calculate the output of the network.
        """
        res = []
        x_in = x
        for layer in self.layers:
            x_in = np.array(layer.forward(x_in))
            res.append(x_in)
        return res

    def update_weights(self, y_pred, deltas,alpha=0.1):
        for layer, delta in zip(self.layers[1:], deltas):
            layer.update(y_pred, delta, alpha=alpha)

    def fit(self, x, y, epochs=10, alpha=0.01):
        for i in range(epochs):
            yhat, deltas = self(x, y)
            self.update_weights(yhat, deltas, alpha=alpha)
        return self

    def __call__(self, x, d):
        y = self.__calc(x)
        deltas = self.backpropagate(y, d)
        return y, deltas

    def backpropagate(self, y, d):
        return _backpropagate(self.layers, y, d)

    def __repr__(self):
        input_shape = self.layers[0].get_weights().shape[1] - 1
        return (
            f"{self.__class__.__name__}(input_shape={input_shape}, layers=[\n"
            + ",\n".join("  " + str(i) for i in self.layers)
            + "\n])\n"
        )

    def __dict__(self):
        return {"layers": [dict(i) for i in self.layers]}
