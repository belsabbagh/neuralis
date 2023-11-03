import numpy as np
from src.layers import Layer


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


def _backpropagate(layers, y, d):
    y, out = y[:-1], y[-1]
    deltas = np.subtract(out, d)
    res = [deltas]
    hidden_layers, out_layer = layers[:-1], layers[-1]
    prev_weights = out_layer.get_weights()
    for yhat, layer in zip(y[::-1], hidden_layers[::-1]):
        deltas = layer.backpropagate(yhat, deltas, prev_weights.T)
        prev_weights = layer.get_weights()
        res.append(deltas)
    return np.array(res[::-1], dtype=object)


class FeedForward:
    layers = None

    def __init__(self, layers: list[Layer] = None) -> None:
        self.layers = [] if layers is None else layers

    def add(self, layer: Layer):
        self.layers.append(layer)

    def __calc(self, x):
        """Calculate the output of the network.
        """
        res = []
        x_in = x
        for layer in self.layers:
            x_in = np.array(layer.calc(x_in))
            res.append(x_in)
        return res

    def update_weights(self, y_pred, deltas, update_fn=None, alpha=0.1):
        if update_fn is None:
            update_fn = update
        for layer, delta in zip(self.layers[1:], deltas):
            for neuron, y, d in zip(layer.neurons, y_pred, delta):
                neuron.weights = update_fn(neuron.weights, y, d, alpha)

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
