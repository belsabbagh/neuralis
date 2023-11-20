import numpy as np
from src.layers.base import Layer
from src.activations import identity


class Convolutional(Layer):
    def __init__(self, input_shape, filter_size, num_filters, activation=None, op=None):
        super().__init__()
        if op is None:
            op = np.dot
        self.op = op
        if activation is None:
            activation = identity
        self.activation, self.derv = activation
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape
        self.filter_shape = (num_filters, filter_size, filter_size)
        self.output_shape = (
            num_filters,
            input_height - filter_size + 1,
            input_width - filter_size + 1,
        )

        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, x, verbose=1):
        if verbose:
            print(f"Input: {x}\nWeights: {self.get_weights()}")
        y = np.zeros(self.output_shape)
        for i in range(self.num_filters):
            y[i] = self.op(self.input_data, self.filters[i])
        y = self.activation(y)
        if verbose:
            print(f"Output: {y}")
            print(f"--------------------------------------")
        return y

    def backward(self, y, deltas):
        dL_dinput = np.zeros(self.input_shape)
        dL_dfilters = np.zeros_like(self.filters)

        for i in range(self.num_filters):
            dL_dfilters[i] = self.op(
                self.input_data,
                deltas[i],
            )
            dL_dinput += self.op(deltas[i], self.filters[i])
        return dL_dinput

    def update(self, y, d, lr=0.01):
        self.filters -= lr * d
        self.biases -= lr * y
