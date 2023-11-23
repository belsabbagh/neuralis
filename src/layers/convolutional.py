import numpy as np
from src.layers.base import Layer
from src.activations import identity


def cross_correlate(mat, kernel):
    """Cross correlate a matrix with a kernel."""
    kernel = np.flip(kernel)
    return np.array(
        [
            [
                np.sum(
                    np.multiply(
                        mat[i : i + kernel.shape[0], j : j + kernel.shape[1]],
                        kernel,
                    )
                )
                for j in range(mat.shape[1] - kernel.shape[1] + 1)
            ]
            for i in range(mat.shape[0] - kernel.shape[0] + 1)
        ]
    )


class Convolutional(Layer):
    def __init__(
        self,
        input_shape,
        filter_size,
        output_depth,
        num_filters,
        activation=None,
        op=None,
        filters=None,
        biases=None,
    ):
        super().__init__()
        if op is None:
            op = np.dot
        self.op = op
        if activation is None:
            activation = identity()
        self.activation, self.derv = activation
        self.input_shape = input_shape
        self.filters = np.array(
            self.__init_filters(num_filters, output_depth, filter_size)
            if filters is None
            else filters
        )
        output_matrix_size = (i - filter_size + 1 for i in input_shape[1:])
        self.output_shape = (len(self.filters), *output_matrix_size)
        self.biases = np.array(
            self.__init_biases(num_filters, output_matrix_size)
            if biases is None
            else biases
        )

    def __init_filters(self, num_filters, depth, filter_size):
        return np.random.uniform(-1, 1, (num_filters, depth, filter_size, filter_size))

    def __init_biases(self, num_filters, output_matrix_size):
        return np.zeros((num_filters, *output_matrix_size))

    def forward(self, x, verbose=1):
        x = np.array(x)
        if verbose:
            print(f"Input: {x}")
        y = [
            np.sum(
                [
                    cross_correlate(x[i], self.filters[i][j])
                    for j in range(len(self.filters[i]))
                ],
                axis=0,
            )
            + self.biases[i]
            for i in range(len(self.filters))
        ]
        if verbose:
            print(f"Output: {y}")
            print(f"--------------------------------------")
        return np.array(y)

    def backward(self, y, deltas):
        dL_dinput = np.zeros(self.input_shape)
        dL_dfilters = np.zeros_like(self.filters)

        for i in range(len(self.filters)):
            dL_dfilters[i] = self.op(
                self.input_data,
                deltas[i],
            )
            dL_dinput += self.op(deltas[i], self.filters[i])
        return dL_dinput

    def update(self, y, d, lr=0.01):
        self.filters -= lr * d
        self.biases -= lr * y
