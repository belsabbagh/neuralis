import numpy as np
from src.layers.base import Layer
from src.activations import identity
from scipy.signal import correlate2d
import os


def matrix_to_latex(mat):
    """Convert a matrix to a LaTeX string."""
    content = "\\\\\n".join([" & ".join([str(j) for j in i]) for i in mat])
    return f"\\begin{{bmatrix}}\n{content}\n\\end{{bmatrix}}"


def convolution_iterator(mat, kernel, stride):
    """Iterate over a matrix with a kernel."""
    for i in range(0, mat.shape[0] - kernel.shape[0] + 1, stride[0]):
        for j in range(0, mat.shape[1] - kernel.shape[1] + 1, stride[1]):
            yield i, j, mat[i : i + kernel.shape[0], j : j + kernel.shape[1]]


def cross_correlate(mat, kernel, stride, biases):
    """Cross correlate a matrix with a kernel."""
    res = correlate2d(mat, kernel, mode="valid")[:: stride[0], :: stride[1]]
    return res


class Convolutional(Layer):
    def __init__(
        self,
        input_shape,
        filter_size,
        output_depth,
        num_filters,
        stride=(1, 1),
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
        self.stride = stride
        output_matrix_size = (
            i - filter_size // j + 1 for i, j in zip(input_shape[1:], stride)
        )
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
        y = np.zeros(self.output_shape)
        print(f"\\text{{Output Shape: }} {y.shape}")
        for i in range(len(self.filters)):
            for j in range(len(self.filters[i])):
                c=cross_correlate(x[j], self.filters[i][j], self.stride, self.biases[i])
                c2= c + self.biases[i]
                y[i] = self.activation(c2)
                print(f"Y_{{{i+1}{j+1}}}=\\sigma({matrix_to_latex(c)} + {matrix_to_latex(self.biases[i])}) = {matrix_to_latex(y[i])} \\\\")
        # y = np.array(
        #     [
        #         self.activation(
        #             np.sum(
        #                 [
        #                     cross_correlate(
        #                         x[i], self.filters[i][j], self.stride, self.biases[i]
        #                     )
        #                     for j in range(len(self.filters[i]))
        #                 ],
        #                 axis=0,
        #             ) + self.biases[i]
        #         )
        #         for i in range(len(self.filters))
        #     ]
        # )
        if verbose:
            print(f"Output: {y}")
            print(f"--------------------------------------")
        return y

    def backward(self, y, deltas):
        return np.array(
            [
                [
                    [
                        cross_correlate(deltas[i], self.filters[i][j])
                        for j in range(len(self.filters[i]))
                    ]
                    for i in range(len(self.filters))
                ],
            ]
        )

    def update(self, y, d, lr=0.01):
        self.filters -= lr * d
        self.biases -= lr * y
