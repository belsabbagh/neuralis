import numpy as np


class Neuron:
    weights = None

    def __init__(self, activation, weights, bias) -> None:
        self.activation = activation
        self.weights = weights
        self.bias = bias

    @staticmethod
    def __sop(x, w, b):
        return np.dot(x, w) - b

    def calc(self, x):
        return self.activation(self.__sop(x, self.weights, self.bias))

    def back_calc(self, y, delta, weights):
        return y * (1 - y) * np.dot(delta, weights)
