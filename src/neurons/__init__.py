import numpy as np


class Neuron:
    weights = None

    def __init__(self, activation, weights) -> None:
        weights = np.array(weights)
        self.activation = activation
        self.weights = weights

    @staticmethod
    def __sop(x, w):
        return np.dot(x, w.T[1:].T) - w.T[0]

    def calc(self, x):
        return self.activation(self.__sop(x, self.weights))

    def back_calc(self, y, delta, weights):
        return y * (1 - y) * np.dot(delta, weights)
