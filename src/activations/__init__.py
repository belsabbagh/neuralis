import numpy as np


def nothing(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def heaviside(x):
    return int(x >= 0)
