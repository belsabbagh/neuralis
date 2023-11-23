import numpy as np


def _identity():
    return lambda x: x


def _identity_der():
    return lambda x: 1


def identity():
    return _identity(), _identity_der()


def sigmoid():
    s = lambda x: 1 / (1 + np.exp(-x))
    return s, lambda x: s(x) * (1 - s(x))


def tanh():
    t = lambda x: np.tanh(x)
    return t, lambda x: 1 - t(x) ** 2


def arctan():
    return lambda x: np.arctan(x), lambda x: 1 / (x**2 + 1)


def prelu(alpha=1):
    return lambda x: alpha * x if x < 0 else x, lambda x: alpha and x >= 0


def relu():
    return lambda x: 0 if x < 0 else x, lambda x: int(x >= 0)


def elu(alpha=1):
    return lambda x: alpha * (np.exp(x) - 1) if x < 0 else x, lambda x: alpha and x >= 0


def softplus():
    return lambda x: np.log(1 + np.exp(x)), sigmoid()


def binstep(threshold=0):
    return lambda x: int(x >= threshold)
