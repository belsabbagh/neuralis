from typing import Callable, TypeAlias
import numpy as np

ActivationFunction: TypeAlias = Callable[[float], float]
ActivationBuilderReturn: TypeAlias = tuple[ActivationFunction, ActivationFunction]


def _identity() -> ActivationFunction:
    return lambda x: x


def _identity_der() -> ActivationFunction:
    return lambda x: 1


def identity() -> ActivationBuilderReturn:
    return _identity(), _identity_der()


def sigmoid() -> ActivationBuilderReturn:
    s: ActivationFunction = lambda x: 1 / (1 + np.exp(-x))
    return s, lambda x: s(x) * (1 - s(x))


def tanh()->ActivationBuilderReturn:
    t: ActivationFunction = lambda x: np.tanh(x)
    return t, lambda x: 1 - t(x) ** 2


def arctan() -> ActivationBuilderReturn:
    return lambda x: np.arctan(x), lambda x: 1 / (x**2 + 1)


def prelu(alpha=1) -> ActivationBuilderReturn:
    return lambda x: alpha * x if x < 0 else x, lambda x: alpha and x >= 0


def relu() -> ActivationBuilderReturn:
    return lambda x: 0 if x < 0 else x, lambda x: int(x >= 0)


def elu(alpha=1) -> ActivationBuilderReturn:
    return lambda x: alpha * (np.exp(x) - 1) if x < 0 else x, lambda x: alpha and x >= 0


def softplus() -> ActivationBuilderReturn:
    return lambda x: np.log(1 + np.exp(x)), sigmoid()[0]


def binstep(threshold=0) -> ActivationFunction:
    return lambda x: int(x >= threshold)
