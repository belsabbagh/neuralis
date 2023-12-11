import numpy as np
from src.networks import FeedForward
from src.activations import sigmoid
from src.layers import Dense

activation = sigmoid()

nn = FeedForward(
    [
        Dense(3, activation, np.array([[0.44] + [1] * 3] * 3)),
        Dense(2, activation, np.array([[0.44] + [1] * 3] * 2)),
    ]
)


if __name__ == "__main__":
    x = [0.2, 0.4, 0.6]
    d = [4.33, 4.13]
    res, deltas = nn(x, d)
    print(res)
    print(deltas)
