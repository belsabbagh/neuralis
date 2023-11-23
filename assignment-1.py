from src.networks import FeedForward
from src.activations import relu
from src.layers import Dense

activation = relu()
units = [4, 3, 1]
weights = [
    [
        [0.8, 0.5, 0.3, -0.1, 0.41],
        [-0.2, 0.1, 0.5, 0.22, -0.1],
        [0.4, 0.7, 0.1, -0.21, 0.2],
        [0.3, 0.3, 0.2, 0.61, -0.2],
    ],
    [
        [0.1, 0.1, 0.2, 0.24, -0.4],
        [-0.1, 0.22, 0.3, 0.7, -0.3],
        [0.3, 0.11, 0.14, -0.13, 0.8],
    ],
    [
        [0.21, 0.2, -0.1, 0.32],
    ],
]

nn = FeedForward(
    [
        *[Dense(u, activation, w) for u, w in zip(units, weights)],
    ]
)

if __name__ == "__main__":
    x = [0.2, -0.3, 0.1, 0.15]
    d = 1
    res, _ = nn(x, d)
    y = res[-1][0]
    e = d - y
    print(y, e)
