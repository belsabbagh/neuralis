from src.networks import FeedForward
from src.activations import sigmoid
from src.layers import Layer


questions = [
    {"x": [0.4, -5.3, 0.03], "d": [1]},
    {"x": [-2.2, 6.5, 3.5], "d": [0]},
]

weights = [
    [
        [0.2, 0.12, -0.5, 0.3],
        [0.53, 0.21, 0.24, -0.5],
        [0.21, 0.31, -0.3, -0.32],
        [0.13, 0.5, -0.18, 0.33],
    ],
    [
        [0.2, 0.14, -0.7, 0.12, 0.3],
        [0.14, 0.34, 0.28, -0.41, 0.1],
        [0.25, 0.22, -0.7, -0.12, 0.2],
        [0.32, 0.6, -0.14, 0.38, 0.03],
    ],
    [[0.12, 0.19, -0.28, 0.11, 0.1]],
]
units = [4, 4, 1]
layers = [
    *[Layer(u, sigmoid, w) for u, w in zip(units, weights)],
]

nn = FeedForward(layers)


if __name__ == "__main__":
    print(nn)
    for q in questions:
        print(f"Question: {q}")
        x, d = q["x"], q["d"]
        res, _ = nn.calc(x, d)
        y = res[-1][0]
        print(f"  Before:", {"y": y, "e": d - y})
        model = nn.fit(x, d, 20000, alpha=0.05)
        res, _ = model.calc(x, d)
        y = res[-1][0]
        print(f"  After:", {"y": y, "e": d - y})
