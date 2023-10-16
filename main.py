from src.networks import FeedForward
from src.activations import sigmoid
from src.layers import Layer, InputLayer


questions = [
    {
        "x": [0.4, -5.3, 0.03],
        "d": 1,
    },
    {"x": [-2.2, 6.5, 3.5], "d": 0},
]

weights = [
    [
        [0.12, -0.5, 0.3],
        [0.21, 0.24, -0.5],
        [0.31, -0.3, -0.32],
        [0.5, -0.18, 0.33],
    ],
    [
        [0.14, -0.7, 0.12, 0.3],
        [0.34, 0.28, -0.41, 0.1],
        [0.22, -0.7, -0.12, 0.2],
        [0.6, -0.14, 0.38, 0.03],
    ],
    [[0.19, -0.28, 0.11, 0.1]],
]
biases = [[0.2, 0.53, 0.21, 0.13], [0.2, 0.14, 0.25, 0.32], [0.12]]
units = [4, 4, 1]
layers = [
    InputLayer(3),
    *[Layer(u, sigmoid, w, b) for u, w, b in zip(units, weights, biases)],
]

nn = FeedForward(layers)


if __name__ == "__main__":
    answers = []
    for q in questions:
        x, d = q["x"], q["d"]
        res, deltas = nn.fit(x, d)
        y = res[-1][0]
        e = d - y
        answers.append({"y": y, "e": e, "deltas": deltas})
    for ans in answers:
        print(f"y={ans['y']}, e={ans['e']}")
        print(ans["deltas"])
