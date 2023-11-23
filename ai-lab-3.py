from src.layers import Convolutional
import numpy as np
mat = [
    [103, 102, 99, 101, 100, 120],
    [98, 97, 100, 98, 99, 94],
    [100, 5, 102, 97, 98, 96],
    [97, 102, 100, 240, 100, 101],
    [102, 100, 99, 102, 98, 96],
    [103, 105, 92, 104, 99, 97],
]

kernel = np.array([
    [0.4, 0.6, 0.8],
    [0.3, 0.9, 1.2],
    [3.1, 1.2, 0.7],
])
if __name__ == "__main__":
    conv = Convolutional((6, 6), 3, 1, filters=[kernel])
    y = conv.forward(mat)
    print(y[0])