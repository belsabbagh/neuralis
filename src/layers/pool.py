import numpy as np
from src.layers.base import Layer


class Pool(Layer):
    def __init__(self, pool_size, input_shape=None, pool_fn=np.max) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.input_shape = input_shape
    
    def forward(self, x, verbose=1):
        if self.input_shape is None:
            self.input_shape = x.shape
        self.num_channels, self.input_height, self.input_width = x.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        # Determining the output shape
        y = np.zeros((self.num_channels, self.output_height, self.output_width))

        # Iterating over different channels
        for c in range(self.num_channels):
            # Looping through the height
            for i in range(self.output_height):
                # looping through the width
                for j in range(self.output_width):

                    # Starting postition
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    # Ending Position
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    # Creating a patch from the input data
                    patch = x[c, start_i:end_i, start_j:end_j]

                    #Finding the maximum value from each patch/window
                    self.output[c, i, j] = np.max(patch)

        return y
    
    def backward(self, y, deltas):
        dL_dinput = np.zeros(self.input_shape)
        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i, end_i = self.__get_bounds(i)
                    start_j, end_j = self.__get_bounds(j)
                    patch = self.input_data[c, start_i:end_i, start_j:end_j]
                    mask = patch == np.max(patch)
                    dL_dinput[c,start_i:end_i, start_j:end_j] = deltas[c, i, j] * mask
        return dL_dinput
    
    
    def __get_bounds(self, i):
        start = i * self.pool_size
        end = start + self.pool_size
        return start, end
    
    
class MaxPool(Pool):
    def __init__(self, pool_size, input_shape=None) -> None:
        super().__init__(pool_size, input_shape=input_shape, pool_fn=np.max)
        
        
class AveragePool(Pool):
    def __init__(self, pool_size, input_shape=None) -> None:
        super().__init__(pool_size, input_shape=input_shape, pool_fn=np.average)
        
        
        