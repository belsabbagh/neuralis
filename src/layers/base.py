

from abc import abstractmethod


class Layer:
    @abstractmethod
    def forward(self, x, verbose=1):
        pass

    @abstractmethod
    def backward(self, y, deltas):
        pass
    
    @abstractmethod
    def update(self, y, d, lr=0.01):
        pass
