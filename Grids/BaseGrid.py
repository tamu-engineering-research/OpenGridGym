from abc import ABC, abstractmethod

class BaseGrid(ABC):

    @abstractmethod
    def reset(self):
        self.t = -1

    @abstractmethod
    def step(self):
        pass