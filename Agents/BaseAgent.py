from abc import ABC, abstractmethod

class BaseAgent(ABC):
    
    @abstractmethod
    def set_market_actions(self):
        ...
    
    @abstractmethod
    def set_grid_actions(self):
        ...

    def get_market_actions(self):
        return self.actions

    def get_grid_actions(self):
        return self.actions