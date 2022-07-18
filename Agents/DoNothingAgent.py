from dataclasses import dataclass
from BaseAgent import BaseAgent

@dataclass
class DoNothingAgent(BaseAgent):

    def __post_init__(self):
        self.actions = []
    
    def set_market_actions(self):
        pass
    
    def set_grid_actions(self):
        pass


if __name__ == '__main__':

    agent = DoNothingAgent()
    print(agent)
    print(isinstance(agent, BaseAgent))