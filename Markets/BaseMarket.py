from abc import ABC, abstractmethod
import os, sys
from pathlib import Path

path_to_OpenEMS = Path(__file__).parents[1]
sys.path.insert(0, os.path.join(path_to_OpenEMS))
for folder in ('utils',):
    path = os.path.join(path_to_OpenEMS, folder)
    sys.path.insert(0, path)

from iterator import episode_iterator

class BaseMarket(ABC):

    @abstractmethod
    def reset(self):
        self.t = -1

    @abstractmethod
    def step(self):
        ...
    
    def iterate(self, num_iter=None, iterator=range, warn=True):
        '''
        Iterate through the market episode starting at the current step
        self.t and ending after num_iter iterations.

        Note: To reset start time, self.reset().
        Any explicit edit of self.t must be done cautiously.

        For explanation of this method's parameters, please refer to
        the docstring of episode_iterator.

        Example:
            
            >>> market.t # check current time
            50
            >>> for t in market.iterate(3): # progress for 3 steps only
            ...     print(t)
            ...
            51
            52
            53
        '''
        # Progress and yield time
        for t in episode_iterator(t_prev=self.t,
                                    max_steps=self.max_steps,
                                    num_iter=num_iter,
                                    iterator=iterator,
                                    warn=warn):
            
            for agent in self.env.agents:
                agent.set_market_actions()

            self.step()

            self.t = t
            yield t