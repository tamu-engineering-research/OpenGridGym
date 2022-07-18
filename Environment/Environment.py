from dataclasses import dataclass, field
from typing import List
import os, sys
from pathlib import Path
import warnings
from tqdm import trange

path_to_OpenGridGym = Path(__file__).parents[1]
sys.path.insert(0, os.path.join(path_to_OpenGridGym))
for folder in ('Grids', 'Markets', 'Agents', 'utils'):
    path = os.path.join(path_to_OpenGridGym, folder)
    sys.path.insert(0, path)

from BaseGrid import BaseGrid
from BaseMarket import BaseMarket
from BaseAgent import BaseAgent

from method_update import update_instance_method
from iterator import episode_iterator

def ensure_type(obj, desired_type):
    '''
    Ensure that a given object is of desired type.

    Example

        >>> ensure_type(1, int)
        >>> ensure_type('1', int)
        TypeError: Expected object of type '<class 'int'>'
        but observed instead type '<class 'str'>'.

    '''
    if not isinstance(obj, desired_type):
        actual_type = type(obj)
        raise TypeError(f"Expected object of type '{desired_type}'\n"
                        f"but instead observed type '{actual_type}'.")

@dataclass
class Environment:
    '''
    Stores all the objects relevant to the simulation
    of electricity markets, and provides the user with
    an interface to manipulate the simulation process.

    Some method names are inspired by OpenAI Gym's style
    of defining environments. We employ the dataclass
    decorator to improve the user's debugging experience.

    '''
    grid:   BaseGrid
    market: BaseMarket
    agents: List[BaseAgent]
    max_steps: int = field(default=24*4)

    def __post_init__(self):
        # Ensure correctness of object types
        ensure_type(self.grid, BaseGrid)
        ensure_type(self.market, BaseMarket)
        ensure_type(self.agents, list)
        for agent in self.agents:
            ensure_type(agent, BaseAgent)

        # Assign pointers to environment
        self.grid.env = self
        self.market.env = self
        for agent in self.agents:
            agent.env = self

        self.reset()

    def reset(self):
        self.grid.reset()
        self.market.reset()
        # Instruct agents to get their first observations
        ...

    def step(self):
        # Gym: action -> done, info, reward, obs
        # Here: Call agent to act & it internally grabs info, reward, obs
        ...

    def iterate(self, num_iter=None, grid_iterator=trange, market_iterator=range, warn=True):
        '''
        Iterate through the episode starting at the current time
        self.grid.t and ending after num_iter iterations.

        Note: To reset start time, self.reset().
        Any explicit edit of self.grid.t must be done cautiously.

        For explanation of this method's parameters, please refer to
        the docstring of episode_iterator.

        Example:
            
            >>> env.grid.t # check current time
            50
            >>> for t in env.iterate(3): # progress for 3 steps only
            ...     print(t)
            ...
            51
            52
            53

        '''
        # Progress and yield time
        for t in episode_iterator(t_prev=self.grid.t, max_steps=self.max_steps,
                                    num_iter=num_iter, iterator=grid_iterator, warn=warn):
            
            # Market interactions
            self.market.reset()
            for t_market in self.market.iterate(iterator=market_iterator, warn=warn):
                self.callback_market_step(t_market)
            
            # # Grid interactions
            for agent in self.agents:
                agent.set_grid_actions()

            self.grid.t = t
            self.grid.step()

            yield t

    def callback_market_step(self, t_market):
        pass

    def render(self):
        ...

    def close(self):
        ...