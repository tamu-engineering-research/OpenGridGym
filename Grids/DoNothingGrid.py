from dataclasses import dataclass, field
from BaseGrid import BaseGrid

@dataclass
class DoNothingGrid(BaseGrid):
    case_folder: str = field(default='')

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.t = -1

    def step(self):
        pass


if __name__ == '__main__':

    grid = DoNothingGrid()
    print(grid)
    print(isinstance(grid, BaseGrid))