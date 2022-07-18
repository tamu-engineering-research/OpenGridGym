from dataclasses import dataclass, field
from BaseMarket import BaseMarket

@dataclass
class DoNothingMarket(BaseMarket):
    max_steps: int = field(default=100)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.t = -1

    def step(self):
        pass


if __name__ == '__main__':

    market = DoNothingMarket()
    print(market)
    print(isinstance(market, BaseMarket))