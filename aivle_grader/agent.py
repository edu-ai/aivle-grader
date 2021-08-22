from abc import ABCMeta, abstractmethod


class Agent(metaclass=ABCMeta):
    @abstractmethod
    def step(self, state):
        pass

    @abstractmethod
    def reset(self):
        pass
