import abc
from typing import Any


class BaseLearner(abc.ABC):
    """Base class for learners."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def eval(self) -> None:
        """set model to evaluation mode."""
        pass

    @abc.abstractmethod
    def train(self) -> None:
        """set model to train mode."""
        pass

    @abc.abstractmethod
    def select_target_action(self, state: Any) -> None:
        """select action for target/test."""
        pass

    @abc.abstractmethod
    def step(self, *args, **kwargs) -> None:
        """update relevant data/parameters for learner."""
        pass
