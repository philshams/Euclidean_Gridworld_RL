import abc
from typing import Any


class BaseLearner(abc.ABC):
    """Base class for learners."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def eval(self) -> None:
        pass

    @abc.abstractmethod
    def train(self) -> None:
        pass

    @abc.abstractmethod
    def select_target_action(self, state: Any) -> None:
        pass

    @abc.abstractmethod
    def step(self, *args, **kwargs) -> None:
        """Update relevant data for learner."""
        pass
