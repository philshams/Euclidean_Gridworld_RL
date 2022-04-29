import abc
from typing import Any


class BaseLearner(abc.ABC):
    """Base class for learners."""

    def __init__(self):
        pass

    def train(self) -> None:
        """set model to train mode."""
        self._training = True
        self._train()

    def eval(self) -> None:
        """set model to evaluation mode."""
        self._training = False
        self._eval()

    @abc.abstractmethod
    def _eval(self) -> None:
        """set model to evaluation mode."""
        pass

    @abc.abstractmethod
    def _train(self) -> None:
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
