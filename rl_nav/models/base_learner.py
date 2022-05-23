import abc
from typing import Any, Dict


class BaseLearner(abc.ABC):
    """Base class for learners."""

    def __init__(self):
        self._env_transition_matrix = None

    @property
    def env_transition_matrix(self):
        return self._env_transition_matrix

    @env_transition_matrix.setter
    def env_transition_matrix(self, transition_matrix: Dict):
        # ensures transition matrix is information available
        # only if explicitly permitted.
        if self._env_transition_matrix is not None:
            self._env_transition_matrix = transition_matrix

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
