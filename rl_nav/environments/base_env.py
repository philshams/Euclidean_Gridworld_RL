import abc
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
from rl_nav import constants
from rl_nav.utils import env_utils


class BaseEnvironment(abc.ABC):
    """Base class for RL environments.

    Abstract methods:
        step: takes action produces reward and next state.
        reset_environment: reset environment and return initial state.
    """
    MAPPING = {
        constants.WALL_CHARACTER: 1,
        constants.OPEN_CHARACTER: 0,
        constants.START_CHARACTER: 0,
        constants.REWARD_CHARACTER: 0,
    }

    def __init__(self):
        self._training: bool
        self._active: bool
        self._episode_step_count: int

    def _setup_environment(self, map_ascii_path: Optional[str] = None):

        if map_ascii_path is not None:
            self._map = env_utils.parse_map_outline(
                map_file_path=map_ascii_path, mapping=self.MAPPING
            )

        self._rewards = env_utils.setup_rewards(
            self._reward_positions, self._reward_attributes
        )

        availability = self._reward_attributes[constants.AVAILABILITY]
        if availability == constants.INFINITE:
            self._total_rewards = np.inf
        else:
            self._total_rewards = availability * len(self._rewards)

        (
            self._positional_state_space,
            self._rewards_received_state_space,
            self._state_space,
            self._wall_state_space,
        ) = env_utils.configure_state_space(
            map_outline=self._map,
            reward_positions=self._reward_positions,
        )

    def average_values_over_positional_states(
        self, values: Dict[Tuple[int], float]
    ) -> Dict[Tuple[int], float]:
        """For certain analyses (e.g. plotting value functions) we want to
        average the values for each position over all non-positional state information--
        in this case the key posessions.
        Args:
            values: full state-action value information
        Returns:
            averaged_values: (positional-state)-action values.
        """
        averaged_values = {}
        for state in self._positional_state_space:
            non_positional_set = [
                v for k, v in values.items() if k[:2] == state
            ]
            non_positional_mean = np.mean(non_positional_set, axis=0)
            averaged_values[state] = non_positional_mean
        return averaged_values

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Take step in environment according to action of agent."""
        pass

    @abc.abstractmethod
    def reset_environment(self, train: bool):
        """Reset environment.

        Args:
            train: whether episode is for train or test
            (may affect e.g. logging).
        """
        pass

    @property
    def active(self) -> bool:
        return self._active

    @property
    def episode_step_count(self) -> int:
        return self._episode_step_count

    @property
    def agent_position(self) -> Tuple[int, int]:
        return tuple(self._agent_position)

    @property
    def action_space(self) -> List[int]:
        return self.ACTION_SPACE

    @property
    def state_space(self) -> List[Tuple[int, int]]:
        return self._state_space

    @property
    def positional_state_space(self):
        return self._positional_state_space

    @property
    def visitation_counts(self) -> np.ndarray:
        return self._visitation_counts

    @property
    def train_episode_history(self) -> List[np.ndarray]:
        return self._train_episode_history

    @property
    def test_episode_history(self) -> List[np.ndarray]:
        return self._test_episode_history

    @property
    def train_episode_partial_history(self) -> List[np.ndarray]:
        return self._train_episode_partial_history

    @property
    def test_episode_partial_history(self) -> List[np.ndarray]:
        return self._test_episode_partial_history

    @property
    def train_episode_position_history(self) -> np.ndarray:
        return np.array(self._train_episode_position_history)

    @property
    def test_episode_position_history(self) -> np.ndarray:
        return np.array(self._test_episode_position_history)
