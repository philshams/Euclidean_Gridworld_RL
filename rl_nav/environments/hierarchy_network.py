import copy
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rl_nav import constants
from rl_nav.environments import escape_env


class HierarchyNetwork(escape_env.EscapeEnv):
    def __init__(
        self,
        training: bool,
        map_path: str,
        representation: str,
        reward_positions: List[Tuple[int]],
        reward_attributes: List[Dict],
        step_cost_factor: Union[float, int],
        transition_structure: str,
        start_position: Optional[Tuple[int]] = None,
        episode_timeout: Optional[Union[int, None]] = None,
        one_dim_blocks: Optional[bool] = True,
        scaling: Optional[int] = 1,
        grayscale: bool = True,
        batch_dimension: bool = True,
        torch_axes: bool = True,
    ):

        super().__init__(
            training=training,
            map_path=map_path,
            representation=representation,
            reward_positions=reward_positions,
            reward_attributes=reward_attributes,
            step_cost_factor=step_cost_factor,
            start_position=start_position,
            episode_timeout=episode_timeout,
            one_dim_blocks=one_dim_blocks,
            scaling=scaling,
            grayscale=grayscale,
            batch_dimension=batch_dimension,
            torch_axes=torch_axes,
            transition_matrix=False,
        )

        with open(transition_structure, "r") as transition_json:
            env_structure = json.load(transition_json)

        self._state_position_mapping = {
            int(i): tuple(k) for i, k in env_structure[constants.NODES].items()
        }
        self._position_state_mapping = {
            tuple(k): int(i) for i, k in env_structure[constants.NODES].items()
        }
        self._available_actions = {
            int(i): k for i, k in env_structure[constants.EDGES].items()
        }

        self._state_space = list(self._position_state_mapping.keys())
        self._action_space = list(self._state_position_mapping.keys())
        self._deltas = {a: a for a in self._action_space}
        self._start_state_space = list(self._position_state_mapping.keys())

        self._inverse_action_mapping = {
            state: {
                action: self._state_space.index(state) for action in self._action_space
            }
            for state in self._state_space
        }

    def _move_agent(
        self, delta: np.ndarray, phantom_position: Optional[np.ndarray] = None
    ):
        current_position = copy.deepcopy(self._agent_position)
        state = self._position_state_mapping[tuple(self._agent_position)]
        if delta in self._available_actions[state]:
            self._agent_position = np.array(self._state_position_mapping[delta])

        delta = current_position - self._agent_position

        return self._compute_reward(delta=delta)

    @property
    def action_deltas(self) -> Dict[int, np.ndarray]:
        return self._deltas

    @property
    def delta_actions(self) -> Dict[Tuple[int], int]:
        return {}

    @property
    def inverse_action_mapping(self) -> Dict[int, int]:
        return self._inverse_action_mapping

    @property
    def active(self) -> bool:
        """Determines episode termination."""
        return self._active

    @property
    def episode_step_count(self) -> int:
        """Number of steps taken in environment."""
        return self._episode_step_count

    @property
    def agent_position(self) -> Tuple[int, int]:
        """Current x, y position of agent."""
        return tuple(self._agent_position)

    @property
    def action_space(self) -> List[int]:
        """Actions (as integer indices) available in environment"""
        return self._action_space

    @property
    def state_space(self) -> List[Tuple[int]]:
        """List of tuples of states available in envrionment."""
        return self._state_space

    @property
    def positional_state_space(self) -> List[Tuple[int]]:
        """List of tuples of positional components
        of the states available in envrionment."""
        return self._positional_state_space

    @property
    def total_rewards_available(self) -> Union[float, int]:
        """Total scalar reward available from the environment
        from instantiation to termination."""
        return self._total_rewards_available

    @property
    def visitation_counts(self) -> np.ndarray:
        """Number of times agent has visited each state"""
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
