import abc
from typing import Dict, List, Tuple, Union

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
        constants.WALL_CHARACTER: constants.WALL_VALUE,
        constants.H_BLOCK_CHARACTER: constants.H_BLOCK_VALUE,
        constants.K_BLOCK_CHARACTER: constants.K_BLOCK_VALUE,
        constants.OPEN_CHARACTER: constants.BLANK_VALUE,
        constants.START_CHARACTER: constants.BLANK_VALUE,
        constants.REWARD_CHARACTER: constants.BLANK_VALUE,
        constants.B_BLOCK_CHARACTER: constants.B_BLOCK_VALUE,
        constants.D_BLOCK_CHARACTER: constants.D_BLOCK_VALUE,
        constants.E_BLOCK_CHARACTER: constants.E_BLOCK_VALUE,
        constants.NO_START_BLOCK_CHAR: constants.NO_START_BLOCK_VALUE,
        constants.Z_BLOCK_CHARACTER: constants.Z_BLOCK_VALUE,
        constants.Y_BLOCK_CHARACTER: constants.Y_BLOCK_VALUE,
        constants.X_BLOCK_CHARACTER: constants.X_BLOCK_VALUE,
        constants.V_BLOCK_CHARACTER: constants.V_BLOCK_VALUE,
        constants.F_BLOCK_CHARACTER: constants.F_BLOCK_VALUE,
        constants.A_BLOCK_CHARACTER: constants.A_BLOCK_VALUE,
        constants.U_BLOCK_CHARACTER: constants.U_BLOCK_VALUE,
        constants.W_BLOCK_CHARACTER: constants.W_BLOCK_VALUE,
    }

    def __init__(self, training: bool):
        """Class constructor.

        Args:
            training: signals whether environment instance is for training
            ot testing.
        """
        self._training: bool = training
        self._active: bool
        self._episode_step_count: int

    def _setup_environment(self, map_ascii_path, reward_state: bool = False) -> None:
        """Environment setup method.

        Args:
            map_ascii_path: path to text file specifying map layout.
            reward_state: whether to include reward in state.
        """

        if map_ascii_path is not None:
            self._map = env_utils.parse_map_outline(
                map_file_path=map_ascii_path, mapping=self.MAPPING
            )[constants.GRID]

        self._rewards = env_utils.setup_rewards(
            self._reward_positions, self._reward_attributes
        )

        if reward_state:
            reward_position_arg = self._reward_positions
        else:
            reward_position_arg = None
        state_space_dictionary = env_utils.configure_state_space(
            map_outline=self._map,
            reward_positions=reward_position_arg,
            one_dim_blocks=self._one_dim_blocks,
        )

        self._positional_state_space = state_space_dictionary.get(
            constants.POSITIONAL_STATE_SPACE
        )
        self._rewards_received_state_space = state_space_dictionary.get(
            constants.REWARDS_RECEIVED_STATE_SPACE
        )
        self._state_space = state_space_dictionary.get(constants.STATE_SPACE)
        self._wall_state_space = state_space_dictionary.get(constants.WALL_STATE_SPACE)
        self._k_block_state_space = state_space_dictionary.get(
            constants.K_BLOCK_STATE_SPACE
        )
        self._h_block_state_space = state_space_dictionary.get(
            constants.H_BLOCK_STATE_SPACE
        )
        self._d_block_state_space = state_space_dictionary.get(
            constants.D_BLOCK_STATE_SPACE
        )
        self._e_block_state_space = state_space_dictionary.get(
            constants.E_BLOCK_STATE_SPACE
        )
        self._b_block_state_space = state_space_dictionary.get(
            constants.B_BLOCK_STATE_SPACE
        )
        self._c_block_state_space = state_space_dictionary.get(
            constants.C_BLOCK_STATE_SPACE
        )
        self._z_block_state_space = state_space_dictionary.get(
            constants.Z_BLOCK_STATE_SPACE
        )
        self._x_block_state_space = state_space_dictionary.get(
            constants.X_BLOCK_STATE_SPACE
        )
        self._v_block_state_space = state_space_dictionary.get(
            constants.V_BLOCK_STATE_SPACE
        )
        self._y_block_state_space = state_space_dictionary.get(
            constants.Y_BLOCK_STATE_SPACE
        )
        self._a_block_state_space = state_space_dictionary.get(
            constants.A_BLOCK_STATE_SPACE
        )
        self._f_block_state_space = state_space_dictionary.get(
            constants.F_BLOCK_STATE_SPACE
        )
        self._w_block_state_space = state_space_dictionary.get(
            constants.W_BLOCK_STATE_SPACE
        )
        self._u_block_state_space = state_space_dictionary.get(
            constants.U_BLOCK_STATE_SPACE
        )
        self._start_state_space = list(
            set(self._positional_state_space)
            - set(self._c_block_state_space)
            - set(self._reward_positions)
        )

    def average_values_over_positional_states(
        self, values: Dict[Tuple[int], float]
    ) -> Dict[Tuple[int], float]:
        """For certain analyses (e.g. plotting value functions) we want to
        average the values for each position over all non-positional state information--
        in this case perhaps the rewards received.
        Args:
            values: full state-action value information
        Returns:
            averaged_values: (positional-state)-action values.
        """
        averaged_values = {}
        for state in self._positional_state_space:
            non_positional_set = [v for k, v in values.items() if k[:2] == state]
            non_positional_mean = np.mean(non_positional_set, axis=0)
            averaged_values[state] = non_positional_mean
        return averaged_values

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Take step in environment according to action of agent."""
        pass

    @abc.abstractmethod
    def reset_environment(self) -> None:
        """Reset environment and associated quantities."""
        pass

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
