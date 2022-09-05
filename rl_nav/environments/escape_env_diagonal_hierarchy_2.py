import copy
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rl_nav import constants
from rl_nav.environments import escape_env
from rl_nav.utils import env_utils


class EscapeEnvDiagonalHierarchy2(escape_env.EscapeEnv):
    def __init__(
        self,
        training: bool,
        map_path: str,
        representation: str,
        reward_positions: List[Tuple[int]],
        reward_attributes: List[Dict],
        step_cost_factor: Union[float, int],
        partitions_path: str,
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

        self._sub_deltas = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([-1, 1]),
            5: np.array([1, 1]),
            6: np.array([1, -1]),
            7: np.array([-1, -1]),
        }

        self._sub_deltas_ = {
            (-1, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (0, -1): 3,
            (-1, 1): 4,
            (1, 1): 5,
            (1, -1): 6,
            (-1, -1): 7,
        }

        self._sub_action_space = [0, 1, 2, 3, 4, 5, 6, 7]
        # 0: LEFT
        # 1: UP
        # 2: RIGHT
        # 3: DOWN
        # 4: LEFT-UP
        # 5: RIGHT-UP
        # 6: RIGHT-DOWN
        # 7: LEFT-DOWN

        self._inverse_sub_action_mapping = {
            0: 2,
            1: 3,
            2: 0,
            3: 1,
            4: 6,
            5: 7,
            6: 4,
            7: 5,
        }

        partitions_ = env_utils.setup_partitions(partitions_path)

        with open(transition_structure, "r") as transition_json:
            env_structure = json.load(transition_json)

        letter_int_mapping = {
            k: int(i)
            for i, k in enumerate(
                [
                    node
                    for node in partitions_.keys()
                    if node in env_structure[constants.NODES].keys()
                ]
            )
        }

        # we want to specify partitions using letters in our txt files.
        # here it is far easier to deal with state/action spaces with integers.
        # the above is messy/hacky but deals with this dichotomy.
        self._partitions = {
            letter_int_mapping[k]: v
            for k, v in partitions_.items()
            if k in letter_int_mapping
        }

        self._state_position_mapping = {
            letter_int_mapping[i]: tuple(k)
            for i, k in env_structure[constants.NODES].items()
        }
        self._position_state_mapping = {
            tuple(k): letter_int_mapping[i]
            for i, k in env_structure[constants.NODES].items()
        }
        self._available_actions = {
            letter_int_mapping[i]: [letter_int_mapping[e] for e in k]
            for i, k in env_structure[constants.EDGES].items()
        }

        self._inverse_partition_mapping = {}
        for partition_index, partition_coordinates in self._partitions.items():
            for coordinate in partition_coordinates:
                self._inverse_partition_mapping[coordinate] = partition_index

        self._state_space = list(self._position_state_mapping.keys())
        self._action_space = list(range(len(self._state_position_mapping.keys())))

        # self._state_state_deltas = {
        #     state: {
        #         next_state: np.sqrt((np.array(state) - np.array(next_state)).sum())
        #         for next_state in self._state_space
        #     }
        #     for state in self._state_space
        # }
        # self._state_action_deltas = {
        #     state: {
        #         action: np.sqrt(
        #             (
        #                 np.array(state) - np.array(self._state_position_mapping[action])
        #             ).sum()
        #         )
        #         for action in self._action_space
        #     }
        #     for state in self._state_space
        # }

        self._start_state_space = list(self._position_state_mapping.keys())

        self._inverse_action_mapping = {
            state: {
                action: self._state_space.index(state) for action in self._action_space
            }
            for state in self._state_space
        }

    @property
    def position_state_mapping(self):
        return self._position_state_mapping

    def _low_step(self, action: int):

        reward = self._low_level_move_agent(delta=self._sub_deltas[action])
        new_state = self.get_state_representation()
        skeleton = self._env_skeleton()

        self._active = self._remain_active(reward=reward)

        self._train_episode_sub_position_history.append(tuple(self._agent_position))
        self._train_episode_sub_history.append(skeleton)

        return new_state

    def _high_step(self, state: Tuple, action: int):
        reward = self._move_agent(delta=action)
        new_state = self.get_state_representation()
        skeleton = self._env_skeleton()

        self._active = self._remain_active(reward=reward)
        self._test_episode_position_history.append(tuple(self._agent_position))
        self._test_episode_history.append(skeleton)

        return reward, new_state

    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Overrides step method in escape_env.py

        Take step in environment according to action of agent.

        Args:
            action: 0: left, 1: up, 2: right, 3: down

        Returns:
            reward: float indicating reward, 1 for target reached, 0 otherwise.
            next_state: new coordinates of agent.
        """
        assert (
            self._active
        ), "Environment not active. call reset_environment() to reset environment and make it active."
        if action is not None:
            assert (
                action in self._action_space
            ), f"Action given as {action}; must be 0: left, 1: up, 2: right or 3: down."

        low_state = copy.deepcopy(tuple(self._agent_position))
        state = self._inverse_partition_mapping[low_state]
        state_sub_states = self._partitions[state]
        low_new_state = copy.deepcopy(tuple(self._agent_position))

        if self._training:
            while low_new_state in state_sub_states:
                action = np.random.choice(self._sub_action_space)
                low_new_state = self._low_step(action=action)
            # use compute_reward here for high level state reward / new_state
            new_state = self._inverse_partition_mapping[low_new_state]
            delta = np.array(low_state) - np.array(low_new_state)
            reward = self._compute_reward(delta=delta)

        else:
            new_state = self._high_step(action=action)

        import pdb

        pdb.set_trace()

        # if training move on low level. If evaluating move on high level?

        # Here I want to do actions on sub-level.
        # Implement a sub-level history (for visualisation etc.)
        # a sub-level everything etc.

        return reward, new_state

    def _move_agent(
        self, delta: np.ndarray, phantom_position: Optional[np.ndarray] = None
    ):
        current_position = copy.deepcopy(self._agent_position)
        state = self._position_state_mapping[tuple(self._agent_position)]
        import pdb

        pdb.set_trace()
        if delta in self._available_actions[state]:
            self._agent_position = np.array(self._state_position_mapping[delta])

        delta = current_position - self._agent_position

        reward = self._compute_reward(delta=delta)

        return reward

    @property
    def action_deltas(self) -> Dict[int, np.ndarray]:
        return None

    @property
    def delta_actions(self) -> Dict[Tuple[int], int]:
        return {}

    @property
    def inverse_action_mapping(self) -> Dict[int, int]:
        return self._inverse_action_mapping

    @property
    def available_actions(self) -> Dict[int, Dict]:
        return self._available_actions

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

    def get_partition(self, state: Tuple):
        return self._inverse_partition_mapping[state] - 1  # for "#"" partition

    def get_centroid(self, partition):
        return self._centroids[partition + 1]  # for "#"" partition

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

    def _env_specific_reset(self, retain_history: bool):
        if not retain_history:
            if self._training:
                self._train_episode_sub_position_history = []
                self._train_episode_sub_history = []
            else:
                self._test_episode_sub_position_history = []
                self._test_episode_sub_history = []

    def _low_level_move_agent(
        self, delta: np.ndarray, phantom_position: Optional[np.ndarray] = None
    ) -> float:
        """Move agent. If provisional new position is a wall, no-op."""
        if phantom_position is None:
            current_position = self._agent_position
        else:
            current_position = phantom_position
        provisional_new_position = current_position + delta

        moving_into_wall = tuple(provisional_new_position) in self._wall_state_space

        moving_into_k_block = (
            (self._k_block_state_space is not None)
            and (tuple(provisional_new_position) in self._k_block_state_space)
            and (
                np.array_equal(delta, self._sub_deltas[2])
                or np.array_equal(delta, self._sub_deltas[3])
                or np.array_equal(delta, self._sub_deltas[6])
            )
        )

        moving_into_h_block = (
            (self._h_block_state_space is not None)
            and (tuple(provisional_new_position) in self._h_block_state_space)
            and (
                np.array_equal(delta, self._sub_deltas[0])
                or np.array_equal(delta, self._sub_deltas[3])
                or np.array_equal(delta, self._sub_deltas[7])
            )
        )

        moving_into_d_block = (
            (self._d_block_state_space is not None)
            and (tuple(provisional_new_position) in self._d_block_state_space)
            and (
                np.array_equal(delta, self._sub_deltas[3])
                or np.array_equal(delta, self._sub_deltas[6])
                or np.array_equal(delta, self._sub_deltas[7])
            )
        )

        moving_into_e_block = (
            (self._e_block_state_space is not None)
            and (tuple(provisional_new_position) in self._e_block_state_space)
            and (
                np.array_equal(delta, self._sub_deltas[0])
                or np.array_equal(delta, self._sub_deltas[2])
                or np.array_equal(delta, self._sub_deltas[3])
                or np.array_equal(delta, self._sub_deltas[6])
                or np.array_equal(delta, self._sub_deltas[7])
            )
        )

        moving_into_b_block = (
            (self._b_block_state_space is not None)
            and (tuple(provisional_new_position) in self._b_block_state_space)
            and (
                np.array_equal(delta, self._sub_deltas[3])
                or np.array_equal(delta, self._sub_deltas[6])
                or np.array_equal(delta, self._sub_deltas[7])
            )
        )

        moving_from_b_block = (
            (self._b_block_state_space is not None)
            and (tuple(current_position) in self._b_block_state_space)
            and (
                np.array_equal(delta, self._sub_deltas[1])
                or np.array_equal(delta, self._sub_deltas[4])
                or np.array_equal(delta, self._sub_deltas[5])
            )
        )

        moving_into_c_block = (self._c_block_state_space is not None) and (
            tuple(provisional_new_position) in self._c_block_state_space
        )

        moving_into_z_block = (
            (self._z_block_state_space is not None)
            and (tuple(provisional_new_position) in self._z_block_state_space)
            and (np.array_equal(delta, self._sub_deltas[7]))
        )

        moving_into_y_block = (
            (self._y_block_state_space is not None)
            and (tuple(provisional_new_position) in self._y_block_state_space)
            and (np.array_equal(delta, self._sub_deltas[6]))
        )

        moving_into_v_block = (
            (self._v_block_state_space is not None)
            and (tuple(provisional_new_position) in self._v_block_state_space)
            and (
                np.array_equal(delta, self._sub_deltas[7])
                or np.array_equal(delta, self._sub_deltas[0])
            )
        )

        moving_into_a_block = (
            (self._a_block_state_space is not None)
            and (tuple(provisional_new_position) in self._a_block_state_space)
            and (
                np.array_equal(delta, self._sub_deltas[0])
                or np.array_equal(delta, self._sub_deltas[3])
                or np.array_equal(delta, self._sub_deltas[6])
                or np.array_equal(delta, self._sub_deltas[7])
            )
        )

        moving_into_f_block = (
            (self._f_block_state_space is not None)
            and (tuple(provisional_new_position) in self._f_block_state_space)
            and (
                np.array_equal(delta, self._sub_deltas[2])
                or np.array_equal(delta, self._sub_deltas[3])
                or np.array_equal(delta, self._sub_deltas[6])
                or np.array_equal(delta, self._sub_deltas[7])
            )
        )

        moving_into_u_block = (
            (self._u_block_state_space is not None)
            and (tuple(provisional_new_position) in self._u_block_state_space)
            and (
                np.array_equal(delta, self._sub_deltas[2])
                or np.array_equal(delta, self._sub_deltas[6])
            )
        )

        moving_into_w_block = (
            (self._w_block_state_space is not None)
            and (tuple(provisional_new_position) in self._w_block_state_space)
            and (
                np.array_equal(delta, self._sub_deltas[2])
                or np.array_equal(delta, self._sub_deltas[3])
                or np.array_equal(delta, self._sub_deltas[5])
                or np.array_equal(delta, self._sub_deltas[6])
            )
        )

        moving_into_x_block = (
            (self._x_block_state_space is not None)
            and (tuple(provisional_new_position) in self._x_block_state_space)
            and (
                np.array_equal(delta, self._sub_deltas[0])
                or np.array_equal(delta, self._sub_deltas[3])
                or np.array_equal(delta, self._sub_deltas[4])
                or np.array_equal(delta, self._sub_deltas[7])
            )
        )

        move_permissible = all(
            [
                not moving_into_wall,
                not moving_into_k_block,
                not moving_into_h_block,
                not moving_into_d_block,
                not moving_into_e_block,
                not moving_into_b_block,
                not moving_from_b_block,
                not moving_into_c_block,
                not moving_into_z_block,
                not moving_into_v_block,
                not moving_into_u_block,
                not moving_into_w_block,
                not moving_into_x_block,
                not moving_into_y_block,
                not moving_into_a_block,
                not moving_into_f_block,
            ]
        )

        if phantom_position is not None:
            if move_permissible:
                return tuple(provisional_new_position)
            else:
                return tuple(current_position)
        else:
            if move_permissible:
                self._agent_position = provisional_new_position

            return self._compute_reward(delta=delta)
