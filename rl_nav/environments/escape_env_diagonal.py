from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rl_nav.environments import escape_env


class EscapeEnvDiagonal(escape_env.EscapeEnv):
    """Grid world environment with multiple rooms.
    Between each room is a door, that requires a key to unlock.
    """

    def __init__(
        self,
        training: bool,
        map_path: str,
        representation: str,
        reward_positions: List[Tuple[int]],
        reward_attributes: List[Dict],
        step_cost_factor: Union[float, int],
        start_position: Optional[Tuple[int]] = None,
        episode_timeout: Optional[Union[int, None]] = None,
        one_dim_blocks: Optional[bool] = True,
        scaling: Optional[int] = 1,
        grayscale: bool = True,
        batch_dimension: bool = True,
        torch_axes: bool = True,
    ) -> None:
        """Class constructor.

        Args:
            training: signals whether environment instance is for training.
            map_path: path to txt or other ascii file with map specifications.
            representation: agent_position (for tabular) or pixel
                (for function approximation).
            reward_positions: list of positions in which reward is located.
            reward_attributes: list of attributes each reward exhibits
                (e.g. statistics).
            start_position: coordinate start position of agent.
            episode_timeout: number of steps before episode automatically terminates.
            scaling: optional integer (for use with pixel representations)
                specifying how much to expand state by.
            grayscale: whether to keep rgb channels or compress to grayscale.
            batch_dimension: (for use with certain techniques);
                numer of examples per optimisation step.
            torch_axes: whether to use torch or tf paradigm of axis ordering.
        """

        self._deltas = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([-1, 1]),
            5: np.array([1, 1]),
            6: np.array([1, -1]),
            7: np.array([-1, -1]),
        }

        self._deltas_ = {
            (-1, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (0, -1): 3,
            (-1, 1): 4,
            (1, 1): 5,
            (1, -1): 6,
            (-1, -1): 7,
        }

        self._action_space = [0, 1, 2, 3, 4, 5, 6, 7]
        # 0: LEFT
        # 1: UP
        # 2: RIGHT
        # 3: DOWN
        # 4: LEFT-UP
        # 5: RIGHT-UP
        # 6: RIGHT-DOWN
        # 7: LEFT-DOWN

        self._inverse_action_mapping = {0: 2, 1: 3, 2: 0, 3: 1, 4: 6, 5: 7, 6: 4, 7: 5}

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
        )

    def _move_agent(
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
                np.array_equal(delta, self._deltas[2])
                or np.array_equal(delta, self._deltas[3])
                or np.array_equal(delta, self._deltas[6])
            )
        )

        moving_into_h_block = (
            (self._h_block_state_space is not None)
            and (tuple(provisional_new_position) in self._h_block_state_space)
            and (
                np.array_equal(delta, self._deltas[0])
                or np.array_equal(delta, self._deltas[3])
                or np.array_equal(delta, self._deltas[7])
            )
        )

        moving_into_d_block = (
            (self._d_block_state_space is not None)
            and (tuple(provisional_new_position) in self._d_block_state_space)
            and (
                np.array_equal(delta, self._deltas[3])
                or np.array_equal(delta, self._deltas[6])
                or np.array_equal(delta, self._deltas[7])
            )
        )

        moving_into_e_block = (
            (self._e_block_state_space is not None)
            and (tuple(provisional_new_position) in self._e_block_state_space)
            and (
                np.array_equal(delta, self._deltas[0])
                or np.array_equal(delta, self._deltas[2])
                or np.array_equal(delta, self._deltas[3])
                or np.array_equal(delta, self._deltas[6])
                or np.array_equal(delta, self._deltas[7])
            )
        )

        moving_into_b_block = (
            (self._b_block_state_space is not None)
            and (tuple(provisional_new_position) in self._b_block_state_space)
            and (
                np.array_equal(delta, self._deltas[3])
                or np.array_equal(delta, self._deltas[6])
                or np.array_equal(delta, self._deltas[7])
            )
        )

        moving_from_b_block = (
            (self._b_block_state_space is not None)
            and (tuple(current_position) in self._b_block_state_space)
            and (
                np.array_equal(delta, self._deltas[1])
                or np.array_equal(delta, self._deltas[4])
                or np.array_equal(delta, self._deltas[5])
            )
        )

        moving_into_c_block = (self._c_block_state_space is not None) and (
            tuple(provisional_new_position) in self._c_block_state_space
        )

        moving_into_z_block = (
            (self._z_block_state_space is not None)
            and (tuple(provisional_new_position) in self._z_block_state_space)
            and (np.array_equal(delta, self._deltas[7]))
        )

        moving_into_y_block = (
            (self._y_block_state_space is not None)
            and (tuple(provisional_new_position) in self._y_block_state_space)
            and (np.array_equal(delta, self._deltas[6]))
        )

        moving_into_v_block = (
            (self._v_block_state_space is not None)
            and (tuple(provisional_new_position) in self._v_block_state_space)
            and (
                np.array_equal(delta, self._deltas[7])
                or np.array_equal(delta, self._deltas[0])
            )
        )

        moving_into_a_block = (
            (self._a_block_state_space is not None)
            and (tuple(provisional_new_position) in self._a_block_state_space)
            and (
                np.array_equal(delta, self._deltas[0])
                or np.array_equal(delta, self._deltas[3])
                or np.array_equal(delta, self._deltas[6])
                or np.array_equal(delta, self._deltas[7])
            )
        )

        moving_into_f_block = (
            (self._f_block_state_space is not None)
            and (tuple(provisional_new_position) in self._f_block_state_space)
            and (
                np.array_equal(delta, self._deltas[2])
                or np.array_equal(delta, self._deltas[3])
                or np.array_equal(delta, self._deltas[6])
                or np.array_equal(delta, self._deltas[7])
            )
        )

        moving_into_u_block = (
            (self._u_block_state_space is not None)
            and (tuple(provisional_new_position) in self._u_block_state_space)
            and (
                np.array_equal(delta, self._deltas[2])
                or np.array_equal(delta, self._deltas[6])
            )
        )

        moving_into_w_block = (
            (self._w_block_state_space is not None)
            and (tuple(provisional_new_position) in self._w_block_state_space)
            and (
                np.array_equal(delta, self._deltas[2])
                or np.array_equal(delta, self._deltas[3])
                or np.array_equal(delta, self._deltas[5])
                or np.array_equal(delta, self._deltas[6])
            )
        )

        moving_into_x_block = (
            (self._x_block_state_space is not None)
            and (tuple(provisional_new_position) in self._x_block_state_space)
            and (
                np.array_equal(delta, self._deltas[0])
                or np.array_equal(delta, self._deltas[3])
                or np.array_equal(delta, self._deltas[4])
                or np.array_equal(delta, self._deltas[7])
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

    def _env_specific_reset(self, retain_history):
        pass
