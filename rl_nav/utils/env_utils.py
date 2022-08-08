import itertools
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from rl_nav import constants


def parse_map_outline(
    map_file_path: str, mapping: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """Method to parse ascii map and map settings from yaml file.

    Args:
        map_file_path: path to file containing map schematic.
        map_yaml_path: path to yaml file containing map config.

    Returns:
        multi_room_grid: numpy array of map state.
    """
    map_rows = []

    with open(map_file_path) as f:
        map_lines = f.read().splitlines()
        mapping = mapping or {
            item: i for i, item in enumerate(sorted(set("".join(map_lines))))
        }

        # flip indices for x, y referencing
        for i, line in enumerate(map_lines[::-1]):
            map_row = [mapping[char] for char in line]
            map_rows.append(map_row)

    assert all(
        len(i) == len(map_rows[0]) for i in map_rows
    ), "ASCII map must specify rectangular grid."

    multi_room_grid = np.array(map_rows, dtype=float)

    return multi_room_grid


def parse_x_positions(map_yaml_path: str, data_key: str):
    with open(map_yaml_path) as yaml_file:
        map_data = yaml.load(yaml_file, yaml.SafeLoader)

    positions = [tuple(p) for p in map_data[data_key]]

    return positions


def parse_map_positions(map_yaml_path: str) -> Tuple[List, List, List, List]:
    """Method to parse map settings from yaml file.

    Args:
        map_yaml_path: path to yaml file containing map config.

    Returns:
        initial_start_position: x,y coordinates for
            agent at start of each episode.
        key_positions: list of x, y coordinates of keys.
        door_positions: list of x, y coordinates of doors.
        reward_positions: list of x, y coordinates of rewards.
    """
    with open(map_yaml_path) as yaml_file:
        map_data = yaml.load(yaml_file, yaml.SafeLoader)

    start_positions = [tuple(map_data[constants.START_POSITION])]

    reward_positions = parse_x_positions(
        map_yaml_path=map_yaml_path, data_key=constants.REWARD_POSITIONS
    )
    key_positions = parse_x_positions(
        map_yaml_path=map_yaml_path, data_key=constants.KEY_POSITIONS
    )
    door_positions = parse_x_positions(
        map_yaml_path=map_yaml_path, data_key=constants.DOOR_POSITIONS
    )

    reward_statistics = map_data[constants.REWARD_STATISTICS]

    assert (
        len(start_positions) == 1
    ), "maximally one start position 'S' should be specified in ASCII map."

    assert len(door_positions) == len(
        key_positions
    ), "number of key positions must equal number of door positions."

    return (
        start_positions[0],
        key_positions,
        door_positions,
        reward_positions,
        reward_statistics,
    )


def setup_rewards(reward_positions, reward_attributes) -> Dict[Tuple, Callable]:
    class RewardFunction:
        def __init__(self, availability: Union[str, int]):
            self._original_availability = availability
            self._reset_availability()

        def _reset_availability(self, availability: Optional[Union[str, int]] = None):
            if availability is not None:
                use_availability = availability
            else:
                use_availability = self._original_availability
            if use_availability == constants.INFINITE:
                self._availability = np.inf
            else:
                self._availability = self._original_availability

        def reset(self, availability: Optional[Union[str, int]] = None):
            self._reset_availability(availability=availability)

        @property
        def availability(self):
            return self._availability

    class GaussianRewardFunction(RewardFunction):
        def __init__(self, availability: Union[str, int], reward_parameters: Dict):
            super().__init__(availability=availability)
            self._reward_parameters = reward_parameters

        def __call__(self):
            if self._availability > 0:
                reward = np.random.normal(
                    loc=self._reward_parameters[constants.MEAN],
                    scale=self._reward_parameters[constants.VARIANCE],
                )
                self._availability -= 1
            else:
                reward = 0
            return reward

    def _get_reward_function(
        availability: Union[str, int], reward_type: str, reward_parameters: Dict
    ) -> Callable:

        if reward_type == constants.GAUSSIAN:
            return GaussianRewardFunction(
                availability=availability, reward_parameters=reward_parameters
            )

    rewards = {}

    for reward_position, reward_availability, reward_specifications in zip(
        reward_positions,
        reward_attributes[constants.AVAILABILITY],
        reward_attributes[constants.PARAMETERS],
    ):
        reward_type, reward_parameters = zip(*reward_specifications.items())

        rewards[reward_position] = _get_reward_function(
            reward_availability, reward_type[0], reward_parameters[0]
        )

    return rewards


def configure_state_space(
    map_outline, reward_positions: Optional, one_dim_blocks: bool = True
):
    """Get state space for the environment from the parsed map.
    Further split state space into walls, valid positions, key possessions etc.
    """

    state_space_dictionary = {}

    state_indices = np.where(map_outline == constants.BLANK_VALUE)
    wall_indices = np.where(map_outline == constants.WALL_VALUE)
    k_block_indices = np.where(map_outline == constants.K_BLOCK_VALUE)
    h_block_indices = np.where(map_outline == constants.H_BLOCK_VALUE)
    d_block_indices = np.where(map_outline == constants.D_BLOCK_VALUE)
    e_block_indices = np.where(map_outline == constants.E_BLOCK_VALUE)
    b_block_indices = np.where(map_outline == constants.B_BLOCK_VALUE)
    c_block_indices = np.where(map_outline == constants.C_BLOCK_VALUE)
    z_block_indices = np.where(map_outline == constants.Z_BLOCK_VALUE)
    y_block_indices = np.where(map_outline == constants.Y_BLOCK_VALUE)
    v_block_indices = np.where(map_outline == constants.V_BLOCK_VALUE)
    x_block_indices = np.where(map_outline == constants.X_BLOCK_VALUE)
    f_block_indices = np.where(map_outline == constants.F_BLOCK_VALUE)
    a_block_indices = np.where(map_outline == constants.A_BLOCK_VALUE)
    w_block_indices = np.where(map_outline == constants.W_BLOCK_VALUE)
    u_block_indices = np.where(map_outline == constants.U_BLOCK_VALUE)

    empty_state_space = list(zip(state_indices[1], state_indices[0]))
    wall_state_space = list(zip(wall_indices[1], wall_indices[0]))
    k_block_state_space = list(zip(k_block_indices[1], k_block_indices[0]))
    h_block_state_space = list(zip(h_block_indices[1], h_block_indices[0]))
    d_block_state_space = list(zip(d_block_indices[1], d_block_indices[0]))
    e_block_state_space = list(zip(e_block_indices[1], e_block_indices[0]))
    b_block_state_space = list(zip(b_block_indices[1], b_block_indices[0]))
    c_block_state_space = list(zip(c_block_indices[1], c_block_indices[0]))
    z_block_state_space = list(zip(z_block_indices[1], z_block_indices[0]))
    y_block_state_space = list(zip(y_block_indices[1], y_block_indices[0]))
    v_block_state_space = list(zip(v_block_indices[1], v_block_indices[0]))
    x_block_state_space = list(zip(x_block_indices[1], x_block_indices[0]))
    f_block_state_space = list(zip(f_block_indices[1], f_block_indices[0]))
    a_block_state_space = list(zip(a_block_indices[1], a_block_indices[0]))
    w_block_state_space = list(zip(w_block_indices[1], w_block_indices[0]))
    u_block_state_space = list(zip(u_block_indices[1], u_block_indices[0]))

    positional_state_space = empty_state_space

    if one_dim_blocks:
        positional_state_space.extend(b_block_state_space)
        state_space_dictionary[constants.B_BLOCK_STATE_SPACE] = b_block_state_space

    if len(k_block_state_space):
        positional_state_space.extend(k_block_state_space)
        state_space_dictionary[constants.K_BLOCK_STATE_SPACE] = k_block_state_space
    if len(h_block_state_space):
        positional_state_space.extend(h_block_state_space)
        state_space_dictionary[constants.H_BLOCK_STATE_SPACE] = h_block_state_space
    if len(d_block_state_space):
        positional_state_space.extend(d_block_state_space)
        state_space_dictionary[constants.D_BLOCK_STATE_SPACE] = d_block_state_space
    if len(e_block_state_space):
        positional_state_space.extend(e_block_state_space)
        state_space_dictionary[constants.E_BLOCK_STATE_SPACE] = e_block_state_space
    if len(c_block_state_space):
        positional_state_space.extend(c_block_state_space)
        state_space_dictionary[constants.C_BLOCK_STATE_SPACE] = c_block_state_space
    if len(z_block_state_space):
        positional_state_space.extend(z_block_state_space)
        state_space_dictionary[constants.Z_BLOCK_STATE_SPACE] = z_block_state_space
    if len(y_block_state_space):
        positional_state_space.extend(y_block_state_space)
        state_space_dictionary[constants.Y_BLOCK_STATE_SPACE] = y_block_state_space
    if len(v_block_state_space):
        positional_state_space.extend(v_block_state_space)
        state_space_dictionary[constants.V_BLOCK_STATE_SPACE] = v_block_state_space
    if len(x_block_state_space):
        positional_state_space.extend(x_block_state_space)
        state_space_dictionary[constants.X_BLOCK_STATE_SPACE] = x_block_state_space
    if len(f_block_state_space):
        positional_state_space.extend(f_block_state_space)
        state_space_dictionary[constants.F_BLOCK_STATE_SPACE] = f_block_state_space
    if len(a_block_state_space):
        positional_state_space.extend(a_block_state_space)
        state_space_dictionary[constants.A_BLOCK_STATE_SPACE] = a_block_state_space
    if len(w_block_state_space):
        positional_state_space.extend(w_block_state_space)
        state_space_dictionary[constants.W_BLOCK_STATE_SPACE] = w_block_state_space
    if len(u_block_state_space):
        positional_state_space.extend(u_block_state_space)
        state_space_dictionary[constants.U_BLOCK_STATE_SPACE] = u_block_state_space
    else:
        state_space_dictionary[constants.C_BLOCK_STATE_SPACE] = []

    state_space_dictionary[constants.WALL_STATE_SPACE] = wall_state_space
    state_space_dictionary[constants.POSITIONAL_STATE_SPACE] = positional_state_space

    if reward_positions is not None:
        rewards_received_state_space = list(
            itertools.product([0, 1], repeat=len(reward_positions))
        )
        state_space_dictionary[
            constants.REWARDS_RECEIVED_STATE_SPACE
        ] = rewards_received_state_space
        state_space = [
            i[0] + i[1]
            for i in itertools.product(
                positional_state_space,
                rewards_received_state_space,
            )
        ]
    else:
        state_space = positional_state_space

    state_space_dictionary[constants.STATE_SPACE] = state_space

    return state_space_dictionary


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    # rgb channel last
    grayscale = np.dot(rgb[..., :3], [[0.299], [0.587], [0.114]])
    return grayscale
