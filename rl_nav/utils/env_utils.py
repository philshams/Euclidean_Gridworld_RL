import itertools
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from rl_nav import constants


def parse_map_outline(map_file_path: str, mapping: Dict[str, int]) -> np.ndarray:
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
                    loc=reward_parameters[constants.MEAN],
                    scale=reward_parameters[constants.VARIANCE],
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

    reward_availability = reward_attributes[constants.AVAILABILITY]
    reward_type = reward_attributes[constants.TYPE]
    reward_parameters = reward_attributes[constants.PARAMETERS]

    rewards = {
        reward_position: _get_reward_function(
            reward_availability, reward_type, reward_parameters
        )
        for reward_position in reward_positions
    }

    return rewards


def configure_state_space(map_outline, reward_positions: Optional):
    """Get state space for the environment from the parsed map.
    Further split state space into walls, valid positions, key possessions etc.
    """

    state_space_dictionary = {}

    state_indices = np.where(map_outline == 0)
    wall_indices = np.where(map_outline == 1)
    k_block_indices = np.where(map_outline == 0.6)
    h_block_indices = np.where(map_outline == 0.4)

    empty_state_space = list(zip(state_indices[1], state_indices[0]))
    wall_state_space = list(zip(wall_indices[1], wall_indices[0]))
    k_block_state_space = list(zip(k_block_indices[1], k_block_indices[0]))
    h_block_state_space = list(zip(h_block_indices[1], h_block_indices[0]))

    positional_state_space = empty_state_space

    if len(k_block_state_space):
        positional_state_space.extend(k_block_state_space)
        state_space_dictionary[constants.K_BLOCK_STATE_SPACE] = k_block_state_space
    if len(h_block_state_space):
        positional_state_space.extend(h_block_state_space)
        state_space_dictionary[constants.H_BLOCK_STATE_SPACE] = h_block_state_space

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
