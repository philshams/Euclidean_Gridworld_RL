import copy
import itertools
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from environments import base_env
from rl_nav import constants
from rl_nav.utils import env_utils


class EscapeEnv(base_env.BaseEnvironment):
    """Grid world environment with multiple rooms.
    Between each room is a door, that requires a key to unlock.
    """

    ACTION_SPACE = [0, 1, 2, 3]
    # 0: LEFT
    # 1: UP
    # 2: RIGHT
    # 3: DOWN

    DELTAS = {
        0: np.array([-1, 0]),
        1: np.array([0, 1]),
        2: np.array([1, 0]),
        3: np.array([0, -1]),
    }

    def __init__(
        self,
        map_ascii_path: str,
        representation: str,
        reward_positions: List[Tuple[int]],
        reward_attributes: List[Dict],
        start_position: Optional[Tuple[int]] = None,
        episode_timeout: Optional[Union[int, None]] = None,
        scaling: Optional[int] = 1,
        field_x: Optional[int] = 1,
        field_y: Optional[int] = 1,
        grayscale: bool = True,
        batch_dimension: bool = True,
        torch_axes: bool = True,
    ) -> None:
        """Class constructor.

        Args:
            map_ascii_path: path to txt or other ascii file with map specifications.
            representation: agent_position (for tabular) or pixel
                (for function approximation).
            episode_timeout: number of steps before episode automatically terminates.
            scaling: optional integer (for use with pixel representations)
                specifying how much to expand state by.
            field_x: integer (required for use with partial observability
                in pixel representations) specifying how many pixels in each x
                direction the agent can see.
            field_y: integer (required for use with partial observability
                in pixel representations) specifying how many pixels in each y
                direction the agent can see.
        """

        self._active: bool = False

        self._training: bool
        self._episode_step_count: int
        self._representation = representation

        self._reward_positions = [tuple(p) for p in reward_positions]
        self._reward_attributes = reward_attributes
        self._starting_xy = start_position

        self._agent_position: np.ndarray
        self._rewards_state: np.ndarray

        self._train_episode_position_history: List[List[int]]
        self._test_episode_position_history: List[List[int]]
        self._train_episode_history: List[np.ndarray]
        self._test_episode_history: List[np.ndarray]

        if self._representation == constants.PO_PIXEL:
            self._train_episode_partial_history: List[np.ndarray]
            self._test_episode_partial_history: List[np.ndarray]

        self._episode_timeout = episode_timeout or np.inf
        self._scaling = scaling
        self._field_x = field_x
        self._field_y = field_y
        self._grayscale = grayscale
        self._batch_dimension = batch_dimension
        self._torch_axes = torch_axes

        self._setup_environment(
            map_ascii_path=map_ascii_path
        )

        # states are zero, -1 removes walls from counts.
        self._visitation_counts = -1 * copy.deepcopy(self._map)

    def _setup_environment(
        self, map_ascii_path: Optional[str] = None
    ):

        if map_ascii_path is not None:
            self._map = env_utils.parse_map_outline(
                map_file_path=map_ascii_path, mapping=self.MAPPING
            )

        self._rewards = env_utils.setup_rewards(
            self._reward_positions, self._reward_attributes
        )
        self._total_rewards = len(self._rewards)

        (
            self._positional_state_space,
            self._rewards_received_state_space,
            self._state_space,
            self._wall_state_space,
        ) = env_utils.configure_state_space(
            map_outline=self._map,
            reward_positions=self._reward_positions,
        )

    def _env_skeleton(
        self,
        rewards: Union[None, str, Tuple[int]] = "state",
        doors: Union[None, str] = "state",
        agent: Union[None, str, np.ndarray] = "state",
        cue: Union[None, str, np.ndarray] = None,
    ) -> np.ndarray:
        """Get a 'skeleton' of map e.g. for visualisation purposes.

        Args:
            rewards: # TODO whether or not to mark out rewards (ignores magnitudes).
            show_doors: whether or not to mark out doors.
            show_agent: whether or not to mark out agent position

        Returns:
            skeleton: np array of map.
        """
        # flip size so indexing is consistent with axis dimensions
        skeleton = np.ones(self._map.shape + (3,))

        # make walls black
        skeleton[self._map == 1] = np.zeros(3)

        if rewards is not None:
            if isinstance(rewards, str):
                if rewards == constants.STATIONARY:
                    reward_iterate = list(self._rewards.keys())
                elif rewards == constants.STATE:
                    reward_positions = list(self._rewards.keys())
                    reward_iterate = [
                        reward_positions[i]
                        for i, r in enumerate(self._rewards_state)
                        if not r
                    ]
                else:
                    raise ValueError(f"Rewards keyword {rewards} not identified.")
            elif isinstance(rewards, tuple):
                reward_positions = list(self._rewards.keys())
                reward_iterate = [
                    reward_positions[i] for i, r in enumerate(rewards) if not r
                ]
            else:
                raise ValueError(
                    "Rewards must be string with relevant keyword or keystate list"
                )
            # show reward in red
            for reward in reward_iterate:
                skeleton[reward[::-1]] = [1.0, 0.0, 0.0]

        if agent is not None:
            if isinstance(agent, str):
                if agent == constants.STATE:
                    agent_position = self._agent_position
                elif agent == constants.STATIONARY:
                    agent_position = self._starting_xy
            else:
                agent_position = agent
            # show agent
            skeleton[tuple(agent_position[::-1])] = 0.5 * np.ones(3)

        return skeleton

    def _partial_observation(self, state, agent_position):

        height = state.shape[0]
        width = state.shape[1]

        # out of bounds needs to be different from wall pixels
        OUT_OF_BOUNDS_PIXEL = 0.2 * np.ones(3)

        # nominal bounds on field of view (pre-edge cases)
        x_min = agent_position[1] - self._field_x
        x_max = agent_position[1] + self._field_x
        y_min = agent_position[0] - self._field_y
        y_max = agent_position[0] + self._field_y

        state = state[
            max(0, x_min) : min(x_max, width) + 1,
            max(0, y_min) : min(y_max, height) + 1,
            :,
        ]

        # edge case contingencies
        if 0 > x_min:
            append_left = 0 - x_min

            fill = np.kron(
                OUT_OF_BOUNDS_PIXEL,
                np.ones((append_left, state.shape[1], 1)),
            )
            state = np.concatenate(
                (fill, state),
                axis=0,
            )
        if x_max >= width:
            append_right = x_max + 1 - width

            fill = np.kron(
                OUT_OF_BOUNDS_PIXEL,
                np.ones((append_right, state.shape[1], 1)),
            )
            state = np.concatenate(
                (state, fill),
                axis=0,
            )
        if 0 > y_min:
            append_below = 0 - y_min

            fill = np.kron(
                OUT_OF_BOUNDS_PIXEL,
                np.ones((state.shape[0], append_below, 1)),
            )
            state = np.concatenate(
                (fill, state),
                axis=1,
            )
        if y_max >= height:
            append_above = y_max + 1 - height

            fill = np.kron(
                OUT_OF_BOUNDS_PIXEL,
                np.ones((state.shape[0], append_above, 1)),
            )
            state = np.concatenate(
                (state, fill),
                axis=1,
            )

        return state

    def get_state_representation(
        self,
        tuple_state: Optional[Tuple] = None,
    ) -> Union[tuple, np.ndarray]:
        """From current state, produce a representation of it.
        This can either be a tuple of the agent and key positions,
        or a top-down pixel view of the environment (for DL)."""
        if self._representation == constants.AGENT_POSITION:
            return tuple(self._agent_position) + tuple(self._rewards_state)
        elif self._representation in [constants.PIXEL, constants.PO_PIXEL]:
            if tuple_state is None:
                agent_position = self._agent_position
                state = self._env_skeleton()  # H x W x C
            else:
                agent_position = tuple_state[:2]
                rewards = tuple_state[2 + len(self._key_positions) :]
                state = self._env_skeleton(
                    rewards=rewards, agent=agent_position
                )  # H x W x C

            if self._representation == constants.PO_PIXEL:
                state = self._partial_observation(
                    state=state, agent_position=agent_position
                )

            if self._grayscale:
                state = utils.rgb_to_grayscale(state)

            if self._torch_axes:
                state = np.transpose(state, axes=(2, 0, 1))  # C x H x W
                state = np.kron(state, np.ones((1, self._scaling, self._scaling)))
            else:
                state = np.kron(state, np.ones((self._scaling, self._scaling, 1)))

            if self._batch_dimension:
                # add batch dimension
                state = np.expand_dims(state, 0)

            return state

    def _move_agent(self, delta: np.ndarray) -> float:
        """Move agent. If provisional new position is a wall, no-op."""
        provisional_new_position = self._agent_position + delta

        moving_into_wall = tuple(provisional_new_position) in self._wall_state_space

        if not moving_into_wall:
            self._agent_position = provisional_new_position

        return self._compute_reward()

    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Take step in environment according to action of agent.

        Args:
            action: 0: left, 1: up, 2: right, 3: down

        Returns:
            reward: float indicating reward, 1 for target reached, 0 otherwise.
            next_state: new coordinates of agent.
        """
        assert (
            self._active
        ), "Environment not active. call reset_environment() to reset environment and make it active."
        assert (
            action in self.ACTION_SPACE
        ), f"Action given as {action}; must be 0: left, 1: up, 2: right or 3: down."

        reward = self._move_agent(delta=self.DELTAS[action])
        new_state = self.get_state_representation()
        skeleton = self._env_skeleton()

        self._active = self._remain_active(reward=reward)
        self._episode_step_count += 1

        if self._training:
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1
            self._train_episode_position_history.append(tuple(self._agent_position))
            self._train_episode_history.append(skeleton)
            if self._representation == constants.PO_PIXEL:
                self._train_episode_partial_history.append(
                    self._partial_observation(
                        state=skeleton, agent_position=self._agent_position
                    )
                )
        else:
            self._test_episode_position_history.append(tuple(self._agent_position))
            self._test_episode_history.append(skeleton)
            if self._representation == constants.PO_PIXEL:
                self._test_episode_partial_history.append(
                    self._partial_observation(
                        state=skeleton, agent_position=self._agent_position
                    )
                )

        return reward, new_state

    def _compute_reward(self) -> float:
        """Check for reward, i.e. whether agent position is equal to a reward position.
        If reward is found, add to rewards received log.
        """
        if (
            tuple(self._agent_position) in self._rewards
            and tuple(self._agent_position) not in self._rewards_received
        ):
            reward = self._rewards.get(tuple(self._agent_position))()
            reward_index = list(self._rewards.keys()).index(tuple(self._agent_position))
            self._rewards_state[reward_index] = 1
            self._rewards_received.append(tuple(self._agent_position))
        else:
            reward = 0.0

        return reward

    def _remain_active(self, reward: float) -> bool:
        """Check on reward / timeout conditions whether episode should be terminated.

        Args:
            reward: total reward accumulated so far.

        Returns:
            remain_active: whether to keep episode active.
        """
        conditions = [
            self._episode_step_count == self._episode_timeout,
            len(self._rewards_received) == self._total_rewards,
        ]
        return not any(conditions)

    def reset_environment(
        self, train: bool = True, map_yaml_path: Optional[str] = None
    ) -> Tuple[int, int, int]:
        """Reset environment.

        Bring agent back to starting position.

        Args:
            train: whether episode is for train or test (affects logging).
        """
        if map_yaml_path is not None:
            self._setup_environment(map_yaml_path=map_yaml_path)

        self._active = True
        self._episode_step_count = 0
        self._training = train
        if self._starting_xy is not None:
            self._agent_position = np.array(self._starting_xy)
        else:
            x_lim, y_lim = self._map.shape
            x_pos = np.random.randint(1, x_lim - 1)
            y_pos = np.random.randint(1, y_lim - 1)
            self._agent_position = np.array([x_pos, y_pos])

        for reward in self._rewards.values():
            reward.reset()

        self._rewards_received = []
        self._rewards_state = np.zeros(len(self._rewards), dtype=int)

        initial_state = self.get_state_representation()
        skeleton = self._env_skeleton()

        if train:
            self._train_episode_position_history = [tuple(self._agent_position)]
            self._train_episode_history = [skeleton]
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1
            if self._representation == constants.PO_PIXEL:
                self._train_episode_partial_history = [
                    self._partial_observation(
                        state=skeleton, agent_position=self._agent_position
                    )
                ]
        else:
            self._test_episode_position_history = [tuple(self._agent_position)]
            self._test_episode_history = [skeleton]
            if self._representation == constants.PO_PIXEL:
                self._test_episode_partial_history = [
                    self._partial_observation(
                        state=skeleton, agent_position=self._agent_position
                    )
                ]

        return initial_state
