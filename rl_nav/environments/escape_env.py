import abc
import copy
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rl_nav import constants
from rl_nav.environments import base_env


class EscapeEnv(base_env.BaseEnvironment, abc.ABC):
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
        transition_matrix: Optional[bool] = True,
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
            step_cost_factor: cost of each step (multplied by euclidean
                distance travelled)
            start_position: coordinate start position of agent.
            episode_timeout: number of steps before episode automatically terminates.
            scaling: optional integer (for use with pixel representations)
                specifying how much to expand state by.
            grayscale: whether to keep rgb channels or compress to grayscale.
            batch_dimension: (for use with certain techniques);
                numer of examples per optimisation step.
            torch_axes: whether to use torch or tf paradigm of axis ordering.
        """

        super().__init__(training=training)

        self._representation = representation

        self._reward_positions = [tuple(p) for p in reward_positions]
        self._reward_attributes = reward_attributes
        self._starting_xy = start_position
        self._step_cost_factor = step_cost_factor

        self._agent_position: np.ndarray
        self._rewards_state: np.ndarray

        self._train_episode_position_history: List[List[int]]
        self._test_episode_position_history: List[List[int]]
        self._train_episode_history: List[np.ndarray]
        self._test_episode_history: List[np.ndarray]

        if self._representation == constants.PO_PIXEL:
            self._train_episode_partial_history: List[np.ndarray]
            self._test_episode_partial_history: List[np.ndarray]

        self._one_dim_blocks = one_dim_blocks
        self._episode_timeout = episode_timeout or np.inf
        self._standard_episode_timeout = episode_timeout or np.inf
        self._scaling = scaling
        self._grayscale = grayscale
        self._batch_dimension = batch_dimension
        self._torch_axes = torch_axes

        self._reward_state = self._representation == constants.AGENT_POSITION_REWARD
        self._setup_environment(
            map_ascii_path=map_path, reward_state=self._reward_state
        )

        # states are zero, -1 removes walls from counts.
        self._visitation_counts = -1 * copy.deepcopy(self._map)

        if transition_matrix:
            self._transition_matrix = {
                state: {
                    action: self._move_agent(
                        delta=self.action_deltas[action],
                        phantom_position=np.array(state),
                    )
                    for action in self.action_space
                }
                for state in self._positional_state_space
            }
        else:
            self._transition_matrix = {}

    @property
    def starting_xy(self) -> Tuple[int]:
        return self._starting_xy

    @property
    def reward_positions(self):
        return self._reward_positions

    @property
    def action_deltas(self) -> Dict[int, np.ndarray]:
        return self._deltas

    @property
    def delta_actions(self) -> Dict[Tuple[int], int]:
        return self._deltas_

    @property
    def inverse_action_mapping(self) -> Dict[int, int]:
        return self._inverse_action_mapping

    @property
    def transition_matrix(self) -> Dict:
        return self._transition_matrix

    def _env_skeleton(
        self,
        rewards: Union[None, str, Tuple[int]] = "state",
        agent: Union[None, str, np.ndarray] = "state",
    ) -> np.ndarray:
        """Get a 'skeleton' of map e.g. for visualisation purposes.

        Args:
            rewards: specifies if/how to represent rewards in env;
                if "state" is given, rewards represented according to
                    rewards_received state.
                if "stationary" is given, rewards represented where they are
                    specified at start.
                if None given, rewards are not represented.
                if tuple of boolean ints given, rewards of indices with True
                    value represented.
            agent: specifies if/how to represent agent in env;
                if "state" is given, agent represented according to
                    current agent position.
                if "stationary" is given, agent represented at start position.
                if None given, agent not represented.

        Returns:
            skeleton: np array of map.
        """
        # flip size so indexing is consistent with axis dimensions
        skeleton = np.ones(self._map.shape + (3,))

        # make walls black
        skeleton[self._map == 1] = np.zeros(3)

        # make b, c blocks black
        skeleton[self._map == 0.5] = np.zeros(3)
        skeleton[self._map == 0.55] = np.zeros(3)

        # make k blocks, h blocks, d blocks, e blocks silver
        skeleton[self._map == 0.4] = 0.75 * np.ones(3)
        skeleton[self._map == 0.6] = 0.75 * np.ones(3)
        skeleton[self._map == 0.51] = 0.75 * np.ones(3)
        skeleton[self._map == 0.52] = 0.75 * np.ones(3)

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
            if agent_position is not None:
                # show agent
                skeleton[tuple(agent_position[::-1])] = 0.5 * np.ones(3)

        return skeleton

    def save_as_array(self, save_path: str):
        env_skeleton = self._env_skeleton(agent=None, rewards=constants.STATIONARY)
        np.save(save_path, env_skeleton)

    def get_state_representation(
        self,
        tuple_state: Optional[Tuple] = None,
    ) -> Union[tuple, np.ndarray]:
        """From current state, produce a representation of it.
        This can either be a tuple of the agent and key positions,
        or a top-down pixel view of the environment (for DL)."""
        if self._representation == constants.AGENT_POSITION:
            return tuple(self._agent_position)
        elif self._representation == constants.AGENT_POSITION_REWARD:
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

    @abc.abstractmethod
    def _move_agent(
        self, delta: np.ndarray, phantom_position: Optional[np.ndarray] = None
    ) -> float:
        """Move agent. If provisional new position is a wall, no-op."""
        pass

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
            action in self._action_space
        ), f"Action given as {action}; must be 0: left, 1: up, 2: right or 3: down."

        reward = self._move_agent(delta=self._deltas[action])
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

    def _compute_reward(self, delta: np.ndarray) -> float:
        """Check for reward, i.e. whether agent position is equal to a reward position.
        If reward is found, add to rewards received log. Step cost factor added as penalty.

        Args:
            delta: distance travelled in x, y directions.
        """
        reward = 0.0
        if tuple(self._agent_position) in self._rewards:
            reward_availability = self._rewards[
                tuple(self._agent_position)
            ].availability
            if reward_availability > 0:
                reward += self._rewards.get(tuple(self._agent_position))()
                self._num_rewards_sampled += 1
                self._reward_received += reward
                if reward_availability == 1:
                    reward_index = list(self._rewards.keys()).index(
                        tuple(self._agent_position)
                    )
                    self._rewards_state[reward_index] = 1

        # step cost
        reward -= self._step_cost_factor * np.sqrt(np.sum(delta**2))

        return reward

    def _remain_active(self, reward: float) -> bool:
        """Check on reward / timeout conditions whether episode should be terminated.

        Args:
            reward: total reward accumulated so far.

        Returns:
            remain_active: whether to keep episode active.
        """
        conditions = [
            self._episode_step_count + 1 == self._episode_timeout,
            self._num_rewards_sampled == self._total_rewards_available,
        ]
        return not any(conditions)

    def save_history(self, save_path: str):
        if self._training:
            np.save(save_path, np.array(self._train_episode_position_history))
        else:
            np.save(save_path, np.array(self._test_episode_position_history))

    def reset_environment(
        self,
        map_yaml_path: Optional[str] = None,
        episode_timeout: Optional[int] = None,
        retain_history: Optional[bool] = False,
        start_position: Optional[Tuple[int]] = None,
        reward_availability: Optional[str] = None,
    ) -> Tuple[int, int, int]:
        """Reset environment.

        Bring agent back to starting position.

        Args:
            train: whether episode is for train or test (affects logging).
        """
        if map_yaml_path is not None:
            self._setup_environment(
                map_yaml_path=map_yaml_path, reward_state=self._reward_state
            )
        if episode_timeout is not None:
            # allow for temporary switch to timeout conditions
            self._episode_timeout = episode_timeout
        else:
            self._episode_timeout = self._standard_episode_timeout

        self._active = True
        self._episode_step_count = 0
        if start_position is not None:
            self._agent_position = np.array(start_position)
        elif self._starting_xy is not None:
            if any(isinstance(pos, list) for pos in self._starting_xy):
                self._agent_position = np.array(random.choice(self._starting_xy))
            else:
                self._agent_position = np.array(self._starting_xy)
        else:
            random_position_index = np.random.choice(len(self._start_state_space))
            self._agent_position = np.array(
                self._start_state_space[random_position_index]
            )

        for reward in self._rewards.values():
            reward.reset(availability=reward_availability)

        availability = (
            reward_availability or self._reward_attributes[constants.AVAILABILITY]
        )
        if constants.INFINITE in availability:
            self._total_rewards_available = np.inf
        else:
            self._total_rewards_available = sum(availability) * len(self._rewards)

        self._reward_received = 0
        self._num_rewards_sampled = 0
        self._rewards_state = np.zeros(len(self._rewards), dtype=int)

        initial_state = self.get_state_representation()
        skeleton = self._env_skeleton()

        if not retain_history:
            if self._training:
                self._train_episode_position_history = []
                self._train_episode_history = []
                if self._representation == constants.PO_PIXEL:
                    self._train_episode_partial_history = []
            else:
                self._test_episode_position_history = []
                self._test_episode_history = []
                if self._representation == constants.PO_PIXEL:
                    self._test_episode_partial_history = []

        if self._training:
            self._train_episode_position_history.append(tuple(self._agent_position))
            self._train_episode_history.append(skeleton)
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1
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

        return initial_state
