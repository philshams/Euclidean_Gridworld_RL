import copy
import itertools
import random
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from rl_nav import constants
from rl_nav.models import base_learner
from rl_nav.utils import feature_utils
from rl_nav.utils import learning_rate_schedules


class StateLinearFeatureLearner(base_learner.BaseLearner):
    """Model with linear combination of features (for state values)."""

    def __init__(
        self,
        features: Dict[str, Dict[str, Any]],
        action_space: List[int],
        state_space: List[Tuple[int, int]],
        learning_rate: Type[learning_rate_schedules.LearningRateSchedule],
        gamma: float,
        initialisation_strategy: Dict,
        behaviour: str,
        target: str,
        imputation_method: str,
        update_no_op: bool,
    ):
        """Class constructor.

        Args:
            action_space: list of actions available.
            state_space: list of states.
            learning_rate: learning_rage.
            gamma: discount factor.
            initialisation_strategy: name of network initialisation strategy.
            behaviour: name of behaviour type e.g. epsilon_greedy.
            target: name of target type e.g. greedy.
            imputation_method: name of method to impute values at test time
                for states not present during training,
                e.g. near_neighbours or random.
        """
        super().__init__()

        self._env_transition_matrix = {}

        self._action_space = action_space
        self._state_space = state_space

        self._state_id_mapping = {state: i for i, state in enumerate(self._state_space)}
        self._id_state_mapping = {i: state for i, state in enumerate(self._state_space)}

        self._state_visitation_counts = {s: 0 for s in self._state_space}

        self._behaviour = behaviour
        self._target = target
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._training = True

        self._imputation_method = imputation_method
        self._allow_state_instantiation = False

        self._update_no_op = update_no_op

        (
            self._feature_extractors,
            self._feature_dim,
        ) = self._setup_feature_extractors(features=features)

        self._weight_matrix = self._initialise_values(
            initialisation_strategy=initialisation_strategy
        )

        self._state_features = np.zeros((len(self._state_space), self._feature_dim))

        for s, state in enumerate(self._state_space):
            self._state_features[s, :] = self._extract_features(state)

        self._wm_change: bool
        self._compute_state_values()

    def _train(self):
        pass

    def _eval(self):
        self._latest_state_values = {
            self._id_state_mapping[i]: values
            for i, values in enumerate(self._state_values)
        }

    @property
    def state_action_id_mapping(self):
        return self._state_action_id_mapping

    @property
    def _state_values(self):
        if self._wm_change:
            self._compute_state_values()
        return self.__state_values

    @property
    def state_values(self):
        if self._training:
            values = {
                self._id_state_mapping[i]: values
                for i, values in enumerate(self._state_values)
            }
            return values
        else:
            return self._latest_state_values

    @property
    def state_visitation_counts(self) -> Dict[Tuple[int, int], int]:
        """number of times each state has been visited."""
        return self._state_visitation_counts

    def _setup_feature_extractors(self, features: Dict[str, Dict[str, Any]]):
        if constants.STATE_ID in features:
            features[constants.STATE_ID][
                constants.STATE_ID_MAPPING
            ] = self._state_id_mapping

        if constants.COARSE_CODING in features:
            features[constants.COARSE_CODING][constants.STATE_SPACE] = self._state_space
            if features[constants.COARSE_CODING][constants.AUGMENT_ACTIONS]:
                features[constants.COARSE_CODING][
                    constants.AUGMENT_ACTIONS
                ] = self._action_space
            else:
                features[constants.COARSE_CODING][constants.AUGMENT_ACTIONS] = None

        if constants.ACTION_ONE_HOT in features:
            features[constants.ACTION_ONE_HOT][
                constants.ACTION_SPACE
            ] = self._action_space

        return feature_utils.get_feature_extractors(features=features)

    def _extract_features(self, state: Tuple[int, int]):
        feature_vector = np.concatenate(
            [extractor(state) for extractor in self._feature_extractors]
        )
        return feature_vector

    def _compute_state_values(self):
        self.__state_values = np.dot(self._weight_matrix, self._state_features.T)
        self._wm_change = False

    def _initialise_values(self, initialisation_strategy: str) -> np.ndarray:
        """Initialise values for each state, action pair in state-action space.

        Args:
            initialisation_strategy: name of method used to initialise.

        Returns:
            initial_values: matrix containing state-action id / value mapping.
        """
        initialisation_strategy_name = list(initialisation_strategy.keys())[0]
        if isinstance(initialisation_strategy_name, (int, float)):
            return initialisation_strategy_name * np.ones(self._feature_dim)
        elif initialisation_strategy_name == constants.RANDOM_UNIFORM:
            return np.random.rand(len(self._feature_dim))
        elif initialisation_strategy_name == constants.RANDOM_NORMAL:
            return np.random.normal(loc=0, scale=0.1, size=(self._feature_dim))
        elif initialisation_strategy_name == constants.ZEROS:
            return np.zeros(self._feature_dim)
        elif initialisation_strategy_name == constants.ONES:
            return np.ones(self._feature_dim)

    def select_target_action(self, state: Tuple[int, int], excess_state_mapping) -> int:
        """Select action according to target policy, i.e. policy being learned.
        Here, action with highest value in given state is selected.

        Args:
            state: current state.

        Returns:
            action: greedy action.
        """
        if self._target == constants.GREEDY:
            action = self._greedy_action(
                state=state, excess_state_mapping=excess_state_mapping
            )
        elif self._target == constants.GREEDY_SAMPLE:
            action = self._greedy_sample_action(
                state=state, excess_state_mapping=excess_state_mapping
            )
        return action

    def _greedy_action(
        self, state: Tuple[int, int], excess_state_mapping: Optional[Dict] = None
    ) -> int:
        """Find action with highest value in given state.

        Args:
            state: state for which to find action with highest value.
            excess_state_mapping: mapping from state to near neighbours.

        Returns:
            action: action with highest value in state given.
        """
        next_state_value_estimates = []

        for action in self._action_space:
            next_state = self._env_transition_matrix[state][action]
            next_state_id = self._state_id_mapping.get(next_state)
            next_state_value_estimate = self._state_values[next_state_id]
            next_state_value_estimates.append(next_state_value_estimate)

        return np.argmax(next_state_value_estimates)

    def _epsilon_greedy_action(
        self,
        state: Tuple[int, int],
        epsilon: float,
        excess_state_mapping: Optional[Dict] = None,
    ) -> int:
        """Choose greedy policy with probability
        (1 - epsilon) and a random action with probability epsilon.

        Args:
            state: state for which to find action.
            epsilon: parameter controlling randomness.

        Returns:
            action: action chosen according to epsilon greedy.
        """
        if random.random() < epsilon:
            action = random.choice(self._action_space)
        else:
            action = self._greedy_action(
                state=state,
                excess_state_mapping=excess_state_mapping,
            )
        return action

    def select_behaviour_action(
        self, state: Tuple[int, int], epsilon: float, excess_state_mapping=None
    ) -> Tuple[int, float]:
        """Select action with behaviour policy, i.e. policy collecting trajectory data
        and generating behaviour. Sarsa lambda is on-policy so this is the same as the
        target policy, namely the greedy action.

        Args:
            state: current state.

        Returns:
            action: greedy action.
        """
        if self._behaviour == constants.GREEDY:
            action = self._greedy_action(
                state=state, excess_state_mapping=excess_state_mapping
            )
        elif self._behaviour == constants.EPSILON_GREEDY:
            action = self._epsilon_greedy_action(
                state=state, epsilon=epsilon, excess_state_mapping=excess_state_mapping
            )
        return action

    def step(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        new_state: Tuple[int, int],
        active: bool,
    ) -> None:
        """Update state-action values.

        Make q-learning update:
        Q(s_t, a_t) <- Q(s_t, a_t) + alpha * [
                            r_{t+1}
                            + gamma * max_a(Q(s_{t+1}, a))
                            - Q(s_t, at_t)
                            ]

        Args:
            state: state before update.
            action: action taken by agent.
            reward: scalar reward received from environment.
            new_state: next state.
            active: whether episode is still ongoing.
        """
        self._state_visitation_counts[state] += 1

        if state == new_state and not self._update_no_op:
            return

        state_id = self._state_id_mapping[state]
        new_state_id = self._state_id_mapping[new_state]

        if active:
            discount = self._gamma
        else:
            discount = 0

        self._step(
            state_id=state_id,
            action=action,
            reward=reward,
            discount=discount,
            new_state_id=new_state_id,
        )

        next(self._learning_rate)

    def _step(
        self,
        state_id,
        action,
        reward,
        discount,
        new_state_id,
    ):
        initial_state_value = self._state_values[state_id]
        state_features = self._state_features[state_id]
        v_target = self._state_values[new_state_id]
        delta = reward + discount * v_target - initial_state_value
        self._weight_matrix += self._learning_rate.value * delta * state_features
        self._wm_change = True
