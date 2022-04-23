import abc
import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from rl_nav import constants
from rl_nav.models import base_learner


class TabularLearner(base_learner.BaseLearner):
    """Base class for learners in tabular settings."""

    def __init__(
        self,
        action_space: List[int],
        state_space: List[Tuple[int, int]],
        learning_rate: float,
        gamma: float,
        initialisation_strategy: Dict,
        behaviour: str,
        target: str,
        imputation_method: str,
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
            epsilon: exploration parameter.
        """
        self._action_space = action_space
        self._state_space = state_space

        self._state_id_mapping = {state: i for i, state in enumerate(self._state_space)}
        self._id_state_mapping = {i: state for i, state in enumerate(self._state_space)}

        self._state_action_values = self._initialise_values(
            initialisation_strategy=initialisation_strategy
        )
        self._latest_state_action_values = {
            self._id_state_mapping[i]: action_values
            for i, action_values in enumerate(self._state_action_values)
        }

        self._state_visitation_counts = {s: 0 for s in self._state_space}

        self._behaviour = behaviour
        self._target = target
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._training = True

        self._imputation_method = imputation_method
        self._allow_state_instantiation = False

    def train(self):
        self._training = True

    def eval(self):
        self._training = False
        self._latest_state_action_values = {
            self._id_state_mapping[i]: action_values
            for i, action_values in enumerate(self._state_action_values)
        }

    @property
    def action_space(self) -> List[int]:
        return self._action_space

    @property
    def state_id_mapping(self) -> Dict:
        return self._state_id_mapping

    @property
    def id_state_mapping(self) -> Dict:
        return self._id_state_mapping

    @property
    def state_visitation_counts(self) -> Dict[Tuple[int, int], int]:
        return self._state_visitation_counts

    @property
    def state_action_values(self) -> Dict[Tuple[int, int], np.ndarray]:
        if self._training:
            values = {
                self._id_state_mapping[i]: action_values
                for i, action_values in enumerate(self._state_action_values)
            }
            return values
        else:
            return self._latest_state_action_values

    def _impute_values(self, state, excess_state_mapping):
        if self._imputation_method == constants.NEAR_NEIGHBOURS:
            return self._impute_near_neighbours()
        elif self._imputation_method == constants.RANDOM:
            return self._impute_randomly()

    def _impute_near_neighbours(self, state, excess_state_mapping):
        near_neighbour_ids = [
            self._state_id_mapping[s] for s in excess_state_mapping[state]
        ]
        neighbour_state_action_values = [
            copy.deepcopy(self._state_action_values[s_id])
            for s_id in near_neighbour_ids
        ]
        state_action_values = np.mean(neighbour_state_action_values, axis=0)
        return state_action_values

    def _impute_randomly(self):
        state_action_values = np.random.normal(size=len(self._action_space))
        return state_action_values

    def _initialise_values(self, initialisation_strategy: str) -> np.ndarray:
        """Initialise values for each state, action pair in state-action space.

        Args:
            initialisation_strategy: name of method used to initialise.

        Returns:
            initial_values: matrix containing state-action id / value mapping.
        """
        initialisation_strategy_name = list(initialisation_strategy.keys())[0]
        if isinstance(initialisation_strategy_name, (int, float)):
            return initialisation_strategy_name * np.ones(
                (len(self._state_space), len(self._action_space))
            )
        elif initialisation_strategy_name == constants.RANDOM_UNIFORM:
            return np.random.rand(len(self._state_space), len(self._action_space))
        elif initialisation_strategy_name == constants.RANDOM_NORMAL:
            return np.random.normal(
                loc=0, scale=0.1, size=(len(self._state_space), len(self._action_space))
            )
        elif initialisation_strategy_name == constants.RANDOM_NORMAL:
            return np.random.normal(
                loc=0, scale=0.1, size=(len(self._state_space), len(self._action_space))
            )
        elif initialisation_strategy_name == constants.ZEROS:
            return np.zeros((len(self._state_space), len(self._action_space)))
        elif initialisation_strategy_name == constants.ONES:
            return np.ones((len(self._state_space), len(self._action_space)))

    def _max_state_action_value(
        self, state: Tuple[int, int], other_state_action_values: Optional[Dict] = None
    ) -> float:
        """Find highest value in given state.

        Args:
            state: state for which to find highest value.

        Returns:
            value: corresponding highest value.
        """
        state_id = self._state_id_mapping[state]

        if other_state_action_values is not None:
            state_action_values = other_state_action_values[state_id]
        else:
            state_action_values = self._state_action_values[state_id]

        return np.amax(state_action_values)

    def _greedy_action(
        self, state: Tuple[int, int], excess_state_mapping: Optional[Dict] = None
    ) -> int:
        """Find action with highest value in given state.

        Args:
            state: state for which to find action with highest value.

        Returns:
            action: action with highest value in state given.
        """

        state_id = self._state_id_mapping.get(state)
        if state_id is not None:
            state_action_values = copy.deepcopy(self._state_action_values[state_id])
        else:
            state_action_values = self._impute_values(
                state=state, excess_state_mapping=excess_state_mapping
            )

        return np.argmax(state_action_values)

    def _non_repeat_greedy_action(
        self, state: Tuple[int, int], excluded_actions: List[int]
    ) -> int:
        """Find action with highest value in given state not included set of excluded actions.

        Args:
            state: state for which to find action with (modified) highest value.
            excluded_actions: set of actions to exclude from consideration.

        Returns:
            action: action with (modified) highest value in state given.
        """
        state_id = self._state_id_mapping[state]
        actions_available = [
            action if i not in excluded_actions else -np.inf
            for (i, action) in enumerate(self._state_action_values[state_id])
        ]
        return np.argmax(actions_available)

    def _greedy_sample_action(
        self, state: Tuple[int, int], excess_state_mapping: Optional[Dict] = None
    ):
        state_id = self._state_id_mapping.get(state)
        if state_id is not None:
            state_action_values = copy.deepcopy(self._state_action_values[state_id])
        else:
            state_action_values = self._impute_values(
                state=state, excess_state_mapping=excess_state_mapping
            )

        exp_values = np.exp(state_action_values)
        softmax_values = exp_values / np.sum(exp_values)

        return np.random.choice(self._action_space, p=softmax_values)

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
                state=state, excess_state_mapping=excess_state_mapping
            )
        return action

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

    @abc.abstractmethod
    def step(self, *args, **kwargs) -> None:
        """Update relevant data for learner."""
        pass

    @property
    def allow_state_instantiation(self):
        return self._allow_state_instantiation

    @allow_state_instantiation.setter
    def allow_state_instantiation(self, allow: bool):
        self._allow_state_instantiation = allow
