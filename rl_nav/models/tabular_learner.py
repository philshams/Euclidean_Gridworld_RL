import abc
import copy
import random
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from rl_nav import constants
from rl_nav.models import base_learner
from rl_nav.utils import learning_rate_schedules


class TabularLearner(base_learner.BaseLearner):
    """Base class for learners in tabular settings."""

    def __init__(
        self,
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

    def _train(self):
        """set model to train mode."""
        pass

    def _eval(self):
        """set model to evaluation mode."""
        self._latest_state_action_values = {
            self._id_state_mapping[i]: action_values
            for i, action_values in enumerate(self._state_action_values)
        }

    @property
    def action_space(self) -> List[int]:
        """actions available to agent."""
        return self._action_space

    @property
    def state_id_mapping(self) -> Dict:
        """one way mapping from states to indices for states;
        inverse mapping of _id_state_mapping."""
        return self._state_id_mapping

    @property
    def id_state_mapping(self) -> Dict:
        """one way mapping from indices for states to states;
        inverse mapping of _state_id_mapping."""
        return self._id_state_mapping

    @property
    def state_visitation_counts(self) -> Dict[Tuple[int, int], int]:
        """number of times each state has been visited."""
        return self._state_visitation_counts

    def _impute_values(
        self,
        state: Tuple[int, int],
        excess_state_mapping: Dict[Tuple[int, int], List[Tuple[int, int]]],
        store_imputation: bool,
    ) -> float:
        """method to impute values for new state that has no entry in table.

        Args:
            state: new state for which value is being imputed.
            excess_state_mapping: mapping from state to near neighbours.
            store_imputation: whether to compute for single-use or store
            as part of model for future use.

        Returns:
            imputed_value for state.
        """
        if self._imputation_method == constants.NEAR_NEIGHBOURS:
            return self._impute_near_neighbours(
                state=state,
                excess_state_mapping=excess_state_mapping,
                store_imputation=store_imputation,
            )
        elif self._imputation_method == constants.RANDOM:
            return self._impute_randomly(state=state, store_imputation=store_imputation)

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
            excess_state_mapping: mapping from state to near neighbours.

        Returns:
            action: action with highest value in state given.
        """

        state_id = self._state_id_mapping.get(state)
        if state_id is not None:
            state_action_values = copy.deepcopy(self._state_action_values[state_id])
        else:
            state_action_values = self._impute_values(
                state=state,
                excess_state_mapping=excess_state_mapping,
                store_imputation=False,
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
        """take action by sampling from distribution of values.

        Args:
            state: state for which to find action.
            excess_state_mapping: mapping from state to near neighbours.
        """
        state_id = self._state_id_mapping.get(state)
        if state_id is not None:
            state_action_values = copy.deepcopy(self._state_action_values[state_id])
        else:
            state_action_values = self._impute_values(
                state=state,
                excess_state_mapping=excess_state_mapping,
                store_imputation=False,
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
                state=state,
                excess_state_mapping=excess_state_mapping,
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
