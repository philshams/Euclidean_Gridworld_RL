import copy
import itertools
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from rl_nav import constants
from rl_nav.models import tabular_learner


class Dyna(tabular_learner.TabularLearner):
    """Dyna (Sutton '91)."""

    MAXSIZE = 500

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
        plan_steps_per_update: int,
        inverse_actions: Optional[Dict] = None,
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
        super().__init__(
            action_space=action_space,
            state_space=state_space,
            learning_rate=learning_rate,
            gamma=gamma,
            initialisation_strategy=initialisation_strategy,
            behaviour=behaviour,
            target=target,
            imputation_method=imputation_method,
        )

        self._plan_steps_per_update = plan_steps_per_update
        if inverse_actions is not None:
            self._undirected = True
            self._inverse_actions = inverse_actions
        else:
            self._undirected = False
            self._inverse_actions = None

        self._state_action_values = self._initialise_values(
            initialisation_strategy=initialisation_strategy
        )
        self._latest_state_action_values = {
            self._id_state_mapping[i]: action_values
            for i, action_values in enumerate(self._state_action_values)
        }

        self._states_visited = []
        self._action_history = {state: [] for state in self._state_space}

        self._model = {
            t[0] + tuple([t[1]]): ()
            for t in itertools.product(self._state_space, self._action_space)
        }

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

    def _impute_near_neighbours(
        self,
        state: Tuple[int, int],
        excess_state_mapping: Dict[Tuple[int, int], List[Tuple[int, int]]],
        store_imputation: bool,
    ):
        """method to impute values for new state that has no entry in table
        using average of values in table that are near neighbours (directly reachable).

        Args:
            state: new state for which value is being imputed.
            excess_state_mapping: mapping from state to near neighbours.
            store_imputation: whether to compute for single-use or store
                as part of model for future use.

        Returns:
            imputed_value for state.
        """
        near_neighbour_ids = [
            self._state_id_mapping[s] for s in excess_state_mapping[state]
        ]
        neighbour_state_action_values = [
            copy.deepcopy(self._state_action_values[s_id])
            for s_id in near_neighbour_ids
        ]
        state_action_values = np.mean(neighbour_state_action_values, axis=0)
        return state_action_values

    def _impute_randomly(self, state: Tuple[int, int], store_imputation: bool):
        """method to impute values for new state that has no entry in table
        by initialising randomly

        Args:
            state: new state for which value is being imputed.
            store_imputation: whether to compute for single-use or store
                as part of model for future use.

        Returns:
            imputed_value for state.
        """
        state_action_values = np.random.normal(size=len(self._action_space))

        if store_imputation:
            self._store_imputation(state=state, imputation=state_action_values)
        return state_action_values

    def _store_imputation(self, state, imputation):
        self._state_id_mapping[state] = len(self._state_id_mapping)
        self._id_state_mapping[len(self._id_state_mapping)] = state

        self._state_action_values = np.vstack(
            (self._state_action_values, imputation.reshape(1, len(self._action_space)))
        )
        self._state_visitation_counts[state] = 0

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

        Then perform planning steps.

        Args:
            state: state before update.
            action: action taken by agent.
            reward: scalar reward received from environment.
            new_state: next state.
            active: whether episode is still ongoing.
        """
        state_id = self._state_id_mapping[state]

        if active:
            discount = self._gamma
        else:
            discount = 0

        self._state_visitation_counts[state] += 1

        self._step(
            state_id=state_id,
            action=action,
            reward=reward,
            discount=discount,
            new_state=new_state,
        )

        if not self._allow_state_instantiation:
            if state not in self._states_visited:
                self._states_visited.append(state)
            if len(self._action_history[state]) == Dyna.MAXSIZE:
                _ = self._action_history[state].pop(0)
            self._action_history[state].append(action)
            state_action = state + tuple([action])
            self._model[state_action] = (new_state, reward)

            if self._undirected:
                if any([isinstance(k, dict) for k in self._inverse_actions.values()]):
                    reverse_action = self._inverse_actions[state][action]
                else:
                    reverse_action = self._inverse_actions[action]
                reverse_state_action = new_state + tuple([reverse_action])
                if reverse_state_action not in self._model:
                    self._model[reverse_state_action] = (state, 0)

            self._plan()

    def _step(
        self,
        state_id,
        action,
        reward,
        discount,
        new_state,
    ):
        initial_state_action_value = self._state_action_values[state_id][action]

        if new_state not in self._state_id_mapping and self._allow_state_instantiation:
            self._impute_randomly(state=new_state, store_imputation=True)

        updated_state_action_value = (
            initial_state_action_value
            + self._learning_rate
            * (
                reward
                + discount * self._max_state_action_value(state=new_state)
                - initial_state_action_value
            )
        )
        self._state_action_values[state_id][action] = updated_state_action_value

    def _plan(self):
        for i in range(self._plan_steps_per_update):
            state = random.choice(self._states_visited)
            action = random.choice(self._action_history[state])
            state_action = state + tuple([action])
            new_state, reward = self._model[state_action]
            state_id = self._state_id_mapping[state]
            self._step(
                state_id=state_id,
                action=action,
                reward=reward,
                discount=self._gamma,
                new_state=new_state,
            )
