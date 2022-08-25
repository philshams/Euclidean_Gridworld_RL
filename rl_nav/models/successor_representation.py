import copy
from typing import Dict, List, Tuple, Type

import numpy as np
from rl_nav.models import tabular_learner
from rl_nav.utils import learning_rate_schedules


class SuccessorRepresentation(tabular_learner.TabularLearner):
    """SR (Dayan) with state-action."""

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
        super().__init__(
            action_space=action_space,
            state_space=state_space,
            learning_rate=learning_rate,
            gamma=gamma,
            initialisation_strategy=initialisation_strategy,
            behaviour=behaviour,
            target=target,
            imputation_method=imputation_method,
            update_no_op=update_no_op,
        )

        self._successor_matrix = np.random.normal(
            loc=0,
            scale=0,
            size=(len(action_space), len(state_space), len(state_space)),
        ).astype(dtype="float16")
        self._reward_function = np.zeros(len(state_space)).astype(dtype="float16")

        self._sr_change: bool

        self._compute_state_action_values()

        self._one_hot_matrix = np.eye(len(state_space))

        self._latest_state_action_values = {
            self._id_state_mapping[i]: action_values
            for i, action_values in enumerate(self._state_action_values)
        }

    def _action_policy(self):
        return {
            state: np.argmax(self._state_action_values[state_id, :])
            for state_id, state in self._id_state_mapping.items()
        }

    def get_place_field(self, state):
        policy = self._action_policy()
        successor_matrix = self.successor_matrix
        return {
            pre_state: successor_matrix[state][action][
                self._state_id_mapping[pre_state]
            ]
            for pre_state, action in policy.items()
        }

    @property
    def reward_function(self):
        return {
            state: self._reward_function[state_id]
            for state_id, state in self._id_state_mapping.items()
        }

    @property
    def successor_matrix(self):
        return {
            state: self._successor_matrix[:, state_id, :]
            for state_id, state in self._id_state_mapping.items()
        }

    @property
    def _state_action_values(self):
        if self._sr_change:
            self._compute_state_action_values()
        return self.__state_action_values

    def _compute_state_action_values(self):
        self.__state_action_values = np.matmul(
            self._successor_matrix, self._reward_function
        ).T
        self._sr_change = False

    @property
    def state_action_values(self):
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
        In SR case this is implemented at the level of the successor matrix and
        reward function.

        Args:
            state: new state for which value is being imputed.
            excess_state_mapping: mapping from state to near neighbours.
            store_imputation: whether to compute for single-use or store
                as part of model for future use.
        """
        raise NotImplementedError

    def _impute_randomly(self, state: Tuple[int, int], store_imputation: bool):
        """method to impute values for new state that has no entry in table
        by initialising randomly. In SR case this is implemented at the level
        of the successor matrix and reward function.

        Args:
            state: new state for which value is being imputed.
            store_imputation: whether to compute for single-use or store
                as part of model for future use.

        Returns:
            imputed_value for state.
        """
        if store_imputation:
            self._state_id_mapping[state] = len(self._state_id_mapping)
            self._id_state_mapping[len(self._id_state_mapping)] = state
            successor_matrix_row_entry = np.random.normal(
                size=(len(self._action_space), 1, len(self._reward_function))
            )
            successor_matrix_column_entry = np.random.normal(
                size=(len(self._action_space), len(self._reward_function) + 1, 1)
            )

            # add row
            self._successor_matrix = np.hstack(
                (self._successor_matrix, successor_matrix_row_entry)
            )
            # add column
            self._successor_matrix = np.dstack(
                (self._successor_matrix, successor_matrix_column_entry)
            )
            self._reward_function = np.hstack(
                (self._reward_function, np.random.normal())
            )

            self._state_visitation_counts[state] = 0
            self._one_hot_matrix = np.eye(len(self._reward_function))

            self._sr_change = True

            return self._state_action_values[-1]
        else:
            return np.random.normal(size=len(self._action_space))

    def step(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        new_state: Tuple[int, int],
        active: bool,
    ) -> None:
        """Update state-action values.

        Make SR update via TD:
        M(s_t, s_t+1) <- M(s_t, s_t+1) + alpha * [
                            1I(s_t=s_t+1)
                            + gamma * (M(s_t+1, s')) - M(s_t, s_t+1)
                            - M(s_t, st_t+1)
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

        if new_state not in self._state_id_mapping and self._allow_state_instantiation:
            self._impute_randomly(state=new_state, store_imputation=True)

        new_state_id = self._state_id_mapping[new_state]

        self._step_reward_function(state_id=new_state_id, reward=reward)
        self._step_successor_matrix(
            state_id=state_id,
            action=action,
            new_state_id=new_state_id,
            discount=self._gamma,
            active=active,
        )

        next(self._learning_rate)

    def _step_reward_function(self, state_id: int, reward: float):
        error = reward - self._reward_function[state_id]
        self._reward_function[state_id] += self._learning_rate.value * error

        self._sr_change = True

    def _step_successor_matrix(self, state_id, action, discount, new_state_id, active):
        if active:
            next_action_values = self._state_action_values[new_state_id]
            max_next_action_value = np.max(next_action_values)
            next_action = np.random.choice(
                np.where(next_action_values == max_next_action_value)[0]
            )
            next_action = np.random.choice(range(len(next_action_values)))
            target_ = self._successor_matrix[next_action][new_state_id]
        else:
            target_ = self._one_hot_matrix[new_state_id]

        td_error = (
            self._one_hot_matrix[state_id]
            + discount * target_
            - self._successor_matrix[action][state_id]
        )

        self._successor_matrix[action][state_id] += self._learning_rate.value * td_error

        self._sr_change = True
