from typing import Dict, List, Tuple

import numpy as np
from rl_nav.models import tabular_learner


class QLearner(tabular_learner.TabularLearner):
    """Q-learning (Watkins)."""

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
            self._state_id_mapping[new_state] = len(self._state_id_mapping)
            self._id_state_mapping[len(self._id_state_mapping)] = new_state
            imputed_values = self._impute_randomly().reshape(1, len(self._action_space))
            self._state_action_values = np.vstack(
                (self._state_action_values, imputed_values)
            )
            self._state_visitation_counts[new_state] = 0

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
