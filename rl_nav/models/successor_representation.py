from typing import Dict, List, Tuple

from rl_nav.models import tabular_learner


class SuccessorRepresetation(tabular_learner.TabularLearner):
    """SR (Dayan)."""

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
        raise NotImplementedError

    def _step(
        self,
        state_id,
        action,
        reward,
        discount,
        new_state,
    ):
        raise NotImplementedError
