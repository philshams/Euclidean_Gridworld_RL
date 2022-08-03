import copy
import itertools
import random
from typing import Dict, List, Tuple

import numpy as np
from rl_nav import constants
from rl_nav.models import base_learner
from rl_nav.utils import a_star_search


class AStar(base_learner.BaseLearner):
    """Simple model + A* planning."""

    def __init__(
        self,
        action_space: List[int],
        state_space: List[Tuple[int, int]],
    ):
        """Class constructor.

        Args:
            action_space: list of actions available.
            state_space: list of states.
        """
        super().__init__()

        self._action_space = action_space
        self._state_space = state_space

        self._state_id_mapping = {state: i for i, state in enumerate(self._state_space)}
        self._id_state_mapping = {i: state for i, state in enumerate(self._state_space)}

        self._allow_state_instantiation = False
        self._state_visitation_counts = {s: 0 for s in self._state_space}

        self._model = {}
        self._transition_matrix = {}
        self._action_history = {}
        self._reward_states = []

    def _train(self):
        pass

    def _eval(self):
        pass

    def select_target_action(self):
        pass

    def plan(self, state):
        if len(self._reward_states):
            path = a_star_search.search(
                transition_matrix=self._transition_matrix,
                start_state=state,
                reward_states=self._reward_states,
            )
        else:
            path = None
        if self._allow_state_instantiation:
            import pdb

            pdb.set_trace()
        return path

    def select_behaviour_action(
        self, state: Tuple[int, int], epsilon: float, excess_state_mapping=None
    ) -> Tuple[int, float]:
        """Select action with behaviour policy, i.e. policy collecting trajectory data
        and generating behaviour.

        Args:
            state: current state.

        Returns:
            action: greedy action.
        """
        return random.choice(self._action_space)

    @property
    def deltas(self):
        return self._deltas

    @deltas.setter
    def deltas(self, deltas):
        self._deltas = deltas

    def step(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        new_state: Tuple[int, int],
        active: bool,
    ) -> None:
        """

        Args:
            state: state before update.
            action: action taken by agent.
            reward: scalar reward received from environment.
            new_state: next state.
            active: whether episode is still ongoing.
        """
        if new_state not in self._state_id_mapping and self._allow_state_instantiation:
            self._state_id_mapping[state] = len(self._state_id_mapping)
            self._id_state_mapping[len(self._id_state_mapping)] = state
            self._state_visitation_counts[state] = 0

        if reward > 0 and new_state not in self._reward_states:
            self._reward_states.append(copy.deepcopy(new_state))

        if state not in self._transition_matrix:
            self._transition_matrix[state] = []
        if new_state != state and new_state not in self._transition_matrix[state]:
            self._transition_matrix[state].append(new_state)

        # undirected
        if new_state not in self._transition_matrix:
            self._transition_matrix[new_state]
        if new_state != state and state not in self._transition_matrix[new_state]:
            self._transition_matrix[new_state].append(state)

        # for action_available in self._action_space:
        #     delta = self._deltas[action_available]
        #     next_state = tuple(np.array(state) + delta)
        #     if next_state in self._state_space:
        #         if next_state not in self._transition_matrix[state]:
        #             self._transition_matrix[state].append(next_state)

        if not self._allow_state_instantiation:
            self._state_visitation_counts[state] += 1

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

    @property
    def allow_state_instantiation(self):
        return self._allow_state_instantiation

    @allow_state_instantiation.setter
    def allow_state_instantiation(self, allow: bool):
        self._allow_state_instantiation = allow
