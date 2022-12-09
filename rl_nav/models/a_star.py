import copy
import itertools
import random
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from scipy import stats

from rl_nav import constants
from rl_nav.models import base_learner
from rl_nav.utils.a_star_search import (
    DeterministicMinSearch,
    RandomMinSearch,
    RandomMinWindowSearch,
)


class AStar(base_learner.BaseLearner):
    """Simple model + A* planning."""

    def __init__(
        self,
        action_space: List[int],
        state_space: List[Tuple[int, int]],
        window_average: int,
        node_cost_resolution: Type[namedtuple],
        inverse_actions: Optional[Dict] = None,
    ):
        """Class constructor.

        Args:
            action_space: list of actions available.
            state_space: list of states.
        """
        super().__init__()

        if inverse_actions is not None:
            self._undirected = True
            self._inverse_actions = inverse_actions
        else:
            self._undirected = False
            self._inverse_actions = None

        self._action_space = action_space
        self._state_space = state_space

        self._state_id_mapping = {state: i for i, state in enumerate(self._state_space)}
        self._id_state_mapping = {i: state for i, state in enumerate(self._state_space)}

        self._allow_state_instantiation = False
        self._state_visitation_counts = {s: 0 for s in self._state_space}

        self._window_average = window_average
        self._model = {}
        self._transition_matrix = {}
        self._action_history = {}
        self._reward_states = {}

        if node_cost_resolution.method == constants.RANDOM_MIN:
            self._search_class = RandomMinSearch()
        elif node_cost_resolution.method == constants.DETERMINISTIC_MIN:
            self._search_class = DeterministicMinSearch()
        elif node_cost_resolution.method == constants.RANDOM_MIN_WINDOW:
            self._search_class = RandomMinWindowSearch(
                tolerance=node_cost_resolution.tolerance
            )
        else:
            raise ValueError(
                f"node cost resolution method {node_cost_resolution.method} not recognised."
            )

    def _train(self):
        pass

    def _eval(self):
        pass

    def select_target_action(self):
        pass

    def _process_transition_matrix(self):
        transition_matrix = {}

        for state, action_next_states in self._transition_matrix.items():
            if state not in transition_matrix:
                transition_matrix[state] = []
            for action in action_next_states:
                if action_next_states[action]:
                    modal_new_state = tuple(
                        stats.mode(action_next_states[action][-self._window_average :])[
                            0
                        ][0]
                    )
                    if (
                        modal_new_state not in transition_matrix[state]
                        and state != modal_new_state
                    ):
                        transition_matrix[state].append(modal_new_state)

        return transition_matrix

    def plan(self, state):
        if len(self._reward_states):
            transition_matrix = self._process_transition_matrix()
            path = self._search_class.search(
                transition_matrix=transition_matrix,
                start_state=state,
                reward_states=list(self._reward_states.keys()),
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

        if reward > 0:
            if new_state not in self._reward_states:
                self._reward_states[new_state] = []
            self._reward_states[new_state].append(reward)

        if state not in self._transition_matrix:
            self._transition_matrix[state] = {}
        if action not in self._transition_matrix[state]:
            self._transition_matrix[state][action] = []
        self._transition_matrix[state][action].append(new_state)

        if self._undirected and state != new_state:
            inverse_action = self._inverse_actions[action]
            if new_state not in self._transition_matrix:
                self._transition_matrix[new_state] = {}
            if inverse_action not in self._transition_matrix[new_state]:
                self._transition_matrix[new_state][inverse_action] = []
            self._transition_matrix[new_state][inverse_action].append(state)

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
