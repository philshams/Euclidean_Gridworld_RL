from typing import Tuple

import numpy as np


class Node:
    def __init__(self, position: Tuple[int], root: bool = False):
        self._position = position
        self._root = root

    @property
    def root(self):
        return self._root

    @property
    def position(self):
        return self._position

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def cost(self):
        return self._h + self._g

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, h):
        self._h = h

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, g):
        self._g = g

    @property
    def root_here_cost(self):
        raise NotImplementedError


def heuristic(state, reward_states):
    def euclidean(s1, s2):
        return np.sqrt((s1[0] - s2[0]) ** 2 + (s1[1] - s2[1]) ** 2)

    distances = [euclidean(state, r_state) for r_state in reward_states]

    return np.min(distances)


def resolve_path(goal_node):
    path = []
    node = goal_node
    while not node.root:
        path.append(node.position)
        node = node.parent

    path.append(node.position)

    return path[::-1]


def search(transition_matrix, start_state, reward_states):

    open_list = []
    closed_list = []

    nodes = {}

    start_node = Node(start_state, True)
    start_node.g = 0
    start_node.h = heuristic(start_node.position, reward_states)
    nodes[start_state] = start_node
    open_list.append(start_node.position)

    while len(open_list):
        # select node with lowest cost
        node_costs = [nodes[i].cost for i in open_list]
        lowest_cost_index = np.argmin(node_costs)
        selected_node_pos = open_list[lowest_cost_index]
        open_list.remove(selected_node_pos)
        closed_list.append(selected_node_pos)
        selected_node = nodes[selected_node_pos]
        if selected_node.position in reward_states:
            path = resolve_path(selected_node)
            return path
        for adj_node_pos in transition_matrix[selected_node.position]:
            step_cost = np.sqrt(
                (adj_node_pos[0] - selected_node.position[0]) ** 2
                + (adj_node_pos[1] - selected_node.position[1]) ** 2
            )
            adj_node_cost = selected_node.g + step_cost
            if adj_node_pos in nodes:
                adj_node = nodes[adj_node_pos]
            else:
                adj_node = Node(adj_node_pos)
                nodes[adj_node_pos] = adj_node
            if nodes[adj_node_pos].position in open_list:
                if adj_node_cost <= nodes[adj_node_pos].g:
                    open_list.remove(nodes[adj_node_pos].position)
            if nodes[adj_node_pos].position in closed_list:
                if adj_node_cost <= nodes[adj_node_pos].g:
                    closed_list.remove(nodes[adj_node_pos].position)
                    # open_list.append(nodes[adj_node_pos].position)
            if (
                nodes[adj_node_pos].position not in open_list
                and nodes[adj_node_pos].position not in closed_list
            ):
                nodes[adj_node_pos].h = heuristic(
                    nodes[adj_node_pos].position, reward_states
                )
                open_list.append(nodes[adj_node_pos].position)

                nodes[adj_node_pos].g = adj_node_cost
                nodes[adj_node_pos].parent = selected_node

        closed_list.append(selected_node.position)

    # what is the right behaviour if path cannot be found?
    return None
