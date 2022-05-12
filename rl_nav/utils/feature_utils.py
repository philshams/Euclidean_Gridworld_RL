import itertools
from typing import Any, Dict, Tuple

import numpy as np
from rl_nav import constants


def get_feature_extractors(features: Dict[str, Dict[str, Any]]):

    full_feature_dim = 0
    extractors = []

    for feature, feature_args in features.items():

        if feature == constants.STATE_ID:

            state_action_id_mapping = feature_args[constants.STATE_ACTION_ID_MAPPING]

            def id_feature(state: Tuple[int, int, int]):

                state_action_id = state_action_id_mapping[state]
                one_hot = np.zeros(len(state_action_id_mapping))
                one_hot[state_action_id] = 1

                return one_hot

            full_feature_dim += len(state_action_id_mapping)
            extractors.append(id_feature)

        elif feature == constants.ACTION_ONE_HOT:

            action_space = feature_args[constants.ACTION_SPACE]

            def action_feature(state: Tuple[int, int, int]):
                one_hot = np.zeros(len(action_space))
                one_hot[state[-1]] = 1
                return one_hot

            full_feature_dim += len(action_space)
            extractors.append(action_feature)

        elif feature == constants.X_COORDINATE:

            def x_feature(state: Tuple[int, int, int]):
                return state[0]

            full_feature_dim += 1
            extractors.append(x_feature)

        elif feature == constants.Y_COORDINATE:

            def y_feature(state: Tuple[int, int, int]):
                return state[1]

            full_feature_dim += 1
            extractors.append(y_feature)

        elif feature == constants.X_SQUARED:

            def x2_feature(state: Tuple[int, int, int]):
                return state[0] ** 2

            full_feature_dim += 1
            extractors.append(x2_feature)

        elif feature == constants.Y_SQUARED:

            def y2_feature(state: Tuple[int, int, int]):
                return state[1] ** 2

            full_feature_dim += 1
            extractors.append(y2_feature)

        elif feature == constants.XY:

            def xy_feature(state: Tuple[int, int, int]):
                return state[0] * state[1]

            full_feature_dim += 1
            extractors.append(xy_feature)

        elif feature == constants.X2Y:

            def x2y_feature(state: Tuple[int, int, int]):
                return state[0] ** 2 * state[1]

            full_feature_dim += 1
            extractors.append(x2y_feature)

        elif feature == constants.XY2:

            def xy2_feature(state: Tuple[int, int, int]):
                return state[0] * state[1] ** 2

            full_feature_dim += 1
            extractors.append(xy2_feature)

        elif feature == constants.X2Y2:

            def x2y2_feature(state: Tuple[int, int, int]):
                return state[0] ** 2 * state[1] ** 2

            full_feature_dim += 1
            extractors.append(x2y2_feature)

        elif feature == constants.COARSE_CODING:

            coding_widths = feature_args[constants.CODING_WIDTHS]
            coding_heights = feature_args[constants.CODING_HEIGHTS]

            augment_actions = feature_args[constants.AUGMENT_ACTIONS]
            if augment_actions:
                action_space = augment_actions

            state_space = feature_args[constants.STATE_SPACE]
            x_, y_ = zip(*state_space)

            # assume square map
            x_min = min(x_)
            x_max = max(x_)
            y_min = min(y_)
            y_max = max(y_)

            square_width = x_max - x_min
            square_height = y_max - y_min

            tiles = []

            for width, height in zip(coding_widths, coding_heights):
                for x in range(x_min, x_max - width + 2):
                    for y in range(y_min, y_max - height + 2):
                        if augment_actions:
                            tile = [
                                (x + xi, y + yi, a)
                                for a, xi, yi in itertools.product(
                                    action_space,
                                    range(width),
                                    range(height),
                                )
                            ]
                        else:
                            tile = [
                                (x + xi, y + yi)
                                for xi, yi in itertools.product(
                                    range(width),
                                    range(height),
                                )
                            ]
                        tiles.append(tile)

            def cc_feature(state: Tuple[int, int, int]):
                if augment_actions:
                    coding_features = np.array([state in tile for tile in tiles])
                else:
                    coding_features = np.array([state[:2] in tile for tile in tiles])

                return coding_features.astype(int)

            full_feature_dim += len(tiles)
            extractors.append(cc_feature)

    return extractors, full_feature_dim
