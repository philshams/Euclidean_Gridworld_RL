import copy
import itertools
from typing import Any, Dict, Tuple

import numpy as np

from rl_nav import constants
from rl_nav.utils import env_utils


def _setup_coarse_coding_features(feature_args):
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

    tiles = []

    for width, height in zip(coding_widths, coding_heights):
        for x in range(x_min, x_max - width + 2):
            for y in range(y_min, y_max - height + 2):
                if augment_actions:
                    for a in action_space:
                        tile = [
                            (x + xi, y + yi, a)
                            for xi, yi in itertools.product(
                                range(width),
                                range(height),
                            )
                        ]
                        tiles.append(tile)
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

    return cc_feature, len(tiles)


def _setup_hard_coded_geometry_features(feature_args):
    tiles = []

    for geometry_path in feature_args[constants.GEOMETRY_OUTLINE_PATHS]:
        geometry = env_utils.parse_map_outline(
            map_file_path=geometry_path,
            mapping=None,
        )[constants.GRID]

        geometry_unzipped_tiles = [
            np.where(geometry == item) for item in set(geometry.flatten())
        ]
        geometry_tiles = [
            list(zip(tile[1], tile[0])) for tile in geometry_unzipped_tiles
        ]

        augment_actions = feature_args[constants.AUGMENT_ACTIONS]
        if augment_actions:
            tiles_ = geometry_tiles
            geometry_tiles = []
            action_space = augment_actions
            for tile in tiles_:
                for a in action_space:
                    augmented_tile = [pos + (a,) for pos in tile]
                    geometry_tiles.append(augmented_tile)

        tiles.extend(geometry_tiles)

    def hard_coded_feature(state: Tuple[int, int, int]):
        if augment_actions:
            features = np.array([state in tile for tile in tiles])
        else:
            features = np.array([state[:2] in tile for tile in tiles])

        return features.astype(int)

    return hard_coded_feature, len(tiles)


def get_feature_extractors(features: Dict[str, Dict[str, Any]]):

    full_feature_dim = 0
    extractors = []

    for feature, feature_args in features.items():

        if feature == constants.STATE_ACTION_ID:

            state_action_id_mapping = feature_args[constants.STATE_ACTION_ID_MAPPING]

            def action_id_feature(state: Tuple[int, int, int]):

                state_action_id = state_action_id_mapping[state]
                one_hot = np.zeros(len(state_action_id_mapping))
                one_hot[state_action_id] = 1

                return one_hot

            full_feature_dim += len(state_action_id_mapping)
            extractors.append(action_id_feature)

        elif feature == constants.STATE_ID:

            state_id_mapping = feature_args[constants.STATE_ID_MAPPING]

            def id_feature(state: Tuple[int, int, int]):

                state_action_id = state_id_mapping[state]
                one_hot = np.zeros(len(state_id_mapping))
                one_hot[state_action_id] = 1

                return one_hot

            full_feature_dim += len(state_id_mapping)
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
            extractor, dim = _setup_coarse_coding_features(feature_args=feature_args)
            full_feature_dim += dim
            extractors.append(extractor)

        elif feature == constants.HARD_CODED_GEOMETRY:
            extractor, dim = _setup_hard_coded_geometry_features(
                feature_args=feature_args
            )
            full_feature_dim += dim
            extractors.append(extractor)

    return extractors, full_feature_dim
