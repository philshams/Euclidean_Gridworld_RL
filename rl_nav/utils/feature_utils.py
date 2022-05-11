from typing import List, Tuple

import numpy as np
from rl_nav import constants


def setup_feature_extractors(features: List[str], learner_class):

    full_feature_dim = 0
    extractors = []

    state_action_id_mapping = learner_class.state_action_id_mapping

    for feature in features:

        if feature == constants.STATE_ID:

            def id_feature(state: Tuple[int, int]):

                state_action_id = state_action_id_mapping[state]
                one_hot = np.zeros(len(state_action_id_mapping))
                one_hot[state_action_id] = 1

                return one_hot

            full_feature_dim += len(state_action_id_mapping)
            extractors.append(id_feature)

        if feature == constants.X_COORDINATE:

            def x_feature(state: Tuple[int, int]):
                return state[0]

            full_feature_dim += 1
            extractors.append(x_feature)

        elif feature == constants.Y_COORDINATE:

            def y_feature(state: Tuple[int, int]):
                return state[1]

            full_feature_dim += 1
            extractors.append(y_feature)

        elif feature == constants.X_SQUARED:

            def x2_feature(state: Tuple[int, int]):
                return state[0] ** 2

            full_feature_dim += 1
            extractors.append(x2_feature)

        elif feature == constants.Y_SQUARED:

            def y2_feature(state: Tuple[int, int]):
                return state[1] ** 2

            full_feature_dim += 1
            extractors.append(y2_feature)

        elif feature == constants.XY:

            def xy_feature(state: Tuple[int, int]):
                return state[0] * state[1]

            full_feature_dim += 1
            extractors.append(xy_feature)

        elif feature == constants.X2Y:

            def x2y_feature(state: Tuple[int, int]):
                return state[0] ** 2 * state[1]

            full_feature_dim += 1
            extractors.append(x2y_feature)

        elif feature == constants.XY2:

            def xy2_feature(state: Tuple[int, int]):
                return state[0] * state[1] ** 2

            full_feature_dim += 1
            extractors.append(xy2_feature)

        elif feature == constants.X2Y2:

            def x2y2_feature(state: Tuple[int, int]):
                return state[0] ** 2 * state[1] ** 2

            full_feature_dim += 1
            extractors.append(x2y2_feature)

    return extractors, full_feature_dim
