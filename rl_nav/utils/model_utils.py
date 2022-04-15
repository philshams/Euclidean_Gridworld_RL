from rl_nav import constants


def get_initialisation_strategy(config):
    if config.initialisation_type == constants.RANDOM_UNIFORM:
        initialisation_strategy = {
            constants.RANDOM_UNIFORM: {
                constants.LOWER_BOUND: config.lower_bound,
                constants.UPPER_BOUND: config.upper_bound,
            }
        }
    elif config.initialisation_type == constants.RANDOM_NORMAL:
        initialisation_strategy = {
            constants.RANDOM_NORMAL: {
                constants.MEAN: config.mean,
                constants.VARIANCE: config.variance,
            }
        }
    else:
        raise ValueError("Unrecognised initialisation_type")
    return initialisation_strategy