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
    elif config.initialisation_type == constants.ZEROS:
        initialisation_strategy = {constants.ZEROS: {}}
    elif config.initialisation_type == constants.ONES:
        initialisation_strategy = {constants.ONES: {}}
    else:
        raise ValueError("Unrecognised initialisation_type")
    return initialisation_strategy
