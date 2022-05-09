from config_manager import config_field, config_template
from rl_nav import constants


def get_template():
    template_class = RLNavConfigTemplate()
    return template_class.base_template


class RLNavConfigTemplate:
    def __init__(self):
        self._initialise()

    def _initialise(self):
        self._training_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.NUM_STEPS,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.MODEL,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [
                            constants.Q_LEARNING,
                            constants.SUCCESSOR_REP,
                            constants.DYNA,
                            constants.UNDIRECTED_DYNA,
                            constants.A_STAR,
                        ]
                    ],
                ),
                config_field.Field(
                    name=constants.BEHAVIOUR,
                    types=[str],
                    requirements=[lambda x: x in [constants.EPSILON_GREEDY]],
                ),
                config_field.Field(
                    name=constants.TARGET,
                    types=[str],
                    requirements=[
                        lambda x: x in [constants.GREEDY, constants.GREEDY_SAMPLE]
                    ],
                ),
                config_field.Field(
                    name=constants.LEARNING_RATE,
                    types=[float, int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.DISCOUNT_FACTOR,
                    types=[float, int],
                    requirements=[lambda x: x > 0 and x <= 1],
                ),
                config_field.Field(
                    name=constants.STEP_COST_FACTOR,
                    types=[float, int],
                    requirements=[lambda x: x >= 0],
                ),
                config_field.Field(
                    name=constants.EPSILON,
                    types=[float, int],
                    requirements=[lambda x: x >= 0 and x <= 1],
                ),
                config_field.Field(name=constants.ONE_DIM_BLOCKS, types=[bool]),
                config_field.Field(
                    name=constants.IMPUTATION_METHOD,
                    types=[str],
                    requirements=[
                        lambda x: x in [constants.NEAR_NEIGHBOURS, constants.RANDOM]
                    ],
                ),
                config_field.Field(
                    name=constants.PLAN_STEPS_PER_UPDATE,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
            ],
            level=[constants.TRAINING],
        )

        self._random_uniform_template = config_template.Template(
            fields=[
                config_field.Field(name=constants.LOWER_BOUND, types=[float, int]),
                config_field.Field(name=constants.UPPER_BOUND, types=[float, int]),
            ],
            level=[constants.INITIALISATION, constants.RANDOM_UNIFORM],
            dependent_variables=[constants.INITIALISATION_TYPE],
            dependent_variables_required_values=[[constants.RANDOM_UNIFORM]],
        )

        self._random_normal_template = config_template.Template(
            fields=[
                config_field.Field(name=constants.MEAN, types=[float, int]),
                config_field.Field(name=constants.VARIANCE, types=[float, int]),
            ],
            level=[constants.INITIALISATION, constants.RANDOM_NORMAL],
            dependent_variables=[constants.INITIALISATION_TYPE],
            dependent_variables_required_values=[[constants.RANDOM_NORMAL]],
        )

        self._initialisation_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.TYPE,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [
                            constants.RANDOM_UNIFORM,
                            constants.RANDOM_NORMAL,
                            constants.ZEROS,
                            constants.ONES,
                        ]
                        or isinstance(x, (int, float))
                    ],
                    key=constants.INITIALISATION_TYPE,
                ),
            ],
            level=[constants.INITIALISATION],
            nested_templates=[
                self._random_uniform_template,
                self._random_normal_template,
            ],
        )

        self._gaussian_statistics_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.MEAN, key=constants.GAUSSIAN_MEAN, types=[float, int]
                ),
                config_field.Field(
                    name=constants.VARIANCE,
                    key=constants.GAUSSIAN_VARIANCE,
                    types=[float, int],
                    requirements=[lambda x: x >= 0],
                ),
            ],
            level=[
                {
                    constants.TRAIN: constants.TRAIN_ENVIRONMENT,
                    constants.TEST: constants.TEST_ENVIRONMENTS,
                },
                constants.REWARD_ATTRIBUTES,
                constants.GAUSSIAN,
            ],
            dependent_variables=[
                [constants.TRAIN_STATISTICS, constants.TEST_STATISTICS]
            ],
            dependent_variables_required_values=[[constants.GAUSSIAN]],
        )

        self._reward_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.AVAILABILITY,
                    types=[int, str],
                    requirements=[
                        lambda x: x == constants.INFINITE
                        or (isinstance(x, int) and x > 0)
                    ],
                ),
                config_field.Field(
                    name=constants.STATISTICS,
                    types=[str],
                    requirements=[lambda x: x in [constants.GAUSSIAN]],
                ),
            ],
            level=[
                {
                    constants.TRAIN: constants.TRAIN_ENVIRONMENT,
                    constants.TEST: constants.TEST_ENVIRONMENTS,
                },
                constants.REWARD_ATTRIBUTES,
            ],
            nested_templates=[self._gaussian_statistics_template],
        )

        self._train_environment_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.ENV_NAME,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [constants.ESCAPE_ENV, constants.ESCAPE_ENV_DIAGONAL]
                    ],
                ),
                config_field.Field(name=constants.MAP_PATH, types=[str]),
                config_field.Field(
                    name=constants.EPISODE_TIMEOUT,
                    types=[int, type(None)],
                    requirements=[lambda x: x is None or x > 0],
                ),
                config_field.Field(
                    name=constants.REPRESENTATION,
                    types=[str],
                    requirements=[
                        lambda x: x in [constants.AGENT_POSITION, constants.PIXEL]
                    ],
                ),
                config_field.Field(
                    name=constants.START_POSITION, types=[type(None), list]
                ),
                config_field.Field(
                    name=constants.REWARD_POSITIONS,
                    types=[list],
                    requirements=[lambda x: all([isinstance(y, list) for y in x])],
                ),
            ],
            level=[constants.TRAIN_ENVIRONMENT],
            nested_templates=[self._reward_template],
            key_prefix=constants.TRAIN,
        )

        self._test_environments_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.ENV_NAME,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [constants.ESCAPE_ENV, constants.ESCAPE_ENV_DIAGONAL]
                    ],
                ),
                config_field.Field(name=constants.MAP_PATHS, types=[list]),
                config_field.Field(
                    name=constants.EPISODE_TIMEOUT,
                    types=[int, type(None)],
                    requirements=[lambda x: x is None or x > 0],
                ),
                config_field.Field(
                    name=constants.REPRESENTATION,
                    types=[str],
                    requirements=[
                        lambda x: x in [constants.AGENT_POSITION, constants.PIXEL]
                    ],
                ),
                config_field.Field(
                    name=constants.START_POSITION, types=[type(None), list]
                ),
                config_field.Field(
                    name=constants.REWARD_POSITIONS,
                    types=[list],
                    requirements=[lambda x: all([isinstance(y, list) for y in x])],
                ),
            ],
            level=[constants.TEST_ENVIRONMENTS],
            nested_templates=[self._reward_template],
            key_prefix=constants.TEST,
        )

        self._logging_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.VISUALISATION_FREQUENCY,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.TEST_FREQUENCY,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.ROLLOUT_FREQUENCY,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.CHECKPOINT_FREQUENCY,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.VISUALISATIONS,
                    types=[list],
                    requirements=[lambda x: all([isinstance(y, str) for y in x])],
                ),
            ],
            level=[constants.LOGGING],
        )

    @property
    def base_template(self):
        return config_template.Template(
            fields=[
                config_field.Field(name=constants.SEED, types=[int]),
            ],
            nested_templates=[
                self._training_template,
                self._initialisation_template,
                self._train_environment_template,
                self._test_environments_template,
                self._logging_template,
            ],
        )
