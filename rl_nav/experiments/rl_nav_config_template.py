from config_manager import config_field, config_template
from rl_nav import constants


def get_template():
    template_class = RLNavConfigTemplate()
    return template_class.base_template


class RLNavConfigTemplate:
    def __init__(self):
        self._initialise()

    def _initialise(self):

        self._dyna_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.PLAN_STEPS_PER_UPDATE,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
            ],
            dependent_variables=[constants.MODEL],
            dependent_variables_required_values=[
                [
                    constants.DYNA,
                    constants.UNDIRECTED_DYNA,
                    constants.DYNA_LINEAR_FEATURES,
                    constants.UNDIRECTED_DYNA_LINEAR_FEATURES,
                ]
            ],
            level=[constants.TRAINING, constants.DYNA],
        )

        self._coarse_coding_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.CODING_WIDTHS,
                    types=[list],
                    requirements=[
                        lambda x: all([isinstance(y, int) and y > 0 for y in x])
                    ],
                ),
                config_field.Field(
                    name=constants.CODING_HEIGHTS,
                    types=[list],
                    requirements=[
                        lambda x: all([isinstance(y, int) and y > 0 for y in x])
                    ],
                ),
                config_field.Field(name=constants.AUGMENT_ACTIONS, types=[bool]),
            ],
            level=[
                constants.TRAINING,
                constants.LINEAR_FEATURES,
                constants.COARSE_CODING,
            ],
        )

        self._hard_coded_geometry_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.GEOMETRY_OUTLINE_PATHS,
                    types=[list],
                ),
                config_field.Field(name=constants.HC_AUGMENT_ACTIONS, types=[bool]),
            ],
            level=[
                constants.TRAINING,
                constants.LINEAR_FEATURES,
                constants.HARD_CODED_GEOMETRY,
            ],
        )

        self._linear_features_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.FEATURES,
                    types=[list],
                    requirements=[lambda x: all([isinstance(y, str) for y in x])],
                ),
            ],
            level=[constants.TRAINING, constants.LINEAR_FEATURES],
            dependent_variables=[constants.MODEL],
            dependent_variables_required_values=[
                [
                    constants.LINEAR_FEATURES,
                    constants.STATE_LINEAR_FEATURES,
                    constants.DYNA_LINEAR_FEATURES,
                    constants.UNDIRECTED_DYNA_LINEAR_FEATURES,
                ]
            ],
            nested_templates=[
                self._coarse_coding_template,
                self._hard_coded_geometry_template,
            ],
        )

        _constant_lr_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.VALUE,
                    types=[int, float],
                    requirements=[lambda x: x >= 0 and x <= 1],
                )
            ],
            level=[
                constants.TRAINING,
                constants.LEARNING_RATE,
                constants.CONSTANT,
            ],
            dependent_variables=[constants.SCHEDULE],
            dependent_variables_required_values=[[constants.CONSTANT]],
            key_prefix=constants.LEARNING_RATE,
        )

        _linear_decay_lr_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.INITIAL_VALUE,
                    types=[int, float],
                    requirements=[lambda x: x >= 0 and x <= 1],
                ),
                config_field.Field(
                    name=constants.FINAL_VALUE,
                    types=[int, float],
                    requirements=[lambda x: x >= 0 and x <= 1],
                ),
                config_field.Field(
                    name=constants.ANNEAL_DURATION,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
            ],
            level=[
                constants.TRAINING,
                constants.LEARNING_RATE,
                constants.LINEAR_DECAY,
            ],
            dependent_variables=[constants.SCHEDULE],
            dependent_variables_required_values=[
                [constants.LINEAR_DECAY],
            ],
            key_prefix=constants.LEARNING_RATE,
        )

        _hard_coded_lr_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.VALUES,
                    types=[list],
                    requirements=[
                        lambda x: all(
                            [(isinstance(y, float) or isinstance(y, int)) for y in x]
                        )
                    ],
                ),
                config_field.Field(
                    name=constants.TIMESTEP_CHANGES,
                    types=[list],
                    requirements=[lambda x: all([isinstance(y, int) for y in x])],
                ),
            ],
            level=[
                constants.TRAINING,
                constants.LEARNING_RATE,
                constants.HARD_CODED,
            ],
            dependent_variables=[constants.SCHEDULE],
            dependent_variables_required_values=[
                [constants.HARD_CODED],
            ],
        )

        self._learning_rate_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.SCHEDULE,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [
                            constants.CONSTANT,
                            constants.LINEAR_DECAY,
                            constants.HARD_CODED,
                        ]
                    ],
                ),
            ],
            level=[constants.TRAINING, constants.LEARNING_RATE],
            nested_templates=[
                _constant_lr_template,
                _linear_decay_lr_template,
                _hard_coded_lr_template,
            ],
            key_prefix=constants.LEARNING_RATE,
        )

        _constant_epsilon_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.VALUE,
                    types=[int, float],
                    requirements=[lambda x: x >= 0 and x <= 1],
                )
            ],
            level=[
                constants.TRAINING,
                constants.EPSILON,
                constants.CONSTANT,
            ],
            dependent_variables=[constants.SCHEDULE],
            dependent_variables_required_values=[[constants.CONSTANT]],
        )

        _linear_decay_epsilon_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.INITIAL_VALUE,
                    types=[int, float],
                    requirements=[lambda x: x >= 0 and x <= 1],
                ),
                config_field.Field(
                    name=constants.FINAL_VALUE,
                    types=[int, float],
                    requirements=[lambda x: x >= 0 and x <= 1],
                ),
                config_field.Field(
                    name=constants.ANNEAL_DURATION,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
            ],
            level=[
                constants.TRAINING,
                constants.EPSILON,
                constants.LINEAR_DECAY,
            ],
            dependent_variables=[constants.SCHEDULE],
            dependent_variables_required_values=[
                [constants.LINEAR_DECAY],
            ],
        )

        self._epsilon_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.SCHEDULE,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [
                            constants.CONSTANT,
                            constants.LINEAR_DECAY,
                        ]
                    ],
                ),
            ],
            level=[constants.TRAINING, constants.EPSILON],
            nested_templates=[
                _constant_epsilon_template,
                _linear_decay_epsilon_template,
            ],
        )

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
                            constants.DYNA_LINEAR_FEATURES,
                            constants.UNDIRECTED_DYNA_LINEAR_FEATURES,
                            constants.UNDIRECTED_DYNA,
                            constants.A_STAR,
                            constants.LINEAR_FEATURES,
                            constants.STATE_LINEAR_FEATURES,
                            constants.SARSA,
                        ]
                    ],
                ),
                config_field.Field(
                    name=constants.BEHAVIOUR,
                    types=[str],
                    requirements=[
                        lambda x: x in [constants.EPSILON_GREEDY, constants.GREEDY]
                    ],
                ),
                config_field.Field(
                    name=constants.TRAIN_RUN_TRIGGER_STATES,
                    types=[list, type(None)],
                    requirements=[
                        lambda x: x is None or all([isinstance(y, list) for y in x])
                    ],
                ),
                config_field.Field(
                    name=constants.TRAIN_RUN_ACTION_SEQUENCES,
                    types=[list, type(None)],
                    requirements=[
                        lambda x: x is None or all([isinstance(y, list) for y in x])
                    ],
                ),
                config_field.Field(
                    name=constants.TRAIN_RUN_TRIGGER_PROBABILITIES,
                    types=[list, type(None)],
                    requirements=[
                        lambda x: x is None
                        or all(
                            [(isinstance(y, float) and y >= 0 and y <= 1) for y in x]
                        )
                    ],
                ),
                config_field.Field(
                    name=constants.TARGET,
                    types=[str],
                    requirements=[
                        lambda x: x in [constants.GREEDY, constants.GREEDY_SAMPLE]
                    ],
                ),
                config_field.Field(
                    name=constants.TEST_EPSILON,
                    types=[float, int],
                    requirements=[lambda x: x >= 0 and x <= 1],
                ),
                config_field.Field(
                    name=constants.DISCOUNT_FACTOR,
                    types=[float, int],
                    requirements=[lambda x: x > 0 and x <= 1],
                ),
                config_field.Field(
                    name=constants.TRAIN_STEP_COST_FACTOR,
                    types=[float, int],
                    requirements=[lambda x: x >= 0],
                ),
                config_field.Field(name=constants.ONE_DIM_BLOCKS, types=[bool]),
                config_field.Field(
                    name=constants.IMPUTATION_METHOD,
                    types=[str],
                    requirements=[
                        lambda x: x in [constants.NEAR_NEIGHBOURS, constants.RANDOM]
                    ],
                ),
                config_field.Field(name=constants.UPDATE_NO_OP, types=[bool]),
                config_field.Field(
                    name=constants.GRADUAL_LEARNER_WINDOW_AVERAGE, types=[int]
                ),
            ],
            level=[constants.TRAINING],
            nested_templates=[
                self._dyna_template,
                self._linear_features_template,
                self._epsilon_template,
                self._learning_rate_template,
            ],
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

        self._reward_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.AVAILABILITY,
                    types=[list, int, str],
                    requirements=[
                        lambda x: isinstance(x, list)
                        or x == constants.INFINITE
                        or (isinstance(x, int) and x > 0)
                    ],
                ),
                config_field.Field(
                    name=constants.STATISTICS,
                    types=[list],
                ),
            ],
            level=[
                {
                    constants.TRAIN: constants.TRAIN_ENVIRONMENT,
                    constants.TEST: constants.TEST_ENVIRONMENTS,
                },
                constants.REWARD_ATTRIBUTES,
            ],
        )

        self._train_hierarchy_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.TRANSITION_STRUCTURE_PATH, types=[str]
                )
            ],
            level=[
                constants.TRAIN_ENVIRONMENT,
                f"{constants.TRAIN}_{constants.HIERARCHY_NETWORK}",
            ],
            dependent_variables=[[f"{constants.TRAIN}_{constants.ENV_NAME}"]],
            dependent_variables_required_values=[[constants.HIERARCHY_NETWORK]],
        )

        self._train_environment_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.ENV_NAME,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [
                            constants.ESCAPE_ENV,
                            constants.ESCAPE_ENV_DIAGONAL,
                            constants.HIERARCHY_NETWORK,
                            constants.ESCAPE_ENV_DIAGONAL_HIERARCHY,
                        ]
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
                        lambda x: x
                        in [
                            constants.AGENT_POSITION,
                            constants.PIXEL,
                            constants.AGENT_POSITION_REWARD,
                        ]
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
            nested_templates=[self._reward_template, self._train_hierarchy_template],
            key_prefix=constants.TRAIN,
        )

        self._test_hierarchy_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.TRANSITION_STRUCTURE_PATHS, types=[list]
                )
            ],
            level=[
                constants.TEST_ENVIRONMENTS,
                f"{constants.TEST}_{constants.HIERARCHY_NETWORK}",
            ],
            dependent_variables=[[f"{constants.TEST}_{constants.ENV_NAME}"]],
            dependent_variables_required_values=[[constants.HIERARCHY_NETWORK]],
        )

        self._test_environments_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.ENV_NAME,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [
                            constants.ESCAPE_ENV,
                            constants.ESCAPE_ENV_DIAGONAL,
                            constants.HIERARCHY_NETWORK,
                            constants.ESCAPE_ENV_DIAGONAL_HIERARCHY,
                        ]
                    ],
                ),
                config_field.Field(name=constants.MAP_PATHS, types=[list]),
                config_field.Field(
                    name=constants.NUM_TRIALS,
                    types=[int],
                    requirements=[lambda x: x > 1],
                ),
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
                config_field.Field(
                    name=constants.STEP_COST_FACTOR,
                    types=[int, float],
                    requirements=[lambda x: x >= 0],
                ),
            ],
            level=[constants.TEST_ENVIRONMENTS],
            nested_templates=[self._reward_template, self._test_hierarchy_template],
            key_prefix=constants.TEST,
        )

        self._logging_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.PRINT_FREQUENCY,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.VISUALISATION_FREQUENCY,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.TRAIN_TEST_FREQUENCY,
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
                    types=[list, type(None)],
                    requirements=[
                        lambda x: x is None or all([isinstance(y, str) for y in x])
                    ],
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
