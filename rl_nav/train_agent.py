import argparse

import yaml

from typing import Dict

from environments import escape_env
from models import q_learning
from runners import episodic_runner
from rl_nav import constants

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config_path",
    metavar="-C",
    default="config.yaml",
    help="path to base configuration file.",
)


def _read_config_from_path(path: str) -> Dict:
    """Read configuration from yaml file path.

    Args:
        path: path to .yaml file.

    Returns:
        configuration: configuration in dictionary format.

    Raises:
        FileNotFoundError if file cannot be found at path specified.
    """
    try:
        with open(path, "r") as yaml_file:
            configuration = yaml.load(yaml_file, yaml.SafeLoader)
    except FileNotFoundError:
        raise FileNotFoundError("Yaml file could not be read.")

    return configuration


def _get_initialisation_strategy(initialisation: Dict):
    initialisation_type = initialisation[constants.TYPE]
    initialisation_spec = initialisation[initialisation_type]
    if initialisation_type == constants.RANDOM_UNIFORM:
        initialisation_strategy = {
            constants.RANDOM_UNIFORM: {
                constants.LOWER_BOUND: initialisation_spec[constants.LOWER_BOUND],
                constants.UPPER_BOUND: initialisation_spec[constants.UPPER_BOUND],
            }
        }
    elif initialisation_type == constants.RANDOM_NORMAL:
        initialisation_strategy = {
            constants.RANDOM_NORMAL: {
                constants.MEAN: initialisation_spec[constants.MEAN],
                constants.VARIANCE: initialisation_spec[constants.VARIANCE],
            }
        }
    else:
        initialisation_strategy == {config.initialisation}
    return initialisation_strategy


if __name__ == "__main__":
    args = parser.parse_args()

    config = _read_config_from_path(path=args.config_path)

    env = escape_env.EscapeEnv(
        map_ascii_path=config[constants.MAP_PATH],
        representation=config[constants.REPRESENTATION],
        reward_positions=config[constants.REWARD_POSITIONS],
        reward_attributes=config[constants.REWARD_ATTRIBUTES],
        start_position=config[constants.START_POSITION],
    )
    model = q_learning.QLearner(
        action_space=env.action_space,
        state_space=env.state_space,
        behaviour=config[constants.BEHAVIOUR],
        target=config[constants.TARGET],
        initialisation_strategy=_get_initialisation_strategy(config[constants.INITIALISATION]),
        learning_rate=config[constants.LEARNING_RATE],
        gamma=config[constants.DISCOUNT_FACTOR]
    )
    runner = episodic_runner.EpisodicRunner(model=model, environment=env)
    runner.train(100)
