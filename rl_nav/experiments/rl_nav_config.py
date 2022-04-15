from typing import Dict
from typing import List
from typing import Union

import yaml
from rl_nav import constants
from rl_nav.experiments.rl_nav_config_template import RLNavConfigTemplate
from config_manager import base_configuration


class RLNavConfig(base_configuration.BaseConfiguration):
    """RLNav Wrapper for base configuration

    Implements a specific validate configuration method for
    non-trivial associations that need checking in config.
    """

    def __init__(self, config: Union[str, Dict], changes: List[Dict] = []) -> None:
        super().__init__(
            configuration=config,
            template=RLNavConfigTemplate.base_template,
            changes=changes,
        )

        self._validate_config()

    def _validate_config(self) -> None:
        """Check for non-trivial associations in config.

        Raises:
            AssertionError: if any rules are broken by config.
        """
        pass

    def _maybe_reconfigure(self, property_name: str) -> None:
        pass
