"""Mapping from configuration attributes to values

For a given experiment we may want to compare several
values of a given configuration attribute while keeping
everything else the same. Rather than write many
configuration files we can use the same base for all and
systematically modify it for each different run.
"""

import itertools

CONFIG_CHANGES = {"baseline": []}
# CONFIG_CHANGES = {"rep0": [{"seed": 0}], "rep1": [{"seed": 1}], "rep2": [{"seed": 2}]}
