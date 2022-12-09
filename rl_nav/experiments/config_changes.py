"""Mapping from configuration attributes to values

For a given experiment we may want to compare several
values of a given configuration attribute while keeping
everything else the same. Rather than write many
configuration files we can use the same base for all and
systematically modify it for each different run.
"""

import itertools

CONFIG_CHANGES = {
    "condition_1": [
        {
            "train_environment": {"map_path": "../circular_maps/obstacle_map.txt"},
            "test_environments": {
                "map_paths": [
                    "../circular_maps/obstacle_map.txt",
                    "../circular_maps/test_map.txt",
                ]
            },
        }
    ],
    "condition_2": [
        {
            "training": {"train_step_cost_factor": 0.0},
            "train_environment": {
                "map_path": "../circular_maps/empty_obstacle_map.txt",
                "reward_positions": [],
            },
            "test_environments": {
                "map_paths": [
                    "../circular_maps/test_map.txt",
                    "../circular_maps/obstacle_map.txt",
                ]
            },
        }
    ],
    "condition_3": [
        {
            "train_environment": {
                "map_path": "../circular_maps/all_edge_blocked_obstacle_map.txt",
            },
            "test_environments": {
                "map_paths": [
                    "../circular_maps/all_edge_blocked_obstacle_map.txt",
                    "../circular_maps/test_map.txt",
                    "../circular_maps/obstacle_map.txt",
                ]
            },
        }
    ],
    "condition_4": [
        {
            "train_environment": {
                "map_path": "../circular_maps/direct_edge_blocked_obstacle_map.txt",
            },
            "test_environments": {
                "map_paths": [
                    "../circular_maps/direct_edge_blocked_obstacle_map.txt",
                    "../circular_maps/test_map.txt",
                    "../circular_maps/obstacle_map.txt",
                ]
            },
        }
    ],
    "condition_5": [
        {
            "train_environment": {
                "map_path": "../maps/shelter_blocked_square_escape_map.txt",
                "train_hierarchy_network": {
                    "transition_structure_path": "../maps/shelter_blocked_hierarchy_transition.json"
                },
            },
            "test_environments": {
                "map_paths": [
                    "../maps/square_escape_test_map.txt",
                    "../maps/square_escape_map.txt",
                    "../maps/shelter_blocked_square_escape_map.txt",
                ],
                "test_hierarchy_network": {
                    "transition_structure_paths": [
                        "../maps/test_hierarchy_transition.json",
                        "../maps/hierarchy_transition.json",
                        "../maps/shelter_blocked_hierarchy_transition.json",
                    ]
                },
            },
        }
    ],
    "condition_6": [
        {
            "train_environment": {
                "map_path": "../maps/vector_threat_only_blocked_square_escape_map.txt",
                "train_hierarchy_network": {
                    "transition_structure_path": "../maps/vector_threat_only_blocked_hierarchy_transition.json"
                },
            },
            "test_environments": {
                "map_paths": [
                    "../maps/square_escape_test_map.txt",
                    "../maps/square_escape_map.txt",
                    "../maps/vector_threat_only_blocked_square_escape_map.txt",
                ],
                "test_hierarchy_network": {
                    "transition_structure_paths": [
                        "../maps/test_hierarchy_transition.json",
                        "../maps/hierarchy_transition.json",
                        "../maps/vector_threat_only_blocked_hierarchy_transition.json",
                    ]
                },
            },
        }
    ],
    "condition_7": [
        {
            "train_environment": {
                "map_path": "../maps/square_escape_map.txt",
                "train_hierarchy_network": {
                    "transition_structure_path": "../maps/hierarchy_transition.json"
                },
            },
            "test_environments": {
                "map_paths": [
                    "../maps/moved_reward_map.txt",
                    "../maps/square_escape_map.txt",
                ],
                "reward_positions": [[7, 6]],
                "test_hierarchy_network": {
                    "transition_structure_paths": [
                        "../maps/test_hierarchy_transition.json",
                        "../maps/hierarchy_transition.json",
                    ]
                },
            },
        }
    ],
    "condition_8": [
        {
            "train_environment": {
                "map_path": "../maps/blocked_threat_zone_trip_wire_map.txt",
            },
            "test_environments": {
                "map_paths": [
                    "../maps/square_escape_test_map.txt",
                    "../maps/square_escape_map.txt",
                ],
            },
        }
    ],
    "condition_9": [
        {
            "train_environment": {
                "map_path": "../maps/blocked_threat_zone_map.txt",
            },
            "test_environments": {
                "map_paths": [
                    "../maps/square_escape_test_map.txt",
                    "../maps/square_escape_map.txt",
                ],
            },
        }
    ],
}
