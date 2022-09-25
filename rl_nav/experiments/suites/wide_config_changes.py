CONFIG_CHANGES = {
    "condition_1_2": [
        {
            "train_environment": {
                "map_path": "../wide_maps/square_escape_map.txt",
                "reward_positions": [[11, 1]],
                "train_hierarchy_network": {
                    "transition_structure_path": "../wide_maps/hierarchy_transition.json"
                },
            },
            "test_environments": {
                "map_paths": [
                    "../wide_maps/square_escape_test_map.txt",
                    "../wide_maps/square_escape_map.txt",
                ],
                "start_position": [11, 12],
                "reward_positions": [[11, 1]],
                "test_hierarchy_network": {
                    "transition_structure_paths": [
                        "../wide_maps/test_hierarchy_transition.json",
                        "../wide_maps/hierarchy_transition.json",
                    ]
                },
            },
        }
    ],
    "condition_3": [
        {
            "train_environment": {
                "map_path": "../wide_maps/empty_square_map.txt",
                "reward_positions": [],
                "train_hierarchy_network": {
                    "transition_structure_path": "../wide_maps/hierarchy_transition.json"
                },
            },
            "test_environments": {
                "map_paths": [
                    "../wide_maps/square_escape_test_map.txt",
                    "../wide_maps/square_escape_map.txt",
                ],
                "start_position": [11, 12],
                "reward_positions": [[11, 1]],
                "test_hierarchy_network": {
                    "transition_structure_paths": [
                        "../wide_maps/test_hierarchy_transition.json",
                        "../wide_maps/hierarchy_transition.json",
                    ]
                },
            },
        }
    ],
    "condition_4": [
        {
            "train_environment": {
                "map_path": "../wide_maps/vector_blocked_square_escape_map.txt",
                "reward_positions": [[11, 1]],
                "train_hierarchy_network": {
                    "transition_structure_path": "../wide_maps/vector_blocked_hierarchy_transition.json"
                },
            },
            "test_environments": {
                "map_paths": [
                    "../wide_maps/square_escape_test_map.txt",
                    "../wide_maps/square_escape_map.txt",
                    "../wide_maps/vector_blocked_square_escape_map.txt",
                ],
                "start_position": [11, 12],
                "reward_positions": [[11, 1]],
                "test_hierarchy_network": {
                    "transition_structure_paths": [
                        "../wide_maps/test_hierarchy_transition.json",
                        "../wide_maps/hierarchy_transition.json",
                        "../wide_maps/vector_blocked_hierarchy_transition.json",
                    ]
                },
            },
        }
    ],
    "condition_5": [
        {
            "train_environment": {
                "map_path": "../wide_maps/shelter_blocked_square_escape_map.txt",
                "reward_positions": [[11, 1]],
                "train_hierarchy_network": {
                    "transition_structure_path": "../wide_maps/shelter_blocked_hierarchy_transition.json"
                },
            },
            "test_environments": {
                "map_paths": [
                    "../wide_maps/square_escape_test_map.txt",
                    "../wide_maps/square_escape_map.txt",
                    "../wide_maps/shelter_blocked_square_escape_map.txt",
                ],
                "start_position": [11, 12],
                "reward_positions": [[11, 1]],
                "test_hierarchy_network": {
                    "transition_structure_paths": [
                        "../wide_maps/test_hierarchy_transition.json",
                        "../wide_maps/hierarchy_transition.json",
                        "../wide_maps/shelter_blocked_hierarchy_transition.json",
                    ]
                },
            },
        }
    ],
    "condition_6": [
        {
            "train_environment": {
                "map_path": "../wide_maps/vector_threat_only_blocked_square_escape_map.txt",
                "reward_positions": [[11, 1]],
                "train_hierarchy_network": {
                    "transition_structure_path": "../wide_maps/vector_threat_only_blocked_hierarchy_transition.json"
                },
            },
            "test_environments": {
                "map_paths": [
                    "../wide_maps/square_escape_test_map.txt",
                    "../wide_maps/square_escape_map.txt",
                    "../wide_maps/vector_threat_only_blocked_square_escape_map.txt",
                ],
                "start_position": [11, 12],
                "reward_positions": [[11, 1]],
                "test_hierarchy_network": {
                    "transition_structure_paths": [
                        "../wide_maps/test_hierarchy_transition.json",
                        "../wide_maps/hierarchy_transition.json",
                        "../wide_maps/vector_threat_only_blocked_hierarchy_transition.json",
                    ]
                },
            },
        }
    ],
    "condition_7": [
        {
            "train_environment": {
                "map_path": "../wide_maps/square_escape_map.txt",
                "reward_positions": [[11, 1]],
                "train_hierarchy_network": {
                    "transition_structure_path": "../wide_maps/hierarchy_transition.json"
                },
            },
            "test_environments": {
                "map_paths": [
                    "../wide_maps/moved_reward_map.txt",
                    "../wide_maps/square_escape_map.txt",
                ],
                "start_position": [11, 12],
                "reward_positions": [[11, 7]],
                "test_hierarchy_network": {
                    "transition_structure_paths": [
                        "../wide_maps/test_hierarchy_transition.json",
                        "../wide_maps/hierarchy_transition.json",
                    ]
                },
            },
        }
    ],
    "condition_8": [
        {
            "train_environment": {
                "map_path": "../wide_maps/blocked_threat_zone_trip_wire_map.txt",
                "reward_positions": [[11, 1]],
            },
            "test_environments": {
                "map_paths": [
                    "../wide_maps/square_escape_test_map.txt",
                    "../wide_maps/square_escape_map.txt",
                ],
                "start_position": [11, 12],
                "reward_positions": [[11, 1]],
            },
        }
    ],
    "condition_9": [
        {
            "train_environment": {
                "map_path": "../wide_maps/blocked_threat_zone_map.txt",
                "reward_positions": [[11, 1]],
            },
            "test_environments": {
                "map_paths": [
                    "../wide_maps/square_escape_test_map.txt",
                    "../wide_maps/square_escape_map.txt",
                ],
                "start_position": [11, 12],
                "reward_positions": [[11, 1]],
            },
        }
    ],
}
