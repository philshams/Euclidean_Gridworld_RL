CONFIG_CHANGES = {
    "condition_1_2": [
        {
            "train_environment": {
                "map_path": "../maps/square_escape_map.txt",
                "episode_timeout": None,
                "reward_attributes": {"availability": ["infinite"]},
                "train_hierarchy_network": {
                    "transition_structure_path": "../maps/hierarchy_transition.json"
                },
            },
            "test_environments": {
                "map_paths": [
                    "../maps/square_escape_test_map.txt",
                    "../maps/square_escape_map.txt",
                ],
                "test_hierarchy_network": {
                    "transition_structure_paths": [
                        "../maps/test_hierarchy_transition.json",
                        "../maps/hierarchy_transition.json",
                    ]
                },
            },
        }
    ],
    "condition_3": [
        {
            "train_environment": {
                "map_path": "../maps/empty_square_map.txt",
                "episode_timeout": None,
                "reward_attributes": {"availability": ["infinite"]},
                "reward_positions": [],
                "train_hierarchy_network": {
                    "transition_structure_path": "../maps/hierarchy_transition.json"
                },
            },
            "test_environments": {
                "map_paths": [
                    "../maps/square_escape_test_map.txt",
                    "../maps/square_escape_map.txt",
                ],
                "test_hierarchy_network": {
                    "transition_structure_paths": [
                        "../maps/test_hierarchy_transition.json",
                        "../maps/hierarchy_transition.json",
                    ]
                },
            },
        }
    ],
    "condition_4": [
        {
            "train_environment": {
                "map_path": "../maps/vector_blocked_square_escape_map.txt",
                "episode_timeout": None,
                "reward_attributes": {"availability": ["infinite"]},
                "train_hierarchy_network": {
                    "transition_structure_path": "../maps/vector_blocked_hierarchy_transition.json"
                },
            },
            "test_environments": {
                "map_paths": [
                    "../maps/square_escape_test_map.txt",
                    "../maps/square_escape_map.txt",
                    "../maps/vector_blocked_square_escape_map.txt",
                ],
                "test_hierarchy_network": {
                    "transition_structure_paths": [
                        "../maps/test_hierarchy_transition.json",
                        "../maps/hierarchy_transition.json",
                        "../maps/vector_blocked_hierarchy_transition.json",
                    ]
                },
            },
        }
    ],
    "condition_5": [
        {
            "train_environment": {
                "map_path": "../maps/shelter_blocked_square_escape_map.txt",
                "episode_timeout": None,
                "reward_attributes": {"availability": ["infinite"]},
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
                "episode_timeout": None,
                "reward_attributes": {"availability": ["infinite"]},
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
                "episode_timeout": None,
                "reward_attributes": {"availability": ["infinite"]},
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
                "episode_timeout": None,
                "reward_attributes": {"availability": ["infinite"]},
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
                "episode_timeout": None,
                "reward_attributes": {"availability": ["infinite"]},
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
