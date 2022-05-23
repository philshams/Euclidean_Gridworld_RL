CONFIG_CHANGES = {
    "condition_1_2": [
        {
            "train_environment": {"map_path": "../maps/square_escape_map.txt"},
            "test_environments": {
                "map_paths": [
                    "../maps/square_escape_test_map.txt",
                    "../maps/square_escape_map.txt",
                ]
            },
        }
    ],
    "condition_3": [
        {
            "train_environment": {
                "map_path": "../maps/empty_square_map.txt",
                "reward_positions": [],
            },
            "test_environments": {
                "map_paths": [
                    "../maps/square_escape_test_map.txt",
                    "../maps/square_escape_map.txt",
                ]
            },
        }
    ],
    "condition_4": [
        {
            "train_environment": {
                "map_path": "../maps/vector_blocked_square_escape_map.txt"
            },
            "test_environments": {
                "map_paths": [
                    "../maps/square_escape_test_map.txt",
                    "../maps/square_escape_map.txt",
                ]
            },
        }
    ],
    "condition_5": [
        {
            "train_environment": {
                "map_path": "../maps/shelter_blocked_square_escape_map.txt"
            },
            "test_environments": {
                "map_paths": [
                    "../maps/square_escape_test_map.txt",
                    "../maps/square_escape_map.txt",
                ]
            },
        }
    ],
    "condition_6": [
        {
            "train_environment": {
                "map_path": "../maps/vector_threat_only_blocked_square_escape_map.txt"
            },
            "test_environments": {
                "map_paths": [
                    "../maps/square_escape_test_map.txt",
                    "../maps/square_escape_map.txt",
                ]
            },
        }
    ],
    "condition_7": [
        {
            "train_environment": {
                "map_path": "../maps/square_escape_map.txt",
            },
            "test_environments": {
                "map_paths": [
                    "../maps/moved_reward_map.txt",
                    "../maps/square_escape_map.txt",
                ],
                "reward_positions": [7, 6],
            },
        }
    ],
}
