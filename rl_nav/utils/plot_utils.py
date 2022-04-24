import os
import re

import matplotlib.pyplot as plt
import numpy as np
import yaml
from rl_nav import constants


def plot_trajectories(folder_path, exp_names):
    def _plot_trajectories(
        exp_path, seed_folders, env_name, env, pattern, save_path, split_by=None
    ):
        fig = plt.figure()

        plt.imshow(env, origin="lower")

        for seed_folder in seed_folders:

            all_rollouts = [
                os.path.join(seed_folder, constants.ROLLOUTS, f)
                for f in os.listdir(os.path.join(seed_folder, constants.ROLLOUTS))
                if pattern.match(f)
            ]

            final_rollout = sorted(
                all_rollouts, key=lambda x: int(x.split(".npy")[0].split("_")[-1])
            )[-1]

            final_rollout_coords = np.load(final_rollout)
            x, y = zip(*final_rollout_coords)

            if split_by is not None:
                split_indices = []
                for split_pos in split_by:
                    split_index = np.where(
                        np.sum(final_rollout_coords == split_pos, axis=1) == 2
                    )[0][0]
                    split_indices.append(split_index)

                split_index = min(split_indices)
                # plt.plot(
                #     x[: split_index + 1], y[: split_index + 1], color="red", alpha=0.2
                # )
                plt.plot(
                    x[split_index + 1 :],
                    y[split_index + 1 :],
                    color="skyblue",
                    alpha=0.6,
                )
            else:
                plt.plot(x, y, color="skyblue", alpha=0.6)

        fig.savefig(save_path)
        plt.close()

    for exp_name in exp_names:
        exp_path = os.path.join(folder_path, exp_name)
        seed_folders = [
            os.path.join(exp_path, p) for p in os.listdir(exp_path) if p.isdigit()
        ]

        # choose first seed arbitrarily to establish maps, config etc.
        config_path = [f for f in os.listdir(seed_folders[0]) if f.endswith(".yaml")][0]
        with open(os.path.join(seed_folders[0], config_path)) as yaml_file:
            config = yaml.load(yaml_file, Loader=yaml.SafeLoader)
            reward_positions = config[constants.TEST_ENVIRONMENTS][
                constants.REWARD_POSITIONS
            ]

        envs = {
            f[: -len(f"_{constants.ENV_SKELETON}.npy")]: np.load(
                os.path.join(seed_folders[0], constants.ENV_SKELETON, f)
            )
            for f in os.listdir(os.path.join(seed_folders[0], constants.ENV_SKELETON))
        }

        for env_name, env in envs.items():

            plain_pattern = re.compile(
                f"{constants.INDIVIDUAL_TEST_RUN}_{env_name}_[0-9]*.npy"
            )

            final_reward_pattern = re.compile(
                f"{constants.INDIVIDUAL_TEST_RUN}_{constants.FINAL_REWARD_RUN}_{env_name}_[0-9]*.npy"
            )

            _plot_trajectories(
                exp_path=exp_path,
                seed_folders=seed_folders,
                env_name=env_name,
                env=env,
                pattern=plain_pattern,
                save_path=os.path.join(
                    exp_path, f"{env_name}_{constants.TRAJECTORIES}.pdf"
                ),
            )

            _plot_trajectories(
                exp_path=exp_path,
                seed_folders=seed_folders,
                env_name=env_name,
                env=env,
                pattern=final_reward_pattern,
                save_path=os.path.join(
                    exp_path,
                    f"{env_name}_{constants.FINAL_REWARD_RUN}_{constants.TRAJECTORIES}.pdf",
                ),
                split_by=reward_positions,
            )
