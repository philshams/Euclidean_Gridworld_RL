import os
import re

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import cm
from rl_nav import constants


def plot_trajectories(folder_path, exp_names):

    cmap = cm.get_cmap("winter")

    def _plot_trajectories(
        exp_path,
        seed_folders,
        env_name,
        env,
        pattern,
        save_path,
        split_by=None,
        gradient=False,
    ):

        path_lengths = []
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

                if gradient:
                    for xi, xi_, yi, yi_ in zip(
                        x[split_index + 1 : -1],
                        x[split_index + 2 :],
                        y[split_index + 1 : -1],
                        y[split_index + 2 :],
                    ):
                        xi_space = np.linspace(xi, xi_, 30)
                        yi_space = [
                            yi + i * (yi_ - yi) / len(xi_space)
                            for i in range(len(xi_space))
                        ]
                        color = [cmap(i / len(xi_space)) for i in range(len(xi_space))]
                        plt.scatter(
                            xi_space,
                            yi_space,
                            c=color,
                            alpha=0.6,
                        )
                else:
                    plt.plot(
                        x[split_index + 1 :],
                        y[split_index + 1 :],
                        color="skyblue",
                        alpha=0.6,
                    )
                    path_lengths.append(len(y[split_index + 1 :]))
            else:
                plt.plot(x, y, color="skyblue", alpha=0.6)
                path_lengths.append(len(y))

        plt.title(
            f"Average Path Length: {round(np.mean(path_lengths), 2)} "
            f"+- {round(np.std(path_lengths), 2)}"
        )

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
            start_position = config[constants.TEST_ENVIRONMENTS][
                constants.START_POSITION
            ]

        envs = {
            f[: -len(f"_{constants.ENV_SKELETON}.npy")]: np.load(
                os.path.join(seed_folders[0], constants.ENV_SKELETON, f)
            )
            for f in os.listdir(os.path.join(seed_folders[0], constants.ENV_SKELETON))
            if f.endswith(".npy")
        }

        for env_name, env in envs.items():

            plain_pattern = re.compile(
                f"{constants.INDIVIDUAL_TEST_RUN}_{env_name}_[0-9]*.npy"
            )

            final_reward_pattern = re.compile(
                f"{constants.INDIVIDUAL_TEST_RUN}_{constants.FINAL_REWARD_RUN}_{env_name}_[0-9]*.npy"
            )

            find_shelter_pattern = re.compile(
                f"{constants.INDIVIDUAL_TEST_RUN}_{constants.FIND_SHELTER_RUN}_{env_name}_[0-9]*.npy"
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

            _plot_trajectories(
                exp_path=exp_path,
                seed_folders=seed_folders,
                env_name=env_name,
                env=env,
                pattern=find_shelter_pattern,
                save_path=os.path.join(
                    exp_path,
                    f"{env_name}_{constants.FIND_SHELTER_RUN}_{constants.TRAJECTORIES}.pdf",
                ),
                split_by=[start_position],
            )


def plot_heatmaps(folder_path, exp_names):
    def _plot_heatmap(save_path, heatmap):

        x, y = zip(*heatmap.keys())

        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)

        array_heatmap = np.zeros((y_max - y_min + 1, x_max - x_min + 1))

        fig = plt.figure()
        for state, value in heatmap.items():
            x_ = state[0]
            y_ = state[1]
            array_heatmap[y_ - y_min][x_ - x_min] = value

        heat_min = np.min(array_heatmap)
        heat_max = np.max(array_heatmap)

        array_heatmap = (array_heatmap - heat_min) / (heat_max - heat_min)

        plt.imshow(array_heatmap, origin="lower", cmap=cm.get_cmap("plasma"))
        plt.colorbar()

        fig.savefig(save_path)

    for exp_name in exp_names:
        exp_path = os.path.join(folder_path, exp_name)
        seed_folders = [
            os.path.join(exp_path, p) for p in os.listdir(exp_path) if p.isdigit()
        ]

        pattern = re.compile(f"[0-9]*_{constants.VALUES}.npy")
        pre_test_pattern = re.compile(f"[0-9]*_{constants.PRE_TEST}_{constants.VALUES}")

        average_heatmap = {}
        average_pre_test_heatmap = {}

        for seed_folder in seed_folders:
            all_heatmaps = [
                os.path.join(seed_folder, constants.VISUALISATIONS, f)
                for f in os.listdir(os.path.join(seed_folder, constants.VISUALISATIONS))
                if pattern.match(f)
            ]
            all_pre_test_heatmaps = [
                os.path.join(seed_folder, constants.VISUALISATIONS, f)
                for f in os.listdir(os.path.join(seed_folder, constants.VISUALISATIONS))
                if pre_test_pattern.match(f)
            ]

            final_heatmap = np.load(
                sorted(
                    all_heatmaps,
                    key=lambda x: int(x.split("_values.npy")[0].split("/")[-1]),
                )[-1],
                allow_pickle=True,
            )[()]
            final_pre_test_heatmap = np.load(
                sorted(
                    all_pre_test_heatmaps,
                    key=lambda x: int(
                        x.split("_pre_test_values.npy")[0].split("/")[-1]
                    ),
                )[-1],
                allow_pickle=True,
            )[()]

            for state, value in final_heatmap.items():
                if state not in average_heatmap:
                    average_heatmap[state] = 0
                average_heatmap[state] += value / len(seed_folders)

            for state, value in final_pre_test_heatmap.items():
                if state not in average_pre_test_heatmap:
                    average_pre_test_heatmap[state] = 0
                average_pre_test_heatmap[state] += value / len(seed_folders)

        _plot_heatmap(
            os.path.join(exp_path, constants.AVERAGE_HEATMAP_PDF), average_heatmap
        )
        _plot_heatmap(
            os.path.join(exp_path, constants.AVERAGE_PRETEST_HEATMAP_PDF),
            average_pre_test_heatmap,
        )
