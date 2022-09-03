import os
import re
import matplotlib
matplotlib.set_loglevel("critical")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import cm
from rl_nav import constants
from typing import List, Tuple


def _split_rollout_by_indices(
    rollout_coordinates: List[Tuple], split_start: Tuple, split_end: Tuple
):
    """Take a list of (x, y) coordinates and split them into chunks based on a
    specified splitting coordinate.

    Args:
        rollout_coordinates: List of x, y coordinates
        split_start: Coordinates by which to split set of coordinates (start)
        split_end: Coordinates by which to split set of coordinates (end)

    Returns:
        chunked_rollout: List of lists with split by specified coordinates
    """

    x, y = zip(*rollout_coordinates)

    start_idx_ = np.where(
        np.sum(rollout_coordinates == split_start, axis=1) == 2
    )[0]
    end_idx_ = np.where(
        np.sum(rollout_coordinates == split_end, axis=1) == 2
    )[0]

    start_idx = []
    prev_end_idx = -1
    for end_idx in end_idx_:
        start_idx_trial = start_idx_[
            np.logical_and(start_idx_>prev_end_idx, start_idx_<end_idx)
            ]
        start_idx.append(min(start_idx_trial))
        prev_end_idx = end_idx

    chunked_rollout = []

    for s, e in zip(
        start_idx, end_idx_
    ):
        x_chunk = x[s + 1 : e]
        y_chunk = y[s + 1 : e]
        chunked_rollout.append([x_chunk, y_chunk])

    return chunked_rollout


def plot_trajectories(folder_path, exp_names, min_rollout):

    cmap = cm.get_cmap("winter")

    def _determine_min_trials(seed_folders, pattern):
        """ This will be used to determine how many trials
        are needed at minimum such that all seeds learn edge
        vectors with the obstacle present"""
        first_fast_rollout_idx = []
        fastest_rollout_lens = []
        for seed_folder in seed_folders:
            try:
                all_rollouts = [
                    os.path.join(seed_folder, constants.ROLLOUTS, f)
                    for f in os.listdir(os.path.join(seed_folder, constants.ROLLOUTS))
                    if pattern.match(f)
                ]
            except FileNotFoundError:
                break

            all_rollouts_sorted = sorted(
                    all_rollouts,
                    key=lambda x: int(x.split(".npy")[0].split("_")[-1]),
                )
            all_rollout_coords = [np.load(rollout) for rollout in all_rollouts_sorted]
            all_rollout_lens = [len(rollout) for rollout in all_rollout_coords]
            # first_fast_rollout_idx.append(np.where(np.array(all_rollout_lens)<50)[0][0])
            first_fast_rollout_idx.append(np.argmin(all_rollout_lens))
            fastest_rollout_lens.append(min(all_rollout_lens))
            print(all_rollout_lens)
        first_fast_rollout_idx_all = max(first_fast_rollout_idx)
        num_steps_in_rollout = int(all_rollouts_sorted[
                first_fast_rollout_idx_all
            ].split(".npy")[0].split("_")[-1])
        print(f"\nType of test: {all_rollouts_sorted[0]}"
              + f"\nTrajectory convergence step: {num_steps_in_rollout}"
              + f"\nMax num steps: {max(fastest_rollout_lens)}")
        return num_steps_in_rollout

    def _plot_trajectories(
        seed_folders,
        env,
        pattern,
        save_path,
        split_by=None,
        gradient=False,
        num_training_steps=None,
    ):

        path_lengths = []
        plot_coordinates = {}

        for seed_folder in seed_folders:

            try:
                all_rollouts = [
                    os.path.join(seed_folder, constants.ROLLOUTS, f)
                    for f in os.listdir(os.path.join(seed_folder, constants.ROLLOUTS))
                    if pattern.match(f)
                ]
            except FileNotFoundError:
                break

            if split_by is not None:

                if num_training_steps:
                    num_training_steps_by_rollout = np.array([int(rollout.split(".npy")[0].split("_")[-1]) for rollout in all_rollouts])
                    plot_rollout = all_rollouts[np.where(num_training_steps_by_rollout==num_training_steps)[0][0]]
                else:
                    try:
                        plot_rollout = sorted(
                            all_rollouts,
                            key=lambda x: int(x.split(".npy")[0].split("_")[-1]),
                        )[-1]
                    except IndexError:
                        break

                plot_rollout_coords = _split_rollout_by_indices(
                    np.load(plot_rollout), split_by[0], split_by[1]
                )

                for t, chunk in enumerate(plot_rollout_coords):
                    if t not in plot_coordinates:
                        plot_coordinates[t] = []
                    plot_coordinates[t].append(chunk)

        for t, all_seed_coordinates in plot_coordinates.items():
            if t>5: break

            fig = plt.figure()
            plt.imshow(env, origin="lower")

            if gradient:
                for xi, xi_, yi, yi_ in zip(
                    coordinates[0][:-1],
                    coordinates[0][1:],
                    coordinates[1][:-1],
                    coordinates[1][1:],
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
                        # alpha=0.6,
                    )
            else:
                for coordinates in all_seed_coordinates:
                    x_plot = np.array(coordinates[0])
                    y_plot = np.array(coordinates[1])
                    plt.plot(
                        x_plot,
                        y_plot,
                        color="skyblue",
                        alpha=0.08,
                        zorder=99,
                    )
                    x_diffs = x_plot[1:] - x_plot[:-1]
                    y_diffs = y_plot[1:] - y_plot[:-1]
                    distance = np.sqrt(x_diffs**2 + y_diffs**2).sum()
                    path_lengths.append(distance)

            # gridlines
            for row in range(env.shape[0]):
                plt.plot(
                    [-0.5, env.shape[1] - 0.5],
                    [row - 0.5, row - 0.5],
                    color=[.8,.8,.8],
                    linewidth=0.75,
                    # alpha=0.2,
                )
            for col in range(env.shape[1]):
                plt.plot(
                    [col - 0.5, col - 0.5],
                    [-0.5, env.shape[0] - 0.5],
                    color=[.8,.8,.8],
                    linewidth=0.75,
                    # alpha=0.2,
                )

            plt.title(
                f"Average Path Length: {round(np.mean(path_lengths), 2)} "
                f"+- {round(np.std(path_lengths), 2)}"
            )

            fig.savefig(f"{save_path}_{t}.eps")
            fig.savefig(f"{save_path}_{t}.png")
            plt.close()
        # print(plot_rollout)

    # else:
    #     plt.plot(x, y, color="skyblue", alpha=0.6)
    #     path_lengths.append(len(y))
    num_training_steps = None
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

            find_threat_pattern = re.compile(
                f"{constants.INDIVIDUAL_TEST_RUN}_{constants.FIND_THREAT_RUN}_{env_name}_[0-9]*.npy"
            )

            if min_rollout and exp_name=="condition_1" and env_name=="obstacle_map":
                num_training_steps = _determine_min_trials(
                    seed_folders=seed_folders,
                    pattern=plain_pattern,
                )
            elif not min_rollout and exp_name=="condition_1" and env_name=="obstacle_map":
                num_training_steps = None

            _plot_trajectories(
                seed_folders=seed_folders,
                env=env,
                pattern=plain_pattern,
                save_path=os.path.join(
                    exp_path, f"{env_name}_{constants.TRAJECTORIES}"
                ),
                split_by=[start_position, (0,0)],
                num_training_steps=num_training_steps,
            )

            _plot_trajectories(
                seed_folders=seed_folders,
                env=env,
                pattern=find_threat_pattern,
                save_path=os.path.join(
                    exp_path,
                    f"{env_name}_{constants.FIND_THREAT_RUN}_{constants.TRAJECTORIES}",
                ),
                split_by=[start_position, (0,0)],
                num_training_steps=num_training_steps,
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
