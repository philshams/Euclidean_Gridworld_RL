import argparse
import json
import os

from plotter import plot_functions
from rl_nav import constants
from rl_nav.utils import plot_utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--results_folder", type=str, default="results\ql1", help="path to results folder to post-process."
)
parser.add_argument(
    "--plot_trajectories",
    action="store_true",
    help="whether to plot (final) trajectory summaries.",
)
parser.add_argument(
    "--plot_min_trajectories",
    action="store_true",
    default=True,
    help="whether to plot (minimum) trajectory summaries.",
)
parser.add_argument(
    "--plot_heatmaps",
    action="store_true",
    help="whether to plot heatmap averages.",
)
parser.add_argument(
    "--plot_scalars",
    action="store_true",
    help="whether to plot logged scalars.",
)
parser.add_argument(
    "--smoothing", type=int, help="window width for moving averaging.", default=40
)
parser.add_argument(
    "--cmap", type=str, help="matplotlib colormap to use for plots.", default=None
)
parser.add_argument("--linewidth", type=int, help="width of line in plots", default=3)
parser.add_argument(
    "--exclude", type=str, help="list of experiments to ignore.", default=None
)
parser.add_argument(
    "--include",
    type=str,
    help="list of experiments to exclusively process.",
    default=None,
)

if __name__ == "__main__":
    args = parser.parse_args()
    config_changes_json_path = os.path.join(
        args.results_folder, "all_config_changes.json"
    )

    if args.include is not None:
        included_experiments = [
            name.strip() for name in args.include.strip("[").strip("]").split(",")
        ]
        exp_names = included_experiments

        for name in included_experiments:
            assert name in os.listdir(
                args.results_folder
            ), f"name in include: {name} not in experiment names."
    else:
        with open(config_changes_json_path, "r") as f:
            changes = json.load(f)
            exp_names = list(changes.keys())

        if args.exclude is not None:
            excluded_experiments = [
                name.strip() for name in args.exclude.strip("[").strip("]").split(",")
            ]

            exp_names = [name for name in exp_names if name not in excluded_experiments]

    if args.plot_trajectories:
        plot_utils.plot_trajectories(
            folder_path=args.results_folder, exp_names=exp_names, min_rollout=False
        )
    if args.plot_min_trajectories:
        plot_utils.plot_trajectories(
            folder_path=args.results_folder, exp_names=exp_names, min_rollout=True
        )

    if args.plot_heatmaps:
        plot_utils.plot_heatmaps(folder_path=args.results_folder, exp_names=exp_names)

    if args.plot_scalars:

        plot_functions.plot_all_multi_seed_multi_run(
            folder_path=args.results_folder,
            exp_names=exp_names,
            window_width=args.smoothing,
            linewidth=args.linewidth,
            colormap=args.cmap,
            index=constants.STEP,
            averaging_method=constants.INTERPOLATION,
        )

        plot_functions.plot_multi_seed_multi_run(
            folder_path=args.results_folder,
            exp_names=exp_names,
            window_width=args.smoothing,
            linewidth=args.linewidth,
            colormap=args.cmap,
            index=constants.STEP,
            averaging_method=constants.INTERPOLATION,
        )
