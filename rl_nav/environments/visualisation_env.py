from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rl_nav import constants
from rl_nav.environments import wrapper

try:
    import cv2
    import matplotlib
    from matplotlib import cm
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ModuleNotFoundError:
    raise AssertionError(
        "To use visualisation wrapper, further package requirements "
        "need to be satisfied. Please consult README."
    )


class VisualisationEnv(wrapper.Wrapper):

    COLORMAP = cm.get_cmap("plasma")
    NORMALISE = True

    def __init__(self, env):
        super().__init__(env=env)

    def render(
        self,
        save_path: Optional[str] = None,
        dpi: Optional[int] = 60,
        format: str = "state",
    ) -> None:
        """Method to render environment.

        Args:
            save_path: optional path to which to save image.
            dpi: optional pixel.
            format: state of environment to render.
        """
        if format == constants.STATE:
            assert (
                self._env.active
            ), "To render map with state, environment must be active."
            "call reset_environment() to reset environment and make it active."
            "Else render stationary environment skeleton using format='stationary'"
        if save_path:
            fig = plt.figure()
            plt.imshow(
                self._env._env_skeleton(
                    rewards=format,
                    agent=format,
                ),
                origin="lower",
            )
            fig.savefig(save_path, dpi=dpi)
        else:
            plt.imshow(
                self._env._env_skeleton(
                    rewards=format,
                    agent=format,
                ),
                origin="lower",
            )

    def visualise_episode_history(
        self, save_path: str, history: Union[str, List[np.ndarray]] = "train"
    ) -> None:
        """Produce video of episode history.

        Args:
            save_path: name of file to be saved.
            history: "train", "test" to plot train or test history,
            else provide an independent history.
        """
        if isinstance(history, str):
            if history == constants.TRAIN:
                history = self._env.train_episode_history
            elif history == constants.TEST:
                history = self._env.test_episode_history
            elif history == constants.TRAIN_PARTIAL:
                history = self._env.train_episode_partial_history
            elif history == constants.TEST_PARTIAL:
                history = self._env.test_episode_partial_history

        SCALING = 20
        FPS = 30

        map_shape = history[0].shape
        frameSize = (SCALING * map_shape[1], SCALING * map_shape[0])

        out = cv2.VideoWriter(
            filename=save_path,
            fourcc=cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            fps=FPS,
            frameSize=frameSize,
        )

        for frame in history:
            bgr_frame = frame[..., ::-1].copy()
            flipped_frame = np.flip(bgr_frame, 0)
            scaled_up_frame = np.kron(flipped_frame, np.ones((SCALING, SCALING, 1)))
            out.write((scaled_up_frame * 255).astype(np.uint8))

        out.release()

    def plot_heatmap_over_env(
        self,
        heatmap: Dict[Tuple[int, int], float],
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        save_name: Optional[str] = None,
    ) -> None:
        """plot quantities over top of environmen (e.g. value function)

        Args:
            heatmap: data to plot; dictionary of states (keys) and quantities (values).
            fig: figure on which to plot.
            ax: axis on which to plot.
            save_name: path to which to save plot.
        """
        assert (
            ax is not None and fig is not None
        ) or save_name is not None, "Either must provide axis to plot heatmap over,"
        "r file name to save separate figure."
        environment_map = self._env._env_skeleton(
            rewards=None,
            agent=None,
        )

        all_values = list(heatmap.values())
        current_max_value = np.max(all_values)
        current_min_value = np.min(all_values)

        if VisualisationEnv.NORMALISE:
            for position, value in heatmap.items():
                # remove alpha from rgba in colormap return
                # normalise value for color mapping
                environment_map[position[::-1]] = self.COLORMAP(
                    (value - current_min_value)
                    / (current_max_value - current_min_value)
                )[:-1]
        else:
            for position, value in heatmap.items():
                # remove alpha from rgba in colormap return
                environment_map[position[::-1]] = self.COLORMAP(value)[:-1]

        fig = plt.figure()
        if save_name is not None:
            plt.imshow(environment_map, origin="lower", cmap=self.COLORMAP)
            plt.colorbar()
            fig.savefig(save_name, dpi=60)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.imshow(environment_map, origin="lower", cmap=self.COLORMAP)
            fig.colorbar(im, ax=ax, cax=cax, orientation="vertical")
        plt.close()

    def plot_numbered_values_over_env(
        self, values: Dict[Tuple[int], np.ndarray], save_name: str
    ):
        fig = plt.figure()
        environment_map = self._env._env_skeleton(
            rewards=None,
            agent=None,
        )
        plt.imshow(environment_map, origin="lower", cmap=self.COLORMAP)
        all_states = list(values.keys())
        for state, action_values in values.items():
            for i, action_value in enumerate(action_values):
                if all_states[i] != state:
                    xytext = np.array(state) + 0.2 * (
                        np.array(all_states[i]) - np.array(state)
                    )
                    plt.annotate(
                        f"{i}: {round(action_value, 2)}",
                        xy=state,
                        xytext=xytext,
                        arrowprops={
                            "headlength": 2,
                            "headwidth": 2,
                            "width": 0.5,
                            "linewidth": 0.1,
                        },
                        color="y",
                        size=5,
                    )
                else:
                    plt.annotate(
                        i,
                        xy=state,
                        color="g",
                        size=5,
                    )
        fig.savefig(save_name, dpi=60)
        plt.close()
