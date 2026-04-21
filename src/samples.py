from pathlib import Path
from typing import Optional
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import colorsys
from scipy.integrate import trapezoid

import plot_basics
from finn.io import load_exp_data_numpy, load_exp_cfg


@dataclasses.dataclass
class Samples:
    """Container for storing and visualizing bootstrap samples and retardation results."""

    core1: np.ndarray
    core2: np.ndarray
    core2b: np.ndarray
    ret_x: np.ndarray
    ret_y: np.ndarray
    quantiles: np.ndarray
    mixed_quantiles: np.ndarray

    @classmethod
    def from_dir(cls, p: str):
        """Loads samples from a directory using the 'y_coreX_samples' naming convention."""

        def try_load(p: Path):
            try:
                return np.load(p)
            except FileNotFoundError:
                return np.array([[]])
        p = Path(p).resolve()
        return cls(
            core1=try_load(p / "y_core1_samples.npy"),
            core2=try_load(p / "y_core2_samples.npy"),
            core2b=try_load(p / "y_core2b_samples.npy"),
            ret_x=try_load(p / "x_ret_samples.npy").squeeze(),
            ret_y=try_load(p / "y_ret_samples.npy"),
            quantiles=try_load(p / "y_core2_quantiles.npy"),
            mixed_quantiles=try_load(p / "y_core2_mixed_quantiles.npy")
        )

    @classmethod
    def from_dir2(cls, p: str):
        """Loads samples from a directory using simple 'coreX' naming convention."""

        def try_load(p: Path):
            try:
                return np.load(p)
            except FileNotFoundError:
                return np.array([[]])
        p = Path(p).resolve()
        return cls(
            core1=try_load(p / "core1.npy"),
            core2=try_load(p / "core2.npy"),
            core2b=try_load(p / "core2b.npy"),
            ret_x=try_load(p / "ret_x.npy").squeeze(),
            ret_y=try_load(p / "ret_y.npy"),
            quantiles=try_load(p / "core2_quantiles.npy"),
            mixed_quantiles=try_load(p / "core2_mixed_quantiles.npy")
        )

    def to_dir(self, p: str):
        """Saves current samples to the specified directory."""

        p = Path(p).resolve()
        np.save(p / "y_core1_samples.npy", self.core1)
        np.save(p / "y_core2_samples.npy", self.core2)
        np.save(p / "y_core2b_samples.npy", self.core2b)
        np.save(p / "x_ret_samples.npy", self.ret_x)
        np.save(p / "y_ret_samples.npy", self.ret_y)

    def plot(self, axs: Optional[list[plt.Axes]] = None, set_titles: bool = True, line_kwargs=None, only_outlines: bool = False, double_colors: bool = False) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plots the experimental cores and retardation factors into a matplotlib grid for given samples.

        Args:
            axs: Optional list of Axes to plot into. If None, a new figure with 2x2 grid is created.
            set_titles: Whether to set titles for each subplot.
            line_kwargs: Additional keyword arguments for line plotting.
            only_outlines: If True, only the min/max outlines of the samples are plotted as filled areas.
        """
        if axs is None:
            fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(2 * plot_basics.FIGURE_WIDTH, 2 * plot_basics.FIGURE_HEIGHT))
            axs = axs.flatten().tolist()
        else:
            fig = plt.gcf()
        assert axs is not None
        line_kwargs = line_kwargs or dict()

        if set_titles:
            axs[0].set_title("Core 1")
            axs[1].set_title("Core 2")
            axs[2].set_title("Core 2B")
            axs[3].set_title("R(c)")

        core1_x = load_exp_data_numpy("Core 1")[0]
        if core1_x.shape[0] != self.core1[0].shape[0]:
            core1_x = np.linspace(core1_x.min(), core1_x.max(), 100, endpoint=True)
        core2_x = load_exp_data_numpy("Core 2")[0]
        core2b_cfg = load_exp_cfg("Core 2B")
        if self.core2b.shape[1] == int(core2b_cfg["Nx"]):
            core2b_x = np.linspace(0, core2b_cfg["X"], int(core2b_cfg["Nx"]))
        else:
            core2b_x = load_exp_data_numpy("Core 2B")[0]

        # line_kwargs.setdefault("color", "C0")

        if double_colors:
            def idxs_of_flats():
                idxs = []
                for i in range(self.ret_y.shape[0]):

                    y0 = self.ret_y[i, 0]
                    idxs.append(1) if 3.5 < y0 < 5.0 else idxs.append(0)
                print(np.sum(idxs), len(idxs))
                return idxs

            base_rgb = np.array(colors.to_rgb("C0"))
            base_hls = colorsys.rgb_to_hls(*base_rgb)
            lightness_values = np.linspace(max(base_hls[1] - 0.3, 0.0), min(1.0, base_hls[1] + 0.3), 2)
            shades = [colors.to_hex(colorsys.hls_to_rgb(base_hls[0], lightness, base_hls[2])) for lightness in lightness_values]
        else:
            line_kwargs.setdefault("linestyle", "-")

        def compute_outlines(arr: np.ndarray):
            return np.array([np.min(arr, axis=0), np.max(arr, axis=0)])

        if only_outlines:
            line_kwargs.setdefault("alpha", 0.5)
            if self.core1.size > 0:
                axs[0].fill_between(core1_x, *compute_outlines(self.core1), **line_kwargs)
            if self.core2.size > 0:
                axs[1].fill_between(core2_x, *compute_outlines(self.core2), **line_kwargs)
            if self.core2b.size > 0:
                axs[2].fill_between(core2b_x, *compute_outlines(self.core2b), **line_kwargs)
            if self.ret_x.size > 0:
                axs[3].fill_between(self.ret_x, *compute_outlines(self.ret_y), **line_kwargs)
        else:
            # line_kwargs["alpha"] = max(1e-2, min(1.0, 6 / self.core2.shape[0]))
            line_kwargs["alpha"] = 0.2
            datasets = [
                (axs[0], core1_x, self.core1),
                (axs[1], core2_x, self.core2),
                (axs[2], core2b_x, self.core2b),
                (axs[3], self.ret_x, self.ret_y),
            ]
            if double_colors:
                idxs = idxs_of_flats()
                for ax, x, data in datasets:
                    if data.size > 0:
                        for i, idx in enumerate(idxs):
                            ax.plot(x, data[i, :], color=shades[1 - idx], **line_kwargs)
            else:
                for ax, x, data in datasets:
                    if data.size > 0:
                        for i in range(data.shape[0]):
                            ax.plot(x, data[i, :], **line_kwargs)

        axs[3].set_xlim(0, 1.5)
        plot_basics.set_retardation_axes_stuff(axs[-1], set_xlabel=True, set_ylabel=True)
        for i, ax in enumerate(axs[:-1]):
            plot_basics.set_concentration_axes_stuff(ax, core="2" if i != 2 else "2B", set_xlabel=True, set_ylabel=True)
        return fig, axs

    def plot_samples(self, axs: Optional[list[plt.Axes]] = None, samples: Optional[set] = None, set_titles: bool = True, line_kwargs=None) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plots the experimental cores and retardation factors into a matplotlib grid.

        Args:
            axs: Optional list of Axes to plot into. If None, a new figure with 2x2 grid is created.
            samples: Optional set of sample indices to plot. If None, all samples are plotted.
            set_titles: Whether to set titles for each subplot.
            line_kwargs: Additional keyword arguments for line plotting.
        """
        if axs is None:
            fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(2 * plot_basics.FIGURE_WIDTH, 2 * plot_basics.FIGURE_HEIGHT))
            axs = axs.flatten().tolist()
        else:
            fig = plt.gcf()
        assert axs is not None
        line_kwargs = line_kwargs or dict()

        if set_titles:
            axs[0].set_title("Bootstrapped\nConcentrations")
            axs[1].set_title("Learned\nRetardation Factors")
            axs[2].set_title("Predicted\nConcentrations")

        core2_x = load_exp_data_numpy("Core 2")[0]

        line_kwargs = {
            "marker": ".",
            "markersize": 4,
            "linestyle": ""
        }

        if samples == None:
            quantiles = self.mixed_quantiles.T
            core2 = self.core2.T
            ret_y = self.ret_y.T

            base_rgb = np.array(colors.to_rgb("C0"))
            base_hls = colorsys.rgb_to_hls(*base_rgb)
            lightness_values = np.linspace(max(base_hls[1] - 0.3, 0.0), min(1.0, base_hls[1] + 0.3), core2.shape[1])
            shades = [colors.to_hex(colorsys.hls_to_rgb(base_hls[0], lightness, base_hls[2])) for lightness in lightness_values]

            if self.core2.size > 0:
                axs[0].plot(core2_x, quantiles, **line_kwargs)
                axs[2].plot(core2_x, core2, **line_kwargs)
            if self.ret_x.size > 0:
                axs[1].plot(self.ret_x, ret_y, **line_kwargs)

        else:
            mixed_quantiles = self.mixed_quantiles[samples].T
            core2 = self.core2[samples].T
            ret_y = self.ret_y[samples].T

            base_rgb = np.array(colors.to_rgb("C0"))
            base_hls = colorsys.rgb_to_hls(*base_rgb)
            lightness_values = np.linspace(max(base_hls[1] - 0.3, 0.0), min(1.0, base_hls[1] + 0.3), core2.shape[1])
            shades = [colors.to_hex(colorsys.hls_to_rgb(base_hls[0], lightness, base_hls[2])) for lightness in lightness_values]

            if self.core2.size > 0:
                [axs[0].plot(core2_x, mixed_quantiles[:, i], color=shades[i], **line_kwargs) for i in range(mixed_quantiles.shape[1])]
                [axs[2].plot(core2_x, core2[:, i], color=shades[1 - i], **line_kwargs) for i in range(core2.shape[1])]

            if self.ret_x.size > 0:
                [axs[1].plot(self.ret_x, ret_y[:, i], color=shades[1 - i], **line_kwargs) for i in range(ret_y.shape[1])]

        for i, ax in enumerate(axs):
            plot_basics.set_concentration_axes_stuff(ax, core="2" if i != 2 else "2B", set_xlabel=i in [0, 1, 2], set_ylabel=i in [0, 1, 2])
        plot_basics.set_retardation_axes_stuff(axs[1], set_xlabel=True, set_ylabel=True)

        fig, [axs[0], axs[2], axs[1]]

    def plot_amount_samples(self, axs: Optional[list[plt.Axes]] = None, amount_samples: list[int] = None, set_titles: bool = True, intervals: float = None, line_kwargs=None) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plots the experimental cores and retardation factors into a matplotlib grid for different given numbers of bootstrap samples.

        Args:
            axs: Optional list of Axes to plot into. If None, a new figure with 2x2 grid is created.
            amount_samples: List of integers specifying how many samples to plot for each dataset. If None, all samples are plotted.
            intervals: If not None, specifies the confidence interval width (e.g., 0.95 for 95% CIs) to plot as filled areas instead of individual samples.
            set_titles: Whether to set titles for each subplot.
            line_kwargs: Additional keyword arguments for line plotting.
        """

        if axs is None:
            fig, axs = plt.subplots(ncols=len(amount_samples), nrows=4, figsize=(len(amount_samples) * plot_basics.FIGURE_WIDTH, 2 * plot_basics.FIGURE_HEIGHT))
            axs = axs.flatten().tolist()
        else:
            fig = plt.gcf()
        assert axs is not None
        line_kwargs = line_kwargs or dict()

        if set_titles:
            for i, amount in enumerate(amount_samples):
                axs[0, i].set_title(r"$N_{boot}$" + f" = {amount}")

        core1_x = load_exp_data_numpy("Core 1")[0]
        if core1_x.shape[0] != self.core1[0].shape[0]:
            core1_x = np.linspace(core1_x.min(), core1_x.max(), 100, endpoint=True)
        core2_x = load_exp_data_numpy("Core 2")[0]
        core2b_cfg = load_exp_cfg("Core 2B")
        if self.core2b.shape[1] == int(core2b_cfg["Nx"]):
            core2b_x = np.linspace(0, core2b_cfg["X"], int(core2b_cfg["Nx"]))
        else:
            core2b_x = load_exp_data_numpy("Core 2B")[0]

        # line_kwargs.setdefault("color", "C0")

        line_kwargs.setdefault("linestyle", "-")

        def compute_outlines(arr: np.ndarray):
            return np.array([np.min(arr, axis=0), np.max(arr, axis=0)])

        datasets = [
            (core2_x, self.core2),
            (self.ret_x, self.ret_y),
            (core1_x, self.core1),
            (core2b_x, self.core2b)
        ]

        for i, amount in enumerate(amount_samples):
            for set, data in enumerate(datasets):
                if data[1].size > 0:
                    subset = data[1][:amount, :]
                    if intervals is not None:
                        line_kwargs.setdefault("alpha", 0.5)
                        # axs[set, i].fill_between(data[0], *compute_outlines(subset), **line_kwargs)
                        axs[set, i].fill_between(data[0], *np.percentile(subset, [int((1 - intervals) / 2 * 100), int((1 + intervals) / 2 * 100)], axis=0), **line_kwargs)
                    else:
                        line_kwargs.setdefault("alpha", 0.2)
                        axs[set, i].plot(data[0], subset.T, color="C0", **line_kwargs)

            plot_basics.set_retardation_axes_stuff(axs[1, i], set_xlabel=True, set_ylabel=True)
            plot_basics.set_concentration_axes_stuff(axs[0, i], core="2", set_xlabel=True, set_ylabel=True)
            plot_basics.set_concentration_axes_stuff(axs[2, i], core="2", set_xlabel=True, set_ylabel=True)
            plot_basics.set_concentration_axes_stuff(axs[3, i], core="2B", set_xlabel=True, set_ylabel=True)

            axs[0, i].set_xlim(0, 40)
            axs[1, i].set_xlim(0, 1.5)
            axs[2, i].set_xlim(0, 39)
            axs[3, i].set_xlim(0.0, 0.105)
            axs[3, i].set_ylim(-0.05, 0.85)

        return fig, axs

    def interval_areas_amount_samples(self, amount_samples: list[int] = None, intervals: float = 0.95, line_kwargs=None) -> tuple[plt.Figure, list[plt.Axes]]:
        fig, ax1 = plt.subplots(figsize=(2 * plot_basics.FIGURE_WIDTH, 2 * plot_basics.FIGURE_HEIGHT))
        ax2 = ax1.twinx()  # Create the secondary axis
        line_kwargs = line_kwargs or dict()

        line_kwargs.setdefault("alpha", 0.8)

        core1_x = load_exp_data_numpy("Core 1")[0]
        if core1_x.shape[0] != self.core1[0].shape[0]:
            core1_x = np.linspace(core1_x.min(), core1_x.max(), 100, endpoint=True)

        core2_x = load_exp_data_numpy("Core 2")[0]
        core2b_cfg = load_exp_cfg("Core 2B")
        core2b_x = np.linspace(0, core2b_cfg["X"], int(core2b_cfg["Nx"])) if self.core2b.shape[1] == int(core2b_cfg["Nx"]) else load_exp_data_numpy("Core 2B")[0]

        primary_datasets = [
            ("Core 2", core2_x, self.core2, "#010B13", "x-"),
            ("Core 1", core1_x, self.core1, "#08306B", "s-"),
            ("Core 2B", core2b_x, self.core2b, "#2171B5", "o-")
        ]
        ret_data = ("R(c)", self.ret_x, self.ret_y, "#5DADE2", "D-")  # Kept the original blue shade

        def calc_areas(x_vals, data_vals):
            areas = []
            for amount in amount_samples:
                subset = data_vals[:amount, :]
                upper = np.percentile(subset, int((1 + intervals) / 2 * 100), axis=0)
                lower = np.percentile(subset, int((1 - intervals) / 2 * 100), axis=0)
                areas.append(trapezoid(upper - lower, x=x_vals))
            return areas

        for name, x, data, color, marker in primary_datasets:
            areas = calc_areas(x, data)
            ax1.plot(amount_samples, areas, marker, color=color, label=name, **line_kwargs)

        ret_name, rx, ry, r_color, r_marker = ret_data
        ret_areas = calc_areas(rx, ry)
        ax2.plot(amount_samples, ret_areas, r_marker, color=r_color, label=ret_name, **line_kwargs)

        ax1.grid(True)
        ax1.set_xlim(0, max(amount_samples) + 1)
        ax1.set_xlabel(r"$N_{boot}$")
        ax1.set_ylabel("Covered CI Area (Cores)", color="#010B13")

        ax2.set_ylabel("Covered CI Area (R(c))", color=r_color)
        ax2.tick_params(axis='y', labelcolor=r_color)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)  # , loc="lower right")

        return fig, [ax1, ax2]
