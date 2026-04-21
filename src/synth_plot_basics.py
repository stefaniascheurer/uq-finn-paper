from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sympy as sp
import numpy as np
from synth_finn.config import Params

# Constants
FIGURE_WIDTH = 3.0
FIGURE_HEIGHT = 1.875
FONT_SIZE = 14

# Update plot parameters globally
plt.rcParams.update({
    "figure.figsize": [FIGURE_WIDTH, FIGURE_HEIGHT],
    "font.size": FONT_SIZE,
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
})

# Label constants
U_Y_LABEL = r"$u$"
U_X_LABEL = r"$x$"
L_X_LABEL = r"$u$"
if Params().model == "burgers":
    L_Y_LABEL = r"$a$"
elif Params().model == "allen-cahn":
    L_Y_LABEL = r"$R$"


def set_learned_axes_stuff(ax: plt.Axes, xticks: Optional[ticker.Locator] = None, yticks: Optional[ticker.Locator] = None, set_xlabel: bool = False, set_ylabel: bool = False):
    """
    Configure the ticks and labels for learned function-related plots.

    Args:
        ax: The matplotlib Axes object to format.
        xticks: Custom ticker locator for the x-axis. Defaults to FixedLocator at [-1.0, 0.0, 1.0].
        yticks: Custom ticker locator for the y-axis. Defaults to MaxNLocator(3).
        set_xlabel: Whether to apply the predefined learned function x-label.
        set_ylabel: Whether to apply the predefined learned function y-label.
    """
    if xticks is None:
        xticks = ticker.FixedLocator([-1.0, 0.0, 1.0])
    if yticks is None:
        yticks = ticker.MaxNLocator(3)

    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)

    if set_xlabel:
        ax.set_xlabel(L_X_LABEL)
    if set_ylabel:
        ax.set_ylabel(L_Y_LABEL)


def set_statevar_axes_stuff(ax: plt.Axes, mode: str, xticks: Optional[ticker.Locator] = None, yticks: Optional[ticker.Locator] = None, set_xlabel: bool = False, set_ylabel: bool = False):
    """
    Configure ticks and labels for state variable plots.

    Args:
        ax: The matplotlib Axes object to format.
        mode: Either "train" or "test".
        xticks: Custom ticker locator.
        yticks: Custom ticker locator.
        set_xlabel: Whether to apply the context-aware x-label.
        set_ylabel: Whether to apply the context-aware y-label.
    """

    if mode == "train":
        if xticks is None:
            xticks = ticker.FixedLocator([-1.0, 0.0, 1.0])
        if yticks is None:
            yticks = ticker.FixedLocator([-0.5, 0., 0.5])
    elif mode == "test":
        if xticks is None:
            xticks = ticker.FixedLocator([-1.0, 0.0, 1.0])
        if yticks is None:
            yticks = ticker.FixedLocator([-0.25, 0.0, 0.25])

    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)

    if set_xlabel:
        ax.set_xlabel(U_X_LABEL)
    if set_ylabel:
        ax.set_ylabel(U_Y_LABEL)


def solve_ax_dimensions(gap: float) -> tuple[float, float]:
    """Solves for AX_HEIGHT and AX_WIDTH given FIG_HEIGHT, FIG_WIDTH, and gap."""

    ax_height, ax_width = sp.symbols("ax_height, ax_width")

    eq1 = sp.Eq(3 * ax_height + 3 * gap, 1)
    eq2 = sp.Eq(2 * ax_width + 2 * gap, 1)

    solutions = sp.solve((eq1, eq2), (ax_height, ax_width))
    return float(solutions[ax_width]), float(solutions[ax_height])


def get_axes_for_data(**fig_kwargs) -> tuple[plt.Figure, np.ndarray]:
    """"Creates a figure and axes layout for plotting data."""

    FIG_WIDTH = 8 * FIGURE_WIDTH
    FIG_HEIGHT = FIGURE_HEIGHT
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), **fig_kwargs)
    GAP = 0.1

    # Calculate width for three axes side by side
    total_gap = 4 * GAP  # Left margin + 2 gaps between axes + right margin
    available_width = 1.0 - total_gap
    ax_width = available_width / 3

    # Calculate height to maintain aspect ratio
    ax_height = FIG_HEIGHT

    # Center vertically
    vertical_offset = (1.0 - ax_height) / 2

    # rect = [left, bottom, width, height]
    ax1 = fig.add_axes([GAP, vertical_offset, ax_width, ax_height])
    ax2 = fig.add_axes([2 * GAP + ax_width, vertical_offset, ax_width, ax_height])
    ax3 = fig.add_axes([3 * GAP + 2 * ax_width, vertical_offset, ax_width, ax_height])

    axs = np.array([ax1, ax2, ax3])
    return fig, axs


def get_axes_for_samples(**fig_kwargs) -> tuple[plt.Figure, np.ndarray]:
    """"Creates a figure and axes layout for plotting samples."""

    FIG_WIDTH = 6 * FIGURE_WIDTH
    FIG_HEIGHT = FIGURE_HEIGHT
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), **fig_kwargs)
    GAP = 0.05
    AX_WIDTH, AX_HEIGHT = solve_ax_dimensions(GAP)

    # Calculate width for three axes side by side
    total_gap = 3 * GAP  # Left margin + 2 gaps between axes + right margin
    available_width = 1.0 - total_gap
    ax_width = available_width / 2

    # Calculate height to maintain aspect ratio
    ax_height = FIG_HEIGHT

    # Center vertically
    vertical_offset = (1.0 - ax_height) / 2

    # rect = [left, bottom, width, height]
    ax1 = fig.add_axes([GAP, vertical_offset, ax_width, ax_height])
    ax2 = fig.add_axes([2 * GAP + ax_width, vertical_offset, ax_width, ax_height])

    axs = np.array([ax1, ax2])
    return fig, axs


def get_axes_for_comparison(**fig_kwargs) -> tuple[plt.Figure, np.ndarray]:
    FIG_WIDTH = 3 * FIGURE_WIDTH
    FIG_HEIGHT = 4 * FIGURE_HEIGHT
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), **fig_kwargs)
    GAP = 0.1
    AX_WIDTH, AX_HEIGHT = solve_ax_dimensions(GAP)

    # Calculate width for two axes side by side
    total_gap = 3 * GAP  # Left margin + gap between axes + right margin
    available_width = 1.0 - total_gap
    ax_width = available_width / 2

    # Calculate height for three axes stacked vertically
    total_vertical_gap = 3 * GAP  # Top margin + 2 gaps between axes + bottom margin
    available_height = 1.0 - total_vertical_gap
    ax_height = available_height / 3

    # rect = [left, bottom, width, height]
    ax1 = fig.add_axes([GAP, 3 * GAP + 2 * ax_height, ax_width, ax_height])
    ax2 = fig.add_axes([2 * GAP + ax_width, 3 * GAP + 2 * ax_height, ax_width, ax_height])

    ax3 = fig.add_axes([GAP, 2 * GAP + ax_height, ax_width, ax_height])
    ax4 = fig.add_axes([2 * GAP + ax_width, 2 * GAP + ax_height, ax_width, ax_height])

    ax5 = fig.add_axes([GAP, GAP, ax_width, ax_height])
    ax6 = fig.add_axes([2 * GAP + ax_width, GAP, ax_width, ax_height])

    axs = np.array([[ax1, ax2], [ax3, ax4], [ax5, ax6]])
    return fig, axs


def savefig(fig: plt.Figure, path: Union[str, Path], tight: bool = True, **kwargs):
    """
    Save the figure into multiple formats (jpg, svg, pdf) in organized subdirectories.

    Args:
        fig: The matplotlib Figure object to save.
        path: The base path/filename (without suffix).
        tight: Whether to call tight_layout() before saving.
    """
    path = Path(path)
    if tight:
        fig.tight_layout()

    for suffix in ["jpg", "svg", "pdf"]:
        folder = path.parent / suffix
        folder.mkdir(exist_ok=True, parents=True)

        save_path = (folder / path.name).with_suffix(f".{suffix}")
        fig.savefig(save_path, dpi=500, bbox_inches="tight", **kwargs)
