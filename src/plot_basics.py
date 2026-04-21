from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sympy as sp
import numpy as np

# Constants
FIGURE_WIDTH = 3.0
FIGURE_HEIGHT = 1.875
FONT_SIZE = 11

# Update plot parameters globally
plt.rcParams.update({
    "figure.figsize": [FIGURE_WIDTH, FIGURE_HEIGHT],
    "font.size": FONT_SIZE,
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
})

# Label constants
C_DISS_Y_LABEL = r"$c$ [$mg/L$]"
C_TOT_Y_LABEL = r"$c_t$ [$mg/L$]"
C_DISS_X_LABEL = r"$t$ [$\text{days}$]"
C_TOT_X_LABEL = r"$x$ [$m$]"
R_X_LABEL = r"$c$ [$mg/L$]"
R_Y_LABEL = r"$R$ [-]"


def set_retardation_axes_stuff(ax: plt.Axes, xticks: Optional[ticker.Locator] = None, yticks: Optional[ticker.Locator] = None, set_xlabel: bool = False, set_ylabel: bool = False):
    """
    Configure the ticks and labels for retardation-related plots.

    Args:
        ax: The matplotlib Axes object to format.
        xticks: Custom ticker locator for the x-axis. Defaults to FixedLocator at [0, 0.5, 1.0, 1.5].
        yticks: Custom ticker locator for the y-axis. Defaults to MaxNLocator(3).
        set_xlabel: Whether to apply the predefined retardation x-label.
        set_ylabel: Whether to apply the predefined retardation y-label.
    """
    if xticks is None:
        xticks = ticker.FixedLocator([0.0, 0.5, 1.0, 1.5])
    if yticks is None:
        yticks = ticker.MaxNLocator(3)

    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)

    if set_xlabel:
        ax.set_xlabel(R_X_LABEL)
    if set_ylabel:
        ax.set_ylabel(R_Y_LABEL)


def set_concentration_axes_stuff(ax: plt.Axes, xticks: Optional[ticker.Locator] = None, yticks: Optional[ticker.Locator] = None, core: str = "2", set_xlabel: bool = False, set_ylabel: bool = False):
    """
    Configure ticks and labels for concentration plots based on the core type.

    Args:
        ax: The matplotlib Axes object to format.
        xticks: Custom ticker locator. Logic varies if core is '2B'.
        yticks: Custom ticker locator. Logic varies if core is '2B'.
        core: The core identifier string. Uses "2B" logic for depth/total concentration.
        set_xlabel: Whether to apply the context-aware x-label.
        set_ylabel: Whether to apply the context-aware y-label.
    """

    if xticks is None:
        if core == "2B":
            xticks = (ticker.FixedLocator([0, 0.05, 0.1]))
        else:
            xticks = (ticker.FixedLocator([0, 20, 40]))
    if yticks is None:
        if core == "2B":
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            yticks = (ticker.FixedLocator([0.0, 0.3, 0.6]))
        else:
            yticks = (ticker.FixedLocator([0, 0.0025, 0.005]))

    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)

    if set_xlabel:
        ax.set_xlabel(C_DISS_X_LABEL if core != "2B" else C_TOT_X_LABEL)
    if set_ylabel:
        ax.set_ylabel(C_DISS_Y_LABEL if core != "2B" else C_TOT_Y_LABEL)


def solve_ax_dimensions(gap: float) -> tuple[float, float]:
    """Solves for AX_HEIGHT and AX_WIDTH given FIG_HEIGHT, FIG_WIDTH, and gap."""

    ax_height, ax_width = sp.symbols("ax_height, ax_width")

    eq1 = sp.Eq(3 * ax_height + 3 * gap, 1)
    eq2 = sp.Eq(2 * ax_width + 2 * gap, 1)

    solutions = sp.solve((eq1, eq2), (ax_height, ax_width))
    return float(solutions[ax_width]), float(solutions[ax_height])


def get_axes_for_results(**fig_kwargs) -> tuple[plt.Figure, np.ndarray]:
    """"Creates a figure and axes layout for plotting results."""

    FIG_WIDTH = 2 * FIGURE_WIDTH
    FIG_HEIGHT = 3 * FIGURE_HEIGHT
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), **fig_kwargs)

    GAP = 0.15
    AX_WIDTH, AX_HEIGHT = solve_ax_dimensions(GAP)

    # rect = [left, bottom, width, height]
    ax1 = fig.add_axes([0.5 - AX_WIDTH / 2, 2.5 * GAP + 2 * AX_HEIGHT, AX_WIDTH, AX_HEIGHT])
    # ax1.set_title('Core 2')
    # set_concentration_axes_stuff(ax1, set_xlabel=True, set_ylabel=True)

    ax2 = fig.add_axes([0.5 - AX_WIDTH / 2, 1.5 * GAP + 1 * AX_HEIGHT, AX_WIDTH, AX_HEIGHT])
    # ax2.set_title('Retardation Factor')

    ax3 = fig.add_axes([0.5 - AX_WIDTH - GAP / 2, GAP / 2, AX_WIDTH, AX_HEIGHT])
    # ax3.set_title('Core 1')

    ax4 = fig.add_axes([0.5 + GAP / 2, GAP / 2, AX_WIDTH, AX_HEIGHT])
    # ax4.set_title('Core 2B')

    axs = np.array([ax3, ax1, ax4, ax2])
    return fig, axs


def get_axes_for_ddb_samples(**fig_kwargs) -> tuple[plt.Figure, np.ndarray]:
    """"Creates a figure and axes layout for plotting DDB samples."""

    FIG_WIDTH = 3 * FIGURE_WIDTH
    FIG_HEIGHT = FIGURE_HEIGHT
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), **fig_kwargs)
    GAP = 0.1
    AX_WIDTH, AX_HEIGHT = solve_ax_dimensions(GAP)

    # Calculate width for three axes side by side
    total_gap = 4 * GAP  # Left margin + 2 gaps between axes + right margin
    available_width = 1.0 - total_gap
    ax_width = available_width / 3

    # Calculate height to maintain aspect ratio
    ax_height = ax_width * (FIG_WIDTH / FIG_HEIGHT)

    # Center vertically
    vertical_offset = (1.0 - ax_height) / 2

    # rect = [left, bottom, width, height]
    ax1 = fig.add_axes([GAP, vertical_offset, ax_width, ax_height])
    ax2 = fig.add_axes([2 * GAP + ax_width, vertical_offset, ax_width, ax_height])
    ax3 = fig.add_axes([3 * GAP + 2 * ax_width, vertical_offset, ax_width, ax_height])

    axs = np.array([ax1, ax2, ax3])
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
