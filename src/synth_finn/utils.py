import numpy as np
import torch
from typing import Literal, Callable

from synth_finn.config import Params
from synth_finn.io import load_synthetic_data
from synth_finn.model import solve_allen_cahn_pde, solve_burgers_pde, interp1D_torch


def construct_learned_fx(u_for_learned: np.ndarray, learned: np.ndarray) -> Callable[[torch.Tensor], torch.Tensor]:
    """Helper to convert discrete learned function points into a continuous interpolation function.

    Args:
        u_for_learned: Values of u where learned function is sampled.
        learned: Learned values L(u).
    """
    # Check for uniform spacing for interp1D_torch
    assert np.allclose(np.diff(u_for_learned), np.full(len(u_for_learned) - 1, u_for_learned[1] - u_for_learned[0]))

    learned_tensor = torch.from_numpy(learned).float()
    u_min, u_max = u_for_learned.min(), u_for_learned.max()

    def learned_fx(u):
        return interp1D_torch(learned_tensor, u_min, u_max, u)

    return learned_fx


def compute_u_last_timestep(u_for_learned: np.ndarray, learned: np.ndarray, mode: Literal["train-test", "in-dis-test", "out-dis-test"]) -> np.ndarray:
    """Solves the PDE and returns the full field u(x, t).

    Args:
        u_for_learned: Grid for the learned function input.
        learned: Learned values.
        mode: "train-test", "in-dis-test", or "out-dis-test" to specify which data to load.
    """
    learned_fun = construct_learned_fx(u_for_learned, learned)

    x, u = load_synthetic_data(mode)

    params = Params.from_dict()
    Nt_total = 2 * (params.Nt - 1) + 1
    t_eval = np.linspace(0, 2.0, Nt_total)

    if mode != "in-dis-test":
        t = t_eval[:201]
    else:
        t = t_eval

    if params.model == "allen-cahn":
        if mode == "train-test" or mode == "in-dis-test":
            u0 = (x**2) * np.cos(np.pi * x)
        else:
            u0 = np.sin(np.pi * x / 2.0)
        # u0 must be [1, Nx] for the Flux module
        u0 = torch.FloatTensor(u0).unsqueeze(0).unsqueeze(-1)
        u_ode = solve_allen_cahn_pde(learned_fun, torch.tensor(t), params, u0)
    else:
        if mode == "train-test" or mode == "in-dis-test":
            u0 = - np.sin(np.pi * x)
        else:
            u0 = np.sin(np.pi * x)
        # u0 must be [1, Nx, 1] for the Flux module
        u0 = torch.FloatTensor(u0).unsqueeze(0).unsqueeze(-1)
        u_ode = solve_burgers_pde(learned_fun, torch.tensor(t), params, u0)

    # Return as [len(t), Nx]
    return u_ode[-1].squeeze()


def compute_u(u_for_learned: np.ndarray, learned: np.ndarray, mode: Literal["train-test", "in-dis-test", "out-dis-test"]) -> np.ndarray:
    """Computes the full spatial field for the given mode"""
    return compute_u_last_timestep(u_for_learned, learned, mode)
