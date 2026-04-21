import numpy as np
import torch
from typing import Literal, Callable, Optional
from pathlib import Path

from finn.config import Params
from finn.io import FINNDir, load_exp_data, load_exp_cfg
from finn.model import solve_diffusion_sorption_pde, interp1D_torch


def construct_ret_fx(c_for_ret: np.ndarray, ret: np.ndarray) -> Callable[[torch.Tensor], torch.Tensor]:
    """Helper to convert discrete retardation points into a continuous interpolation function.

    Args:
        c_for_ret: Concentration.
        ret: Retardation values.
    """
    assert np.allclose(np.diff(c_for_ret), np.full(len(c_for_ret) - 1, c_for_ret[1] - c_for_ret[0]))
    ret_tensor = torch.from_numpy(ret)
    c_min, c_max = c_for_ret.min(), c_for_ret.max()
    def ret_fx(c): return interp1D_torch(ret_tensor, c_min, c_max, c)
    return ret_fx


def compute_btc(c_for_ret: np.ndarray, ret: np.ndarray, cauchy_mult: float, D_eff: float, core_type: Literal["Core 1", "Core 2"]) -> np.ndarray:
    """Solves the PDE and extracts the breakthrough curve.

    Args:
        c_for_ret: Concentration points for retardation curve.
        ret: Retardation values.
        cauchy_mult: Cauchy multiplier for breakthrough curve scaling.
        D_eff: Effective diffusion coefficient.
        core_type: 'Core 1' or 'Core 2' to load appropriate experiment.
    """
    ret_fun = construct_ret_fx(c_for_ret, ret)
    data, cfg = load_exp_data(core_type), load_exp_cfg(core_type)
    t = torch.FloatTensor(data["time"])
    params = Params.from_dict(is_exp_data=True, **cfg)
    params.p_exp_flux = [0.0, 0.0]
    c0 = torch.zeros(2, params.Nx, 1).to(torch.float32)
    c_ode = solve_diffusion_sorption_pde(ret_fun, t, params, c0)
    cm = 0.0836712021582612 if core_type == "Core 1" else cauchy_mult * D_eff
    return ((c_ode[:, 0, -2] - c_ode[:, 0, -1]) * cm).squeeze()


def compute_core1_btc(c_for_ret: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """Post-processing breakthrough curve for Core 1.

    Args:
        c_for_ret: Concentration points for retardation curve.
        ret: Retardation values.
    """
    return compute_btc(c_for_ret, ret, None, None, "Core 1")


def compute_core2_btc(finn_dir: FINNDir, c_for_ret: np.ndarray, ret: np.ndarray, cauchy_mult: Optional[float] = None, D_eff: Optional[float] = None) -> np.ndarray:
    """Post-processing breakthrough curve for Core 2.

    Args:
        finn_dir: FINNDir instance to load parameters from.
        c_for_ret: Concentration points for retardation curve.
        ret: Retardation values.
        cauchy_mult: Optional Cauchy multiplier. If None, loaded from finn_dir.
        D_eff: Optional effective diffusion coefficient. If None, loaded from finn_dir."""
    params = finn_dir.load_params()
    cm = cauchy_mult if cauchy_mult is not None else params["cauchy_mult"][0]
    de = D_eff if D_eff is not None else params["D_eff"][0]
    return compute_btc(c_for_ret, ret, cm, de, "Core 2")


def compute_core2B_profile(c_for_ret, ret):
    """Computes spatial concentration profile for Core 2B.

    Args:
        c_for_ret: Concentration points for retardation curve.
        ret: Retardation values.
    """
    ret_fx = construct_ret_fx(c_for_ret, ret)
    data, cfg = load_exp_data("Core 2B"), load_exp_cfg("Core 2B")
    t = torch.linspace(0.0, cfg["T"], 101)
    params = Params.from_dict(is_exp_data=True, **cfg)
    params.p_exp_flux = [0.0, 0.0]
    c0 = torch.zeros(2, params.Nx, 1).to(torch.float32)
    c_ode = solve_diffusion_sorption_pde(ret_fx, t, params, c0)
    x = data["x"].to_numpy()
    xp = np.linspace(0.0, cfg["X"], int(cfg["Nx"]))
    return np.interp(x, xp, c_ode[-1, 1, :, 0])


def iter_final_retardation_files(root: Path, min_epoch: int = 100, is_ret_OK: Optional[Callable[[np.ndarray], bool]] = None, verbose: bool = False):
    """
    Iterate trough a directory containing multiple folders with FINN simulation results and return the path to the final retardation curve file.

    Args:
        root: Root directory to search for FINN results.
        min_epoch: Minimum epoch number to consider a result valid.
        is_ret_OK: Optional function to validate the retardation curve.
        verbose: If True, print skipped directories and reasons.
    """
    finn_dirs = (p.parent for p in root.rglob("c_for_ret.npy"))
    for p in finn_dirs:
        all_ret_file_paths = sorted(
            (p / "ret").glob("ret_pred_*.npy"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        if not all_ret_file_paths:
            if verbose:
                print(f"Skipped {p}. No files found.")
            continue

        epoch = int(all_ret_file_paths[-1].stem.split("_")[-1])
        if epoch < min_epoch:
            if verbose:
                print(f"Skipped {p}. Epoch < {min_epoch}.")
            continue

        ret = np.load(all_ret_file_paths[-1])
        if np.any(np.isnan(ret)):
            if verbose:
                print(f"Skipped {p}. Ret contains NaNs.")
            continue

        if np.any(np.isinf(ret)):
            if verbose:
                print(f"Skipped {p}. Ret contains infs.")
            continue

        if np.any(ret > 1e6) or np.any(ret < 1e-6):
            if verbose:
                print(f"Skipped {p}. Ret is not in [1e-6, 1e6].")
            continue

        if is_ret_OK is not None:
            if not is_ret_OK(ret):
                if verbose:
                    print(f"Skipped {p}. Ret is not OK.")
                continue

        print(p, all_ret_file_paths[-1])
