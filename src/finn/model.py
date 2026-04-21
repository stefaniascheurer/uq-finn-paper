import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np

from finn.config import Params


def create_mlp(layers: list[int] = [1, 10, 20, 10, 1], activation_fun: nn.Module = nn.Tanh(), activation_fun_end: nn.Module = nn.Sigmoid(), dropout: int = 0) -> nn.Sequential:
    """Create a multi-layer perceptron (MLP) neural network.

    Args:
        layers: List of integers defining the number of neurons in each layer.
        activation_fun: Activation function to use between layers.
        activation_fun_end: Activation function to use at the output layer.
        dropout: Dropout percentage to apply after each hidden layer (0 for no dropout).
    """
    network_layers = []

    for i in range(len(layers) - 1):
        network_layers.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            network_layers.append(activation_fun)
            if dropout > 0:
                print("Dropout added.")
                network_layers.append(nn.Dropout(p=dropout / 100.0))

    network_layers.append(activation_fun_end)

    return nn.Sequential(*network_layers)


class AnalyticRetardation:
    """Analytic retardation functions for different isotherm models."""

    @staticmethod
    def linear(u: float, por: float, rho_s: float, Kd: float) -> np.ndarray:
        """Linear isotherm model."""
        factor = 1 + (1 - por) / por * rho_s * Kd
        return factor * np.ones_like(u)

    @staticmethod
    def freundlich(u: float, por: float, rho_s: float, Kf: float, nf: float) -> np.ndarray:
        """Freundlich isotherm model."""
        return 1 + (1 - por) / por * rho_s * Kf * nf * (u + 1e-6) ** (nf - 1)

    @staticmethod
    def langmuir(u: float, por: float, rho_s: float, smax: float, Kl: float) -> np.ndarray:
        """Langmuir isotherm model."""
        return 1 + (1 - por) / por * rho_s * smax * Kl / ((u + Kl) ** 2)


class Flux(torch.nn.Module):
    """Construct flux modules for all variables in the system."""

    def __init__(self, u0: torch.Tensor, cfg: Params, ret_funs: nn.ModuleList):
        """
        Args:
            u0: Initial condition, dim: [num_features, Nx, Ny].
            cfg: Configuration object of the model setup, containing boundary condition types, values, learnable parameter settings, etc.
            ret_funs: List of retardation function neural networks for each variable in the system.
        """

        super(Flux, self).__init__()

        assert len(ret_funs) == 2

        self.cfg = cfg
        self.num_vars = u0.size(0)
        self.flux_modules = torch.nn.ModuleList([FluxKernels(u0[i], self.cfg, i, coeff_nn=ret_funs[i]) for i in range(self.num_vars)])

    def forward(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """Compute du/dt for all variables in the system.

        Args:
            t: Time, taken from the ODE solver.
            u: Unknown variables to be calculated taken from the previous time step, dim: [num_features, Nx, Ny].

        """

        flux = [self.flux_modules[i](u[[0]], u[[0]], t) for i in range(self.num_vars)]
        dudt = torch.stack(flux)

        return dudt


class FluxKernels(torch.nn.Module):
    """Construct flux modules for one variable in the system."""

    def __init__(self, u0: torch.Tensor, cfg: Params, var_idx: int, coeff_nn=None):
        """
        Args:
            u0: Initial condition, dim: [num_features, Nx, Ny].
            cfg: Configuration object of the model setup, containing boundary condition types, values, learnable parameter settings, etc.
            var_idx: Index of the variable to be calculated.
            coeff_nn: Neural network to predict retardation factor as a function of the unknown variable.
        """

        super(FluxKernels, self).__init__()

        # Extracting the spatial dimension and initial condition of the problem and store the initial condition value u0
        self.Nx = u0.size(0)
        self.Ny = u0.size(1)
        self.u0 = u0

        # Variables that act as switch to use different types of boundary condition
        # Each variable consists of boolean values at all 2D domain boundaries:
        # [left (x = 0), right (x = Nx), top (y = 0), bottom (y = Ny)]
        # For 1D, only the first two values matter, set the last two values to be no-flux boundaries (zero neumann_val)
        self.dirichlet_bool = cfg.dirichlet_bool[var_idx]
        self.neumann_bool = cfg.neumann_bool[var_idx]
        self.cauchy_bool = cfg.cauchy_bool[var_idx]

        # Variables that store the values of the boundary condition of each type
        # Values = 0 if not used, otherwise specify in the configuration file
        # Each variable consists of real values at all 2D domain boundaries:
        # [left (x = 0), right (x = Nx), top (y = 0), bottom (y = Ny)]
        # For 1D, only the first two values matter, set the last two values to be no-flux boundaries
        self.dirichlet_val = cfg.dirichlet_val[var_idx]
        self.neumann_val = cfg.neumann_val[var_idx]

        # For Cauchy BC, the initial Cauchy value is set to be the initial condition at each corresponding domain boundary, and will be updated through time
        self.cauchy_val = [u0[0, :], u0[-1, :], u0[:, 0], u0[:, -1]]

        # Set the Cauchy BC multiplier (to be multiplied with the gradient of the unknown variable and the diffusion coefficient)
        self.cauchy_mult = cfg.cauchy_mult[var_idx]

        self.stencil = (1.0, -1.0)

        # Extract the diffusion coefficient scalar value and set to be learnable if desired
        self.D_eff = cfg.D_eff[var_idx]
        if cfg.learn_coeff[var_idx]:
            self.D_eff = torch.nn.Parameter(torch.tensor([self.D_eff], dtype=torch.float))  # type: ignore

        # Extract value of the normalizing constant to be applied to the output of the NN that predicts the diffusion coefficient function
        self.p_exp = cfg.p_exp_flux[var_idx]

        # Initialize a NN to predict diffusion coefficient as a function of the unknown variable if necessary
        if coeff_nn is not None:
            self.coeff_nn = coeff_nn
            self.p_exp = torch.nn.Parameter(torch.tensor([self.p_exp], dtype=torch.float))  # type: ignore
        else:
            self.coeff_nn = None

    def forward(self, u_main: torch.Tensor, u_coupled: torch.Tensor, t: float) -> torch.Tensor:
        """Compute du/dt for one variable in the system.

        Args: 
                u_main: Unknown variable to be calculated taken from the previous time step, dim: [1, Nx, Ny].
                u_coupled: All necessary unknown variables required to calculate the diffusion coeffient as a function, dim: [num_features, Nx, Ny].
                t: Time, taken from the ODE solver.
        """

        return compute_flux(
            u_main=u_main,
            u_coupled=u_coupled,
            t=t,
            u0=self.u0,
            Nx=self.Nx,
            Ny=self.Ny,
            D_eff=self.D_eff,  # type: ignore
            cauchy_mult=self.cauchy_mult,
            stencil=self.stencil,
            dirichlet_bool=self.dirichlet_bool,
            dirichlet_val=self.dirichlet_val,
            neumann_bool=self.neumann_bool,
            neumann_val=self.neumann_val,
            cauchy_bool=self.cauchy_bool,
            cauchy_val=self.cauchy_val,
            coeff_nn=self.coeff_nn,
            p_exp=self.p_exp,
        )


def compute_flux(
    u_main: torch.Tensor,
    u_coupled: torch.Tensor,
    t: float,
    u0: torch.Tensor,
    Nx: int,
    Ny: int,
    D_eff: float,
    cauchy_mult: float,
    stencil: tuple[float, float],
    dirichlet_bool: list[bool],
    dirichlet_val: list[float],
    neumann_bool: list[bool],
    neumann_val: list[float],
    cauchy_bool: list[bool],
    cauchy_val: list[float],  # has to be list because this is modified in here
    coeff_nn=None,
    p_exp=None,
) -> torch.Tensor:
    """Computes the integrated flux between each control volume and its neighbors.

    Args:
        u_main: Unknown variable to be used to calculate the flux, dim: [1, Nx, Ny].
        u_coupled: All necessary unknown variables required to calculate the diffusion coeffient as a function, dim: [num_features, Nx, Ny].
        t: Time (scalar value, taken from the ODE solver).
    """

    # Reshape the input dimension for the coeff_nn model into [Nx, Ny, num_features]
    u_coupled = u_coupled.permute(1, 2, 0)

    # Calculate the flux multiplier (diffusion coefficient function) if set to be a function, otherwise set as tensor of ones
    if coeff_nn is not None:
        assert p_exp is not None
        flux_mult = coeff_nn(u_coupled).squeeze(2) * 10**p_exp
    else:
        flux_mult = torch.ones(Nx, Ny)

    # Squeeze the u_main dimension into [Nx, Ny]
    u_main = u_main.squeeze(0)

    # Left boundary condition
    if dirichlet_bool[0]:
        # If Dirichlet, calculate the flux at the boundary using the Dirichlet value as a constant
        left_bound_flux = ((stencil[0] * dirichlet_val[0] + stencil[1] * u_main[0, :]).unsqueeze(0) * D_eff * flux_mult[0, :])

    elif neumann_bool[0]:
        # If Neumann, set the Neumann value as the flux at the boundary
        left_bound_flux = torch.full((1, Ny), neumann_val[0])

    elif cauchy_bool[0]:
        # If Cauchy, first set the value to be equal to the initial condition at t = 0.0, otherwise update the value according to the previous time step value
        if t == 0.0:
            cauchy_val[0] = u0[0, :]
        else:
            cauchy_val[0] = (u_main[0, :] - cauchy_val[0]) * cauchy_mult * D_eff
        # Calculate the flux at the boundary using the updated Cauchy value
        left_bound_flux = ((stencil[0] * cauchy_val[0] + stencil[1] * u_main[0, :]).unsqueeze(0) * D_eff * flux_mult[0, :])

    # Calculate the fluxes of each control volume with its left neighboring cell
    left_neighbors = ((stencil[0] * u_main[:-1, :] + stencil[1] * u_main[1:, :]) * D_eff * flux_mult[1:, :])
    # Concatenate the left boundary fluxes with the left neighbors fluxes
    left_flux = torch.cat((left_bound_flux, left_neighbors))

    # Right boundary condition
    if dirichlet_bool[1]:
        # If Dirichlet, calculate the flux at the boundary using the Dirichlet value as a constant
        right_bound_flux = ((stencil[0] * dirichlet_val[1] + stencil[1] * u_main[-1, :]).unsqueeze(0) * D_eff * flux_mult[-1, :])

    elif neumann_bool[1]:
        # If Neumann, set the Neumann value as the flux at the boundary
        right_bound_flux = torch.full((1, Ny), neumann_val[1])

    elif cauchy_bool[1]:
        # If Cauchy, first set the value to be equal to the initial condition at t = 0.0, otherwise update the value according to the previous time step value
        if t == 0.0:
            cauchy_val[1] = u0[-1, :]
        else:
            cauchy_val[1] = (u_main[-1, :] - cauchy_val[1]) * cauchy_mult * D_eff
        # Calculate the flux at the boundary using the updated Cauchy value
        right_bound_flux = ((stencil[0] * cauchy_val[1] + stencil[1] * u_main[-1, :]).unsqueeze(0) * D_eff * flux_mult[-1, :])

    # Calculate the fluxes of each control volume with its right neighboring cell
    right_neighbors = ((stencil[0] * u_main[1:, :] + stencil[1] * u_main[:-1, :]) * D_eff * flux_mult[:-1, :])
    # Concatenate the right neighbors fluxes with the right boundary fluxes
    right_flux = torch.cat((right_neighbors, right_bound_flux))

    # Top boundary condition
    if dirichlet_bool[2]:
        # If Dirichlet, calculate the flux at the boundary using theDirichlet value as a constant
        top_bound_flux = ((stencil[0] * dirichlet_val[2] + stencil[1] * u_main[:, 0]).unsqueeze(1) * D_eff * flux_mult[:, 0])

    elif neumann_bool[2]:
        # If Neumann, set the Neumann value as the flux at the boundary
        top_bound_flux = torch.full((Nx, 1), neumann_val[2])

    elif cauchy_bool[2]:
        # If Cauchy, first set the value to be equal to the initial condition at t = 0.0, otherwise update the value according to the previous time step value
        if t == 0.0:
            cauchy_val[2] = u0[:, 0]
        else:
            cauchy_val[2] = (u_main[:, 0] - cauchy_val[2]) * cauchy_mult * D_eff
        # Calculate the flux at the boundary using the updated Cauchy value
        top_bound_flux = ((stencil[0] * cauchy_val[2] + stencil[1] * u_main[:, 0]).unsqueeze(1) * D_eff * flux_mult[:, 0])

    # Calculate the fluxes of each control volume with its top neighboring cell
    top_neighbors = ((stencil[0] * u_main[:, :-1] + stencil[1] * u_main[:, 1:]) * D_eff * flux_mult[:, 1:])
    # Concatenate the top boundary fluxes with the top neighbors fluxes
    top_flux = torch.cat((top_bound_flux, top_neighbors), dim=1)

    # Bottom boundary condition
    if dirichlet_bool[3]:
        # If Dirichlet, calculate the flux at the boundary using the Dirichlet value as a constant
        bottom_bound_flux = ((stencil[0] * dirichlet_val[3] + stencil[1] * u_main[:, -1]).unsqueeze(1) * D_eff * flux_mult[:, -1])

    elif neumann_bool[3]:
        # If Neumann, set the Neumann value as the flux at the boundary
        bottom_bound_flux = torch.full((Nx, 1), neumann_val[3])

    elif cauchy_bool[3]:
        # If Cauchy, first set the value to be equal to the initial condition at t = 0.0, otherwise update the value according to the previous time step value
        if t == 0.0:
            cauchy_val[3] = u0[:, -1]
        else:
            cauchy_val[3] = (u_main[:, -1] - cauchy_val[3]) * cauchy_mult * D_eff
        # Calculate the flux at the boundary using the updated Cauchy value
        bottom_bound_flux = ((stencil[0] * cauchy_val[3] + stencil[1] * u_main[:, -1]).unsqueeze(1) * D_eff * flux_mult[:, -1])

    # Calculate the fluxes of each control volume with its bottom neighboring cell
    bottom_neighbors = ((stencil[0] * u_main[:, 1:] + stencil[1] * u_main[:, :-1]) * D_eff * flux_mult[:, :-1])
    # Concatenate the bottom neighbors fluxes with the bottom boundary fluxes
    bottom_flux = torch.cat((bottom_neighbors, bottom_bound_flux), dim=1)

    # Integrate all fluxes at all control volume boundaries
    flux = left_flux + right_flux + top_flux + bottom_flux

    return flux


def interp1D_torch(y: torch.Tensor, xmin, xmax, x: torch.Tensor) -> torch.Tensor:
    """Linear interpolation on Torch tensors for retardation function mapping.

    Args:
        y: values to be interpolated, shape: [n_points, 1, 1].
        xmin: minimum x value of the known data points.
        xmax: maximum x value of the known data points.
        x: x values where interpolation is desired, shape: [m_points, 1, 1].
    """
    y, x = y.reshape(-1), x.reshape(-1)
    n = len(y) - 1
    xp, dx = torch.linspace(xmin, xmax, n + 1), (xmax - xmin) / n
    i = torch.clip(((x - xmin) / dx).to(int), 0, n - 1)
    y_interp = y[i] + (y[i + 1] - y[i]) * (x - xp[i]) / dx
    return y_interp.reshape(-1, 1, 1)


def solve_diffusion_sorption_pde(ret_fx: callable, t: torch.Tensor, params, c0: torch.Tensor = None) -> np.ndarray:
    """Wrapper to solve the PDE using torchdiffeq odeint.

    Args:
        ret_fx: Retardation function.
        t: Time points to solve the PDE at, shape: [n_time_points].
        params: FINN Params object containing model parameters.
        c0: Initial condition, dim: [num_features, Nx, 1]. If None, zeros are used.
    """
    if c0 is None:
        c0 = torch.zeros(2, params.Nx, 1).to(torch.float32)

    def coeff_nn_fun(c): return 1 / ret_fx(c)
    model = Flux(c0, params, ret_funs=[coeff_nn_fun, None])
    model.eval()
    return odeint(model, c0, t, rtol=1e-5, atol=1e-6).detach().numpy()
