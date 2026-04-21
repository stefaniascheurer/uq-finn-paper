import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np

from synth_finn.config import Params


def create_mlp(layers: list[int] = [1, 10, 20, 10, 1], activation_fun: nn.Module = nn.Tanh(), activation_fun_end: nn.Module = None, dropout: int = 0) -> nn.Sequential:
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
    if activation_fun_end is not None:
        network_layers.append(activation_fun_end)

    return nn.Sequential(*network_layers)


class Flux_AllenCahn(torch.nn.Module):
    """Construct flux modules for all variables in the system."""

    def __init__(self, u0: torch.Tensor, cfg: Params, react_fun: nn.ModuleList):
        """
        Args:
            u0: Initial condition, dim: [1, Nx, 1].
            cfg: Configuration object of the model setup, containing boundary condition types, values, learnable parameter settings, etc.
            react_fun: Reaction function neural network for the variable in the system.
        """

        super(Flux_AllenCahn, self).__init__()

        assert len(react_fun) == 1

        self.cfg = cfg
        self.num_vars = u0.size(0)
        self.flux_modules = torch.nn.ModuleList([FluxKernels_AllenCahn(u0[i], self.cfg, coeff_nn=react_fun[i]) for i in range(self.num_vars)])

    def forward(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """Compute du/dt.

        Args:
            t: Time, taken from the ODE solver.
            u: Unknown variables to be calculated taken from the previous time step, dim: [Nx].

        """
        flux = [self.flux_modules[i](u, t) for i in range(self.num_vars)]
        dudt = torch.stack(flux)

        return dudt


class FluxKernels_AllenCahn(torch.nn.Module):
    def __init__(self, u0: torch.Tensor, cfg: Params, coeff_nn=None):
        """
        Args:
            u0: Initial condition, dim: [Nx, 1].
            cfg: Configuration object of the model setup, containing boundary condition types, values, learnable parameter settings, etc.
            coeff_nn: Neural network to predict reaction function as a function of the unknown variable.
        """

        super(FluxKernels_AllenCahn, self).__init__()
        self.Nx = u0.size(0)
        self.u0 = u0
        self.D = cfg.D
        self.dx = cfg.dx

        # NN predicting reaction term R(u)
        self.coeff_nn = coeff_nn
        self.p_mult = nn.Parameter(torch.tensor([10.0], dtype=torch.float))

    def forward(self, u: torch.Tensor, t: float) -> torch.Tensor:
        """Computes du/dt for the given variable.

        Args:
            u: Unknown variable to be used to calculate the flux, dim: [Nx, 1].
            t: Time, taken from the ODE solver.
        """

        return compute_flux_ac(
            u=u,
            t=t,
            D=self.D,
            dx=self.dx,
            coeff_nn=self.coeff_nn,
            p_mult=self.p_mult
        )


def compute_flux_ac(
    u: torch.Tensor,
    t: float,
    D: float,
    dx: float,
    p_mult: float = 10.0,
    coeff_nn=None,
) -> torch.Tensor:
    """Computes the integrated flux between each control volume and its neighbors.

    Args:
        u: Unknown variable to be used to calculate the flux, dim: [Nx, 1].
        t: Time, taken from the ODE solver.
        D: Diffusion coefficient.
        dx: Grid spacing.
        coeff_nn: Neural network to predict reaction function as a function of the unknown variable.
    """

    u = u.squeeze()  # [Nx]

    # Left boundary
    left_bound_flux = (u[-1] - 2 * u[0] + u[1]) / (dx**2)
    # Internal nodes
    internal_flux = (u[:-2] - 2 * u[1:-1] + u[2:]) / (dx**2)
    # Right boundary
    right_bound_flux = (u[-2] - 2 * u[-1] + u[0]) / (dx**2)

    diffusion = D * torch.cat((left_bound_flux.unsqueeze(0), internal_flux, right_bound_flux.unsqueeze(0)))

    # Raction calculation (Learnable via NN)
    if coeff_nn is not None:
        # Input to NN is [Nx, 1], output is [Nx, 1]
        reaction = coeff_nn(u.unsqueeze(-1)).squeeze(-1) * p_mult
    else:
        # Default Allen-Cahn reaction: 5u - 5u^3
        reaction = 5 * u - 5 * u**3

    dudt = (diffusion + reaction).unsqueeze(-1)
    return dudt


class Flux_Burgers(torch.nn.Module):
    """Construct flux modules for all variables in the system."""

    def __init__(self, u0: torch.Tensor, cfg: Params, adv_fun: nn.ModuleList):
        """
        Args:
            u0: Initial condition, dim: [1, Nx, 1].
            cfg: Configuration object of the model setup, containing boundary condition types, values, learnable parameter settings, etc.
            adv_fun: Advection function neural network for the variable in the system.
        """

        super(Flux_Burgers, self).__init__()

        assert len(adv_fun) == 1

        self.cfg = cfg
        self.num_vars = u0.size(0)
        self.flux_modules = torch.nn.ModuleList([FluxKernels_Burgers(u0[i], self.cfg, coeff_nn=adv_fun[i]) for i in range(self.num_vars)])

    def forward(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """Compute du/dt.

        Args:
            t: Time, taken from the ODE solver.
            u: Unknown variables to be calculated taken from the previous time step, dim: [Nx].

        """
        flux = [self.flux_modules[i](u, t) for i in range(self.num_vars)]
        dudt = torch.stack(flux)

        return dudt


class FluxKernels_Burgers(torch.nn.Module):
    def __init__(self, u0: torch.Tensor, cfg: Params, coeff_nn=None):
        """
        Args:
            u0: Initial condition, dim: [Nx, 1].
            cfg: Configuration object of the model setup, containing boundary condition types, values, learnable parameter settings, etc.
            coeff_nn: Neural network to predict reaction function as a function of the unknown variable.
        """

        super(FluxKernels_Burgers, self).__init__()
        self.Nx = u0.size(0)
        self.u0 = u0
        self.D = cfg.D
        self.dx = cfg.dx

        # NN predicting adcvection term a(u)
        self.coeff_nn = coeff_nn

    def forward(self, u: torch.Tensor, t: float) -> torch.Tensor:
        """Computes du/dt for the given variable.

        Args:
            u: Unknown variable to be used to calculate the flux, dim: [Nx, 1].
            t: Time, taken from the ODE solver.
        """
        return compute_flux_b(
            u=u,
            t=t,
            D=self.D,
            dx=self.dx,
            coeff_nn=self.coeff_nn
        )


def compute_flux_b(
    u: torch.Tensor,
    t: float,
    D: float,
    dx: float,
    coeff_nn=None,
) -> torch.Tensor:
    """Computes the integrated flux between each control volume and its neighbors.

    Args:
        u: Unknown variable to be used to calculate the flux, dim: [Nx, 1].
        t: Time, taken from the ODE solver.
        D: Diffusion coefficient.
        dx: Grid spacing.
        coeff_nn: Neural network to predict reaction function as a function of the unknown variable.
    """

    u = u.squeeze()
    if coeff_nn is not None:
        a = coeff_nn(u.unsqueeze(-1)).squeeze()
    else:
        a = u

    # Upwind scheme for advection term to ensure stability at shock fronts
    # a_plus handles flow to the right, a_min handles flow to the left
    a_plus = torch.relu(a)
    a_min = -torch.relu(-a)
    u_l = torch.cat((torch.tensor([0.0], device=u.device), u[:-1]))
    u_r = torch.cat((u[1:], torch.tensor([0.0], device=u.device)))

    adv_flux = (a_plus * (u - u_l) + a_min * (u_r - u)) / dx

    diff_flux = (u_l - 2 * u + u_r) / (dx**2)

    dudt = -adv_flux + D * diff_flux

    return dudt.unsqueeze(-1)


def interp1D_torch(y: torch.Tensor, xmin, xmax, x: torch.Tensor) -> torch.Tensor:
    """Linear interpolation on Torch tensors for reaction function mapping.

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


def solve_allen_cahn_pde(react_fx: callable, t: torch.Tensor, params, u0: torch.Tensor) -> np.ndarray:
    """Wrapper to solve the PDE using torchdiffeq odeint.

    Args:
        react_fx: Reaction function.
        t: Time points to solve the PDE at, shape: [n_time_points].
        params: FINN Params object containing model parameters.
        u0: Initial condition, dim: [Nx, 1]. If None, zeros are used.
    """

    model = Flux_AllenCahn(u0, params, react_fun=[react_fx])
    return odeint(model, u0, t, rtol=1e-5, atol=1e-6).detach().numpy()


def solve_burgers_pde(adv_fx: callable, t: torch.Tensor, params, u0: torch.Tensor) -> np.ndarray:
    """Wrapper to solve the PDE using torchdiffeq odeint.

    Args:
        adv_fx: Advection function.
        t: Time points to solve the PDE at, shape: [n_time_points].
        params: FINN Params object containing model parameters.
        u0: Initial condition, dim: [Nx, 1]. If None, zeros are used.
    """

    model = Flux_Burgers(u0, params, adv_fun=[adv_fx])
    return odeint(model, u0, t).detach().numpy()
