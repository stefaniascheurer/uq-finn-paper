from typing import Any
import numpy as np


class Params:
    """Manages FINN parameters."""

    def __init__(
        self,
        *,
        # Training params
        n_epochs: int = 32,
        error_mult: float = 1e5,
        phys_mult: float = 1e2,
        start_lr: float = 1e-1,
        use_adam_optim: bool = True,
        # Domain params
        X: float = 1.0,  # length of sample [m]
        T: float = 10000,  # simulation time [days]
        Nx: int = 26,
        Nt: int = 201,
        # Soil params
        D: float = 0.0005,  # effective diffusion coefficient [m^2/day]
        por: float = 0.29,  # porosity [-]
        rho_s: float = 2880,  # bulk density [kg/m^3]
        solubility: float = 1.0,  # top boundary value [kg/m^3]
        # FINN params
        c_diss_max: float = 1.0,  # for evaluating retardation (inclusive; for phys loss and output)
        n_c_diss: int = 100,
        **kwargs,
    ):
        """
        Args:
            n_epochs: Number of training epochs.
            error_mult: Multiplier for the data loss term.
            phys_mult: Multiplier for the physics loss term.
            start_lr: Starting learning rate.
            use_adam_optim: Whether to use Adam optimizer.
            X: Length of the sample in meters.
            T: Total simulation time in days.
            Nx: Number of spatial grid points (excluding boundaries).
            Nt: Number of temporal grid points.
            D: Effective diffusion coefficient in m^2/day.
            por: Porosity of the medium (dimensionless).
            rho_s: Bulk density in kg/m^3.
            solubility: Solubility limit for the dissolved concentration in kg/m^3.
            c_diss_max: Maximum dissolved concentration
        """

        # Training params
        self.n_epochs = n_epochs
        self.error_mult = error_mult
        self.phys_mult = phys_mult
        self.start_lr = start_lr
        self.use_adam_optim = use_adam_optim

        # Domain params
        self.X = X
        self.T = T
        self.Nx = int(Nx)  # no. inner grid points (aka. no. cells - 1)
        self.Nt = int(Nt)
        self.dx = self.X / (self.Nx + 1)  # length of discrete control volume [m]
        self.dt = self.T / (self.Nt + 1)  # time step [days]

        # Soil params
        self.D = D
        self.por = por
        self.rho_s = rho_s
        self.solubility = solubility
        self.D_eff = kwargs.pop("D_eff", [self.D / self.dx**2, 0.25])  # effective diffusion coefficient for each variable
        # self.D_eff = [self.D / self.dx**2, self.D * self.por / (self.rho_s/1000) / self.dx**2]
        # Note: Insert the params the above equation does not yield 0.25

        self.Kf = 0.716 / self.rho_s  # Freundlich's K [(m^3/kg)^nf]
        self.nf = 0.874  # freundlich exponent [-]
        self.smax = 1 / 1500  # sorption capacity [m^3/kg]
        self.Kl = 1  # half-concentration [kg/m^3]
        self.Kd = 0.429 / 1000  # organic carbon partitioning [m^3/kg]

        # Boundary conditions
        self.dirichlet_bool = kwargs.pop("dirichlet_bool", [[True, False, False, False]] * 2)
        self.neumann_bool = kwargs.pop("neumann_bool", [[False, False, True, True]] * 2)
        self.cauchy_bool = kwargs.pop("cauchy_bool", [[False, True, False, False]] * 2)
        self.dirichlet_val = kwargs.pop("dirichlet_val", [[solubility, 0, 0, 0]] * 2)
        self.neumann_val = kwargs.pop("neumann_val", [[0] * 4] * 2)
        self.cauchy_mult = kwargs.pop("cauchy_mult", [self.dx, self.dx])

        # FINN params
        self.c_diss_max = c_diss_max
        self.n_c_diss = n_c_diss
        self.p_exp_flux = kwargs.pop("p_exp_flux", [0.0, 0.0])  # Normalizer for functions that are approximated with a NN
        self.learn_coeff = kwargs.pop("learn_coeff", [False, True])  # Diffusion coefficient is learnable or not

        # Check all kwargs where used
        assert len(kwargs) == 0, kwargs

    @classmethod
    def from_dict(cls, is_exp_data: bool = False, **kwargs) -> "Params":
        """Creates FINN parameters from a dictionary.

        Args:
            is_exp_data: Whether the parameters are for experimental data.
            **kwargs: Keyword arguments corresponding to FINN parameters.
        """
        if "Dirichlet" in kwargs:
            kwargs["dirichlet_bool"] = [[True, bool(kwargs.get("Dirichlet")), False, False]] * 2
            kwargs.pop("Dirichlet")
        if "Cauchy" in kwargs:
            kwargs["cauchy_bool"] = [[False, bool(kwargs.get("Cauchy")), False, False]] * 2
            kwargs.pop("Cauchy")

        r = kwargs.pop("sample_radius")
        Q = kwargs.pop("Q")
        A = np.pi * r**2

        finn_params = cls(**kwargs)

        if is_exp_data:
            kwargs["learn_coeff"] = [False, False]
            finn_params.D_eff[1] = (finn_params.D * finn_params.por / (finn_params.rho_s / 1000) / (finn_params.dx**2))

        cauchy_val = finn_params.por * A / Q * finn_params.dx
        finn_params.cauchy_mult = [cauchy_val, cauchy_val]

        return finn_params

    def to_dict(self) -> dict[str, Any]:
        """Converts the FINN parameters to a dictionary."""
        return {
            "n_epochs": self.n_epochs,
            "error_mult": self.error_mult,
            "phys_mult": self.phys_mult,
            "start_lr": self.start_lr,
            "use_adam_optim": self.use_adam_optim,
            "X": self.X,
            "T": self.T,
            "Nx": self.Nx,
            "Nt": self.Nt,
            "dirichlet_bool": self.dirichlet_bool,
            "neumann_bool": self.neumann_bool,
            "cauchy_bool": self.cauchy_bool,
            "dirichlet_val": self.dirichlet_val,
            "neumann_val": self.neumann_val,
            "cauchy_mult": self.cauchy_mult,
            "D": self.D,
            "por": self.por,
            "rho_s": self.rho_s,
            "solubility": self.solubility,
            "D_eff": self.D_eff,
            "c_diss_max": self.c_diss_max,
            "n_c_diss": self.n_c_diss,
            "p_exp_flux": self.p_exp_flux,
            "learn_coeff": self.learn_coeff,
        }
