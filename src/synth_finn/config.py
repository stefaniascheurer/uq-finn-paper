from typing import Any

import numpy as np


class Params:
    """Manages FINN parameters."""

    def __init__(
        self,
        *,
        # Training params
        n_epochs: int = 30,
        error_mult: float = 1,
        phys_mult: float = 0,
        start_lr: float = 1e-1,
        use_adam_optim: bool = False,
        # Domain params (Defaulted to the experiment specs)
        X: float = 2.0,      # Domain length from -1 to 1
        T: float = 1.0,  # 0.5,    # Simulation time (0.5 for train, 0.5 for test)
        Nx: int = 49,
        Nt: int = 201,
        # Physical params
        D: float = 0.01 / np.pi,  # 1e-4 # 0.01/np.pi for burgers, 1e-4 for allen_cahn,  # Diffusion coefficient D
        model: str = "burgers"  # allen_cahn or burgers
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
            D: Diffusion coefficient in m^2/day.
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
        self.Nx = int(Nx)
        self.Nt = int(Nt)

        # Grid spacing
        # For a grid of 49 points on length 2.0:
        self.dx = self.X / (self.Nx + 1)
        self.dt = self.T / (self.Nt + 1)

        # Physics params
        # We use a list to support the indexing [var_idx] in FluxKernels
        self.D = D

        self.model = model

    @classmethod
    def from_dict(cls, **kwargs) -> "Params":
        """Creates FINN parameters from a dictionary.

        Args:
            **kwargs: Keyword arguments corresponding to FINN parameters.
        """
        return cls(**kwargs)

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
            "D": self.D,
            "dx": self.dx
        }
