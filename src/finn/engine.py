import time
import random
import numpy as np
import torch
import json
import pandas as pd
from torchdiffeq import odeint

from finn.io import FINNDir
from finn.config import Params
from finn.model import Flux, create_mlp


def random_logunif(min_val: float, max_val: float) -> float:
    """Samples a random value from a log-uniform distribution between min_val and max_val."""
    log_v = np.random.uniform(np.log10(min_val), np.log10(max_val))
    return 10**log_v


class Trainer:
    """Orchestrates the training process including losses and checkpointing."""

    def __init__(
        self,
        model,
        finn_dir: FINNDir,
        continue_training: bool = False,
        n_epochs: int = 30,
        error_mult: float = 1e5,
        phys_mult: float = 1e2,
        start_lr: float = 1e-1,
        use_adam_optim: bool = False,
        **kwargs
    ):
        self.model = model
        self.finn_dir = finn_dir

        self.n_epochs = n_epochs
        self.error_mult = error_mult
        self.phys_mult = phys_mult
        self.start_lr = start_lr

        if use_adam_optim:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
        else:
            self.optimizer = torch.optim.LBFGS(model.parameters(), lr=start_lr)

        self.start_epoch = 0
        self.train_losses = []
        self.best_loss = np.inf

        self.latest_mse_loss = None
        self.latest_physical_loss = None
        self.latest_pred = None
        self.latest_ode_pred = None
        self.latest_D_eff = None
        self.latest_cauchy_mult = None

        # Load the model from checkpoint
        if continue_training:
            print("Restoring model (that is the network's weights) from file...\n")

            self.checkpoint = torch.load(self.finn_dir.ckpt_path)

            # Load the model state_dict (all the network parameters)
            self.model.load_state_dict(self.checkpoint["state_dict"])

            # Load the optimizer state dict (important because ADAM and LBFGS
            # requires past states, e.g. momentum information and approximate
            # Hessian)
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v

            # Load the epoch and loss values from the previous training up until
            # the checkpoint to enable complete history of the training
            self.start_epoch = self.checkpoint["epoch"]
            self.train_losses = self.checkpoint["loss_train"]

    def train_model(self, *, c0: torch.Tensor, t: torch.Tensor, data: torch.Tensor) -> None:
        """Trains the model to fit the provided data.

        Args:
            c0: Initial concentration tensor.
            t: Time points tensor.
            data: Training data tensor.
        """
        c_for_ret = torch.linspace(0.0, 2.0, 100).view(-1, 1)
        np.save(self.finn_dir.c_for_ret_path, c_for_ret)
        criterion = torch.nn.MSELoss()
        loss_phys_fac = 1000 * self.phys_mult
        loss_mse_fac = self.error_mult

        c0 = c0.detach()
        t = t.detach()
        data = data.detach()

        c_btc_data = data[:, 1]

        def closure():

            c_for_ret = torch.linspace(0.0, 2.0, 100).view(-1, 1).detach()

            # Set the model to training mode
            self.model.train()

            # Reset the gradient buffer (set to 0)
            self.optimizer.zero_grad()

            # Calculate the model prediction (full field solution)
            c_full_pred: torch.Tensor = odeint(self.model, c0, t, rtol=1e-5, atol=1e-6)
            self.latest_c_full_pred = c_full_pred.clone().detach().numpy()

            # Extract the breakthrough curve from the full field solution prediction
            cauchy_mult = (self.model.flux_modules[0].cauchy_mult * self.model.flux_modules[0].D_eff)
            c_btc_pred = ((c_full_pred[:, 0, -2] - c_full_pred[:, 0, -1]) * cauchy_mult).squeeze()

            loss = loss_mse_fac * criterion(c_btc_data, c_btc_pred)

            # Extract the predicted retardation factor function for physical regularization
            ret_temp = self.model.flux_modules[0].coeff_nn(c_for_ret)
            # Normalize retardation such that there is no difference whether the rets are orders of magnitude apart but just if the slope is different
            ret_temp = ret_temp / ret_temp.max()

            # Physical regularization: value of the retardation factor should decrease with increasing concentration
            loss_physical = (loss_phys_fac * torch.mean(torch.relu(ret_temp[:-1] - ret_temp[1:])))
            self.latest_mse_loss = loss.detach().numpy()
            self.latest_physical_loss = loss_physical.detach().numpy()

            loss = loss + loss_physical

            loss.backward()

            self.latest_c_btc_pred = c_btc_pred.clone().detach().numpy()
            self.latest_D_eff = self.model.flux_modules[0].D_eff
            self.latest_cauchy_mult = self.model.flux_modules[0].cauchy_mult

            return loss

        lr_min, lr_max, decay_factor, T_0 = 1e-2, 0.1, 0.8, 10
        num_restarts = 30

        learning_rates = np.concatenate([lr_min + 0.5 * (lr_max * decay_factor**i - lr_min) * (1 + np.cos(np.linspace(0, np.pi, T_0))) for i in range(num_restarts)])

        # Iterate until maximum epoch number is reached
        for epoch in range(self.start_epoch, self.n_epochs):
            for g in self.optimizer.param_groups:
                g["lr"] = learning_rates[epoch]

            dt = time.time()

            # Update the model parameters and record the loss value
            loss = self.optimizer.step(closure)
            self.train_losses.append(loss.item())  # type: ignore

            dt = time.time() - dt

            print(
                f"It {epoch+1:>3}/{self.n_epochs}"
                f" | mse = {np.square(self.latest_c_btc_pred - c_btc_data.detach().numpy()).mean():.2e}"
                f" | loss = {self.train_losses[-1]:.2e}"
                f" | dt = {dt:.1f}s"
                f" | lr = {learning_rates[epoch]:.1e}"
                f" | loss_mse = {self.latest_mse_loss:.1e}"
                f" | loss_phys = {self.latest_physical_loss:.1e}"
            )

            ret_pred = (1 / self.model.flux_modules[0].coeff_nn(c_for_ret) / 10 ** self.model.flux_modules[0].p_exp)

            assert self.latest_c_full_pred is not None
            assert self.latest_c_btc_pred is not None
            assert self.latest_D_eff is not None
            assert self.latest_cauchy_mult is not None

            np.save(self.finn_dir.get_pred_ret_path(epoch), ret_pred.detach().numpy())
            np.save(self.finn_dir.get_pred_c_btc_path(epoch), self.latest_c_btc_pred)
            np.save(self.finn_dir.get_pred_c_full_path(epoch), self.latest_c_full_pred)

            np.save(self.finn_dir.get_D_eff_path(epoch), self.latest_D_eff)
            np.save(self.finn_dir.get_cauchy_mult_path(epoch), self.latest_cauchy_mult)
            np.save(self.finn_dir.get_p_exp_path(epoch), self.model.flux_modules[0].p_exp.detach().numpy())

            if (epoch + 1) % 2 == 0:
                self.save_model_to_file(epoch)

        np.save(self.finn_dir.loss_path, self.train_losses)

    def save_model_to_file(self, epoch: int) -> None:
        """Saves the current model and optimizer state to a checkpoint file."""
        torch.save({"epoch": epoch + 1, "state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "loss_train": self.train_losses}, self.finn_dir.ckpt_path)


def evaluate_model(finn_dir: FINNDir, t: torch.Tensor, is_exp_data: bool = True, **add_cfg) -> torch.Tensor:
    """Evaluate the trained FINN model to obtain breakthrough curve predictions.

    Args:
        finn_dir: The FINN directory containing the trained model.
        t: Time points tensor.
        is_exp_data: Whether the data is experimental.
        **add_cfg: Additional configuration for the model.
    """
    params = Params.from_dict(is_exp_data=is_exp_data, **add_cfg)
    c0 = torch.zeros(2, params.Nx, 1)

    model = Flux(c0, params, ret_funs=[create_mlp(), None])
    checkpoint = torch.load(finn_dir.ckpt_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    c_full_pred: torch.Tensor = odeint(model, c0, t, rtol=1e-5, atol=1e-6)
    cauchy_mult = (model.flux_modules[0].cauchy_mult * model.flux_modules[0].D_eff)
    c_btc_pred = ((c_full_pred[:, 0, -2] - c_full_pred[:, 0, -1]) * cauchy_mult).squeeze()

    return c_btc_pred


def setup_and_train_model(finn_dir: FINNDir, data: pd.DataFrame, is_exp_data: bool = True, random_seed: bool = False, **add_cfg):
    """Setup and train the FINN model.

    Args:
        finn_dir: The FINN directory to save results.
        data: Training data as a pandas DataFrame.
        is_exp_data: Whether the data is experimental.
        random_seed: Whether to randomize the random seed for training.
        **add_cfg: Additional configuration for the model.
    """
    if finn_dir.is_done:
        return

    if not finn_dir.c_train_path.exists():
        np.save(finn_dir.c_train_path, np.squeeze(data))
    c_train = torch.Tensor(np.load(finn_dir.c_train_path))

    if not finn_dir.t_train_path.exists():
        t_train = data["time"].to_numpy().ravel()
        np.save(finn_dir.t_train_path, t_train)
    t_train = torch.Tensor(np.load(finn_dir.t_train_path))

    if random_seed:
        seed = time.time_ns() % 10**9
    else:
        seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    params = Params.from_dict(is_exp_data=is_exp_data, **add_cfg)
    finn_dir.params_path.write_text(json.dumps(params.to_dict()))

    c0 = c_train[0].clone() if c_train.ndim == 4 else torch.zeros(2, params.Nx, 1)  # Synthetic data case: c_train has shape (Nt, 2, Nx, Ny)

    model = Flux(c0, params, ret_funs=[create_mlp(), None])
    trainer = Trainer(model, finn_dir, **params.to_dict())
    trainer.train_model(c0=c0, t=t_train, data=c_train)

    finn_dir.done_marker_path.touch()
