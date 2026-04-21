import time
import random
import numpy as np
import torch
import json
from torchdiffeq import odeint

from synth_finn.io import FINNDir, load_synthetic_data
from synth_finn.config import Params
from synth_finn.model import Flux_AllenCahn, Flux_Burgers, create_mlp


def random_logunif(min_val: float, max_val: float) -> float:
    """Samples a random value from a log-uniform distribution between min_val and max_val."""
    log_v = np.random.uniform(np.log10(min_val), np.log10(max_val))
    return 10**log_v


class Trainer:
    """Orchestrates the training for FINN of synthetic data, learning a function f(u)."""

    def __init__(
        self,
        model,
        finn_dir: FINNDir,
        continue_training: bool = False,
        n_epochs: int = 30,
        error_mult: float = 1e5,
        phys_mult: float = 1e1,
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
        self.use_adam_optim = use_adam_optim

        if use_adam_optim:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
        else:
            self.optimizer = torch.optim.LBFGS(model.parameters(), lr=start_lr)

        self.start_epoch = 0
        self.train_losses = []
        self.best_loss = np.inf

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

    def train_model(self, *, u0: torch.Tensor, t: torch.Tensor, data: torch.Tensor) -> None:
        """Trains the model to fit the provided data.

        Args:
            u0: Initial tensor.
            t: Time tensor.
            data: Training data tensor.
        """

        u_for_learned = torch.linspace(-1.0, 1.0, 100).view(-1, 1)
        np.save(self.finn_dir.u_for_learned_path, u_for_learned)
        criterion = torch.nn.MSELoss()
        loss_phys_fac = self.phys_mult
        loss_mse_fac = self.error_mult

        u0 = u0.detach()
        t = t.detach()
        data = data.detach()

        def closure():

            # u_for_learned = torch.linspace(-1.5, 1.5, 100).view(-1, 1).detach()

            # Set the model to training mode
            self.model.train()

            # Reset the gradient buffer (set to 0)
            self.optimizer.zero_grad()

            # Calculate the model prediction (full field solution)
            u_pred = odeint(self.model, u0, t)  # , rtol=1e-5, atol=1e-6)
            self.latest_u_pred = u_pred.detach().cpu().numpy()
            self.latest_u_train_test_pred = u_pred.squeeze()[-1].detach().cpu().numpy()

            # MSE against synthetic u_train
            # Squeeze to [Nt, Nx] to match data, use only last timestep to perform UQ (not possible for > 1D data)
            # loss_mse = criterion(u_pred.squeeze(), data.squeeze())
            if data.squeeze().dim() > 1:
                loss_mse = criterion(u_pred[-1].squeeze(), data[-1].squeeze())
            else:
                loss_mse = criterion(u_pred[-1].squeeze(), data.squeeze())

            self.latest_mse_loss = loss_mse.detach().numpy()

            # # Physical Regularization
            if self.phys_mult > 0:
                if self.model.cfg.model == "allen_cahn":
                    u_roots = torch.tensor([[-1.0], [0.0], [1.0]], device=u_pred.device)
                    react_roots = self.model.flux_modules[0].coeff_nn(u_roots) * self.model.flux_modules[0].p_mult
                    loss_physical = torch.mean(react_roots**2)
                elif self.model.cfg.model == "burgers":
                    u_check = torch.linspace(0.0, 1.0, 100).view(-1, 1)
                    adv_pos = self.model.flux_modules[0].coeff_nn(u_check)
                    adv_neg = self.model.flux_modules[0].coeff_nn(-u_check)
                    loss_physical = torch.mean((adv_pos + adv_neg)**2)

                self.latest_physical_loss = loss_physical.detach().numpy()
                total_loss = loss_mse * loss_mse_fac + loss_physical * loss_phys_fac
            else:
                total_loss = loss_mse * loss_mse_fac

            total_loss.backward()

            return total_loss

        lr_min, lr_max, decay_factor, T_0 = 1e-5, self.start_lr, 0.8, 10
        num_restarts = 30

        learning_rates = np.concatenate([lr_min + 0.5 * (lr_max * decay_factor**i - lr_min) * (1 + np.cos(np.linspace(0, np.pi, T_0))) for i in range(num_restarts)])

        # Iterate until maximum epoch number is reached
        for epoch in range(self.start_epoch, self.n_epochs):
            for g in self.optimizer.param_groups:
                if self.use_adam_optim:
                    g["lr"] = self.start_lr
                else:
                    g["lr"] = learning_rates[epoch]

                dt = time.time()

            # Update the model parameters and record the loss value
            loss = self.optimizer.step(closure)
            self.train_losses.append(loss.item())

            dt = time.time() - dt

            print(
                f"It {epoch+1:>3}/{self.n_epochs}"
                f" | mse = {self.latest_mse_loss:.2e}"
                f" | loss = {self.train_losses[-1]:.2e}"
                f" | dt = {dt:.1f}s"
                # f" | lr = {learning_rates[epoch]:.1e}"
                # f" | loss_mse = {self.latest_mse_loss:.1e}"
                # f" | loss_phys = {self.latest_physical_loss:.1e}"
            )

            if self.model.cfg.model == "allen_cahn":
                learned_pred = self.model.flux_modules[0].coeff_nn(u_for_learned) * self.model.flux_modules[0].p_mult
            else:
                learned_pred = self.model.flux_modules[0].coeff_nn(u_for_learned)

            np.save(self.finn_dir.get_pred_learned_path(epoch), learned_pred.detach().numpy())
            np.save(self.finn_dir.get_pred_u_train_test_path(epoch), self.latest_u_train_test_pred)
            np.save(self.finn_dir.get_pred_u_train_path(epoch), self.latest_u_pred)

            if (epoch + 1) % 2 == 0:
                self.save_model_to_file(epoch)

        np.save(self.finn_dir.loss_path, self.train_losses)

    def save_model_to_file(self, epoch: int) -> None:
        """Saves the current model and optimizer state to a checkpoint file."""
        torch.save({"epoch": epoch + 1, "state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "loss_train": self.train_losses}, self.finn_dir.ckpt_path)


def setup_and_train_model(finn_dir: FINNDir, t_train: np.ndarray, x_train: np.ndarray, u_train: np.ndarray, random_seed: bool = False):
    """Setup and train the FINN model.

    Args:
        finn_dir: The FINN directory to save results.
        t_train: Training time points.
        x_train: Training spatial points.
        u_train: Training data for the unknown variable.
        random_seed: Whether to randomize the random seed for training.
    """

    if finn_dir.is_done:
        return

    if not finn_dir.u_train_path.exists():
        np.save(finn_dir.u_train_path, np.squeeze(u_train))
    u_train = torch.Tensor(np.load(finn_dir.u_train_path))

    if not finn_dir.t_train_path.exists():
        np.save(finn_dir.t_train_path, t_train)
    t_train = torch.Tensor(np.load(finn_dir.t_train_path))

    if not finn_dir.x_train_path.exists():
        np.save(finn_dir.x_train_path, x_train)

    if random_seed:
        seed = time.time_ns() % 10**9
    else:
        seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    params = Params.from_dict()
    finn_dir.params_path.write_text(json.dumps(params.to_dict()))

    # u0 shape [1, Nx, 1] - First time step of u_train
    if u_train.dim() == 2:
        u0 = torch.FloatTensor(u_train[0]).unsqueeze(0).unsqueeze(-1)
    else:
        _, _, u = load_synthetic_data("train")
        u0 = u[0]
        u0 = torch.FloatTensor(u0).unsqueeze(0).unsqueeze(-1)

    if params.model == "allen_cahn":
        model = Flux_AllenCahn(u0, params, react_fun=[create_mlp()])
    else:
        model = Flux_Burgers(u0, params, adv_fun=[create_mlp()])

    trainer = Trainer(model, finn_dir, **params.to_dict())
    if u_train.dim() == 2:
        trainer.train_model(u0=u0, t=t_train, data=u_train)
    else:
        trainer.train_model(u0=u0, t=t_train, data=u_train.unsqueeze(0))

    finn_dir.done_marker_path.touch()
