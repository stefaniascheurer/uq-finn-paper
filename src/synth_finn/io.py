import dataclasses
import json
import numpy as np
from pathlib import Path
from typing import Any, Literal, Optional


@dataclasses.dataclass(frozen=True)
class FINNDir:
    """Handles directory paths and file loading for FINN instance.

    Attributes:
        path: The root directory path for the instance.
    """
    path: Path

    def __post_init__(self):
        self.path.mkdir(exist_ok=True, parents=True)

    @property
    def params_path(self) -> Path:
        return self.path / "params.json"

    @property
    def t_train_path(self) -> Path:
        return self.path / "t_train.npy"

    @property
    def x_train_path(self) -> Path:
        return self.path / "x_train.npy"

    @property
    def u_train_path(self) -> Path:
        return self.path / "u_train.npy"

    @property
    def u_for_learned_path(self) -> Path:
        return self.path / "u_for_learned.npy"

    @property
    def ckpt_path(self) -> Path:
        return self.path / "ckpt.pt"

    @property
    def training_dir(self) -> Path:
        p = self.path / "training/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    @property
    def learned_dir(self) -> Path:
        p = self.path / "learned/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    def get_pred_learned_path(self, epoch: int) -> Path:
        return self.learned_dir / f"learned_pred_{epoch}.npy"

    def get_pred_u_train_path(self, epoch: int) -> Path:
        return self.training_dir / f"u_train_pred_{epoch}.npy"

    def get_pred_u_train_test_path(self, epoch: int) -> Path:
        return self.training_dir / f"u_train-test_pred_{epoch}.npy"

    def get_pred_u_in_dis_test_path(self, epoch: int) -> Path:
        return self.training_dir / f"u_in-dis-test_pred_{epoch}.npy"

    def get_pred_u_out_dis_test_path(self, epoch: int) -> Path:
        return self.training_dir / f"u_out-dis-test_pred_{epoch}.npy"

    def load_params(self) -> dict[str, Any]:
        return json.loads(self.params_path.read_text())

    @property
    def n_epochs(self) -> int:
        return len(list(self.learned_dir.glob("learned_pred_*.npy")))

    @property
    def best_epoch(self) -> int:
        mses = []
        data = np.load(self.u_train_path)
        if data.ndim > 1:
            data = data[-1]
        for i in range(self.n_epochs):
            pred = np.load(self.get_pred_u_train_test_path(i))
            mses.append(np.square(data.squeeze() - pred).mean())
        return int(np.argmin(mses))

    @property
    def best_learned(self) -> tuple[np.ndarray, np.ndarray]:
        return np.load(self.get_pred_learned_path(self.best_epoch)).reshape(-1)

    @property
    def best_pred_u_train(self) -> np.ndarray:
        return np.load(self.get_pred_u_train_path(self.best_epoch))

    @property
    def best_pred_u_train_test(self) -> np.ndarray:
        return np.load(self.get_pred_u_train_test_path(self.best_epoch))

    @property
    def best_pred_u_in_dis_test(self) -> np.ndarray:
        return np.load(self.get_pred_u_in_dis_test_path(self.best_epoch))

    @property
    def best_pred_u_out_dis_test(self) -> np.ndarray:
        return np.load(self.get_pred_u_out_dis_test_path(self.best_epoch))

    @property
    def loss_path(self) -> Path:
        return self.path / "loss.npy"

    @property
    def done_marker_path(self) -> Path:
        return self.path / "finn_done.marker"

    @property
    def is_done(self) -> bool:
        return self.done_marker_path.exists()


def load_synthetic_data(mode: Literal["train", "train-test", "in-dis-test", "out-dis-test"]) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Loads synthetic data.

    Returns:
        For "train": (t, x, u) where t is time grid, x is spatial grid, and u is the full field u(x, t).
        For test modes: (x, u) where x is spatial grid and u is the field at the test time.

    """

    base_dir = Path(__file__).parent
    p = base_dir / (f"../../in/synthetic_data")

    if mode == "train":
        t = np.load(p / "t_train.npy")
        x = np.load(p / "x_train.npy")
        u = np.load(p / "u_train.npy")
        return t, x, u
    elif mode == "train-test":
        x = np.load(p / "x_train-test.npy")
        u = np.load(p / "u_train-test.npy")
    elif mode == "in-dis-test":
        x = np.load(p / "x_in-test.npy")
        u = np.load(p / "u_in-test.npy")
    else:
        x = np.load(p / "x_out-test.npy")
        u = np.load(p / "u_out-test.npy")

    return x, u
