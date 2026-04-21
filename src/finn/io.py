import dataclasses
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Literal


@dataclasses.dataclass(frozen=True)
class FINNDir:
    """Handles directory paths and file loading for a FINN instance.

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
    def c_train_path(self) -> Path:
        return self.path / "c_train.npy"

    @property
    def c_for_ret_path(self) -> Path:
        return self.path / "c_for_ret.npy"

    @property
    def ckpt_path(self) -> Path:
        return self.path / "ckpt.pt"

    @property
    def training_dir(self) -> Path:
        p = self.path / "training/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    @property
    def ret_dir(self) -> Path:
        p = self.path / "ret/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    @property
    def btc_dir(self) -> Path:
        p = self.path / "btc/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    def get_pred_ret_path(self, epoch: int) -> Path:
        return self.ret_dir / f"ret_pred_{epoch}.npy"

    def get_pred_c_btc_path(self, epoch: int) -> Path:
        return self.btc_dir / f"c_btc_pred_{epoch}.npy"

    def get_pred_c_full_path(self, epoch: int) -> Path:
        return self.training_dir / f"c_full_pred_{epoch}.npy"

    def get_D_eff_path(self, epoch: int) -> Path:
        return self.training_dir / f"D_eff_{epoch}.npy"

    def get_cauchy_mult_path(self, epoch: int) -> Path:
        return self.training_dir / f"cauchy_mult_{epoch}.npy"

    def get_p_exp_path(self, epoch: int) -> Path:
        return self.training_dir / f"p_exp_{epoch}.npy"

    def load_params(self) -> dict[str, Any]:
        return json.loads(self.params_path.read_text())

    @property
    def n_epochs(self) -> int:
        return len(list(self.ret_dir.glob("ret_pred_*.npy")))

    @property
    def best_epoch(self) -> int:
        mses = []
        data = np.load(self.c_train_path)
        for i in range(self.n_epochs):
            pred = np.load(self.get_pred_c_btc_path(i))
            mses.append(np.square(data[:, 1].squeeze() - pred).mean())
        return int(np.argmin(mses))

    @property
    def best_ret(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.load(self.c_for_ret_path).reshape(-1), np.load(self.get_pred_ret_path(self.best_epoch)).reshape(-1))

    @property
    def best_pred_c_btc(self) -> np.ndarray:
        return np.load(self.get_pred_c_btc_path(self.best_epoch))

    @property
    def loss_path(self) -> Path:
        return self.path / "loss.npy"

    @property
    def done_marker_path(self) -> Path:
        return self.path / "finn_done.marker"

    @property
    def is_done(self) -> bool:
        return self.done_marker_path.exists()


def _load_exp_df(name: Literal["Core 1", "Core 2", "Core 2B"], sheet: int) -> pd.DataFrame:
    """Loads a DataFrame from experimental Excel sheets."""
    base_dir = Path(__file__).parent
    p = base_dir / (f"../../in/experimental_data/data_{name.replace(' ', '').lower()}.xlsx")
    return pd.read_excel(p, index_col=None, header=None, sheet_name=sheet)


def load_exp_data(name: Literal["Core 1", "Core 2", "Core 2B"], physical_model=False, csv: bool = False) -> pd.DataFrame:
    """Loads concentration or breakthrough data from experimental Excel sheets or csv."""
    if csv:
        base_dir = Path(__file__).parent
        p = base_dir / (f"../../in/experimental_data/{name.replace(' ', '').lower()}.csv")
        df = pd.read_csv(p, index_col=None)
    else:
        df = _load_exp_df(name, sheet=2 if physical_model else 0)
        if name == "Core 2B":
            df.columns = ["x", "c_tot"]
            df["c_tot"] /= 1000.0
        else:
            df.columns = ["time", "c_diss"]
            df["c_diss"] /= 1000.0
    return df


def load_exp_data_numpy(name: Literal["Core 1", "Core 2", "Core 2B"], physical_model: bool = False, csv: bool = False):
    """Loads concentration or breakthrough data from experimental Excel sheets as numpy arrays."""
    return load_exp_data(name, physical_model, csv).to_numpy().T


def load_exp_cfg(name: Literal["Core 1", "Core 2", "Core 2B"]) -> dict[str, Any]:
    """Loads configuration metadata from experimental Excel sheets."""
    df = _load_exp_df(name, sheet=1).dropna(how="all")
    return df[[0, 1]].set_index(0, drop=True).to_dict()[1]
