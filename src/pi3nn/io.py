import dataclasses
from pathlib import Path
import numpy as np


@dataclasses.dataclass(frozen=True)
class PI3NNDir():
    """Handles directory paths and file loading for a PI3NN instance.

    Attributes:
        path: The root directory path for the PI3NN instance.
    """
    path: Path

    def __post_init__(self):
        self.path.mkdir(exist_ok=True, parents=True)

    @property
    def t_path(self) -> Path:
        return self.path / "t.npy"

    @property
    def c_data_path(self) -> Path:
        return self.path / "c_data.npy"

    @property
    def c_pred_mean_path(self) -> Path:
        return self.path / "c_pred_mean.npy"

    @property
    def c_pred_median_path(self) -> Path:
        return self.path / "c_pred_median.npy"

    @property
    def done_marker_path(self) -> Path:
        return self.path / "pi3nn_done.marker"

    @property
    def is_done(self) -> bool:
        return self.done_marker_path.exists()

    @property
    def quantiles_dir(self) -> Path:
        p = self.path / "quantiles/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    @property
    def upper_dir(self) -> Path:
        p = self.path / "upper/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    @property
    def lower_dir(self) -> Path:
        p = self.path / "lower/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    @property
    def c_pred_upper_res_path(self) -> Path:
        return self.path / "c_pred_upper_res.npy"

    @property
    def c_pred_lower_res_path(self) -> Path:
        return self.path / "c_pred_lower_res.npy"

    def get_quantile_path(self, quantile: float) -> Path:
        """Get the file path for a specific quantile's predictions."""
        quantile_str = f"{quantile:g}".replace("0.", "0-").replace("1.", "1-")
        return self.quantiles_dir / f"{quantile_str}.npy"

    def iter_quantiles(self) -> dict[float, np.ndarray]:
        """Iterate over and load all stored quantile predictions."""
        quantiles: dict[float, np.ndarray] = {}

        for path in self.quantiles_dir.glob("*.npy"):
            quantile = float(path.stem.replace("0-", "0.").replace("1-", "1."))
            quantiles[quantile] = np.load(path)

        return quantiles
