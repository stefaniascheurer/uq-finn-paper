import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class DDBDir:
    """Handles directory paths and file loading for a DDB instance.

    Attributes:
        path: The root directory path for the DDB instance.
    """
    path: Path

    def __post_init__(self):
        self.path.mkdir(exist_ok=True, parents=True)

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
    def quantiles_path(self) -> Path:
        return self.path / "u_quantiles.npy"

    @property
    def mixed_quantiles_path(self) -> Path:
        return self.path / "u_mixed_quantiles.npy"

    @property
    def finn_mixed_quantiles_dir(self) -> Path:
        p = self.path / "finn_mixed_quantiles/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    def iter_finn_mixed_quantiles(self) -> list[str]:
        """Iterate over available FINN mixed quantile result directories."""
        # Assumes self.directory_path holds the path to search
        path = self.finn_mixed_quantiles_dir
        return [f.name for f in path.iterdir() if f.is_dir()]

    @property
    def u_train_test_path(self) -> Path:
        return self.path / "u_train_test_samples.npy"

    @property
    def u_in_dis_test_path(self) -> Path:
        return self.path / "u_in-test_samples.npy"

    @property
    def u_out_dis_test_path(self) -> Path:
        return self.path / "u_out-test_samples.npy"

    @property
    def learned_x_path(self) -> Path:
        return self.path / "learned_x.npy"

    @property
    def learned_y_path(self) -> Path:
        return self.path / "learned_y.npy"
