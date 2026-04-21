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
    def c_train_path(self) -> Path:
        return self.path / "c_train.npy"

    @property
    def quantiles_path(self) -> Path:
        return self.path / "core2_quantiles.npy"

    @property
    def mixed_quantiles_path(self) -> Path:
        return self.path / "core2_mixed_quantiles.npy"

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
    def core1_path(self) -> Path:
        return self.path / "core1.npy"

    @property
    def core2_path(self) -> Path:
        return self.path / "core2.npy"

    @property
    def core2b_path(self) -> Path:
        return self.path / "core2b.npy"

    @property
    def ret_x_path(self) -> Path:
        return self.path / "ret_x.npy"

    @property
    def ret_y_path(self) -> Path:
        return self.path / "ret_y.npy"
