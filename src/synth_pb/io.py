import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class PBDir:
    """Handles directory paths and file loading for a PB instance.

    Attributes:
        path: The root directory path for the PB instance.
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
    def bootstraps_path(self) -> Path:
        return self.path / "u_bootstraps.npy"

    @property
    def finn_bootstraps_dir(self) -> Path:
        p = self.path / "finn_bootstraps/"
        p.mkdir(exist_ok=True, parents=True)
        return p

    def iter_finn_bootstraps(self) -> list[str]:
        """Iterate over available FINN bootstrap result directories."""
        # Assumes self.directory_path holds the path to search
        path = self.finn_bootstraps_dir
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
