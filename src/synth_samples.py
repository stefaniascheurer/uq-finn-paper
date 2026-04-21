from pathlib import Path
import dataclasses
import numpy as np
from scipy.integrate import trapezoid


@dataclasses.dataclass
class DDBSamples:
    """Container for storing and visualizing data-driven bootstrap samples and learned function results."""

    x_train: np.ndarray
    u_train_test: np.ndarray
    u_in_dis_test: np.ndarray
    u_out_dis_test: np.ndarray
    learned_x: np.ndarray
    learned_y: np.ndarray
    quantiles: np.ndarray
    mixed_quantiles: np.ndarray

    @classmethod
    def from_dir(cls, p: str):
        """Loads samples."""

        def try_load(p: Path):
            try:
                return np.load(p)
            except FileNotFoundError:
                return np.array([[]])
        p = Path(p).resolve()
        return cls(
            x_train=try_load(p / "x_train.npy"),
            u_train_test=try_load(p / "u_train_test_samples.npy"),
            u_in_dis_test=try_load(p / "u_in-test_samples.npy"),
            u_out_dis_test=try_load(p / "u_out-test_samples.npy"),
            learned_x=try_load(p / "learned_x.npy"),
            learned_y=try_load(p / "learned_y.npy"),
            quantiles=try_load(p / "u_quantiles.npy"),
            mixed_quantiles=try_load(p / "u_mixed_quantiles.npy")
        )

    def to_dir(self, p: str):
        """Saves current samples to the specified directory."""

        p = Path(p).resolve()
        np.save(p / "u_train_test_samples.npy", self.u_train_test)
        np.save(p / "u_in-test_samples.npy", self.u_in_dis_test)
        np.save(p / "u_out-test_samples.npy", self.u_out_dis_test)
        np.save(p / "learned_x.npy", self.learned_x)
        np.save(p / "learned_y.npy", self.learned_y)
        np.save(p / "u_quantiles.npy", self.quantiles)
        np.save(p / "u_mixed_quantiles.npy", self.mixed_quantiles)

    def calc_areas(self, interval=0.95):
        areas = {}

        upper = np.percentile(self.u_train_test.T, int((1 + interval) / 2 * 100), axis=0)
        lower = np.percentile(self.u_train_test.T, int((1 - interval) / 2 * 100), axis=0)
        area = trapezoid(upper - lower, x=self.x_train)
        areas["u_train_test"] = area

        upper = np.percentile(self.u_in_dis_test.T, int((1 + interval) / 2 * 100), axis=0)
        lower = np.percentile(self.u_in_dis_test.T, int((1 - interval) / 2 * 100), axis=0)
        area = trapezoid(upper - lower, x=self.x_train)
        areas["u_in_dis_test"] = area

        upper = np.percentile(self.u_out_dis_test.T, int((1 + interval) / 2 * 100), axis=0)
        lower = np.percentile(self.u_out_dis_test.T, int((1 - interval) / 2 * 100), axis=0)
        area = trapezoid(upper - lower, x=self.x_train)
        areas["u_out_dis_test"] = area

        upper = np.percentile(self.learned_y.T, int((1 + interval) / 2 * 100), axis=0)
        lower = np.percentile(self.learned_y.T, int((1 - interval) / 2 * 100), axis=0)
        area = trapezoid(upper - lower, x=self.learned_x.squeeze())
        areas["learned_y"] = area

        return areas


@dataclasses.dataclass
class PBSamples:
    """Container for storing and visualizing parametric bootstrap samples and learned function results."""

    x_train: np.ndarray
    u_train_test: np.ndarray
    u_in_dis_test: np.ndarray
    u_out_dis_test: np.ndarray
    learned_x: np.ndarray
    learned_y: np.ndarray
    bootstraps: np.ndarray

    @classmethod
    def from_dir(cls, p: str):
        """Loads samples."""

        def try_load(p: Path):
            try:
                return np.load(p)
            except FileNotFoundError:
                return np.array([[]])
        p = Path(p).resolve()
        return cls(
            x_train=try_load(p / "x_train.npy"),
            u_train_test=try_load(p / "u_train_test_samples.npy"),
            u_in_dis_test=try_load(p / "u_in-test_samples.npy"),
            u_out_dis_test=try_load(p / "u_out-test_samples.npy"),
            learned_x=try_load(p / "learned_x.npy"),
            learned_y=try_load(p / "learned_y.npy"),
            bootstraps=try_load(p / "u_bootstraps.npy"),
        )

    def to_dir(self, p: str):
        """Saves current samples to the specified directory."""

        p = Path(p).resolve()
        np.save(p / "u_train_test_samples.npy", self.u_train_test)
        np.save(p / "u_in-test_samples.npy", self.u_in_dis_test)
        np.save(p / "u_out-test_samples.npy", self.u_out_dis_test)
        np.save(p / "learned_x.npy", self.learned_x)
        np.save(p / "learned_y.npy", self.learned_y)
        np.save(p / "u_bootstraps.npy", self.bootstraps)

    def calc_areas(self, interval=0.95):
        areas = {}

        upper = np.percentile(self.u_train_test.T, int((1 + interval) / 2 * 100), axis=0)
        lower = np.percentile(self.u_train_test.T, int((1 - interval) / 2 * 100), axis=0)
        area = trapezoid(upper - lower, x=self.x_train)
        areas["u_train_test"] = area

        upper = np.percentile(self.u_in_dis_test.T, int((1 + interval) / 2 * 100), axis=0)
        lower = np.percentile(self.u_in_dis_test.T, int((1 - interval) / 2 * 100), axis=0)
        area = trapezoid(upper - lower, x=self.x_train)
        areas["u_in_dis_test"] = area

        upper = np.percentile(self.u_out_dis_test.T, int((1 + interval) / 2 * 100), axis=0)
        lower = np.percentile(self.u_out_dis_test.T, int((1 - interval) / 2 * 100), axis=0)
        area = trapezoid(upper - lower, x=self.x_train)
        areas["u_out_dis_test"] = area

        upper = np.percentile(self.learned_y.T, int((1 + interval) / 2 * 100), axis=0)
        lower = np.percentile(self.learned_y.T, int((1 - interval) / 2 * 100), axis=0)
        area = trapezoid(upper - lower, x=self.learned_x.squeeze())
        areas["learned_y"] = area

        return areas
