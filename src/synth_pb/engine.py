import numpy as np
import pandas as pd
from pathlib import Path

from synth_finn.io import FINNDir, load_synthetic_data
import synth_finn.engine as feng
from synth_finn.utils import compute_u
from synth_pb.io import PBDir
from synth_pb.config import Params


def boostrap(pb_dir: PBDir, params: Params) -> np.ndarray:
    """Generate bootstrapped dataset for PB.

    Args:
        pb_dir: The PB directory to save bootstrapped datasets.
        params: PB parameters.
    """

    np.random.seed(42)

    # Load clean data
    current_dir = Path(__file__).parent
    u_full = np.load(current_dir / "../../in/synthetic_data/u_full.npy").T
    u_clean = np.tile(u_full[:, 200].reshape(-1, 1), (1, params.n_bootstraps))
    noise = np.random.normal(0, params.noise, u_clean.shape)

    bootstraps = u_clean + noise
    np.save(pb_dir.bootstraps_path, bootstraps)

    return bootstraps


def finn_per_bootstrap(pb_dir: PBDir, t_train: np.ndarray, x_train: np.ndarray, bootstraps: np.ndarray):
    """Train a FINN model for each mixed quantile dataset.

    Args:
        pb_dir: The DDB directory containing mixed quantiles.
        t_train: Training time points.
        x_train: Training spatial points.
        bootstraps: The bootstrapped datasets.
    """
    for b in np.arange(bootstraps.shape[1]):

        finn_dir = FINNDir(pb_dir.finn_bootstraps_dir / f"b_{b}/")
        u_train = bootstraps[:, b]

        feng.setup_and_train_model(finn_dir, t_train, x_train, u_train, random_seed=True)


def postprocess_finn_results(pb_dir: PBDir, threshold: list[float] = [-1.5, 1.5]):
    """Postprocess FINN results for bootstrap dataset for visualization.

    Args:
        pb_dir: The PB directory containing FINN results.
        threshold: The threshold for identifying non-converged samples.
    """
    finn_results = pb_dir.iter_finn_bootstraps()

    x, u_train_test_data = load_synthetic_data("train-test")
    x, u_in_dis_test_data = load_synthetic_data("in-dis-test")
    x, u_out_dis_test_data = load_synthetic_data("out-dis-test")

    u_train_test = np.zeros((len(u_train_test_data), len(finn_results)))
    learned_x = np.load(pb_dir.finn_bootstraps_dir / finn_results[0] / "u_for_learned.npy")
    learned_y = np.zeros((len(learned_x), len(finn_results)))

    for idx, finn_result in enumerate(finn_results):

        finn_dir = FINNDir(pb_dir.finn_bootstraps_dir / finn_result / "")
        best_pred_u_train_test = finn_dir.best_pred_u_train_test
        learned = finn_dir.best_learned

        u_train_test[:, idx] = best_pred_u_train_test
        learned_y[:, idx] = learned

    # Check for convergence issues
    converged_idxs = []
    for l in range(learned_y.shape[1]):
        learned = learned_y[:, l]
        if learned[0] > -0.5 or np.min(learned) < threshold[0] or np.max(learned) > threshold[1] or learned[0] > learned[-1] or not np.all(np.diff(learned) >= 0):
            converged_idxs.append(0)
        else:
            converged_idxs.append(1)

    converged_idxs = np.array(converged_idxs, dtype=bool)
    u_train_test = u_train_test[:, converged_idxs]
    learned_y = learned_y[:, converged_idxs]

    u_in_dis_test = np.zeros((len(u_in_dis_test_data), learned_y.shape[1]))
    u_out_dis_test = np.zeros((len(u_out_dis_test_data), learned_y.shape[1]))

    for idx in range(learned_y.shape[1]):
        u_in_dis_test[:, idx] = compute_u(np.squeeze(learned_x), np.squeeze(learned_y[:, idx]), mode="in-dis-test")
        u_out_dis_test[:, idx] = compute_u(np.squeeze(learned_x), np.squeeze(learned_y[:, idx]), mode="out-dis-test")

    np.save(pb_dir.u_train_test_path, u_train_test)
    np.save(pb_dir.u_in_dis_test_path, u_in_dis_test)
    np.save(pb_dir.u_out_dis_test_path, u_out_dis_test)
    np.save(pb_dir.learned_x_path, learned_x)
    np.save(pb_dir.learned_y_path, learned_y)


def setup_and_train_pb(pb_dir: PBDir, finn_dir: FINNDir):
    """Setup and train PB fresults after generating parametric bootstraps.
    Args:
        pb_dir: The PB directory to save results.
        finn_dir: The FINN directory containing quantile results.
    """

    if not pb_dir.u_train_path.exists():
        np.save(pb_dir.u_train_path, np.load(finn_dir.u_train_path))
    u_train = np.load(pb_dir.u_train_path)

    if not pb_dir.t_train_path.exists():
        np.save(pb_dir.t_train_path, np.load(finn_dir.t_train_path))
    t_train = np.load(pb_dir.t_train_path)

    if not pb_dir.x_train_path.exists():
        np.save(pb_dir.x_train_path, np.load(finn_dir.x_train_path))
    x_train = np.load(pb_dir.x_train_path)

    params = Params()

    bootstraps = boostrap(pb_dir, params)
    finn_per_bootstrap(pb_dir, t_train, x_train, bootstraps)
    postprocess_finn_results(pb_dir, threshold=[-1.5, 1.5])
