import numpy as np
import pandas as pd

from synth_pi3nn.config import Params
from synth_pi3nn.io import PI3NNDir
from synth_finn.io import FINNDir, load_synthetic_data
import synth_finn.engine as feng
from synth_finn.utils import compute_u
from synth_ddb.io import DDBDir
from synth_ddb.config import Params


def mix_quantiles(ddb_dir: DDBDir, pi3nn_dir: PI3NNDir, x_train: np.ndarray, params: Params) -> np.ndarray:
    """Mix quantiles from PI3NN results to create mixed quantile datasets for DDB.

    Args:
        ddb_dir: The DDB directory to save mixed quantiles.
        pi3nn_dir: The PI3NN directory containing quantile results.
        t_train: Training spatial points.
        params: DDB parameters.
    """
    # Load quantiles from DDB directory into one array
    quantiles_data = pi3nn_dir.iter_quantiles()
    quantiles = np.zeros((len(x_train), len(quantiles_data.keys())))

    for idx, quantile in enumerate(sorted(quantiles_data.keys())):
        quantiles[:, idx] = quantiles_data[quantile]

    np.save(ddb_dir.quantiles_path, quantiles)

    # Mix quantiles
    mixed_quantiles = np.zeros((len(x_train), params.n_mixed_quantiles))
    n_x = np.arange(len(x_train))

    for mq in np.arange(params.n_mixed_quantiles):
        quantile_idxs = np.random.randint(0, len(quantiles_data.keys()), size=len(n_x))
        mixed_quantiles[:, mq] = quantiles[n_x, quantile_idxs]

    np.save(ddb_dir.mixed_quantiles_path, mixed_quantiles)

    return mixed_quantiles


def finn_per_mixed_quantile(ddb_dir: DDBDir, t_train: np.ndarray, x_train: np.ndarray, mixed_quantiles: np.ndarray):
    """Train a FINN model for each mixed quantile dataset.

    Args:
        ddb_dir: The DDB directory containing mixed quantiles.
        t_train: Training time points.
        x_train: Training spatial points.
        mixed_quantiles: The mixed quantile datasets.
    """
    for mq in np.arange(mixed_quantiles.shape[1]):

        finn_dir = FINNDir(ddb_dir.finn_mixed_quantiles_dir / f"mq_{mq}/")
        u_train = mixed_quantiles[:, mq]

        feng.setup_and_train_model(finn_dir, t_train, x_train, u_train, random_seed=True)


def postprocess_finn_results(ddb_dir: DDBDir, threshold: list[float] = [-1.5, 1.5]):
    """Postprocess FINN results for each mixed quantile dataset for visualization.

    Args:
        ddb_dir: The DDB directory containing FINN results.
        threshold: The threshold for identifying non-converged samples.
    """
    finn_results = ddb_dir.iter_finn_mixed_quantiles()

    x, u_train_test_data = load_synthetic_data("train-test")
    x, u_in_dis_test_data = load_synthetic_data("in-dis-test")
    x, u_out_dis_test_data = load_synthetic_data("out-dis-test")

    u_train_test = np.zeros((len(u_train_test_data), len(finn_results)))
    learned_x = np.load(ddb_dir.finn_mixed_quantiles_dir / finn_results[0] / "u_for_learned.npy")
    learned_y = np.zeros((len(learned_x), len(finn_results)))

    for idx, finn_result in enumerate(finn_results):

        finn_dir = FINNDir(ddb_dir.finn_mixed_quantiles_dir / finn_result / "")
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

    np.save(ddb_dir.u_train_test_path, u_train_test)
    np.save(ddb_dir.u_in_dis_test_path, u_in_dis_test)
    np.save(ddb_dir.u_out_dis_test_path, u_out_dis_test)
    np.save(ddb_dir.learned_x_path, learned_x)
    np.save(ddb_dir.learned_y_path, learned_y)


def setup_and_train_ddb(ddb_dir: DDBDir, pi3nn_dir: PI3NNDir):
    """Setup and train DDB using mixed quantiles from PI3NN results.
    Args:
        ddb_dir: The DDB directory to save results.
        pi3nn_dir: The PI3NN directory containing quantile results.
    """
    if not ddb_dir.u_train_path.exists():
        np.save(ddb_dir.u_train_path, np.load(pi3nn_dir.u_data_path))
    u_train = np.load(ddb_dir.u_train_path)

    if not ddb_dir.t_train_path.exists():
        np.save(ddb_dir.t_train_path, np.load(pi3nn_dir.t_path))
    t_train = np.load(ddb_dir.t_train_path)

    if not ddb_dir.x_train_path.exists():
        np.save(ddb_dir.x_train_path, np.load(pi3nn_dir.x_path))
    x_train = np.load(ddb_dir.x_train_path)

    params = Params()

    mixed_quantiles = mix_quantiles(ddb_dir, pi3nn_dir, x_train, params)
    finn_per_mixed_quantile(ddb_dir, t_train, x_train, mixed_quantiles)
    postprocess_finn_results(ddb_dir, threshold=[-1.5, 1.5])
