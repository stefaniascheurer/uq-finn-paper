import numpy as np
import pandas as pd

from pi3nn.config import Params
from pi3nn.io import PI3NNDir
from finn.io import FINNDir, load_exp_data_numpy
import finn.engine as feng
from finn.utils import compute_core1_btc, compute_core2B_profile
from ddb.io import DDBDir
from ddb.config import Params


def mix_quantiles(ddb_dir: DDBDir, pi3nn_dir: PI3NNDir, t_train: np.ndarray, params: Params) -> np.ndarray:
    """Mix quantiles from PI3NN results to create mixed quantile datasets for DDB.

    Args:
        ddb_dir: The DDB directory to save mixed quantiles.
        pi3nn_dir: The PI3NN directory containing quantile results.
        t_train: Training time points.
        params: DDB parameters.
    """
    # Load quantiles from DDB directory into one array
    quantiles_data = pi3nn_dir.iter_quantiles()
    quantiles = np.zeros((len(t_train), len(quantiles_data.keys())))

    for idx, quantile in enumerate(sorted(quantiles_data.keys())):
        quantiles[:, idx] = quantiles_data[quantile]

    np.save(ddb_dir.quantiles_path, quantiles)

    # Mix quantiles
    mixed_quantiles = np.zeros((len(t_train), params.n_mixed_quantiles))
    n_t = np.arange(len(t_train))

    for mq in np.arange(params.n_mixed_quantiles):
        quantile_idxs = np.random.randint(0, len(quantiles_data.keys()), size=len(n_t))
        mixed_quantiles[:, mq] = quantiles[n_t, quantile_idxs]

    np.save(ddb_dir.mixed_quantiles_path, mixed_quantiles)

    return mixed_quantiles


def finn_per_mixed_quantile(ddb_dir: DDBDir, t_train: np.ndarray, mixed_quantiles: np.ndarray, **core2_cfg):
    """Train a FINN model for each mixed quantile dataset.

    Args:
        ddb_dir: The DDB directory containing mixed quantiles.
        t_train: Training time points.
        mixed_quantiles: The mixed quantile datasets.
        **core2_cfg: Additional configuration for core 2 FINN training.
    """
    for mq in np.arange(mixed_quantiles.shape[1]):

        finn_dir = FINNDir(ddb_dir.finn_mixed_quantiles_dir / f"mq_{mq}/")
        data = pd.DataFrame({"time": t_train, "c_diss": mixed_quantiles[:, mq]})

        feng.setup_and_train_model(finn_dir, data, random_seed=True, **core2_cfg)


def postprocess_finn_results(ddb_dir: DDBDir):
    """Postprocess FINN results for each mixed quantile dataset for visualization.

    Args:
        ddb_dir: The DDB directory containing FINN results.
    """
    finn_results = ddb_dir.iter_finn_mixed_quantiles()

    core1_exp = load_exp_data_numpy("Core 1")
    core2_exp = load_exp_data_numpy("Core 2")
    core2b_exp = load_exp_data_numpy("Core 2B")

    core1 = np.zeros((core1_exp.shape[1], len(finn_results)))
    core2 = np.zeros((core2_exp.shape[1], len(finn_results)))
    core2b = np.zeros((core2b_exp.shape[1], len(finn_results)))
    ret_x = np.load(ddb_dir.finn_mixed_quantiles_dir / finn_results[0] / "c_for_ret.npy")
    ret_y = np.zeros((len(ret_x), len(finn_results)))

    for idx, finn_result in enumerate(finn_results):
        finn_dir = FINNDir(ddb_dir.finn_mixed_quantiles_dir / finn_result / "")
        c_btc_pred = finn_dir.best_pred_c_btc
        ret = finn_dir.best_ret

        core2[:, idx] = c_btc_pred
        ret_y[:, idx] = ret[1]

        # Interpolate to experimental time points
        core1[:, idx] = compute_core1_btc(np.squeeze(ret_x), np.squeeze(ret[1]))
        core2b[:, idx] = compute_core2B_profile(np.squeeze(ret_x), np.squeeze(ret[1]))

    np.save(ddb_dir.core1_path, core1)
    np.save(ddb_dir.core2_path, core2)
    np.save(ddb_dir.core2b_path, core2b)
    np.save(ddb_dir.ret_x_path, ret_x)
    np.save(ddb_dir.ret_y_path, ret_y)


def setup_and_train_ddb(ddb_dir: DDBDir, pi3nn_dir: PI3NNDir, **core2_cfg):
    """Setup and train DDB using mixed quantiles from PI3NN results.
    Args:
        ddb_dir: The DDB directory to save results.
        pi3nn_dir: The PI3NN directory containing quantile results.
        **core2_cfg: Additional configuration for core 2 FINN training.
    """
    if not ddb_dir.c_train_path.exists():
        np.save(ddb_dir.c_train_path, np.load(pi3nn_dir.c_data_path))
    c_train = np.load(ddb_dir.c_train_path)

    if not ddb_dir.t_train_path.exists():
        np.save(ddb_dir.t_train_path, np.load(pi3nn_dir.t_path))
    t_train = np.load(ddb_dir.t_train_path)

    params = Params()

    mixed_quantiles = mix_quantiles(ddb_dir, pi3nn_dir, t_train, params)
    finn_per_mixed_quantile(ddb_dir, t_train, mixed_quantiles, **core2_cfg)
    postprocess_finn_results(ddb_dir)
