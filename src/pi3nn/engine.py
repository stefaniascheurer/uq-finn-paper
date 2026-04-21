import numpy as np
from scipy import optimize
import keras
import dataclasses as dc

from pi3nn.config import Params
from pi3nn.io import PI3NNDir
from pi3nn.model import make_model

from finn.io import FINNDir


@dc.dataclass
class TrainingResult:
    """Container for the results of a neural network training session.

    Attributes:
        model: The trained Keras or ScaledTrainingModel instance.
        x_train: The original input features used for training.
        y_train: The original target values used for training.
        x_eval: The input features used for evaluation/plotting.
        y_eval_pred: Model predictions on the evaluation set.
        y_train_pred: Model predictions on the training set.
        loss_train: Array of loss values recorded per epoch.
        learning_rates: Array of learning rate values used per epoch.
    """
    model: keras.Model
    x_train: np.ndarray
    y_train: np.ndarray
    x_eval: np.ndarray
    y_eval_pred: np.ndarray
    y_train_pred: np.ndarray
    loss_train: np.ndarray
    learning_rates: np.ndarray


def train_model(model: keras.Model, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, params: Params, lr_schedule) -> TrainingResult:
    """Executes the training loop for a given model.

    Args:
        model: The model instance to train.
        x_train: Training input features.
        y_train: Training target values.
        x_eval: Evaluation input features.
        t_params: Instance of TrainingParams (hyperparameters).
        e_params: Instance of ExperimentParams (global settings).
        lr_schedule: A Keras LearningRateSchedule or a callable function.
    """
    if isinstance(model, keras.Model):
        model.compile(loss=params.loss_fun, optimizer=keras.optimizers.get({"class_name": params.optimizer, "config": {"learning_rate": params.initial_learning_rate}}))
        callbacks = [keras.callbacks.LearningRateScheduler(lambda s: float(lr_schedule(s)))]
        if params.stop_early:
            callbacks.append(keras.callbacks.EarlyStopping(monitor="loss", patience=20, restore_best_weights=True, min_delta=0.0005))
        history = model.fit(x_train, y_train, epochs=params.max_epochs, batch_size=params.batch_size or x_train.shape[0], validation_split=params.validation_fraction, callbacks=callbacks, verbose=params.verbose)
        return TrainingResult(model, x_train, y_train, x_eval, model.predict(x_eval), model.predict(x_train), np.array(history.history["loss"]), np.array([lr_schedule(s) for s in range(len(history.history["loss"]))]))
    return TrainingResult(model, x_train, y_train, x_eval, model.predict(x_eval), model.predict(x_train), np.array([]), np.array([]))


def _c_for_bound_quantile(mode: str, y_train: np.ndarray, pred_median: np.ndarray, pred_res: np.ndarray) -> float:
    """Calculates the scaling constant 'c' required to cover 100% of data points.

    Args:
        mode: Direction of the bound ('up' or 'down').
        y_train: Ground truth training targets.
        pred_median: Predicted median values from the mean model.
        pred_res: Predicted absolute residuals from the residual model.
    """
    sign = 1 if mode == "up" else -1

    def bound(c):
        return pred_median + sign * c * pred_res
    c2 = float(0.0 - (sign * (bound(0.0) - y_train)).min() / pred_res[np.argmin(sign * (bound(0.0) - y_train))])
    while not np.all(sign * (bound(c2) - y_train) >= 0):
        c2 += 1e-6
    return c2


def optimize_quantile(mode: str, y_train: np.ndarray, pred_median: np.ndarray, pred_res: np.ndarray, quantile: float) -> float:
    """Finds the optimal scaling constant 'c' for a specific quantile.

    Args:
        mode: Direction of the bound ('up' or 'down').
        y_train: Ground truth training targets.
        pred_median: Predicted median values.
        pred_res: Predicted absolute residuals.
        quantile: The target coverage probability (e.g., 0.95).
    """
    if quantile == 1.0 or quantile == 0.0:
        return _c_for_bound_quantile(mode, y_train, pred_median, pred_res)

    num_outliers = int(len(y_train) * (1 - quantile))

    def objective(c):
        b = pred_median + (c * pred_res if mode == "up" else -c * pred_res)
        return (np.count_nonzero(y_train >= b)) - num_outliers
        # return (np.count_nonzero(y_train >= b) if mode == "up" else np.count_nonzero(y_train <= b)) - num_outliers
    return float(optimize.bisect(objective, 0.0, 1e6, maxiter=1000))


def compute_quantiles(pi3nn_dir: PI3NNDir, mean_finn_dir: FINNDir, quantiles: np.ndarray):
    """
    Perform PI3NN on FINN results for quantiles.

    Args:
        pi3nn_dir: The PI3NN directory to store results.
        mean_finn_dir: The FINN directory containing mean predictions.
        quantiles: Array of quantiles to compute.
    """

    if not pi3nn_dir.is_done:

        # Data
        t = np.load(mean_finn_dir.t_train_path)
        c_data = np.load(mean_finn_dir.c_train_path)

        t_eval = t.reshape(-1, 1)

        np.save(pi3nn_dir.t_path, np.squeeze(t))
        np.save(pi3nn_dir.c_data_path, np.squeeze(c_data))

        # Mean predictions
        c_pred_mean = mean_finn_dir.best_pred_c_btc
        np.save(pi3nn_dir.c_pred_mean_path, c_pred_mean)

        # Residuals & Shift to median
        # Ensuring 50% of data is above and 50% below the center line
        raw_residuals = c_data[:, 1] - c_pred_mean
        median_shift = np.median(raw_residuals)
        c_pred_median = c_pred_mean + median_shift
        np.save(pi3nn_dir.c_pred_median_path, c_pred_median)

        # Train residual models
        # One network for 'up' errors, one for 'down' errors
        params = Params()
        res_results = {}
        for mask, sign, mode in [(raw_residuals < median_shift, -1, "down"), (raw_residuals > median_shift, 1, "up")]:
            t_subset = t[mask.flat].reshape(-1, 1)
            c_res_subset = (sign * (raw_residuals - median_shift))[mask.flat].reshape(-1, 1)

            model = make_model(t_subset, c_res_subset, params.n_neurons_per_layer, params.activation, positivity_method="softplus", l2_reg=params.l2_regularization)
            lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay([10**8], [params.initial_learning_rate, 1e-7])
            res_results[mode] = train_model(model, t_subset, c_res_subset, t_eval, params, lr_schedule)

            if mode == "up":
                upper_dir = pi3nn_dir.upper_dir
                res_results[mode].model.save_weights(upper_dir / "weights.weights.h5")

                np.save(upper_dir / "t_train.npy", res_results[mode].x_train)
                np.save(upper_dir / "c_res_train.npy", res_results[mode].y_train)
                np.save(upper_dir / "c_res_pred_train.npy", res_results[mode].y_train_pred)
                upper_res = res_results[mode].y_eval_pred.squeeze()
                np.save(pi3nn_dir.c_pred_upper_res_path, upper_res)

            else:
                lower_dir = pi3nn_dir.lower_dir
                res_results[mode].model.save_weights(lower_dir / "weights.weights.h5")

                np.save(lower_dir / "t_train.npy", res_results[mode].x_train)
                np.save(lower_dir / "c_res_train.npy", res_results[mode].y_train)
                np.save(lower_dir / "c_res_pred_train.npy", res_results[mode].y_train_pred)
                lower_res = res_results[mode].y_eval_pred.squeeze()
                np.save(pi3nn_dir.c_pred_lower_res_path, lower_res)

        pi3nn_dir.done_marker_path.touch()

    for q in quantiles:
        if not pi3nn_dir.get_quantile_path(q).is_file():
            c_data = np.load(pi3nn_dir.c_data_path)
            c_pred_median = np.load(pi3nn_dir.c_pred_median_path)
            upper_res = np.load(pi3nn_dir.c_pred_upper_res_path)
            lower_res = np.load(pi3nn_dir.c_pred_lower_res_path)

            # Find optimal c for both directions
            if q > 0.5:
                c = optimize_quantile("up", c_data[:, 1], c_pred_median, upper_res, q)
                quantile = (c_pred_median + c * upper_res).squeeze()
            elif q < 0.5:
                c = optimize_quantile("down", c_data[:, 1], c_pred_median, lower_res, q)
                quantile = (c_pred_median - c * lower_res).squeeze()
            else:
                quantile = c_pred_median.squeeze()

            # Save quantile files
            np.save(pi3nn_dir.get_quantile_path(q), quantile)
