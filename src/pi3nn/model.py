import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MaxOneScaler:
    """Scaler that transforms data such that the maximum absolute value is 1.0."""

    def __init__(self):
        self.scale_ = None

    def fit(self, X: np.ndarray):
        """Calculates the scaling factor based on the max absolute value in X."""
        x_abs_max = np.max(np.abs(X), axis=0)
        self.scale_ = 1.0 / x_abs_max
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applies the scaling factor to X."""
        return np.asarray(X).copy() * self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverses the scaling to return data to original units."""
        return np.asarray(X).copy() / self.scale_


class ScaledTrainingModel(keras.Model):
    """Wrapper for Keras Models to handle automated input/output scaling."""

    def __init__(self, model: keras.Model, x_scaler, y_scaler):
        """Initializes with a compiled model and fitted scalers."""
        super().__init__()
        self.model, self.x_scaler, self.y_scaler = model, x_scaler, y_scaler

    def fit(self, x: np.ndarray, y: np.ndarray, *args, **kwargs):
        """Fits the model using transformed x and y."""
        return super().fit(self.x_scaler.transform(x), self.y_scaler.transform(y), *args, **kwargs)

    def call(self, x: np.ndarray, *args, **kwargs):
        """Executes the forward pass."""
        return self.model.call(x, *args, **kwargs)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts and inverse-transforms the output back to original units."""
        return self.y_scaler.inverse_transform(self.model.predict(self.x_scaler.transform(x)))


def make_model(x_train: np.ndarray, y_train: np.ndarray, layer_sizes: list[int], activation: str, positivity_method: str = None, l2_reg: float = 0.0):
    """Factory function to build and wrap a PI3NN Neural Network.

    Args:
        x_train: Training features used to fit the input scaler.
        y_train: Training targets used to fit the output scaler.
        layer_sizes: List defining hidden layer neuron counts.
        activation: Hidden layer activation function.
        positivity_method: Optional string ('softplus', 'sqrt_sqr', etc.) 
                           to enforce positive outputs for residual nets.
        l2_reg: L2 regularization penalty.
    """
    model = keras.Sequential([
        keras.Input(shape=(x_train.shape[1],)),
        *[keras.layers.Dense(n, activation=activation,
                             kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg != 0.0 else None)
          for n in layer_sizes],
        keras.layers.Dense(y_train.shape[1]),
    ])
    if positivity_method:
        x = model(model.inputs)
        methods = {
            "sqrt_sqr": lambda: keras.ops.sqrt(keras.ops.square(x) + 1e-12),
            "abs": lambda: keras.ops.abs(x),
            "softplus": lambda: keras.layers.Activation("softplus")(x),
            "exp": lambda: keras.layers.Activation("exponential")(x),
            "relu": lambda: keras.layers.ReLU()(x),
        }
        model = keras.Model(model.input, methods[positivity_method]())
        return ScaledTrainingModel(model, MinMaxScaler().fit(x_train), MaxOneScaler().fit(y_train))
    return ScaledTrainingModel(model, MinMaxScaler().fit(x_train), StandardScaler().fit(y_train))
