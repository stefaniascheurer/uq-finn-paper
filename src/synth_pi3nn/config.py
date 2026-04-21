import dataclasses as dc
from typing import Optional


@dc.dataclass
class Params:
    """Manages PI3NN parameters.

    Attributes:
        max_epochs: Maximum number of training epochs.
        initial_learning_rate: Initial learning rate for the optimizer.
        batch_size: Size of training batches; if None, uses full-batch training.
        activation: Activation function to use in hidden layers.
        loss_fun: Loss function to optimize.
        optimizer: Optimizer to use during training.
        validation_fraction: Fraction of training data to use for validation.
        stop_early: Whether to use early stopping based on validation loss.
        n_neurons_per_layer: List specifying the number of neurons in each hidden layer.
        positivity_method: Method to enforce positivity in outputs ('softplus', 'relu', etc.).
        verbose: Verbosity level during training (0 = silent, 1 = progress bar, 2 = one line per epoch).
        l2_regularization: L2 regularization factor for weights.
    """

    max_epochs: int = 600
    initial_learning_rate: float = 0.1
    batch_size: Optional[int] = 8
    activation: str = "tanh"
    loss_fun: str = "mse"
    optimizer: str = "adam"
    validation_fraction: float = 0.2
    stop_early: bool = True
    n_neurons_per_layer: list[int] = dc.field(default_factory=lambda: [32])
    positivity_method: str = "softplus"
    verbose: int = 2
    l2_regularization: float = 0.00001
