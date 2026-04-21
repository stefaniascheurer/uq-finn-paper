import dataclasses as dc


@dc.dataclass
class Params:
    """Manages PB parameters.

    Attributes:
        n_bootstraps: Number of bootstraps to generate.
    """
    noise = 0.03
    n_bootstraps: int = 100
