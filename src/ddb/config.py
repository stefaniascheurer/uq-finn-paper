import dataclasses as dc


@dc.dataclass
class Params:
    """Manages DDB parameters.

    Attributes:
        n_mixed_quantiles: Number of mixed quantiles to generate.
    """

    n_mixed_quantiles: int = 70
