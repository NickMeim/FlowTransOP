from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeBackends:
    """Device choices used by package wrappers.

    The model device and the TRANSACT/pre-alignment backend are intentionally
    separate because users may want, for example, model training on CUDA while
    running the initial alignment on CPU for reproducibility checks.
    """

    model_device: str = "cuda"
    transact_backend: str = "gpu"
    transact_device: str = "cuda"

    def __post_init__(self) -> None:
        if self.transact_backend not in {"gpu", "cpu"}:
            raise ValueError("transact_backend must be either 'gpu' or 'cpu'")

    def as_env(self) -> dict[str, str]:
        return {
            "FLOWTRANSOP_MODEL_DEVICE": self.model_device,
            "FLOWTRANSOP_TRANSACT_BACKEND": self.transact_backend,
            "FLOWTRANSOP_TRANSACT_DEVICE": self.transact_device,
        }
