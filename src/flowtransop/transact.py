from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .backends import RuntimeBackends


@dataclass(frozen=True)
class TransactBackend:
    name: str
    align: Callable[..., Any]
    transform: Callable[..., Any]
    config: RuntimeBackends


def _add_learning_path(repo_root: str | Path | None) -> None:
    if repo_root is None:
        return
    learning = Path(repo_root).resolve() / "learning"
    if str(learning) not in sys.path:
        sys.path.insert(0, str(learning))


def load_transact_backend(
    repo_root: str | Path | None = None,
    backends: RuntimeBackends | None = None,
    backend: str | None = None,
    transact_device: str | None = None,
) -> TransactBackend:
    """Load the requested TRANSACT implementation from the research scripts."""

    config = backends or RuntimeBackends()
    if backend is not None or transact_device is not None:
        config = RuntimeBackends(
            model_device=config.model_device,
            transact_backend=backend or config.transact_backend,
            transact_device=transact_device or config.transact_device,
        )

    _add_learning_path(repo_root)

    if config.transact_backend == "gpu":
        try:
            module = importlib.import_module("transact_utility_gpu")
        except ModuleNotFoundError as exc:
            if exc.name != "transact_utility_gpu":
                raise ModuleNotFoundError(
                    f"Could not import the GPU TRANSACT backend because dependency "
                    f"{exc.name!r} is missing. Install the reproduction dependencies first."
                ) from exc
            raise ModuleNotFoundError(
                "Could not import learning/transact_utility_gpu.py. Pass repo_root "
                "or run from the repository root so the learning scripts are importable."
            ) from exc

        def align(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("device", config.transact_device)
            return module.transact_align_gpu(*args, **kwargs)

        return TransactBackend(
            name="gpu",
            align=align,
            transform=module.transact_transform_gpu,
            config=config,
        )

    try:
        module = importlib.import_module("utility")
    except ModuleNotFoundError as exc:
        if exc.name != "utility":
            raise ModuleNotFoundError(
                f"Could not import the CPU TRANSACT backend because dependency "
                f"{exc.name!r} is missing. Install the reproduction dependencies first."
            ) from exc
        raise ModuleNotFoundError(
            "Could not import learning/utility.py. Pass repo_root or run from "
            "the repository root so the learning scripts are importable."
        ) from exc

    return TransactBackend(
        name="cpu",
        align=module.transact_align,
        transform=module.transact_transform,
        config=config,
    )
