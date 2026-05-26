"""FlowTransOP package helpers."""

from .backends import RuntimeBackends
from .transact import TransactBackend, load_transact_backend

__all__ = [
    "RuntimeBackends",
    "TransactBackend",
    "load_transact_backend",
    "FlowTransOPTranslator",
    "load_archs4_translator",
    "translate_array",
]
__version__ = "0.1.0"


def __getattr__(name):
    if name in {"FlowTransOPTranslator", "load_archs4_translator", "translate_array"}:
        from . import inference

        return getattr(inference, name)
    raise AttributeError(name)
