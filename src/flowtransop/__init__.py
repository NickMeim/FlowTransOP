"""FlowTransOP package helpers."""

from .backends import RuntimeBackends
from .transact import TransactBackend, load_transact_backend

__all__ = [
    "RuntimeBackends",
    "TransactBackend",
    "load_transact_backend",
    "ARCHS4EnsembleTranslator",
    "FlowTransOPTranslator",
    "finetune_archs4_ensemble_member",
    "load_archs4_ensemble",
    "load_archs4_translator",
    "translate_archs4_ensemble_array",
    "translate_array",
]
__version__ = "0.1.0"


def __getattr__(name):
    if name in {"FlowTransOPTranslator", "load_archs4_translator", "translate_array"}:
        from . import inference

        return getattr(inference, name)
    if name in {
        "ARCHS4EnsembleTranslator",
        "finetune_archs4_ensemble_member",
        "load_archs4_ensemble",
        "translate_archs4_ensemble_array",
    }:
        from . import archs4

        return getattr(archs4, name)
    raise AttributeError(name)
