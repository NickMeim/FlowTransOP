"""FlowTransOP package helpers."""

__all__ = ["FlowTransOPTranslator", "load_archs4_translator", "translate_array"]
__version__ = "0.1.0"


def __getattr__(name):
    if name in __all__:
        from . import inference

        return getattr(inference, name)
    raise AttributeError(name)
