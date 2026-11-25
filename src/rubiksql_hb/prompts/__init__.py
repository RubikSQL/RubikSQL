"""Prompt runtime facade; bootstrap imports do not initialize the runtime."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["PromptStoreError", "Release", "RenderRecorder", "current", "load_release",
           "resolve_baseline", "same_prompts"]


def __getattr__(name: str) -> Any:
    """Load the public runtime symbol only when a consumer asks for it."""
    if name not in __all__:
        raise AttributeError(name)
    module = import_module(".store" if name == "PromptStoreError" else ".runtime", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
