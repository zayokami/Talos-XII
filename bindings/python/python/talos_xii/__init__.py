"""Talos-XII deep-learning framework Python API."""

from contextlib import ContextDecorator as _ContextDecorator
from functools import wraps as _wraps
import os as _os
from pathlib import Path as _Path
from threading import local as _local


_dll_directories = []


def _configure_windows_cuda_dlls():
    if _os.name != "nt" or not hasattr(_os, "add_dll_directory"):
        return
    candidates = []
    cuda_path = _os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(_Path(cuda_path) / "bin")
    candidates.extend(_Path(entry) for entry in _os.environ.get("PATH", "").split(";") if entry)

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        normalized = str(resolved).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        if not (resolved / "cudart64_12.dll").is_file():
            continue
        try:
            _dll_directories.append(_os.add_dll_directory(str(resolved)))
        except OSError:
            continue


_configure_windows_cuda_dlls()

from . import _native as _C
from ._native import *
from ._native import __version__


class _GradMode(_ContextDecorator):
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self._state = _local()

    def _stack(self):
        if not hasattr(self._state, "previous"):
            self._state.previous = []
        return self._state.previous

    def __enter__(self):
        self._stack().append(_C._set_grad_enabled(self.enabled))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        stack = self._stack()
        if stack:
            _C._set_grad_enabled(stack.pop())
        return False

    def __call__(self, function):
        @_wraps(function)
        def wrapped(*args, **kwargs):
            with type(self)(self.enabled):
                return function(*args, **kwargs)

        return wrapped


def no_grad():
    """Disable gradient recording in a context or decorated function."""

    return _GradMode(False)


def enable_grad():
    """Enable gradient recording in a context or decorated function."""

    return _GradMode(True)


def set_grad_enabled(enabled: bool):
    """Set gradient recording for a context or decorated function."""

    return _GradMode(enabled)


def is_grad_enabled() -> bool:
    """Return whether the current thread records autograd operations."""

    return _C._is_grad_enabled()


class _CudaNamespace:
    @staticmethod
    def is_available() -> bool:
        return _C._cuda_is_available()

    @staticmethod
    def device_count() -> int:
        return _C._cuda_device_count()


cuda = _CudaNamespace()


__all__ = [name for name in globals() if not name.startswith("_")]
