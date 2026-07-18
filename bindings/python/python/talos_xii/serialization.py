"""Safe, versioned checkpoint serialization without pickle."""

import json
import os
import tempfile
import zipfile
from collections import OrderedDict
from os import PathLike
from typing import Any, Mapping, Optional, Union

from . import Tensor, bfloat16, float32, float64, int8, tensor


_FORMAT = "talos_xii.checkpoint"
_VERSION = 1
_DTYPES = {
    "float32": float32,
    "float64": float64,
    "bfloat16": bfloat16,
    "int8": int8,
}


def _dtype_name(value: Any) -> str:
    for name, candidate in _DTYPES.items():
        if value == candidate:
            return name
    raise TypeError(f"unsupported checkpoint dtype: {value!r}")


def _encode(value: Any) -> Any:
    if isinstance(value, Tensor):
        return {
            "__type__": "tensor",
            "shape": list(value.shape),
            "dtype": _dtype_name(value.dtype),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
            "data": value.to_list(),
        }
    if isinstance(value, Mapping):
        return {
            "__type__": "mapping",
            "items": [[str(key), _encode(item)] for key, item in value.items()],
        }
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")


def _decode(value: Any, map_location: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item, map_location) for item in value]
    if not isinstance(value, dict) or "__type__" not in value:
        return value
    kind = value["__type__"]
    if kind == "tensor":
        device = value["device"] if map_location is None else map_location
        return tensor(
            value["data"],
            value["shape"],
            dtype=_DTYPES[value["dtype"]],
            device=device,
            requires_grad=value["requires_grad"],
        )
    if kind == "mapping":
        return OrderedDict(
            (key, _decode(item, map_location)) for key, item in value["items"]
        )
    if kind == "tuple":
        return tuple(_decode(item, map_location) for item in value["items"])
    raise ValueError(f"unsupported checkpoint record type: {kind!r}")


def save(obj: Any, path: Union[str, PathLike]) -> None:
    """Atomically save a module, optimizer, state dictionary, or Tensor tree."""

    value = obj.state_dict() if hasattr(obj, "state_dict") else obj
    payload = {"format": _FORMAT, "version": _VERSION, "value": _encode(value)}
    destination = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(destination))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".talos-xii-", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr(
                "checkpoint.json",
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def load(path: Union[str, PathLike], map_location: Optional[Any] = None) -> Any:
    """Load a checkpoint produced by :func:`save` without executing code."""

    with zipfile.ZipFile(os.fspath(path), "r") as archive:
        names = archive.namelist()
        if names != ["checkpoint.json"]:
            raise ValueError("checkpoint must contain exactly checkpoint.json")
        payload = json.loads(archive.read("checkpoint.json"))
    if payload.get("format") != _FORMAT or payload.get("version") != _VERSION:
        raise ValueError("unsupported Talos-XII checkpoint format or version")
    return _decode(payload["value"], map_location)


__all__ = ["save", "load"]
