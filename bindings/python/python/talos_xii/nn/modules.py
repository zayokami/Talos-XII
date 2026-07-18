"""PyTorch-style module and parameter primitives."""

import math
from collections import OrderedDict, namedtuple
from typing import Any, Iterable, Iterator, Mapping, Optional, Tuple

from .. import Tensor, _next_seed, no_grad, ones, rand, zeros
from . import functional as F


_IncompatibleKeys = namedtuple("_IncompatibleKeys", ["missing_keys", "unexpected_keys"])


class _ParameterMeta(type):
    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, Tensor) and bool(instance._is_parameter)


class Parameter(metaclass=_ParameterMeta):
    """A leaf Tensor registered automatically when assigned to a Module."""

    def __new__(cls, data: Any = None, requires_grad: bool = True) -> Tensor:
        if data is None:
            data = []
        if isinstance(data, Tensor):
            parameter = Tensor(
                data,
                dtype=data.dtype,
                device=data.device,
                requires_grad=requires_grad,
            )
        else:
            parameter = Tensor(data, requires_grad=requires_grad)
        parameter._set_parameter(True)
        return parameter


class Module:
    """Base class for composable trainable Python modules."""

    training: bool

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_non_persistent_buffers", set())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        parameters = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        buffers = self.__dict__.get("_buffers")
        if parameters is None or modules is None or buffers is None:
            object.__setattr__(self, name, value)
            return
        if isinstance(value, Parameter):
            modules.pop(name, None)
            if name in buffers:
                buffers.pop(name)
                self._non_persistent_buffers.discard(name)
            parameters[name] = value
        elif name in parameters:
            if value is not None:
                raise TypeError(
                    f"cannot assign {type(value).__name__} as parameter {name!r} "
                    "(talos_xii.nn.Parameter or None expected)"
                )
            parameters[name] = None
        elif isinstance(value, Module):
            parameters.pop(name, None)
            if name in buffers:
                buffers.pop(name)
                self._non_persistent_buffers.discard(name)
            modules[name] = value
        elif name in modules:
            if value is not None:
                raise TypeError(
                    f"cannot assign {type(value).__name__} as child module {name!r} "
                    "(talos_xii.nn.Module or None expected)"
                )
            modules[name] = None
        elif name in buffers:
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(
                    f"cannot assign {type(value).__name__} as buffer {name!r} "
                    "(talos_xii.Tensor or None expected)"
                )
            buffers[name] = value
        else:
            object.__setattr__(self, name, value)
            return
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers.discard(name)
        elif name in self._modules:
            del self._modules[name]
        object.__delattr__(self, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.forward() is not implemented")

    def register_parameter(self, name: str, parameter: Optional[Tensor]) -> None:
        self._validate_member_name(name)
        if parameter is not None and not isinstance(parameter, Parameter):
            raise TypeError("parameter must be a talos_xii.nn.Parameter or None")
        if hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute {name!r} already exists")
        setattr(self, name, parameter)

    def register_buffer(
        self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        self._validate_member_name(name)
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError("buffer must be a talos_xii.Tensor or None")
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute {name!r} already exists")
        self._buffers[name] = tensor
        if persistent:
            self._non_persistent_buffers.discard(name)
        else:
            self._non_persistent_buffers.add(name)
        object.__setattr__(self, name, tensor)

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        self._validate_member_name(name)
        if module is not None and not isinstance(module, Module):
            raise TypeError("module must be a talos_xii.nn.Module or None")
        if hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute {name!r} already exists")
        setattr(self, name, module)

    def parameters(self, recurse: bool = True) -> Iterator[Tensor]:
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        memo = set()
        yield from self._named_parameters(prefix, recurse, memo)

    def _named_parameters(
        self, prefix: str, recurse: bool, memo: set
    ) -> Iterator[Tuple[str, Tensor]]:
        for name, parameter in self._parameters.items():
            if parameter is None or id(parameter) in memo:
                continue
            memo.add(id(parameter))
            yield self._join_name(prefix, name), parameter
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    yield from module._named_parameters(
                        self._join_name(prefix, name), True, memo
                    )

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        for name, buffer in self._buffers.items():
            if buffer is not None:
                yield self._join_name(prefix, name), buffer
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    yield from module.named_buffers(self._join_name(prefix, name), True)

    def children(self) -> Iterator["Module"]:
        for module in self._modules.values():
            if module is not None:
                yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        for name, module in self._modules.items():
            if module is not None:
                yield name, module

    def modules(self) -> Iterator["Module"]:
        yield self
        for module in self.children():
            yield from module.modules()

    def train(self, mode: bool = True) -> "Module":
        if not isinstance(mode, bool):
            raise ValueError("training mode must be a boolean")
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def to(self, device: Any = None, dtype: Any = None) -> "Module":
        with no_grad():
            for parameter in self._parameters.values():
                if parameter is None:
                    continue
                converted = parameter.to(device, dtype=dtype)
                if (
                    converted.device != parameter.device
                    or converted.dtype != parameter.dtype
                ):
                    parameter._replace_data_(converted)
            for name, buffer in tuple(self._buffers.items()):
                if buffer is None:
                    continue
                converted = buffer.to(device, dtype=dtype)
                if converted.device == buffer.device and converted.dtype == buffer.dtype:
                    continue
                self._buffers[name] = converted
                object.__setattr__(self, name, converted)
        for child in self.children():
            child.to(device, dtype)
        return self

    def cpu(self) -> "Module":
        return self.to("cpu")

    def cuda(self, device: int = 0) -> "Module":
        return self.to(f"cuda:{device}")

    def float(self) -> "Module":
        from .. import float32

        return self.to(dtype=float32)

    def double(self) -> "Module":
        from .. import float64

        return self.to(dtype=float64)

    def bfloat16(self) -> "Module":
        from .. import bfloat16

        return self.to(dtype=bfloat16)

    def state_dict(self) -> "OrderedDict[str, Tensor]":
        state = OrderedDict()
        with no_grad():
            for name, parameter in self.named_parameters():
                state[name] = parameter.detach().clone()
            for name, buffer in self._named_persistent_buffers():
                state[name] = buffer.detach().clone()
        return state

    def load_state_dict(
        self, state_dict: Mapping[str, Tensor], strict: bool = True
    ) -> _IncompatibleKeys:
        targets = OrderedDict(self.named_parameters())
        targets.update(self._named_persistent_buffers())
        missing = [name for name in targets if name not in state_dict]
        unexpected = [name for name in state_dict if name not in targets]
        errors = []
        with no_grad():
            for name, target in targets.items():
                if name not in state_dict:
                    continue
                source = state_dict[name]
                if not isinstance(source, Tensor):
                    errors.append(f"{name}: expected Tensor, found {type(source).__name__}")
                    continue
                if source.shape != target.shape:
                    errors.append(
                        f"{name}: shape {source.shape} does not match {target.shape}"
                    )
                    continue
                target.copy_(source.to(target, copy=True))
        if strict and (missing or unexpected or errors):
            details = []
            if missing:
                details.append(f"Missing key(s): {', '.join(missing)}")
            if unexpected:
                details.append(f"Unexpected key(s): {', '.join(unexpected)}")
            details.extend(errors)
            raise RuntimeError("Error(s) in loading state_dict:\n\t" + "\n\t".join(details))
        if errors:
            raise RuntimeError("Error(s) in loading state_dict:\n\t" + "\n\t".join(errors))
        return _IncompatibleKeys(missing, unexpected)

    def _named_persistent_buffers(
        self, prefix: str = ""
    ) -> Iterator[Tuple[str, Tensor]]:
        for name, buffer in self._buffers.items():
            if buffer is not None and name not in self._non_persistent_buffers:
                yield self._join_name(prefix, name), buffer
        for name, module in self._modules.items():
            if module is not None:
                yield from module._named_persistent_buffers(
                    self._join_name(prefix, name)
                )

    def _validate_member_name(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("module member name must be a non-empty string")
        if "." in name:
            raise KeyError("module member name cannot contain '.'")

    @staticmethod
    def _join_name(prefix: str, name: str) -> str:
        return f"{prefix}.{name}" if prefix else name

    def __repr__(self) -> str:
        children = list(self.named_children())
        if not children:
            return f"{type(self).__name__}()"
        lines = [f"{type(self).__name__}("]
        for name, module in children:
            representation = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {representation}")
        lines.append(")")
        return "\n".join(lines)


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight = Parameter(
            rand(
                [self.out_features, self.in_features],
                min=-bound,
                max=bound,
                seed=_next_seed(),
                device=device,
                dtype=dtype,
            )
        )
        self.bias = (
            Parameter(
                rand(
                    [self.out_features],
                    min=-bound,
                    max=bound,
                    seed=_next_seed(),
                    device=device,
                    dtype=dtype,
                )
            )
            if bias
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for index, module in enumerate(modules):
            self.add_module(str(index), module)

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for module in self.children():
            output = module(output)
        return output

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return self.children()

    def __getitem__(self, index: int) -> Module:
        return list(self.children())[index]


class ModuleList(Module):
    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def append(self, module: Module) -> "ModuleList":
        self.add_module(str(len(self._modules)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> "ModuleList":
        for module in modules:
            self.append(module)
        return self

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return self.children()

    def __getitem__(self, index: int) -> Module:
        return list(self.children())[index]


class Identity(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input


class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)


class GELU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.tanh(input)


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        return input.flatten(self.start_dim, self.end_dim)


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be positive")
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.weight = (
            Parameter(ones([normalized_shape], device=device, dtype=dtype))
            if elementwise_affine
            else None
        )
        self.bias = (
            Parameter(zeros([normalized_shape], device=device, dtype=dtype))
            if elementwise_affine
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[-1] != self.normalized_shape:
            raise RuntimeError(
                f"expected last dimension {self.normalized_shape}, found {input.shape[-1]}"
            )
        return F.layer_norm(input, self.weight, self.bias, self.eps)


class RMSNorm(Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be positive")
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.weight = (
            Parameter(ones([normalized_shape], device=device, dtype=dtype))
            if elementwise_affine
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[-1] != self.normalized_shape:
            raise RuntimeError(
                f"expected last dimension {self.normalized_shape}, found {input.shape[-1]}"
            )
        return F.rms_norm(input, self.weight, self.eps)


class MSELoss(Module):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)
