from typing import Any, Callable, Iterable, Mapping, Optional

from .. import Tensor

class Optimizer:
    param_groups: list[dict[str, Any]]
    state: dict[int, dict[str, Any]]
    def __init__(
        self, params: Iterable[Tensor], defaults: Mapping[str, Any]
    ) -> None: ...
    def step(
        self, closure: Optional[Callable[[], Tensor]] = ...
    ) -> Optional[Tensor]: ...
    def zero_grad(self) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None: ...

class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = ...,
        momentum: float = ...,
        dampening: float = ...,
        weight_decay: float = ...,
        nesterov: bool = ...,
        maximize: bool = ...,
    ) -> None: ...

class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = ...,
        betas: tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        maximize: bool = ...,
    ) -> None: ...

class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = ...,
        betas: tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        maximize: bool = ...,
    ) -> None: ...
