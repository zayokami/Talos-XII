"""Gradient-based optimizers for Talos-XII."""

import math
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from .. import Tensor, no_grad, zeros


class Optimizer:
    def __init__(self, params: Iterable[Tensor], defaults: Mapping[str, Any]) -> None:
        parameters = []
        seen = set()
        for parameter in params:
            if not isinstance(parameter, Tensor):
                raise TypeError("optimizer parameters must be talos_xii.Tensor objects")
            if not parameter.requires_grad:
                continue
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
        if not parameters:
            raise ValueError("optimizer received an empty parameter list")
        self.param_groups = [dict(defaults, params=parameters)]
        self.state: Dict[int, Dict[str, Any]] = {}

    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                parameter.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        parameters = self.param_groups[0]["params"]
        serialized_state = []
        for parameter in parameters:
            values = {}
            for name, value in self.state.get(id(parameter), {}).items():
                values[name] = value.detach().clone() if isinstance(value, Tensor) else value
            serialized_state.append(values)
        group = {
            name: value
            for name, value in self.param_groups[0].items()
            if name != "params"
        }
        group["params"] = list(range(len(parameters)))
        return {"state": serialized_state, "param_groups": [group]}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        states = state_dict.get("state")
        groups = state_dict.get("param_groups")
        parameters = self.param_groups[0]["params"]
        if not isinstance(states, list) or len(states) != len(parameters):
            raise ValueError("optimizer state does not match the parameter count")
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError("optimizer state must contain exactly one parameter group")
        loaded_group = groups[0]
        for name in tuple(self.param_groups[0]):
            if name != "params" and name in loaded_group:
                self.param_groups[0][name] = loaded_group[name]
        self.state.clear()
        for parameter, values in zip(parameters, states):
            restored = {}
            for name, value in values.items():
                restored[name] = (
                    value.to(parameter, copy=True) if isinstance(value, Tensor) else value
                )
            self.state[id(parameter)] = restored


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
    ) -> None:
        if lr < 0 or momentum < 0 or weight_decay < 0:
            raise ValueError("lr, momentum, and weight_decay must be non-negative")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and zero dampening")
        super().__init__(
            params,
            {
                "lr": float(lr),
                "momentum": float(momentum),
                "dampening": float(dampening),
                "weight_decay": float(weight_decay),
                "nesterov": bool(nesterov),
                "maximize": bool(maximize),
            },
        )

    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        loss = closure() if closure is not None else None
        with no_grad():
            for group in self.param_groups:
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    if group["maximize"]:
                        gradient = -gradient
                    if group["weight_decay"] != 0:
                        gradient = gradient + parameter * group["weight_decay"]
                    if group["momentum"] != 0:
                        state = self.state.setdefault(id(parameter), {})
                        buffer = state.get("momentum_buffer")
                        if buffer is None:
                            buffer = gradient.detach().clone()
                            state["momentum_buffer"] = buffer
                        else:
                            buffer.copy_(
                                buffer * group["momentum"]
                                + gradient * (1.0 - group["dampening"])
                            )
                        gradient = (
                            gradient + buffer * group["momentum"]
                            if group["nesterov"]
                            else buffer
                        )
                    parameter.copy_(parameter - gradient * group["lr"])
        return loss


class _AdamBase(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        betas: tuple,
        eps: float,
        weight_decay: float,
        maximize: bool,
        decoupled_weight_decay: bool,
    ) -> None:
        beta1, beta2 = betas
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("lr, eps, and weight_decay must be non-negative")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in the interval [0, 1)")
        super().__init__(
            params,
            {
                "lr": float(lr),
                "betas": (float(beta1), float(beta2)),
                "eps": float(eps),
                "weight_decay": float(weight_decay),
                "maximize": bool(maximize),
                "decoupled_weight_decay": bool(decoupled_weight_decay),
            },
        )

    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        loss = closure() if closure is not None else None
        with no_grad():
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    if group["maximize"]:
                        gradient = -gradient
                    if group["weight_decay"] != 0 and not group["decoupled_weight_decay"]:
                        gradient = gradient + parameter * group["weight_decay"]
                    state = self.state.setdefault(id(parameter), {})
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = zeros(
                            parameter.shape,
                            dtype=parameter.dtype,
                            device=parameter.device,
                        )
                        state["exp_avg_sq"] = zeros(
                            parameter.shape,
                            dtype=parameter.dtype,
                            device=parameter.device,
                        )
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.copy_(exp_avg * beta1 + gradient * (1.0 - beta1))
                    exp_avg_sq.copy_(
                        exp_avg_sq * beta2 + gradient.square() * (1.0 - beta2)
                    )
                    if group["decoupled_weight_decay"] and group["weight_decay"] != 0:
                        parameter.copy_(
                            parameter * (1.0 - group["lr"] * group["weight_decay"])
                        )
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    denominator = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                    denominator = denominator + group["eps"]
                    step_size = group["lr"] / bias_correction1
                    parameter.copy_(parameter - exp_avg / denominator * step_size)
        return loss


class Adam(_AdamBase):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        maximize: bool = False,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, maximize, False)


class AdamW(_AdamBase):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        maximize: bool = False,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, maximize, True)


__all__ = ["Optimizer", "SGD", "Adam", "AdamW"]
