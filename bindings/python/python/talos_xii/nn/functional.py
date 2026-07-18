"""Functional neural-network operations."""

from typing import Optional

from .. import Tensor


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    output = input.matmul(weight.T)
    return output if bias is None else output + bias


def relu(input: Tensor) -> Tensor:
    return input.relu()


def gelu(input: Tensor) -> Tensor:
    return input.gelu()


def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    return input.tanh()


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    return input.mse_loss(target)


def layer_norm(
    input: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    mean = input.mean(-1, True)
    variance = (input - mean).square().mean(-1, True)
    output = (input - mean) * (variance + eps).rsqrt()
    if weight is not None:
        output = output * weight
    if bias is not None:
        output = output + bias
    return output


def rms_norm(input: Tensor, weight: Optional[Tensor] = None, eps: float = 1e-5) -> Tensor:
    variance = input.square().mean(-1, True)
    output = input * (variance + eps).rsqrt()
    return output if weight is None else output * weight
