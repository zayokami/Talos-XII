import pytest

import talos_xii as tx


pytestmark = pytest.mark.skipif(
    not tx.cuda.is_available(), reason="Talos-XII CUDA runtime is unavailable"
)


def test_cuda_leaf_backward_exports_materialized_gradient():
    leaf = tx.tensor([1.0, 2.0], device="cuda", requires_grad=True)

    leaf.square().sum().backward()

    assert leaf.grad is not None
    assert leaf.grad.device == tx.device("cuda")
    assert leaf.grad.tolist() == [2.0, 4.0]


def test_cpu_to_cuda_transfer_preserves_backward_edge():
    leaf = tx.tensor([1.0, 2.0], requires_grad=True)
    moved = leaf.cuda()

    (moved * 3.0).sum().backward()

    assert moved.device == tx.device("cuda")
    assert leaf.grad is not None
    assert leaf.grad.device == tx.device("cpu")
    assert leaf.grad.tolist() == [3.0, 3.0]


def test_cuda_to_cpu_transfer_preserves_backward_edge():
    leaf = tx.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    moved = leaf.cpu()

    (moved * 4.0).sum().backward()

    assert moved.device == tx.device("cpu")
    assert leaf.grad is not None
    assert leaf.grad.device == tx.device("cuda")
    assert leaf.grad.tolist() == [4.0, 4.0]


def test_cuda_broadcast_backward_reduces_gradient_to_operand_shape():
    matrix = tx.ones([2, 3], device="cuda", requires_grad=True)
    bias = tx.tensor([0.5, 1.0, 1.5], device="cuda", requires_grad=True)

    (matrix + bias).sum().backward()

    assert matrix.grad is not None
    assert matrix.grad.device == tx.device("cuda")
    assert matrix.grad.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert bias.grad is not None
    assert bias.grad.device == tx.device("cuda")
    assert bias.grad.tolist() == [2.0, 2.0, 2.0]
