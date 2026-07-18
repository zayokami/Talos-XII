import math

import pytest

import talos_xii as tx


def test_nested_constructor_and_metadata_contract():
    tensor = tx.tensor([[1.0, 2.0], [3.0, 4.0]])

    assert tensor.shape == (2, 2)
    assert tensor.size() == (2, 2)
    assert tensor.size(-1) == 2
    assert tensor.ndim == 2
    assert tensor.dim() == 2
    assert tensor.dtype == tx.float32
    assert tensor.device == tx.device("cpu")
    assert tensor.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert len(tensor) == 2


def test_scalar_and_f64_item_preserve_python_semantics():
    scalar = tx.tensor(1.123456789012345, dtype=tx.float64)

    assert scalar.shape == ()
    assert scalar.tolist() == 1.123456789012345
    assert scalar.item() == 1.123456789012345
    with pytest.raises(TypeError, match=r"len\(\) of a 0-d tensor"):
        len(scalar)


def test_broadcast_negative_dimensions_and_shape_ops():
    matrix = tx.ones([2, 3])
    vector = tx.tensor([1.0, 2.0, 3.0])

    result = matrix + vector
    assert result.shape == (2, 3)
    assert result.tolist() == [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]
    assert result.sum(-1).tolist() == [9.0, 9.0]
    assert result.mean(-2).tolist() == [2.0, 3.0, 4.0]
    assert result.reshape([-1, 2]).shape == (3, 2)
    assert result.unsqueeze(-1).shape == (2, 3, 1)
    assert result.unsqueeze(-1).squeeze(-1).shape == (2, 3)
    assert result.flatten(0, -1).shape == (6,)
    assert result[0].tolist() == [2.0, 3.0, 4.0]
    assert result[:, 1:].tolist() == [[3.0, 4.0], [3.0, 4.0]]


def test_requires_grad_and_grad_tensor_contract():
    leaf = tx.tensor([1.0, 2.0], requires_grad=True)
    assert leaf.requires_grad is True
    assert leaf.is_leaf is True
    assert leaf.grad is None

    output = (leaf * 3.0).sum()
    assert output.shape == ()
    assert output.requires_grad is True
    assert output.is_leaf is False
    assert output.grad_fn is not None
    output.backward()

    assert isinstance(leaf.grad, tx.Tensor)
    assert leaf.grad.tolist() == [3.0, 3.0]
    assert output.grad is None


def test_non_scalar_backward_requires_explicit_gradient():
    leaf = tx.tensor([1.0, 2.0], requires_grad=True)
    output = leaf * 2.0

    with pytest.raises(
        RuntimeError, match="grad can be implicitly created only for scalar outputs"
    ):
        output.backward()

    output.backward(tx.tensor([1.0, 0.5]))
    assert leaf.grad.tolist() == [2.0, 1.0]


def test_no_grad_context_and_decorator():
    leaf = tx.tensor([1.0], requires_grad=True)

    with tx.no_grad():
        result = leaf * 2.0
    assert result.requires_grad is False
    assert tx.is_grad_enabled() is True

    @tx.no_grad()
    def inference(value):
        return value + 1.0

    assert inference(leaf).requires_grad is False
    assert tx.is_grad_enabled() is True


def test_detach_alias_clone_copy_and_in_place_version_guard():
    leaf = tx.tensor([1.0, 2.0], requires_grad=True)
    detached = leaf.detach()
    cloned = leaf.clone()

    with tx.no_grad():
        detached.fill_(5.0)
    assert leaf.tolist() == [5.0, 5.0]
    assert cloned.tolist() == [1.0, 2.0]

    guarded = tx.tensor([2.0], requires_grad=True)
    output = guarded.square().sum()
    with tx.no_grad():
        guarded.fill_(3.0)
    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        output.backward()


def test_graph_release_and_retained_graph_accumulate_leaf_gradients():
    leaf = tx.tensor([2.0], requires_grad=True)
    output = leaf.square().sum()
    output.backward(retain_graph=True)
    assert leaf.grad.tolist() == [4.0]

    output.backward()
    assert leaf.grad.tolist() == [8.0]
    with pytest.raises(RuntimeError, match="backward through the graph a second time"):
        output.backward()


def test_dtype_conversion_remains_differentiable():
    leaf = tx.tensor([1.5], requires_grad=True)
    converted = leaf.double()

    assert converted.dtype == tx.float64
    assert converted.device == tx.device("cpu")
    converted.sum().backward()
    assert leaf.grad.tolist() == [1.0]


def test_loss_is_scalar_and_backward_reaches_inputs():
    prediction = tx.tensor([[1.0], [3.0]], requires_grad=True)
    target = tx.tensor([[0.0], [1.0]])
    loss = prediction.mse_loss(target)

    assert loss.shape == ()
    assert math.isclose(loss.item(), 2.5)
    loss.backward()
    assert prediction.grad.tolist() == [[1.0], [2.0]]


def test_invalid_contract_boundaries_are_typed_errors():
    with pytest.raises(ValueError, match="ragged"):
        tx.tensor([[1.0], [2.0, 3.0]])
    with pytest.raises(RuntimeError, match="floating point"):
        tx.tensor([1], dtype=tx.int8, requires_grad=True)
    if not tx.cuda.is_available():
        with pytest.raises(RuntimeError, match="built without CUDA"):
            tx.ones([1], device="cuda")
    with pytest.raises(RuntimeError, match="cannot be multiplied"):
        tx.ones([2]).matmul(tx.ones([3, 2]))
