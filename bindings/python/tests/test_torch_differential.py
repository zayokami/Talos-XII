import pytest

import talos_xii as tx

torch = pytest.importorskip("torch")


def _torch_210_or_skip():
    version = torch.__version__.split("+", 1)[0]
    if tuple(version.split(".")[:2]) != ("2", "10"):
        pytest.skip(f"PyTorch 2.10.x oracle required, found {torch.__version__}")


def _assert_nested_close(actual, expected, tolerance=1e-6):
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_close(actual_item, expected_item, tolerance)
    else:
        assert abs(actual - expected) <= tolerance


@pytest.mark.torch_contract
def test_forward_shape_dtype_and_broadcast_match_torch_210():
    _torch_210_or_skip()
    torch_value = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    talos_value = tx.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tx.float64)
    torch_bias = torch.tensor([0.5, -1.0], dtype=torch.float64)
    talos_bias = tx.tensor([0.5, -1.0], dtype=tx.float64)

    torch_output = (torch_value + torch_bias).transpose(0, 1).sum(-1)
    talos_output = (talos_value + talos_bias).transpose(0, 1).sum(-1)

    assert talos_output.shape == tuple(torch_output.shape)
    assert talos_output.dtype == tx.float64
    _assert_nested_close(talos_output.tolist(), torch_output.tolist())


@pytest.mark.torch_contract
def test_backward_and_explicit_vector_gradient_match_torch_210():
    _torch_210_or_skip()
    torch_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
    talos_leaf = tx.tensor([1.0, 2.0], requires_grad=True)

    torch_output = torch_leaf.square()
    talos_output = talos_leaf.square()
    torch_output.backward(torch.tensor([0.5, 2.0]))
    talos_output.backward(tx.tensor([0.5, 2.0]))

    _assert_nested_close(talos_leaf.grad.tolist(), torch_leaf.grad.tolist())


@pytest.mark.torch_contract
def test_no_grad_detach_and_clone_match_torch_210():
    _torch_210_or_skip()
    torch_leaf = torch.tensor([1.0], requires_grad=True)
    talos_leaf = tx.tensor([1.0], requires_grad=True)

    with torch.no_grad():
        torch_result = torch_leaf * 2.0
    with tx.no_grad():
        talos_result = talos_leaf * 2.0

    assert talos_result.requires_grad == torch_result.requires_grad
    assert talos_leaf.detach().requires_grad == torch_leaf.detach().requires_grad
    assert talos_leaf.clone().is_leaf == torch_leaf.clone().is_leaf


@pytest.mark.torch_contract
def test_error_contract_matches_torch_210_categories():
    _torch_210_or_skip()
    torch_output = torch.ones(2, requires_grad=True) * 2.0
    talos_output = tx.ones([2], requires_grad=True) * 2.0

    with pytest.raises(RuntimeError):
        torch_output.backward()
    with pytest.raises(RuntimeError):
        talos_output.backward()

    with pytest.raises(TypeError):
        len(torch.tensor(1.0))
    with pytest.raises(TypeError):
        len(tx.tensor(1.0))
