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


@pytest.mark.torch_contract
def test_linear_forward_and_backward_match_torch_210():
    _torch_210_or_skip()
    torch_layer = torch.nn.Linear(3, 2)
    talos_layer = tx.nn.Linear(3, 2)
    weight = [[0.5, -0.25, 0.75], [-1.0, 0.5, 0.125]]
    bias = [0.2, -0.3]
    with torch.no_grad():
        torch_layer.weight.copy_(torch.tensor(weight))
        torch_layer.bias.copy_(torch.tensor(bias))
    with tx.no_grad():
        talos_layer.weight.copy_(tx.tensor(weight))
        talos_layer.bias.copy_(tx.tensor(bias))

    torch_input = torch.tensor(
        [[1.0, 2.0, -1.0], [-0.5, 0.25, 2.0]], requires_grad=True
    )
    talos_input = tx.tensor(
        [[1.0, 2.0, -1.0], [-0.5, 0.25, 2.0]], requires_grad=True
    )
    torch_output = torch_layer(torch_input)
    talos_output = talos_layer(talos_input)
    torch_output.square().mean().backward()
    talos_output.square().mean().backward()

    _assert_nested_close(talos_output.tolist(), torch_output.tolist(), 1e-5)
    _assert_nested_close(talos_input.grad.tolist(), torch_input.grad.tolist(), 1e-5)
    _assert_nested_close(
        talos_layer.weight.grad.tolist(), torch_layer.weight.grad.tolist(), 1e-5
    )
    _assert_nested_close(
        talos_layer.bias.grad.tolist(), torch_layer.bias.grad.tolist(), 1e-5
    )


@pytest.mark.torch_contract
@pytest.mark.parametrize("optimizer_name", ["SGD", "Adam", "AdamW"])
def test_optimizer_updates_match_torch_210(optimizer_name):
    _torch_210_or_skip()
    torch_parameter = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
    talos_parameter = tx.nn.Parameter([1.0, -2.0, 0.5])
    if optimizer_name == "SGD":
        kwargs = {
            "lr": 0.05,
            "momentum": 0.8,
            "weight_decay": 0.03,
            "nesterov": True,
        }
    else:
        kwargs = {
            "lr": 0.01,
            "betas": (0.8, 0.95),
            "eps": 1e-7,
            "weight_decay": 0.03,
        }
    torch_optimizer = getattr(torch.optim, optimizer_name)([torch_parameter], **kwargs)
    talos_optimizer = getattr(tx.optim, optimizer_name)([talos_parameter], **kwargs)
    torch_target = torch.tensor([-0.25, 0.75, 1.5])
    talos_target = tx.tensor([-0.25, 0.75, 1.5])

    for _ in range(4):
        torch_optimizer.zero_grad()
        talos_optimizer.zero_grad()
        (torch_parameter - torch_target).square().mean().backward()
        (talos_parameter - talos_target).square().mean().backward()
        torch_optimizer.step()
        talos_optimizer.step()

    _assert_nested_close(talos_parameter.tolist(), torch_parameter.tolist(), 2e-5)
