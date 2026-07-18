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


def test_cuda_scalar_arithmetic_stays_on_device():
    value = tx.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    result = ((2.0 - value) * 3.0 + 1.0) / 2.0

    assert result.device == tx.device("cuda")
    assert result.tolist() == pytest.approx([2.0, 0.5])
    result.sum().backward()
    assert value.grad.device == tx.device("cuda")
    assert value.grad.tolist() == pytest.approx([-1.5, -1.5])


def test_cuda_copy_from_cpu_does_not_trigger_small_tensor_cpu_fallback():
    destination = tx.zeros([2], device="cuda")
    destination.copy_(tx.tensor([1.0, 2.0]))

    negated = -destination
    combined = destination + tx.tensor([3.0, 4.0], device="cuda")
    assert destination.device == tx.device("cuda")
    assert negated.device == tx.device("cuda")
    assert combined.device == tx.device("cuda")
    assert negated.tolist() == [-1.0, -2.0]
    assert combined.tolist() == [4.0, 6.0]


@pytest.mark.parametrize("dtype", [tx.float32, tx.float64])
def test_cuda_sqrt_forward_backward(dtype):
    value = tx.tensor(
        [1.0, 4.0, 9.0], dtype=dtype, device="cuda", requires_grad=True
    )
    result = value.sqrt()

    assert result.device == tx.device("cuda")
    assert result.dtype == dtype
    assert result.tolist() == pytest.approx([1.0, 2.0, 3.0])
    result.sum().backward()
    assert value.grad.device == tx.device("cuda")
    assert value.grad.tolist() == pytest.approx([0.5, 0.25, 1.0 / 6.0])


def test_cuda_python_module_and_sgd_training_loop():
    tx.manual_seed(19)
    model = tx.nn.Linear(2, 1)
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    optimizer = tx.optim.SGD(model.parameters(), lr=0.1)
    model.cuda()
    inputs = tx.tensor([[1.0, 2.0], [2.0, -1.0]], device="cuda")
    targets = tx.tensor([[3.0], [0.0]], device="cuda")

    initial_loss = model(inputs).mse_loss(targets).item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = model(inputs).mse_loss(targets)
        loss.backward()
        optimizer.step()
    final_loss = model(inputs).mse_loss(targets).item()

    assert all(parameter.device == tx.device("cuda") for parameter in model.parameters())
    assert [id(parameter) for parameter in model.parameters()] == parameter_ids
    assert all(
        optimizer_parameter is model_parameter
        for optimizer_parameter, model_parameter in zip(
            optimizer.param_groups[0]["params"], model.parameters()
        )
    )
    assert final_loss < initial_loss * 0.1


@pytest.mark.parametrize("optimizer_name", ["SGD", "Adam", "AdamW"])
def test_cuda_optimizer_matches_cpu(optimizer_name):
    cpu_parameter = tx.nn.Parameter([1.0, -2.0, 0.5])
    cuda_parameter = tx.nn.Parameter(
        tx.tensor([1.0, -2.0, 0.5], device="cuda")
    )
    if optimizer_name == "SGD":
        kwargs = {"lr": 0.05, "momentum": 0.8, "nesterov": True}
    else:
        kwargs = {"lr": 0.01, "betas": (0.8, 0.95), "eps": 1e-7}
    cpu_optimizer = getattr(tx.optim, optimizer_name)([cpu_parameter], **kwargs)
    cuda_optimizer = getattr(tx.optim, optimizer_name)([cuda_parameter], **kwargs)
    cpu_target = tx.tensor([-0.25, 0.75, 1.5])
    cuda_target = tx.tensor([-0.25, 0.75, 1.5], device="cuda")

    for _ in range(4):
        cpu_optimizer.zero_grad()
        cuda_optimizer.zero_grad()
        (cpu_parameter - cpu_target).square().mean().backward()
        (cuda_parameter - cuda_target).square().mean().backward()
        cpu_optimizer.step()
        cuda_optimizer.step()

    assert cuda_parameter.cpu().tolist() == pytest.approx(
        cpu_parameter.tolist(), abs=2e-5
    )
