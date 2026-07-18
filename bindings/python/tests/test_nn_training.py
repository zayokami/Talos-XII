from collections import OrderedDict

import pytest

import talos_xii as tx


def test_module_registers_parameters_and_propagates_training_mode():
    tx.manual_seed(7)
    model = tx.nn.Sequential(
        tx.nn.Linear(2, 4),
        tx.nn.ReLU(),
        tx.nn.Linear(4, 1),
    )

    names = [name for name, _ in model.named_parameters()]
    assert names == ["0.weight", "0.bias", "2.weight", "2.bias"]
    assert all(isinstance(parameter, tx.nn.Parameter) for parameter in model.parameters())
    assert all(parameter.is_leaf for parameter in model.parameters())

    model.eval()
    assert all(not module.training for module in model.modules())
    model.train()
    assert all(module.training for module in model.modules())


def test_sgd_trains_linear_regression_and_preserves_parameter_identity():
    tx.manual_seed(11)
    model = tx.nn.Linear(1, 1)
    optimizer = tx.optim.SGD(model.parameters(), lr=0.1)
    inputs = tx.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    targets = tx.tensor([[-5.0], [-2.0], [1.0], [4.0], [7.0]])
    parameter_ids = [id(parameter) for parameter in model.parameters()]

    initial_loss = model(inputs).mse_loss(targets).item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = model(inputs).mse_loss(targets)
        loss.backward()
        optimizer.step()
    final_loss = model(inputs).mse_loss(targets).item()

    assert final_loss < initial_loss * 1e-3
    assert [id(parameter) for parameter in model.parameters()] == parameter_ids


def test_adamw_state_roundtrip_and_safe_checkpoint(tmp_path):
    parameter = tx.nn.Parameter([1.0, -1.0])
    optimizer = tx.optim.AdamW([parameter], lr=0.05)

    optimizer.zero_grad()
    parameter.square().mean().backward()
    optimizer.step()
    state = optimizer.state_dict()

    assert state["state"][0]["step"] == 1
    assert state["state"][0]["exp_avg"].shape == (2,)

    checkpoint = tmp_path / "optimizer.txckpt"
    tx.save(optimizer, checkpoint)
    loaded = tx.load(checkpoint, map_location="cpu")
    restored = tx.optim.AdamW([tx.nn.Parameter([0.0, 0.0])], lr=1.0)
    restored.load_state_dict(loaded)

    assert restored.param_groups[0]["lr"] == pytest.approx(0.05)
    assert restored.state_dict()["state"][0]["step"] == 1


def test_module_state_dict_save_load_roundtrip(tmp_path):
    tx.manual_seed(101)
    source = tx.nn.Sequential(tx.nn.Linear(2, 3), tx.nn.Tanh(), tx.nn.Linear(3, 1))
    input = tx.tensor([[0.25, -0.5], [1.0, 2.0]])
    expected = source(input).tolist()
    checkpoint = tmp_path / "model.txckpt"

    tx.save(source, checkpoint)
    state = tx.load(checkpoint, map_location="cpu")
    assert isinstance(state, OrderedDict)

    tx.manual_seed(202)
    destination = tx.nn.Sequential(
        tx.nn.Linear(2, 3), tx.nn.Tanh(), tx.nn.Linear(3, 1)
    )
    result = destination.load_state_dict(state)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert destination(input).tolist() == expected


def test_parameter_copy_requires_no_grad():
    parameter = tx.nn.Parameter([1.0])
    with pytest.raises(RuntimeError, match="leaf Tensor"):
        parameter.copy_(tx.tensor([2.0]))
    with tx.no_grad():
        result = parameter.copy_(tx.tensor([2.0]))
    assert result is parameter
    assert parameter.tolist() == [2.0]


def test_module_to_preserves_parameter_identity_and_existing_gradient():
    model = tx.nn.Linear(2, 1)
    optimizer = tx.optim.SGD(model.parameters(), lr=0.1)
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    model(tx.tensor([[1.0, 2.0]])).sum().backward()
    gradients = [parameter.grad.tolist() for parameter in model.parameters()]

    model.double()

    assert [id(parameter) for parameter in model.parameters()] == parameter_ids
    assert optimizer.param_groups[0]["params"] == list(model.parameters())
    assert all(parameter.dtype == tx.float64 for parameter in model.parameters())
    assert [parameter.grad.tolist() for parameter in model.parameters()] == gradients
    optimizer.step()


def test_non_persistent_buffers_are_excluded_from_state_dict():
    module = tx.nn.Module()
    module.register_buffer("persistent", tx.ones([1]))
    module.register_buffer("ephemeral", tx.ones([1]), persistent=False)

    assert [name for name, _ in module.named_buffers()] == [
        "persistent",
        "ephemeral",
    ]
    assert list(module.state_dict()) == ["persistent"]
    result = module.load_state_dict({"persistent": tx.zeros([1])})
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_module_registration_rejects_invalid_reassignment_and_tracks_deletion():
    module = tx.nn.Module()
    module.parameter = tx.nn.Parameter([1.0])
    module.child = tx.nn.Identity()
    module.register_buffer("buffer", tx.ones([1]), persistent=False)

    with pytest.raises(TypeError, match="as parameter"):
        module.parameter = tx.ones([1])
    with pytest.raises(TypeError, match="as parameter"):
        module.parameter = tx.nn.Identity()
    with pytest.raises(TypeError, match="as child module"):
        module.child = tx.ones([1])
    with pytest.raises(TypeError, match="as buffer"):
        module.buffer = 1

    del module.parameter
    del module.child
    del module.buffer
    assert list(module.parameters()) == []
    assert list(module.children()) == []
    assert list(module.buffers()) == []


def test_explicit_registration_does_not_overwrite_plain_attributes():
    module = tx.nn.Module()
    module.existing = "value"

    with pytest.raises(KeyError, match="already exists"):
        module.register_parameter("existing", tx.nn.Parameter([1.0]))
    with pytest.raises(KeyError, match="already exists"):
        module.register_buffer("existing", tx.ones([1]))
    with pytest.raises(KeyError, match="already exists"):
        module.add_module("existing", tx.nn.Identity())
