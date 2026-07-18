"""Train a small linear model with the PyTorch-style Talos-XII API."""

import sys

import talos_xii as tx


def train(device: str = "cpu", steps: int = 120) -> tx.nn.Module:
    if device.startswith("cuda") and not tx.cuda.is_available():
        raise RuntimeError("CUDA was requested but this Talos-XII build cannot use it")

    tx.manual_seed(23)
    model = tx.nn.Linear(1, 1, device=device)
    optimizer = tx.optim.AdamW(
        model.parameters(), lr=0.05, weight_decay=0.0
    )
    inputs = tx.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], device=device)
    targets = tx.tensor([[-5.0], [-2.0], [1.0], [4.0], [7.0]], device=device)

    initial_loss = model(inputs).mse_loss(targets).item()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model(inputs).mse_loss(targets)
        loss.backward()
        optimizer.step()
    final_loss = model(inputs).mse_loss(targets).item()

    print(f"device={device} initial_loss={initial_loss:.6f} final_loss={final_loss:.6f}")
    print(f"weight={model.weight.cpu().tolist()} bias={model.bias.cpu().tolist()}")
    return model


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    checkpoint = sys.argv[3] if len(sys.argv) > 3 else None
    model = train(device, steps)
    if checkpoint is not None:
        tx.save(model, checkpoint)
        print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
