import sys

import talos_xii as tx


def main():
    target_value = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

    x = tx.tensor([1.0, 2.0], [1, 2])
    w = tx.tensor([0.25, -0.5], [2, 1], requires_grad=True)
    target = tx.tensor([target_value], [1, 1])

    prediction = x.matmul(w) + 0.1
    loss = prediction.mse_loss(target)
    loss.backward()

    print(f"talos_xii={tx.version()}", flush=True)
    print(f"prediction={prediction.item():.6f}", flush=True)
    print(f"loss={loss.item():.6f}", flush=True)
    print(f"grad_w={w.grad.tolist()}", flush=True)


main()
