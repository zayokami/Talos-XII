"""Fit y ≈ w·x + b with manual SGD using Talos-XII autograd.

Run:
  cargo run --features python -- python examples/python/linear_regression_sgd.py
  cargo run --features python -- python examples/python/linear_regression_sgd.py -- 0.01 200
"""

import functools
import sys

import talos_xii as tx

print = functools.partial(print, flush=True)

# y = 2.5 * x + 0.7
X = tx.tensor([0.0, 1.0, 2.0, 3.0, 4.0], [5, 1])
Y = tx.tensor([0.7, 3.2, 5.7, 8.2, 10.7], [5, 1])


def add_bias(x, bias):
    return x + tx.broadcast(bias, x.shape)


def predict(x, w, b):
    return add_bias(x.matmul(w), b)


def train(steps=150, lr=0.05):
    w = tx.tensor([0.0], [1, 1], requires_grad=True)
    b = tx.tensor([0.0], [1, 1], requires_grad=True)

    for step in range(steps):
        pred = predict(X, w, b)
        loss = pred.mse_loss(Y)
        loss.backward()
        w_data = w.to_list()[0]
        b_data = b.to_list()[0]
        gw = w.grad.to_list()[0]
        gb = b.grad.to_list()[0]

        with tx.no_grad():
            w.fill_(w_data - lr * gw)
            b.fill_(b_data - lr * gb)
        w.zero_grad()
        b.zero_grad()

        if step == 0 or step == steps - 1 or (step + 1) % 50 == 0:
            print(f"step={step + 1:4d} loss={loss.item():.6f} w={w.item():.4f} b={b.item():.4f}")

    final = predict(X, w, b)
    print("predictions:", final.to_list())
    print("targets:    ", Y.to_list())
    return w.item(), b.item()


def main():
    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    print(f"talos_xii={tx.version()} lr={lr} steps={steps}")
    w, b = train(steps=steps, lr=lr)
    print(f"learned w~2.5 -> {w:.4f}, b~0.7 -> {b:.4f}")


if __name__ == "__main__":
    main()
