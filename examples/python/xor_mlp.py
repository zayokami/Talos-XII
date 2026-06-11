"""Two-layer MLP that learns XOR with ReLU hidden units.

Run:
  cargo run --features python -- python examples/python/xor_mlp.py
  cargo run --features python -- python examples/python/xor_mlp.py -- 500 0.5
"""

import functools
import sys

import talos_xii as tx

print = functools.partial(print, flush=True)

X = tx.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])
Y = tx.tensor([0.0, 1.0, 1.0, 0.0], [4, 1])


def add_bias(x, bias):
    return x + tx.broadcast(bias, x.shape)


def sgd_step(param, lr):
    updated = [v - lr * g for v, g in zip(param.to_list(), param.grad())]
    return tx.tensor(updated, param.shape)


def forward(x, w1, b1, w2, b2):
    hidden = add_bias(x.matmul(w1), b1)
    activated = hidden.relu()
    return add_bias(activated.matmul(w2), b2)


def train(steps=400, lr=0.5):
    w1 = tx.randn([2, 4], seed=11) * 0.5
    b1 = tx.zeros([1, 4])
    w2 = tx.randn([4, 1], seed=12) * 0.5
    b2 = tx.zeros([1, 1])

    for step in range(steps):
        pred = forward(X, w1, b1, w2, b2)
        loss = pred.mse_loss(Y)
        loss.backward()

        w1 = sgd_step(w1, lr)
        b1 = sgd_step(b1, lr)
        w2 = sgd_step(w2, lr)
        b2 = sgd_step(b2, lr)

        if step == 0 or step == steps - 1 or (step + 1) % 100 == 0:
            print(f"step={step + 1:4d} loss={loss.item():.6f}")

    pred = forward(X, w1, b1, w2, b2)
    print("XOR predictions (target in parentheses):")
    for i in range(4):
        x0, x1 = X.to_list()[i * 2], X.to_list()[i * 2 + 1]
        p = pred.to_list()[i]
        t = Y.to_list()[i]
        print(f"  ({x0:.0f}, {x1:.0f}) -> {p:.4f}  (want {t:.0f})")


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    print(f"talos_xii={tx.version()} steps={steps} lr={lr}")
    train(steps=steps, lr=lr)


if __name__ == "__main__":
    main()
