"""Three-class linear classifier trained with softmax cross-entropy.

Run:
  cargo run --features python -- python examples/python/softmax_classifier.py
"""

import functools

import talos_xii as tx

print = functools.partial(print, flush=True)

# Simple 2D points: class 0 = left, 1 = middle band, 2 = right
FEATURES = tx.tensor(
    [
        -2.0, 0.0,
        -1.5, 0.5,
        -1.0, -0.5,
        0.0, 0.0,
        0.1, 0.2,
        -0.1, -0.2,
        2.0, 0.0,
        1.8, -0.3,
        1.5, 0.4,
    ],
    [9, 2],
)
LABELS = tx.tensor(
    [
        1.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
    ],
    [9, 3],
)


def add_bias(x, bias):
    return x + tx.broadcast(bias, x.shape)


def sgd_step(param, lr):
    updated = [v - lr * g for v, g in zip(param.to_list(), param.grad())]
    return tx.tensor(updated, param.shape)


def train(steps=300, lr=0.2):
    w = tx.randn([2, 3], seed=1) * 0.1
    b = tx.zeros([1, 3])

    for step in range(steps):
        logits = add_bias(FEATURES.matmul(w), b)
        loss = logits.softmax_cross_entropy_with_logits(LABELS)
        loss.backward()

        w = sgd_step(w, lr)
        b = sgd_step(b, lr)

        if step == 0 or step == steps - 1 or (step + 1) % 100 == 0:
            print(f"step={step + 1:4d} loss={loss.item():.6f}")

    logits = add_bias(FEATURES.matmul(w), b)
    probs = logits.softmax(1)
    print("probabilities (rows = samples):")
    flat = probs.to_list()
    for row in range(9):
        p = flat[row * 3 : (row + 1) * 3]
        print(f"  sample {row}: {[round(v, 3) for v in p]}")


def main():
    print(f"talos_xii={tx.version()}")
    train()


if __name__ == "__main__":
    main()
