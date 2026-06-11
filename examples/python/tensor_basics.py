"""Talos-XII tensor API quick tour (no autograd).

Run:
  cargo run --features python -- python examples/python/tensor_basics.py
"""

import functools

import talos_xii as tx

print = functools.partial(print, flush=True)


def main():
    print(f"talos_xii {tx.version()}")

    # Constructors
    x = tx.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    zeros = tx.zeros([2, 2])
    ones = tx.ones([3])
    filled = tx.full([2, 2], 3.5)
    seq = tx.arange(0, 6, 2)  # [0, 2, 4]
    identity = tx.eye(3)
    noise = tx.randn([4], seed=7)

    print("x shape/dtype/device:", x.shape, x.dtype, x.device)
    print("x:", x.to_list())
    print("x + ones_like:", tx.ones_like(x).add(x).to_list())
    print("rand [0,1):", tx.rand([3], min=0.0, max=1.0, seed=42).to_list())

    # Element-wise math and activations
    v = tx.tensor([-1.0, 0.0, 2.0, 4.0], [4])
    print("relu:", v.relu().to_list())
    print("gelu:", v.gelu().to_list())
    print("sigmoid:", v.sigmoid().to_list())

    # Reductions along axis
    mat = tx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    print("row sums:", mat.sum(1).to_list())
    print("col means:", mat.mean(0).to_list())
    print("softmax row sums:", mat.softmax(1).sum(1).to_list())

    # Shape ops
    flat = mat.flatten()
    reshaped = tx.reshape(mat, [3, 2])
    transposed = mat.transpose(0, 1)
    cat = tx.concat([x, tx.tensor([5.0, 6.0], [1, 2])], 0)
    print("flatten len:", flat.numel())
    print("reshape:", reshaped.shape, reshaped.to_list())
    print("transpose:", transposed.to_list())
    print("concat:", cat.shape, cat.to_list())

    # Matrix multiply
    a = tx.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = tx.tensor([2.0, 0.0, 1.0, 2.0], [2, 2])
    print("matmul:", a.matmul(b).to_list())

    # Broadcast and slice
    col = tx.tensor([10.0, 20.0], [2, 1])
    broadcasted = tx.broadcast(col, [2, 3])
    sliced = cat.strided_slice([0, 0], [2, 2], [1, 1])
    print("broadcast:", broadcasted.to_list())
    print("slice:", sliced.to_list())

    # Dtype constants
    bf16 = tx.tensor([1.0, 2.0], [2], tx.BF16)
    print("bf16 tensor dtype:", bf16.dtype)


if __name__ == "__main__":
    main()
