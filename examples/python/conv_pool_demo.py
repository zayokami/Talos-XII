"""Minimal conv2d + pooling forward pass.

Run:
  cargo run --features python -- python examples/python/conv_pool_demo.py
"""

import functools

import talos_xii as tx

print = functools.partial(print, flush=True)


def main():
    print(f"talos_xii {tx.version()}")

    # Conv/pool kernels currently require f64 tensors.
    image = tx.tensor(
        [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ],
        [1, 1, 4, 4],
        "f64",
    )

    edge = tx.tensor(
        [
            -1.0, -1.0, -1.0,
            -1.0, 8.0, -1.0,
            -1.0, -1.0, -1.0,
        ],
        [1, 1, 3, 3],
        "f64",
    )

    conv_out = image.conv2d(edge, stride=1, padding=0)
    pooled = conv_out.avg_pool2d(kernel_size=2, stride=2)
    maxed = conv_out.max_pool2d(kernel_size=2, stride=2)

    print("input shape:", image.shape)
    print("conv shape:", conv_out.shape, "values:", conv_out.to_list())
    print("avg pool:", pooled.to_list())
    print("max pool:", maxed.to_list())

    # Depthwise scaling
    dw_weight = tx.tensor([0.5], [1, 1, 1, 1], "f64")
    dw_out = image.depthwise_conv2d(dw_weight)
    print("depthwise x0.5 first 4:", dw_out.to_list()[:4])

    mix = tx.tensor([2.0], [1, 1, 1, 1], "f64")
    mixed = image.conv2d(mix)
    print("1x1 conv x2 corner:", mixed.to_list()[0], mixed.to_list()[15])


if __name__ == "__main__":
    main()
