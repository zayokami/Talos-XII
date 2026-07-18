# PyTorch Compatibility Contract

Talos-XII uses PyTorch eager-mode behavior as the reference contract for its
Python frontend. It is not a drop-in replacement for the complete PyTorch API,
and it does not import or embed PyTorch at runtime.

## Reference baseline

- Behavioral oracle: PyTorch 2.10.x eager mode.
- Python package: talos_xii.
- Execution scope: dense strided tensors on CPU and the existing single-device
  CUDA backend.
- Differential tests: bindings/python/tests/test_torch_differential.py.
- Framework-owned tests: bindings/python/tests/test_tensor_contract.py.

The baseline is versioned deliberately. A future PyTorch release does not
silently redefine Talos-XII behavior; the compatibility matrix and differential
tests must be reviewed before the baseline changes.

## Guaranteed core contract

The following behavior is treated as stable and release-blocking:

- Tensor metadata: tuple shape, ndim, size, first-dimension len, first-class
  dtype and device objects, scalar item, and nested tolist.
- Constructors for scalar and rectangular nested Python data, with optional
  legacy explicit shapes.
- Float64, float32, bfloat16, and int8 storage. Only floating tensors may
  require gradients.
- Scalar arithmetic, right-aligned tensor broadcasting, matrix multiplication
  for the documented 1-D/2-D subset, negative dimensions, reshape with one
  inferred dimension, transpose, flatten, squeeze, unsqueeze, and basic
  integer/positive-step slice indexing.
- Reverse-mode autograd metadata and behavior: requires_grad, is_leaf, grad_fn,
  optional Tensor-valued grad, explicit gradients for non-scalar outputs,
  gradient accumulation, retain_graph, retain_grad, no_grad, enable_grad,
  detach aliasing, differentiable clone, and in-place version checks.
- Explicit device and dtype conversions. Explicit CUDA requests fail when the
  wheel lacks CUDA support or initialization fails; they never silently fall
  back to CPU.
- Typed Python exceptions at validated API boundaries. Unsupported behavior is
  reported as NotImplementedError instead of being approximated silently.

## Deliberate extensions and legacy APIs

- Tensor.to_list() is a Talos-XII legacy flat export. PyTorch-compatible nested
  export is Tensor.tolist().
- Constructors accept the historical explicit shape positional argument.
- gemm, softmax_v2, log_softmax_v2, compressed convolution aliases, and ACHF
  APIs are Talos-XII extensions.
- clear_graph, Tensor-level zero_grad, and the flat top-level compatibility
  functions remain available for existing scripts but are not presented as
  PyTorch APIs.

## Partial or unsupported behavior

- Integer inference currently defaults Python numeric input to float32 because
  int64 storage is not implemented.
- Complex, float16, bool, sparse, quantized-autograd, named, nested, and
  distributed tensors are not implemented.
- General batched PyTorch matmul, negative-step slicing, advanced/boolean
  indexing, view strides, storage offsets, and non-contiguous layouts are not
  implemented. Talos-XII materializes contiguous results.
- Higher-order gradients (create_graph=True), forward-mode AD, hooks, anomaly
  detection, and custom Python autograd functions are not implemented.
- CUDA currently exposes the framework's production single-device backend, not
  PyTorch streams, events, graphs, peer access, or distributed collectives.
- PyTorch pickle/checkpoint binary compatibility is not claimed.

## Running the contract

Build and install the wheel, then run:

    python -m pytest bindings/python/tests/test_tensor_contract.py
    python -m pytest bindings/python/tests/test_torch_differential.py
    python -m pytest bindings/python/tests/test_cuda_tensor_contract.py

The differential suite skips when PyTorch 2.10.x is unavailable. Release CI
always runs the framework-owned contract; the PyTorch oracle job installs the
versioned CPU reference package explicitly. The CUDA contract skips without a
CUDA-enabled wheel and available runtime, while the opt-in GPU CI job requires
all CUDA transfer and backward paths to pass.
