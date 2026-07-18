use crate::autograd::{Device, Tensor as AutoTensor};
use crate::dtype::Dtype;
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PySystemExit, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyModule, PyTuple};
use std::any::Any;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Once;

static REGISTER_PY_MODULE: Once = Once::new();
const MAX_PY_TENSOR_ELEMENTS: usize = 64 * 1024 * 1024;
const MAX_PY_EXPORT_ELEMENTS: usize = 8 * 1024 * 1024;

pub fn run_script(script: &Path, cwd: Option<&Path>, args: &[String]) -> Result<i32, String> {
    REGISTER_PY_MODULE.call_once(|| {
        pyo3::append_to_inittab!(talos_xii);
    });

    let script_path = std::fs::canonicalize(script)
        .map_err(|err| format!("failed to resolve script path {}: {err}", script.display()))?;
    let script_dir = script_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let source = std::fs::read_to_string(&script_path)
        .map_err(|err| format!("failed to read script {}: {err}", script_path.display()))?;

    let original_cwd = std::env::current_dir()
        .map_err(|err| format!("failed to read current working directory: {err}"))?;
    if let Some(cwd) = cwd {
        std::env::set_current_dir(cwd)
            .map_err(|err| format!("failed to set cwd {}: {err}", cwd.display()))?;
    }

    let result = Python::attach(|py| -> PyResult<i32> {
        let sys = py.import("sys")?;
        let original_argv = sys.getattr("argv")?.call_method0("copy")?;
        let original_path = sys.getattr("path")?.call_method0("copy")?;
        if let Err(err) = configure_python_process(py, &script_path, &script_dir, args) {
            let _ = sys.setattr("argv", &original_argv);
            let _ = sys.setattr("path", &original_path);
            return Err(err);
        }

        let code = CString::new(source)
            .map_err(|_| PyValueError::new_err("script source contains an embedded NUL byte"))?;
        let filename = CString::new(script_path.to_string_lossy().as_bytes())
            .map_err(|_| PyValueError::new_err("script path contains an embedded NUL byte"))?;

        let script_result =
            match PyModule::from_code(py, code.as_c_str(), filename.as_c_str(), c"__main__") {
                Ok(_) => Ok(0),
                Err(err) => {
                    if let Some(code) = system_exit_code(py, &err) {
                        Ok(code)
                    } else {
                        Err(err)
                    }
                }
            };

        let restore_result = sys
            .setattr("argv", &original_argv)
            .and_then(|_| sys.setattr("path", &original_path));
        match (script_result, restore_result) {
            (Ok(code), Ok(())) => Ok(code),
            (Ok(_), Err(err)) | (Err(err), Ok(())) | (Err(err), Err(_)) => Err(err),
        }
    });

    let restore_result = if cwd.is_some() {
        std::env::set_current_dir(&original_cwd)
            .map_err(|err| format!("failed to restore cwd {}: {err}", original_cwd.display()))
    } else {
        Ok(())
    };

    match (result, restore_result) {
        (Ok(code), Ok(())) => Ok(code),
        (Ok(_), Err(err)) => Err(err),
        (Err(err), restore) => {
            Python::attach(|py| err.print(py));
            if let Err(restore_err) = restore {
                Err(format!("Python script failed; additionally, {restore_err}"))
            } else {
                Err("Python script failed".to_string())
            }
        }
    }
}

fn configure_python_process(
    py: Python<'_>,
    script_path: &Path,
    script_dir: &Path,
    args: &[String],
) -> PyResult<()> {
    let sys = py.import("sys")?;
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push(script_path.to_string_lossy().to_string());
    argv.extend(args.iter().cloned());
    sys.setattr("argv", PyList::new(py, argv)?)?;

    let sys_path = sys.getattr("path")?;
    sys_path.call_method1("insert", (0, script_dir.to_string_lossy().to_string()))?;
    Ok(())
}

fn system_exit_code(py: Python<'_>, err: &PyErr) -> Option<i32> {
    if !err.is_instance_of::<PySystemExit>(py) {
        return None;
    }
    let value = err.value(py);
    let Ok(code) = value.getattr("code") else {
        return Some(1);
    };
    if code.is_none() {
        return Some(0);
    }
    code.extract::<i32>().ok().or(Some(1))
}

#[pyclass(name = "Tensor", unsendable, skip_from_py_object)]
#[derive(Clone)]
struct PyTensor {
    inner: AutoTensor,
}

enum TensorOperand<'py> {
    Tensor(PyRef<'py, PyTensor>),
    Scalar(f64),
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, shape, dtype = "f32"))]
    fn new(data: &Bound<'_, PyAny>, shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        make_tensor(data, shape, dtype)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype = "f32"))]
    fn zeros(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        crate::python_bridge::zeros(shape, dtype)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype = "f32"))]
    fn ones(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        crate::python_bridge::ones(shape, dtype)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, fill_value, dtype = "f32"))]
    fn full(shape: Vec<usize>, fill_value: f64, dtype: &str) -> PyResult<Self> {
        crate::python_bridge::full(shape, fill_value, dtype)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, *, min = 0.0, max = 1.0, seed = 42, dtype = "f32"))]
    fn rand(shape: Vec<usize>, min: f64, max: f64, seed: u64, dtype: &str) -> PyResult<Self> {
        crate::python_bridge::py_rand(shape, min, max, seed, dtype)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, *, seed = 42, dtype = "f32"))]
    fn randn(shape: Vec<usize>, seed: u64, dtype: &str) -> PyResult<Self> {
        crate::python_bridge::py_randn(shape, seed, dtype)
    }

    #[staticmethod]
    #[pyo3(signature = (start, stop = None, step = 1.0, dtype = "f32"))]
    fn arange(start: f64, stop: Option<f64>, step: f64, dtype: &str) -> PyResult<Self> {
        crate::python_bridge::arange(start, stop, step, dtype)
    }

    #[staticmethod]
    #[pyo3(signature = (n, m = None, dtype = "f32"))]
    fn eye(n: usize, m: Option<usize>, dtype: &str) -> PyResult<Self> {
        crate::python_bridge::eye(n, m, dtype)
    }

    #[staticmethod]
    fn concat(tensors: &Bound<'_, PyAny>, dim: usize) -> PyResult<Self> {
        let tensors = tensor_sequence(tensors)?;
        validate_concat_tensors(&tensors, dim)?;
        wrap_tensor_op(|| AutoTensor::concat(&tensors, dim))
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_name(self.inner.dtype)
    }

    #[getter]
    fn device(&self) -> &'static str {
        device_name(self.inner.device)
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn to_list(&self) -> PyResult<Vec<f64>> {
        validate_export_len(self.inner.numel(), "to_list")?;
        Ok(self.inner.data_as_f64_vec())
    }

    fn tolist(&self) -> PyResult<Vec<f64>> {
        self.to_list()
    }

    fn data(&self) -> PyResult<Vec<f64>> {
        self.to_list()
    }

    fn grad(&self) -> PyResult<Vec<f64>> {
        validate_export_len(self.inner.numel(), "grad")?;
        Ok(self.inner.grad_to_f64_vec())
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn backward(&self) -> PyResult<()> {
        wrap_unit_op(|| self.inner.backward())
    }

    fn clear_graph(&mut self) {
        self.inner.clear_graph();
    }

    fn detach(&self) -> Self {
        Self::from_tensor(self.inner.detach())
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn fill_(&mut self, value: f64) -> PyResult<()> {
        wrap_unit_op(|| {
            self.inner.fill_(value);
        })
    }

    fn item(&self) -> PyResult<f64> {
        wrap_value_op(|| self.inner.item() as f64)
    }

    fn matmul(&self, rhs: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.matmul(&rhs.inner))
    }

    fn gemm(&self, rhs: &PyTensor) -> PyResult<Self> {
        self.matmul(rhs)
    }

    fn add(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => Ok(&self.inner + &rhs.inner),
            TensorOperand::Scalar(rhs) => Ok(&self.inner + &self.scalar_tensor_like(rhs)),
        })
    }

    fn sub(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => Ok(&self.inner - &rhs.inner),
            TensorOperand::Scalar(rhs) => Ok(&self.inner - &self.scalar_tensor_like(rhs)),
        })
    }

    fn mul(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => Ok(&self.inner * &rhs.inner),
            TensorOperand::Scalar(rhs) => Ok(&self.inner * &self.scalar_tensor_like(rhs)),
        })
    }

    fn div(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => Ok(&self.inner / &rhs.inner),
            TensorOperand::Scalar(rhs) => Ok(&self.inner / &self.scalar_tensor_like(rhs)),
        })
    }

    fn neg(&self) -> PyResult<Self> {
        wrap_tensor_op(|| -&self.inner)
    }

    fn relu(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.relu())
    }

    fn gelu(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.gelu())
    }

    #[pyo3(signature = (dim = None))]
    fn softmax(&self, dim: Option<usize>) -> PyResult<Self> {
        let resolved_dim = resolve_dim_option(&self.inner, dim, "softmax")?;
        wrap_tensor_op(|| self.inner.softmax_dim(resolved_dim))
    }

    #[pyo3(signature = (dim = None))]
    fn softmax_v2(&self, dim: Option<usize>) -> PyResult<Self> {
        self.softmax(dim)
    }

    #[pyo3(signature = (dim = None))]
    fn log_softmax(&self, dim: Option<usize>) -> PyResult<Self> {
        let resolved_dim = resolve_dim_option(&self.inner, dim, "log_softmax")?;
        wrap_tensor_op(|| self.inner.log_softmax_dim(resolved_dim))
    }

    #[pyo3(signature = (dim = None))]
    fn log_softmax_v2(&self, dim: Option<usize>) -> PyResult<Self> {
        self.log_softmax(dim)
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn sum(&self, dim: Option<usize>, keepdim: bool) -> PyResult<Self> {
        validate_dim_option(&self.inner, dim, "sum")?;
        wrap_tensor_op(|| match dim {
            Some(dim) => self.inner.reduce_sum_dim(dim, keepdim),
            None => self.inner.sum(),
        })
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn mean(&self, dim: Option<usize>, keepdim: bool) -> PyResult<Self> {
        validate_dim_option(&self.inner, dim, "mean")?;
        wrap_tensor_op(|| match dim {
            Some(dim) => self.inner.reduce_mean_dim(dim, keepdim),
            None => self.inner.mean(),
        })
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn max(&self, dim: Option<usize>, keepdim: bool) -> PyResult<Self> {
        validate_dim_option(&self.inner, dim, "max")?;
        wrap_tensor_op(|| match dim {
            Some(dim) => self.inner.reduce_max_dim(dim, keepdim),
            None => self.inner.max(),
        })
    }

    #[pyo3(signature = (dim, keepdim = false))]
    fn reduce_sum(&self, dim: usize, keepdim: bool) -> PyResult<Self> {
        validate_dim(&self.inner, dim, "reduce_sum")?;
        wrap_tensor_op(|| self.inner.reduce_sum_dim(dim, keepdim))
    }

    #[pyo3(signature = (dim, keepdim = false))]
    fn reduce_mean(&self, dim: usize, keepdim: bool) -> PyResult<Self> {
        validate_dim(&self.inner, dim, "reduce_mean")?;
        wrap_tensor_op(|| self.inner.reduce_mean_dim(dim, keepdim))
    }

    #[pyo3(signature = (dim, keepdim = false))]
    fn reduce_max(&self, dim: usize, keepdim: bool) -> PyResult<Self> {
        validate_dim(&self.inner, dim, "reduce_max")?;
        wrap_tensor_op(|| self.inner.reduce_max_dim(dim, keepdim))
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn reduce_all(&self, dim: Option<usize>, keepdim: bool) -> PyResult<Self> {
        validate_dim_option(&self.inner, dim, "reduce_all")?;
        wrap_tensor_op(|| match dim {
            Some(dim) => self.inner.reduce_all_dim(dim, keepdim),
            None => self.inner.all(),
        })
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn reduce_any(&self, dim: Option<usize>, keepdim: bool) -> PyResult<Self> {
        validate_dim_option(&self.inner, dim, "reduce_any")?;
        wrap_tensor_op(|| match dim {
            Some(dim) => self.inner.reduce_any_dim(dim, keepdim),
            None => self.inner.any(),
        })
    }

    fn log(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.log())
    }

    fn exp(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.exp())
    }

    fn expm1(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.expm1())
    }

    fn abs(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.abs())
    }

    fn pow(&self, exponent: f64) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.pow(exponent))
    }

    fn sin(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.sin())
    }

    fn cos(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.cos())
    }

    fn acos(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.acos())
    }

    fn asin(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.asin())
    }

    fn asinh(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.asinh())
    }

    fn atan(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.atan())
    }

    fn cosh(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.cosh())
    }

    fn sinh(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.sinh())
    }

    fn erf(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.erf())
    }

    fn erfc(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.erfc())
    }

    fn sqrt(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.sqrt())
    }

    fn rsqrt(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.rsqrt())
    }

    fn tanh(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.tanh())
    }

    fn sigmoid(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.sigmoid())
    }

    fn log1p(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.log1p())
    }

    fn inv(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.inv())
    }

    fn reciprocal(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.reciprocal())
    }

    fn square(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.square())
    }

    fn ceil(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.ceil())
    }

    fn floor(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.floor())
    }

    fn round(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.round())
    }

    fn rint(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.rint())
    }

    fn sign(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.sign())
    }

    #[pyo3(signature = (alpha = 1.0))]
    fn elu(&self, alpha: f64) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.elu(alpha))
    }

    fn selu(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.selu())
    }

    fn relu6(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.relu6())
    }

    fn prelu(&self, weight: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.prelu(&weight.inner))
    }

    fn softplus(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.softplus())
    }

    fn softsign(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.softsign())
    }

    fn clamp(&self, min: f64, max: f64) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.clamp(min, max))
    }

    fn clip(&self, min: f64, max: f64) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.clip(min, max))
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        validate_reshape_shape(&self.inner, &shape)?;
        wrap_tensor_op(|| self.inner.reshape(shape))
    }

    fn flatten(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.flatten())
    }

    fn broadcast(&self, shape: Vec<usize>) -> PyResult<Self> {
        validate_broadcast_shape(&self.inner, &shape)?;
        wrap_tensor_op(|| self.inner.broadcast(shape))
    }

    fn transpose2d(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.transpose2d())
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<Self> {
        validate_dim(&self.inner, dim0, "transpose")?;
        validate_dim(&self.inner, dim1, "transpose")?;
        wrap_tensor_op(|| self.inner.transpose(dim0, dim1))
    }

    fn split(&self, dim: usize, sizes: Vec<usize>) -> PyResult<Vec<Self>> {
        validate_split(&self.inner, dim, &sizes)?;
        wrap_value_op(|| {
            self.inner
                .split(dim, sizes)
                .into_iter()
                .map(Self::from_tensor)
                .collect()
        })
    }

    fn strided_slice(
        &self,
        begin: Vec<usize>,
        end: Vec<usize>,
        strides: Vec<usize>,
    ) -> PyResult<Self> {
        validate_strided_slice(&self.inner, &begin, &end, &strides)?;
        wrap_tensor_op(|| self.inner.strided_slice(begin, end, strides))
    }

    fn maximum(&self, rhs: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.maximum(&rhs.inner))
    }

    fn modulo(&self, rhs: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.modulo(&rhs.inner))
    }

    fn equal(&self, rhs: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.equal(&rhs.inner))
    }

    fn ones_like(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.ones_like())
    }

    fn bias_add(&self, bias: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.bias_add(&bias.inner))
    }

    #[pyo3(signature = (dim = None, eps = 1e-12))]
    fn l2_normalize(&self, dim: Option<usize>, eps: f64) -> PyResult<Self> {
        let resolved_dim = resolve_dim_option(&self.inner, dim, "l2_normalize")?;
        validate_positive_finite(eps, "l2_normalize eps")?;
        wrap_tensor_op(|| self.inner.l2_normalize(resolved_dim, eps))
    }

    #[pyo3(signature = (groups, eps = 1e-5))]
    fn group_norm(&self, groups: usize, eps: f64) -> PyResult<Self> {
        validate_positive_usize(groups, "group_norm groups")?;
        validate_positive_finite(eps, "group_norm eps")?;
        wrap_tensor_op(|| self.inner.group_norm(groups, eps))
    }

    #[pyo3(signature = (eps = 1e-5))]
    fn instance_norm(&self, eps: f64) -> PyResult<Self> {
        validate_positive_finite(eps, "instance_norm eps")?;
        wrap_tensor_op(|| self.inner.instance_norm(eps))
    }

    #[pyo3(signature = (scale, bias, eps = 1e-5))]
    fn batch_norm2d(&self, scale: &PyTensor, bias: &PyTensor, eps: f64) -> PyResult<Self> {
        validate_positive_finite(eps, "batch_norm2d eps")?;
        wrap_tensor_op(|| self.inner.batch_norm2d(&scale.inner, &bias.inner, eps))
    }

    #[pyo3(signature = (kernel_size, stride, padding = 0, count_include_pad = false))]
    fn avg_pool2d(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> PyResult<Self> {
        validate_pool_args(kernel_size, stride, "avg_pool2d")?;
        wrap_tensor_op(|| {
            self.inner
                .avg_pool2d(kernel_size, stride, padding, count_include_pad)
        })
    }

    #[pyo3(signature = (kernel_size, stride, padding = 0, count_include_pad = false))]
    fn avg_pool(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> PyResult<Self> {
        self.avg_pool2d(kernel_size, stride, padding, count_include_pad)
    }

    #[pyo3(signature = (kernel_size, stride, padding = 0))]
    fn max_pool2d(&self, kernel_size: usize, stride: usize, padding: usize) -> PyResult<Self> {
        validate_pool_args(kernel_size, stride, "max_pool2d")?;
        wrap_tensor_op(|| self.inner.max_pool2d(kernel_size, stride, padding))
    }

    #[pyo3(signature = (kind, kernel_size, stride, padding = 0))]
    fn pooling(
        &self,
        kind: &str,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> PyResult<Self> {
        validate_pool_args(kernel_size, stride, "pooling")?;
        match kind.to_ascii_lowercase().as_str() {
            "avg" | "average" | "mean" => self.avg_pool2d(kernel_size, stride, padding, false),
            "max" => self.max_pool2d(kernel_size, stride, padding),
            _ => Err(PyValueError::new_err(
                "pooling kind must be 'max', 'avg', 'average', or 'mean'",
            )),
        }
    }

    #[pyo3(signature = (weight, stride = 1, padding = 0))]
    fn conv2d(&self, weight: &PyTensor, stride: usize, padding: usize) -> PyResult<Self> {
        validate_positive_usize(stride, "conv2d stride")?;
        wrap_tensor_op(|| self.inner.conv2d(&weight.inner, stride, padding))
    }

    #[pyo3(signature = (weight, stride = 1, padding = 0))]
    fn conv2d_compress(&self, weight: &PyTensor, stride: usize, padding: usize) -> PyResult<Self> {
        validate_positive_usize(stride, "conv2d_compress stride")?;
        wrap_tensor_op(|| self.inner.conv2d_compress(&weight.inner, stride, padding))
    }

    #[pyo3(signature = (weight, stride = 1, padding = 0))]
    fn conv2d_transpose(&self, weight: &PyTensor, stride: usize, padding: usize) -> PyResult<Self> {
        validate_positive_usize(stride, "conv2d_transpose stride")?;
        wrap_tensor_op(|| self.inner.conv2d_transpose(&weight.inner, stride, padding))
    }

    #[pyo3(signature = (weight, stride = 1, padding = 0))]
    fn deconvolution(&self, weight: &PyTensor, stride: usize, padding: usize) -> PyResult<Self> {
        validate_positive_usize(stride, "deconvolution stride")?;
        wrap_tensor_op(|| self.inner.deconvolution(&weight.inner, stride, padding))
    }

    #[pyo3(signature = (weight, stride = 1, padding = 0))]
    fn depthwise_conv2d(&self, weight: &PyTensor, stride: usize, padding: usize) -> PyResult<Self> {
        validate_positive_usize(stride, "depthwise_conv2d stride")?;
        wrap_tensor_op(|| self.inner.depthwise_conv2d(&weight.inner, stride, padding))
    }

    #[pyo3(signature = (weight, stride = 1, padding = 0))]
    fn conv3d(&self, weight: &PyTensor, stride: usize, padding: usize) -> PyResult<Self> {
        validate_positive_usize(stride, "conv3d stride")?;
        wrap_tensor_op(|| self.inner.conv3d(&weight.inner, stride, padding))
    }

    fn mse_loss(&self, target: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.mse_loss(&target.inner))
    }

    fn weighted_mse_loss(&self, target: &PyTensor, weights: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.weighted_mse_loss(&target.inner, &weights.inner))
    }

    fn l2_loss(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.l2_loss())
    }

    #[pyo3(signature = (target, beta = 1.0))]
    fn smooth_l1_loss(&self, target: &PyTensor, beta: f64) -> PyResult<Self> {
        validate_positive_finite(beta, "smooth_l1_loss beta")?;
        wrap_tensor_op(|| self.inner.smooth_l1_loss(&target.inner, beta))
    }

    fn sigmoid_cross_entropy_with_logits(&self, target: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.sigmoid_cross_entropy_with_logits(&target.inner))
    }

    fn softmax_cross_entropy_with_logits(&self, target: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.softmax_cross_entropy_with_logits(&target.inner))
    }

    #[pyo3(signature = (other, target, margin = 0.0))]
    fn cosine_embedding_loss(
        &self,
        other: &PyTensor,
        target: &PyTensor,
        margin: f64,
    ) -> PyResult<Self> {
        wrap_tensor_op(|| {
            self.inner
                .cosine_embedding_loss(&other.inner, &target.inner, margin)
        })
    }

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.add(rhs)
    }

    fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => Ok(&lhs.inner + &self.inner),
            TensorOperand::Scalar(lhs) => Ok(&self.scalar_tensor_like(lhs) + &self.inner),
        })
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.sub(rhs)
    }

    fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => Ok(&lhs.inner - &self.inner),
            TensorOperand::Scalar(lhs) => Ok(&self.scalar_tensor_like(lhs) - &self.inner),
        })
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.mul(rhs)
    }

    fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => Ok(&lhs.inner * &self.inner),
            TensorOperand::Scalar(lhs) => Ok(&self.scalar_tensor_like(lhs) * &self.inner),
        })
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.div(rhs)
    }

    fn __rtruediv__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => Ok(&lhs.inner / &self.inner),
            TensorOperand::Scalar(lhs) => Ok(&self.scalar_tensor_like(lhs) / &self.inner),
        })
    }

    fn __mod__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => Ok(self.inner.modulo(&rhs.inner)),
            TensorOperand::Scalar(rhs) => Ok(self.inner.modulo(&self.scalar_tensor_like(rhs))),
        })
    }

    fn __rmod__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => Ok(lhs.inner.modulo(&self.inner)),
            TensorOperand::Scalar(lhs) => Ok(self.scalar_tensor_like(lhs).modulo(&self.inner)),
        })
    }

    fn __neg__(&self) -> PyResult<Self> {
        self.neg()
    }

    fn __abs__(&self) -> PyResult<Self> {
        self.abs()
    }

    fn __pow__(
        &self,
        exponent: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if let Some(modulo) = modulo {
            if !modulo.is_none() {
                return Err(PyValueError::new_err(
                    "Tensor.__pow__ does not support the modulo argument",
                ));
            }
        }
        let exponent = exponent.extract::<f64>().map_err(|_| {
            PyTypeError::new_err("Tensor.__pow__ expects a numeric scalar exponent")
        })?;
        self.pow(exponent)
    }

    fn __len__(&self) -> usize {
        self.inner.numel()
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype='{}', device='{}', numel={})",
            self.inner.shape,
            dtype_name(self.inner.dtype),
            device_name(self.inner.device),
            self.inner.numel()
        )
    }
}

impl PyTensor {
    fn from_tensor(inner: AutoTensor) -> Self {
        Self { inner }
    }

    fn scalar_tensor_like(&self, value: f64) -> AutoTensor {
        AutoTensor::with_dtype(
            vec![value; self.inner.numel()],
            self.inner.shape.clone(),
            self.inner.dtype,
        )
    }
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
#[pyo3(signature = (data, shape, dtype = "f32"))]
fn tensor(data: &Bound<'_, PyAny>, shape: Vec<usize>, dtype: &str) -> PyResult<PyTensor> {
    make_tensor(data, shape, dtype)
}

#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype = "f32"))]
fn full(shape: Vec<usize>, fill_value: f64, dtype: &str) -> PyResult<PyTensor> {
    let dtype = parse_dtype(dtype)?;
    let len = checked_numel(&shape)?;
    Ok(PyTensor::from_tensor(AutoTensor::with_dtype(
        vec![fill_value; len],
        shape,
        dtype,
    )))
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = "f32"))]
fn zeros(shape: Vec<usize>, dtype: &str) -> PyResult<PyTensor> {
    let dtype = parse_dtype(dtype)?;
    let len = checked_numel(&shape)?;
    Ok(PyTensor::from_tensor(AutoTensor::with_dtype(
        vec![0.0; len],
        shape,
        dtype,
    )))
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = "f32"))]
fn ones(shape: Vec<usize>, dtype: &str) -> PyResult<PyTensor> {
    let dtype = parse_dtype(dtype)?;
    let len = checked_numel(&shape)?;
    Ok(PyTensor::from_tensor(AutoTensor::with_dtype(
        vec![1.0; len],
        shape,
        dtype,
    )))
}

#[pyfunction(name = "rand", signature = (shape, *, min = 0.0, max = 1.0, seed = 42, dtype = "f32"))]
fn py_rand(shape: Vec<usize>, min: f64, max: f64, seed: u64, dtype: &str) -> PyResult<PyTensor> {
    let dtype = parse_dtype(dtype)?;
    let len = checked_numel(&shape)?;
    let mut rng = crate::rng::Rng::from_seed(seed);
    let data: Vec<f64> = (0..len)
        .map(|_| {
            let t = rng.next_f64();
            min + t * (max - min)
        })
        .collect();
    Ok(PyTensor::from_tensor(AutoTensor::with_dtype(
        data, shape, dtype,
    )))
}

#[pyfunction(name = "randn", signature = (shape, *, seed = 42, dtype = "f32"))]
fn py_randn(shape: Vec<usize>, seed: u64, dtype: &str) -> PyResult<PyTensor> {
    let dtype = parse_dtype(dtype)?;
    let len = checked_numel(&shape)?;
    let mut rng = crate::rng::Rng::from_seed(seed);
    let data: Vec<f64> = (0..len).map(|_| rng.next_f64_normal()).collect();
    Ok(PyTensor::from_tensor(AutoTensor::with_dtype(
        data, shape, dtype,
    )))
}

#[pyfunction]
#[pyo3(signature = (start, stop = None, step = 1.0, dtype = "f32"))]
fn arange(start: f64, stop: Option<f64>, step: f64, dtype: &str) -> PyResult<PyTensor> {
    if step == 0.0 || !step.is_finite() {
        return Err(PyValueError::new_err(
            "arange step must be finite and non-zero",
        ));
    }
    if !start.is_finite() || stop.is_some_and(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "arange start/stop values must be finite",
        ));
    }

    let dtype = parse_dtype(dtype)?;
    let (start, stop) = match stop {
        Some(stop) => (start, stop),
        None => (0.0, start),
    };
    let len_estimate = if step > 0.0 {
        if start >= stop {
            0usize
        } else {
            checked_float_len(((stop - start) / step).ceil(), "arange")?
        }
    } else if start <= stop {
        0usize
    } else {
        checked_float_len(((start - stop) / -step).ceil(), "arange")?
    };
    let mut data = Vec::new();
    data.try_reserve_exact(len_estimate)
        .map_err(|_| PyMemoryError::new_err("arange allocation would exceed available memory"))?;
    for idx in 0..len_estimate {
        let value = start + step * idx as f64;
        if !value.is_finite() {
            return Err(PyValueError::new_err("arange generated a non-finite value"));
        }
        data.push(value);
    }
    let len = data.len();
    Ok(PyTensor::from_tensor(AutoTensor::with_dtype(
        data,
        vec![len],
        dtype,
    )))
}

#[pyfunction]
#[pyo3(signature = (n, m = None, dtype = "f32"))]
fn eye(n: usize, m: Option<usize>, dtype: &str) -> PyResult<PyTensor> {
    let dtype = parse_dtype(dtype)?;
    let cols = m.unwrap_or(n);
    let len = n
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("eye shape element count overflow"))?;
    validate_tensor_len(len, "eye")?;
    let mut data = vec![0.0; len];
    for idx in 0..n.min(cols) {
        data[idx * cols + idx] = 1.0;
    }
    Ok(PyTensor::from_tensor(AutoTensor::with_dtype(
        data,
        vec![n, cols],
        dtype,
    )))
}

macro_rules! tensor_unary_pyfunction {
    ($name:ident) => {
        #[pyfunction]
        fn $name(input: &PyTensor) -> PyResult<PyTensor> {
            input.$name()
        }
    };
}

#[pyfunction]
fn numel(input: &PyTensor) -> usize {
    input.numel()
}

#[pyfunction]
fn shape(input: &PyTensor) -> Vec<usize> {
    input.shape()
}

#[pyfunction]
fn dtype(input: &PyTensor) -> &'static str {
    input.dtype()
}

#[pyfunction]
fn device(input: &PyTensor) -> &'static str {
    input.device()
}

#[pyfunction]
fn to_list(input: &PyTensor) -> PyResult<Vec<f64>> {
    input.to_list()
}

#[pyfunction]
fn tolist(input: &PyTensor) -> PyResult<Vec<f64>> {
    input.tolist()
}

#[pyfunction]
fn data(input: &PyTensor) -> PyResult<Vec<f64>> {
    input.data()
}

#[pyfunction]
fn grad(input: &PyTensor) -> PyResult<Vec<f64>> {
    input.grad()
}

#[pyfunction]
fn zero_grad(input: &PyTensor) {
    input.zero_grad();
}

#[pyfunction]
fn backward(input: &PyTensor) -> PyResult<()> {
    input.backward()
}

#[pyfunction]
fn clear_graph(mut input: PyRefMut<'_, PyTensor>) {
    input.clear_graph();
}

#[pyfunction]
fn detach(input: &PyTensor) -> PyTensor {
    input.detach()
}

#[pyfunction]
fn copy(input: &PyTensor) -> PyTensor {
    input.copy()
}

#[pyfunction]
fn fill_(mut input: PyRefMut<'_, PyTensor>, value: f64) -> PyResult<()> {
    input.fill_(value)
}

#[pyfunction]
fn item(input: &PyTensor) -> PyResult<f64> {
    input.item()
}

#[pyfunction]
fn add(lhs: &PyTensor, rhs: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    lhs.add(rhs)
}

#[pyfunction]
fn sub(lhs: &PyTensor, rhs: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    lhs.sub(rhs)
}

#[pyfunction]
fn mul(lhs: &PyTensor, rhs: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    lhs.mul(rhs)
}

#[pyfunction]
fn div(lhs: &PyTensor, rhs: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    lhs.div(rhs)
}

#[pyfunction]
fn neg(input: &PyTensor) -> PyResult<PyTensor> {
    input.neg()
}

tensor_unary_pyfunction!(relu);
tensor_unary_pyfunction!(gelu);
tensor_unary_pyfunction!(exp);
tensor_unary_pyfunction!(expm1);
tensor_unary_pyfunction!(abs);
tensor_unary_pyfunction!(sin);
tensor_unary_pyfunction!(cos);
tensor_unary_pyfunction!(acos);
tensor_unary_pyfunction!(asin);
tensor_unary_pyfunction!(asinh);
tensor_unary_pyfunction!(atan);
tensor_unary_pyfunction!(cosh);
tensor_unary_pyfunction!(sinh);
tensor_unary_pyfunction!(erf);
tensor_unary_pyfunction!(erfc);
tensor_unary_pyfunction!(sqrt);
tensor_unary_pyfunction!(rsqrt);
tensor_unary_pyfunction!(tanh);
tensor_unary_pyfunction!(sigmoid);
tensor_unary_pyfunction!(log1p);
tensor_unary_pyfunction!(inv);
tensor_unary_pyfunction!(reciprocal);
tensor_unary_pyfunction!(square);
tensor_unary_pyfunction!(ceil);
tensor_unary_pyfunction!(floor);
tensor_unary_pyfunction!(round);
tensor_unary_pyfunction!(rint);
tensor_unary_pyfunction!(sign);
tensor_unary_pyfunction!(selu);
tensor_unary_pyfunction!(relu6);
tensor_unary_pyfunction!(softplus);
tensor_unary_pyfunction!(softsign);
tensor_unary_pyfunction!(flatten);
tensor_unary_pyfunction!(transpose2d);

#[pyfunction(name = "log")]
fn py_log(input: &PyTensor) -> PyResult<PyTensor> {
    input.log()
}

#[pyfunction]
fn pow(input: &PyTensor, exponent: f64) -> PyResult<PyTensor> {
    input.pow(exponent)
}

#[pyfunction]
#[pyo3(signature = (input, alpha = 1.0))]
fn elu(input: &PyTensor, alpha: f64) -> PyResult<PyTensor> {
    input.elu(alpha)
}

#[pyfunction]
fn prelu(input: &PyTensor, weight: &PyTensor) -> PyResult<PyTensor> {
    input.prelu(weight)
}

#[pyfunction]
fn clamp(input: &PyTensor, min: f64, max: f64) -> PyResult<PyTensor> {
    input.clamp(min, max)
}

#[pyfunction]
fn clip(input: &PyTensor, min: f64, max: f64) -> PyResult<PyTensor> {
    input.clip(min, max)
}

#[pyfunction]
fn reshape(input: &PyTensor, shape: Vec<usize>) -> PyResult<PyTensor> {
    input.reshape(shape)
}

#[pyfunction]
fn broadcast(input: &PyTensor, shape: Vec<usize>) -> PyResult<PyTensor> {
    input.broadcast(shape)
}

#[pyfunction]
fn transpose(input: &PyTensor, dim0: usize, dim1: usize) -> PyResult<PyTensor> {
    input.transpose(dim0, dim1)
}

#[pyfunction]
fn split(input: &PyTensor, dim: usize, sizes: Vec<usize>) -> PyResult<Vec<PyTensor>> {
    input.split(dim, sizes)
}

#[pyfunction]
fn strided_slice(
    input: &PyTensor,
    begin: Vec<usize>,
    end: Vec<usize>,
    strides: Vec<usize>,
) -> PyResult<PyTensor> {
    input.strided_slice(begin, end, strides)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None, keepdim = false))]
fn sum(input: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    input.sum(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None, keepdim = false))]
fn mean(input: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    input.mean(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None, keepdim = false))]
fn max(input: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    input.max(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim, keepdim = false))]
fn reduce_sum(input: &PyTensor, dim: usize, keepdim: bool) -> PyResult<PyTensor> {
    input.reduce_sum(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim, keepdim = false))]
fn reduce_mean(input: &PyTensor, dim: usize, keepdim: bool) -> PyResult<PyTensor> {
    input.reduce_mean(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim, keepdim = false))]
fn reduce_max(input: &PyTensor, dim: usize, keepdim: bool) -> PyResult<PyTensor> {
    input.reduce_max(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None, keepdim = false))]
fn reduce_all(input: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    input.reduce_all(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None, keepdim = false))]
fn reduce_any(input: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    input.reduce_any(dim, keepdim)
}

#[pyfunction]
fn concat(tensors: &Bound<'_, PyAny>, dim: usize) -> PyResult<PyTensor> {
    let tensors = tensor_sequence(tensors)?;
    validate_concat_tensors(&tensors, dim)?;
    wrap_tensor_op(|| AutoTensor::concat(&tensors, dim))
}

#[pyfunction]
fn ones_like(tensor: &PyTensor) -> PyResult<PyTensor> {
    wrap_tensor_op(|| tensor.inner.ones_like())
}

#[pyfunction]
fn bias_add(input: &PyTensor, bias: &PyTensor) -> PyResult<PyTensor> {
    input.bias_add(bias)
}

#[pyfunction]
fn maximum(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    wrap_tensor_op(|| lhs.inner.maximum(&rhs.inner))
}

#[pyfunction]
fn equal(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    wrap_tensor_op(|| lhs.inner.equal(&rhs.inner))
}

#[pyfunction]
fn matmul(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    wrap_tensor_op(|| lhs.inner.matmul(&rhs.inner))
}

#[pyfunction]
fn gemm(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    matmul(lhs, rhs)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None))]
fn softmax(input: &PyTensor, dim: Option<usize>) -> PyResult<PyTensor> {
    let resolved_dim = resolve_dim_option(&input.inner, dim, "softmax")?;
    wrap_tensor_op(|| input.inner.softmax_dim(resolved_dim))
}

#[pyfunction]
#[pyo3(signature = (input, dim = None))]
fn softmax_v2(input: &PyTensor, dim: Option<usize>) -> PyResult<PyTensor> {
    softmax(input, dim)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None))]
fn log_softmax(input: &PyTensor, dim: Option<usize>) -> PyResult<PyTensor> {
    let resolved_dim = resolve_dim_option(&input.inner, dim, "log_softmax")?;
    wrap_tensor_op(|| input.inner.log_softmax_dim(resolved_dim))
}

#[pyfunction]
#[pyo3(signature = (input, dim = None))]
fn log_softmax_v2(input: &PyTensor, dim: Option<usize>) -> PyResult<PyTensor> {
    log_softmax(input, dim)
}

#[pyfunction]
fn modulo(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    wrap_tensor_op(|| lhs.inner.modulo(&rhs.inner))
}

#[pyfunction]
fn l2_loss(input: &PyTensor) -> PyResult<PyTensor> {
    wrap_tensor_op(|| input.inner.l2_loss())
}

#[pyfunction]
fn mse_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    wrap_tensor_op(|| input.inner.mse_loss(&target.inner))
}

#[pyfunction]
fn weighted_mse_loss(
    input: &PyTensor,
    target: &PyTensor,
    weights: &PyTensor,
) -> PyResult<PyTensor> {
    wrap_tensor_op(|| input.inner.weighted_mse_loss(&target.inner, &weights.inner))
}

#[pyfunction]
#[pyo3(signature = (input, target, beta = 1.0))]
fn smooth_l1_loss(input: &PyTensor, target: &PyTensor, beta: f64) -> PyResult<PyTensor> {
    validate_positive_finite(beta, "smooth_l1_loss beta")?;
    wrap_tensor_op(|| input.inner.smooth_l1_loss(&target.inner, beta))
}

#[pyfunction]
fn sigmoid_cross_entropy_with_logits(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    wrap_tensor_op(|| input.inner.sigmoid_cross_entropy_with_logits(&target.inner))
}

#[pyfunction]
fn softmax_cross_entropy_with_logits(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    wrap_tensor_op(|| input.inner.softmax_cross_entropy_with_logits(&target.inner))
}

#[pyfunction]
#[pyo3(signature = (input, other, target, margin = 0.0))]
fn cosine_embedding_loss(
    input: &PyTensor,
    other: &PyTensor,
    target: &PyTensor,
    margin: f64,
) -> PyResult<PyTensor> {
    wrap_tensor_op(|| {
        input
            .inner
            .cosine_embedding_loss(&other.inner, &target.inner, margin)
    })
}

#[pyfunction]
#[pyo3(signature = (input, weight, stride = 1, padding = 0))]
fn conv2d(
    input: &PyTensor,
    weight: &PyTensor,
    stride: usize,
    padding: usize,
) -> PyResult<PyTensor> {
    validate_positive_usize(stride, "conv2d stride")?;
    wrap_tensor_op(|| input.inner.conv2d(&weight.inner, stride, padding))
}

#[pyfunction]
#[pyo3(signature = (input, weight, stride = 1, padding = 0))]
fn conv2d_compress(
    input: &PyTensor,
    weight: &PyTensor,
    stride: usize,
    padding: usize,
) -> PyResult<PyTensor> {
    validate_positive_usize(stride, "conv2d_compress stride")?;
    wrap_tensor_op(|| input.inner.conv2d_compress(&weight.inner, stride, padding))
}

#[pyfunction]
#[pyo3(signature = (input, weight, stride = 1, padding = 0))]
fn conv2d_transpose(
    input: &PyTensor,
    weight: &PyTensor,
    stride: usize,
    padding: usize,
) -> PyResult<PyTensor> {
    validate_positive_usize(stride, "conv2d_transpose stride")?;
    wrap_tensor_op(|| input.inner.conv2d_transpose(&weight.inner, stride, padding))
}

#[pyfunction]
#[pyo3(signature = (input, weight, stride = 1, padding = 0))]
fn deconvolution(
    input: &PyTensor,
    weight: &PyTensor,
    stride: usize,
    padding: usize,
) -> PyResult<PyTensor> {
    validate_positive_usize(stride, "deconvolution stride")?;
    wrap_tensor_op(|| input.inner.deconvolution(&weight.inner, stride, padding))
}

#[pyfunction]
#[pyo3(signature = (input, weight, stride = 1, padding = 0))]
fn depthwise_conv2d(
    input: &PyTensor,
    weight: &PyTensor,
    stride: usize,
    padding: usize,
) -> PyResult<PyTensor> {
    validate_positive_usize(stride, "depthwise_conv2d stride")?;
    wrap_tensor_op(|| input.inner.depthwise_conv2d(&weight.inner, stride, padding))
}

#[pyfunction]
#[pyo3(signature = (input, weight, stride = 1, padding = 0))]
fn conv3d(
    input: &PyTensor,
    weight: &PyTensor,
    stride: usize,
    padding: usize,
) -> PyResult<PyTensor> {
    validate_positive_usize(stride, "conv3d stride")?;
    wrap_tensor_op(|| input.inner.conv3d(&weight.inner, stride, padding))
}

#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride, padding = 0, count_include_pad = false))]
fn avg_pool2d(
    input: &PyTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
) -> PyResult<PyTensor> {
    validate_pool_args(kernel_size, stride, "avg_pool2d")?;
    wrap_tensor_op(|| {
        input
            .inner
            .avg_pool2d(kernel_size, stride, padding, count_include_pad)
    })
}

#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride, padding = 0, count_include_pad = false))]
fn avg_pool(
    input: &PyTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
) -> PyResult<PyTensor> {
    input.avg_pool(kernel_size, stride, padding, count_include_pad)
}

#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride, padding = 0))]
fn max_pool2d(
    input: &PyTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> PyResult<PyTensor> {
    validate_pool_args(kernel_size, stride, "max_pool2d")?;
    wrap_tensor_op(|| input.inner.max_pool2d(kernel_size, stride, padding))
}

#[pyfunction]
#[pyo3(signature = (input, kind, kernel_size, stride, padding = 0))]
fn pooling(
    input: &PyTensor,
    kind: &str,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> PyResult<PyTensor> {
    input.pooling(kind, kernel_size, stride, padding)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None, eps = 1e-12))]
fn l2_normalize(input: &PyTensor, dim: Option<usize>, eps: f64) -> PyResult<PyTensor> {
    let resolved_dim = resolve_dim_option(&input.inner, dim, "l2_normalize")?;
    validate_positive_finite(eps, "l2_normalize eps")?;
    wrap_tensor_op(|| input.inner.l2_normalize(resolved_dim, eps))
}

#[pyfunction]
#[pyo3(signature = (input, groups, eps = 1e-5))]
fn group_norm(input: &PyTensor, groups: usize, eps: f64) -> PyResult<PyTensor> {
    validate_positive_usize(groups, "group_norm groups")?;
    validate_positive_finite(eps, "group_norm eps")?;
    wrap_tensor_op(|| input.inner.group_norm(groups, eps))
}

#[pyfunction]
#[pyo3(signature = (input, eps = 1e-5))]
fn instance_norm(input: &PyTensor, eps: f64) -> PyResult<PyTensor> {
    validate_positive_finite(eps, "instance_norm eps")?;
    wrap_tensor_op(|| input.inner.instance_norm(eps))
}

#[pyfunction]
#[pyo3(signature = (input, scale, bias, eps = 1e-5))]
fn batch_norm2d(
    input: &PyTensor,
    scale: &PyTensor,
    bias: &PyTensor,
    eps: f64,
) -> PyResult<PyTensor> {
    validate_positive_finite(eps, "batch_norm2d eps")?;
    wrap_tensor_op(|| input.inner.batch_norm2d(&scale.inner, &bias.inner, eps))
}

#[pymodule]
pub fn talos_xii(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("F32", "f32")?;
    m.add("F64", "f64")?;
    m.add("BF16", "bf16")?;
    m.add("I8", "i8")?;
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(py_rand, m)?)?;
    m.add_function(wrap_pyfunction!(py_randn, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(numel, m)?)?;
    m.add_function(wrap_pyfunction!(shape, m)?)?;
    m.add_function(wrap_pyfunction!(dtype, m)?)?;
    m.add_function(wrap_pyfunction!(device, m)?)?;
    m.add_function(wrap_pyfunction!(to_list, m)?)?;
    m.add_function(wrap_pyfunction!(tolist, m)?)?;
    m.add_function(wrap_pyfunction!(data, m)?)?;
    m.add_function(wrap_pyfunction!(grad, m)?)?;
    m.add_function(wrap_pyfunction!(zero_grad, m)?)?;
    m.add_function(wrap_pyfunction!(backward, m)?)?;
    m.add_function(wrap_pyfunction!(clear_graph, m)?)?;
    m.add_function(wrap_pyfunction!(detach, m)?)?;
    m.add_function(wrap_pyfunction!(copy, m)?)?;
    m.add_function(wrap_pyfunction!(fill_, m)?)?;
    m.add_function(wrap_pyfunction!(item, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(neg, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(py_log, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(expm1, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(acos, m)?)?;
    m.add_function(wrap_pyfunction!(asin, m)?)?;
    m.add_function(wrap_pyfunction!(asinh, m)?)?;
    m.add_function(wrap_pyfunction!(atan, m)?)?;
    m.add_function(wrap_pyfunction!(cosh, m)?)?;
    m.add_function(wrap_pyfunction!(sinh, m)?)?;
    m.add_function(wrap_pyfunction!(erf, m)?)?;
    m.add_function(wrap_pyfunction!(erfc, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(rsqrt, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(log1p, m)?)?;
    m.add_function(wrap_pyfunction!(inv, m)?)?;
    m.add_function(wrap_pyfunction!(reciprocal, m)?)?;
    m.add_function(wrap_pyfunction!(square, m)?)?;
    m.add_function(wrap_pyfunction!(ceil, m)?)?;
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(round, m)?)?;
    m.add_function(wrap_pyfunction!(rint, m)?)?;
    m.add_function(wrap_pyfunction!(sign, m)?)?;
    m.add_function(wrap_pyfunction!(relu6, m)?)?;
    m.add_function(wrap_pyfunction!(selu, m)?)?;
    m.add_function(wrap_pyfunction!(softplus, m)?)?;
    m.add_function(wrap_pyfunction!(softsign, m)?)?;
    m.add_function(wrap_pyfunction!(pow, m)?)?;
    m.add_function(wrap_pyfunction!(elu, m)?)?;
    m.add_function(wrap_pyfunction!(prelu, m)?)?;
    m.add_function(wrap_pyfunction!(clamp, m)?)?;
    m.add_function(wrap_pyfunction!(clip, m)?)?;
    m.add_function(wrap_pyfunction!(reshape, m)?)?;
    m.add_function(wrap_pyfunction!(flatten, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(transpose2d, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(strided_slice, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_sum, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_mean, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_max, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_all, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_any, m)?)?;
    m.add_function(wrap_pyfunction!(concat, m)?)?;
    m.add_function(wrap_pyfunction!(ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(bias_add, m)?)?;
    m.add_function(wrap_pyfunction!(maximum, m)?)?;
    m.add_function(wrap_pyfunction!(equal, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(gemm, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(softmax_v2, m)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(log_softmax_v2, m)?)?;
    m.add_function(wrap_pyfunction!(modulo, m)?)?;
    m.add_function(wrap_pyfunction!(l2_loss, m)?)?;
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_cross_entropy_with_logits, m)?)?;
    m.add_function(wrap_pyfunction!(softmax_cross_entropy_with_logits, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_embedding_loss, m)?)?;
    m.add_function(wrap_pyfunction!(conv2d, m)?)?;
    m.add_function(wrap_pyfunction!(conv2d_compress, m)?)?;
    m.add_function(wrap_pyfunction!(conv2d_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(deconvolution, m)?)?;
    m.add_function(wrap_pyfunction!(depthwise_conv2d, m)?)?;
    m.add_function(wrap_pyfunction!(conv3d, m)?)?;
    m.add_function(wrap_pyfunction!(avg_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(avg_pool, m)?)?;
    m.add_function(wrap_pyfunction!(max_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(pooling, m)?)?;
    m.add_function(wrap_pyfunction!(l2_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(group_norm, m)?)?;
    m.add_function(wrap_pyfunction!(instance_norm, m)?)?;
    m.add_function(wrap_pyfunction!(batch_norm2d, m)?)?;
    Ok(())
}

fn make_tensor(data: &Bound<'_, PyAny>, shape: Vec<usize>, dtype: &str) -> PyResult<PyTensor> {
    let dtype = parse_dtype(dtype)?;
    let expected = checked_numel(&shape)?;
    let data_len = data
        .len()
        .map_err(|_| PyTypeError::new_err("tensor data must be a sized sequence of numbers"))?;
    validate_tensor_len(data_len, "tensor data")?;
    if data_len != expected {
        return Err(PyValueError::new_err(format!(
            "data length {} does not match shape {:?} ({} elements)",
            data_len, shape, expected
        )));
    }
    let data = data.extract::<Vec<f64>>()?;
    Ok(PyTensor::from_tensor(AutoTensor::with_dtype(
        data, shape, dtype,
    )))
}

fn parse_dtype(dtype: &str) -> PyResult<Dtype> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "f64" | "float64" | "double" => Ok(Dtype::F64),
        "f32" | "float32" | "float" => Ok(Dtype::F32),
        "bf16" | "bfloat16" => Ok(Dtype::BF16),
        "i8" | "int8" => Ok(Dtype::I8),
        other => Err(PyValueError::new_err(format!(
            "unsupported dtype '{other}', expected one of: f32, f64, bf16, i8"
        ))),
    }
}

fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F64 => "f64",
        Dtype::F32 => "f32",
        Dtype::BF16 => "bf16",
        Dtype::I8 => "i8",
    }
}

fn device_name(device: Device) -> &'static str {
    #[cfg(cuda)]
    if device == Device::Cuda {
        return "cuda";
    }
    let _ = device;
    "cpu"
}

fn checked_numel(shape: &[usize]) -> PyResult<usize> {
    let len = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| PyValueError::new_err("shape element count overflow"))
    })?;
    validate_tensor_len(len, "tensor allocation")?;
    Ok(len)
}

fn checked_float_len(value: f64, op: &str) -> PyResult<usize> {
    if !value.is_finite() || value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{op} produced an invalid element count"
        )));
    }
    if value > usize::MAX as f64 {
        return Err(PyValueError::new_err(format!(
            "{op} element count overflow"
        )));
    }
    let len = value as usize;
    validate_tensor_len(len, op)?;
    Ok(len)
}

fn validate_tensor_len(len: usize, op: &str) -> PyResult<()> {
    if len > MAX_PY_TENSOR_ELEMENTS {
        return Err(PyMemoryError::new_err(format!(
            "{op} would allocate {len} elements; Python bridge limit is {MAX_PY_TENSOR_ELEMENTS}"
        )));
    }
    Ok(())
}

fn validate_export_len(len: usize, op: &str) -> PyResult<()> {
    if len > MAX_PY_EXPORT_ELEMENTS {
        return Err(PyMemoryError::new_err(format!(
            "{op} would export {len} elements; Python list export limit is {MAX_PY_EXPORT_ELEMENTS}"
        )));
    }
    Ok(())
}

fn validate_dim(tensor: &AutoTensor, dim: usize, op: &str) -> PyResult<()> {
    if dim >= tensor.shape.len() {
        return Err(PyValueError::new_err(format!(
            "{op} dim {dim} out of bounds for rank {}",
            tensor.shape.len()
        )));
    }
    Ok(())
}

fn validate_dim_option(tensor: &AutoTensor, dim: Option<usize>, op: &str) -> PyResult<()> {
    if let Some(dim) = dim {
        validate_dim(tensor, dim, op)?;
    }
    Ok(())
}

fn resolve_dim_option(tensor: &AutoTensor, dim: Option<usize>, op: &str) -> PyResult<usize> {
    match dim {
        Some(dim) => {
            validate_dim(tensor, dim, op)?;
            Ok(dim)
        }
        None if tensor.shape.is_empty() => Err(PyValueError::new_err(format!(
            "{op} requires rank >= 1 when dim is not provided"
        ))),
        None => Ok(tensor.shape.len() - 1),
    }
}

fn shapes_broadcastable(old: &[usize], new: &[usize]) -> bool {
    let old_len = old.len();
    let new_len = new.len();
    let max_len = old_len.max(new_len);

    for idx in 0..max_len {
        let old_axis = idx as isize - (max_len as isize - old_len as isize);
        let new_axis = idx as isize - (max_len as isize - new_len as isize);
        let old_dim = if old_axis < 0 {
            1
        } else {
            old[old_axis as usize]
        };
        let new_dim = if new_axis < 0 {
            1
        } else {
            new[new_axis as usize]
        };
        if old_dim != new_dim && old_dim != 1 && new_dim != 1 {
            return false;
        }
    }
    true
}

fn validate_reshape_shape(tensor: &AutoTensor, shape: &[usize]) -> PyResult<()> {
    let len = checked_numel(shape)?;
    if len != tensor.numel() {
        return Err(PyValueError::new_err(format!(
            "reshape size mismatch: tensor has {} elements, requested shape {:?} has {len}",
            tensor.numel(),
            shape
        )));
    }
    Ok(())
}

fn validate_broadcast_shape(tensor: &AutoTensor, shape: &[usize]) -> PyResult<()> {
    checked_numel(shape)?;
    if !shapes_broadcastable(&tensor.shape, shape) {
        return Err(PyValueError::new_err(format!(
            "Shapes {:?} and {:?} are not broadcast-compatible",
            tensor.shape, shape
        )));
    }
    Ok(())
}

fn validate_concat_tensors(tensors: &[AutoTensor], dim: usize) -> PyResult<()> {
    if tensors.is_empty() {
        return Err(PyValueError::new_err("concat requires at least one Tensor"));
    }
    let rank = tensors[0].shape.len();
    if dim >= rank {
        return Err(PyValueError::new_err(format!(
            "concat dim {dim} out of bounds for rank {rank}"
        )));
    }
    let mut shape = tensors[0].shape.clone();
    shape[dim] = 0;
    for tensor in tensors {
        if tensor.shape.len() != rank {
            return Err(PyValueError::new_err("concat rank mismatch"));
        }
        for axis in 0..rank {
            if axis != dim && tensor.shape[axis] != tensors[0].shape[axis] {
                return Err(PyValueError::new_err(format!(
                    "concat shape mismatch at dim {axis}"
                )));
            }
        }
        shape[dim] = shape[dim]
            .checked_add(tensor.shape[dim])
            .ok_or_else(|| PyValueError::new_err("concat output dimension overflow"))?;
    }
    checked_numel(&shape)?;
    Ok(())
}

fn validate_split(tensor: &AutoTensor, dim: usize, sizes: &[usize]) -> PyResult<()> {
    validate_dim(tensor, dim, "split")?;
    if sizes.is_empty() {
        return Err(PyValueError::new_err("split requires at least one output"));
    }
    let total = sizes.iter().try_fold(0usize, |acc, &size| {
        acc.checked_add(size)
            .ok_or_else(|| PyValueError::new_err("split sizes overflow"))
    })?;
    if total != tensor.shape[dim] {
        return Err(PyValueError::new_err(
            "split sizes must sum to the selected dimension",
        ));
    }
    for &size in sizes {
        let mut output_shape = tensor.shape.clone();
        output_shape[dim] = size;
        checked_numel(&output_shape)?;
    }
    Ok(())
}

fn validate_strided_slice(
    tensor: &AutoTensor,
    begin: &[usize],
    end: &[usize],
    strides: &[usize],
) -> PyResult<()> {
    let rank = tensor.shape.len();
    if begin.len() != rank || end.len() != rank || strides.len() != rank {
        return Err(PyValueError::new_err(format!(
            "strided_slice expects begin/end/strides rank {rank}"
        )));
    }
    let mut output_shape = Vec::with_capacity(rank);
    for dim in 0..rank {
        if strides[dim] == 0 {
            return Err(PyValueError::new_err(
                "strided_slice strides must be positive",
            ));
        }
        if begin[dim] > end[dim] || end[dim] > tensor.shape[dim] {
            return Err(PyValueError::new_err(format!(
                "strided_slice bounds invalid at dim {dim}"
            )));
        }
        output_shape.push((end[dim] - begin[dim]).div_ceil(strides[dim]));
    }
    checked_numel(&output_shape)?;
    Ok(())
}

fn validate_positive_usize(value: usize, name: &str) -> PyResult<()> {
    if value == 0 {
        return Err(PyValueError::new_err(format!("{name} must be positive")));
    }
    Ok(())
}

fn validate_positive_finite(value: f64, name: &str) -> PyResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and positive"
        )));
    }
    Ok(())
}

fn validate_pool_args(kernel_size: usize, stride: usize, op: &str) -> PyResult<()> {
    validate_positive_usize(kernel_size, &format!("{op} kernel_size"))?;
    validate_positive_usize(stride, &format!("{op} stride"))?;
    Ok(())
}

fn tensor_operand<'py>(obj: &'py Bound<'py, PyAny>) -> PyResult<TensorOperand<'py>> {
    if let Ok(tensor) = obj.extract::<PyRef<'py, PyTensor>>() {
        return Ok(TensorOperand::Tensor(tensor));
    }
    if let Ok(scalar) = obj.extract::<f64>() {
        return Ok(TensorOperand::Scalar(scalar));
    }
    Err(PyTypeError::new_err("expected Tensor or numeric scalar"))
}

fn tensor_sequence(obj: &Bound<'_, PyAny>) -> PyResult<Vec<AutoTensor>> {
    if let Ok(list) = obj.cast::<PyList>() {
        let mut tensors = Vec::with_capacity(list.len());
        for item in list.iter() {
            let tensor = item.extract::<PyRef<'_, PyTensor>>()?;
            tensors.push(tensor.inner.clone());
        }
        return Ok(tensors);
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let mut tensors = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            let tensor = item.extract::<PyRef<'_, PyTensor>>()?;
            tensors.push(tensor.inner.clone());
        }
        return Ok(tensors);
    }
    Err(PyTypeError::new_err(
        "expected a list or tuple of Tensor objects",
    ))
}

fn wrap_tensor_op<F>(op: F) -> PyResult<PyTensor>
where
    F: FnOnce() -> AutoTensor,
{
    wrap_value_op(|| PyTensor::from_tensor(op()))
}

fn wrap_tensor_result_op<F>(op: F) -> PyResult<PyTensor>
where
    F: FnOnce() -> PyResult<AutoTensor>,
{
    wrap_value_op(|| op().map(PyTensor::from_tensor))?
}

fn wrap_unit_op<F>(op: F) -> PyResult<()>
where
    F: FnOnce(),
{
    wrap_value_op(|| {
        op();
    })
}

fn wrap_value_op<T, F>(op: F) -> PyResult<T>
where
    F: FnOnce() -> T,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(op)).map_err(|payload| {
        PyRuntimeError::new_err(format!(
            "Talos-XII tensor operation failed: {}",
            panic_payload_to_string(payload)
        ))
    })
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_script_path(prefix: &str) -> PathBuf {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}_{}.py", prefix, std::process::id(), now))
    }

    #[test]
    fn parses_dtype_aliases() {
        assert_eq!(parse_dtype("f32").unwrap(), Dtype::F32);
        assert_eq!(parse_dtype("float64").unwrap(), Dtype::F64);
        assert_eq!(parse_dtype("bfloat16").unwrap(), Dtype::BF16);
        assert!(parse_dtype("complex64").is_err());
    }

    #[test]
    fn run_script_smoke_executes_basic_tensor_ops() {
        let script_path = temp_script_path("talos_xii_python_bridge_smoke");
        let script = r#"
import math
import talos_xii as tx

def assert_close_list(got, expected, tol=1e-5):
    assert len(got) == len(expected), (got, expected)
    for g, e in zip(got, expected):
        assert abs(g - e) <= tol, (got, expected)

x = tx.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
assert x.shape == [2, 2]
assert x.dtype == "f32"
assert x.to_list() == [1.0, 2.0, 3.0, 4.0]

expected_constructor_api = [
    "full", "zeros", "ones", "rand", "randn", "arange", "eye", "concat",
]
for function_name in expected_constructor_api:
    assert hasattr(tx, function_name), function_name
    assert hasattr(tx.Tensor, function_name), function_name

expected_method_function_api = [
    "numel", "shape", "dtype", "device", "to_list", "tolist", "data", "grad",
    "zero_grad", "backward", "clear_graph", "detach", "copy", "fill_", "item",
    "add", "sub", "mul", "div", "neg", "relu", "gelu",
    "softmax", "softmax_v2", "log_softmax", "log_softmax_v2", "sum", "mean",
    "max", "reduce_sum", "reduce_mean", "reduce_max", "reduce_all",
    "reduce_any", "log", "exp", "expm1", "abs", "pow", "sin", "cos",
    "acos", "asin", "asinh", "atan", "cosh", "sinh", "erf", "erfc",
    "sqrt", "rsqrt", "tanh", "sigmoid", "log1p", "inv", "reciprocal",
    "square", "ceil", "floor", "round", "rint", "sign", "elu", "selu",
    "relu6", "prelu", "softplus", "softsign", "clamp", "clip", "reshape",
    "flatten", "broadcast", "transpose2d", "transpose", "split",
    "strided_slice", "maximum", "modulo", "equal", "ones_like",
    "bias_add", "matmul", "gemm", "l2_loss", "mse_loss", "weighted_mse_loss",
    "smooth_l1_loss", "sigmoid_cross_entropy_with_logits",
    "softmax_cross_entropy_with_logits", "cosine_embedding_loss", "conv2d",
    "conv2d_compress", "conv2d_transpose", "deconvolution",
    "depthwise_conv2d", "conv3d", "avg_pool2d", "avg_pool", "max_pool2d",
    "pooling", "l2_normalize", "group_norm", "instance_norm", "batch_norm2d",
]
for function_name in expected_method_function_api:
    assert hasattr(tx, function_name), function_name
    assert hasattr(x, function_name), function_name
assert hasattr(tx, "tensor")

y = tx.ones([2, 2])
z = x + y
assert z.shape == [2, 2]
assert z.to_list() == [2.0, 3.0, 4.0, 5.0]
assert z.tolist() == z.to_list()
assert abs(z.sum().item() - 14.0) < 1e-9

w = tx.tensor([2.0, 0.0, 1.0, 2.0], [2, 2])
m = x.matmul(w)
assert m.shape == [2, 2]
assert m.to_list() == [4.0, 4.0, 10.0, 8.0]
assert_close_list(tx.matmul(x, w).to_list(), m.to_list())
assert_close_list(tx.gemm(x, w).to_list(), m.to_list())

assert tx.full([2, 3], 7.0).to_list() == [7.0] * 6
assert tx.Tensor.full([2], -3.0).to_list() == [-3.0, -3.0]
assert tx.arange(4).to_list() == [0.0, 1.0, 2.0, 3.0]
assert tx.arange(1.0, 4.0, 1.5).to_list() == [1.0, 2.5]
assert len(tx.arange(1e16, 1e16 + 4.0, 1.0).to_list()) == 4
assert tx.Tensor.arange(3).to_list() == [0.0, 1.0, 2.0]
assert tx.eye(2, 3).to_list() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
assert tx.Tensor.eye(2).to_list() == [1.0, 0.0, 0.0, 1.0]

u = tx.tensor([-1.0, 0.0, 2.0, 4.0], [4])
assert tx.numel(u) == 4
assert tx.shape(u) == [4]
assert tx.dtype(u) == "f32"
assert tx.device(u) == "cpu"
assert tx.to_list(u) == u.to_list()
assert tx.tolist(u) == u.to_list()
assert tx.data(u) == u.to_list()
assert tx.grad(u) == [0.0, 0.0, 0.0, 0.0]
assert tx.copy(u).to_list() == u.to_list()
assert tx.detach(u).to_list() == u.to_list()
tmp = tx.zeros([2])
tx.fill_(tmp, 3.0)
assert tmp.to_list() == [3.0, 3.0]
tx.clear_graph(tmp)
assert tx.item(tx.tensor([7.0], [1])) == 7.0
assert_close_list(tx.relu(u).to_list(), u.relu().to_list())
assert_close_list(tx.gelu(u).to_list(), u.gelu().to_list())
assert_close_list(tx.log(tx.tensor([1.0, 2.0], [2])).to_list(), tx.tensor([1.0, 2.0], [2]).log().to_list())
assert_close_list(tx.exp(u).to_list(), u.exp().to_list())
assert_close_list(tx.expm1(u).to_list(), u.expm1().to_list())
assert_close_list(u.abs().to_list(), [1.0, 0.0, 2.0, 4.0])
assert_close_list(abs(u).to_list(), [1.0, 0.0, 2.0, 4.0])
assert_close_list(tx.abs(u).to_list(), u.abs().to_list())
assert_close_list(u.pow(2.0).to_list(), [1.0, 0.0, 4.0, 16.0])
assert_close_list((u ** 2.0).to_list(), [1.0, 0.0, 4.0, 16.0])
assert_close_list(tx.pow(u, 2.0).to_list(), u.pow(2.0).to_list())
assert_close_list(u.sin().to_list(), [math.sin(v) for v in [-1.0, 0.0, 2.0, 4.0]])
assert_close_list(u.cos().to_list(), [math.cos(v) for v in [-1.0, 0.0, 2.0, 4.0]])
assert_close_list(tx.sin(u).to_list(), u.sin().to_list())
assert_close_list(tx.cos(u).to_list(), u.cos().to_list())
assert_close_list(tx.acos(tx.tensor([0.0, 0.5], [2])).to_list(), tx.tensor([0.0, 0.5], [2]).acos().to_list())
assert_close_list(tx.asin(tx.tensor([0.0, 0.5], [2])).to_list(), tx.tensor([0.0, 0.5], [2]).asin().to_list())
assert_close_list(tx.asinh(u).to_list(), u.asinh().to_list())
assert_close_list(tx.atan(u).to_list(), u.atan().to_list())
assert_close_list(tx.cosh(u).to_list(), u.cosh().to_list())
assert_close_list(tx.sinh(u).to_list(), u.sinh().to_list())
assert_close_list(tx.erf(tx.tensor([0.0, 1.0], [2])).to_list(), tx.tensor([0.0, 1.0], [2]).erf().to_list(), tol=1e-5)
assert_close_list(tx.erfc(tx.tensor([0.0, 1.0], [2])).to_list(), tx.tensor([0.0, 1.0], [2]).erfc().to_list(), tol=1e-5)
assert_close_list(tx.sqrt(tx.tensor([1.0, 4.0], [2])).to_list(), [1.0, 2.0])
assert_close_list(tx.rsqrt(tx.tensor([1.0, 4.0], [2])).to_list(), [1.0, 0.5])
assert_close_list(u.tanh().to_list(), [math.tanh(v) for v in [-1.0, 0.0, 2.0, 4.0]])
assert_close_list(u.sigmoid().to_list(), [1.0 / (1.0 + math.exp(-v)) for v in [-1.0, 0.0, 2.0, 4.0]])
assert_close_list(tx.tanh(u).to_list(), u.tanh().to_list())
assert_close_list(tx.sigmoid(u).to_list(), u.sigmoid().to_list())
assert_close_list(tx.log1p(tx.tensor([0.0, 2.0, 4.0], [3])).to_list(), tx.tensor([0.0, 2.0, 4.0], [3]).log1p().to_list())
assert_close_list(tx.inv(tx.tensor([1.0, 2.0], [2])).to_list(), [1.0, 0.5])
assert_close_list(tx.reciprocal(tx.tensor([1.0, 2.0], [2])).to_list(), [1.0, 0.5])
assert_close_list(tx.square(u).to_list(), u.square().to_list())
assert_close_list(tx.ceil(u).to_list(), u.ceil().to_list())
assert_close_list(tx.floor(u).to_list(), u.floor().to_list())
assert_close_list(tx.round(u).to_list(), u.round().to_list())
assert_close_list(tx.rint(u).to_list(), u.rint().to_list())
assert_close_list(tx.sign(u).to_list(), u.sign().to_list())
assert_close_list(u.clamp(-0.5, 1.0).to_list(), [-0.5, 0.0, 1.0, 1.0])
assert_close_list(u.clip(-0.25, 2.5).to_list(), [-0.25, 0.0, 2.0, 2.5])
assert_close_list(tx.clamp(u, -0.5, 1.0).to_list(), u.clamp(-0.5, 1.0).to_list())
assert_close_list(tx.clip(u, -0.25, 2.5).to_list(), u.clip(-0.25, 2.5).to_list())
assert_close_list(tx.tensor([1.0, 4.0, 9.0, 16.0], [4]).sqrt().to_list(), [1.0, 2.0, 3.0, 4.0])
assert_close_list(u.relu6().to_list(), [0.0, 0.0, 2.0, 4.0])
assert_close_list(tx.relu6(u).to_list(), u.relu6().to_list())
assert_close_list(tx.selu(u).to_list(), u.selu().to_list())
assert_close_list(tx.softplus(u).to_list(), u.softplus().to_list())
assert_close_list(tx.softsign(u).to_list(), u.softsign().to_list())
assert_close_list(u.softplus().to_list(), [math.log1p(math.exp(v)) for v in [-1.0, 0.0, 2.0, 4.0]])
assert_close_list(tx.tensor([0.0, 2.0, 4.0], [3]).log1p().to_list(), [math.log1p(v) for v in [0.0, 2.0, 4.0]])

v = tx.tensor([1.0, 2.0, 4.0], [3])
assert_close_list((v + 1.0).to_list(), [2.0, 3.0, 5.0])
assert_close_list(v.add(1.0).to_list(), [2.0, 3.0, 5.0])
assert_close_list(tx.add(v, 1.0).to_list(), [2.0, 3.0, 5.0])
assert_close_list(tx.add(v, tx.tensor([1.0, 1.0, 1.0], [3])).to_list(), [2.0, 3.0, 5.0])
assert_close_list((1.0 + v).to_list(), [2.0, 3.0, 5.0])
assert_close_list((v - 1.0).to_list(), [0.0, 1.0, 3.0])
assert_close_list(v.sub(1.0).to_list(), [0.0, 1.0, 3.0])
assert_close_list(tx.sub(v, 1.0).to_list(), [0.0, 1.0, 3.0])
assert_close_list((10.0 - v).to_list(), [9.0, 8.0, 6.0])
assert_close_list((v * 2.0).to_list(), [2.0, 4.0, 8.0])
assert_close_list(v.mul(2.0).to_list(), [2.0, 4.0, 8.0])
assert_close_list(tx.mul(v, 2.0).to_list(), [2.0, 4.0, 8.0])
assert_close_list((2.0 * v).to_list(), [2.0, 4.0, 8.0])
assert_close_list((v / 2.0).to_list(), [0.5, 1.0, 2.0])
assert_close_list(v.div(2.0).to_list(), [0.5, 1.0, 2.0])
assert_close_list(tx.div(v, 2.0).to_list(), [0.5, 1.0, 2.0])
assert_close_list((8.0 / v).to_list(), [8.0, 4.0, 2.0])
assert_close_list(v.neg().to_list(), [-1.0, -2.0, -4.0])
assert_close_list(tx.neg(v).to_list(), [-1.0, -2.0, -4.0])
assert_close_list((v % 2.0).to_list(), [1.0, 0.0, 0.0])
assert_close_list(v.maximum(tx.tensor([3.0, 1.0, 5.0], [3])).to_list(), [3.0, 2.0, 5.0])
assert_close_list(v.equal(tx.tensor([1.0, 0.0, 4.0], [3])).to_list(), [1.0, 0.0, 1.0])

mat = tx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
assert_close_list(mat.sum(1).to_list(), [6.0, 15.0])
assert_close_list(mat.mean(0).to_list(), [2.5, 3.5, 4.5])
assert mat.max(1, keepdim=True).shape == [2, 1]
assert_close_list(tx.sum(mat, 1).to_list(), mat.sum(1).to_list())
assert_close_list(tx.mean(mat, 0).to_list(), mat.mean(0).to_list())
assert tx.max(mat, 1, keepdim=True).shape == mat.max(1, keepdim=True).shape
assert_close_list(tx.reduce_sum(mat, 1).to_list(), mat.reduce_sum(1).to_list())
assert_close_list(tx.reduce_mean(mat, 0).to_list(), mat.reduce_mean(0).to_list())
assert tx.reduce_max(mat, 1, keepdim=True).shape == [2, 1]
truth = tx.tensor([1.0, 0.0, 1.0, 1.0], [2, 2])
assert_close_list(tx.reduce_all(truth, 1).to_list(), [0.0, 1.0])
assert_close_list(tx.reduce_any(truth, 1).to_list(), [1.0, 1.0])
assert_close_list(mat.softmax(0).sum(0).to_list(), [1.0, 1.0, 1.0])
assert_close_list(mat.softmax_v2(1).sum(1).to_list(), [1.0, 1.0])
assert_close_list(tx.softmax_v2(mat, 0).sum(0).to_list(), [1.0, 1.0, 1.0])
assert_close_list(tx.softmax(mat, 1).sum(1).to_list(), [1.0, 1.0])
assert_close_list(tx.log_softmax(mat, 1).exp().sum(1).to_list(), [1.0, 1.0])
assert_close_list(tx.log_softmax_v2(mat, 1).exp().sum(1).to_list(), [1.0, 1.0])
assert_close_list(mat.l2_normalize(1).square().sum(1).to_list(), [1.0, 1.0])
assert_close_list(tx.l2_normalize(mat, 1).square().sum(1).to_list(), [1.0, 1.0])
assert tx.reshape(mat, [3, 2]).shape == [3, 2]
assert tx.flatten(mat).shape == [6]
assert_close_list(tx.broadcast(tx.tensor([1.0, 2.0, 3.0], [3, 1]), [3, 2]).to_list(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
assert_close_list(tx.transpose2d(mat).to_list(), mat.transpose2d().to_list())
assert_close_list(tx.transpose(mat, 0, 1).to_list(), mat.transpose(0, 1).to_list())
cat = tx.concat([tx.tensor([1.0, 2.0], [1, 2]), tx.tensor([3.0, 4.0], [1, 2])], 0)
assert cat.shape == [2, 2]
assert_close_list(cat.to_list(), [1.0, 2.0, 3.0, 4.0])
cat_tuple = tx.concat((tx.tensor([1.0], [1]), tx.tensor([2.0], [1])), 0)
assert_close_list(cat_tuple.to_list(), [1.0, 2.0])
parts = cat.split(0, [1, 1])
assert_close_list(parts[0].to_list(), [1.0, 2.0])
parts_fn = tx.split(cat, 0, [1, 1])
assert_close_list(parts_fn[1].to_list(), [3.0, 4.0])
assert_close_list(cat.strided_slice([0, 0], [2, 2], [1, 2]).to_list(), [1.0, 3.0])
assert_close_list(tx.strided_slice(cat, [0, 0], [2, 2], [1, 2]).to_list(), [1.0, 3.0])
assert_close_list(tx.ones_like(v).to_list(), [1.0, 1.0, 1.0])
assert_close_list(tx.maximum(v, tx.tensor([3.0, 1.0, 5.0], [3])).to_list(), [3.0, 2.0, 5.0])
assert_close_list(tx.equal(v, tx.tensor([1.0, 0.0, 4.0], [3])).to_list(), [1.0, 0.0, 1.0])
assert_close_list(tx.modulo(v, tx.tensor([2.0, 2.0, 2.0], [3])).to_list(), [1.0, 0.0, 0.0])
assert_close_list(tx.bias_add(tx.tensor([1.0, 2.0, 3.0, 4.0], [2, 2]), tx.tensor([10.0, 20.0], [2])).to_list(), [11.0, 22.0, 13.0, 24.0])

pool = tx.tensor([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]).avg_pool2d(2, 1)
assert_close_list(pool.to_list(), [2.5])
assert_close_list(tx.tensor([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]).pooling("avg", 2, 1).to_list(), [2.5])
pool_input = tx.tensor([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2], "f64")
assert_close_list(tx.avg_pool2d(pool_input, 2, 1).to_list(), [2.5])
assert_close_list(tx.avg_pool(pool_input, 2, 1).to_list(), [2.5])
assert_close_list(tx.max_pool2d(pool_input, 2, 1).to_list(), [4.0])
assert_close_list(tx.pooling(pool_input, "avg", 2, 1).to_list(), [2.5])
conv1_weight = tx.tensor([1.0], [1, 1, 1, 1], "f64")
conv2_weight = tx.tensor([2.0], [1, 1, 1, 1], "f64")
assert_close_list(tx.conv2d(pool_input, conv1_weight).to_list(), pool_input.to_list())
assert_close_list(tx.conv2d_compress(pool_input, conv1_weight).to_list(), pool_input.to_list())
assert_close_list(tx.conv2d_transpose(tx.tensor([1.0], [1, 1, 1, 1], "f64"), conv2_weight).to_list(), [2.0])
assert_close_list(tx.deconvolution(tx.tensor([1.0], [1, 1, 1, 1], "f64"), conv2_weight).to_list(), [2.0])
dw = tx.tensor([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]).depthwise_conv2d(tx.tensor([2.0], [1, 1, 1, 1]))
assert_close_list(dw.to_list(), [2.0, 4.0, 6.0, 8.0])
assert_close_list(tx.depthwise_conv2d(pool_input, conv2_weight).to_list(), [2.0, 4.0, 6.0, 8.0])
assert_close_list(tx.conv3d(tx.tensor([1.0], [1, 1, 1, 1, 1], "f64"), tx.tensor([2.0], [1, 1, 1, 1, 1], "f64")).to_list(), [2.0])
assert_close_list(tx.prelu(u, tx.tensor([0.25], [1])).to_list(), u.prelu(tx.tensor([0.25], [1])).to_list())
assert_close_list(tx.group_norm(tx.tensor([1.0, 2.0], [1, 2]), 1).to_list(), tx.tensor([1.0, 2.0], [1, 2]).group_norm(1).to_list())
assert_close_list(tx.instance_norm(tx.tensor([1.0, 2.0], [1, 2])).to_list(), tx.tensor([1.0, 2.0], [1, 2]).instance_norm().to_list())
assert_close_list(tx.batch_norm2d(tx.tensor([1.0, 2.0], [1, 2]), tx.tensor([1.0, 1.0], [2]), tx.tensor([0.0, 0.0], [2])).to_list(), tx.tensor([1.0, 2.0], [1, 2]).batch_norm2d(tx.tensor([1.0, 1.0], [2]), tx.tensor([0.0, 0.0], [2])).to_list())
ce = tx.tensor([2.0, 0.0], [1, 2]).softmax_cross_entropy_with_logits(tx.tensor([1.0, 0.0], [1, 2]))
assert ce.item() >= 0.0
assert tx.softmax_cross_entropy_with_logits(tx.tensor([2.0, 0.0], [1, 2]), tx.tensor([1.0, 0.0], [1, 2])).item() >= 0.0
assert tx.mse_loss(tx.tensor([1.0], [1]), tx.tensor([0.0], [1])).item() == 1.0
assert tx.weighted_mse_loss(tx.tensor([1.0], [1]), tx.tensor([0.0], [1]), tx.tensor([1.0], [1])).item() == 1.0
assert tx.l2_loss(tx.tensor([2.0], [1])).item() == 2.0
assert tx.smooth_l1_loss(tx.tensor([1.0], [1]), tx.tensor([0.0], [1])).item() == 0.5
assert tx.sigmoid_cross_entropy_with_logits(tx.tensor([0.0], [1]), tx.tensor([1.0], [1])).item() > 0.0
assert tx.cosine_embedding_loss(tx.tensor([1.0, 0.0], [1, 2]), tx.tensor([1.0, 0.0], [1, 2]), tx.tensor([1.0], [1])).item() == 0.0

def assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError("expected exception")

assert_raises(ValueError, lambda: mat.softmax(99))
assert_raises(ValueError, lambda: tx.tensor([1.0], []).softmax())
assert_raises(ValueError, lambda: tx.tensor([1.0], []).l2_normalize())
assert_raises(ValueError, lambda: tx.tensor([1.0], [1]).reshape([2]))
assert_raises(ValueError, lambda: mat.avg_pool2d(0, 1))
assert_raises(ValueError, lambda: mat.group_norm(0))
assert_raises(MemoryError, lambda: tx.zeros([67108865]))
assert_raises(MemoryError, lambda: tx.tensor([1.0], [67108865]))
assert_close_list(tx.tensor([7.0], [1]).to_list(), [7.0])

loss = (v * 2.0 + 1.0).sum()
loss.backward()
assert_close_list(v.grad(), [2.0, 2.0, 2.0])
tx.zero_grad(v)
assert_close_list(tx.grad(v), [0.0, 0.0, 0.0])
v2 = tx.tensor([1.0, 2.0], [2])
loss2 = tx.sum(v2 * 3.0)
tx.backward(loss2)
assert_close_list(tx.grad(v2), [3.0, 3.0])
"#;

        std::fs::write(&script_path, script).unwrap();
        let result = run_script(&script_path, None, &[]);
        let _ = std::fs::remove_file(&script_path);

        assert_eq!(result.unwrap(), 0);
    }
}
