use crate::autograd::{Device, Tensor as AutoTensor};
use crate::dtype::Dtype;
use pyo3::exceptions::{PyRuntimeError, PySystemExit, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyModule};
use std::any::Any;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Once;

static REGISTER_PY_MODULE: Once = Once::new();

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
        configure_python_process(py, &script_path, &script_dir, args)?;

        let code = CString::new(source)
            .map_err(|_| PyValueError::new_err("script source contains an embedded NUL byte"))?;
        let filename = CString::new(script_path.to_string_lossy().as_bytes())
            .map_err(|_| PyValueError::new_err("script path contains an embedded NUL byte"))?;

        match PyModule::from_code(py, code.as_c_str(), filename.as_c_str(), c"__main__") {
            Ok(_) => Ok(0),
            Err(err) => {
                if let Some(code) = system_exit_code(py, &err) {
                    Ok(code)
                } else {
                    Err(err)
                }
            }
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
    fn new(data: Vec<f64>, shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
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

    fn to_list(&self) -> Vec<f64> {
        self.inner.data_as_f64_vec()
    }

    fn tolist(&self) -> Vec<f64> {
        self.to_list()
    }

    fn data(&self) -> Vec<f64> {
        self.to_list()
    }

    fn grad(&self) -> Vec<f64> {
        self.inner.grad_to_f64_vec()
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

    fn relu(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.relu())
    }

    fn gelu(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.gelu())
    }

    fn softmax(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.softmax())
    }

    #[pyo3(signature = (dim = None))]
    fn log_softmax(&self, dim: Option<usize>) -> PyResult<Self> {
        wrap_tensor_op(|| match dim {
            Some(dim) => self.inner.log_softmax_dim(dim),
            None => self.inner.log_softmax(),
        })
    }

    fn sum(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.sum())
    }

    fn mean(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.mean())
    }

    fn log(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.log())
    }

    fn exp(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.exp())
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

    fn sqrt(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.sqrt())
    }

    fn tanh(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.tanh())
    }

    fn sigmoid(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.sigmoid())
    }

    fn clamp(&self, min: f64, max: f64) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.clamp(min, max))
    }

    fn clip(&self, min: f64, max: f64) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.clip(min, max))
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.reshape(shape))
    }

    fn broadcast(&self, shape: Vec<usize>) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.broadcast(shape))
    }

    fn transpose2d(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.transpose2d())
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.transpose(dim0, dim1))
    }

    fn mse_loss(&self, target: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.mse_loss(&target.inner))
    }

    fn weighted_mse_loss(&self, target: &PyTensor, weights: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.weighted_mse_loss(&target.inner, &weights.inner))
    }

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => Ok(&self.inner + &rhs.inner),
            TensorOperand::Scalar(rhs) => Ok(&self.inner + &self.scalar_tensor_like(rhs)),
        })
    }

    fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => Ok(&lhs.inner + &self.inner),
            TensorOperand::Scalar(lhs) => Ok(&self.scalar_tensor_like(lhs) + &self.inner),
        })
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => Ok(&self.inner - &rhs.inner),
            TensorOperand::Scalar(rhs) => Ok(&self.inner - &self.scalar_tensor_like(rhs)),
        })
    }

    fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => Ok(&lhs.inner - &self.inner),
            TensorOperand::Scalar(lhs) => Ok(&self.scalar_tensor_like(lhs) - &self.inner),
        })
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => Ok(&self.inner * &rhs.inner),
            TensorOperand::Scalar(rhs) => Ok(&self.inner * &self.scalar_tensor_like(rhs)),
        })
    }

    fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => Ok(&lhs.inner * &self.inner),
            TensorOperand::Scalar(lhs) => Ok(&self.scalar_tensor_like(lhs) * &self.inner),
        })
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => Ok(&self.inner / &rhs.inner),
            TensorOperand::Scalar(rhs) => Ok(&self.inner / &self.scalar_tensor_like(rhs)),
        })
    }

    fn __rtruediv__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        wrap_tensor_result_op(|| match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => Ok(&lhs.inner / &self.inner),
            TensorOperand::Scalar(lhs) => Ok(&self.scalar_tensor_like(lhs) / &self.inner),
        })
    }

    fn __neg__(&self) -> PyResult<Self> {
        wrap_tensor_op(|| -&self.inner)
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
fn tensor(data: Vec<f64>, shape: Vec<usize>, dtype: &str) -> PyResult<PyTensor> {
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
    let mut data = Vec::new();
    let mut value = start;
    if step > 0.0 {
        while value < stop {
            data.push(value);
            value += step;
        }
    } else {
        while value > stop {
            data.push(value);
            value += step;
        }
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

#[pymodule]
fn talos_xii(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    Ok(())
}

fn make_tensor(data: Vec<f64>, shape: Vec<usize>, dtype: &str) -> PyResult<PyTensor> {
    let dtype = parse_dtype(dtype)?;
    let expected = checked_numel(&shape)?;
    if data.len() != expected {
        return Err(PyValueError::new_err(format!(
            "data length {} does not match shape {:?} ({} elements)",
            data.len(),
            shape,
            expected
        )));
    }
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
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| PyValueError::new_err("shape element count overflow"))
    })
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

assert tx.full([2, 3], 7.0).to_list() == [7.0] * 6
assert tx.Tensor.full([2], -3.0).to_list() == [-3.0, -3.0]
assert tx.arange(4).to_list() == [0.0, 1.0, 2.0, 3.0]
assert tx.arange(1.0, 4.0, 1.5).to_list() == [1.0, 2.5]
assert tx.Tensor.arange(3).to_list() == [0.0, 1.0, 2.0]
assert tx.eye(2, 3).to_list() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
assert tx.Tensor.eye(2).to_list() == [1.0, 0.0, 0.0, 1.0]

u = tx.tensor([-1.0, 0.0, 2.0, 4.0], [4])
assert_close_list(u.abs().to_list(), [1.0, 0.0, 2.0, 4.0])
assert_close_list(abs(u).to_list(), [1.0, 0.0, 2.0, 4.0])
assert_close_list(u.pow(2.0).to_list(), [1.0, 0.0, 4.0, 16.0])
assert_close_list((u ** 2.0).to_list(), [1.0, 0.0, 4.0, 16.0])
assert_close_list(u.sin().to_list(), [math.sin(v) for v in [-1.0, 0.0, 2.0, 4.0]])
assert_close_list(u.cos().to_list(), [math.cos(v) for v in [-1.0, 0.0, 2.0, 4.0]])
assert_close_list(u.tanh().to_list(), [math.tanh(v) for v in [-1.0, 0.0, 2.0, 4.0]])
assert_close_list(u.sigmoid().to_list(), [1.0 / (1.0 + math.exp(-v)) for v in [-1.0, 0.0, 2.0, 4.0]])
assert_close_list(u.clamp(-0.5, 1.0).to_list(), [-0.5, 0.0, 1.0, 1.0])
assert_close_list(u.clip(-0.25, 2.5).to_list(), [-0.25, 0.0, 2.0, 2.5])
assert_close_list(tx.tensor([1.0, 4.0, 9.0, 16.0], [4]).sqrt().to_list(), [1.0, 2.0, 3.0, 4.0])

v = tx.tensor([1.0, 2.0, 4.0], [3])
assert_close_list((v + 1.0).to_list(), [2.0, 3.0, 5.0])
assert_close_list((1.0 + v).to_list(), [2.0, 3.0, 5.0])
assert_close_list((v - 1.0).to_list(), [0.0, 1.0, 3.0])
assert_close_list((10.0 - v).to_list(), [9.0, 8.0, 6.0])
assert_close_list((v * 2.0).to_list(), [2.0, 4.0, 8.0])
assert_close_list((2.0 * v).to_list(), [2.0, 4.0, 8.0])
assert_close_list((v / 2.0).to_list(), [0.5, 1.0, 2.0])
assert_close_list((8.0 / v).to_list(), [8.0, 4.0, 2.0])

loss = (v * 2.0 + 1.0).sum()
loss.backward()
assert_close_list(v.grad(), [2.0, 2.0, 2.0])
"#;

        std::fs::write(&script_path, script).unwrap();
        let result = run_script(&script_path, None, &[]);
        let _ = std::fs::remove_file(&script_path);

        assert_eq!(result.unwrap(), 0);
    }
}
