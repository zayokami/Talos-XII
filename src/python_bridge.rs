use crate::autograd::{Context, Device, Tensor as AutoTensor};
use crate::dtype::{Dtype, Storage};
use pyo3::exceptions::{
    PyIndexError, PyMemoryError, PyNotImplementedError, PyRuntimeError, PySystemExit, PyTypeError,
    PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyModule, PySlice, PyTuple};
use std::any::Any;
use std::cell::Cell;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};

mod state;
mod types;

use state::{mark_gradients_ready, PythonAutogradState};
use types::{parse_device_arg, parse_dtype_arg, PyDType, PyDevice};

static REGISTER_PY_MODULE: Once = Once::new();
const MAX_PY_TENSOR_ELEMENTS: usize = 64 * 1024 * 1024;
const MAX_PY_EXPORT_ELEMENTS: usize = 8 * 1024 * 1024;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

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
    state: Arc<PythonAutogradState>,
}

enum TensorOperand<'py> {
    Tensor(PyRef<'py, PyTensor>),
    Scalar(f64),
}

#[pyclass(name = "_NativeGradMode", unsendable)]
struct PyGradMode {
    enabled: bool,
    previous: Vec<bool>,
}

#[pymethods]
impl PyGradMode {
    fn __enter__(mut self_: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        let previous = GRAD_ENABLED.with(|state| state.replace(self_.enabled));
        self_.previous.push(previous);
        self_
    }

    fn __exit__(
        &mut self,
        _exception_type: Option<&Bound<'_, PyAny>>,
        _exception_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        if let Some(previous) = self.previous.pop() {
            GRAD_ENABLED.with(|state| state.set(previous));
        }
        false
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, shape = None, dtype = None, *, device = None, requires_grad = false))]
    fn new(
        data: &Bound<'_, PyAny>,
        shape: Option<Vec<usize>>,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        make_tensor(data, shape, dtype, device, requires_grad)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype = None, *, device = None, requires_grad = false))]
    fn zeros(
        shape: Vec<usize>,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        crate::python_bridge::zeros(shape, dtype, device, requires_grad)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype = None, *, device = None, requires_grad = false))]
    fn ones(
        shape: Vec<usize>,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        crate::python_bridge::ones(shape, dtype, device, requires_grad)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, fill_value, dtype = None, *, device = None, requires_grad = false))]
    fn full(
        shape: Vec<usize>,
        fill_value: f64,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        crate::python_bridge::full(shape, fill_value, dtype, device, requires_grad)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, *, min = 0.0, max = 1.0, seed = 42, dtype = None, device = None, requires_grad = false))]
    fn rand(
        shape: Vec<usize>,
        min: f64,
        max: f64,
        seed: u64,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        crate::python_bridge::py_rand(shape, min, max, seed, dtype, device, requires_grad)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, *, seed = 42, dtype = None, device = None, requires_grad = false))]
    fn randn(
        shape: Vec<usize>,
        seed: u64,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        crate::python_bridge::py_randn(shape, seed, dtype, device, requires_grad)
    }

    #[staticmethod]
    #[pyo3(signature = (start, stop = None, step = 1.0, dtype = None, *, device = None, requires_grad = false))]
    fn arange(
        start: f64,
        stop: Option<f64>,
        step: f64,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        crate::python_bridge::arange(start, stop, step, dtype, device, requires_grad)
    }

    #[staticmethod]
    #[pyo3(signature = (n, m = None, dtype = None, *, device = None, requires_grad = false))]
    fn eye(
        n: usize,
        m: Option<usize>,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        crate::python_bridge::eye(n, m, dtype, device, requires_grad)
    }

    #[staticmethod]
    fn concat(tensors: &Bound<'_, PyAny>, dim: isize) -> PyResult<Self> {
        let tensors = tensor_sequence(tensors)?;
        let dim = resolve_sequence_dim(&tensors, dim, "concat")?;
        validate_concat_tensors(&tensors, dim)?;
        wrap_tensor_op(|| AutoTensor::concat(&tensors, dim))
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape.iter().copied())
    }

    #[getter]
    fn dtype(&self) -> PyDType {
        PyDType::new(self.inner.dtype)
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice::from_core(self.inner.device)
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.shape.len()
    }

    fn dim(&self) -> usize {
        self.inner.shape.len()
    }

    #[pyo3(signature = (dim = None))]
    fn size(&self, dim: Option<isize>) -> PyResult<Py<PyAny>> {
        Python::attach(|py| match dim {
            Some(dim) => {
                let dim = resolve_dim(&self.inner, dim, "size")?;
                Ok(self.inner.shape[dim].into_pyobject(py)?.into_any().unbind())
            }
            None => Ok(PyTuple::new(py, self.inner.shape.iter().copied())?
                .into_any()
                .unbind()),
        })
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.state.requires_grad()
    }

    #[setter]
    fn set_requires_grad(&self, value: bool) -> PyResult<()> {
        self.set_requires_grad_value(value)
    }

    #[pyo3(signature = (value = true))]
    fn requires_grad_(&self, value: bool) -> PyResult<Self> {
        self.set_requires_grad_value(value)?;
        Ok(self.clone())
    }

    #[getter]
    fn is_leaf(&self) -> bool {
        self.state.is_leaf()
    }

    #[getter]
    fn _is_parameter(&self) -> bool {
        self.state.is_parameter()
    }

    fn _set_parameter(&self, value: bool) {
        self.state.set_parameter(value);
    }

    #[getter]
    fn grad_fn(&self) -> Option<&'static str> {
        if self.state.requires_grad() && !self.state.is_leaf() {
            Some("TalosBackward")
        } else {
            None
        }
    }

    #[getter]
    fn _version(&self) -> u64 {
        self.state.version()
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn to_list(&self) -> PyResult<Vec<f64>> {
        validate_export_len(self.inner.numel(), "to_list")?;
        Ok(self.inner.data_as_f64_vec())
    }

    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        validate_export_len(self.inner.numel(), "tolist")?;
        nested_list(
            py,
            &self.inner.data_as_f64_vec(),
            &self.inner.shape,
            self.inner.dtype,
        )
    }

    #[getter]
    fn data(&self) -> Self {
        Self::from_alias(self.inner.detach(), self, false, true)
    }

    #[getter]
    fn grad(&self) -> PyResult<Option<Self>> {
        if !self.state.grad_ready() {
            return Ok(None);
        }
        validate_export_len(self.inner.numel(), "grad")?;
        let grad_dtype = self.inner.grad.dtype();
        let grad = AutoTensor::with_dtype(
            self.inner.grad_to_f64_vec(),
            self.inner.shape.clone(),
            grad_dtype,
        );
        #[cfg(cuda)]
        let grad = if self.inner.device == Device::Cuda {
            grad.to_cuda().map_err(|error| {
                PyRuntimeError::new_err(format!("CUDA gradient export failed: {error}"))
            })?
        } else {
            grad
        };
        Ok(Some(Self::from_leaf(grad, false)))
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
        self.state.set_grad_ready(true);
    }

    #[pyo3(signature = (gradient = None, retain_graph = None, create_graph = false))]
    fn backward(
        &mut self,
        gradient: Option<&PyTensor>,
        retain_graph: Option<bool>,
        create_graph: bool,
    ) -> PyResult<()> {
        if create_graph {
            return Err(PyNotImplementedError::new_err(
                "create_graph=True is not implemented",
            ));
        }
        if !self.state.requires_grad() {
            return Err(PyRuntimeError::new_err(
                "element 0 of tensors does not require grad and does not have a grad_fn",
            ));
        }
        if gradient.is_none() && self.inner.numel() != 1 {
            return Err(PyRuntimeError::new_err(
                "grad can be implicitly created only for scalar outputs",
            ));
        }
        if self.state.graph_consumed() {
            return Err(PyRuntimeError::new_err(
                "Trying to backward through the graph a second time; specify retain_graph=True on the first call",
            ));
        }
        self.state
            .check_versions()
            .map_err(PyRuntimeError::new_err)?;

        wrap_value_op(|| {
            self.inner
                .backward_with_gradient(gradient.map(|gradient| &gradient.inner))
        })?
        .map_err(PyRuntimeError::new_err)?;
        mark_gradients_ready(&self.inner);

        if !retain_graph.unwrap_or(false) && !self.state.is_leaf() {
            self.inner.clear_graph();
            self.state.set_graph_consumed();
        }
        Ok(())
    }

    fn clear_graph(&mut self) {
        self.inner.clear_graph();
        if !self.state.is_leaf() {
            self.state.set_graph_consumed();
        }
    }

    fn detach(&self) -> Self {
        Self::from_alias(self.inner.detach(), self, false, true)
    }

    fn clone(&self) -> Self {
        self.clone_tensor()
    }

    fn retain_grad(&self) -> PyResult<()> {
        if !self.state.requires_grad() {
            return Err(PyRuntimeError::new_err(
                "can't retain_grad on Tensor that has requires_grad=False",
            ));
        }
        self.state.set_retain_grad(true);
        Ok(())
    }

    fn copy(&self) -> Self {
        self.clone_tensor()
    }

    fn copy_<'py>(
        mut self_: PyRefMut<'py, Self>,
        source: &PyTensor,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if GRAD_ENABLED.with(Cell::get) && self_.state.requires_grad() {
            let message = if self_.state.is_leaf() {
                "a leaf Tensor that requires grad is being used in an in-place operation"
            } else {
                "in-place mutation of a Tensor tracked by autograd is not implemented; use no_grad() for parameter updates"
            };
            return Err(PyRuntimeError::new_err(message));
        }
        self_
            .inner
            .copy_data_from(&source.inner)
            .map_err(PyRuntimeError::new_err)?;
        self_.state.increment_version();
        Ok(self_)
    }

    fn _replace_data_(&mut self, source: &PyTensor) -> PyResult<()> {
        if self.inner.shape != source.inner.shape {
            return Err(PyRuntimeError::new_err(format!(
                "source shape {:?} does not match destination shape {:?}",
                source.inner.shape, self.inner.shape
            )));
        }
        if GRAD_ENABLED.with(Cell::get) && self.state.requires_grad() {
            return Err(PyRuntimeError::new_err(
                "a leaf Tensor that requires grad is being used in an in-place operation",
            ));
        }
        if source.inner.dtype == Dtype::I8 && self.state.requires_grad() {
            return Err(PyRuntimeError::new_err(
                "only floating point tensors can require gradients",
            ));
        }

        let existing_grad = self
            .state
            .grad_ready()
            .then(|| self.inner.grad_to_f64_vec());
        let mut replacement = source.inner.clone();
        replacement.clear_graph();
        if let Some(existing_grad) = existing_grad {
            replacement
                .grad
                .copy_from_f64_slice(&existing_grad)
                .map_err(PyRuntimeError::new_err)?;
        }
        self.inner = replacement;
        self.state.increment_version();
        self.state.rebind(&self.inner);
        Ok(())
    }

    fn fill_(&mut self, value: f64) -> PyResult<()> {
        if GRAD_ENABLED.with(Cell::get) && self.state.requires_grad() {
            let message = if self.state.is_leaf() {
                "a leaf Tensor that requires grad is being used in an in-place operation"
            } else {
                "in-place mutation of a Tensor tracked by autograd is not implemented; use no_grad() for parameter updates"
            };
            return Err(PyRuntimeError::new_err(message));
        }
        wrap_unit_op(|| {
            self.inner.fill_(value);
        })?;
        self.state.increment_version();
        Ok(())
    }

    fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = wrap_value_op(|| self.inner.item())?;
        match self.inner.dtype {
            Dtype::I8 => Ok((value as i8).into_pyobject(py)?.into_any().unbind()),
            _ => Ok(value.into_pyobject(py)?.into_any().unbind()),
        }
    }

    #[pyo3(signature = (device = None, dtype = None, non_blocking = false, copy = false))]
    fn to(
        &self,
        device: Option<&Bound<'_, PyAny>>,
        dtype: Option<&Bound<'_, PyAny>>,
        non_blocking: bool,
        copy: bool,
    ) -> PyResult<Self> {
        if non_blocking {
            return Err(PyNotImplementedError::new_err(
                "non_blocking=True is not implemented",
            ));
        }

        let mut requested_device = None;
        let mut requested_dtype = dtype
            .map(|value| parse_dtype_arg(Some(value)))
            .transpose()?;
        if let Some(value) = device {
            if let Ok(other) = value.extract::<PyRef<'_, PyTensor>>() {
                if requested_dtype.is_some() {
                    return Err(PyTypeError::new_err(
                        "to(other) cannot be combined with an explicit dtype",
                    ));
                }
                requested_dtype = Some(other.inner.dtype);
                requested_device = Some(PyDevice::from_core(other.inner.device));
            } else if let Ok(dtype) = value.extract::<PyRef<'_, PyDType>>() {
                if requested_dtype.is_some() {
                    return Err(PyTypeError::new_err("dtype specified twice"));
                }
                requested_dtype = Some(dtype.inner());
            } else if !value.is_none() {
                requested_device = Some(parse_device_arg(Some(value))?);
            }
        }

        self.converted_tensor(
            requested_dtype.unwrap_or(self.inner.dtype),
            requested_device.unwrap_or_else(|| PyDevice::from_core(self.inner.device)),
            copy,
        )
    }

    fn cpu(&self) -> PyResult<Self> {
        self.converted_tensor(self.inner.dtype, PyDevice::cpu(), false)
    }

    #[pyo3(signature = (device = None))]
    fn cuda(&self, device: Option<usize>) -> PyResult<Self> {
        self.converted_tensor(self.inner.dtype, PyDevice::cuda(device.unwrap_or(0)), false)
    }

    fn float(&self) -> PyResult<Self> {
        self.converted_tensor(Dtype::F32, PyDevice::from_core(self.inner.device), false)
    }

    fn double(&self) -> PyResult<Self> {
        self.converted_tensor(Dtype::F64, PyDevice::from_core(self.inner.device), false)
    }

    fn bfloat16(&self) -> PyResult<Self> {
        self.converted_tensor(Dtype::BF16, PyDevice::from_core(self.inner.device), false)
    }

    fn matmul(&self, rhs: &PyTensor) -> PyResult<Self> {
        validate_matmul(&self.inner, &rhs.inner)?;
        wrap_tensor_op(|| self.inner.matmul(&rhs.inner))
    }

    fn gemm(&self, rhs: &PyTensor) -> PyResult<Self> {
        self.matmul(rhs)
    }

    fn add(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => broadcast_binary(self, &rhs, |lhs, rhs| lhs + rhs),
            TensorOperand::Scalar(rhs) => {
                let scalar = self.scalar_tensor_like(rhs)?;
                wrap_tensor_op(|| &self.inner + &scalar)
            }
        }
    }

    fn sub(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => broadcast_binary(self, &rhs, |lhs, rhs| lhs - rhs),
            TensorOperand::Scalar(rhs) => {
                let scalar = self.scalar_tensor_like(rhs)?;
                wrap_tensor_op(|| &self.inner - &scalar)
            }
        }
    }

    fn mul(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => broadcast_binary(self, &rhs, |lhs, rhs| lhs * rhs),
            TensorOperand::Scalar(rhs) => {
                let scalar = self.scalar_tensor_like(rhs)?;
                wrap_tensor_op(|| &self.inner * &scalar)
            }
        }
    }

    fn div(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => broadcast_binary(self, &rhs, |lhs, rhs| lhs / rhs),
            TensorOperand::Scalar(rhs) => {
                let scalar = self.scalar_tensor_like(rhs)?;
                wrap_tensor_op(|| &self.inner / &scalar)
            }
        }
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
    fn softmax(&self, dim: Option<isize>) -> PyResult<Self> {
        let resolved_dim = resolve_dim_option(&self.inner, dim, "softmax")?;
        wrap_tensor_op(|| self.inner.softmax_dim(resolved_dim))
    }

    #[pyo3(signature = (dim = None))]
    fn softmax_v2(&self, dim: Option<isize>) -> PyResult<Self> {
        self.softmax(dim)
    }

    #[pyo3(signature = (dim = None))]
    fn log_softmax(&self, dim: Option<isize>) -> PyResult<Self> {
        let resolved_dim = resolve_dim_option(&self.inner, dim, "log_softmax")?;
        wrap_tensor_op(|| self.inner.log_softmax_dim(resolved_dim))
    }

    #[pyo3(signature = (dim = None))]
    fn log_softmax_v2(&self, dim: Option<isize>) -> PyResult<Self> {
        self.log_softmax(dim)
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn sum(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        let dim = resolve_optional_dim(&self.inner, dim, "sum")?;
        match dim {
            Some(dim) => wrap_tensor_op(|| self.inner.reduce_sum_dim(dim, keepdim)),
            None => {
                let reduced = PyTensor::from_tensor(wrap_value_op(|| self.inner.sum())?);
                wrap_tensor_op(|| reduced.inner.reshape(Vec::new()))
            }
        }
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn mean(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        let dim = resolve_optional_dim(&self.inner, dim, "mean")?;
        match dim {
            Some(dim) => wrap_tensor_op(|| self.inner.reduce_mean_dim(dim, keepdim)),
            None => {
                let reduced = PyTensor::from_tensor(wrap_value_op(|| self.inner.mean())?);
                wrap_tensor_op(|| reduced.inner.reshape(Vec::new()))
            }
        }
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn max(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        let dim = resolve_optional_dim(&self.inner, dim, "max")?;
        match dim {
            Some(dim) => wrap_tensor_op(|| self.inner.reduce_max_dim(dim, keepdim)),
            None => {
                let reduced = PyTensor::from_tensor(wrap_value_op(|| self.inner.max())?);
                wrap_tensor_op(|| reduced.inner.reshape(Vec::new()))
            }
        }
    }

    #[pyo3(signature = (dim, keepdim = false))]
    fn reduce_sum(&self, dim: isize, keepdim: bool) -> PyResult<Self> {
        let dim = resolve_dim(&self.inner, dim, "reduce_sum")?;
        wrap_tensor_op(|| self.inner.reduce_sum_dim(dim, keepdim))
    }

    #[pyo3(signature = (dim, keepdim = false))]
    fn reduce_mean(&self, dim: isize, keepdim: bool) -> PyResult<Self> {
        let dim = resolve_dim(&self.inner, dim, "reduce_mean")?;
        wrap_tensor_op(|| self.inner.reduce_mean_dim(dim, keepdim))
    }

    #[pyo3(signature = (dim, keepdim = false))]
    fn reduce_max(&self, dim: isize, keepdim: bool) -> PyResult<Self> {
        let dim = resolve_dim(&self.inner, dim, "reduce_max")?;
        wrap_tensor_op(|| self.inner.reduce_max_dim(dim, keepdim))
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn reduce_all(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        let dim = resolve_optional_dim(&self.inner, dim, "reduce_all")?;
        wrap_tensor_op(|| match dim {
            Some(dim) => self.inner.reduce_all_dim(dim, keepdim),
            None => self.inner.all(),
        })
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn reduce_any(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        let dim = resolve_optional_dim(&self.inner, dim, "reduce_any")?;
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

    fn reshape(&self, shape: Vec<isize>) -> PyResult<Self> {
        let shape = resolve_reshape_shape(&self.inner, &shape)?;
        wrap_tensor_op(|| self.inner.reshape(shape))
    }

    #[pyo3(signature = (start_dim = 0, end_dim = -1))]
    fn flatten(&self, start_dim: isize, end_dim: isize) -> PyResult<Self> {
        let shape = flatten_shape(&self.inner, start_dim, end_dim)?;
        wrap_tensor_op(|| self.inner.reshape(shape))
    }

    fn view(&self, shape: Vec<isize>) -> PyResult<Self> {
        self.reshape(shape)
    }

    fn unsqueeze(&self, dim: isize) -> PyResult<Self> {
        let dim = resolve_insert_dim(&self.inner, dim, "unsqueeze")?;
        let mut shape = self.inner.shape.clone();
        shape.insert(dim, 1);
        wrap_tensor_op(|| self.inner.reshape(shape))
    }

    #[pyo3(signature = (dim = None))]
    fn squeeze(&self, dim: Option<isize>) -> PyResult<Self> {
        let mut shape = self.inner.shape.clone();
        match dim {
            Some(dim) => {
                let dim = resolve_dim(&self.inner, dim, "squeeze")?;
                if shape[dim] == 1 {
                    shape.remove(dim);
                }
            }
            None => shape.retain(|dimension| *dimension != 1),
        }
        wrap_tensor_op(|| self.inner.reshape(shape))
    }

    fn broadcast(&self, shape: Vec<usize>) -> PyResult<Self> {
        validate_broadcast_shape(&self.inner, &shape)?;
        wrap_tensor_op(|| self.inner.broadcast(shape))
    }

    fn transpose2d(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.transpose2d())
    }

    #[getter]
    #[allow(non_snake_case)]
    fn T(&self) -> PyResult<Self> {
        match self.inner.shape.len() {
            0 | 1 => Ok(Clone::clone(self)),
            2 => self.transpose(0, 1),
            _ => Err(PyNotImplementedError::new_err(
                "Tensor.T currently supports tensors with at most 2 dimensions",
            )),
        }
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn contiguous(&self) -> Self {
        Clone::clone(self)
    }

    fn transpose(&self, dim0: isize, dim1: isize) -> PyResult<Self> {
        let dim0 = resolve_dim(&self.inner, dim0, "transpose")?;
        let dim1 = resolve_dim(&self.inner, dim1, "transpose")?;
        wrap_tensor_op(|| self.inner.transpose(dim0, dim1))
    }

    fn split(&self, dim: isize, sizes: Vec<usize>) -> PyResult<Vec<Self>> {
        let dim = resolve_dim(&self.inner, dim, "split")?;
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

    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        tensor_getitem(self, key)
    }

    fn maximum(&self, rhs: &PyTensor) -> PyResult<Self> {
        broadcast_binary(self, rhs, AutoTensor::maximum)
    }

    fn modulo(&self, rhs: &PyTensor) -> PyResult<Self> {
        broadcast_binary(self, rhs, AutoTensor::modulo)
    }

    fn equal(&self, rhs: &PyTensor) -> PyResult<Self> {
        broadcast_binary(self, rhs, AutoTensor::equal)
    }

    fn ones_like(&self) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.ones_like())
    }

    fn bias_add(&self, bias: &PyTensor) -> PyResult<Self> {
        wrap_tensor_op(|| self.inner.bias_add(&bias.inner))
    }

    #[pyo3(signature = (dim = None, eps = 1e-12))]
    fn l2_normalize(&self, dim: Option<isize>, eps: f64) -> PyResult<Self> {
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
        wrap_scalar_tensor_op(|| self.inner.mse_loss(&target.inner))
    }

    fn weighted_mse_loss(&self, target: &PyTensor, weights: &PyTensor) -> PyResult<Self> {
        wrap_scalar_tensor_op(|| self.inner.weighted_mse_loss(&target.inner, &weights.inner))
    }

    fn l2_loss(&self) -> PyResult<Self> {
        wrap_scalar_tensor_op(|| self.inner.l2_loss())
    }

    #[pyo3(signature = (target, beta = 1.0))]
    fn smooth_l1_loss(&self, target: &PyTensor, beta: f64) -> PyResult<Self> {
        validate_positive_finite(beta, "smooth_l1_loss beta")?;
        wrap_scalar_tensor_op(|| self.inner.smooth_l1_loss(&target.inner, beta))
    }

    fn sigmoid_cross_entropy_with_logits(&self, target: &PyTensor) -> PyResult<Self> {
        wrap_scalar_tensor_op(|| self.inner.sigmoid_cross_entropy_with_logits(&target.inner))
    }

    fn softmax_cross_entropy_with_logits(&self, target: &PyTensor) -> PyResult<Self> {
        wrap_scalar_tensor_op(|| self.inner.softmax_cross_entropy_with_logits(&target.inner))
    }

    #[pyo3(signature = (other, target, margin = 0.0))]
    fn cosine_embedding_loss(
        &self,
        other: &PyTensor,
        target: &PyTensor,
        margin: f64,
    ) -> PyResult<Self> {
        wrap_scalar_tensor_op(|| {
            self.inner
                .cosine_embedding_loss(&other.inner, &target.inner, margin)
        })
    }

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.add(rhs)
    }

    fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => broadcast_binary(&lhs, self, |lhs, rhs| lhs + rhs),
            TensorOperand::Scalar(lhs) => {
                let scalar = self.scalar_tensor_like(lhs)?;
                wrap_tensor_op(|| &scalar + &self.inner)
            }
        }
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.sub(rhs)
    }

    fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => broadcast_binary(&lhs, self, |lhs, rhs| lhs - rhs),
            TensorOperand::Scalar(lhs) => {
                let scalar = self.scalar_tensor_like(lhs)?;
                wrap_tensor_op(|| &scalar - &self.inner)
            }
        }
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.mul(rhs)
    }

    fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => broadcast_binary(&lhs, self, |lhs, rhs| lhs * rhs),
            TensorOperand::Scalar(lhs) => {
                let scalar = self.scalar_tensor_like(lhs)?;
                wrap_tensor_op(|| &scalar * &self.inner)
            }
        }
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.div(rhs)
    }

    fn __rtruediv__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => broadcast_binary(&lhs, self, |lhs, rhs| lhs / rhs),
            TensorOperand::Scalar(lhs) => {
                let scalar = self.scalar_tensor_like(lhs)?;
                wrap_tensor_op(|| &scalar / &self.inner)
            }
        }
    }

    fn __matmul__(&self, rhs: &PyTensor) -> PyResult<Self> {
        self.matmul(rhs)
    }

    fn __rmatmul__(&self, lhs: &PyTensor) -> PyResult<Self> {
        lhs.matmul(self)
    }

    fn __mod__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        match tensor_operand(rhs)? {
            TensorOperand::Tensor(rhs) => self.modulo(&rhs),
            TensorOperand::Scalar(rhs) => {
                let scalar = self.scalar_tensor_like(rhs)?;
                wrap_tensor_op(|| self.inner.modulo(&scalar))
            }
        }
    }

    fn __rmod__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        match tensor_operand(lhs)? {
            TensorOperand::Tensor(lhs) => lhs.modulo(self),
            TensorOperand::Scalar(lhs) => {
                let scalar = self.scalar_tensor_like(lhs)?;
                wrap_tensor_op(|| scalar.modulo(&self.inner))
            }
        }
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

    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .shape
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("len() of a 0-d tensor"))
    }

    fn __repr__(&self) -> String {
        let requires_grad = if self.state.requires_grad() {
            ", requires_grad=True"
        } else {
            ""
        };
        format!(
            "tensor({:?}, dtype=talos_xii.{}, device='{}'{requires_grad})",
            self.inner.data_as_f64_vec(),
            PyDType::new(self.inner.dtype).public_name(),
            device_name(self.inner.device),
        )
    }
}

impl PyTensor {
    fn from_tensor(mut inner: AutoTensor) -> Self {
        let grad_enabled = GRAD_ENABLED.with(Cell::get);
        let state = PythonAutogradState::operation(&mut inner, grad_enabled);
        Self { inner, state }
    }

    fn from_leaf(inner: AutoTensor, requires_grad: bool) -> Self {
        let state = PythonAutogradState::leaf(&inner, requires_grad);
        Self { inner, state }
    }

    fn from_alias(
        inner: AutoTensor,
        source: &PyTensor,
        requires_grad: bool,
        is_leaf: bool,
    ) -> Self {
        let state = PythonAutogradState::alias(&inner, &source.state, requires_grad, is_leaf);
        Self { inner, state }
    }

    fn scalar_tensor_like(&self, value: f64) -> PyResult<AutoTensor> {
        let scalar = AutoTensor::with_dtype(
            vec![value; self.inner.numel()],
            self.inner.shape.clone(),
            self.inner.dtype,
        );
        #[cfg(cuda)]
        if self.inner.device == Device::Cuda {
            return scalar.to_cuda().map_err(|error| {
                PyRuntimeError::new_err(format!("CUDA scalar transfer failed: {error}"))
            });
        }
        Ok(scalar)
    }

    fn set_requires_grad_value(&self, value: bool) -> PyResult<()> {
        if !self.state.is_leaf() {
            return Err(PyRuntimeError::new_err(
                "you can only change requires_grad flags of leaf variables",
            ));
        }
        if value && self.inner.dtype == Dtype::I8 {
            return Err(PyRuntimeError::new_err(
                "only floating point tensors can require gradients",
            ));
        }
        self.state.set_requires_grad(value);
        Ok(())
    }

    fn clone_tensor(&self) -> Self {
        let mut cloned = AutoTensor {
            data: Storage::from_f64_vec(self.inner.data_as_f64_vec(), self.inner.dtype),
            grad: Storage::zeros(
                self.inner.numel(),
                AutoTensor::grad_dtype_for(self.inner.dtype),
            ),
            shape: self.inner.shape.clone(),
            device: self.inner.device,
            dtype: self.inner.dtype,
            _ctx: None,
        };
        if GRAD_ENABLED.with(Cell::get) && self.state.requires_grad() {
            cloned._ctx = Some(Arc::new(Context {
                parents: vec![self.inner.clone()],
                backward_op: Box::new(|grad_out, parents| {
                    parents[0].accumulate_external_gradient(&grad_out.to_f64_vec());
                }),
            }));
        }
        Self::from_tensor(cloned)
    }

    fn converted_tensor(&self, dtype: Dtype, device: PyDevice, copy: bool) -> PyResult<Self> {
        let target_device = device.to_core()?;
        if dtype == self.inner.dtype && target_device == self.inner.device && !copy {
            return Ok(Clone::clone(self));
        }
        if dtype == Dtype::I8 && self.state.requires_grad() {
            return Err(PyRuntimeError::new_err(
                "only floating point tensors can require gradients",
            ));
        }

        let mut converted = AutoTensor {
            data: Storage::from_f64_vec(self.inner.data_as_f64_vec(), dtype),
            grad: Storage::zeros(self.inner.numel(), AutoTensor::grad_dtype_for(dtype)),
            shape: self.inner.shape.clone(),
            device: Device::Cpu,
            dtype,
            _ctx: None,
        };
        if GRAD_ENABLED.with(Cell::get) && self.state.requires_grad() {
            converted._ctx = Some(Arc::new(Context {
                parents: vec![self.inner.clone()],
                backward_op: Box::new(|grad_out, parents| {
                    parents[0].accumulate_external_gradient(&grad_out.to_f64_vec());
                }),
            }));
        }

        #[cfg(cuda)]
        if target_device == Device::Cuda {
            converted = converted.to_cuda().map_err(|error| {
                PyRuntimeError::new_err(format!("CUDA transfer failed: {error}"))
            })?;
        }
        let _ = target_device;
        Ok(Self::from_tensor(converted))
    }
}

fn nested_list(py: Python<'_>, data: &[f64], shape: &[usize], dtype: Dtype) -> PyResult<Py<PyAny>> {
    if shape.is_empty() {
        let value = data
            .first()
            .copied()
            .ok_or_else(|| PyRuntimeError::new_err("scalar tensor has no data"))?;
        return match dtype {
            Dtype::I8 => Ok((value as i8).into_pyobject(py)?.into_any().unbind()),
            _ => Ok(value.into_pyobject(py)?.into_any().unbind()),
        };
    }

    let chunk = shape[1..]
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| PyRuntimeError::new_err("tensor shape element count overflow"))?;
    let mut items = Vec::with_capacity(shape[0]);
    for index in 0..shape[0] {
        let start = index
            .checked_mul(chunk)
            .ok_or_else(|| PyRuntimeError::new_err("tensor index overflow"))?;
        let end = start
            .checked_add(chunk)
            .ok_or_else(|| PyRuntimeError::new_err("tensor index overflow"))?;
        items.push(nested_list(py, &data[start..end], &shape[1..], dtype)?);
    }
    Ok(PyList::new(py, items)?.into_any().unbind())
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction(name = "_is_grad_enabled")]
fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

#[pyfunction(name = "_set_grad_enabled")]
fn set_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.with(|state| state.replace(enabled))
}

#[pyfunction]
fn no_grad() -> PyGradMode {
    PyGradMode {
        enabled: false,
        previous: Vec::new(),
    }
}

#[pyfunction]
fn enable_grad() -> PyGradMode {
    PyGradMode {
        enabled: true,
        previous: Vec::new(),
    }
}

#[pyfunction(name = "set_grad_enabled")]
fn set_grad_enabled_context(enabled: bool) -> PyGradMode {
    PyGradMode {
        enabled,
        previous: Vec::new(),
    }
}

#[pyfunction(name = "is_grad_enabled")]
fn is_grad_enabled_public() -> bool {
    is_grad_enabled()
}

#[pyfunction(name = "_cuda_is_available")]
fn cuda_is_available() -> bool {
    crate::cuda::is_available()
}

#[pyfunction(name = "_cuda_device_count")]
fn cuda_device_count() -> usize {
    crate::cuda::device_count().unwrap_or(0)
}

#[pyfunction]
#[pyo3(signature = (data, shape = None, dtype = None, *, device = None, requires_grad = false))]
fn tensor(
    data: &Bound<'_, PyAny>,
    shape: Option<Vec<usize>>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    make_tensor(data, shape, dtype, device, requires_grad)
}

#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype = None, *, device = None, requires_grad = false))]
fn full(
    shape: Vec<usize>,
    fill_value: f64,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let dtype = parse_dtype_arg(dtype)?;
    let len = checked_numel(&shape)?;
    finish_leaf(
        AutoTensor::with_dtype(vec![fill_value; len], shape, dtype),
        device,
        requires_grad,
    )
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None, *, device = None, requires_grad = false))]
fn zeros(
    shape: Vec<usize>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let dtype = parse_dtype_arg(dtype)?;
    let len = checked_numel(&shape)?;
    finish_leaf(
        AutoTensor::with_dtype(vec![0.0; len], shape, dtype),
        device,
        requires_grad,
    )
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None, *, device = None, requires_grad = false))]
fn ones(
    shape: Vec<usize>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let dtype = parse_dtype_arg(dtype)?;
    let len = checked_numel(&shape)?;
    finish_leaf(
        AutoTensor::with_dtype(vec![1.0; len], shape, dtype),
        device,
        requires_grad,
    )
}

#[pyfunction(name = "rand", signature = (shape, *, min = 0.0, max = 1.0, seed = 42, dtype = None, device = None, requires_grad = false))]
fn py_rand(
    shape: Vec<usize>,
    min: f64,
    max: f64,
    seed: u64,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let dtype = parse_dtype_arg(dtype)?;
    let len = checked_numel(&shape)?;
    if !min.is_finite() || !max.is_finite() || min >= max {
        return Err(PyValueError::new_err(
            "rand requires finite bounds with min < max",
        ));
    }
    let mut rng = crate::rng::Rng::from_seed(seed);
    let data: Vec<f64> = (0..len)
        .map(|_| {
            let t = rng.next_f64();
            min + t * (max - min)
        })
        .collect();
    finish_leaf(
        AutoTensor::with_dtype(data, shape, dtype),
        device,
        requires_grad,
    )
}

#[pyfunction(name = "randn", signature = (shape, *, seed = 42, dtype = None, device = None, requires_grad = false))]
fn py_randn(
    shape: Vec<usize>,
    seed: u64,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let dtype = parse_dtype_arg(dtype)?;
    let len = checked_numel(&shape)?;
    let mut rng = crate::rng::Rng::from_seed(seed);
    let data: Vec<f64> = (0..len).map(|_| rng.next_f64_normal()).collect();
    finish_leaf(
        AutoTensor::with_dtype(data, shape, dtype),
        device,
        requires_grad,
    )
}

#[pyfunction]
#[pyo3(signature = (start, stop = None, step = 1.0, dtype = None, *, device = None, requires_grad = false))]
fn arange(
    start: f64,
    stop: Option<f64>,
    step: f64,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
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

    let dtype = parse_dtype_arg(dtype)?;
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
    finish_leaf(
        AutoTensor::with_dtype(data, vec![len], dtype),
        device,
        requires_grad,
    )
}

#[pyfunction]
#[pyo3(signature = (n, m = None, dtype = None, *, device = None, requires_grad = false))]
fn eye(
    n: usize,
    m: Option<usize>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let dtype = parse_dtype_arg(dtype)?;
    let cols = m.unwrap_or(n);
    let len = n
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("eye shape element count overflow"))?;
    validate_tensor_len(len, "eye")?;
    let mut data = vec![0.0; len];
    for idx in 0..n.min(cols) {
        data[idx * cols + idx] = 1.0;
    }
    finish_leaf(
        AutoTensor::with_dtype(data, vec![n, cols], dtype),
        device,
        requires_grad,
    )
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
fn shape(input: &PyTensor, py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(PyTuple::new(py, input.inner.shape.iter().copied())?
        .into_any()
        .unbind())
}

#[pyfunction(name = "get_dtype")]
fn tensor_dtype(input: &PyTensor) -> PyDType {
    input.dtype()
}

#[pyfunction(name = "get_device")]
fn tensor_device(input: &PyTensor) -> PyDevice {
    input.device()
}

#[pyfunction]
fn to_list(input: &PyTensor) -> PyResult<Vec<f64>> {
    input.to_list()
}

#[pyfunction]
fn tolist(input: &PyTensor, py: Python<'_>) -> PyResult<Py<PyAny>> {
    input.tolist(py)
}

#[pyfunction]
fn data(input: &PyTensor) -> PyTensor {
    input.data()
}

#[pyfunction]
fn grad(input: &PyTensor) -> PyResult<Option<PyTensor>> {
    input.grad()
}

#[pyfunction]
fn zero_grad(input: &PyTensor) {
    input.zero_grad();
}

#[pyfunction]
#[pyo3(signature = (input, gradient = None, retain_graph = None, create_graph = false))]
fn backward(
    mut input: PyRefMut<'_, PyTensor>,
    gradient: Option<&PyTensor>,
    retain_graph: Option<bool>,
    create_graph: bool,
) -> PyResult<()> {
    input.backward(gradient, retain_graph, create_graph)
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
fn item(input: &PyTensor, py: Python<'_>) -> PyResult<Py<PyAny>> {
    input.item(py)
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
tensor_unary_pyfunction!(transpose2d);

#[pyfunction]
#[pyo3(signature = (input, start_dim = 0, end_dim = -1))]
fn flatten(input: &PyTensor, start_dim: isize, end_dim: isize) -> PyResult<PyTensor> {
    input.flatten(start_dim, end_dim)
}

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
fn reshape(input: &PyTensor, shape: Vec<isize>) -> PyResult<PyTensor> {
    input.reshape(shape)
}

#[pyfunction]
fn unsqueeze(input: &PyTensor, dim: isize) -> PyResult<PyTensor> {
    input.unsqueeze(dim)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None))]
fn squeeze(input: &PyTensor, dim: Option<isize>) -> PyResult<PyTensor> {
    input.squeeze(dim)
}

#[pyfunction]
fn broadcast(input: &PyTensor, shape: Vec<usize>) -> PyResult<PyTensor> {
    input.broadcast(shape)
}

#[pyfunction]
fn transpose(input: &PyTensor, dim0: isize, dim1: isize) -> PyResult<PyTensor> {
    input.transpose(dim0, dim1)
}

#[pyfunction]
fn split(input: &PyTensor, dim: isize, sizes: Vec<usize>) -> PyResult<Vec<PyTensor>> {
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
fn sum(input: &PyTensor, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    input.sum(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None, keepdim = false))]
fn mean(input: &PyTensor, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    input.mean(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None, keepdim = false))]
fn max(input: &PyTensor, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    input.max(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim, keepdim = false))]
fn reduce_sum(input: &PyTensor, dim: isize, keepdim: bool) -> PyResult<PyTensor> {
    input.reduce_sum(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim, keepdim = false))]
fn reduce_mean(input: &PyTensor, dim: isize, keepdim: bool) -> PyResult<PyTensor> {
    input.reduce_mean(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim, keepdim = false))]
fn reduce_max(input: &PyTensor, dim: isize, keepdim: bool) -> PyResult<PyTensor> {
    input.reduce_max(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None, keepdim = false))]
fn reduce_all(input: &PyTensor, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    input.reduce_all(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None, keepdim = false))]
fn reduce_any(input: &PyTensor, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    input.reduce_any(dim, keepdim)
}

#[pyfunction]
fn concat(tensors: &Bound<'_, PyAny>, dim: isize) -> PyResult<PyTensor> {
    let tensors = tensor_sequence(tensors)?;
    let dim = resolve_sequence_dim(&tensors, dim, "concat")?;
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
    lhs.maximum(rhs)
}

#[pyfunction]
fn equal(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.equal(rhs)
}

#[pyfunction]
fn matmul(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.matmul(rhs)
}

#[pyfunction]
fn gemm(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    matmul(lhs, rhs)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None))]
fn softmax(input: &PyTensor, dim: Option<isize>) -> PyResult<PyTensor> {
    let resolved_dim = resolve_dim_option(&input.inner, dim, "softmax")?;
    wrap_tensor_op(|| input.inner.softmax_dim(resolved_dim))
}

#[pyfunction]
#[pyo3(signature = (input, dim = None))]
fn softmax_v2(input: &PyTensor, dim: Option<isize>) -> PyResult<PyTensor> {
    softmax(input, dim)
}

#[pyfunction]
#[pyo3(signature = (input, dim = None))]
fn log_softmax(input: &PyTensor, dim: Option<isize>) -> PyResult<PyTensor> {
    let resolved_dim = resolve_dim_option(&input.inner, dim, "log_softmax")?;
    wrap_tensor_op(|| input.inner.log_softmax_dim(resolved_dim))
}

#[pyfunction]
#[pyo3(signature = (input, dim = None))]
fn log_softmax_v2(input: &PyTensor, dim: Option<isize>) -> PyResult<PyTensor> {
    log_softmax(input, dim)
}

#[pyfunction]
fn modulo(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.modulo(rhs)
}

#[pyfunction]
fn l2_loss(input: &PyTensor) -> PyResult<PyTensor> {
    input.l2_loss()
}

#[pyfunction]
fn mse_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    input.mse_loss(target)
}

#[pyfunction]
fn weighted_mse_loss(
    input: &PyTensor,
    target: &PyTensor,
    weights: &PyTensor,
) -> PyResult<PyTensor> {
    input.weighted_mse_loss(target, weights)
}

#[pyfunction]
#[pyo3(signature = (input, target, beta = 1.0))]
fn smooth_l1_loss(input: &PyTensor, target: &PyTensor, beta: f64) -> PyResult<PyTensor> {
    input.smooth_l1_loss(target, beta)
}

#[pyfunction]
fn sigmoid_cross_entropy_with_logits(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    input.sigmoid_cross_entropy_with_logits(target)
}

#[pyfunction]
fn softmax_cross_entropy_with_logits(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    input.softmax_cross_entropy_with_logits(target)
}

#[pyfunction]
#[pyo3(signature = (input, other, target, margin = 0.0))]
fn cosine_embedding_loss(
    input: &PyTensor,
    other: &PyTensor,
    target: &PyTensor,
    margin: f64,
) -> PyResult<PyTensor> {
    input.cosine_embedding_loss(other, target, margin)
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
fn l2_normalize(input: &PyTensor, dim: Option<isize>, eps: f64) -> PyResult<PyTensor> {
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
    m.add_class::<PyDType>()?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PyGradMode>()?;
    let py = m.py();
    let float32 = Py::new(py, PyDType::new(Dtype::F32))?;
    let float64 = Py::new(py, PyDType::new(Dtype::F64))?;
    let bfloat16 = Py::new(py, PyDType::new(Dtype::BF16))?;
    let int8 = Py::new(py, PyDType::new(Dtype::I8))?;
    m.add("float32", float32.clone_ref(py))?;
    m.add("float64", float64.clone_ref(py))?;
    m.add("bfloat16", bfloat16.clone_ref(py))?;
    m.add("int8", int8.clone_ref(py))?;
    m.add("float", float32.clone_ref(py))?;
    m.add("double", float64.clone_ref(py))?;
    m.add("F32", float32)?;
    m.add("F64", float64)?;
    m.add("BF16", bfloat16)?;
    m.add("I8", int8)?;
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(is_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(set_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(no_grad, m)?)?;
    m.add_function(wrap_pyfunction!(enable_grad, m)?)?;
    m.add_function(wrap_pyfunction!(set_grad_enabled_context, m)?)?;
    m.add_function(wrap_pyfunction!(is_grad_enabled_public, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_device_count, m)?)?;
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
    m.add_function(wrap_pyfunction!(tensor_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_device, m)?)?;
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
    m.add_function(wrap_pyfunction!(unsqueeze, m)?)?;
    m.add_function(wrap_pyfunction!(squeeze, m)?)?;
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

fn make_tensor(
    data: &Bound<'_, PyAny>,
    shape: Option<Vec<usize>>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let dtype = parse_dtype_arg(dtype)?;
    let (values, inferred_shape) = flatten_python_data(data)?;
    let shape = shape.unwrap_or(inferred_shape);
    let expected = checked_numel(&shape)?;
    validate_tensor_len(values.len(), "tensor data")?;
    if values.len() != expected {
        return Err(PyValueError::new_err(format!(
            "data length {} does not match shape {:?} ({} elements)",
            values.len(),
            shape,
            expected
        )));
    }
    finish_leaf(
        AutoTensor::with_dtype(values, shape, dtype),
        device,
        requires_grad,
    )
}

fn flatten_python_data(data: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>)> {
    if let Ok(tensor) = data.extract::<PyRef<'_, PyTensor>>() {
        return Ok((tensor.inner.data_as_f64_vec(), tensor.inner.shape.clone()));
    }
    if let Ok(value) = data.extract::<f64>() {
        return Ok((vec![value], Vec::new()));
    }

    let items: Vec<Bound<'_, PyAny>> = if let Ok(list) = data.cast::<PyList>() {
        list.iter().collect()
    } else if let Ok(tuple) = data.cast::<PyTuple>() {
        tuple.iter().collect()
    } else {
        return Err(PyTypeError::new_err(
            "tensor data must be a number or a nested list/tuple of numbers",
        ));
    };

    if items.is_empty() {
        return Ok((Vec::new(), vec![0]));
    }

    let mut values = Vec::new();
    let mut child_shape: Option<Vec<usize>> = None;
    for item in items {
        let (child_values, shape) = flatten_python_data(&item)?;
        if let Some(expected) = &child_shape {
            if expected != &shape {
                return Err(PyValueError::new_err(
                    "expected a rectangular nested sequence, found a ragged value",
                ));
            }
        } else {
            child_shape = Some(shape);
        }
        values.extend(child_values);
    }
    let mut shape = vec![values_shape_len(data)?];
    shape.extend(child_shape.unwrap_or_default());
    Ok((values, shape))
}

fn values_shape_len(data: &Bound<'_, PyAny>) -> PyResult<usize> {
    data.len()
        .map_err(|_| PyTypeError::new_err("tensor data must be a sized sequence"))
}

fn finish_leaf(
    tensor: AutoTensor,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    #[allow(unused_mut)]
    let mut tensor = tensor;
    if requires_grad && tensor.dtype == Dtype::I8 {
        return Err(PyRuntimeError::new_err(
            "only floating point tensors can require gradients",
        ));
    }
    let requested_device = parse_device_arg(device)?;
    let _ = requested_device.to_core()?;
    if requested_device.is_cuda() {
        #[cfg(cuda)]
        {
            tensor = tensor.to_cuda().map_err(|error| {
                PyRuntimeError::new_err(format!("CUDA transfer failed: {error}"))
            })?;
        }
        #[cfg(not(cuda))]
        unreachable!("parse_device_arg rejects CUDA for CPU-only builds");
    }
    Ok(PyTensor::from_leaf(tensor, requires_grad))
}

#[cfg(test)]
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

fn resolve_dim(tensor: &AutoTensor, dim: isize, op: &str) -> PyResult<usize> {
    let rank = isize::try_from(tensor.shape.len())
        .map_err(|_| PyValueError::new_err(format!("{op} tensor rank is too large")))?;
    let resolved = if dim < 0 { rank + dim } else { dim };
    if resolved < 0 || resolved >= rank {
        return Err(PyValueError::new_err(format!(
            "{op} dim {dim} out of range for rank {}",
            tensor.shape.len()
        )));
    }
    usize::try_from(resolved)
        .map_err(|_| PyValueError::new_err(format!("{op} dim {dim} is invalid")))
}

fn resolve_insert_dim(tensor: &AutoTensor, dim: isize, op: &str) -> PyResult<usize> {
    let rank = isize::try_from(tensor.shape.len())
        .map_err(|_| PyValueError::new_err(format!("{op} tensor rank is too large")))?;
    let resolved = if dim < 0 { rank + dim + 1 } else { dim };
    if resolved < 0 || resolved > rank {
        return Err(PyValueError::new_err(format!(
            "{op} dim {dim} out of range for tensor of dimension {}",
            tensor.shape.len()
        )));
    }
    usize::try_from(resolved)
        .map_err(|_| PyValueError::new_err(format!("{op} dim {dim} is invalid")))
}

fn flatten_shape(tensor: &AutoTensor, start_dim: isize, end_dim: isize) -> PyResult<Vec<usize>> {
    if tensor.shape.is_empty() {
        if (start_dim == 0 || start_dim == -1) && (end_dim == 0 || end_dim == -1) {
            return Ok(vec![1]);
        }
        return Err(PyValueError::new_err(
            "flatten dimensions out of range for a scalar tensor",
        ));
    }
    let start = resolve_dim(tensor, start_dim, "flatten")?;
    let end = resolve_dim(tensor, end_dim, "flatten")?;
    if start > end {
        return Err(PyValueError::new_err(
            "flatten start_dim cannot come after end_dim",
        ));
    }
    let flattened = tensor.shape[start..=end]
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| PyValueError::new_err("flatten shape element count overflow"))?;
    let mut shape = tensor.shape[..start].to_vec();
    shape.push(flattened);
    shape.extend_from_slice(&tensor.shape[end + 1..]);
    Ok(shape)
}

fn resolve_optional_dim(
    tensor: &AutoTensor,
    dim: Option<isize>,
    op: &str,
) -> PyResult<Option<usize>> {
    dim.map(|dim| resolve_dim(tensor, dim, op)).transpose()
}

fn resolve_dim_option(tensor: &AutoTensor, dim: Option<isize>, op: &str) -> PyResult<usize> {
    match dim {
        Some(dim) => resolve_dim(tensor, dim, op),
        None if tensor.shape.is_empty() => Err(PyValueError::new_err(format!(
            "{op} requires rank >= 1 when dim is not provided"
        ))),
        None => Ok(tensor.shape.len() - 1),
    }
}

fn resolve_sequence_dim(tensors: &[AutoTensor], dim: isize, op: &str) -> PyResult<usize> {
    let first = tensors
        .first()
        .ok_or_else(|| PyValueError::new_err(format!("{op} requires at least one Tensor")))?;
    resolve_dim(first, dim, op)
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

fn resolve_reshape_shape(tensor: &AutoTensor, shape: &[isize]) -> PyResult<Vec<usize>> {
    let mut resolved = Vec::with_capacity(shape.len());
    let mut inferred_index = None;
    let mut known_elements = 1usize;
    for (index, dim) in shape.iter().copied().enumerate() {
        match dim {
            -1 => {
                if inferred_index.replace(index).is_some() {
                    return Err(PyValueError::new_err("only one dimension can be inferred"));
                }
                resolved.push(1);
            }
            dim if dim < 0 => {
                return Err(PyValueError::new_err(format!(
                    "invalid shape dimension {dim}"
                )));
            }
            dim => {
                let dim = usize::try_from(dim)
                    .map_err(|_| PyValueError::new_err("shape dimension overflow"))?;
                known_elements = known_elements
                    .checked_mul(dim)
                    .ok_or_else(|| PyValueError::new_err("shape element count overflow"))?;
                resolved.push(dim);
            }
        }
    }

    if let Some(index) = inferred_index {
        if known_elements == 0 || !tensor.numel().is_multiple_of(known_elements) {
            return Err(PyValueError::new_err(format!(
                "shape {:?} is invalid for input of size {}",
                shape,
                tensor.numel()
            )));
        }
        resolved[index] = tensor.numel() / known_elements;
    }

    let len = checked_numel(&resolved)?;
    if len != tensor.numel() {
        return Err(PyValueError::new_err(format!(
            "reshape size mismatch: tensor has {} elements, requested shape {:?} has {len}",
            tensor.numel(),
            shape
        )));
    }
    Ok(resolved)
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

fn validate_matmul(lhs: &AutoTensor, rhs: &AutoTensor) -> PyResult<()> {
    if lhs.device != rhs.device {
        return Err(PyRuntimeError::new_err(
            "expected both matmul tensors to be on the same device",
        ));
    }
    if !(lhs.shape.len() == 1 || lhs.shape.len() == 2) || rhs.shape.len() != 2 {
        return Err(PyNotImplementedError::new_err(
            "matmul currently supports a 1-D or 2-D left operand and a 2-D right operand",
        ));
    }
    let lhs_inner = *lhs
        .shape
        .last()
        .ok_or_else(|| PyRuntimeError::new_err("matmul left operand has no dimensions"))?;
    if lhs_inner != rhs.shape[0] {
        return Err(PyRuntimeError::new_err(format!(
            "mat1 and mat2 shapes cannot be multiplied ({:?} and {:?})",
            lhs.shape, rhs.shape
        )));
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

fn broadcast_binary<F>(lhs: &PyTensor, rhs: &PyTensor, operation: F) -> PyResult<PyTensor>
where
    F: FnOnce(&AutoTensor, &AutoTensor) -> AutoTensor,
{
    if lhs.inner.device != rhs.inner.device {
        return Err(PyRuntimeError::new_err(format!(
            "expected all tensors to be on the same device, found {} and {}",
            device_name(lhs.inner.device),
            device_name(rhs.inner.device)
        )));
    }
    let shape = broadcast_shape(&lhs.inner.shape, &rhs.inner.shape)?;
    let lhs = if lhs.inner.shape == shape {
        Clone::clone(lhs)
    } else {
        PyTensor::from_tensor(wrap_value_op(|| lhs.inner.broadcast(shape.clone()))?)
    };
    let rhs = if rhs.inner.shape == shape {
        Clone::clone(rhs)
    } else {
        PyTensor::from_tensor(wrap_value_op(|| rhs.inner.broadcast(shape.clone()))?)
    };
    wrap_tensor_op(|| operation(&lhs.inner, &rhs.inner))
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> PyResult<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut shape = vec![1usize; rank];
    for offset in 0..rank {
        let lhs_dim = lhs
            .len()
            .checked_sub(offset + 1)
            .map_or(1, |index| lhs[index]);
        let rhs_dim = rhs
            .len()
            .checked_sub(offset + 1)
            .map_or(1, |index| rhs[index]);
        let output = if lhs_dim == rhs_dim {
            lhs_dim
        } else if lhs_dim == 1 {
            rhs_dim
        } else if rhs_dim == 1 {
            lhs_dim
        } else {
            return Err(PyRuntimeError::new_err(format!(
                "The size of tensor a ({lhs_dim}) must match the size of tensor b ({rhs_dim}) at non-singleton dimension {}",
                rank - offset - 1
            )));
        };
        shape[rank - offset - 1] = output;
    }
    checked_numel(&shape)?;
    Ok(shape)
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

fn tensor_getitem(tensor: &PyTensor, key: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    if tensor.inner.shape.is_empty() {
        return Err(PyIndexError::new_err(
            "invalid index of a 0-dim tensor; use tensor.item()",
        ));
    }
    let keys: Vec<Bound<'_, PyAny>> = if let Ok(tuple) = key.cast::<PyTuple>() {
        tuple.iter().collect()
    } else {
        vec![key.clone()]
    };
    if keys.len() > tensor.inner.shape.len() {
        return Err(PyIndexError::new_err(format!(
            "too many indices for tensor of dimension {}",
            tensor.inner.shape.len()
        )));
    }

    let rank = tensor.inner.shape.len();
    let mut begin = Vec::with_capacity(rank);
    let mut end = Vec::with_capacity(rank);
    let mut strides = Vec::with_capacity(rank);
    let mut drop_dimension = vec![false; rank];

    for (axis, drop) in drop_dimension.iter_mut().enumerate().take(rank) {
        let dim_size = tensor.inner.shape[axis];
        let Some(index) = keys.get(axis) else {
            begin.push(0);
            end.push(dim_size);
            strides.push(1);
            continue;
        };

        if let Ok(raw_index) = index.extract::<isize>() {
            let dim_size_signed = isize::try_from(dim_size)
                .map_err(|_| PyIndexError::new_err("tensor dimension is too large"))?;
            let resolved = if raw_index < 0 {
                dim_size_signed + raw_index
            } else {
                raw_index
            };
            if resolved < 0 || resolved >= dim_size_signed {
                return Err(PyIndexError::new_err(format!(
                    "index {raw_index} is out of bounds for dimension {axis} with size {dim_size}"
                )));
            }
            let resolved = usize::try_from(resolved)
                .map_err(|_| PyIndexError::new_err("tensor index is invalid"))?;
            begin.push(resolved);
            end.push(resolved + 1);
            strides.push(1);
            *drop = true;
            continue;
        }

        if let Ok(slice) = index.cast::<PySlice>() {
            let (start, stop, step) =
                slice
                    .call_method1("indices", (dim_size,))?
                    .extract::<(isize, isize, isize)>()?;
            if step <= 0 {
                return Err(PyNotImplementedError::new_err(
                    "negative and zero slice steps are not implemented",
                ));
            }
            begin.push(
                usize::try_from(start)
                    .map_err(|_| PyIndexError::new_err("slice start could not be represented"))?,
            );
            end.push(
                usize::try_from(stop)
                    .map_err(|_| PyIndexError::new_err("slice stop could not be represented"))?,
            );
            strides.push(
                usize::try_from(step)
                    .map_err(|_| PyIndexError::new_err("slice step could not be represented"))?,
            );
            continue;
        }

        return Err(PyIndexError::new_err(
            "indices must be integers, slices, or tuples of those values",
        ));
    }

    let sliced = PyTensor::from_tensor(wrap_value_op(|| {
        tensor.inner.strided_slice(begin, end, strides)
    })?);
    if !drop_dimension.iter().any(|drop| *drop) {
        return Ok(sliced);
    }
    let shape: Vec<usize> = sliced
        .inner
        .shape
        .iter()
        .copied()
        .zip(drop_dimension)
        .filter_map(|(dim, drop)| (!drop).then_some(dim))
        .collect();
    wrap_tensor_op(|| sliced.inner.reshape(shape))
}

fn wrap_tensor_op<F>(op: F) -> PyResult<PyTensor>
where
    F: FnOnce() -> AutoTensor,
{
    wrap_value_op(|| PyTensor::from_tensor(op()))
}

fn wrap_scalar_tensor_op<F>(op: F) -> PyResult<PyTensor>
where
    F: FnOnce() -> AutoTensor,
{
    let output = PyTensor::from_tensor(wrap_value_op(op)?);
    if output.inner.shape.is_empty() {
        Ok(output)
    } else if output.inner.numel() == 1 {
        wrap_tensor_op(|| output.inner.reshape(Vec::new()))
    } else {
        Err(PyRuntimeError::new_err(
            "loss operation did not produce a scalar tensor",
        ))
    }
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
assert x.shape == (2, 2)
assert x.dtype == tx.float32
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
assert z.shape == (2, 2)
assert z.to_list() == [2.0, 3.0, 4.0, 5.0]
assert z.tolist() == [[2.0, 3.0], [4.0, 5.0]]
assert abs(z.sum().item() - 14.0) < 1e-9

w = tx.tensor([2.0, 0.0, 1.0, 2.0], [2, 2])
m = x.matmul(w)
assert m.shape == (2, 2)
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
assert tx.shape(u) == (4,)
assert tx.get_dtype(u) == tx.float32
assert tx.get_device(u) == tx.device("cpu")
assert tx.to_list(u) == u.to_list()
assert tx.tolist(u) == u.to_list()
assert tx.data(u).to_list() == u.to_list()
assert tx.grad(u) is None
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

v = tx.tensor([1.0, 2.0, 4.0], [3], requires_grad=True)
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
assert mat.max(1, keepdim=True).shape == (2, 1)
assert_close_list(tx.sum(mat, 1).to_list(), mat.sum(1).to_list())
assert_close_list(tx.mean(mat, 0).to_list(), mat.mean(0).to_list())
assert tx.max(mat, 1, keepdim=True).shape == mat.max(1, keepdim=True).shape
assert_close_list(tx.reduce_sum(mat, 1).to_list(), mat.reduce_sum(1).to_list())
assert_close_list(tx.reduce_mean(mat, 0).to_list(), mat.reduce_mean(0).to_list())
assert tx.reduce_max(mat, 1, keepdim=True).shape == (2, 1)
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
assert tx.reshape(mat, [3, 2]).shape == (3, 2)
assert tx.flatten(mat).shape == (6,)
assert_close_list(tx.broadcast(tx.tensor([1.0, 2.0, 3.0], [3, 1]), [3, 2]).to_list(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
assert_close_list(tx.transpose2d(mat).to_list(), mat.transpose2d().to_list())
assert_close_list(tx.transpose(mat, 0, 1).to_list(), mat.transpose(0, 1).to_list())
cat = tx.concat([tx.tensor([1.0, 2.0], [1, 2]), tx.tensor([3.0, 4.0], [1, 2])], 0)
assert cat.shape == (2, 2)
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
assert_close_list(v.grad.to_list(), [2.0, 2.0, 2.0])
tx.zero_grad(v)
assert_close_list(tx.grad(v).to_list(), [0.0, 0.0, 0.0])
v2 = tx.tensor([1.0, 2.0], [2], requires_grad=True)
loss2 = tx.sum(v2 * 3.0)
tx.backward(loss2)
assert_close_list(tx.grad(v2).to_list(), [3.0, 3.0])

nested = tx.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tx.float64)
assert nested.shape == (2, 2)
assert nested.tolist() == [[1.0, 2.0], [3.0, 4.0]]
precise = tx.tensor(1.123456789012345, dtype=tx.float64)
assert precise.item() == 1.123456789012345
broadcasted = tx.ones([2, 3]) + tx.tensor([1.0, 2.0, 3.0])
assert broadcasted.tolist() == [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]
assert broadcasted.sum(-1).tolist() == [9.0, 9.0]
tracked = tx.tensor([1.0], requires_grad=True)
with tx.no_grad():
    untracked = tracked * 2.0
assert not untracked.requires_grad
assert tx.is_grad_enabled()
"#;

        std::fs::write(&script_path, script).unwrap();
        let result = run_script(&script_path, None, &[]);
        let _ = std::fs::remove_file(&script_path);

        assert_eq!(result.unwrap(), 0);
    }
}
