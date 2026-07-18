use crate::autograd::Device;
use crate::dtype::Dtype;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass(name = "dtype", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
pub(super) struct PyDType {
    inner: Dtype,
}

impl PyDType {
    pub(super) const fn new(inner: Dtype) -> Self {
        Self { inner }
    }

    pub(super) const fn inner(&self) -> Dtype {
        self.inner
    }

    pub(super) const fn public_name(&self) -> &'static str {
        match self.inner {
            Dtype::F64 => "float64",
            Dtype::F32 => "float32",
            Dtype::BF16 => "bfloat16",
            Dtype::I8 => "int8",
        }
    }
}

#[pymethods]
impl PyDType {
    fn __repr__(&self) -> String {
        format!("talos_xii.{}", self.public_name())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __hash__(&self) -> u8 {
        match self.inner {
            Dtype::F64 => 0,
            Dtype::F32 => 1,
            Dtype::BF16 => 2,
            Dtype::I8 => 3,
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyDType>>()
            .is_ok_and(|other| self.inner == other.inner)
    }
}

#[pyclass(name = "device", frozen, skip_from_py_object)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PyDevice {
    kind: String,
    index: Option<usize>,
}

impl PyDevice {
    pub(super) fn cpu() -> Self {
        Self {
            kind: "cpu".to_string(),
            index: None,
        }
    }

    pub(super) fn cuda(index: usize) -> Self {
        Self {
            kind: "cuda".to_string(),
            index: Some(index),
        }
    }

    pub(super) fn from_core(device: Device) -> Self {
        match device {
            Device::Cpu => Self::cpu(),
            #[cfg(cuda)]
            Device::Cuda => Self {
                kind: "cuda".to_string(),
                index: Some(0),
            },
        }
    }

    pub(super) fn is_cuda(&self) -> bool {
        self.kind == "cuda"
    }

    pub(super) fn to_core(&self) -> PyResult<Device> {
        match self.kind.as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" => {
                if self.index.unwrap_or(0) != 0 {
                    return Err(PyRuntimeError::new_err(
                        "Talos-XII currently supports only CUDA device index 0",
                    ));
                }
                #[cfg(cuda)]
                {
                    Ok(Device::Cuda)
                }
                #[cfg(not(cuda))]
                {
                    Err(PyRuntimeError::new_err(
                        "Talos-XII was built without CUDA support",
                    ))
                }
            }
            _ => Err(PyValueError::new_err(format!(
                "unsupported device type '{}'",
                self.kind
            ))),
        }
    }

    fn parse_text(spec: &str, explicit_index: Option<usize>) -> PyResult<Self> {
        let normalized = spec.trim().to_ascii_lowercase();
        let (kind, parsed_index) = match normalized.split_once(':') {
            Some((kind, index)) => {
                let index = index.parse::<usize>().map_err(|_| {
                    PyValueError::new_err(format!("invalid device string '{spec}'"))
                })?;
                (kind, Some(index))
            }
            None => (normalized.as_str(), None),
        };
        let index = match (parsed_index, explicit_index) {
            (Some(parsed), Some(explicit)) if parsed != explicit => {
                return Err(PyValueError::new_err(
                    "device index specified twice with different values",
                ));
            }
            (Some(parsed), _) => Some(parsed),
            (None, explicit) => explicit,
        };
        match kind {
            "cpu" => {
                if index.is_some() {
                    return Err(PyValueError::new_err("CPU device does not accept an index"));
                }
                Ok(Self::cpu())
            }
            "cuda" => Ok(Self {
                kind: "cuda".to_string(),
                index: Some(index.unwrap_or(0)),
            }),
            _ => Err(PyValueError::new_err(format!(
                "expected a cpu or cuda device, got '{spec}'"
            ))),
        }
    }
}

#[pymethods]
impl PyDevice {
    #[new]
    #[pyo3(signature = (device, index = None))]
    fn new(device: &Bound<'_, PyAny>, index: Option<usize>) -> PyResult<Self> {
        if let Ok(device) = device.extract::<PyRef<'_, PyDevice>>() {
            if index.is_some() && index != device.index {
                return Err(PyValueError::new_err(
                    "device index specified twice with different values",
                ));
            }
            return Ok(device.clone());
        }
        let spec = device
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("device() expects a string or talos_xii.device"))?;
        Self::parse_text(&spec, index)
    }

    #[getter]
    fn r#type(&self) -> &str {
        &self.kind
    }

    #[getter]
    fn index(&self) -> Option<usize> {
        self.index
    }

    fn __repr__(&self) -> String {
        match self.index {
            Some(index) => format!("device(type='{}', index={index})", self.kind),
            None => format!("device(type='{}')", self.kind),
        }
    }

    fn __str__(&self) -> String {
        match self.index {
            Some(index) => format!("{}:{index}", self.kind),
            None => self.kind.clone(),
        }
    }

    fn __hash__(&self) -> isize {
        let kind = if self.kind == "cpu" { 0isize } else { 1isize };
        kind.wrapping_mul(31)
            .wrapping_add(self.index.unwrap_or(usize::MAX) as isize)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyDevice>>()
            .is_ok_and(|other| *self == *other)
    }
}

pub(super) fn parse_dtype_arg(value: Option<&Bound<'_, PyAny>>) -> PyResult<Dtype> {
    let Some(value) = value else {
        return Ok(Dtype::F32);
    };
    if value.is_none() {
        return Ok(Dtype::F32);
    }
    if let Ok(dtype) = value.extract::<PyRef<'_, PyDType>>() {
        return Ok(dtype.inner());
    }
    let dtype = value
        .extract::<String>()
        .map_err(|_| PyTypeError::new_err("dtype must be a talos_xii.dtype or string"))?;
    match dtype.trim().to_ascii_lowercase().as_str() {
        "f64" | "float64" | "double" | "talos_xii.float64" => Ok(Dtype::F64),
        "f32" | "float32" | "float" | "talos_xii.float32" => Ok(Dtype::F32),
        "bf16" | "bfloat16" | "talos_xii.bfloat16" => Ok(Dtype::BF16),
        "i8" | "int8" | "talos_xii.int8" => Ok(Dtype::I8),
        _ => Err(PyValueError::new_err(
            "unsupported dtype; expected float64, float32, bfloat16, or int8",
        )),
    }
}

pub(super) fn parse_device_arg(value: Option<&Bound<'_, PyAny>>) -> PyResult<PyDevice> {
    let Some(value) = value else {
        return Ok(PyDevice::cpu());
    };
    if value.is_none() {
        return Ok(PyDevice::cpu());
    }
    PyDevice::new(value, None)
}
