use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

/// Data type enumeration for tensor elements.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum Dtype {
    /// 64-bit float (backward compatibility)
    F64,
    /// 32-bit float (default)
    #[default]
    F32,
    /// Brain float 16
    BF16,
    /// Signed 8-bit integer (quantized weights)
    I8,
}

/// Brain float 16 wrapper (16 bits: 1 sign, 8 exponent, 7 mantissa).
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct bf16(pub u16);

impl bf16 {
    /// Convert from f32 to bf16.
    pub fn from_f32(v: f32) -> Self {
        let bits = v.to_bits();
        // Round to nearest even: add 0x7FFF + LSB before shifting
        let rounded = ((bits >> 16) & 1) + 0x7FFF + (bits & 0xFFFF);
        bf16(((bits + rounded) >> 16) as u16)
    }

    /// Convert from bf16 to f32.
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    pub fn from_f64(v: f64) -> Self {
        Self::from_f32(v as f32)
    }
}

impl std::fmt::Debug for bf16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bf16({})", self.to_f32())
    }
}

/// Typed tensor storage backing a `Tensor`.
#[derive(Clone)]
pub enum Storage {
    F64(Arc<RwLock<Vec<f64>>>),
    F32(Arc<RwLock<Vec<f32>>>),
    BF16(Arc<RwLock<Vec<bf16>>>),
    I8(Arc<RwLock<Vec<i8>>>),
}

impl Storage {
    /// Create a new F64 storage.
    pub fn f64(data: Vec<f64>) -> Self {
        Storage::F64(Arc::new(RwLock::new(data)))
    }

    /// Create a new F32 storage.
    pub fn f32(data: Vec<f32>) -> Self {
        Storage::F32(Arc::new(RwLock::new(data)))
    }

    /// Create a new BF16 storage.
    pub fn bf16(data: Vec<bf16>) -> Self {
        Storage::BF16(Arc::new(RwLock::new(data)))
    }

    /// Create a new I8 storage.
    pub fn i8(data: Vec<i8>) -> Self {
        Storage::I8(Arc::new(RwLock::new(data)))
    }

    /// Return the dtype of this storage.
    pub fn dtype(&self) -> Dtype {
        match self {
            Storage::F64(_) => Dtype::F64,
            Storage::F32(_) => Dtype::F32,
            Storage::BF16(_) => Dtype::BF16,
            Storage::I8(_) => Dtype::I8,
        }
    }

    /// Return the number of elements.
    pub fn len(&self) -> usize {
        match self {
            Storage::F64(v) => v.read().unwrap().len(),
            Storage::F32(v) => v.read().unwrap().len(),
            Storage::BF16(v) => v.read().unwrap().len(),
            Storage::I8(v) => v.read().unwrap().len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if two storages point to the same underlying Arc (same buffer).
    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Storage::F64(a), Storage::F64(b)) => Arc::ptr_eq(a, b),
            (Storage::F32(a), Storage::F32(b)) => Arc::ptr_eq(a, b),
            (Storage::BF16(a), Storage::BF16(b)) => Arc::ptr_eq(a, b),
            (Storage::I8(a), Storage::I8(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Convert storage data to Vec<f64>, regardless of original dtype.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Storage::F64(v) => v.read().unwrap().clone(),
            Storage::F32(v) => v.read().unwrap().iter().map(|&x| x as f64).collect(),
            Storage::BF16(v) => v.read().unwrap().iter().map(|&x| x.to_f64()).collect(),
            Storage::I8(v) => v.read().unwrap().iter().map(|&x| x as f64).collect(),
        }
    }

    /// Create storage from Vec<f64> with the specified dtype.
    pub fn from_f64_vec(data: Vec<f64>, dtype: Dtype) -> Self {
        match dtype {
            Dtype::F64 => Storage::F64(Arc::new(RwLock::new(data))),
            Dtype::F32 => Storage::F32(Arc::new(RwLock::new(
                data.into_iter().map(|v| v as f32).collect(),
            ))),
            Dtype::BF16 => Storage::BF16(Arc::new(RwLock::new(
                data.into_iter().map(bf16::from_f64).collect(),
            ))),
            Dtype::I8 => Storage::I8(Arc::new(RwLock::new(
                data.into_iter().map(|v| v as i8).collect(),
            ))),
        }
    }

    /// Create storage from Vec<f32> with the specified dtype.
    pub fn from_f32_vec(data: Vec<f32>, dtype: Dtype) -> Self {
        match dtype {
            Dtype::F64 => Storage::F64(Arc::new(RwLock::new(
                data.into_iter().map(|v| v as f64).collect(),
            ))),
            Dtype::F32 => Storage::F32(Arc::new(RwLock::new(data))),
            Dtype::BF16 => Storage::BF16(Arc::new(RwLock::new(
                data.into_iter().map(bf16::from_f32).collect(),
            ))),
            Dtype::I8 => Storage::I8(Arc::new(RwLock::new(
                data.into_iter().map(|v| v as i8).collect(),
            ))),
        }
    }

    /// Create zero-initialized storage with the specified dtype.
    pub fn zeros(len: usize, dtype: Dtype) -> Self {
        match dtype {
            Dtype::F64 => Storage::F64(Arc::new(RwLock::new(vec![0.0; len]))),
            Dtype::F32 => Storage::F32(Arc::new(RwLock::new(vec![0.0_f32; len]))),
            Dtype::BF16 => Storage::BF16(Arc::new(RwLock::new(vec![bf16(0); len]))),
            Dtype::I8 => Storage::I8(Arc::new(RwLock::new(vec![0_i8; len]))),
        }
    }

    /// Convert storage data to Vec<f32>, regardless of original dtype.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Storage::F64(v) => v.read().unwrap().iter().map(|&x| x as f32).collect(),
            Storage::F32(v) => v.read().unwrap().iter().copied().collect(),
            Storage::BF16(v) => v.read().unwrap().iter().map(|&x| x.to_f32()).collect(),
            Storage::I8(v) => v.read().unwrap().iter().map(|&x| x as f32).collect(),
        }
    }

    /// Convert storage data to Vec<bf16>, regardless of original dtype.
    pub fn to_bf16_vec(&self) -> Vec<bf16> {
        match self {
            Storage::F64(v) => v
                .read()
                .unwrap()
                .iter()
                .map(|&x| bf16::from_f64(x))
                .collect(),
            Storage::F32(v) => v
                .read()
                .unwrap()
                .iter()
                .map(|&x| bf16::from_f32(x))
                .collect(),
            Storage::BF16(v) => v.read().unwrap().clone(),
            Storage::I8(v) => v
                .read()
                .unwrap()
                .iter()
                .map(|&x| bf16::from_f32(x as f32))
                .collect(),
        }
    }

    /// Accumulate an f32 slice into an F32 storage (panics if not F32).
    pub fn accumulate_f32(&self, slice: &[f32]) {
        match self {
            Storage::F32(v) => {
                let mut guard = v.write().unwrap();
                assert_eq!(guard.len(), slice.len());
                for (dst, &src) in guard.iter_mut().zip(slice.iter()) {
                    *dst += src;
                }
            }
            _ => panic!(
                "accumulate_f32 called on non-F32 storage: {:?}",
                self.dtype()
            ),
        }
    }

    /// Accumulate an f64 slice into an F64 storage (panics if not F64).
    pub fn accumulate_f64(&self, slice: &[f64]) {
        match self {
            Storage::F64(v) => {
                let mut guard = v.write().unwrap();
                assert_eq!(guard.len(), slice.len());
                for (dst, &src) in guard.iter_mut().zip(slice.iter()) {
                    *dst += src;
                }
            }
            _ => panic!(
                "accumulate_f64 called on non-F64 storage: {:?}",
                self.dtype()
            ),
        }
    }

    /// Accumulate an f64 slice, converting to native dtype as needed.
    /// Supports F64 (direct) and F32 (cast). Panics for other dtypes.
    pub fn accumulate_f64_slice(&self, slice: &[f64]) {
        match self {
            Storage::F64(v) => {
                let mut guard = v.write().unwrap();
                assert_eq!(guard.len(), slice.len());
                for (dst, &src) in guard.iter_mut().zip(slice.iter()) {
                    *dst += src;
                }
            }
            Storage::F32(v) => {
                let mut guard = v.write().unwrap();
                assert_eq!(guard.len(), slice.len());
                for (dst, &src) in guard.iter_mut().zip(slice.iter()) {
                    *dst += src as f32;
                }
            }
            _ => panic!("accumulate_f64_slice not supported for {:?}", self.dtype()),
        }
    }

    /// Fill storage with a value (converted to native dtype).
    pub fn fill_f64(&self, value: f64) {
        match self {
            Storage::F64(v) => {
                let mut guard = v.write().unwrap();
                guard.fill(value);
            }
            Storage::F32(v) => {
                let mut guard = v.write().unwrap();
                let v32 = value as f32;
                guard.fill(v32);
            }
            Storage::BF16(v) => {
                let mut guard = v.write().unwrap();
                let vbf = bf16::from_f64(value);
                guard.fill(vbf);
            }
            Storage::I8(v) => {
                let mut guard = v.write().unwrap();
                let vi8 = value as i8;
                guard.fill(vi8);
            }
        }
    }

    /// Zero all elements.
    pub fn zero(&self) {
        match self {
            Storage::F64(v) => v.write().unwrap().fill(0.0),
            Storage::F32(v) => v.write().unwrap().fill(0.0_f32),
            Storage::BF16(v) => v.write().unwrap().fill(bf16(0)),
            Storage::I8(v) => v.write().unwrap().fill(0_i8),
        }
    }

    /// Return true if this is F64 storage.
    pub fn is_f64(&self) -> bool {
        matches!(self, Storage::F64(_))
    }

    /// Return true if this is F32 storage.
    pub fn is_f32(&self) -> bool {
        matches!(self, Storage::F32(_))
    }

    /// Return a unique identifier for this storage (based on Arc pointer).
    pub fn id(&self) -> usize {
        match self {
            Storage::F64(v) => Arc::as_ptr(v) as usize,
            Storage::F32(v) => Arc::as_ptr(v) as usize,
            Storage::BF16(v) => Arc::as_ptr(v) as usize,
            Storage::I8(v) => Arc::as_ptr(v) as usize,
        }
    }
}

/// Trait for types that can be tensor elements.
#[allow(dead_code)]
pub trait TensorElement: Copy + Default + 'static + Send + Sync {
    const DTYPE: Dtype;
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
}

impl TensorElement for f64 {
    const DTYPE: Dtype = Dtype::F64;
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

impl TensorElement for f32 {
    const DTYPE: Dtype = Dtype::F32;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl TensorElement for bf16 {
    const DTYPE: Dtype = Dtype::BF16;
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
    fn from_f64(v: f64) -> Self {
        bf16::from_f64(v)
    }
}

impl TensorElement for i8 {
    const DTYPE: Dtype = Dtype::I8;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as i8
    }
}
