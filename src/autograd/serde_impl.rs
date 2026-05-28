use crate::autograd::Tensor;
use crate::dtype::Dtype;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let data = self.data_as_f64_vec();
        let mut state = serializer.serialize_struct("Tensor", 3)?;
        state.serialize_field("data", &data)?;
        state.serialize_field("shape", &self.shape)?;
        state.serialize_field("dtype", &self.dtype)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TensorData {
            data: Vec<f64>,
            shape: Vec<usize>,
            #[serde(default)]
            dtype: Dtype,
        }

        let helper = TensorData::deserialize(deserializer)?;
        Ok(Tensor::with_dtype(helper.data, helper.shape, helper.dtype))
    }
}
