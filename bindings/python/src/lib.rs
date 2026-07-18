use pyo3::prelude::*;

#[pymodule(name = "_native")]
fn talos_xii_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    talos_xii::python_bridge::talos_xii(module)
}
