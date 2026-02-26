use pyo3::prelude::*;

#[pyfunction]
fn soma_rapida(a: i32, b: i32) -> i32 {
    a + b
}

#[pymodule]
fn template_rust_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soma_rapida, m)?)?;
    Ok(())
}
