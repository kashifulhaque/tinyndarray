mod array;
mod conversions;
mod operations;
mod utils;

use pyo3::prelude::*;

pub use array::NdArray;

#[pymodule]
fn ferray(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NdArray>()?;
    Ok(())
}
