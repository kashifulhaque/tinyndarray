mod array;
mod operations;
mod utils;
mod conversions;

use pyo3::prelude::*;

pub use array::NdArray;

#[pymodule]
fn tinyndarray(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NdArray>()?;
    Ok(())
}
