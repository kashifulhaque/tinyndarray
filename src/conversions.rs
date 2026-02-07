use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::array::NdArray;
use crate::utils::compute_strides;

pub fn parse_nested_list(py_obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(flat) = py_obj.extract::<Vec<f32>>() {
        // 1D array
        let shape = vec![flat.len()];
        let strides = compute_strides(&shape);
        Ok(NdArray {
            data: flat,
            shape,
            strides,
        })
    } else if let Ok(nested) = py_obj.extract::<Vec<Bound<'_, PyAny>>>() {
        // Multi-dimensional array
        let mut data = vec![];
        let mut shape = vec![];

        for sublist in &nested {
            let subarray = parse_nested_list(sublist)?;
            if shape.is_empty() {
                shape = vec![nested.len()];
                shape.extend(subarray.shape);
            } else if subarray.shape != shape[1..] {
                return Err(PyErr::new::<exceptions::PyValueError, _>(
                    "Inconsistent sub-array shapes",
                ));
            }
            data.extend(subarray.data);
        }

        let strides = compute_strides(&shape);
        Ok(NdArray {
            data,
            shape,
            strides,
        })
    } else {
        Err(PyErr::new::<exceptions::PyValueError, _>(
            "Input must be a list or nested list of floats",
        ))
    }
}

pub fn to_nested_list<'py>(
    py: Python<'py>,
    data: &[f32],
    shape: &[usize],
    strides: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    if shape.is_empty() {
        return Ok(data[0].into_py(py).into_bound(py));
    }

    let dim = shape[0];
    let rest_shape = &shape[1..];
    let rest_strides = &strides[1..];
    let stride = strides[0];
    let mut list = Vec::with_capacity(dim);

    for i in 0..dim {
        let offset = i * stride;
        let sub_data = &data[offset..];
        list.push(to_nested_list(py, sub_data, rest_shape, rest_strides)?);
    }

    Ok(PyList::new_bound(py, list).into_any())
}

pub fn from_numpy(py_array: PyReadonlyArrayDyn<f32>) -> PyResult<NdArray> {
    let array = py_array.as_array().to_owned();
    let shape = array.shape().to_vec();
    let strides = compute_strides(&shape);
    let data = array.iter().cloned().collect();

    Ok(NdArray {
        data,
        shape,
        strides,
    })
}

pub fn to_numpy<'py>(array: &NdArray, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let ndarray = ndarray::Array::from_shape_vec(array.shape.clone(), array.data.clone())
        .map_err(|_| PyErr::new::<exceptions::PyValueError, _>("Invalid shape for array"))?;
    Ok(PyArrayDyn::from_owned_array_bound(py, ndarray))
}
