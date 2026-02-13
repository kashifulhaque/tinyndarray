use cblas::{sgemm, Layout, Transpose};
use pyo3::exceptions;
use pyo3::prelude::*;

use crate::array::NdArray;
use crate::utils::{broadcast_shapes, compute_strides, ravel_index, unravel_index};

pub fn mul_array(left: &NdArray, right: &NdArray) -> PyResult<NdArray> {
    let (broadcast_shape, self_strides, other_strides) =
        broadcast_shapes(&left.shape, &right.shape)?;
    let total_size: usize = broadcast_shape.iter().product();
    let mut result_data = vec![0.0; total_size];

    for (idx, result) in result_data.iter_mut().enumerate() {
        let idx_multi = unravel_index(idx, &broadcast_shape);
        let self_idx = ravel_index(&idx_multi, &self_strides, &left.shape);
        let other_idx = ravel_index(&idx_multi, &other_strides, &right.shape);
        *result = left.data[self_idx] * right.data[other_idx];
    }

    Ok(NdArray {
        data: result_data,
        shape: broadcast_shape.clone(),
        strides: compute_strides(&broadcast_shape),
    })
}

pub fn mul_scalar(array: &NdArray, scalar: f32) -> PyResult<NdArray> {
    let result_data: Vec<f32> = array.data.iter().map(|&x| x * scalar).collect();
    Ok(NdArray {
        data: result_data,
        shape: array.shape.clone(),
        strides: array.strides.clone(),
    })
}

pub fn add_array(left: &NdArray, right: &NdArray) -> PyResult<NdArray> {
    let (broadcast_shape, self_strides, other_strides) =
        broadcast_shapes(&left.shape, &right.shape)?;
    let total_size: usize = broadcast_shape.iter().product();
    let mut result_data = vec![0.0; total_size];

    for (idx, result) in result_data.iter_mut().enumerate() {
        let idx_multi = unravel_index(idx, &broadcast_shape);
        let self_idx = ravel_index(&idx_multi, &self_strides, &left.shape);
        let other_idx = ravel_index(&idx_multi, &other_strides, &right.shape);
        *result = left.data[self_idx] + right.data[other_idx];
    }

    Ok(NdArray {
        data: result_data,
        shape: broadcast_shape.clone(),
        strides: compute_strides(&broadcast_shape),
    })
}

pub fn add_scalar(array: &NdArray, scalar: f32) -> PyResult<NdArray> {
    let result_data: Vec<f32> = array.data.iter().map(|&x| x + scalar).collect();
    Ok(NdArray {
        data: result_data,
        shape: array.shape.clone(),
        strides: array.strides.clone(),
    })
}

pub fn sub_array(left: &NdArray, right: &NdArray) -> PyResult<NdArray> {
    let (broadcast_shape, self_strides, other_strides) =
        broadcast_shapes(&left.shape, &right.shape)?;
    let total_size: usize = broadcast_shape.iter().product();
    let mut result_data = vec![0.0; total_size];

    for (idx, result) in result_data.iter_mut().enumerate() {
        let idx_multi = unravel_index(idx, &broadcast_shape);
        let self_idx = ravel_index(&idx_multi, &self_strides, &left.shape);
        let other_idx = ravel_index(&idx_multi, &other_strides, &right.shape);
        *result = left.data[self_idx] - right.data[other_idx];
    }

    Ok(NdArray {
        data: result_data,
        shape: broadcast_shape.clone(),
        strides: compute_strides(&broadcast_shape),
    })
}

pub fn sub_scalar(array: &NdArray, scalar: f32) -> PyResult<NdArray> {
    let result_data: Vec<f32> = array.data.iter().map(|&x| x - scalar).collect();
    Ok(NdArray {
        data: result_data,
        shape: array.shape.clone(),
        strides: array.strides.clone(),
    })
}

pub fn div_array(left: &NdArray, right: &NdArray) -> PyResult<NdArray> {
    let (broadcast_shape, self_strides, other_strides) =
        broadcast_shapes(&left.shape, &right.shape)?;
    let total_size: usize = broadcast_shape.iter().product();
    let mut result_data = vec![0.0; total_size];

    for (idx, result) in result_data.iter_mut().enumerate() {
        let idx_multi = unravel_index(idx, &broadcast_shape);
        let self_idx = ravel_index(&idx_multi, &self_strides, &left.shape);
        let other_idx = ravel_index(&idx_multi, &other_strides, &right.shape);
        *result = left.data[self_idx] / right.data[other_idx];
    }

    Ok(NdArray {
        data: result_data,
        shape: broadcast_shape.clone(),
        strides: compute_strides(&broadcast_shape),
    })
}

pub fn div_scalar(array: &NdArray, scalar: f32) -> PyResult<NdArray> {
    let result_data: Vec<f32> = array.data.iter().map(|&x| x / scalar).collect();
    Ok(NdArray {
        data: result_data,
        shape: array.shape.clone(),
        strides: array.strides.clone(),
    })
}

pub fn matmul(left: &NdArray, right: &NdArray) -> PyResult<NdArray> {
    let (m, k1) = (left.shape[0], left.shape[1]);
    let (k2, n) = (right.shape[0], right.shape[1]);

    if k1 != k2 {
        return Err(PyErr::new::<exceptions::PyValueError, _>(
            "Shape mismatch for matmul.",
        ));
    }

    let mut result_data = vec![0.0f32; m * n];
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            m as i32,
            n as i32,
            k1 as i32,
            1.0,
            &left.data,
            k1 as i32,
            &right.data,
            n as i32,
            0.0,
            &mut result_data,
            n as i32,
        );
    }

    Ok(NdArray {
        data: result_data,
        shape: vec![m, n],
        strides: compute_strides(&[m, n]),
    })
}
