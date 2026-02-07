use pyo3::exceptions;
use pyo3::prelude::*;

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn broadcast_shapes(
    shape1: &[usize],
    shape2: &[usize],
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let rank = std::cmp::max(shape1.len(), shape2.len());
    let mut shape = Vec::with_capacity(rank);
    let mut s_strides = vec![0; rank];
    let mut o_strides = vec![0; rank];
    let mut s_shape = vec![1; rank];
    let mut o_shape = vec![1; rank];

    s_shape[(rank - shape1.len())..].copy_from_slice(shape1);
    o_shape[(rank - shape2.len())..].copy_from_slice(shape2);

    let s_strides_base = compute_strides(&s_shape);
    let o_strides_base = compute_strides(&o_shape);

    for i in 0..rank {
        match (s_shape[i], o_shape[i]) {
            (a, b) if a == b => {
                shape.push(a);
                s_strides[i] = s_strides_base[i];
                o_strides[i] = o_strides_base[i];
            }
            (1, b) => {
                shape.push(b);
                s_strides[i] = 0;
                o_strides[i] = o_strides_base[i];
            }
            (a, 1) => {
                shape.push(a);
                s_strides[i] = s_strides_base[i];
                o_strides[i] = 0;
            }
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Shapes are not broadcastable",
                ));
            }
        }
    }
    Ok((shape, s_strides, o_strides))
}

pub fn unravel_index(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        indices[i] = idx % shape[i];
        idx /= shape[i];
    }
    indices
}

pub fn ravel_index(indices: &[usize], strides: &[usize], shape: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides.iter())
        .zip(shape.iter())
        .map(|((&i, &s), &dim)| if dim == 1 { 0 } else { i * s })
        .sum()
}
