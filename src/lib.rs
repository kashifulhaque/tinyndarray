use pyo3::prelude::*;
use pyo3::exceptions;
use pyo3::types::PyType;

#[pyclass]
pub struct NdArray {
    data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>
}

#[pymethods]
impl NdArray {
    #[new]
    fn new(shape: Vec<usize>) -> PyResult<Self> {
        if shape.is_empty() {
            return Err(exceptions::PyValueError::new_err("Shape cannot be empty"));
        }

        let size = shape.iter().product();
        let data = vec![0.0; size];
        let strides = compute_strides(&shape);

        Ok(NdArray { data, shape, strides })
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn get(&self, indices: Vec<usize>) -> PyResult<f64> {
        if indices.len() != self.shape.len() {
            return Err(exceptions::PyIndexError::new_err("Incorrect number of indices"));
        }

        let flat_idx: usize = self
            .strides
            .iter()
            .zip(indices.iter())
            .map(|(s, i)| s * i)
            .sum();

        self.data.get(flat_idx).copied().ok_or_else(|| {
            exceptions::PyIndexError::new_err("Index out of bounds")
        })
    }

    fn set(&mut self, indices: Vec<usize>, value: f64) -> PyResult<()> {
        if indices.len() != self.shape.len() {
            return Err(exceptions::PyIndexError::new_err("Incorrect number of indices"))
        }

        let flat_idx: usize = self
            .strides
            .iter()
            .zip(indices.iter())
            .map(|(s, i)| s * i)
            .sum();

        if let Some(elem) = self.data.get_mut(flat_idx) {
            *elem = value;
            Ok(())
        } else {
            Err(exceptions::PyIndexError::new_err("Index out of bounds"))
        }
    }

    #[classmethod]
    fn ones(_cls: &PyType, shape: Vec<usize>) -> PyResult<Self> {
        if shape.is_empty() {
            return Err(exceptions::PyValueError::new_err("Shape cannot be empty"));
        }

        let size = shape.iter().product();
        let data = vec![1.0; size];
        let strides = compute_strides(&shape);

        Ok(NdArray { data, shape, strides })
    }

    #[classmethod]
    fn zeros(_cls: &PyType, shape: Vec<usize>) -> PyResult<Self> {
        if shape.is_empty() {
            return Err(exceptions::PyValueError::new_err("Shape cannot be empty"));
        }

        let size = shape.iter().product();
        let data = vec![0.0; size];
        let strides = compute_strides(&shape);

        Ok(NdArray { data, shape, strides })
    }

    fn reshape(&mut self, new_shape: Vec<usize>) -> PyResult<()> {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();

        if old_size != new_size {
            return Err(exceptions::PyValueError::new_err("Total size must remain unchanged when reshaping"));
        }

        self.shape = new_shape;
        self.strides = compute_strides(&self.shape);
        Ok(())
    }

    fn add(&self, other: &NdArray) -> PyResult<NdArray> {
        let (broadcast_shape, self_strides, other_strides) = broadcast_shapes(&self.shape, &other.shape)?;

        let total_size: usize = broadcast_shape.iter().product();
        let mut result_data = vec![0.0; total_size];

        for idx in 0..total_size {
            let idx_multi = unravel_index(idx, &broadcast_shape);
            let self_idx = ravel_index(&idx_multi, &self_strides, &self.shape);
            let other_idx = ravel_index(&idx_multi, &other_strides, &other.shape);

            result_data[idx] = self.data[self_idx] + other.data[other_idx];
        }

        Ok(NdArray {
            data: result_data,
            shape: broadcast_shape.clone(),
            strides: compute_strides(&broadcast_shape),
        })
    }

    fn __getitem__(&self, indices: Bound<'_, PyAny>) -> PyResult<f64> {
        let indices: Vec<usize> = if let Ok(i) = indices.extract::<usize>() {
            vec![i]
        } else if let Ok(t) = indices.extract::<Vec<usize>>() {
            t
        } else if let Ok(t) = indices.extract::<(usize, usize)>() {
            vec![t.0, t.1]
        } else {
            return Err(exceptions::PyTypeError::new_err("Invalid index format"))
        };

        self.get(indices)
    }

    fn __setitem__(&mut self, indices: Bound<'_, PyAny>, value: f64) -> PyResult<()> {
        let indices: Vec<usize> = if let Ok(i) = indices.extract::<usize>() {
            vec![i]
        } else if let Ok(t) = indices.extract::<Vec<usize>>() {
            t
        } else if let Ok(t) = indices.extract::<(usize, usize)>() {
            vec![t.0, t.1]
        } else {
            return Err(exceptions::PyTypeError::new_err("Invalid index format"))
        };

        self.set(indices, value)
    }

    fn __repr__(&self) -> String {
        format!(
            "NdArray(shape={:?}, ndim={}, data={:?})",
            self.shape,
            self.shape.len(),
            self.data
        )
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> PyResult<(Vec<usize>, Vec<usize>, Vec<usize>)> {
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
                return Err(exceptions::PyValueError::new_err("Shapes are not broadcastable"));
            }
        }
    }

    Ok((shape, s_strides, o_strides))
}

fn unravel_index(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        indices[i] = idx % shape[i];
        idx /= shape[i];
    }
    indices
}

fn ravel_index(indices: &[usize], strides: &[usize], shape: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides.iter())
        .zip(shape.iter())
        .map(|((&i, &s), &dim)| {
            if dim == 1 {
                0
            } else {
                i * s
            }
        })
        .sum()
}

#[pymodule]
fn tinyndarray(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NdArray>()?;
    Ok(())
}
