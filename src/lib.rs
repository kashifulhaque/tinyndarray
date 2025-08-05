use pyo3::prelude::*;
use pyo3::exceptions;
use pyo3::types::PyType;
use pyo3::types::PyList;
use cblas::{Layout, Transpose, sgemm};
use numpy::{PyReadonlyArrayDyn, PyArrayDyn};

#[pyclass]
#[derive(Clone)]
pub struct NdArray {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>
}

impl NdArray {
    fn mul_array(&self, other: NdArray) -> PyResult<NdArray> {
        let (broadcast_shape, self_strides, other_strides) =
            broadcast_shapes(&self.shape, &other.shape)?;

        let total_size: usize = broadcast_shape.iter().product();
        let mut result_data = vec![0.0; total_size];

        for idx in 0..total_size {
            let idx_multi = unravel_index(idx, &broadcast_shape);

            let self_idx = ravel_index(&idx_multi, &self_strides, &self.shape);
            let other_idx = ravel_index(&idx_multi, &other_strides, &other.shape);

            result_data[idx] = self.data[self_idx] * other.data[other_idx];
        }

        Ok(NdArray {
            data: result_data,
            shape: broadcast_shape.clone(),
            strides: compute_strides(&broadcast_shape),
        })
    }

    fn mul_scalar(&self, scalar: f32) -> PyResult<NdArray> {
        let result_data: Vec<f32> = self.data.iter().map(|&x| x * scalar).collect();
        Ok(NdArray {
            data: result_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    fn add_array(&self, other: NdArray) -> PyResult<NdArray> {
        let (broadcast_shape, self_strides, other_strides) =
            broadcast_shapes(&self.shape, &other.shape)?;

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

    fn sub_array(&self, other: NdArray) -> PyResult<NdArray> {
        let (broadcast_shape, self_strides, other_strides) =
            broadcast_shapes(&self.shape, &other.shape)?;

        let total_size: usize = broadcast_shape.iter().product();
        let mut result_data = vec![0.0; total_size];

        for idx in 0..total_size {
            let idx_multi = unravel_index(idx, &broadcast_shape);
            let self_idx = ravel_index(&idx_multi, &self_strides, &self.shape);
            let other_idx = ravel_index(&idx_multi, &other_strides, &other.shape);

            result_data[idx] = self.data[self_idx] - other.data[other_idx];
        }

        Ok(NdArray {
            data: result_data,
            shape: broadcast_shape.clone(),
            strides: compute_strides(&broadcast_shape),
        })
    }

    fn div_array(&self, other: NdArray) -> PyResult<NdArray> {
        let (broadcast_shape, self_strides, other_strides) =
            broadcast_shapes(&self.shape, &other.shape)?;

        let total_size: usize = broadcast_shape.iter().product();
        let mut result_data = vec![0.0; total_size];

        for idx in 0..total_size {
            let idx_multi = unravel_index(idx, &broadcast_shape);
            let self_idx = ravel_index(&idx_multi, &self_strides, &self.shape);
            let other_idx = ravel_index(&idx_multi, &other_strides, &other.shape);

            result_data[idx] = self.data[self_idx] / other.data[other_idx];
        }

        Ok(NdArray {
            data: result_data,
            shape: broadcast_shape.clone(),
            strides: compute_strides(&broadcast_shape),
        })
    }

    fn add_scalar(&self, scalar: f32) -> PyResult<NdArray> {
        let result_data: Vec<f32> = self.data.iter().map(|&x| x + scalar).collect();
        Ok(NdArray {
            data: result_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    fn sub_scalar(&self, scalar: f32) -> PyResult<NdArray> {
        let result_data: Vec<f32> = self.data.iter().map(|&x| x - scalar).collect();
        Ok(NdArray {
            data: result_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    fn div_scalar(&self, scalar: f32) -> PyResult<NdArray> {
        let result_data: Vec<f32> = self.data.iter().map(|&x| x / scalar).collect();
        Ok(NdArray {
            data: result_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    fn matmul(&self, other: &NdArray) -> PyResult<NdArray> {
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k1 != k2 {
            return Err(PyErr::new::<exceptions::PyValueError, _>("Shape mismatch for matmul."));
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
                &self.data,
                k1 as i32,
                &other.data,
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

}

#[pymethods]
impl NdArray {
    #[classmethod]
    fn from_list(_cls: &Bound<'_, PyType>, py_list: &PyAny) -> PyResult<Self> {
        parse_nested_list(py_list)
    }

    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        to_nested_list(py, &self.data, &self.shape, &self.strides)
    }

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

    fn get(&self, indices: Vec<usize>) -> PyResult<f32> {
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

    fn set(&mut self, indices: Vec<usize>, value: f32) -> PyResult<()> {
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

    fn __mul__(&self, other: &PyAny) -> PyResult<NdArray> {
        if let Ok(array) = other.extract::<NdArray>() {
            self.mul_array(array)
        } else if let Ok(scalar) = other.extract::<f32>() {
            self.mul_scalar(scalar)
        } else {
            Err(exceptions::PyTypeError::new_err("Unsupported operand for *"))
        }
    }

    fn __add__(&self, other: &PyAny) -> PyResult<NdArray> {
        if let Ok(array) = other.extract::<NdArray>() {
            self.add_array(array)
        } else if let Ok(scalar) = other.extract::<f32>() {
            self.add_scalar(scalar)
        } else {
            Err(exceptions::PyTypeError::new_err("Unsupported operand for +"))
        }
    }

    fn __sub__(&self, other: &PyAny) -> PyResult<NdArray> {
        if let Ok(array) = other.extract::<NdArray>() {
            self.sub_array(array)
        } else if let Ok(scalar) = other.extract::<f32>() {
            self.sub_scalar(scalar)
        } else {
            Err(exceptions::PyTypeError::new_err("Unsupported operand for -"))
        }
    }

    fn __truediv__(&self, other: &PyAny) -> PyResult<NdArray> {
        if let Ok(array) = other.extract::<NdArray>() {
            self.div_array(array)
        } else if let Ok(scalar) = other.extract::<f32>() {
            self.div_scalar(scalar)
        } else {
            Err(exceptions::PyTypeError::new_err("Unsupported operand for /"))
        }
    }

    fn transpose(&self) -> PyResult<NdArray> {
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.reverse();
        new_strides.reverse();

        Ok(NdArray {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        })
    }

    fn __matmul__(&self, other: &NdArray)-> PyResult<NdArray> {
        self.matmul(other)
    }

    fn __getitem__(&self, indices: Bound<'_, PyAny>) -> PyResult<f32> {
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

    fn __setitem__(&mut self, indices: Bound<'_, PyAny>, value: f32) -> PyResult<()> {
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

    #[staticmethod]
    fn from_numpy(py_array: PyReadonlyArrayDyn<f32>) -> PyResult<Self> {
        let array = py_array.as_array().to_owned();
        let shape = array.shape().to_vec();
        let strides = compute_strides(&shape);
        let data = array.iter().cloned().collect();

        Ok(NdArray {
            data,
            shape,
            strides
        })
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArrayDyn<f32>> {
        let array = ndarray::Array::from_shape_vec(self.shape.clone(), self.data.clone())
            .map_err(|_| PyErr::new::<exceptions::PyValueError, _>("Invalid shape for array"))?;
        Ok(PyArrayDyn::from_owned_array(py, array))
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

fn parse_nested_list(py_obj: &PyAny) -> PyResult<NdArray> {
    if let Ok(flat) = py_obj.extract::<Vec<f32>>() {    // 1D
        let shape = vec![flat.len()];
        let strides = compute_strides(&shape);
        Ok(NdArray {
            data: flat,
            shape,
            strides,
        })
    } else if let Ok(nested) = py_obj.extract::<Vec<&PyAny>>() {    // Multi-D
        let mut data = vec![];
        let mut shape = vec![];

        for sublist in &nested {
            let subarray = parse_nested_list(sublist)?;
            if shape.is_empty() {
                shape = vec![nested.len()];
                shape.extend(subarray.shape);
            } else if subarray.shape != shape[1..] {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
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
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Input must be a list or nested list of floats",
        ))
    }
}

fn to_nested_list<'py>(
    py: Python<'py>,
    data: &[f32],
    shape: &[usize],
    strides: &[usize],
) -> PyResult<&'py PyAny> {
    if shape.is_empty() {
        return Ok(data[0].into_py(py).into_ref(py));
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

    Ok(PyList::new(py, list).into())
}

#[pymodule]
fn tinyndarray(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NdArray>()?;
    Ok(())
}
