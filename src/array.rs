use pyo3::prelude::*;
use pyo3::exceptions;
use pyo3::types::{PyType, PyAny};
use pyo3::Bound;

use crate::operations;
use crate::utils::{compute_strides};
use crate::conversions::{parse_nested_list, to_nested_list};

#[pyclass]
#[derive(Clone)]
pub struct NdArray {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl NdArray {
    pub fn new_with_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        NdArray { data, shape, strides }
    }
}

#[pymethods]
impl NdArray {
    #[classmethod]
    fn from_list(_cls: &Bound<'_, PyType>, py_list: &Bound<'_, PyAny>) -> PyResult<Self> {
        parse_nested_list(py_list)
    }

    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
    fn ones(_cls: &Bound<'_, PyType>, shape: Vec<usize>) -> PyResult<Self> {
        if shape.is_empty() {
            return Err(exceptions::PyValueError::new_err("Shape cannot be empty"));
        }

        let size = shape.iter().product();
        let data = vec![1.0; size];
        let strides = compute_strides(&shape);
        Ok(NdArray { data, shape, strides })
    }

    #[classmethod]
    fn zeros(_cls: &Bound<'_, PyType>, shape: Vec<usize>) -> PyResult<Self> {
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
            return Err(exceptions::PyValueError::new_err(
                "Total size must remain unchanged when reshaping"
            ));
        }

        self.shape = new_shape;
        self.strides = compute_strides(&self.shape);
        Ok(())
    }

    fn add(&self, other: &NdArray) -> PyResult<NdArray> {
        operations::add_array(self, other)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        if let Ok(array) = other.extract::<NdArray>() {
            operations::mul_array(self, &array)
        } else if let Ok(scalar) = other.extract::<f32>() {
            operations::mul_scalar(self, scalar)
        } else {
            Err(exceptions::PyTypeError::new_err("Unsupported operand for *"))
        }
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        if let Ok(array) = other.extract::<NdArray>() {
            operations::add_array(self, &array)
        } else if let Ok(scalar) = other.extract::<f32>() {
            operations::add_scalar(self, scalar)
        } else {
            Err(exceptions::PyTypeError::new_err("Unsupported operand for +"))
        }
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        if let Ok(array) = other.extract::<NdArray>() {
            operations::sub_array(self, &array)
        } else if let Ok(scalar) = other.extract::<f32>() {
            operations::sub_scalar(self, scalar)
        } else {
            Err(exceptions::PyTypeError::new_err("Unsupported operand for -"))
        }
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        if let Ok(array) = other.extract::<NdArray>() {
            operations::div_array(self, &array)
        } else if let Ok(scalar) = other.extract::<f32>() {
            operations::div_scalar(self, scalar)
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

    fn __matmul__(&self, other: &NdArray) -> PyResult<NdArray> {
        operations::matmul(self, other)
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
    fn from_numpy(py_array: numpy::PyReadonlyArrayDyn<f32>) -> PyResult<Self> {
        use crate::conversions::from_numpy;
        from_numpy(py_array)
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
        use crate::conversions::to_numpy;
        to_numpy(self, py)
    }
}
