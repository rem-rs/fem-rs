use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyList;
use numpy::{PyArray1, PyArrayMethods};
use fem_mesh::SimplexMesh;
use fem_space::{FESpace, H1Space, L2Space, VectorH1Space, HCurlSpace, ComplexGridFunction, ComplexSpace};
use crate::mesh::PyMesh;

/// H¹ finite element space (continuous Lagrange).
#[pyclass(name = "H1Space")]
pub struct PyH1Space {
    pub(crate) inner_2d: Option<H1Space<SimplexMesh<2>>>,
    pub(crate) inner_3d: Option<H1Space<SimplexMesh<3>>>,
    pub(crate) dim: u8,
}

#[pymethods]
impl PyH1Space {
    #[new]
    pub fn new(mesh: &PyMesh, order: u8) -> PyResult<Self> {
        if order == 0 {
            return Err(PyValueError::new_err("H1Space: order must be >= 1"));
        }
        match mesh.dim {
            2 => {
                let m = mesh.inner_2d.as_ref().ok_or_else(||
                    PyValueError::new_err("H1Space: mesh is not a 2-D mesh")
                )?.clone();
                Ok(PyH1Space { inner_2d: Some(H1Space::new(m, order)), inner_3d: None, dim: 2 })
            }
            3 => {
                let m = mesh.inner_3d.as_ref().ok_or_else(||
                    PyValueError::new_err("H1Space: mesh is not a 3-D mesh")
                )?.clone();
                Ok(PyH1Space { inner_2d: None, inner_3d: Some(H1Space::new(m, order)), dim: 3 })
            }
            _ => Err(PyValueError::new_err("H1Space: unsupported mesh dimension")),
        }
    }

    pub fn n_dofs(&self) -> PyResult<usize> {
        match self.dim {
            2 => Ok(self.inner_2d.as_ref().unwrap().n_dofs()),
            3 => Ok(self.inner_3d.as_ref().unwrap().n_dofs()),
            _ => Err(PyValueError::new_err("invalid space state")),
        }
    }

    pub fn dim(&self) -> u8 { self.dim }
}

/// L² (discontinuous) finite element space.
#[pyclass(name = "L2Space")]
pub struct PyL2Space {
    pub(crate) inner_2d: Option<L2Space<SimplexMesh<2>>>,
    pub(crate) inner_3d: Option<L2Space<SimplexMesh<3>>>,
    pub(crate) dim: u8,
}

#[pymethods]
impl PyL2Space {
    #[new]
    pub fn new(mesh: &PyMesh, order: u8) -> PyResult<Self> {
        match mesh.dim {
            2 => {
                let m = mesh.inner_2d.as_ref().ok_or_else(||
                    PyValueError::new_err("L2Space: mesh is not 2-D")
                )?.clone();
                Ok(PyL2Space { inner_2d: Some(L2Space::new(m, order)), inner_3d: None, dim: 2 })
            }
            3 => {
                let m = mesh.inner_3d.as_ref().ok_or_else(||
                    PyValueError::new_err("L2Space: mesh is not 3-D")
                )?.clone();
                Ok(PyL2Space { inner_2d: None, inner_3d: Some(L2Space::new(m, order)), dim: 3 })
            }
            _ => Err(PyValueError::new_err("L2Space: unsupported mesh dimension")),
        }
    }

    pub fn n_dofs(&self) -> PyResult<usize> {
        match self.dim {
            2 => Ok(self.inner_2d.as_ref().unwrap().n_dofs()),
            3 => Ok(self.inner_3d.as_ref().unwrap().n_dofs()),
            _ => Err(PyValueError::new_err("invalid space state")),
        }
    }

    pub fn dim(&self) -> u8 { self.dim }
}

/// Vector H¹ space [H¹]ᵈ for elasticity and Stokes.
#[pyclass(name = "VectorH1Space")]
pub struct PyVectorH1Space {
    pub(crate) inner_2d: Option<VectorH1Space<SimplexMesh<2>>>,
    pub(crate) inner_3d: Option<VectorH1Space<SimplexMesh<3>>>,
    pub(crate) dim: u8,
}

#[pymethods]
impl PyVectorH1Space {
    #[new]
    pub fn new(mesh: &PyMesh, order: u8) -> PyResult<Self> {
        match mesh.dim {
            2 => {
                let m = mesh.inner_2d.as_ref().ok_or_else(||
                    PyValueError::new_err("VectorH1Space: mesh is not 2-D")
                )?.clone();
                Ok(PyVectorH1Space { inner_2d: Some(VectorH1Space::new(m, order, 2)), inner_3d: None, dim: 2 })
            }
            3 => {
                let m = mesh.inner_3d.as_ref().ok_or_else(||
                    PyValueError::new_err("VectorH1Space: mesh is not 3-D")
                )?.clone();
                Ok(PyVectorH1Space { inner_2d: None, inner_3d: Some(VectorH1Space::new(m, order, 3)), dim: 3 })
            }
            _ => Err(PyValueError::new_err("VectorH1Space: unsupported mesh dimension")),
        }
    }

    pub fn n_dofs(&self) -> PyResult<usize> {
        match self.dim {
            2 => Ok(self.inner_2d.as_ref().unwrap().n_dofs()),
            3 => Ok(self.inner_3d.as_ref().unwrap().n_dofs()),
            _ => Err(PyValueError::new_err("invalid space state")),
        }
    }

    pub fn dim(&self) -> u8 { self.dim }
}

/// H(curl) finite element space (Nedelec edge elements).
#[pyclass(name = "HCurlSpace")]
pub struct PyHCurlSpace {
    pub(crate) inner_2d: Option<HCurlSpace<SimplexMesh<2>>>,
    pub(crate) inner_3d: Option<HCurlSpace<SimplexMesh<3>>>,
    pub(crate) dim: u8,
}

#[pymethods]
impl PyHCurlSpace {
    #[new]
    pub fn new(mesh: &PyMesh, order: u8) -> PyResult<Self> {
        match mesh.dim {
            2 => {
                let m = mesh.inner_2d.as_ref().ok_or_else(||
                    PyValueError::new_err("HCurlSpace: mesh is not 2-D")
                )?.clone();
                Ok(PyHCurlSpace { inner_2d: Some(HCurlSpace::new(m, order)), inner_3d: None, dim: 2 })
            }
            3 => {
                let m = mesh.inner_3d.as_ref().ok_or_else(||
                    PyValueError::new_err("HCurlSpace: mesh is not 3-D")
                )?.clone();
                Ok(PyHCurlSpace { inner_2d: None, inner_3d: Some(HCurlSpace::new(m, order)), dim: 3 })
            }
            _ => Err(PyValueError::new_err("HCurlSpace: unsupported mesh dimension")),
        }
    }

    pub fn n_dofs(&self) -> PyResult<usize> {
        match self.dim {
            2 => Ok(self.inner_2d.as_ref().unwrap().n_dofs()),
            3 => Ok(self.inner_3d.as_ref().unwrap().n_dofs()),
            _ => Err(PyValueError::new_err("invalid space state")),
        }
    }

    pub fn dim(&self) -> u8 { self.dim }
}

/// Complex-valued H¹ grid function (real + imaginary parts).
#[pyclass(name = "ComplexGridFunction")]
pub struct PyComplexGridFunction {
    pub u_re: Vec<f64>,
    pub u_im: Vec<f64>,
}

#[pymethods]
impl PyComplexGridFunction {
    #[new]
    pub fn new(space: &PyH1Space) -> PyResult<Self> {
        let n = space.n_dofs()?;
        Ok(PyComplexGridFunction { u_re: vec![0.0; n], u_im: vec![0.0; n] })
    }

    pub fn n_dofs(&self) -> usize { self.u_re.len() }

    pub fn amplitude<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let a: Vec<f64> = self.u_re.iter().zip(self.u_im.iter())
            .map(|(&r, &i)| (r*r + i*i).sqrt()).collect();
        PyArray1::from_vec(py, a)
    }
}
