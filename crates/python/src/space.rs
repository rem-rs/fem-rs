use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use fem_mesh::SimplexMesh;
use fem_space::{FESpace, H1Space};
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
    /// Direct construction is disabled; use H1Space(mesh, order).
    #[new]
    pub fn new(mesh: &PyMesh, order: u8) -> PyResult<Self> {
        if order == 0 {
            return Err(PyValueError::new_err("H1Space: order must be \u{2265} 1"));
        }
        match mesh.dim {
            2 => {
                let m = mesh.inner_2d.as_ref().ok_or_else(|| {
                    PyValueError::new_err("H1Space: mesh is not a 2-D mesh")
                })?.clone();
                Ok(PyH1Space {
                    inner_2d: Some(H1Space::new(m, order)),
                    inner_3d: None,
                    dim: 2,
                })
            }
            3 => {
                let m = mesh.inner_3d.as_ref().ok_or_else(|| {
                    PyValueError::new_err("H1Space: mesh is not a 3-D mesh")
                })?.clone();
                Ok(PyH1Space {
                    inner_2d: None,
                    inner_3d: Some(H1Space::new(m, order)),
                    dim: 3,
                })
            }
            _ => Err(PyValueError::new_err("H1Space: unsupported mesh dimension")),
        }
    }

    /// Number of degrees of freedom.
    pub fn n_dofs(&self) -> PyResult<usize> {
        match self.dim {
            2 => Ok(self.inner_2d.as_ref().unwrap().n_dofs()),
            3 => Ok(self.inner_3d.as_ref().unwrap().n_dofs()),
            _ => Err(PyValueError::new_err("invalid space state")),
        }
    }
}
