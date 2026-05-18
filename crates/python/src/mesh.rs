use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use fem_mesh::SimplexMesh;
use fem_mesh::boundary_nodes_with_tags;

/// Unstructured simplex mesh in 2-D or 3-D.
#[pyclass(name = "Mesh")]
pub struct PyMesh {
    pub(crate) inner_2d: Option<SimplexMesh<2>>,
    pub(crate) inner_3d: Option<SimplexMesh<3>>,
    pub(crate) dim: u8,
}

#[pymethods]
impl PyMesh {
    /// Create a 2-D unit-square mesh of `n×n` subdivisions (each quad → 2 triangles).
    #[staticmethod]
    pub fn unit_square_tri(n: usize) -> PyResult<Self> {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        Ok(PyMesh { inner_2d: Some(mesh), inner_3d: None, dim: 2 })
    }

    /// Create a 3-D unit-cube mesh of `n×n×n` tetrahedra.
    #[staticmethod]
    pub fn unit_cube_tet(n: usize) -> PyResult<Self> {
        let mesh = SimplexMesh::<3>::unit_cube_tet(n);
        Ok(PyMesh { inner_2d: None, inner_3d: Some(mesh), dim: 3 })
    }

    /// Number of nodes (vertices).
    pub fn n_nodes(&self) -> usize {
        match self.dim {
            2 => self.inner_2d.as_ref().unwrap().n_nodes(),
            3 => self.inner_3d.as_ref().unwrap().n_nodes(),
            _ => unreachable!(),
        }
    }

    /// Number of elements (cells).
    pub fn n_elements(&self) -> usize {
        match self.dim {
            2 => self.inner_2d.as_ref().unwrap().n_elems(),
            3 => self.inner_3d.as_ref().unwrap().n_elems(),
            _ => unreachable!(),
        }
    }

    /// Node indices on boundary faces matching the given tags.
    ///
    /// Supported for 2-D meshes. Each tag corresponds to a side:
    /// tag 1 = bottom (y≈0), tag 2 = right (x≈1), tag 3 = top (y≈1), tag 4 = left (x≈0).
    pub fn boundary_nodes(&self, tags: Vec<i32>) -> PyResult<Vec<u32>> {
        match self.dim {
            2 => {
                let mesh = self.inner_2d.as_ref().unwrap();
                let nodes = boundary_nodes_with_tags(mesh, &tags);
                Ok(nodes)
            }
            _ => Err(PyValueError::new_err(
                "boundary_nodes is only supported for 2-D meshes"
            )),
        }
    }
}
