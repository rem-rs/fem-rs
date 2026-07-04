use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use fem_mesh::SimplexMesh;
use fem_mesh::boundary_nodes_with_tags;
use fem_mesh::extrude_tri3_to_prisms;
use fem_mesh::extrude_quad4_to_hex8;
use fem_mesh::build_supermesh;

/// Unstructured simplex mesh in 2-D or 3-D.
///
/// Construct via the static methods:
///   ``fem.Mesh.unit_square_tri(n)``  — 2-D triangle mesh
///   ``fem.Mesh.unit_cube_tet(n)``    — 3-D tetrahedral mesh
#[pyclass(name = "Mesh")]
pub struct PyMesh {
    pub(crate) inner_2d: Option<SimplexMesh<2>>,
    pub(crate) inner_3d: Option<SimplexMesh<3>>,
    pub(crate) dim: u8,
}

#[pymethods]
impl PyMesh {
    /// Direct construction is disabled; use one of the static factory methods.
    #[new]
    pub fn new() -> PyResult<Self> {
        Err(PyValueError::new_err(
            "Mesh cannot be constructed directly. Use Mesh.unit_square_tri(n) or Mesh.unit_cube_tet(n)."
        ))
    }

    /// Create a 2-D unit-square mesh of `n×n` subdivisions (each quad → 2 triangles).
    ///
    /// `n` must be ≥ 1.
    #[staticmethod]
    pub fn unit_square_tri(n: usize) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err(
                "unit_square_tri: n must be ≥ 1"
            ));
        }
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        Ok(PyMesh { inner_2d: Some(mesh), inner_3d: None, dim: 2 })
    }

    /// Create a 3-D unit-cube mesh of `n×n×n` tetrahedra.
    ///
    /// `n` must be ≥ 1.
    #[staticmethod]
    pub fn unit_cube_tet(n: usize) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err(
                "unit_cube_tet: n must be ≥ 1"
            ));
        }
        let mesh = SimplexMesh::<3>::unit_cube_tet(n);
        Ok(PyMesh { inner_2d: None, inner_3d: Some(mesh), dim: 3 })
    }

    /// Spatial dimension (2 or 3).
    pub fn dim(&self) -> u8 {
        self.dim
    }

    /// Number of nodes (vertices).
    pub fn n_nodes(&self) -> PyResult<usize> {
        match self.dim {
            2 => Ok(self.inner_2d.as_ref().unwrap().n_nodes()),
            3 => Ok(self.inner_3d.as_ref().unwrap().n_nodes()),
            _ => Err(PyValueError::new_err("invalid mesh state")),
        }
    }

    /// Number of elements (cells).
    pub fn n_elements(&self) -> PyResult<usize> {
        match self.dim {
            2 => Ok(self.inner_2d.as_ref().unwrap().n_elems()),
            3 => Ok(self.inner_3d.as_ref().unwrap().n_elems()),
            _ => Err(PyValueError::new_err("invalid mesh state")),
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
            3 => Err(PyValueError::new_err(
                "boundary_nodes is only supported for 2-D meshes"
            )),
            _ => Err(PyValueError::new_err("invalid mesh state")),
        }
    }

    /// Extrude a 2-D Tri3 mesh into a 3-D Prism6 mesh.
    #[pyo3(signature = (n_layers, height))]
    pub fn extrude_to_prisms(&self, n_layers: usize, height: f64) -> PyResult<PyMesh> {
        let mesh = self.inner_2d.as_ref().ok_or_else(||
            PyValueError::new_err("extrude_to_prisms requires a 2-D Tri3 mesh")
        )?;
        let m3 = extrude_tri3_to_prisms(mesh, n_layers, height);
        Ok(PyMesh { inner_2d: None, inner_3d: Some(m3), dim: 3 })
    }

    /// Extrude a 2-D Quad4 mesh into a 3-D Hex8 mesh.
    #[pyo3(signature = (n_layers, height))]
    pub fn extrude_to_hex(&self, n_layers: usize, height: f64) -> PyResult<PyMesh> {
        let mesh = self.inner_2d.as_ref().ok_or_else(||
            PyValueError::new_err("extrude_to_hex requires a 2-D Quad4 mesh")
        )?;
        let m3 = extrude_quad4_to_hex8(mesh, n_layers, height);
        Ok(PyMesh { inner_2d: None, inner_3d: Some(m3), dim: 3 })
    }

    /// Compute the supermesh (intersection) of two 2-D Tri3 meshes.
    #[staticmethod]
    pub fn supermesh(mesh_a: &PyMesh, mesh_b: &PyMesh) -> PyResult<PyMesh> {
        let a = mesh_a.inner_2d.as_ref().ok_or_else(||
            PyValueError::new_err("supermesh: mesh_a must be 2-D Tri3")
        )?;
        let b = mesh_b.inner_2d.as_ref().ok_or_else(||
            PyValueError::new_err("supermesh: mesh_b must be 2-D Tri3")
        )?;
        let (super_mesh, _) = build_supermesh(a, b);
        Ok(PyMesh { inner_2d: Some(super_mesh), inner_3d: None, dim: 2 })
    }
}
