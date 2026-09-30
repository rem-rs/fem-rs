use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use fem_mesh::Mesh;
use fem_mesh::extrude_tri3_to_prisms;
use fem_mesh::extrude_quad4_to_hex8;
use fem_mesh::build_supermesh;
use fem_mesh::{refine_uniform, refine_uniform_3d};
use fem_mesh::topology::MeshTopology;

/// Collect node ids lying on boundary faces whose tag is in `tags`.
///
/// Reimplemented here: the former `fem_mesh::boundary_nodes_with_tags` lived in
/// the deleted `moving_mesh` module (removed as dead code in 47e09a6 — fem-py,
/// its only caller, is outside the dead-code audit). Same semantics: sorted
/// unique nodes over boundary faces with a matching tag. Dimension-generic:
/// 2-D boundary faces are edges, 3-D ones are triangles/quads — the
/// `MeshTopology` face accessors cover both.
fn boundary_nodes_with_tags<const D: usize>(mesh: &Mesh<D>, tags: &[i32]) -> Vec<u32> {
    let mut out = std::collections::BTreeSet::<u32>::new();
    for f in mesh.face_iter() {
        if tags.contains(&mesh.face_tag(f)) {
            for &n in mesh.face_nodes(f) {
                out.insert(n);
            }
        }
    }
    out.into_iter().collect()
}

/// Unstructured simplex mesh in 2-D or 3-D.
///
/// Construct via the static methods:
///   ``fem.Mesh.unit_square_tri(n)``  — 2-D triangle mesh
///   ``fem.Mesh.unit_cube_tet(n)``    — 3-D tetrahedral mesh
#[pyclass(name = "Mesh")]
pub struct PyMesh {
    pub(crate) inner_2d: Option<Mesh<2>>,
    pub(crate) inner_3d: Option<Mesh<3>>,
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
        let mesh = Mesh::<2>::unit_square_tri(n);
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
        let mesh = Mesh::<3>::unit_cube_tet(n);
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
    /// Supported for 2-D and 3-D meshes. On the built-in generators each tag
    /// corresponds to a side:
    /// - 2-D (`unit_square_tri`): tag 1 = bottom (y≈0), tag 2 = right (x≈1),
    ///   tag 3 = top (y≈1), tag 4 = left (x≈0).
    /// - 3-D (`unit_cube_tet`): tag 1 = bottom (z≈0), tag 2 = front (y≈0),
    ///   tag 3 = right (x≈1), tag 4 = back (y≈1), tag 5 = left (x≈0),
    ///   tag 6 = top (z≈1).
    pub fn boundary_nodes(&self, tags: Vec<i32>) -> PyResult<Vec<u32>> {
        match self.dim {
            2 => {
                let mesh = self.inner_2d.as_ref().unwrap();
                let nodes = boundary_nodes_with_tags(mesh, &tags);
                Ok(nodes)
            }
            3 => {
                let mesh = self.inner_3d.as_ref().unwrap();
                let nodes = boundary_nodes_with_tags(mesh, &tags);
                Ok(nodes)
            }
            _ => Err(PyValueError::new_err("invalid mesh state")),
        }
    }

    /// Uniform refinement: split every element, return the refined mesh.
    ///
    /// Delegates to `fem_mesh::refine_uniform` (2-D) / `refine_uniform_3d`
    /// (3-D); the original mesh is left untouched.
    pub fn refine(&self) -> PyResult<PyMesh> {
        match self.dim {
            2 => {
                let m = refine_uniform(self.inner_2d.as_ref().unwrap());
                Ok(PyMesh { inner_2d: Some(m), inner_3d: None, dim: 2 })
            }
            3 => {
                let m = refine_uniform_3d(self.inner_3d.as_ref().unwrap());
                Ok(PyMesh { inner_2d: None, inner_3d: Some(m), dim: 3 })
            }
            _ => Err(PyValueError::new_err("invalid mesh state")),
        }
    }

    /// Geometry type of element `elem` (e.g. `"Tri3"`, `"Tet4"`).
    pub fn element_type(&self, elem: usize) -> PyResult<String> {
        let t = match self.dim {
            2 => {
                let mesh = self.inner_2d.as_ref().unwrap();
                if elem >= mesh.n_elems() {
                    return Err(PyValueError::new_err(
                        format!("element_type: element {} out of range (n_elements = {})", elem, mesh.n_elems())
                    ));
                }
                mesh.element_type(elem as u32)
            }
            3 => {
                let mesh = self.inner_3d.as_ref().unwrap();
                if elem >= mesh.n_elems() {
                    return Err(PyValueError::new_err(
                        format!("element_type: element {} out of range (n_elements = {})", elem, mesh.n_elems())
                    ));
                }
                mesh.element_type(elem as u32)
            }
            _ => return Err(PyValueError::new_err("invalid mesh state")),
        };
        Ok(format!("{t:?}"))
    }

    /// Node ids (connectivity) of element `elem`.
    pub fn element_nodes(&self, elem: usize) -> PyResult<Vec<u32>> {
        match self.dim {
            2 => {
                let mesh = self.inner_2d.as_ref().unwrap();
                if elem >= mesh.n_elems() {
                    return Err(PyValueError::new_err(
                        format!("element_nodes: element {} out of range (n_elements = {})", elem, mesh.n_elems())
                    ));
                }
                Ok(mesh.element_nodes(elem as u32).to_vec())
            }
            3 => {
                let mesh = self.inner_3d.as_ref().unwrap();
                if elem >= mesh.n_elems() {
                    return Err(PyValueError::new_err(
                        format!("element_nodes: element {} out of range (n_elements = {})", elem, mesh.n_elems())
                    ));
                }
                Ok(mesh.element_nodes(elem as u32).to_vec())
            }
            _ => Err(PyValueError::new_err("invalid mesh state")),
        }
    }

    /// Coordinates of node `node` (list of `dim` floats).
    pub fn node_coords(&self, node: usize) -> PyResult<Vec<f64>> {
        match self.dim {
            2 => {
                let mesh = self.inner_2d.as_ref().unwrap();
                if node >= mesh.n_nodes() {
                    return Err(PyValueError::new_err(
                        format!("node_coords: node {} out of range (n_nodes = {})", node, mesh.n_nodes())
                    ));
                }
                Ok(mesh.node_coords(node as u32).to_vec())
            }
            3 => {
                let mesh = self.inner_3d.as_ref().unwrap();
                if node >= mesh.n_nodes() {
                    return Err(PyValueError::new_err(
                        format!("node_coords: node {} out of range (n_nodes = {})", node, mesh.n_nodes())
                    ));
                }
                Ok(mesh.node_coords(node as u32).to_vec())
            }
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
