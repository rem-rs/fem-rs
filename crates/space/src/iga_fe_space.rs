//! Bridge between single-patch IGA and [`FESpace`].
//!
//! The generic [`FESpace`](crate::fe_space::FESpace) trait is tied to [`MeshTopology`]. This module
//! provides a minimal 2D tensor-product IGA "mesh" view (one `Quad4` or `Quad9` cell per
//! non-empty knot span) so that IGA can participate in the same type ecosystem as simplicial FE.
//! For Poisson/IGA assembly, the `fem-assembly` crate provides `iga_assembler` and
//! `Assembler::assemble_bilinear_iga_1d` / `assemble_bilinear_iga_2d` (B-spline / NURBS basis);
//! the generic `Assembler::assemble_bilinear` path on this mesh uses Lagrange shapes and is not
//! equivalent.
//!
//! **1D supported:** `IgaFESpace1D` when each knot span has `p+1 ∈ {2, 3}` (`Line2` / `Line3`).
//! Node locations use [Greville abscissae](https://en.wikipedia.org/wiki/Collocation_method#Example)
//! from [`IgaSpace1D::greville_param_coords`].
//! **2D supported:** `IgaFESpace2D` when each knot-span element has
//! `(p+1)(q+1) ∈ {4, 9}` (biquadratic `p=q=2` or bilinear `p=q=1` on quads), matching
//! [`ElementType::Quad4`] and [`fem_mesh::ElementType::Quad9`].

use fem_core::types::{DofId, ElemId, FaceId, NodeId};
use fem_linalg::Vector;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;

use crate::fe_space::{FESpace, SpaceType};
use crate::iga::IgaSpace1D;
use crate::iga::IgaSpace2D;

// ─── 1D single-patch IGA FESpace ────────────────────────────────────────────

/// One line segment (per non-empty parametric span) in the B-spline / NURBS 1D patch.
/// Global node id equals the basis / control index; coordinates are Greville abscissae in
/// \([0,1]\) (one coordinate per point).
#[derive(Debug, Clone)]
pub struct IgaSinglePatchMesh1D {
    n_nodes:     usize,
    node_coords: Vec<Vec<f64>>,
    element_type: ElementType,
    connectivity: Vec<Vec<NodeId>>,
    bdy:         [Vec<NodeId>; 2],
}

impl IgaSinglePatchMesh1D {
    /// Build the mesh. Requires `p+1 ∈ {2, 3}`.
    pub fn from_iga_space(space: &IgaSpace1D) -> Result<Self, String> {
        let p = space.degree();
        let nloc = p + 1;
        let el_ty = match nloc {
            2 => ElementType::Line2,
            3 => ElementType::Line3,
            _ => {
                return Err(format!(
                    "IgaFESpace1D: p+1 must be 2 or 3 for the FESpace bridge, got {nloc} (p={p})"
                ));
            }
        };
        let g = space.greville_param_coords()?;
        if g.len() != space.n_dofs() {
            return Err("IgaFESpace1D: greville count mismatch".to_string());
        }
        let nctrl = space.n_dofs();
        let node_coords: Vec<Vec<f64>> = g.into_iter().map(|u| vec![u]).collect();

        let mut connectivity = Vec::new();
        for span in space.non_empty_spans() {
            let active = space.active_dofs_for_span(span)?;
            if active.len() != nloc {
                return Err(format!(
                    "IgaSinglePatchMesh1D: active len {} != {nloc} at span {span}",
                    active.len()
                ));
            }
            connectivity.push(active.into_iter().map(|i| i as u32).collect());
        }
        let n0: NodeId = 0;
        let n1: NodeId = (nctrl.saturating_sub(1)) as u32;
        let bdy = [vec![n0], vec![n1]];

        Ok(Self {
            n_nodes: nctrl,
            node_coords,
            element_type: el_ty,
            connectivity,
            bdy,
        })
    }
}

/// [`FESpace`] for a 1D B-spline / NURBS patch on an interval in parametric \([0,1]\) (see [`IgaSpace1D`]).
#[derive(Debug, Clone)]
pub struct IgaFESpace1D {
    iga:  IgaSpace1D,
    mesh: IgaSinglePatchMesh1D,
}

impl IgaFESpace1D {
    /// Build from an [`IgaSpace1D`]. Fails if `p+1` is not 2 or 3.
    pub fn new(iga: IgaSpace1D) -> Result<Self, String> {
        let mesh = IgaSinglePatchMesh1D::from_iga_space(&iga)?;
        Ok(Self { iga, mesh })
    }

    /// Underlying IGA space.
    pub fn iga(&self) -> &IgaSpace1D { &self.iga }

    /// Take the IGA space.
    pub fn into_iga(self) -> IgaSpace1D { self.iga }
}

impl FESpace for IgaFESpace1D {
    type Mesh = IgaSinglePatchMesh1D;

    fn mesh(&self) -> &Self::Mesh { &self.mesh }

    fn n_dofs(&self) -> usize { self.iga.n_dofs() }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        node_ids_as_dof_ids(self.mesh.element_nodes(elem))
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let n = self.n_dofs();
        let mut v = Vector::zeros(n);
        let s = v.as_slice_mut();
        for i in 0..n {
            s[i] = f(self.mesh.node_coords[i].as_slice());
        }
        v
    }

    fn space_type(&self) -> SpaceType { SpaceType::H1 }

    fn order(&self) -> u8 { self.iga.degree() as u8 }
}

impl MeshTopology for IgaSinglePatchMesh1D {
    fn dim(&self) -> u8 { 1 }

    fn n_nodes(&self) -> usize { self.n_nodes }

    fn n_elements(&self) -> usize { self.connectivity.len() }

    /// Two endpoint faces (0-D) on a 1D interval mesh.
    fn n_boundary_faces(&self) -> usize { 2 }

    fn element_nodes(&self, elem: ElemId) -> &[NodeId] {
        &self.connectivity[elem as usize]
    }

    fn element_type(&self, _elem: ElemId) -> ElementType { self.element_type }

    fn element_tag(&self, _elem: ElemId) -> i32 { 0 }

    fn node_coords(&self, node: NodeId) -> &[f64] { &self.node_coords[node as usize] }

    fn face_nodes(&self, face: FaceId) -> &[NodeId] {
        match face {
            0 => &self.bdy[0],
            1 => &self.bdy[1],
            _ => &[],
        }
    }

    fn face_tag(&self, _face: FaceId) -> i32 { 0 }

    fn face_elements(&self, face: FaceId) -> (ElemId, Option<ElemId>) {
        let n_el = self.connectivity.len();
        if n_el == 0 { return (0, None); }
        match face {
            0 => (0, None),
            1 => ((n_el - 1) as u32, None),
            _ => (0, None),
        }
    }
}

// NodeId and DofId are both u32; safe same-layout view.
#[inline]
fn node_ids_as_dof_ids(s: &[NodeId]) -> &[DofId] {
    if s.is_empty() { return &[]; }
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const DofId, s.len()) }
}

/// Minimal mesh view for a single 2D IGA patch: one quadrilateral "element" per non-empty
/// `(u,v)` knot span, with global node ids equal to B-spline / NURBS control-point indices.
#[derive(Debug, Clone)]
pub struct IgaSinglePatchMesh2D {
    n_nodes:        usize,
    node_coords:    Vec<Vec<f64>>,
    /// Same for all elements (from patch degree / tensor size).
    element_type:   ElementType,
    connectivity:   Vec<Vec<NodeId>>,
}

impl IgaSinglePatchMesh2D {
    /// Build connectivity from a space. Fails if `(p+1)(q+1)` is not 4 or 9.
    pub fn from_iga_space(space: &IgaSpace2D) -> Result<Self, String> {
        let p = space.degree_u();
        let q = space.degree_v();
        let n_local = (p + 1) * (q + 1);
        let el_ty = match n_local {
            4  => ElementType::Quad4,
            9  => ElementType::Quad9,
            _ => {
                return Err(format!(
                    "IgaFESpace2D: (p+1)(q+1) must be 4 or 9 for the FESpace bridge, got {n_local} (p={p}, q={q})"
                ));
            }
        };

        let nctrl = space.n_dofs();
        let mut node_coords = vec![vec![0.0_f64; 2]; nctrl];
        for (g, c) in space.control_points().iter().enumerate() {
            node_coords[g][0] = c[0];
            node_coords[g][1] = c[1];
        }

        let mut connectivity = Vec::with_capacity(space.non_empty_spans().len());
        for (span_u, span_v) in space.non_empty_spans() {
            let active = space.active_dofs_for_span(span_u, span_v)?;
            if active.len() != n_local {
                return Err(format!(
                    "IgaSinglePatchMesh2D: active dofs per span {span_u},{span_v} len {} != {n_local}",
                    active.len()
                ));
            }
            connectivity.push(active.into_iter().map(|i| i as u32).collect());
        }

        Ok(Self {
            n_nodes: nctrl,
            node_coords,
            element_type: el_ty,
            connectivity,
        })
    }
}

/// [`FESpace`] wrapper for a single 2D IGA / NURBS patch (see [`IgaSpace2D`]).
#[derive(Debug, Clone)]
pub struct IgaFESpace2D {
    iga:  IgaSpace2D,
    mesh: IgaSinglePatchMesh2D,
}

impl IgaFESpace2D {
    /// Take ownership of an [`IgaSpace2D`] and build the mesh bridge.
    pub fn new(iga: IgaSpace2D) -> Result<Self, String> {
        let mesh = IgaSinglePatchMesh2D::from_iga_space(&iga)?;
        Ok(Self { iga, mesh })
    }

    /// Underlying IGA space (parametric + control net + weights).
    pub fn iga(&self) -> &IgaSpace2D { &self.iga }

    /// Extract the IGA space.
    pub fn into_iga(self) -> IgaSpace2D { self.iga }
}

impl FESpace for IgaFESpace2D {
    type Mesh = IgaSinglePatchMesh2D;

    fn mesh(&self) -> &Self::Mesh { &self.mesh }

    fn n_dofs(&self) -> usize { self.iga.n_dofs() }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        node_ids_as_dof_ids(self.mesh.element_nodes(elem))
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let n = self.n_dofs();
        let mut v = Vector::zeros(n);
        let slice = v.as_slice_mut();
        for i in 0..n {
            let c = &self.mesh.node_coords[i];
            slice[i] = f(c.as_slice());
        }
        v
    }

    fn space_type(&self) -> SpaceType {
        // Scalar, smooth (C0) IGA for standard Poisson-type problems.
        SpaceType::H1
    }

    fn order(&self) -> u8 {
        std::cmp::max(self.iga.degree_u(), self.iga.degree_v()) as u8
    }
}

impl MeshTopology for IgaSinglePatchMesh2D {
    fn dim(&self) -> u8 { 2 }

    fn n_nodes(&self) -> usize { self.n_nodes }

    fn n_elements(&self) -> usize { self.connectivity.len() }

    fn n_boundary_faces(&self) -> usize {
        // Minimal bridge: no boundary face discretization. Use IGA-specific APIs for sides.
        0
    }

    fn element_nodes(&self, elem: ElemId) -> &[NodeId] {
        &self.connectivity[elem as usize]
    }

    fn element_type(&self, _elem: ElemId) -> ElementType { self.element_type }

    fn element_tag(&self, _elem: ElemId) -> i32 { 0 }

    fn node_coords(&self, node: NodeId) -> &[f64] {
        &self.node_coords[node as usize]
    }

    fn face_nodes(&self, _face: FaceId) -> &[NodeId] {
        &[]
    }

    fn face_tag(&self, _face: FaceId) -> i32 { 0 }

    fn face_elements(&self, _face: FaceId) -> (ElemId, Option<ElemId>) { (0, None) }
}

#[cfg(test)]
mod tests {
    use super::{IgaFESpace1D, IgaFESpace2D};
    use crate::fe_space::FESpace;
    use crate::iga::{IgaSpace1D, IgaSpace2D};
    use fem_mesh::MeshTopology;

    #[test]
    fn iga_fespace2d_quadratic_matches_span_count() {
        let iga = IgaSpace2D::new_uniform_clamped(2, 2, 6, 5).expect("space");
        let fe = IgaFESpace2D::new(iga).expect("fespace");
        // Non-empty parametric spans: u from p..(nu-1) and v from q..(nv-1) with strict knot inequality.
        // nu=6, p=2 => spans 2..5 => 4; nv=5, q=2 => spans 2..4 => 3; product = 12.
        let n_u_spans = 4usize;
        let n_v_spans = 3usize;
        assert_eq!(fe.mesh().n_elements(), n_u_spans * n_v_spans);
        assert_eq!(fe.n_dofs(), 30);
    }

    #[test]
    fn iga_fespace2d_bilinear_dofs_per_element_4() {
        let iga = IgaSpace2D::new_uniform_clamped(1, 1, 3, 3).expect("space");
        let fe = IgaFESpace2D::new(iga).expect("fespace");
        assert_eq!(fe.element_dofs(0).len(), 4);
    }

    #[test]
    fn iga_fespace1d_linear_line2_spans() {
        let iga = IgaSpace1D::new_uniform_clamped(1, 4).expect("1d");
        let fe = IgaFESpace1D::new(iga).expect("fes");
        // p=1, 4 cps, uniform: n_spans = n - p = 3; Line2, 2 dofs/elem
        assert_eq!(fe.mesh().n_elements(), 3);
        assert_eq!(fe.mesh().dim(), 1);
        assert_eq!(fe.element_dofs(0).len(), 2);
    }

    #[test]
    fn iga_fespace1d_greville_ends_01() {
        let iga = IgaSpace1D::new_uniform_clamped(1, 2).expect("1d");
        let g = iga.greville_param_coords().expect("g");
        assert!((g[0] - 0.0).abs() < 1e-12);
        assert!((g[1] - 1.0).abs() < 1e-12);
    }
}
