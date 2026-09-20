//! `Mesh::PrintCharacteristics` / `Mesh::PrintInfo` — port of MFEM
//! `mesh/mesh.cpp:255-323` (D496, driven by
//! `miniapps/nurbs/nurbs_mesh_info.cpp` and the `mesh-explorer` style
//! `PrintInfo` banner).
//!
//! The output reproduces the C++ line-for-line: entity counts by geometry
//! (`Mesh::PrintElementsByGeometry`), the Euler number, and the min/max
//! element size `h = |det J|^{1/dim}` and aspect ratio `κ = σ₀/σ_{dim-1}`
//! of the P1 Jacobian at the *reference-element center* (`Geometry::GetCenter`
//! — `1/3` for triangles, `0.5` for squares/cubes, `1/4` for tets), evaluated
//! exactly like `Mesh::GetCharacteristics` → `GetElementJacobian` →
//! `IsoparametricTransformation::EvalJacobian` (`Mult(PointMat, dshape)` with
//! ascending accumulation).
//!
//! # Scope (honest gaps)
//!
//! - Straight (P1) geometry only, matching MFEM **when `Nodes == NULL`**.
//!   For curved meshes MFEM evaluates the isoparametric FE — this port
//!   panics instead of silently printing wrong numbers.
//! - Jacobians are implemented for Quad4/Tri3 (2-D) and Hex8/Tet4 (3-D).
//!   Wedge/prism and pyramid meshes still get their *entity counts* but panic
//!   on the size/aspect-ratio statistics.
//! - 0-D and 1-D (`Point`/`Segment`) meshes are not handled.

use crate::element_type::ElementType;
use crate::simplex::Mesh;
use crate::topology::MeshTopology as _;

/// MFEM `Geometry::Name` (`fem/geom.cpp:19`): capitalized geometry names.
fn geom_name(et: ElementType) -> &'static str {
    match et {
        ElementType::Point1 => "Point",
        ElementType::Line2 | ElementType::Line3 => "Segment",
        ElementType::Tri3 | ElementType::Tri6 => "Triangle",
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => "Square",
        ElementType::Tet4 | ElementType::Tet10 => "Tetrahedron",
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => "Cube",
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => "Prism",
        ElementType::Pyramid5 | ElementType::Pyramid13 => "Pyramid",
        ElementType::Polygon => "Polygon",
    }
}

/// The 2-D/3-D base geometry id used for grouping: 2 = Triangle, 3 = Square,
/// 4 = Tetrahedron, 5 = Cube, 6 = Prism, 7 = Pyramid (MFEM `Geometry::Type`).
fn geom_id(et: ElementType) -> u8 {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => 2,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => 3,
        ElementType::Tet4 | ElementType::Tet10 => 4,
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => 5,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => 6,
        ElementType::Pyramid5 | ElementType::Pyramid13 => 7,
        _ => 0,
    }
}

/// MFEM `Mesh::PrintElementsByGeometry` (`mesh/mesh.cpp:243-253`): the
/// `"N Name(s) + M Name(s)"` summary for the present geometry ids.
fn print_elems_by_geometry(counts: &[(u8, usize)], label: &mut String) {
    let mut first = true;
    for &(g, n) in counts {
        if n == 0 {
            continue;
        }
        if !first {
            label.push_str(" + ");
        } else {
            first = false;
        }
        let name = match g {
            2 => "Triangle",
            3 => "Square",
            4 => "Tetrahedron",
            5 => "Cube",
            6 => "Prism",
            7 => "Pyramid",
            _ => "Point",
        };
        label.push_str(&format!("{} {}(s)", n, name));
    }
}

/// `%g` with 6 significant digits — C++ `operator<<(ostream, double)` at the
/// default precision (reuse of the `nurbs_patch` formatter).
use crate::nurbs_patch::format_g as g6;

/// MFEM `Geometry::PerfGeomToGeomJac[TRIANGLE]` (`fem/geom.cpp:211-226`): the
/// constant right factor applied by `Geometry::JacToPerfJac` to every triangle
/// Jacobian (`PJ = J · M`), i.e. the inverse of the unit-right-triangle →
/// equilateral-reference-triangle map at the element center.  Values dumped
/// at %.17g by the D496 probe `tmp/d496/perf_jac_probe.cpp`
/// (`Geometries.JacToPerfJac(TRIANGLE, I, PJ)` against MFEM 4.10).
const PERF_TO_GEOM_JAC_TRI: [[f64; 2]; 2] = [
    [1.0, -0.57735026918962584],
    [0.0, 1.1547005383792517],
];

/// `Geometry::PerfGeomToGeomJac[TETRAHEDRON]` (same probe; SQUARE/CUBE use
/// identity performance jacobians, so quads/hexes need no factor).
const PERF_TO_GEOM_JAC_TET: [[f64; 3]; 3] = [
    [1.0, -0.57735026918962584, -0.40824829046386302],
    [0.0, 1.1547005383792517, -0.40824829046386302],
    [0.0, 0.0, 1.2247448713915892],
];

/// P1 Jacobian `J = ∂x/∂ξ` (row-major `dim×dim`) of element `e` at MFEM's
/// reference-element center, with `Mult(PointMat, dshape)` accumulation
/// order.  `nodes[e]` holds the vertex ids in MFEM connectivity order.
fn p1_jacobian_at_center<const D: usize>(
    mesh: &Mesh<D>,
    e: u32,
    nodes: &[u32],
) -> [[f64; 3]; 3] {
    let et = mesh.element_type_at(e);
    let coords: Vec<[f64; D]> = nodes.iter().map(|&n| mesh.coords_of(n)).collect();
    let mut j = [[0.0f64; 3]; 3];
    // `dshape(j, d)` at the center; values are exact binary fractions in
    // every case, so the accumulation is bit-identical to the C++ `Mult`.
    match et {
        ElementType::Quad4 => {
            // BiLinear2DFiniteElement::CalcDShape at (0.5, 0.5).
            let ds: [[f64; 2]; 4] = [
                [-0.5, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [-0.5, 0.5],
            ];
            for (jj, d) in ds.iter().enumerate() {
                for (k, c) in coords[jj].iter().take(2).enumerate() {
                    j[k][0] += c * d[0];
                    j[k][1] += c * d[1];
                }
            }
        }
        ElementType::Tri3 => {
            // Linear2DFiniteElement::CalcDShape (constant).
            let ds: [[f64; 2]; 3] = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
            for (jj, d) in ds.iter().enumerate() {
                for (k, c) in coords[jj].iter().take(2).enumerate() {
                    j[k][0] += c * d[0];
                    j[k][1] += c * d[1];
                }
            }
        }
        ElementType::Hex8 => {
            // TriLinear3DFiniteElement::CalcDShape at (0.5, 0.5, 0.5).
            let ds: [[f64; 3]; 8] = [
                [-0.25, -0.25, -0.25],
                [0.25, -0.25, -0.25],
                [0.25, 0.25, -0.25],
                [-0.25, 0.25, -0.25],
                [-0.25, -0.25, 0.25],
                [0.25, -0.25, 0.25],
                [0.25, 0.25, 0.25],
                [-0.25, 0.25, 0.25],
            ];
            for (jj, d) in ds.iter().enumerate() {
                for (k, c) in coords[jj].iter().take(3).enumerate() {
                    j[k][0] += c * d[0];
                    j[k][1] += c * d[1];
                    j[k][2] += c * d[2];
                }
            }
        }
        ElementType::Tet4 => {
            // Linear3DFiniteElement::CalcDShape (constant).
            let ds: [[f64; 3]; 4] = [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ];
            for (jj, d) in ds.iter().enumerate() {
                for (k, c) in coords[jj].iter().take(3).enumerate() {
                    j[k][0] += c * d[0];
                    j[k][1] += c * d[1];
                    j[k][2] += c * d[2];
                }
            }
        }
        other => panic!(
            "Mesh::PrintCharacteristics: no P1 Jacobian for element type {other:?} \
             (only Quad4/Tri3/Hex8/Tet4 are supported; see the module docs)"
        ),
    }
    // `Geometries.JacToPerfJac(geom, eltransf->Jacobian(), J)` — for
    // triangles/tets the performance jacobian is `J · PerfGeomToGeomJac`
    // (kernels::Mult, ascending k); for squares/cubes it is `J` itself.
    match et {
        ElementType::Tri3 => {
            let m = PERF_TO_GEOM_JAC_TRI;
            let pj = [
                [j[0][0] * m[0][0] + j[0][1] * m[1][0], j[0][0] * m[0][1] + j[0][1] * m[1][1]],
                [j[1][0] * m[0][0] + j[1][1] * m[1][0], j[1][0] * m[0][1] + j[1][1] * m[1][1]],
            ];
            j[0][0] = pj[0][0];
            j[0][1] = pj[0][1];
            j[1][0] = pj[1][0];
            j[1][1] = pj[1][1];
        }
        ElementType::Tet4 => {
            let m = PERF_TO_GEOM_JAC_TET;
            let mut pj = [[0.0f64; 3]; 3];
            for (i, row) in pj.iter_mut().enumerate() {
                for (jj, v) in row.iter_mut().enumerate() {
                    *v = j[i][0] * m[0][jj] + j[i][1] * m[1][jj] + j[i][2] * m[2][jj];
                }
            }
            j = pj;
        }
        _ => {}
    }
    j
}

/// MFEM `DenseMatrix::Det()` for the 2×2 row-major Jacobian.
fn det2(j: &[[f64; 3]; 3]) -> f64 {
    j[0][0] * j[1][1] - j[1][0] * j[0][1]
}

/// MFEM `DenseMatrix::Det()` for the 3×3 row-major Jacobian — the exact
/// `d[0]*(d[4]*d[8]-d[5]*d[7]) + d[3]*(d[2]*d[7]-d[1]*d[8]) +
///  d[6]*(d[1]*d[5]-d[2]*d[4])` operand order of `linalg/densemat.cpp`.
fn det3(j: &[[f64; 3]; 3]) -> f64 {
    j[0][0] * (j[1][1] * j[2][2] - j[2][1] * j[1][2])
        + j[0][1] * (j[2][0] * j[1][2] - j[1][0] * j[2][2])
        + j[0][2] * (j[1][0] * j[2][1] - j[2][0] * j[1][1])
}

/// h/κ statistics of [`Mesh::print_characteristics`].
struct HStats {
    h_min: f64,
    h_max: f64,
    kappa_min: f64,
    kappa_max: f64,
}

/// `Mesh::GetCharacteristics` (`mesh/mesh.cpp:206-241`).
fn get_characteristics<const D: usize>(mesh: &Mesh<D>, dim: usize) -> HStats {
    let mut st = HStats {
        h_min: f64::INFINITY,
        h_max: f64::NEG_INFINITY,
        kappa_min: f64::INFINITY,
        kappa_max: f64::NEG_INFINITY,
    };
    let inv_dim = 1.0 / dim as f64;
    for e in 0..mesh.n_elems() as u32 {
        let nodes: Vec<u32> = mesh.element_nodes(e).to_vec();
        let j = p1_jacobian_at_center(mesh, e, &nodes);
        let weight = if dim == 2 { det2(&j) } else { det3(&j) };
        let h = weight.abs().powf(inv_dim);
        let kappa = if dim == D {
            // Column-major packing for the MFEM singular-value kernels.
            if dim == 2 {
                let d = [j[0][0], j[1][0], j[0][1], j[1][1]];
                crate::mfem_kernels::calc_singularvalue_2(&d, 0)
                    / crate::mfem_kernels::calc_singularvalue_2(&d, 1)
            } else {
                let d = [
                    j[0][0], j[1][0], j[2][0], j[0][1], j[1][1], j[2][1], j[0][2], j[1][2],
                    j[2][2],
                ];
                crate::mfem_kernels::calc_singularvalue_3(&d, 0)
                    / crate::mfem_kernels::calc_singularvalue_3(&d, 2)
            }
        } else {
            -1.0
        };
        if h < st.h_min {
            st.h_min = h;
        }
        if h > st.h_max {
            st.h_max = h;
        }
        if kappa < st.kappa_min {
            st.kappa_min = kappa;
        }
        if kappa > st.kappa_max {
            st.kappa_max = kappa;
        }
    }
    st
}

/// All element faces as canonical (sorted) node-id keys plus their base
/// geometry id — the counting surrogate for MFEM's `faces`/`faces_info`
/// tables (`GetNFaces`, per-geometry face counts).
fn element_faces(et: ElementType, nodes: &[u32]) -> Vec<(u8, Vec<u32>)> {
    let quad = |n: &[u32; 4]| (3u8, {
        let mut v = n.to_vec();
        v.sort_unstable();
        v
    });
    let tri = |n: &[u32; 3]| (2u8, {
        let mut v = n.to_vec();
        v.sort_unstable();
        v
    });
    match et {
        ElementType::Hex8 => vec![
            quad(&[nodes[3], nodes[2], nodes[1], nodes[0]]),
            quad(&[nodes[0], nodes[1], nodes[5], nodes[4]]),
            quad(&[nodes[1], nodes[2], nodes[6], nodes[5]]),
            quad(&[nodes[2], nodes[3], nodes[7], nodes[6]]),
            quad(&[nodes[3], nodes[0], nodes[4], nodes[7]]),
            quad(&[nodes[4], nodes[5], nodes[6], nodes[7]]),
        ],
        ElementType::Tet4 => vec![
            tri(&[nodes[1], nodes[2], nodes[3]]),
            tri(&[nodes[0], nodes[3], nodes[2]]),
            tri(&[nodes[0], nodes[1], nodes[3]]),
            tri(&[nodes[0], nodes[2], nodes[1]]),
        ],
        ElementType::Prism6 => vec![
            tri(&[nodes[0], nodes[2], nodes[1]]),
            tri(&[nodes[3], nodes[4], nodes[5]]),
            quad(&[nodes[0], nodes[1], nodes[4], nodes[3]]),
            quad(&[nodes[1], nodes[2], nodes[5], nodes[4]]),
            quad(&[nodes[2], nodes[0], nodes[3], nodes[5]]),
        ],
        ElementType::Pyramid5 => vec![
            quad(&[nodes[3], nodes[2], nodes[1], nodes[0]]),
            tri(&[nodes[0], nodes[1], nodes[4]]),
            tri(&[nodes[1], nodes[2], nodes[4]]),
            tri(&[nodes[2], nodes[3], nodes[4]]),
            tri(&[nodes[3], nodes[0], nodes[4]]),
        ],
        other => panic!(
            "Mesh::PrintCharacteristics: face table for element type {other:?} \
             is not supported (see the module docs)"
        ),
    }
}

fn push_common_header<const D: usize>(mesh: &Mesh<D>, dim: usize, s: &mut String) {
    s.push_str("Mesh Characteristics:\n");
    s.push_str(&format!("Dimension          : {}\n", dim));
    s.push_str(&format!("Space dimension    : {}\n", D));
    let _ = mesh;
}

impl Mesh<2> {
    /// MFEM `Mesh::PrintCharacteristics(NULL, NULL, os)` for a 2-D mesh —
    /// returns the printed block (with the trailing blank line).
    ///
    /// Requires straight-sided (P1) geometry and Quad4/Tri3 elements; the
    /// edge table must be built (`Mesh::build_edge_connectivity`).
    pub fn print_characteristics(&self) -> String {
        let mut s = String::new();
        push_common_header(self, 2, &mut s);

        // Mesh::build_edge_connectivity is lazy — make sure it is current.
        let mut mesh = self.clone();
        if mesh.edge_conn.is_empty() {
            mesh.build_edge_connectivity();
        }

        // num_elems_by_geom: single id entry per present geometry, in
        // `Geometry::Type` order (triangle=2 before square=3).
        let mut elem_counts: Vec<(u8, usize)> = Vec::new();
        for e in 0..self.n_elems() as u32 {
            let g = geom_id(self.element_type_at(e));
            match elem_counts.iter_mut().find(|(gi, _)| *gi == g) {
                Some((_, n)) => *n += 1,
                None => elem_counts.push((g, 1)),
            }
        }
        elem_counts.sort_by_key(|(g, _)| *g);

        s.push_str(&format!("Number of vertices : {}\n", self.n_nodes()));
        s.push_str(&format!("Number of edges    : {}\n", mesh.edge_conn.len() / 2));
        let mut el_line = format!("Number of elements : {}", self.n_elems());
        if !elem_counts.is_empty() {
            el_line.push_str("  --  ");
            print_elems_by_geometry(&elem_counts, &mut el_line);
        }
        s.push_str(&el_line);
        s.push('\n');
        s.push_str(&format!("Number of bdr elem : {}\n", self.n_faces()));
        // EulerNumber2D = NV - NE + NE(lements).
        let euler = self.n_nodes() as i64 - (mesh.edge_conn.len() / 2) as i64
            + self.n_elems() as i64;
        s.push_str(&format!("Euler Number       : {}\n", euler));

        let st = get_characteristics(self, 2);
        s.push_str(&format!("h_min              : {}\n", g6(st.h_min, 6)));
        s.push_str(&format!("h_max              : {}\n", g6(st.h_max, 6)));
        s.push_str(&format!("kappa_min          : {}\n", g6(st.kappa_min, 6)));
        s.push_str(&format!("kappa_max          : {}\n", g6(st.kappa_max, 6)));
        s.push('\n');
        s
    }

    /// MFEM `Mesh::PrintInfo(os)` — [`Self::print_characteristics`] in serial.
    pub fn print_info(&self) -> String {
        self.print_characteristics()
    }
}

impl Mesh<3> {
    /// MFEM `Mesh::PrintCharacteristics(NULL, NULL, os)` for a 3-D mesh —
    /// returns the printed block (with the trailing blank line).
    ///
    /// Requires straight-sided (P1) geometry and Hex8/Tet4 elements (prisms
    /// and pyramids get entity counts only — the size statistics panic); the
    /// edge table must be built (`Mesh::build_edge_connectivity`).
    pub fn print_characteristics(&self) -> String {
        let mut s = String::new();
        push_common_header(self, 3, &mut s);

        let mut mesh = self.clone();
        if mesh.edge_conn.is_empty() {
            mesh.build_edge_connectivity();
        }

        // Boundary-element geometry counts (in Geometry::Type order).
        let mut bdr_counts: Vec<(u8, usize)> = Vec::new();
        for f in 0..self.n_faces() as u32 {
            let g = geom_id(self.face_type_at(f));
            match bdr_counts.iter_mut().find(|(gi, _)| *gi == g) {
                Some((_, n)) => *n += 1,
                None => bdr_counts.push((g, 1)),
            }
        }
        bdr_counts.sort_by_key(|(g, _)| *g);

        // All faces (interior + boundary) via canonical node-set keys.
        let mut face_keys: std::collections::HashMap<Vec<u32>, u8> =
            std::collections::HashMap::new();
        for e in 0..self.n_elems() as u32 {
            let et = self.element_type_at(e);
            let nodes = self.element_nodes(e);
            for (g, key) in element_faces(et, nodes) {
                face_keys.entry(key).or_insert(g);
            }
        }
        let mut face_counts: Vec<(u8, usize)> = Vec::new();
        for g in face_keys.values() {
            match face_counts.iter_mut().find(|(gi, _)| gi == g) {
                Some((_, n)) => *n += 1,
                None => face_counts.push((*g, 1)),
            }
        }
        face_counts.sort_by_key(|(g, _)| *g);

        let mut elem_counts: Vec<(u8, usize)> = Vec::new();
        for e in 0..self.n_elems() as u32 {
            let g = geom_id(self.element_type_at(e));
            match elem_counts.iter_mut().find(|(gi, _)| *gi == g) {
                Some((_, n)) => *n += 1,
                None => elem_counts.push((g, 1)),
            }
        }
        elem_counts.sort_by_key(|(g, _)| *g);

        s.push_str(&format!("Number of vertices : {}\n", self.n_nodes()));
        s.push_str(&format!("Number of edges    : {}\n", mesh.edge_conn.len() / 2));
        let mut face_line = format!("Number of faces    : {}", face_keys.len());
        if !face_counts.is_empty() {
            face_line.push_str("  --  ");
            print_elems_by_geometry(&face_counts, &mut face_line);
        }
        s.push_str(&face_line);
        s.push('\n');
        let mut el_line = format!("Number of elements : {}", self.n_elems());
        if !elem_counts.is_empty() {
            el_line.push_str("  --  ");
            print_elems_by_geometry(&elem_counts, &mut el_line);
        }
        s.push_str(&el_line);
        s.push('\n');
        let mut bdr_line = format!("Number of bdr elem : {}", self.n_faces());
        if !bdr_counts.is_empty() {
            bdr_line.push_str("  --  ");
            print_elems_by_geometry(&bdr_counts, &mut bdr_line);
        }
        s.push_str(&bdr_line);
        s.push('\n');
        // EulerNumber = NV - NE + NF - NE.
        let euler = self.n_nodes() as i64
            - (mesh.edge_conn.len() / 2) as i64
            - self.n_elems() as i64
            + face_keys.len() as i64;
        s.push_str(&format!("Euler Number       : {}\n", euler));

        let st = get_characteristics(self, 3);
        s.push_str(&format!("h_min              : {}\n", g6(st.h_min, 6)));
        s.push_str(&format!("h_max              : {}\n", g6(st.h_max, 6)));
        s.push_str(&format!("kappa_min          : {}\n", g6(st.kappa_min, 6)));
        s.push_str(&format!("kappa_max          : {}\n", g6(st.kappa_max, 6)));
        s.push('\n');
        s
    }

    /// MFEM `Mesh::PrintInfo(os)` — [`Self::print_characteristics`] in serial.
    pub fn print_info(&self) -> String {
        self.print_characteristics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned unit cube grid: h = 1, kappa = 1.
    #[test]
    fn unit_hex_grid() {
        let mut mesh = Mesh::<3>::unit_cube_hex(2);
        mesh.build_edge_connectivity();
        let out = mesh.print_characteristics();
        assert!(out.contains("Dimension          : 3"));
        assert!(out.contains("Number of elements : 8  --  8 Cube(s)\n"));
        // 2×2×2 cubes of side 1/2 on the unit cube.
        assert!(out.contains("h_min              : 0.5\n"));
        assert!(out.contains("kappa_max          : 1\n"));
    }
}
