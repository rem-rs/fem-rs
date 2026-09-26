//! Weak Galerkin (WG) discretizations — Poisson, Stokes, Maxwell.
//!
//! # The family's face geometry (D810-1)
//!
//! All three modules assemble a face **stabilizer** `α · ∫_F φᵢφⱼ dS` with
//! `α = penalty/h`.  Until D810-1 each of them carried its own private copy of
//! a **chord route** for that integral — `face_geom_2d`/`face_geom_3d` built the
//! face measure from the face's **corner nodes** (`|c₁−c₀|` /
//! `|(c₁−c₀)×(c₂−c₀)|/2`), the quadrature point was the **linear interpolation
//! of those corners**, and the element reference point was recovered by
//! *inverting* the element's vertex Jacobian (`local_phys_to_ref`).  That is
//! exact only while the mesh is straight: on a mesh carrying an order-`g`
//! geometry table the element's face is the isoparametric image of its
//! reference face, which the corner route does not represent (the same disease
//! round 75 deleted from `dg.rs` / `dg_advection` / `dg_elasticity` /
//! `dg_hyperbolic` and round 76 from the PA kernels).
//!
//! The whole face path is now the **single arithmetic source** the DG modules
//! use — [`crate::dg::dg_base::face_point_geom`] /
//! [`crate::dg::dg_base::face_point_geom_3d`] — which reproduce MFEM's
//! `FaceElementTransformations`:
//!
//! * the face **measure** `|nor|` from `CalcOrtho(J_elem(eip)·d(eip)/dξ)`,
//! * the face **physical point** `xp = Elem->Transform(eip)`,
//! * the **reference point** `eip` by direct reference composition (`Loc1`) —
//!   no physical-to-reference inversion exists on the path any more,
//! * and the face rule MFEM's DG face terms use: `seg_rule` in 2-D (Σw = 1) and
//!   `tri_rule` in 3-D (Σw = 1/2), i.e. `IntRules.Get(Geometry::SEGMENT /
//!   TRIANGLE, quad_order)` — the pre-fix 2-D loop fed a *triangle* rule to a
//!   segment parameterisation (only `ξ₀` was used, and the weights summed to
//!   1/2 while the parameterisation was one-dimensional).
//!
//! Truth: `tmp/d78c/d810_probe.cpp` dumps MFEM 4.10's own
//! `FaceElementTransformations` (`nor`, `eip`, `Elem->Transform(eip)`,
//! `Elem->Weight()`) per face-QP on the same curved fixtures, and
//! `crates/assembly/tests/d810_wg_face_geometry.rs` compares them with
//! `face_point_geom`/`face_point_geom_3d` entry by entry (plus the in-repo
//! per-element divergence identity and the straight-mesh pins).
//!
//! Registered residual (not fixed here): the **volume** path of all three
//! modules still builds its element Jacobian from the element's vertices
//! (`local_jac`) and interpolates the physical point affinely, so on a curved
//! mesh the volume term is inconsistent with the (now isoparametric) face term
//! — the D808-4/D783 class, in `wg`: `weak_gradient_matrix` /
//! `weak_curl_matrix` / the Stokes body-force loop.  See `tmp/d78c/README.md`.

pub mod wg_maxwell;
pub mod wg_stokes;
pub mod wg_poisson;

pub use wg_poisson::*;
pub use wg_stokes::*;
pub use wg_maxwell::*;

use std::collections::HashMap;

use fem_element::quadrature::{seg_rule, tri_rule};
use fem_element::reference::QuadratureRule;
use fem_mesh::topology::MeshTopology;

/// The face reference rule of the WG family: MFEM's `IntRules.Get(Geometry::
/// SEGMENT / TRIANGLE, qo)` — the same rules the DG face terms use
/// (`seg_rule`: `Σw = 1` on the reference segment; `tri_rule`: `Σw = 1/2`, the
/// area of the reference triangle).  Pairing them with `|nor|` (the segment's
/// arc-length derivative in 2-D, the **cross-product magnitude** = twice the
/// triangle's area in 3-D, MFEM `CalcOrtho`) makes `Σ ipw·|nor|` the physical
/// face measure in both dimensions.
pub(crate) fn wg_face_rule(dim: usize, quad_order: u8) -> QuadratureRule {
    match dim {
        2 => seg_rule(quad_order),
        3 => tri_rule(quad_order),
        _ => panic!("wg_face_rule: unsupported dimension {dim}"),
    }
}

/// One face quadrature point of the WG family, i.e. MFEM
/// `FaceElementTransformations` at the face's own reference coordinate `xi`
/// (the face is parameterised from its node `fnodes[0]`).
pub(crate) struct WgFacePoint {
    /// Element reference point (`Loc1` reference composition — MFEM
    /// `GetElement1IntPoint()`).  Basis functions of the owning element are
    /// evaluated here; **no physical inversion is involved**.
    pub eip: Vec<f64>,
    /// Physical face point through the element's own order-`g` map
    /// (`Elem1->Transform(eip)`).
    pub xp: Vec<f64>,
    /// `nor = CalcOrtho(Trans.Jacobian())`, **outward** from the owning
    /// element; `|nor|` is the face measure element (`dS = |nor|·dξ`).
    pub nor: Vec<f64>,
}

impl WgFacePoint {
    /// `|nor|` — the face measure element (`dS = |nor|·dξ`).
    pub(crate) fn nor_mag(&self) -> f64 {
        self.nor.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

/// MFEM `FaceElementTransformations` at `xi` for the face of `elem` whose nodes
/// are `fnodes`, through [`crate::dg::dg_base::face_point_geom`] (2-D) or
/// [`crate::dg::dg_base::face_point_geom_3d`] (3-D).
pub(crate) fn wg_face_point<M: MeshTopology + ?Sized>(
    mesh: &M,
    elem: u32,
    fnodes: &[u32],
    xi: &[f64],
) -> WgFacePoint {
    match fnodes.len() {
        2 => {
            let g = crate::dg::dg_base::face_point_geom(mesh, elem, fnodes[0], fnodes[1], xi[0]);
            WgFacePoint { eip: g.eip.to_vec(), xp: g.xp.to_vec(), nor: g.nor.to_vec() }
        }
        3 => {
            let g = crate::dg::dg_base::face_point_geom_3d(
                mesh, elem, fnodes[0], fnodes[1], fnodes[2], [xi[0], xi[1]],
            );
            WgFacePoint { eip: g.eip.to_vec(), xp: g.xp.to_vec(), nor: g.nor.to_vec() }
        }
        n => panic!("wg_face_point: unsupported face with {n} nodes (Tri3/Tet4 faces only)"),
    }
}

/// The face's physical measure `∫_F dS = Σ_q ipw_q·|nor|_q` on the isoparametric
/// face — the length of a 2-D face / the area of a 3-D one.  This is the `h` of
/// the stabilizer's `α = penalty/h`; on a straight mesh it equals the chord
/// (respectively corner-triangle) value the pre-fix code computed, and on a
/// curved face it follows the element's own order-`g` face.
pub(crate) fn wg_face_measure<M: MeshTopology + ?Sized>(
    mesh: &M,
    elem: u32,
    fnodes: &[u32],
    quad_order: u8,
) -> f64 {
    let dim = mesh.dim() as usize;
    let qf = wg_face_rule(dim, quad_order);
    let mut h = 0.0;
    for (qi, xi) in qf.points.iter().enumerate() {
        h += qf.weights[qi] * wg_face_point(mesh, elem, fnodes, xi).nor_mag();
    }
    h
}

/// Boundary face → owning element, through the DG layer's single
/// implementation ([`crate::dg::dg_base::build_face_elem_map`]).
pub(crate) fn wg_boundary_face_map<M: MeshTopology>(mesh: &M) -> HashMap<u32, u32> {
    crate::dg::dg_base::build_face_elem_map(mesh, mesh.dim() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::quadrature::{seg_rule, tri_rule};
    use fem_mesh::element_type::ElementType;
    use fem_mesh::topology::MeshTopology;
    use fem_mesh::Mesh;

    // ─── Fixtures: the D787/D793 curved cell family ────────────────────────
    //
    // The MFEM-side counterpart (`tmp/d78c/d810_probe.cpp`) builds
    // `MakeCartesian{2,3}D(..., TRIANGLE/TETRAHEDRON)` + `SetCurvature(2)` +
    // the same warp applied to every geometry dof and every vertex, so the two
    // sides describe the same physical cells (pinned by
    // `crates/assembly/tests/d810_wg_face_geometry.rs`'s vertex/element/volume
    // cross-check).

    fn warp3(x: [f64; 3]) -> [f64; 3] {
        [
            x[0] + 0.1 * x[1] * x[2] + 0.2 * x[0] * x[0],
            x[1] + 0.05 * x[0] * x[2],
            x[2] + 0.1 * x[0] * x[1],
        ]
    }

    fn warp2(x: [f64; 2]) -> [f64; 2] {
        [x[0] + 0.2 * x[0] * x[0] + 0.1 * x[0] * x[1], x[1] + 0.05 * x[0] * x[1]]
    }

    fn warp_all<const D: usize>(mesh: &mut Mesh<D>, f: fn([f64; D]) -> [f64; D]) {
        let n_geom = mesh.geometry.as_ref().expect("curved table").coords.len() / D;
        {
            let geo = mesh.geometry.as_mut().expect("curved table");
            for k in 0..n_geom {
                let mut x = [0.0_f64; D];
                x.copy_from_slice(&geo.coords[k * D..(k + 1) * D]);
                let y = f(x);
                geo.coords[k * D..(k + 1) * D].copy_from_slice(&y);
            }
        }
        for k in 0..mesh.n_nodes() {
            let mut x = [0.0_f64; D];
            x.copy_from_slice(&mesh.coords[k * D..(k + 1) * D]);
            let y = f(x);
            mesh.coords[k * D..(k + 1) * D].copy_from_slice(&y);
        }
    }

    /// Unit cube's 6-tet Kuhn split (`Mesh::unit_cube_tet(1)`), order-2
    /// geometry, warped — `geom_order == 2`.
    fn curved_tet_mesh() -> Mesh<3> {
        let mut m = Mesh::<3>::unit_cube_tet(1);
        m.set_curvature(2);
        warp_all(&mut m, warp3);
        m
    }

    /// The two triangles of `[0,1]²` split along MFEM's `MakeCartesian2D`
    /// diagonal (nodes 0–2), order-2 geometry, warped.
    fn curved_tri_mesh() -> Mesh<2> {
        let mut m = Mesh::<2>::uniform(
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            vec![0, 1, 2, 0, 2, 3],
            vec![1, 1],
            ElementType::Tri3,
            vec![0, 1, 1, 2, 2, 3, 3, 0],
            vec![1, 1, 1, 1],
            ElementType::Line2,
        );
        m.set_curvature(2);
        warp_all(&mut m, warp2);
        m
    }

    /// Every face of every element as `(elem, face_nodes, is_boundary)` — the
    /// (element, face) pairs the wg stabilizer visits (interior list + boundary
    /// list), without duplicating the list construction.
    fn element_faces<const D: usize>(mesh: &Mesh<D>) -> Vec<(u32, Vec<u32>)> {
        let mut v: Vec<(u32, Vec<u32>)> = Vec::new();
        for f in &crate::InteriorFaceList::build(mesh).faces {
            v.push((f.elem_left, f.face_nodes.clone()));
            v.push((f.elem_right, f.face_nodes.clone()));
        }
        let bmap = wg_boundary_face_map(mesh);
        for bf in mesh.face_iter() {
            if let Some(&el) = bmap.get(&bf) {
                v.push((el, mesh.face_nodes(bf).to_vec()));
            }
        }
        v
    }

    /// `∫_T div(x − x₀) dV = ∮_∂T (x − x₀)·n̂ dS` per element, with the right
    /// side assembled **through the wg face route** (`wg_face_point`, hence
    /// `face_point_geom`/`_3d`): the unnormalised form is
    /// `Σ_F Σ_q ipw_q · (x_q − x₀)·nor_q = d·|T|`.
    ///
    /// This is an **identity**, not an implementation comparison: it holds only
    /// if the face measure (`|nor|`), the physical point (`xp`) and the outward
    /// orientation are all the ones of the element's own order-`g` boundary —
    /// the element measure on the left comes from `element_jacobian_at`, a
    /// different code path.  A chord/corner face route fails it on a curved
    /// element (there the face does not tile the element boundary).
    fn divergence_defect<const D: usize>(mesh: &Mesh<D>) -> f64 {
        let dim = D;
        let qf = match dim {
            2 => seg_rule(12),
            3 => tri_rule(12),
            _ => unreachable!(),
        };
        let mut worst = 0.0_f64;
        for e in 0..mesh.n_elems() as u32 {
            let x0: Vec<f64> = {
                let c = mesh.node_coords(mesh.element_nodes(e)[0]);
                (0..dim).map(|d| c[d]).collect()
            };
            let mut flux = 0.0;
            for (fe, fnodes) in element_faces(mesh)
                .into_iter()
                .filter(|(fe, _)| *fe == e)
            {
                for (qi, xi) in qf.points.iter().enumerate() {
                    let g = wg_face_point(mesh, fe, &fnodes, xi);
                    let dot: f64 =
                        (0..dim).map(|d| (g.xp[d] - x0[d]) * g.nor[d]).sum();
                    flux += qf.weights[qi] * dot;
                }
            }
            let vol = element_volume(mesh, e, dim);
            worst = worst.max((flux - dim as f64 * vol).abs() / vol.abs().max(1e-30));
        }
        worst
    }

    /// `∫_T |det J|` through the mesh's own order-`g` map.
    fn element_volume<const D: usize>(mesh: &Mesh<D>, e: u32, dim: usize) -> f64 {
        let qf = match dim {
            2 => tri_rule(12),
            3 => tet_rule_for_volume(),
            _ => unreachable!(),
        };
        let mut v = 0.0;
        for (qi, xi) in qf.points.iter().enumerate() {
            let arr: [f64; D] = std::array::from_fn(|k| xi[k]);
            let (j, _x) = fem_mesh::transformation::element_jacobian_at(mesh, e, &arr, dim);
            v += qf.weights[qi] * j.determinant().abs();
        }
        v
    }

    fn tet_rule_for_volume() -> fem_element::reference::QuadratureRule {
        // `tet_rule(8)` — a high-order (positive-weight) rule for the measure.
        fem_element::quadrature::tet_rule(8)
    }

    /// The divergence identity through the wg face route holds to round-off on
    /// **curved** cells of both dimensions — the oracle that the chord route
    /// could not pass (`chord_route_witness` below measures what it scored).
    #[test]
    fn d810_1_divergence_identity_holds_on_curved_cells() {
        let tet = curved_tet_mesh();
        assert_eq!(tet.geom_order(), 2);
        let d3 = divergence_defect(&tet);
        let tri = curved_tri_mesh();
        assert_eq!(tri.geom_order(), 2);
        let d2 = divergence_defect(&tri);
        println!("D810-1 divergence defect: tet {d3:.3e}, tri {d2:.3e}");
        assert!(d2 < 1e-11, "2-D curved: (∮(x−x₀)·nor) − 2|T| = {d2:.3e}");
        assert!(d3 < 1e-11, "3-D curved: (∮(x−x₀)·nor) − 3|T| = {d3:.3e}");
    }

    /// The straight-mesh case is exact for the very same route (a polynomial
    /// identity on an affine cell), so the fixtures are not "passing by luck".
    #[test]
    fn d810_1_divergence_identity_holds_on_straight_cells() {
        let tet = Mesh::<3>::unit_cube_tet(1);
        assert_eq!(tet.geom_order(), 1);
        let tri = Mesh::<2>::uniform(
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            vec![0, 1, 2, 0, 2, 3],
            vec![1, 1],
            ElementType::Tri3,
            vec![0, 1, 1, 2, 2, 3, 3, 0],
            vec![1, 1, 1, 1],
            ElementType::Line2,
        );
        let d3 = divergence_defect(&tet);
        let d2 = divergence_defect(&tri);
        println!("D810-1 straight divergence defect: tet {d3:.3e}, tri {d2:.3e}");
        assert!(d2 < 1e-13, "2-D straight: {d2:.3e}");
        assert!(d3 < 1e-13, "3-D straight: {d3:.3e}");
    }

    /// The **pre-fix chord route** (reproduced here as the witness — the
    /// library no longer contains it) fails the same identity on the curved
    /// cells: the chord measure/triangle does not tile the element boundary and
    /// the corner-interpolated point is not on it.  This is the teeth of
    /// [`d810_1_divergence_identity_holds_on_curved_cells`].
    #[test]
    fn d810_1_chord_route_fails_the_identity_on_curved_cells() {
        let tet = curved_tet_mesh();
        let tri = curved_tri_mesh();
        let d3 = chord_divergence_defect(&tet);
        let d2 = chord_divergence_defect(&tri);
        println!("D810-1 pre-fix chord route defect: tet {d3:.3e}, tri {d2:.3e}");
        assert!(d2 > 1e-3, "2-D chord route: {d2:.3e} — witness has no teeth");
        assert!(d3 > 1e-3, "3-D chord route: {d3:.3e} — witness has no teeth");
    }

    /// The pre-fix route, verbatim: corner chord / corner triangle for the
    /// measure, corner interpolation for the point, and its own outward
    /// corner-chord normal, with the pre-fix rule (`tri_rule` in both
    /// dimensions) and the pre-fix `h`.
    fn chord_divergence_defect<const D: usize>(mesh: &Mesh<D>) -> f64 {
        let dim = D;
        let qf = tri_rule(12);
        let mut worst = 0.0_f64;
        for e in 0..mesh.n_elems() as u32 {
            let c0 = mesh.node_coords(mesh.element_nodes(e)[0]);
            let x0: Vec<f64> = (0..dim).map(|d| c0[d]).collect();
            let mut flux = 0.0;
            for (fe, fnodes) in element_faces(mesh).into_iter().filter(|(fe, _)| *fe == e) {
                let c: Vec<[f64; 3]> = fnodes
                    .iter()
                    .map(|&n| {
                        let p = mesh.node_coords(n);
                        [p[0], p[1], if dim == 3 { p[2] } else { 0.0 }]
                    })
                    .collect();
                let (nor, xs) = if dim == 2 {
                    // The chord of the element's **own** CCW edge `le → le+1`
                    // (the pre-fix `face_geom_2d` orientation), then
                    // `CalcOrtho`'s 2-D rule `(t_y, −t_x)` — outward.
                    let (le, _fwd) =
                        crate::dg::dg_base::find_local_edge(mesh, fe, fnodes[0], fnodes[1]);
                    let en = mesh.element_nodes(fe);
                    let p0 = mesh.node_coords(en[le]);
                    let p1 = mesh.node_coords(en[(le + 1) % en.len()]);
                    let t = [p1[0] - p0[0], p1[1] - p0[1]];
                    (vec![t[1], -t[0]], vec![c[0][0], c[0][1]])
                } else {
                    let u1 = [c[1][0] - c[0][0], c[1][1] - c[0][1], c[1][2] - c[0][2]];
                    let u2 = [c[2][0] - c[0][0], c[2][1] - c[0][1], c[2][2] - c[0][2]];
                    let nn = [
                        u1[1] * u2[2] - u1[2] * u2[1],
                        u1[2] * u2[0] - u1[0] * u2[2],
                        u1[0] * u2[1] - u1[1] * u2[0],
                    ];
                    (vec![nn[0], nn[1], nn[2]], vec![c[0][0], c[0][1], c[0][2]])
                };
                // Corner-chord normal, oriented out of the element by the
                // element's own fourth vertex (3-D) / CCW winding (2-D).
                let nor = orient_outward(mesh, fe, &fnodes, nor);
                for (qi, xi) in qf.points.iter().enumerate() {
                    let xp: Vec<f64> = if dim == 2 {
                        vec![xs[0] + xi[0] * (c[1][0] - c[0][0]), xs[1] + xi[0] * (c[1][1] - c[0][1])]
                    } else {
                        vec![
                            xs[0] + xi[0] * (c[1][0] - c[0][0]) + xi[1] * (c[2][0] - c[0][0]),
                            xs[1] + xi[0] * (c[1][1] - c[0][1]) + xi[1] * (c[2][1] - c[0][1]),
                            xs[2] + xi[0] * (c[1][2] - c[0][2]) + xi[1] * (c[2][2] - c[0][2]),
                        ]
                    };
                    let dot: f64 = (0..dim).map(|d| (xp[d] - x0[d]) * nor[d]).sum();
                    flux += qf.weights[qi] * dot;
                }
            }
            let vol = element_volume(mesh, e, dim);
            worst = worst.max((flux - dim as f64 * vol).abs() / vol.abs().max(1e-30));
        }
        worst
    }

    /// Make a corner-built face normal outward from `elem`.
    fn orient_outward<const D: usize>(
        mesh: &Mesh<D>,
        elem: u32,
        fnodes: &[u32],
        mut nor: Vec<f64>,
    ) -> Vec<f64> {
        let dim = D;
        let en = mesh.element_nodes(elem);
        if dim == 2 {
            // The chord (a→b) with the element's own CCW winding: the outward
            // normal is already the one `chord_divergence_defect` built.
            return nor;
        }
        let idx = |n: u32| fnodes.iter().position(|&m| m == n);
        let fourth = en.iter().find(|n| idx(**n).is_none()).copied();
        let Some(fourth) = fourth else { return nor };
        let c0 = mesh.node_coords(fnodes[0]);
        let cf = mesh.node_coords(fourth);
        let to4 = [cf[0] - c0[0], cf[1] - c0[1], cf[2] - c0[2]];
        if nor[0] * to4[0] + nor[1] * to4[1] + nor[2] * to4[2] > 0.0 {
            nor = nor.iter().map(|v| -v).collect();
        }
        nor
    }
}
