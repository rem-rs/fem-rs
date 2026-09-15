//! D158 regression: 3-D H(div)/H(curl) global DOF numbering and element
//! semantics must match MFEM 4.10 so that `tools_get_values` reads VisIt
//! data-collection coefficient vectors (which are stored in MFEM's global
//! ordering) correctly.
//!
//! Reference values dumped from MFEM 4.10 (`mfem-4.10_ser`) for the meshes
//! below; see `tmp/d158/` for the generators and the C++ `get-values`
//! transcripts.
//!
//! * MFEM numbers global space DOFs **entity-major**: all edge DOFs (mesh
//!   edge index order = first-encounter order), then all face DOFs
//!   (first-encounter face order), then the element-interior DOFs in element
//!   order.  The round-38-and-earlier fem-rs layout interleaved each
//!   element's faces/interiors right after its edges, which scrambled every
//!   multi-element ND2/RT1/RT2 coefficient vector read from a DC file.
//! * MFEM's `Mesh::FindPoints`/`GetValue` chain evaluates hex tensor
//!   elements in the `[-1,1]^3` reference frame; the fem-rs element bases
//!   (HexQk / HexL2GL / HexNDk / HexRTk) share that convention, while
//!   `fem_mesh::find_points` reports `[0,1]^3` coordinates — the tool layer
//!   (`miniapps/tools/get_values.rs`) converts.

use fem_io::mfem::read_mfem;
use fem_mesh::MeshTopology;
use fem_space::{HCurlSpace, HDivSpace};

/// The 6-tetrahedron split of one hexahedron (sizes 1.0 x 1.2 x 0.8), exactly
/// as `Mesh::MakeCartesian3D(1, 1, 1, Element::TETRAHEDRON, ...)` writes it.
const TET_MESH: &str = "MFEM mesh v1.0

dimension
3

elements
6
1 4 7 0 3 1
1 4 7 0 1 5
1 4 7 0 5 4
1 4 7 0 2 3
1 4 7 0 6 2
1 4 7 0 4 6

boundary
12
1 2 3 0 2
1 2 3 0 2
1 2 0 3 1
6 2 7 4 5
6 2 4 7 6
5 2 6 0 4
5 2 0 6 2
3 2 7 1 3
3 2 1 7 5
2 2 5 0 1
2 2 0 5 4
4 2 7 2 6

vertices
8
3
0 0 0
1 0 0
0 1.2 0
1 1.2 0
0 0 0.8
1 0 0.8
0 1.2 0.8
1 1.2 0.8
";

/// A single hexahedron (1.0 x 1.2 x 0.8).
const HEX_MESH: &str = "MFEM mesh v1.0

dimension
3

elements
1
1 5 0 1 2 3 4 5 6 7

boundary
6
1 3 0 1 3 2
1 3 1 2 6 5
1 3 4 5 6 7
1 3 0 4 7 3
1 3 0 3 2 1
1 3 4 6 7 5

vertices
8
3
0 0 0
1 0 0
1 1.2 0
0 1.2 0
0 0 0.8
1 0 0.8
1 1.2 0.8
0 1.2 0.8
";

fn tet_mesh() -> fem_mesh::Mesh<3> {
    read_mfem(TET_MESH.as_bytes())
        .expect("tet mesh parses")
        .mesh3d
        .expect("expected a 3-D mesh")
}

fn hex_mesh() -> fem_mesh::Mesh<3> {
    read_mfem(HEX_MESH.as_bytes())
        .expect("hex mesh parses")
        .mesh3d
        .expect("expected a 3-D mesh")
}

/// MFEM `GetElementVDofs(0)` for `ND_FECollection(1, 3)` on the 6-tet mesh:
/// `[-1, -2, -3, 3, 4, -6]` — global DOFs `0..6` with orientation signs
/// `[-1, -1, -1, +1, +1, -1]` (the min→max vertex rule).
#[test]
fn tet_hcurl_nd1_vdofs_match_mfem() {
    let mesh = tet_mesh();
    let sp = HCurlSpace::new(mesh, 1);
    assert_eq!(sp.n_dofs(), 19);
    let dofs = sp.element_dofs(0);
    assert_eq!(dofs, &[0, 1, 2, 3, 4, 5]);
    let signs: &[f64] = sp.element_signs(0);
    for (got, want) in signs.iter().zip([-1.0_f64, -1.0, -1.0, 1.0, 1.0, -1.0]) {
        assert!((got - want).abs() < 1e-14, "sign {got} vs {want}");
    }
}

/// MFEM `GetElementVDofs(0)` for `RT_FECollection(1, 3)` on the 6-tet mesh:
/// tet 0 owns face DOFs `0..12` (three per face, first-encounter face order)
/// and interior DOFs `54, 55, 56` — i.e. **all face DOFs precede all
/// interior DOFs** (MFEM's entity-major layout; the pre-D158 interleaved
/// layout put tet 0's interiors at 12, 13, 14).
#[test]
fn tet_hdiv_rt1_vdofs_match_mfem() {
    let mesh = tet_mesh();
    let sp = HDivSpace::new(mesh, 1);
    assert_eq!(sp.n_dofs(), 72);
    let dofs = sp.element_dofs(0);
    let want: Vec<u32> = (0..12).chain(54..57).collect();
    assert_eq!(dofs, &want[..]);
    let signs = sp.element_signs(0);
    assert!(signs.iter().all(|&s| s == 1.0));
}

/// Single hexahedron: `ND_FECollection(1, 3)` has exactly the 12 edge DOFs
/// `0..12` (no faces/interiors) and `RT_FECollection(1, 3)` has 36 DOFs
/// (24 face + 12 interior, all belonging to the one element) — both
/// `GetElementVDofs(0) = 0..n` with positive orientation.
#[test]
fn hex_hcurl_hdiv_vdofs_match_mfem() {
    let mesh = hex_mesh();

    let nd = HCurlSpace::new(mesh.clone(), 1);
    assert_eq!(nd.n_dofs(), 12);
    assert_eq!(nd.element_dofs(0), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);

    let rt = HDivSpace::new(mesh, 1);
    assert_eq!(rt.n_dofs(), 36);
    let want: Vec<u32> = (0..36).collect();
    assert_eq!(rt.element_dofs(0), &want[..]);
}

/// End-to-end value check through the get-values pipeline's ingredients:
/// load the coefficient vector `e_0` (MFEM global DOF 0 = 1) of an
/// `ND_FECollection(1, 3)` field on the 6-tet mesh and evaluate at the
/// centroid of tet 0 with the *correct* covariant Piola transform
/// (`phi = J^{-T} psi`, the formula the D158 arbitration patch restores in
/// `GridFunction::evaluate_vector_at_element`).  MFEM 4.10 `GetVectorValue`
/// gives `(0.25, 0, 0.3125)` for this configuration.
#[test]
fn tet_nd1_unit_dof_value_matches_mfem_getvectorvalue() {
    use fem_element::nedelec::TetNDk;
    use fem_element::reference::VectorReferenceElement;

    let mesh = tet_mesh();
    let sp = HCurlSpace::new(mesh.clone(), 1);
    let k = 1usize;
    let vre = TetNDk::new(k);
    let n = vre.n_dofs();

    // The file's coefficient vector for field "f0": unit at global DOF 0.
    let mut data = vec![0.0_f64; sp.n_dofs()];
    data[0] = 1.0;

    // Reference coordinates of tet 0's centroid, as find_points reports
    // them (MFEM's tetrahedron reference frame is the unit simplex).
    let xi = [0.25_f64, 0.25, 0.25];

    let mut psi = vec![0.0_f64; n * 3];
    vre.eval_basis_vec(&xi, &mut psi);
    let dofs = sp.element_dofs(0);
    let signs = sp.element_signs(0);

    // Geometry Jacobian from the element's corner nodes (simplex).
    let nodes = mesh.element_nodes(0);
    let x0 = mesh.node_coords(nodes[0]);
    let mut jac = [[0.0_f64; 3]; 3];
    for c in 0..3 {
        let xc = mesh.node_coords(nodes[c + 1]);
        for r in 0..3 {
            jac[r][c] = xc[r] - x0[r];
        }
    }
    // J^{-T}: adjugate-transpose of the 3x3 column-major J.
    let det = jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1])
        - jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0])
        + jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]);
    let mut j_inv_t = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let a = jac[(i + 1) % 3][(j + 1) % 3];
            let b = jac[(i + 2) % 3][(j + 2) % 3];
            let c = jac[(i + 1) % 3][(j + 2) % 3];
            let d = jac[(i + 2) % 3][(j + 1) % 3];
            j_inv_t[i][j] = (a * b - c * d) / det;
        }
    }

    let mut val = [0.0_f64; 3];
    for i in 0..n {
        let c = signs[i] * data[dofs[i] as usize];
        for r in 0..3 {
            for cc in 0..3 {
                val[r] += c * j_inv_t[r][cc] * psi[i * 3 + cc];
            }
        }
    }

    // MFEM 4.10 `GridFunction::GetVectorValue(0, ip(1/4,1/4,1/4))` for e_0.
    for (got, want) in val.iter().zip([0.25, 0.0, 0.3125]) {
        assert!((got - want).abs() < 5e-13, "val {got} vs {want}");
    }
}
