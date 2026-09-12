//! D85: 2-D `Tri3` `Mesh::set_curvature(p)` for any order (`TriPk` geometry).
//!
//! Three checks:
//!
//! 1. **`p = 2` is bit-identical to the pre-generalisation implementation.**
//!    The two FNV-1a checksums below cover the whole `GeometryData`
//!    (`order`, `nodes_per_elem`, `n_nodes`, the connectivity, and the
//!    coordinates' bit patterns) and were captured *before*
//!    `set_curvature_tri3_2d` was generalised; the old code built the edge
//!    midpoints as `0.5·(c_a + c_b)` in first-encounter edge order, which is
//!    exactly what `t = 1/2` in the general path reproduces.
//! 2. **The curved geometry is the right one.**  For `p = 3, 4` every
//!    element's geometry map must reproduce the original *straight* geometry:
//!    `Σ_k x_k·φ_k(ξ) = Σ_v λ_v(ξ)·c_v` for the `H1TriPk` basis and the
//!    reference barycentric coordinates `λ`.  This pins the node positions and
//!    their association with the reference nodes (a wrong ordering or a
//!    mis-shared edge node breaks it), and the node count pins the sharing:
//!    `n_nodes = V + E·(p−1) + NE·(p−1)(p−2)/2`.
//! 3. **MFEM agreement** (one-off, `tmp/probe_d85.cpp`): `Mesh::SetCurvature(4)`
//!    on `data/solid-cht.mesh` gives per element the same multiset of node
//!    positions to `max|Δ| = 5.0e-16` (`p = 3`: 4.6e-16), with identical vertex
//!    coordinates and identical node counts (`p = 4`: 15 per element, 221 DOFs;
//!    `p = 3`: 10 per element, 130 DOFs).  MFEM's probe reads the mesh with
//!    `refine = 1`, which rotates each element's local vertex list
//!    (`MarkTriMeshForRefinement`), so the comparison is per element as a set.

use fem_element::lagrange::factory::H1TriPk;
use fem_element::quadrature::tri_rule;
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;

fn solid_cht() -> Mesh<2> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/solid-cht.mesh");
    read_mfem_file(path)
        .expect("read solid-cht.mesh")
        .mesh2d
        .expect("2-D")
}

/// FNV-1a over `order`, `nodes_per_elem`, `n_nodes`, `conn` and the raw bits of
/// `coords` — bit-exact by construction.
fn geometry_checksum(m: &Mesh<2>) -> u64 {
    let g = m.geometry.as_ref().expect("geometry");
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(&(g.order as u32).to_le_bytes());
    bytes.extend_from_slice(&(g.nodes_per_elem as u32).to_le_bytes());
    bytes.extend_from_slice(&(g.n_nodes as u32).to_le_bytes());
    for &c in &g.conn {
        bytes.extend_from_slice(&c.to_le_bytes());
    }
    for &v in &g.coords {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `p = 2` output, byte for byte (checksums captured from the previous
/// implementation, which only supported `p = 2`).
#[test]
fn tri3_2d_set_curvature_p2_is_bit_identical() {
    let mut square = Mesh::<2>::unit_square_tri(3);
    square.set_curvature(2);
    assert_eq!(
        geometry_checksum(&square),
        0x1bcd_25a5_b6e2_3861,
        "p=2 geometry of unit_square_tri(3) changed"
    );
    assert_eq!(square.geometry.as_ref().unwrap().nodes_per_elem, 6);
    assert_eq!(square.geometry.as_ref().unwrap().n_nodes, 49);

    let mut solid = solid_cht();
    solid.set_curvature(2);
    assert_eq!(
        geometry_checksum(&solid),
        0x351c_585a_3e41_9f7a,
        "p=2 geometry of solid-cht.mesh changed"
    );
}

/// Every element's order-`p` geometry must interpolate the original straight
/// geometry exactly, at every quadrature point of the rule the assembly would
/// use, and the nodes must be shared as MFEM shares them.
#[test]
fn tri3_2d_set_curvature_reproduces_the_linear_geometry() {
    for &p in &[2usize, 3, 4, 6] {
        for mut mesh in [solid_cht(), Mesh::<2>::unit_square_tri(2)] {
            let (nv, ne) = (mesh.n_nodes(), mesh.n_elems());
            let mut edges = mesh.clone();
            edges.build_edge_connectivity();
            let n_edges = edges.n_edges();

            mesh.build_face_to_elem();
            mesh.set_curvature(p as u8);
            let g = mesh.geometry.as_ref().expect("geometry");

            // Sharing: vertices are re-used, each edge owns `p-1` shared nodes,
            // each element owns `(p-1)(p-2)/2` interior nodes.
            let n_interior = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
            assert_eq!(g.nodes_per_elem, (p + 1) * (p + 2) / 2);
            assert_eq!(
                g.n_nodes,
                nv + n_edges * (p - 1) + ne * n_interior,
                "p={p}: unexpected number of geometry nodes"
            );

            let fe = H1TriPk::new(p);
            let n = fe.n_dofs();
            let rule = tri_rule((p + 1).min(20) as u8);
            let mut phi = vec![0.0_f64; n];
            let mut worst = 0.0_f64;
            let mut scale = 1.0_f64;
            for e in 0..ne as u32 {
                let verts = mesh.element_nodes(e);
                let c: Vec<[f64; 2]> = verts.iter().map(|&v| [mesh.node_coords(v)[0], mesh.node_coords(v)[1]]).collect();
                let nds = mesh.geometry_nodes(e);
                assert_eq!(nds.len(), n);
                for xi in &rule.points {
                    let lam = [1.0 - xi[0] - xi[1], xi[0], xi[1]];
                    fe.eval_basis(xi, &mut phi);
                    let mut x = [0.0_f64; 2];
                    for k in 0..n {
                        let xk = mesh.geom_coords_of(nds[k]);
                        for d in 0..2 {
                            x[d] += phi[k] * xk[d];
                            scale = scale.max(xk[d].abs());
                        }
                    }
                    for d in 0..2 {
                        let want = lam[0] * c[0][d] + lam[1] * c[1][d] + lam[2] * c[2][d];
                        worst = worst.max((x[d] - want).abs());
                    }
                }
            }
            eprintln!(
                "p={p} nv={nv} ne={ne} nodes={} max|geom − affine| = {worst:.3e} (scale {scale:.3})",
                g.n_nodes
            );
            assert!(
                worst <= 1e-12 * scale,
                "p={p}: the order-{p} node table does not reproduce the linear geometry \
                 (max|Δ| = {worst:.3e}) — node positions or their ordering are wrong"
            );
        }
    }
}

/// The interior nodes are private to their element and the edge nodes are
/// shared, not duplicated: a node index may be reachable from several elements
/// only through the edges.
#[test]
fn tri3_2d_set_curvature_shares_edge_nodes() {
    let mut mesh = Mesh::<2>::unit_square_tri(2);
    mesh.set_curvature(4);
    let g = mesh.geometry.as_ref().unwrap();
    let n_elems = mesh.n_elems();
    let npe = g.nodes_per_elem;
    // Nodes used by more than one element must be vertices or edge nodes; the
    // last `(p-1)(p-2)/2 = 3` slots of every element are its private interior.
    let mut owners = vec![0usize; g.n_nodes];
    for e in 0..n_elems {
        for &nd in &g.conn[e * npe..(e + 1) * npe] {
            owners[nd as usize] += 1;
        }
    }
    let mut n_shared = 0;
    let mut n_private = 0;
    for e in 0..n_elems {
        for k in 3 + 3 * (4 - 1)..npe {
            let nd = g.conn[e * npe + k] as usize;
            assert_eq!(owners[nd], 1, "interior node {nd} is shared by element {e}");
        }
        for &nd in &g.conn[e * npe..e * npe + 3 + 3 * (4 - 1)] {
            if owners[nd as usize] > 1 {
                n_shared += 1;
            } else {
                n_private += 1;
            }
        }
    }
    eprintln!("p=4: shared/private boundary-node slots = {n_shared}/{n_private}");
    assert!(n_shared > 0 && n_private > 0);
}
