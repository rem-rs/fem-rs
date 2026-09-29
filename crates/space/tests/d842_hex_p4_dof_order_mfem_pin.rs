//! D842-3: the hex H¹ **global** dof numbering must be MFEM's, slot for slot,
//! for every order — and the refined-hex meshes feeding it must carry MFEM's
//! vertex numbering.
//!
//! Round 91 registered the ex26 hex-mode divergence (D842-3) as "the hex-P4 H1
//! dof ordering is a permutation of MFEM's".  The round-92 probes
//! (`tmp/d92a/probe_hex_dofs.cpp`, MFEM 4.10 serial) re-rooted that
//! registration:
//!
//! 1. `build_pk_hex`'s edge blocks were *already* canonical — the implicit
//!    reversal inside `get_edge_dofs_pk` assigns the ids ascending along the
//!    sorted (min, max) vertex pair, which is MFEM's mesh-edge convention
//!    (`Mesh::GetEdgeVertices`: `vert[0] < vert[1]`, "consistent with the
//!    global edge orientation"; element-local slots re-map through
//!    `H1_FECollection::DofOrderForOrientation(SEGMENT, ori)`).  The dof-table
//!    tests below pin that, but they passed **before** the round-92 edit too
//!    (the edit only makes the convention explicit), so they are regression
//!    guards, not the red→green proof.
//! 2. The actual permutation was one level lower: `refine_uniform_3d` numbered
//!    a *straight* hex mesh's new vertices **per parent cell** (12 edge
//!    midpoints, 6 face centers, 1 center in one 19-id block per element)
//!    while MFEM's `Mesh::UniformRefinement` numbers them in global entity
//!    phases — *all* edge midpoints (mesh-edge order, `oedge + E`), then *all*
//!    face centers (`oface + F`), then *all* element centers (`oelem + C`).
//!    On ex26's inline-hex hierarchy the first refinement already diverged
//!    (`tmp/d92a/refined1_mesh_diff.txt`, 2460 diff lines pre-fix, 0 after),
//!    permuting every later level and with it the whole V-cycle trajectory.
//!    The third test pins the refined mesh against MFEM's own print.
//!
//! Ground truth data (all in `tests/data/`; `.mesh.txt` because the repo's
//! `.gitignore` covers `*.mesh`):
//! * `d842_hex_dof_order_mfem_dump.txt` — `GetElementDofs` tables with global
//!   ids (unlike D157, which compared physical positions only, the global
//!   numbering is the point here) for two 2×2×2 hex cubes and orders 1–4:
//!   - `cart222` (`d842_hex222_cart222.mesh.txt`):
//!     `Mesh::MakeCartesian3D(2,2,2,HEXAHEDRON)` (every local edge direction
//!     ascends — regression anchor),
//!   - `scr222` (`d842_hex222_scrambled.mesh.txt`): the same cube with vertex
//!     ids reversed (26 − id), so most local `CUBE::Edges` directions descend
//!     (e.g. p = 4, element 0, local edge (26→25) must carry block ids
//!     `29 28 27`, not `27 28 29`).
//! * `d842_inline_hex_coarse.mesh.txt` + `d842_inline_hex_ref1_mfem.mesh.txt`
//!   — MFEM's `Print` of `data/inline-hex.mesh` (the ex26 coarse mesh) before
//!   and after one `UniformRefinement`.

use fem_io::mfem::{read_mfem_file, write_mfem_file_3d};
use fem_mesh::refine_uniform_3d;
use fem_mesh::MeshTopology as _;
use fem_space::{fe_space::FESpace, H1Space};
use std::collections::HashMap;

const DUMP: &str = include_str!("data/d842_hex_dof_order_mfem_dump.txt");
const CART222: &str = include_str!("data/d842_hex222_cart222.mesh.txt");
const SCR222: &str = include_str!("data/d842_hex222_scrambled.mesh.txt");
const INLINE_COARSE: &str = include_str!("data/d842_inline_hex_coarse.mesh.txt");
const INLINE_REF1_MFEM: &str = include_str!("data/d842_inline_hex_ref1_mfem.mesh.txt");

/// `mesh_name -> (ndofs per order, element -> dof id table per order)`.
fn parse_mfem_dump() -> HashMap<String, Vec<(u8, usize, Vec<Vec<u32>>)>> {
    let mut out: HashMap<String, Vec<(u8, usize, Vec<Vec<u32>>)>> = HashMap::new();
    let mut mesh = String::new();
    let mut order = 0u8;
    let mut ndofs = 0usize;
    let mut elems: Vec<Vec<u32>> = Vec::new();
    let flush = |mesh: &str, order: u8, ndofs: usize, elems: &mut Vec<Vec<u32>>,
                 out: &mut HashMap<String, Vec<(u8, usize, Vec<Vec<u32>>)>>| {
        if order > 0 {
            out.entry(mesh.to_string())
                .or_default()
                .push((order, ndofs, std::mem::take(elems)));
        }
    };
    for line in DUMP.lines() {
        if let Some(rest) = line.strip_prefix("MESH ") {
            flush(&mesh, order, ndofs, &mut elems, &mut out);
            order = 0;
            mesh = rest.split_whitespace().next().unwrap().to_string();
        } else if let Some(rest) = line.strip_prefix("ORDER ") {
            flush(&mesh, order, ndofs, &mut elems, &mut out);
            let t: Vec<&str> = rest.split_whitespace().collect();
            order = t[0].parse().unwrap();
            ndofs = t[1]["NDofs=".len()..].parse().unwrap();
        } else if let Some(rest) = line.strip_prefix("ELD ") {
            let (head, tail) = rest.split_once(':').unwrap();
            let p: u8 = head.split_whitespace().next().unwrap().parse().unwrap();
            assert_eq!(p, order);
            let e: usize = head.split_whitespace().nth(1).unwrap().parse().unwrap();
            let dofs: Vec<u32> = tail
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            if elems.len() <= e {
                elems.resize(e + 1, Vec::new());
            }
            elems[e] = dofs;
        }
    }
    flush(&mesh, order, ndofs, &mut elems, &mut out);
    out
}

fn mesh_from_str(name: &str, text: &str) -> fem_mesh::Mesh<3> {
    let path = std::env::temp_dir().join(format!("d842_pin_{name}.mesh"));
    std::fs::write(&path, text).unwrap();
    read_mfem_file(&path)
        .unwrap_or_else(|e| panic!("read {name}: {e}"))
        .mesh3d
        .expect("3-D mesh")
}

fn check_mesh(mesh_name: &str, mesh_text: &str) {
    let truth = parse_mfem_dump();
    let sections = truth
        .get(mesh_name)
        .unwrap_or_else(|| panic!("no dump section for {mesh_name}"));
    let mesh = mesh_from_str(mesh_name, mesh_text);
    for (p, ndofs, elem_dofs) in sections {
        let space: H1Space<fem_mesh::Mesh<3>> = H1Space::new(mesh.clone(), *p);
        assert_eq!(
            space.n_dofs(),
            *ndofs,
            "{mesh_name} p={p}: total dof count differs from MFEM"
        );
        assert_eq!(
            space.mesh().n_elements(),
            elem_dofs.len(),
            "{mesh_name} p={p}: element count differs"
        );
        for e in 0..elem_dofs.len() {
            let got = space.element_dofs_u32(e as u32);
            assert_eq!(
                got,
                elem_dofs[e].as_slice(),
                "{mesh_name} p={p} element {e}: GetElementDofs table differs (global ids)"
            );
        }
    }
}

#[test]
fn d842_cart222_hex_h1_numbering_matches_mfem_all_orders() {
    check_mesh("cart222", CART222);
}

#[test]
fn d842_scr222_hex_h1_numbering_matches_mfem_all_orders() {
    // The discriminator for the dof-table convention: vertex ids reversed so
    // most local CUBE::Edges directions descend.
    check_mesh("scr222", SCR222);
}

#[test]
fn d842_uniform_refined_inline_hex_matches_mfem_numbering() {
    // The red→green core of round 92's D842-3: one `refine_uniform_3d` on the
    // ex26 coarse mesh must reproduce MFEM `Mesh::UniformRefinement`'s
    // global-phase new-vertex numbering (all edge midpoints in mesh-edge
    // order, then all face centers, then all element centers) — byte-for-byte
    // against MFEM's own `Mesh::Print`, which also pins the hex writer
    // format.  (Pre-fix this file diffed in every refined element row; see
    // `tmp/d92a/refined1_mesh_diff.txt`.)
    let coarse_path = std::env::temp_dir().join("d842_pin_inline_coarse.mesh");
    std::fs::write(&coarse_path, INLINE_COARSE).unwrap();
    let mesh = read_mfem_file(&coarse_path)
        .expect("read coarse")
        .mesh3d
        .expect("3-D mesh");
    assert_eq!(mesh.n_elements(), 64);
    let refined = refine_uniform_3d(&mesh);
    assert_eq!(refined.n_elements(), 512);
    assert_eq!(refined.n_nodes(), 729); // 125 + 300 edge mids + 240 face ctrs + 64 centers

    let out_path = std::env::temp_dir().join("d842_pin_inline_ref1_rs.mesh");
    write_mfem_file_3d(&out_path, &refined).expect("write refined");
    let got = std::fs::read_to_string(&out_path).expect("read back");
    let want = INLINE_REF1_MFEM.trim_end_matches(['\n', '\r']);
    let got = got.trim_end_matches(['\n', '\r']);
    assert_eq!(got, want, "refined inline-hex mesh differs from MFEM's print");
}
