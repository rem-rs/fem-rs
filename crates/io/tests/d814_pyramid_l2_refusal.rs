//! D813-5 — the pyramid `L2_T1_3D_P*` numbering: known, pinned, and *refused
//! at the writer* for the right reason.
//!
//! Round 39 closed D325/D335 with "MFEM itself rejects the element" — but the
//! premise probe (`$HOME/work/d325/dginv2_out.txt`) shows the rejection was
//! `FiniteElement::GetDofToQuad`'s MFEM_ABORT for the **TENSOR mode**, not the
//! element: `L2_FECollection(p, 3, GaussLobatto).FiniteElementForGeometry(
//! Geometry::PYRAMID)` returns a fully working `L2_FuentesPyramidElement`
//! (`dof = (p+1)³`, `GetNodes()` and FULL dof-to-quad fine —
//! `$HOME/work/d80b/pyr_probe.cpp`).  So there is no "rejection parity" to
//! pin: the honest closure is the **real numbering**, which is now in
//! `mfem_l2_slots` and pinned per slot in `crates/io/src/mfem.rs`'s unit test
//! `d814_pyramid_l2_nodes_match_mfem_in_order` against MFEM's own
//! `GetNodes()` dump (`tests/fixtures/d814_mfem_l2pyr_nodes_p1to3.txt`).
//!
//! What cannot close yet is the *mesh-side* half: fem-rs's pyramid geometry
//! element is the Bergot family with `(p+1)(p+2)(2p+3)/6` dofs, while MFEM's
//! L2 pyramid table has `(p+1)³` — no permutation can map one onto the other
//! (D306, `fem_element`, outside this crate's lane).  This file pins the
//! resulting behaviour end to end: a pyramid mesh with an L2-shaped geometry
//! table is refused **loudly, naming the real reason**, and nothing is
//! written — never silently mis-numbered.

use fem_io::mfem::{read_mfem_file, write_mfem_nodes, NodesSpace};
use fem_mesh::simplex::{GeometryData, Mesh};
use std::path::{Path, PathBuf};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

/// A straight unit-pyramid mesh (`data/d525_unit_pyramid.mesh`) carrying a
/// hand-built *L2-shaped* geometry table: order 1 with 8 private dofs per
/// element — the count MFEM's `L2_FuentesPyramidElement(1)` has (the values
/// themselves are irrelevant; the refusal must fire on the slot mapping).
#[test]
fn d814_pyramid_l2_table_is_refused_with_the_real_reason() {
    let file = read_mfem_file(data_dir().join("d525_unit_pyramid.mesh")).expect("read");
    let mut mesh = file.mesh3d.expect("3-D pyramid mesh");
    assert_eq!(mesh.n_elems(), 1, "one pyramid");

    let npe = 8usize; // (p+1)³ at p = 1 — the Fuentes lattice size
    mesh.geometry = Some(GeometryData {
        order: 1,
        conn: (0..npe as u32).collect(),
        nodes_per_elem: npe,
        coords: vec![0.0; npe * 3],
        n_nodes: npe,
    });

    let mut buf: Vec<u8> = Vec::new();
    // The same 2-D placeholder `write_mfem_file_3d_nodes` uses.
    let err = write_mfem_nodes(
        &mut buf,
        &Mesh::<2>::unit_square_tri(2),
        Some(&mesh),
        NodesSpace::Discontinuous,
    )
    .expect_err("a pyramid L2 table must be refused while the mesh-side lattice is Bergot");
    let msg = err.to_string();
    assert!(
        msg.contains("L2 Fuentes pyramid") && msg.contains("D306"),
        "the refusal must name the real reason (MFEM numbering known, mesh-side \
         Bergot lattice not mappable — D306), got: {msg}"
    );
    assert!(buf.is_empty(), "a refused write must emit nothing");
}
