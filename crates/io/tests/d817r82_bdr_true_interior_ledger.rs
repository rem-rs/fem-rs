//! D817-1 — full-corpus ledger for **interior-coincident boundary entries**.
//!
//! MFEM's `Mesh::GetBdrFaceTransformations` (`mesh/mesh.cpp:1312`) answers
//! INVALID for a boundary element whose face is a *true interior* face
//! (`FaceIsTrueInterior` = both sides registered in this mesh — the shape of
//! a periodically identified seam kept in the file's `boundary` section), and
//! every bdr-face integration in `BilinearForm`/`LinearForm`/`NonlinearForm`
//! skips the resulting nullptr: **zero boundary assembly** on such entries.
//! The fem-rs sink is `MeshTopology::bdr_face_true_interior`, applied at every
//! bdr-assembly traversal (D817-1).
//!
//! The golden account below was taken with an MFEM 4.10 probe over the whole
//! `data/` corpus (`$HOME/work/d817/d817_probe.cpp`, account dump
//! `tmp/d817/golden_account.txt`): exactly **four** of the 106 meshes carry
//! interior-coincident boundary entries; every other readable mesh counts 0.
//! `d445_one_pyramid.mesh` is asserted 0 from the fem-rs side alone — MFEM
//! 4.10 itself SIGSEGVs loading that file (the D113-2 pyramid fragility
//! family), so no MFEM truth exists for it.
//!
//! This ledger is the forward guard for the whole corpus: a later change to
//! the detection (or to the skip sites) has to move these counts deliberately.

use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;
use std::path::{Path, PathBuf};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

/// MFEM-probe golden: files whose `boundary` section lists interior-coincident
/// entries, with the probed count (NBE in parentheses).
const MFEM_TRUE_INTERIOR: &[(&str, u32)] = &[
    ("d525_pyramid_pair.mesh", 2),   // NBE=10: the glued base faces
    ("d667_refined_curved.mesh", 96), // NBE=352: refined curved hex interfaces
    ("multidomain-hex.mesh", 24),    // NBE=88: the subdomain interface faces
    ("periodic-cube.mesh", 54),      // NBE=54: every stored entry (ex9-3D)
];

/// Files the reader legitimately refuses — same rows as the D812-1 write
/// ledger's `read_err` status (nc_mesh / nurbs patches).  Counted as *n/a*
/// here, not 0: no fem-rs boundary list exists to test.
const READER_REFUSES: &[&str] = &[
    "amr-hex.mesh",
    "amr-quad.mesh",
    "beam-quad-amr.mesh",
    "fichera-amr.mesh",
    "nc-nurbs3d.mesh",
    "nc3-nurbs.mesh",
    "square-disc-nurbs-patch.mesh",
];

#[test]
fn d817r82_bdr_true_interior_matches_the_mfem_probe() {
    let mut files: Vec<PathBuf> = std::fs::read_dir(data_dir())
        .expect("data/ directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "mesh"))
        .collect();
    files.sort();
    assert!(
        files.len() >= 100,
        "expected the full data/ corpus (106 meshes), saw {}",
        files.len()
    );

    let refuses: Vec<&str> = READER_REFUSES.to_vec();
    let mut wrong: Vec<String> = Vec::new();
    let mut unexpected_skip: Vec<String> = Vec::new();
    let mut counted = 0usize;
    let mut skipped = 0usize;

    for p in &files {
        let name = p.file_name().unwrap().to_str().unwrap().to_string();
        if refuses.contains(&name.as_str()) {
            skipped += 1;
            continue;
        }
        let mfem = read_mfem_file(p).unwrap_or_else(|e| panic!("{name}: reader refused: {e}"));
        let count = mfem
            .mesh1d
            .as_ref()
            .map(|m| (0..m.n_boundary_faces() as u32).filter(|&f| m.bdr_face_true_interior(f)).count())
            .or_else(|| {
                mfem.mesh2d.as_ref().map(|m| {
                    (0..m.n_boundary_faces() as u32).filter(|&f| m.bdr_face_true_interior(f)).count()
                })
            })
            .or_else(|| {
                mfem.mesh3d.as_ref().map(|m| {
                    (0..m.n_boundary_faces() as u32).filter(|&f| m.bdr_face_true_interior(f)).count()
                })
            })
            .unwrap_or_else(|| panic!("{name}: no mesh dimension parsed"));
        counted += 1;
        let expected = MFEM_TRUE_INTERIOR
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, c)| *c as usize)
            .unwrap_or(0);
        if count != expected {
            wrong.push(format!("{name}: got {count}, expected {expected}"));
        }
    }
    assert_eq!(skipped, refuses.len(), "some refused files vanished from data/");
    assert!(
        counted + skipped == files.len(),
        "unaccounted files: {:?}",
        files.len() - counted - skipped
    );
    if !unexpected_skip.is_empty() {
        panic!("{unexpected_skip:?}");
    }
    assert!(
        wrong.is_empty(),
        "true-interior counts diverge from the MFEM probe:\n{}",
        wrong.join("\n")
    );
}
