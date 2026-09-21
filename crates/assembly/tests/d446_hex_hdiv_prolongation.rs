//! D446 — `build_prolongation_hdiv` must walk HDiv face blocks by the face's
//! **shape**, not by a global triangular stride.
//!
//! Debt: `hdiv_face_dofs_per_face(dim, order)` sized every 3-D face block
//! `(k+1)(k+2)/2` (the triangular size) and the 3-D walk enumerated only the
//! tet's local face table.  After D377/D393 made `HDivSpace::face_dofs` size
//! blocks by shape — a quadrilateral face carries `(k+1)^2` — the
//! prolongation on any mesh with quad faces (hex/prism/pyramid/mixed)
//! mis-walked: on a pure hex hierarchy the tet-tuple `FaceKey`s degenerate to
//! one accidental bottom-face hit per element, the sub-face fallback never
//! matches, and the returned P is **empty** even though every coarse face dof
//! has fine children (RT1 quad faces: 4 dofs, not 3).
//!
//! Fix (same shape rule as `HDivSpace`): enumerate each element's faces from
//! its element type (3-entry slice = tri face, 4-entry = quad), build the
//! quad `FaceKey` as the sorted first 3 of the 4 verts (the HDivSpace
//! builders' rule), and take the per-face block size from the face's vertex
//! count.  Sub-face matching extends by the coarse face's own shape (tri:
//! edge midpoints; quad: edge midpoints + face center).
//!
//! The original audit re-derived the expected structure from the meshes alone
//! and asserted, for every fine face that is a coarse face or one of its
//! uniform-refinement sub-faces:
//!   1. every dof of its `face_dofs` block carries exactly one P entry, in
//!      the same block offset, with value `ratio × visits` (ratio 1.0 for an
//!      unrefined face, the 0.25 area ratio for a sub-face; `visits` = the
//!      number of fine elements sharing the face — P is piecewise identity
//!      per face block);
//!   2. every coarse face dof is claimed by at least one entry (Pᵀ covers
//!      all face blocks), while coarse interior dofs receive none (the
//!      documented higher-order approximation);
//!   3. every nonzero P row's support lies inside a single coarse element's
//!      dof set (Pᵀ single-parent partition, d386 style);
//!   4. fine faces strictly inside a coarse element stay zero (new flux
//!      information has no coarse counterpart).
//!
//! ── Status (round 55): documentation-only target ───────────────────────────
//!
//! The structural audit has been fully superseded: every hierarchy it covered
//! now takes the MFEM-exact `LocalInterpolation_RT` path, so the structural
//! walk assertions no longer describe the truth, and their stronger bitwise
//! replacements live in `d468_hdiv_prolongation_mfem_parity.rs` and
//! `d493_rt1_prolongation_mfem_parity.rs` (see the four notes below).  Under
//! the zero-dead-code rule the audit machinery — `Qc`, `q`, `elem_faces`,
//! `face_key`, `mesh_faces`, `check_hierarchy`, ~270 lines with zero callers
//! and zero `#[test]`s in the file — was deleted rather than left as an empty
//! target carrying dead code.  The notes below are kept as the supersession
//! record for D446/D468/D481/D482/D494.

// HEX hierarchy, RT1: (D494, round 54) hex RT1 prolongation now uses the
// MFEM-exact `LocalInterpolation_RT` path via the published order-1 nodal
// table (`hex_rt1::mfem_hex_nodal_dofs`) — dense interpolation rows for
// every fine dof, bubble rows included — so the legacy block-identity
// structural walk pin no longer describes the truth (same supersession as
// the hex RT0 pin above).  The stronger bitwise pin lives in
// tests/d493_rt1_prolongation_mfem_parity.rs::d493_hex_rt1_matches_mfem
// (1728 MFEM entries, max|delta| 0e0) plus the semantic companion
// d493_hex_rt1_constant_field_prolongs_exactly.  The superseded legacy pin
// was removed rather than ignored, per this file's round-52 precedent.

// HEX hierarchy, RT0: block size 1 hides a wrong *stride* but not a wrong
// *walk* — the old tet-only face enumeration still returned an empty P here.
// (D468, round 52: hex RT0 prolongation now uses the MFEM-exact
// `LocalInterpolation_RT` semantics — dense midline rows, single-write sub-face
// ratios — so the old structural walk pin no longer described the truth; the
// stronger bitwise pin lives in
// tests/d468_hdiv_prolongation_mfem_parity.rs::d468_hex_rt0_matches_mfem,
// which also asserts the sparse structure bidirectionally.  The superseded
// legacy pin was removed rather than ignored.)

// TET hierarchy, RT1: (D494, round 54) tet RT1 on fem-rs's own meshes now
// takes the MFEM-exact `LocalInterpolation_RT` path too — the mirrored
// corner children are served through their own (negative-determinant)
// frames, whose mesh slot signs already carry the orientation — so dense
// interpolation rows replace the legacy block-identity structure and this
// old structural pin no longer describes the truth.  The stronger pins:
// tests/d468_hdiv_prolongation_mfem_parity.rs::d461_tet_rt1_matches_mfem
// (bitwise on MFEM's construction) and
// tests/d493_rt1_prolongation_mfem_parity.rs::
// d493_tet_rt1_own_mesh_exact_path_and_exact_fields (exact path served on
// this very hierarchy; constant and linear fields prolong exactly).  The
// superseded legacy pin was removed rather than ignored, per this file's
// round-52 precedent.

// TET/PRISM hierarchy, RT0: (D481/D482, round 53) both geometries now use the
// MFEM-exact `LocalInterpolation_RT` semantics — mirrored-sliver frames on the
// tet, the Prism family slot rows on the wedge — so midline/interior sub-faces
// carry dense interpolation rows and the old "unparented ⇒ empty row" /
// block-identity structural pins no longer describe the truth.  The stronger
// bitwise pins live in tests/d468_hdiv_prolongation_mfem_parity.rs
// (tet_rt0 264/264 rows at 5.551e-17, prism_rt0 88/88 rows bitwise), together
// with the constant-field P·x_c ≡ x_f semantics tests.  The superseded legacy
// pins were removed rather than ignored.
