//! Low-order refined (LOR) discretizations for H(curl), H(div) and H1 spaces
//! on tensor-product (quad / hex) meshes — the counterpart of MFEM's
//! `LORDiscretization` for `ND_FECollection` / `RT_FECollection` / `H1_FECollection`
//! (`fem/lor/lor.hpp`, `fem/lor/lor.cpp`, `fem/lor/lor_h1.hpp`).
//!
//! # What is built
//!
//! For a high-order (HO) space of order `p` on a quad/hex mesh, [`LorNd`]
//! (H(curl)) / [`LorRt`] (H(div)) provide
//!
//! 1. the LOR mesh — each HO element subdivided into `ref^dim` sub-elements
//!    at the Gauss-Lobatto nodes (`Mesh::MakeRefined`, here
//!    [`make_refined_2d`]/[`make_refined_3d`]), where `ref = p` for ND
//!    (LOR space = ND1) and `ref = p + 1` for RT (LOR space = RT0) — exactly
//!    MFEM's `LORBase::GetLOROrder` / `LORDiscretization::FormLORSpace`
//!    refinement choice;
//! 2. the LOR space itself — ND1 / RT0 on the refined mesh;
//! 3. `perm` — the signed dof permutation `perm[i] = ±j` mapping LOR dof `i`
//!    to the HO dof `j` it corresponds to ("assumed constraint" map, MFEM
//!    `LORBase::ConstructLocalDofPermutation`).  For tensor-product ND/RT
//!    spaces the LOR and HO dof sets are in bijection and the map is a signed
//!    permutation (verified at construction time).
//!
//! # Scalar / vector H1 (elasticity)
//!
//! [`LorH1`] is the counterpart for `H1_FECollection` (MFEM
//! `LORBase::ConstructDofPermutation` returns the identity for H1/L2 — the
//! refined mesh vertices *are* the H1 dofs, in H1 dof order, so LOR and HO
//! dof sets correspond 1:1).  In fem-rs the refined-mesh vertex numbering
//! (MFEM `MakeRefined_` order, see [`crate::make_refined`]) can differ from
//! the `DofManager` H1 numbering (edge-DOF directions), so [`LorH1`] builds
//! the (unsigned) renumbering permutation `perm[i] = j` explicitly and
//! validates it as a bijection.
//!
//! A vector H1 space ([`crate::vector_h1::VectorH1Space`], MFEM
//! `Ordering::byNODES`) has LOR = per-component scalar H1 LOR (block
//! structure): vector LOR dof `(c, i)` corresponds to HO vector dof
//! `(c, perm[i])` — see [`LorH1::vector_perm`].  This is the discretization
//! used by MFEM `miniapps/solvers/lor_elast.cpp` (block-diagonal LOR-AMG
//! elasticity preconditioning).
//!
//! # Scope and limitations
//!
//! - quad meshes (2-D) and hex meshes (3-D), single element type, uniform
//!   order; the LOR ND1/RT0 dof count must equal the HO dof count (checked);
//! - `ref = 1` (ND1 / RT0 HO spaces) gives the identity permutation and the
//!   unrefined mesh, matching MFEM;
//! - signs are the same-functional relative orientations of [`pair_sign`]
//!   (MFEM's `ConstructLocalDofPermutation` sign-product semantics); the
//!   hex interior slot tables follow the actual `HexNDk`/`HexRTk` layouts;
//! - the perm-transfer preconditioner (`A_LOR ≈ Πᵀ A_HO Π`) is only
//!   h/p-robust for MFEM's LOR-compatible basis pair (GaussLobatto,
//!   IntegratedGLL) — MFEM's own default (…, GaussLegendre) vector basis
//!   shows the same refinement growth (see the D40 note in
//!   `fem-assembly::lor_factory`).
//!
//! # Use as a preconditioner
//!
//! Assemble the same bilinear form on `lor_space()` (ND1 curl-curl + vector
//! mass, or RT0 div-div + vector mass), reorder it into the HO numbering with
//! [`LorNd::ho_numbering`]/[`LorRt::ho_numbering`] and use the result as the
//! AMG/V-cycle preconditioner operator for PCG on the HO system — the
//! `LORSolver` pattern of MFEM `miniapps/solvers/plor_solvers.cpp`.
//! `fem-assembly`'s `lor_factory` module wires this end to end.

use std::collections::HashMap;

use fem_core::{FemError, FemResult};
use fem_element::raviart_thomas::HEX_RT_FACES;
use fem_linalg::CsrMatrix;
use fem_mesh::simplex::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_mesh::ElementType;

use crate::dof_manager::{DofManager, EdgeKey};
use crate::fe_space::FESpace;
use crate::hcurl::HCurlSpace;
use crate::hdiv::HDivSpace;
use crate::make_refined::{make_refined_2d, make_refined_3d};

// ─── Shared tables ──────────────────────────────────────────────────────────

/// MFEM `Geometry::Constants<Geometry::CUBE>::Edges` (= `HCurlSpace::HEX_EDGES`).
const HEX_EDGES: [(usize, usize); 12] = [
    (0, 1), (1, 2), (3, 2), (0, 3),
    (4, 5), (5, 6), (7, 6), (4, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
];

/// Hex corner lattice offsets (MFEM vertex order: bottom CCW then top CCW).
const HEX_CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];

/// MFEM `Geometry::Constants<Geometry::CUBE>::FaceVert` (= `HDivSpace::HEX_FACES`):
/// bottom z−, front y−, right x+, back y+, left x−, top z+.
const HEX_FACES: [[usize; 4]; 6] = [
    [3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5],
    [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7],
];

/// Quad edges in local connectivity order (= `HCurlSpace::QUAD_EDGES` /
/// `HDivSpace::QUAD_FACES`): bottom, right, top, left.
const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

/// Quad corner lattice offsets (local connectivity order).
const QUAD_CORNERS: [[usize; 2]; 4] = [[0, 0], [1, 0], [1, 1], [0, 1]];

// ─── LOR H(curl) ────────────────────────────────────────────────────────────

/// Low-order refined H(curl) discretization of an ND space on a quad/hex mesh.
///
/// See the [module docs](self) for the construction and MFEM references.

pub struct LorNd<const D: usize> {
    lor_mesh: Mesh<D>,
    lor_space: HCurlSpace<Mesh<D>>,
    /// `perm[i] = s * j`: LOR dof `i` corresponds to HO dof `j` with sign
    /// `s = ±1` (never zero; a full bijection onto `0..n_ho`).
    perm: Vec<i32>,
    n_ho: usize,
    refinement: usize,
}

impl<const D: usize> LorNd<D> {
    /// The refined (LOR) mesh.
    pub fn lor_mesh(&self) -> &Mesh<D> { &self.lor_mesh }

    /// ND1 space on the LOR mesh.
    pub fn lor_space(&self) -> &HCurlSpace<Mesh<D>> { &self.lor_space }

    /// Signed permutation `perm[i] = ±j` (LOR dof → HO dof).
    pub fn perm(&self) -> &[i32] { &self.perm }

    /// Number of HO dofs (== number of LOR dofs).
    pub fn n_ho(&self) -> usize { self.n_ho }

    /// Refinement factor per direction (`p` for ND).
    pub fn refinement(&self) -> usize { self.refinement }

    /// Reorder an LOR-numbered matrix into HO dof numbering:
    /// `B[|perm[i]|, |perm[j]|] = perm[i]·perm[j]·A[i][j]`.
    pub fn ho_numbering(&self, a_lor: &CsrMatrix<f64>) -> CsrMatrix<f64> {
        assert_eq!(a_lor.nrows, a_lor.ncols, "ho_numbering: matrix must be square");
        assert_eq!(a_lor.nrows, self.perm.len(), "ho_numbering: matrix size mismatch");
        let mut coo = fem_linalg::CooMatrix::<f64>::new(self.n_ho, self.n_ho);
        for i in 0..a_lor.nrows {
            let pi = self.perm[i] as isize;
            let si = if pi < 0 { -1.0_f64 } else { 1.0 };
            let gi = pi.unsigned_abs() as usize;
            for r in a_lor.row_ptr[i]..a_lor.row_ptr[i + 1] {
                let j = a_lor.col_idx[r] as usize;
                let pj = self.perm[j] as isize;
                let sj = if pj < 0 { -1.0_f64 } else { 1.0 };
                let gj = pj.unsigned_abs() as usize;
                let v = a_lor.values[r];
                if v != 0.0 {
                    coo.add(gi, gj, si * sj * v);
                }
            }
        }
        coo.into_csr()
    }

    /// Prolongate: `x_ho[|perm[i]|] = perm[i]·x_lor[i]`.
    pub fn prolongate(&self, x_lor: &[f64], x_ho: &mut [f64]) {
        assert_eq!(x_lor.len(), self.perm.len());
        assert_eq!(x_ho.len(), self.n_ho);
        for (i, &p) in self.perm.iter().enumerate() {
            let s = if p < 0 { -1.0_f64 } else { 1.0 };
            x_ho[p.unsigned_abs() as usize] = s * x_lor[i];
        }
    }

    /// Restrict: `x_lor[i] = perm[i]·x_ho[|perm[i]|]` (transpose of [`LorNd::prolongate`]).
    pub fn restrict(&self, x_ho: &[f64], x_lor: &mut [f64]) {
        assert_eq!(x_lor.len(), self.perm.len());
        assert_eq!(x_ho.len(), self.n_ho);
        for (i, &p) in self.perm.iter().enumerate() {
            let s = if p < 0 { -1.0_f64 } else { 1.0 };
            x_lor[i] = s * x_ho[p.unsigned_abs() as usize];
        }
    }

    fn finish(
        k: usize,
        lor_mesh: Mesh<D>,
        lor_space: HCurlSpace<Mesh<D>>,
        perm: Vec<i32>,
        n_ho: usize,
    ) -> FemResult<Self> {
        // Bijectivity validation (mirrors the MFEM assumed-constraint map).
        if perm.len() != lor_space.n_dofs() {
            return Err(FemError::Other(
                "LOR: permutation size does not match LOR dof count".into(),
            ));
        }
        let mut seen_ho = vec![false; n_ho];
        for &p in &perm {
            let a = p.unsigned_abs() as usize;
            if a >= n_ho || seen_ho[a] {
                return Err(FemError::Other(format!(
                    "LOR: permutation is not a bijection (entry {p})"
                )));
            }
            seen_ho[a] = true;
        }
        Ok(LorNd { lor_mesh, lor_space, perm, n_ho, refinement: k })
    }
}

/// LOR H(div) discretization of an RT space on a quad/hex mesh.
///
/// Refinement factor is `p + 1` (MFEM `RT_FECollection` reports order `p+1`);
/// the LOR space is RT0 on the refined mesh.

pub struct LorRt<const D: usize> {
    lor_mesh: Mesh<D>,
    lor_space: HDivSpace<Mesh<D>>,
    perm: Vec<i32>,
    n_ho: usize,
    refinement: usize,
}

impl<const D: usize> LorRt<D> {
    /// The refined (LOR) mesh.
    pub fn lor_mesh(&self) -> &Mesh<D> { &self.lor_mesh }

    /// RT0 space on the LOR mesh.
    pub fn lor_space(&self) -> &HDivSpace<Mesh<D>> { &self.lor_space }

    /// Signed permutation `perm[i] = ±j` (LOR dof → HO dof).
    pub fn perm(&self) -> &[i32] { &self.perm }

    /// Number of HO dofs (== number of LOR dofs).
    pub fn n_ho(&self) -> usize { self.n_ho }

    /// Refinement factor per direction (`p + 1` for RT).
    pub fn refinement(&self) -> usize { self.refinement }

    /// Reorder an LOR-numbered matrix into HO dof numbering (see [`LorNd::ho_numbering`]).
    pub fn ho_numbering(&self, a_lor: &CsrMatrix<f64>) -> CsrMatrix<f64> {
        assert_eq!(a_lor.nrows, a_lor.ncols, "ho_numbering: matrix must be square");
        assert_eq!(a_lor.nrows, self.perm.len(), "ho_numbering: matrix size mismatch");
        let mut coo = fem_linalg::CooMatrix::<f64>::new(self.n_ho, self.n_ho);
        for i in 0..a_lor.nrows {
            let pi = self.perm[i] as isize;
            let si = if pi < 0 { -1.0_f64 } else { 1.0 };
            let gi = pi.unsigned_abs() as usize;
            for r in a_lor.row_ptr[i]..a_lor.row_ptr[i + 1] {
                let j = a_lor.col_idx[r] as usize;
                let pj = self.perm[j] as isize;
                let sj = if pj < 0 { -1.0_f64 } else { 1.0 };
                let gj = pj.unsigned_abs() as usize;
                let v = a_lor.values[r];
                if v != 0.0 {
                    coo.add(gi, gj, si * sj * v);
                }
            }
        }
        coo.into_csr()
    }

    /// Prolongate: `x_ho[|perm[i]|] = perm[i]·x_lor[i]`.
    pub fn prolongate(&self, x_lor: &[f64], x_ho: &mut [f64]) {
        assert_eq!(x_lor.len(), self.perm.len());
        assert_eq!(x_ho.len(), self.n_ho);
        for (i, &p) in self.perm.iter().enumerate() {
            let s = if p < 0 { -1.0_f64 } else { 1.0 };
            x_ho[p.unsigned_abs() as usize] = s * x_lor[i];
        }
    }

    /// Restrict (transpose of [`LorRt::prolongate`]).
    pub fn restrict(&self, x_ho: &[f64], x_lor: &mut [f64]) {
        assert_eq!(x_lor.len(), self.perm.len());
        assert_eq!(x_ho.len(), self.n_ho);
        for (i, &p) in self.perm.iter().enumerate() {
            let s = if p < 0 { -1.0_f64 } else { 1.0 };
            x_lor[i] = s * x_ho[p.unsigned_abs() as usize];
        }
    }
}

// ─── 3-D ND (hex) ───────────────────────────────────────────────────────────

impl LorNd<3> {
    /// LOR discretization of a hex ND space (order >= 1, uniform order, all
    /// elements `Hex8`).
    pub fn new_hex(ho: &HCurlSpace<Mesh<3>>) -> FemResult<Self> {
        let mesh = ho.mesh().clone();
        let k = ho.order() as usize;
        check_uniform(&mesh, ElementType::Hex8, "LOR ND 3D")?;

        let (lor_mesh, lor_space) = if k == 1 {
            (mesh.clone(), HCurlSpace::new(mesh.clone(), 1))
        } else {
            let lm = make_refined_3d(&mesh, k);
            let ls = HCurlSpace::new(lm.clone(), 1);
            (lm, ls)
        };

        let perm = if k == 1 {
            (0..lor_space.n_dofs() as i32).collect::<Vec<i32>>()
        } else {
            build_nd_perm_3d(&mesh, ho, &lor_space, k)?
        };

        Self::finish(k, lor_mesh, lor_space, perm, ho.n_dofs())
    }
}

fn check_uniform<const D: usize>(mesh: &Mesh<D>, et: ElementType, what: &str) -> FemResult<()> {
    if mesh.n_elements() == 0 {
        return Err(FemError::Other(format!("{what}: empty mesh")));
    }
    for e in 0..mesh.n_elements() as u32 {
        if mesh.element_type(e) != et {
            return Err(FemError::Other(format!(
                "{what}: only {et:?} meshes are supported (element {e} is {:?})",
                mesh.element_type(e)
            )));
        }
    }
    Ok(())
}

/// Canonical-orientation sign of the paired dofs: the product
/// `sign(σ^HO_j(c)) · sign(σ^LOR_i(c))` of both spaces' canonical functionals
/// evaluated on the constant field `c ≡ (1,…,1)` (`xh_one` / `xl_one`, from
/// the spaces' canonical `interpolate_vector`).  On lattice meshes every
/// ND/RT functional reduces to `±(F·e_axis)·(positive length)` for `c`, so the
/// signs are nonzero and the product is exactly the relative orientation of
/// the two canonical dofs — the fem-rs counterpart of the
/// `s1·s2·s3·s4` sign product MFEM accumulates in
/// `LORBase::ConstructLocalDofPermutation` (dof-map signs × vdof signs).
///
/// This is what makes the LOR/HO dof pairs *same-functional* (MFEM `fem/lor/`
/// convention): with this sign the transfer `x_HO[|perm[i]|] = s·x_LOR[i]`
/// maps the LOR coefficients of any field onto the corresponding HO
/// coefficients (constant fields map with the exact ratio `k`).
fn pair_sign(xh_one: &[f64], xl_one: &[f64], ho_dof: usize, lor_dof: usize) -> f64 {
    xh_one[ho_dof].signum() * xl_one[lor_dof].signum()
}

/// Signed permutation for hex ND: LOR ND1 dof → HO ND_k dof.
///
/// Lattice bookkeeping per macro element (local parametrization, indices
/// `a` along the edge direction, the other two across):
/// - x-family lattice d-edge `(a, b, c)`: `a ∈ [0,k)`, `b, c ∈ [0,k]`;
/// - macro mesh edges carry `k` modes (interval `m` counted from the
///   element's parametric origin — all `CUBE::Edges` point +lattice);
/// - macro mesh faces carry `2k(k-1)` dofs in two tangent groups
///   (`HexNDk` face-block order matches `HCurlSpace::HEX_QUAD_FACES`);
/// - element interiors carry `3k(k-1)²` dofs (x/y/z blocks).
fn build_nd_perm_3d(
    mesh: &Mesh<3>,
    ho: &HCurlSpace<Mesh<3>>,
    lor: &HCurlSpace<Mesh<3>>,
    k: usize,
) -> FemResult<Vec<i32>> {
    let n_elem = mesh.n_elements() as u32;
    let n_lor = lor.n_dofs();
    let n_ho = ho.n_dofs();
    if n_lor != n_ho {
        return Err(FemError::Other(format!(
            "LOR ND 3D: dof count mismatch (LOR {n_lor} != HO {n_ho}); \
             the assumed-constraint bijection requires equal counts"
        )));
    }

    // Macro mesh edge ownership: EdgeKey -> (owner element, local edge index).
    let mut edge_owner: HashMap<EdgeKey, (u32, usize)> = HashMap::new();
    for e in 0..n_elem {
        let verts = mesh.element_nodes(e);
        for (lei, &(li, lj)) in HEX_EDGES.iter().enumerate() {
            let key = EdgeKey::new(verts[li], verts[lj]);
            edge_owner.entry(key).or_insert((e, lei));
        }
    }
    // Macro mesh face ownership: sorted 4-vertex key -> owner element.
    let mut face_owner: HashMap<[u32; 4], u32> = HashMap::new();
    for e in 0..n_elem {
        let verts = mesh.element_nodes(e);
        for fq in HEX_FACES.iter() {
            let mut key = [
                verts[fq[0]], verts[fq[1]], verts[fq[2]], verts[fq[3]],
            ];
            key.sort_unstable();
            face_owner.entry(key).or_insert(e);
        }
    }

    // Hand-transcribed HO dof → lattice-edge tables (see the slot arms in
    // the loop below).

    let mut perm = vec![0i32; n_lor];
    let mut filled = vec![false; n_lor];
    let k3 = k * k * k;

    // Canonical values of both spaces on the constant unit field — the
    // same-functional orientation signs ([`pair_sign`]).
    let unit = |_: &[f64]| vec![1.0_f64; 3];
    let xh_one = ho.interpolate_vector(&unit);
    let xl_one = lor.interpolate_vector(&unit);
    let xh_one = xh_one.as_slice();
    let xl_one = xl_one.as_slice();

    /// HCurl face-block index for the macro face fixed on axis `axis_fixed`
    /// at plane 0 (`false`) or k (`true`).  Block order: bottom z−, top z+,
    /// front y−, back y+, left x−, right x+.
    fn macro_face_idx(axis_fixed: usize, plane_high: bool) -> usize {
        match (axis_fixed, plane_high) {
            (0, false) => 4, (0, true) => 5,
            (1, false) => 2, (1, true) => 3,
            (2, false) => 0, (2, true) => 1,
            _ => unreachable!(),
        }
    }

    fn macro_edge_idx(axis: usize, o1: usize, o2: usize) -> usize {
        match (axis, o1, o2) {
            (0, 0, 0) => 0, (0, 1, 0) => 2, (0, 0, 1) => 4, (0, 1, 1) => 6,
            (1, 0, 0) => 3, (1, 1, 0) => 1, (1, 0, 1) => 7, (1, 1, 1) => 5,
            (2, 0, 0) => 8, (2, 1, 0) => 9, (2, 1, 1) => 10, (2, 0, 1) => 11,
            _ => unreachable!(),
        }
    }

    for eh in 0..n_elem {
        let verts = mesh.element_nodes(eh);
        let hd = ho.element_dofs(eh);

                for kz in 0..k {
                    for ky in 0..k {
                        for kx in 0..k {
                            let lor_elem = eh * (k3 as u32) + (kx + k * ky + k * k * kz) as u32;
                            let ld = lor.element_dofs(lor_elem);

                            for (sei, &(si, sj)) in HEX_EDGES.iter().enumerate() {
                        // Determine edge axis and lattice coordinates.
                        let c0 = HEX_CORNERS[si];
                        let c1 = HEX_CORNERS[sj];
                        let axis = (0..3).find(|&d| c0[d] != c1[d]).unwrap();
                        let base = [kx, ky, kz];
                        let a = base[axis]; // interval index along the axis
                        let (d1, d2) = match axis {
                            0 => (1usize, 2usize),
                            1 => (0usize, 2usize),
                            _ => (0usize, 1usize),
                        };
                        let i1 = base[d1] + c0[d1]; // cross coord 1 (absolute)
                        let i2 = base[d2] + c0[d2]; // cross coord 2 (absolute)
                        let bnd1 = i1 == 0 || i1 == k;
                        let bnd2 = i2 == 0 || i2 == k;

                        // Sign of the map: the relative orientation of the two
                        // canonical functionals (MFEM's same-functional sign
                        // product), measured on the constant unit field.
                        let n_edge_dofs = 12 * k;
                        let n_face_blk = 2 * k * (k - 1);
                        let int_base = n_edge_dofs + 6 * n_face_blk;
                        let slot = if bnd1 && bnd2 {
                            // Macro mesh edge: mode m = interval (all CUBE
                            // edges point +lattice, modes from param origin).
                            let key = macro_edge_key(verts, axis, i1, i2);
                            let (owner, _lei) = edge_owner[&key];
                            if owner != eh {
                                continue; // written by the owner element
                            }
                            let lei = macro_edge_idx(axis, (i1 == k) as usize, (i2 == k) as usize);
                            lei * k + a
                        } else if bnd1 || bnd2 {
                            // Macro mesh face (one cross coordinate on the
                            // boundary): face-interior lattice edge.  Owner
                            // check via the sorted-vertex face key.
                            let face_hcurl = if bnd1 {
                                macro_face_idx(d1, i1 == k)
                            } else {
                                macro_face_idx(d2, i2 == k)
                            };
                            let key = macro_face_key(verts, HCURL_TO_FACEVERT[face_hcurl]);
                            let owner = face_owner[&key];
                            if owner != eh {
                                continue;
                            }
                            // Flat index within the face block.  Each block
                            // holds two tangent groups of k(k-1): group dof
                            // (i-cross-power, j-free-lag) at group*k(k-1) +
                            // i*k + j.  On z-faces the groups are (x, y); on
                            // y-faces (x, z); on x-faces (y, z) — and the
                            // x-face groups are transposed (power on the free
                            // axis, lag on the cross).
                            let blk = face_hcurl * n_face_blk;
                            let flat = match (axis, bnd1, i1, i2) {
                                (0, false, b, _) => (b - 1) * k + a,
                                (0, true, _, c) => (c - 1) * k + a,
                                (1, false, b, _) => k * (k - 1) + (b - 1) * k + a,
                                (1, true, _, c) => (c - 1) * k + a,
                                (2, false, b, _) => k * (k - 1) + (b - 1) * k + a,
                                (2, true, _, c) => k * (k - 1) + (c - 1) * k + a,
                                _ => unreachable!("edge axis is 0, 1 or 2"),
                            };
                            n_edge_dofs + blk + flat
                        } else {
                            // Element-interior lattice edge.  HexNDk interior
                            // block order: for each component the two closed
                            // cross factors run (second-cross outer,
                            // first-cross inner), the open interval innermost
                            // (see `HexNDk::eval_basis_vec`).
                            let flat = match axis {
                                0 => ((i2 - 1) * (k - 1) + (i1 - 1)) * k + a,
                                1 => {
                                    k * (k - 1) * (k - 1)
                                        + ((i2 - 1) * (k - 1) + (i1 - 1)) * k
                                        + a
                                }
                                _ => {
                                    2 * k * (k - 1) * (k - 1)
                                        + ((i2 - 1) * (k - 1) + (i1 - 1)) * k
                                        + a
                                }
                            };
                            int_base + flat
                        };

                        let g_lor = ld[sei] as usize;
                        let s = pair_sign(xh_one, xl_one, hd[slot] as usize, g_lor);
                        let target = (s as isize * hd[slot] as isize) as i32;
                        if filled[g_lor] {
                            if perm[g_lor] != target {
                                return Err(FemError::Other(format!(
                                    "LOR ND 3D: inconsistent mapping for LOR dof {g_lor}"
                                )));
                            }
                        } else {
                            perm[g_lor] = target;
                            filled[g_lor] = true;
                        }
                    }
                }
            }
        }
    }

    if filled.iter().any(|&f| !f) {
        return Err(FemError::Other(
            "LOR ND 3D: some LOR dofs were never mapped (mesh not a lattice?)".into(),
        ));
    }
    Ok(perm)
}

/// Global vertex ids of the macro edge of element `verts` at lattice
/// cross-coordinates `(i1, i2)` (values 0 or k) along `axis`.
fn macro_edge_key(verts: &[u32], axis: usize, i1: usize, i2: usize) -> EdgeKey {
    let (d1, d2) = match axis {
        0 => (1usize, 2usize),
        1 => (0usize, 2usize),
        _ => (0usize, 1usize),
    };
    let corner = |o: usize| -> u32 {
        // Corner lattice offsets are 0/1; normalize the absolute cross coords.
        let mut lat = [0usize; 3];
        lat[axis] = o;
        lat[d1] = if i1 == 0 { 0 } else { 1 };
        lat[d2] = if i2 == 0 { 0 } else { 1 };
        let ci = HEX_CORNERS
            .iter()
            .position(|c| c[0] == lat[0] && c[1] == lat[1] && c[2] == lat[2])
            .unwrap();
        verts[ci]
    };
    EdgeKey::new(corner(0), corner(1))
}

/// `HCurlSpace` face-block index (`HEX_QUAD_FACES`: bottom z−, top z+,
/// front y−, back y+, left x−, right x+) -> MFEM `HEX_FACES` (FaceVert) index.
const HCURL_TO_FACEVERT: [usize; 6] = [0, 5, 1, 3, 4, 2];

/// Sorted 4-vertex key of the macro face with `HEX_FACES` (FaceVert) index
/// `face`.
fn macro_face_key(verts: &[u32], face: usize) -> [u32; 4] {
    let mut key = [
        verts[HEX_FACES[face][0]],
        verts[HEX_FACES[face][1]],
        verts[HEX_FACES[face][2]],
        verts[HEX_FACES[face][3]],
    ];
    key.sort_unstable();
    key
}

// ─── 2-D ND (quad) ──────────────────────────────────────────────────────────

impl LorNd<2> {
    /// LOR discretization of a quad ND space (order >= 1, uniform order, all
    /// elements `Quad4`).
    pub fn new_quad(ho: &HCurlSpace<Mesh<2>>) -> FemResult<Self> {
        let mesh = ho.mesh().clone();
        let k = ho.order() as usize;
        check_uniform(&mesh, ElementType::Quad4, "LOR ND 2D")?;

        let (lor_mesh, lor_space) = if k == 1 {
            (mesh.clone(), HCurlSpace::new(mesh.clone(), 1))
        } else {
            let lm = make_refined_2d(&mesh, k);
            let ls = HCurlSpace::new(lm.clone(), 1);
            (lm, ls)
        };

        let perm = if k == 1 {
            (0..lor_space.n_dofs() as i32).collect::<Vec<i32>>()
        } else {
            build_nd_perm_2d(&mesh, ho, &lor_space, k)?
        };

        Self::finish(k, lor_mesh, lor_space, perm, ho.n_dofs())
    }
}

/// Signed permutation for quad ND: LOR ND1 dof → HO ND_k dof.
fn build_nd_perm_2d(
    mesh: &Mesh<2>,
    ho: &HCurlSpace<Mesh<2>>,
    lor: &HCurlSpace<Mesh<2>>,
    k: usize,
) -> FemResult<Vec<i32>> {
    let n_elem = mesh.n_elements() as u32;
    let n_lor = lor.n_dofs();
    let n_ho = ho.n_dofs();
    if n_lor != n_ho {
        return Err(FemError::Other(format!(
            "LOR ND 2D: dof count mismatch (LOR {n_lor} != HO {n_ho})"
        )));
    }

    let mut edge_owner: HashMap<EdgeKey, (u32, usize)> = HashMap::new();
    for e in 0..n_elem {
        let verts = mesh.element_nodes(e);
        for (lei, &(li, lj)) in QUAD_EDGES.iter().enumerate() {
            let key = EdgeKey::new(verts[li], verts[lj]);
            edge_owner.entry(key).or_insert((e, lei));
        }
    }

    let n_edge_dofs = 4 * k;
    let mut perm = vec![0i32; n_lor];
    let mut filled = vec![false; n_lor];
    let k2 = k * k;

    // Canonical values of both spaces on the constant unit field — the
    // same-functional orientation signs ([`pair_sign`]).
    let unit = |_: &[f64]| vec![1.0_f64; 2];
    let xh_one = ho.interpolate_vector(&unit);
    let xl_one = lor.interpolate_vector(&unit);
    let xh_one = xh_one.as_slice();
    let xl_one = xl_one.as_slice();

    for eh in 0..n_elem {
        let verts = mesh.element_nodes(eh);
        let hd = ho.element_dofs(eh);

        for ky in 0..k {
            for kx in 0..k {
                let lor_elem = eh * (k2 as u32) + (kx + k * ky) as u32;
                let ld = lor.element_dofs(lor_elem);

                for (sei, &(si, sj)) in QUAD_EDGES.iter().enumerate() {
                    let c0 = QUAD_CORNERS[si];
                    let c1 = QUAD_CORNERS[sj];
                    let axis = (0..2).find(|&d| c0[d] != c1[d]).unwrap();
                    let base = [kx, ky];
                    let a = base[axis];
                    let d1 = 1 - axis;
                    let i1 = base[d1] + c0[d1];
                    let bnd = i1 == 0 || i1 == k;

                    let slot: usize = if bnd {
                        // Macro mesh edge: local edge index + mode.
                        let c_lo = QUAD_CORNERS
                            .iter()
                            .position(|c| c[axis] == 0 && c[d1] == (i1 == k) as usize)
                            .unwrap();
                        let c_hi = QUAD_CORNERS
                            .iter()
                            .position(|c| c[axis] == 1 && c[d1] == (i1 == k) as usize)
                            .unwrap();
                        let key = EdgeKey::new(verts[c_lo], verts[c_hi]);
                        let (owner, _lei) = edge_owner[&key];
                        if owner != eh {
                            continue;
                        }
                        // Local macro edge index: QUAD_EDGES order
                        // (bottom 0, right 1, top 2, left 3); mode = interval
                        // counted from the parametric origin (fem-rs modes use
                        // l_j(x) from the origin on every edge).
                        let lei = match (axis, i1 == k) {
                            (0, false) => 0, // bottom
                            (0, true) => 2,  // top
                            (1, true) => 1,  // right
                            (1, false) => 3, // left
                            _ => unreachable!(),
                        };
                        lei * k + a
                    } else {
                        // Element-interior lattice edge.
                        let flat = match axis {
                            // x-family: (j, i+1) = (a, i1)
                            0 => (i1 - 1) * k + a,
                            // y-family: (i+1, j) = (i1, a)
                            _ => k * (k - 1) + (i1 - 1) * k + a,
                        };
                        n_edge_dofs + flat
                    };

                    let g_lor = ld[sei] as usize;
                    let s = pair_sign(xh_one, xl_one, hd[slot] as usize, g_lor);
                    let target = (s as isize * hd[slot] as isize) as i32;
                    if filled[g_lor] {
                        if perm[g_lor] != target {
                            return Err(FemError::Other(format!(
                                "LOR ND 2D: inconsistent mapping for LOR dof {g_lor}"
                            )));
                        }
                    } else {
                        perm[g_lor] = target;
                        filled[g_lor] = true;
                    }
                }
            }
        }
    }

    if filled.iter().any(|&f| !f) {
        return Err(FemError::Other(
            "LOR ND 2D: some LOR dofs were never mapped (mesh not a lattice?)".into(),
        ));
    }
    Ok(perm)
}

// ─── 3-D RT (hex) ───────────────────────────────────────────────────────────

impl LorRt<3> {
    /// LOR discretization of a hex RT space (order >= 0, uniform order, all
    /// elements `Hex8`).  Refinement factor `order + 1`; LOR space is RT0.
    pub fn new_hex(ho: &HDivSpace<Mesh<3>>) -> FemResult<Self> {
        let mesh = ho.mesh().clone();
        let q = ho.order() as usize;
        let k = q + 1;
        check_uniform(&mesh, ElementType::Hex8, "LOR RT 3D")?;

        let (lor_mesh, lor_space) = if k == 1 {
            (mesh.clone(), HDivSpace::new(mesh.clone(), 0))
        } else {
            let lm = make_refined_3d(&mesh, k);
            let ls = HDivSpace::new(lm.clone(), 0);
            (lm, ls)
        };

        let perm = if k == 1 {
            (0..lor_space.n_dofs() as i32).collect::<Vec<i32>>()
        } else {
            build_rt_perm_3d(&mesh, ho, &lor_space, k)?
        };

        // Bijectivity validation.
        let mut seen = vec![false; ho.n_dofs()];
        for (i, &p) in perm.iter().enumerate() {
            let a = p.unsigned_abs() as usize;
            if a >= ho.n_dofs() || seen[a] {
                let prev = perm.iter().position(|&q| q.unsigned_abs() as usize == a);
                return Err(FemError::Other(format!(
                    "LOR RT 3D: permutation is not a bijection (dofs {prev:?} and {i} both map to {a}")
                ));
            }
            seen[a] = true;
        }

        Ok(LorRt { lor_mesh, lor_space, perm, n_ho: ho.n_dofs(), refinement: k })
    }
}

/// Signed permutation for hex RT: LOR RT0 face dof → HO RT_q dof.
///
/// Lattice bookkeeping (order q, refinement `k = q+1`):
/// - x-family lattice face `(α, β, γ)`: `α ∈ [0,k]` (plane), `β, γ ∈ [0,k)`;
/// - macro faces carry `(q+1)² = k²` dofs (flat `a*k + b` over the two
///   in-plane coordinates, `HexRTk` face-block order = `HDivSpace::HEX_FACES`);
/// - interiors carry `q(q+1)² = (k-1)k²` dofs per direction.
fn build_rt_perm_3d(
    mesh: &Mesh<3>,
    ho: &HDivSpace<Mesh<3>>,
    lor: &HDivSpace<Mesh<3>>,
    k: usize,
) -> FemResult<Vec<i32>> {
    let n_elem = mesh.n_elements() as u32;
    let n_lor = lor.n_dofs();
    let n_ho = ho.n_dofs();
    if n_lor != n_ho {
        return Err(FemError::Other(format!(
            "LOR RT 3D: dof count mismatch (LOR {n_lor} != HO {n_ho})"
        )));
    }

    let mut face_owner: HashMap<[u32; 4], u32> = HashMap::new();
    for e in 0..n_elem {
        let verts = mesh.element_nodes(e);
        for fq in HEX_FACES.iter() {
            let mut key = [verts[fq[0]], verts[fq[1]], verts[fq[2]], verts[fq[3]]];
            key.sort_unstable();
            face_owner.entry(key).or_insert(e);
        }
    }

    let n_face_dofs = 6 * k * k;
    let mut perm = vec![0i32; n_lor];
    let mut filled = vec![false; n_lor];
    let k3 = k * k * k;

    // Canonical values of both spaces on the constant unit field — the
    // same-functional orientation signs ([`pair_sign`]).
    let unit = |_: &[f64]| vec![1.0_f64; 3];
    let xh_one = ho.interpolate_vector(&unit);
    let xl_one = lor.interpolate_vector(&unit);
    let xh_one = xh_one.as_slice();
    let xl_one = xl_one.as_slice();

    for eh in 0..n_elem {
        let hd = ho.element_dofs(eh);

        for kz in 0..k {
            for ky in 0..k {
                for kx in 0..k {
                    let lor_elem = eh * (k3 as u32) + (kx + k * ky + k * k * kz) as u32;
                    let ld = lor.element_dofs(lor_elem);

                    // LOR RT0 subcell faces in HEX_FACES order: bottom z−,
                    // front y−, right x+, back y+, left x−, top z+.
                    for f in 0..6 {
                        // (axis, plane) of the subcell face.
                        let (axis, plane) = match f {
                            0 => (2usize, kz),
                            1 => (1, ky),
                            2 => (0, kx + 1),
                            3 => (1, ky + 1),
                            4 => (0, kx),
                            _ => (2, kz + 1),
                        };
                        let base = [kx, ky, kz];
                        let (d1, d2) = match axis {
                            0 => (1usize, 2usize),
                            1 => (0usize, 2usize),
                            _ => (0usize, 1usize),
                        };
                        let alpha = plane; // plane index along `axis`
                        let beta = base[d1];
                        let gamma = base[d2];

                        let slot: usize = if alpha == 0 || alpha == k {
                            // Macro mesh face: local face index (HEX_FACES
                            // order: bottom z− 0, front y− 1, right x+ 2,
                            // back y+ 3, left x− 4, top z+ 5).
                            let face_local = match (axis, alpha == k) {
                                (0, true) => 2,
                                (0, false) => 4,
                                (1, true) => 3,
                                (1, false) => 1,
                                (2, true) => 5,
                                (2, false) => 0,
                                _ => unreachable!(),
                            };
                            let key = macro_face_key(&mesh.element_nodes(eh), face_local);
                            let owner = face_owner[&key];
                            if owner != eh {
                                continue;
                            }
                            // Flat within the k² face block.  `HexRTk`'s face
                            // blocks enumerate the two free GLL indices in the
                            // face frame `HEX_RT_FACES` prescribes (MFEM
                            // `CUBE::FaceVert`, `u × v = outward normal`), i.e.
                            // slot `i + j*k` with `i` along the first free axis
                            // and `j` along the second, each reversed when the
                            // frame's flag says so.
                            let (_, _, _, f1, f2) = HEX_RT_FACES[face_local];
                            let (fi, fj) = (
                                if f1 { k - 1 - beta } else { beta },
                                if f2 { k - 1 - gamma } else { gamma },
                            );
                            face_local * k * k + fi + fj * k
                        } else {
                            // Element-interior lattice face.  HexRTk interior
                            // block order (see `HexRTk::eval_basis_vec`):
                            // x-block: z open outer, y open middle, x closed
                            // innermost; y-block: z open outer, y closed
                            // middle, x open innermost; z-block: z closed
                            // outer, y open middle, x open innermost.  The
                            // closed interior index equals the lattice plane.
                            let blk_sz = (k - 1) * k * k;
                            let flat = match axis {
                                0 => gamma * (k * (k - 1)) + beta * (k - 1) + (alpha - 1),
                                1 => blk_sz + gamma * ((k - 1) * k) + (alpha - 1) * k + beta,
                                _ => 2 * blk_sz + (alpha - 1) * (k * k) + gamma * k + beta,
                            };
                            n_face_dofs + flat
                        };

                        let g_lor = ld[f] as usize;
                        // Sign: relative orientation of the two canonical
                        // functionals (same-functional convention).
                        let s = pair_sign(xh_one, xl_one, hd[slot] as usize, g_lor);
                        let target = (s as isize * hd[slot] as isize) as i32;
                        if !filled[g_lor] {
                            perm[g_lor] = target;
                            filled[g_lor] = true;
                        } else if perm[g_lor] != target {
                            return Err(FemError::Other(format!(
                                "LOR RT 3D: inconsistent mapping for LOR dof {g_lor}"
                            )));
                        }
                    }
                }
            }
        }
    }

    if filled.iter().any(|&f| !f) {
        let miss: Vec<usize> = (0..n_lor).filter(|&i| !filled[i]).collect();
        return Err(FemError::Other(format!(
            "LOR RT 3D: some LOR dofs were never mapped: {miss:?}"
        )));
    }
    Ok(perm)
}

// ─── 2-D RT (quad) ──────────────────────────────────────────────────────────

impl LorRt<2> {
    /// LOR discretization of a quad RT space (order >= 0, uniform order, all
    /// elements `Quad4`).  Refinement factor `order + 1`; LOR space is RT0.
    pub fn new_quad(ho: &HDivSpace<Mesh<2>>) -> FemResult<Self> {
        let mesh = ho.mesh().clone();
        let q = ho.order() as usize;
        let k = q + 1;
        check_uniform(&mesh, ElementType::Quad4, "LOR RT 2D")?;

        let (lor_mesh, lor_space) = if k == 1 {
            (mesh.clone(), HDivSpace::new(mesh.clone(), 0))
        } else {
            let lm = make_refined_2d(&mesh, k);
            let ls = HDivSpace::new(lm.clone(), 0);
            (lm, ls)
        };

        let perm = if k == 1 {
            (0..lor_space.n_dofs() as i32).collect::<Vec<i32>>()
        } else {
            build_rt_perm_2d(&mesh, ho, &lor_space, k)?
        };

        let mut seen = vec![false; ho.n_dofs()];
        for (i, &p) in perm.iter().enumerate() {
            let a = p.unsigned_abs() as usize;
            if a >= ho.n_dofs() || seen[a] {
                let prev = perm.iter().position(|&q| q.unsigned_abs() as usize == a);
                return Err(FemError::Other(format!(
                    "LOR RT 2D: permutation is not a bijection (dofs {prev:?} and {i} both map to {a}")
                ));
            }
            seen[a] = true;
        }

        Ok(LorRt { lor_mesh, lor_space, perm, n_ho: ho.n_dofs(), refinement: k })
    }
}

/// Signed permutation for quad RT: LOR RT0 edge dof → HO RT_q dof.
///
/// Slot tables follow `QuadRTk`'s `build_dof_map` slot assignment:
/// bottom `[0,k)`, right `[k,2k)`, top `[2k,3k)` (x-index reversed),
/// left `[3k,4k)` (y-index reversed), interior-x, interior-y.
fn build_rt_perm_2d(
    mesh: &Mesh<2>,
    ho: &HDivSpace<Mesh<2>>,
    lor: &HDivSpace<Mesh<2>>,
    k: usize,
) -> FemResult<Vec<i32>> {
    let n_elem = mesh.n_elements() as u32;
    let n_lor = lor.n_dofs();
    let n_ho = ho.n_dofs();
    if n_lor != n_ho {
        return Err(FemError::Other(format!(
            "LOR RT 2D: dof count mismatch (LOR {n_lor} != HO {n_ho})"
        )));
    }

    let mut edge_owner: HashMap<EdgeKey, u32> = HashMap::new();
    for e in 0..n_elem {
        let verts = mesh.element_nodes(e);
        for &(li, lj) in QUAD_EDGES.iter() {
            let key = EdgeKey::new(verts[li], verts[lj]);
            edge_owner.entry(key).or_insert(e);
        }
    }

    let q = k - 1;
    let n_edge_dofs = 4 * k;
    let mut perm = vec![0i32; n_lor];
    let mut filled = vec![false; n_lor];
    let k2 = k * k;

    // Canonical values of both spaces on the constant unit field — the
    // same-functional orientation signs ([`pair_sign`]).
    let unit = |_: &[f64]| vec![1.0_f64; 2];
    let xh_one = ho.interpolate_vector(&unit);
    let xl_one = lor.interpolate_vector(&unit);
    let xh_one = xh_one.as_slice();
    let xl_one = xl_one.as_slice();

    for eh in 0..n_elem {
        let hd = ho.element_dofs(eh);

        for ky in 0..k {
            for kx in 0..k {
                let lor_elem = eh * (k2 as u32) + (kx + k * ky) as u32;
                let ld = lor.element_dofs(lor_elem);

                // LOR RT0 subcell "faces" = 4 edges (QUAD_FACES order:
                // bottom, right, top, left).
                for f in 0..4 {
                    // (axis, plane): x-faces are x-normal (right/left),
                    // y-faces y-normal (bottom/top).
                    let (axis, plane) = match f {
                        0 => (1usize, ky),        // bottom: y-normal, plane ky
                        1 => (0, kx + 1),         // right: x-normal, plane kx+1
                        2 => (1, ky + 1),         // top
                        _ => (0, kx),             // left
                    };
                    let base = [kx, ky];
                    let d1 = 1 - axis;
                    let alpha = plane; // plane along `axis` (0..k)
                    let beta = base[d1]; // interval across

                    let slot: usize = if alpha == 0 || alpha == k {
                        // Macro mesh edge.  Local edge index in QUAD_FACES
                        // order; slot per the QuadRTk build_dof_map tables
                        // (bottom: α; right: k+β; top: 2k + (k-1-α);
                        //  left: 3k + (k-1-β)).
                        // The sub-edge on a macro mesh edge is the subcell.s own
                        // face f; its endpoints are the face.s corner pair.
                        let verts = mesh.element_nodes(eh);
                        let key = EdgeKey::new(verts[QUAD_EDGES[f].0], verts[QUAD_EDGES[f].1]);
                        let owner = edge_owner[&key];
                        if owner != eh {
                            continue;
                        }
                        let slot = match (axis, alpha == k) {
                            // bottom (y-face, plane 0): slot = α_x
                            (1, false) => beta,
                            // top (plane k): reversed: slot = 2k + (k-1-α_x)
                            (1, true) => 2 * k + (k - 1 - beta),
                            // right (x-face, plane k): slot = k + β
                            (0, true) => k + beta,
                            // left (plane 0): reversed: slot = 3k + (k-1-β)
                            (0, false) => 3 * k + (k - 1 - beta),
                            _ => unreachable!(),
                        };
                        slot
                    } else {
                        // Element-interior lattice face (2D: interior dofs).
                        let flat = match axis {
                            // x-family interior: slot 4k + β·q + (α-1)
                            0 => n_edge_dofs + beta * q + (alpha - 1),
                            // y-family interior: slot 4k + q(q+1) + (β_y-1)(q+1) + α_x
                            _ => {
                                n_edge_dofs
                                    + q * (q + 1)
                                    + (alpha - 1) * (q + 1)
                                    + beta
                            }
                        };
                        flat
                    };

                    let g_lor = ld[f] as usize;
                    // Sign: relative orientation of the two canonical
                    // functionals (same-functional convention).
                    let s = pair_sign(xh_one, xl_one, hd[slot] as usize, g_lor);
                    let target = (s as isize * hd[slot] as isize) as i32;
                    if !filled[g_lor] {
                        perm[g_lor] = target;
                        filled[g_lor] = true;
                    } else if perm[g_lor] != target {
                        return Err(FemError::Other(format!(
                            "LOR RT 2D: inconsistent mapping for LOR dof {g_lor}"
                        )));
                    }
                }
            }
        }
    }

    if filled.iter().any(|&f| !f) {
        let miss: Vec<usize> = (0..n_lor).filter(|&i| !filled[i]).collect();
        return Err(FemError::Other(format!(
            "LOR RT 2D: some LOR dofs were never mapped: {miss:?}"
        )));
    }
    Ok(perm)
}

// ─── LOR H1 (scalar / vector elasticity) ────────────────────────────────────

/// Low-order refined H1 discretization of a scalar Lagrange space — MFEM
/// `LORDiscretization` for `H1_FECollection` (`fem/lor/`, batched kernel in
/// `fem/lor/lor_h1.hpp`).
///
/// The LOR mesh subdivides every HO element into `order^dim` P1 sub-elements
/// whose corners are the H1(`order`) Gauss-Lobatto dof positions
/// ([`crate::make_refined::make_refined_2d`] / [`crate::make_refined::make_refined_3d`],
/// i.e. MFEM `Mesh::MakeRefined`); the LOR space is P1 on that mesh, so an
/// LOR dof is a refined-mesh node id.
///
/// `perm[i] = j` (all signs `+1`: H1 dofs are positive point evaluations)
/// maps LOR dof `i` to the HO scalar dof `j` at the same lattice point.  MFEM
/// returns the identity permutation for H1 (both spaces number dofs
/// identically); here the two fem-rs numbering conventions can differ, so the
/// map is constructed element-locally (by matching the dof coordinates at
/// each lattice point — the same technique `make_refined` itself uses) and
/// validated as a bijection.
///
/// For an `order = 1` space the LOR space is the space itself (identity
/// permutation, unrefined mesh) — matching MFEM.
pub struct LorH1<const D: usize> {
    lor_mesh: Mesh<D>,
    /// `perm[i] = j`: LOR dof `i` (refined-mesh node id) ↔ HO scalar dof `j`.
    perm: Vec<u32>,
    n_ho: usize,
    refinement: usize,
}

impl<const D: usize> LorH1<D> {
    /// The refined (LOR) mesh.  Its P1 dof numbering is the node numbering.
    pub fn lor_mesh(&self) -> &Mesh<D> { &self.lor_mesh }

    /// Number of scalar LOR dofs (refined-mesh nodes).
    pub fn n_lor(&self) -> usize { self.lor_mesh.n_nodes() }

    /// Number of HO scalar dofs.
    pub fn n_ho(&self) -> usize { self.n_ho }

    /// Permutation `perm[i] = j` (LOR dof → HO scalar dof), all signs `+1`.
    pub fn perm(&self) -> &[u32] { &self.perm }

    /// Refinement factor per direction (`order` for H1, like MFEM).
    pub fn refinement(&self) -> usize { self.refinement }

    /// Vector-space (byNODES) permutation: LOR vector dof
    /// `c * n_scalar_lor + i` ↔ HO vector dof `c * n_scalar + perm[i]`.
    ///
    /// This is the dof correspondence of the LOR image of a vector H1 space
    /// (`VectorH1Space` of `dim` components; MFEM `ParLORDiscretization` of a
    /// `byNODES` vector fespace).
    pub fn vector_perm(&self, n_scalar: usize, dim: usize) -> Vec<u32> {
        let n_lor = self.n_lor();
        let mut vp = vec![0u32; n_lor * dim];
        for c in 0..dim {
            for (i, &p) in self.perm.iter().enumerate() {
                vp[c * n_lor + i] = (c * n_scalar + p as usize) as u32;
            }
        }
        vp
    }

    /// Reorder an LOR-numbered matrix into HO scalar dof numbering:
    /// `B[perm[i], perm[j]] = A[i][j]`.
    pub fn ho_numbering(&self, a_lor: &CsrMatrix<f64>) -> CsrMatrix<f64> {
        assert_eq!(a_lor.nrows, a_lor.ncols, "ho_numbering: matrix must be square");
        assert_eq!(a_lor.nrows, self.perm.len(), "ho_numbering: matrix size mismatch");
        let mut coo = fem_linalg::CooMatrix::<f64>::new(self.n_ho, self.n_ho);
        for i in 0..a_lor.nrows {
            let gi = self.perm[i] as usize;
            for r in a_lor.row_ptr[i]..a_lor.row_ptr[i + 1] {
                let gj = self.perm[a_lor.col_idx[r] as usize] as usize;
                let v = a_lor.values[r];
                if v != 0.0 {
                    coo.add(gi, gj, v);
                }
            }
        }
        coo.into_csr()
    }

    /// Prolongate: `x_ho[perm[i]] = x_lor[i]`.
    pub fn prolongate(&self, x_lor: &[f64], x_ho: &mut [f64]) {
        assert_eq!(x_lor.len(), self.perm.len());
        assert_eq!(x_ho.len(), self.n_ho);
        for (i, &p) in self.perm.iter().enumerate() {
            x_ho[p as usize] = x_lor[i];
        }
    }

    /// Restrict (transpose of [`LorH1::prolongate`]): `x_lor[i] = x_ho[perm[i]]`.
    pub fn restrict(&self, x_ho: &[f64], x_lor: &mut [f64]) {
        assert_eq!(x_lor.len(), self.perm.len());
        assert_eq!(x_ho.len(), self.n_ho);
        for (i, &p) in self.perm.iter().enumerate() {
            x_lor[i] = x_ho[p as usize];
        }
    }

    fn finish(lor_mesh: Mesh<D>, perm: Vec<u32>, n_ho: usize, k: usize) -> FemResult<Self> {
        // Bijectivity validation (the H1 assumed-constraint map is a plain
        // permutation; MFEM returns the identity for H1).
        if perm.len() != lor_mesh.n_nodes() {
            return Err(FemError::Other(
                "LOR H1: permutation size does not match LOR dof count".into(),
            ));
        }
        let mut seen_ho = vec![false; n_ho];
        for &p in &perm {
            let a = p as usize;
            if a >= n_ho || seen_ho[a] {
                return Err(FemError::Other(format!(
                    "LOR H1: permutation is not a bijection (entry {p})"
                )));
            }
            seen_ho[a] = true;
        }
        Ok(LorH1 { lor_mesh, perm, n_ho, refinement: k })
    }
}

impl LorH1<2> {
    /// LOR H1 discretization of the scalar Lagrange space of `order` on a 2-D
    /// `mesh` (uniform order; Quad4 at any order, Tri3 at order <= 3;
    /// `order == 1` is the identity path).
    pub fn new(mesh: &Mesh<2>, order: u8) -> FemResult<Self> {
        let k = order as usize;
        if k == 0 {
            return Err(FemError::Other("LOR H1: order must be >= 1".into()));
        }
        let dm = DofManager::new(mesh, order);
        if k == 1 {
            // MFEM: order-1 H1 LOR = the space itself (identity permutation).
            let n = dm.n_dofs;
            return Self::finish(mesh.clone(), (0..n as u32).collect(), n, 1);
        }
        let et = mesh.elem_type;
        let supported = matches!(et, ElementType::Quad4) || (et == ElementType::Tri3 && k <= 3);
        if !supported {
            return Err(FemError::Other(format!(
                "LOR H1 2D: {et:?} meshes at order {k} are not supported yet \
                 (Quad4 any order, Tri3 <= 3, or order 1)"
            )));
        }
        let lor_mesh = make_refined_2d(mesh, k);
        let perm = build_h1_perm(mesh, &dm, &lor_mesh, k)?;
        Self::finish(lor_mesh, perm, dm.n_dofs, k)
    }
}

impl LorH1<3> {
    /// LOR H1 discretization of the scalar Lagrange space of `order` on a 3-D
    /// `mesh` (uniform order; Hex8 at any order, Tet4 at order 2;
    /// `order == 1` is the identity path).
    pub fn new(mesh: &Mesh<3>, order: u8) -> FemResult<Self> {
        let k = order as usize;
        if k == 0 {
            return Err(FemError::Other("LOR H1: order must be >= 1".into()));
        }
        let dm = DofManager::new(mesh, order);
        if k == 1 {
            // MFEM: order-1 H1 LOR = the space itself (identity permutation).
            let n = dm.n_dofs;
            return Self::finish(mesh.clone(), (0..n as u32).collect(), n, 1);
        }
        let et = mesh.elem_type;
        let supported = matches!(et, ElementType::Hex8) || (et == ElementType::Tet4 && k == 2);
        if !supported {
            return Err(FemError::Other(format!(
                "LOR H1 3D: {et:?} meshes at order {k} are not supported yet \
                 (Hex8 any order, Tet4 == 2, or order 1)"
            )));
        }
        let lor_mesh = make_refined_3d(mesh, k);
        let perm = build_h1_perm(mesh, &dm, &lor_mesh, k)?;
        Self::finish(lor_mesh, perm, dm.n_dofs, k)
    }
}

/// Build the H1 LOR permutation: for every macro element, its lattice points
/// are exactly the corner nodes of its `k^dim` sub-elements on the LOR mesh;
/// match each lattice node to the macro element's HO dof at the same
/// physical position (both coordinate sets come from the same GLL
/// interpolation, cf. `make_refined`).
fn build_h1_perm<const D: usize>(
    mesh: &Mesh<D>,
    dm: &DofManager,
    lor_mesh: &Mesh<D>,
    k: usize,
) -> FemResult<Vec<u32>> {
    let n_lor = lor_mesh.n_nodes();
    let mut perm = vec![u32::MAX; n_lor];
    let mut filled = vec![false; n_lor];

    // Coordinate tolerance: same convention as `make_refined` (both sides
    // evaluate the same interpolation, so they agree to round-off).
    let mut scale = 0.0_f64;
    for n in 0..mesh.n_nodes() as u32 {
        for d in 0..D {
            let c = mesh.node_coords(n)[d].abs();
            if c > scale {
                scale = c;
            }
        }
    }
    let tol = 1e-7 * scale.max(1.0);

    let n_sub = k.pow(D as u32);
    let n_elem = mesh.n_elements() as u32;
    for e in 0..n_elem {
        let ho = dm.element_dofs(e);
        // Lattice points of this macro element = union of sub-element corners.
        let mut seen = std::collections::HashSet::new();
        let mut lattice: Vec<u32> = Vec::new();
        for s in 0..n_sub {
            let le = e * (n_sub as u32) + s as u32;
            for &v in lor_mesh.element_nodes(le) {
                if seen.insert(v) {
                    lattice.push(v);
                }
            }
        }
        for v in lattice {
            let c = lor_mesh.node_coords(v);
            let mut target: Option<u32> = None;
            for &d in ho {
                let dc = dm.dof_coord(d);
                if (0..D).all(|q| (dc[q] - c[q]).abs() <= tol) {
                    target = Some(d);
                    break;
                }
            }
            let t = target.ok_or_else(|| {
                FemError::Other(format!(
                    "LOR H1: lattice node {v} of element {e} has no matching \
                     HO dof (mesh not a Gauss-Lobatto lattice?)"
                ))
            })?;
            if filled[v as usize] {
                if perm[v as usize] != t {
                    return Err(FemError::Other(format!(
                        "LOR H1: inconsistent mapping for LOR dof {v}"
                    )));
                }
            } else {
                perm[v as usize] = t;
                filled[v as usize] = true;
            }
        }
    }

    if filled.iter().any(|&f| !f) {
        return Err(FemError::Other(
            "LOR H1: some LOR dofs were never mapped (nodes outside the \
             macro lattice?)"
                .into(),
        ));
    }
    Ok(perm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lor_nd_hex_dof_counts_and_bijection() {
        for (n, k) in [(1usize, 2usize), (2, 2), (2, 3), (3, 2)] {
            let mesh = Mesh::<3>::make_cartesian_3d(n, n, n,
                ElementType::Hex8, 1.0, 1.0, 1.0, false);
            let ho = HCurlSpace::new(mesh.clone(), k as u8);
            let lor = LorNd::<3>::new_hex(&ho).expect("LOR ND 3D");
            assert_eq!(lor.lor_space().n_dofs(), ho.n_dofs(), "n={n} k={k}");
            assert_eq!(lor.refinement(), k);
            assert_eq!(lor.lor_mesh().n_elems(), n * n * n * k * k * k);
        }
    }

    #[test]
    fn lor_nd_quad_dof_counts_and_bijection() {
        for k in [2usize, 3usize] {
            let mesh = Mesh::<2>::unit_square_quad(3);
            let ho = HCurlSpace::new(mesh.clone(), k as u8);
            let lor = LorNd::<2>::new_quad(&ho).expect("LOR ND 2D");
            assert_eq!(lor.lor_space().n_dofs(), ho.n_dofs(), "k={k}");
            assert_eq!(lor.lor_mesh().n_elems(), 9 * k * k);
        }
    }

    #[test]
    fn lor_rt_hex_dof_counts_and_bijection() {
        for (n, q) in [(1usize, 0usize), (1, 1), (2, 1), (2, 2), (2, 0), (3, 1)] {
            let mesh = Mesh::<3>::make_cartesian_3d(n, n, n,
                ElementType::Hex8, 1.0, 1.0, 1.0, false);
            let ho = HDivSpace::new(mesh.clone(), q as u8);
            let lor = LorRt::<3>::new_hex(&ho).expect("LOR RT 3D");
            assert_eq!(lor.lor_space().n_dofs(), ho.n_dofs(), "n={n} q={q}");
            assert_eq!(lor.refinement(), q + 1);
        }
    }

    #[test]
    fn lor_rt_quad_dof_counts_and_bijection() {
        for q in [0usize, 1usize, 2usize] {
            let mesh = Mesh::<2>::unit_square_quad(3);
            let ho = HDivSpace::new(mesh.clone(), q as u8);
            let lor = LorRt::<2>::new_quad(&ho).expect("LOR RT 2D");
            assert_eq!(lor.lor_space().n_dofs(), ho.n_dofs(), "q={q}");
            assert_eq!(lor.refinement(), q + 1);
        }
    }

    #[test]
    fn lor_identity_for_lowest_order() {
        let mesh = Mesh::<3>::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
        let ho = HCurlSpace::new(mesh.clone(), 1);
        let lor = LorNd::<3>::new_hex(&ho).unwrap();
        assert_eq!(lor.refinement(), 1);
        assert!(lor.perm().iter().enumerate().all(|(i, &p)| p == i as i32));

        let ho_rt = HDivSpace::new(mesh, 0);
        let lor_rt = LorRt::<3>::new_hex(&ho_rt).unwrap();
        assert!(lor_rt.perm().iter().enumerate().all(|(i, &p)| p == i as i32));
    }

    #[test]
    fn lor_ho_numbering_signed_similarity() {
        // Apply ho_numbering twice with the inverse permutation: identity.
        // Here: check B is symmetric when A is (sign squares to +1).
        let mesh = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
        let ho = HCurlSpace::new(mesh, 2);
        let lor = LorNd::<3>::new_hex(&ho).unwrap();
        let n = lor.n_ho();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i + 1 < n {
                coo.add(i, i + 1, -0.5);
                coo.add(i + 1, i, -0.5);
            }
        }
        let a = coo.into_csr();
        let b = lor.ho_numbering(&a);
        // A signed permutation congruence of a symmetric matrix stays
        // symmetric with an identical diagonal (as a multiset).
        let mut diag_a: Vec<f64> = (0..n).map(|i| a.get(i, i)).collect();
        let mut diag_b: Vec<f64> = (0..n).map(|i| b.get(i, i)).collect();
        diag_a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        diag_b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for i in 0..n {
            assert!((diag_a[i] - diag_b[i]).abs() < 1e-14, "diag {i}: {} vs {}", diag_a[i], diag_b[i]);
            for r in b.row_ptr[i]..b.row_ptr[i + 1] {
                let j = b.col_idx[r] as usize;
                let v = b.values[r];
                assert!(
                    (v - b.get(j, i)).abs() < 1e-14,
                    "permuted matrix must remain symmetric"
                );
                let off = if i < j { v } else { continue };
                let _ = off;
            }
        }
        // Off-diagonal |value| multiset preserved (signs may flip under a
        // signed permutation).
        let mut off_a: Vec<f64> = Vec::new();
        let mut off_b: Vec<f64> = Vec::new();
        for i in 0..n {
            for r in a.row_ptr[i]..a.row_ptr[i + 1] {
                let j = a.col_idx[r] as usize;
                if j > i { off_a.push(a.values[r].abs()); }
            }
            for r in b.row_ptr[i]..b.row_ptr[i + 1] {
                let j = b.col_idx[r] as usize;
                if j > i { off_b.push(b.values[r].abs()); }
            }
        }
        off_a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        off_b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(off_a.len(), off_b.len());
        for (x, y) in off_a.iter().zip(off_b.iter()) {
            assert!((x - y).abs() < 1e-14);
        }
    }

    // ── LOR H1 tests ─────────────────────────────────────────────────────

    fn assert_perm_bijection(perm: &[u32], n_ho: usize) {
        assert_eq!(perm.len(), n_ho, "LOR and HO dof counts must match");
        let mut seen = vec![false; n_ho];
        for &p in perm {
            assert!((p as usize) < n_ho, "perm entry {p} out of range");
            assert!(!seen[p as usize], "duplicate perm target {p}");
            seen[p as usize] = true;
        }
    }

    #[test]
    fn lor_h1_identity_for_order_one() {
        let q = Mesh::<2>::unit_square_quad(2);
        let lor = LorH1::<2>::new(&q, 1).unwrap();
        assert_eq!(lor.refinement(), 1);
        assert_eq!(lor.n_lor(), q.n_nodes());
        assert!(lor.perm().iter().enumerate().all(|(i, &p)| p == i as u32));

        let h = Mesh::<3>::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
        let lor3 = LorH1::<3>::new(&h, 1).unwrap();
        assert!(lor3.perm().iter().enumerate().all(|(i, &p)| p == i as u32));
    }

    #[test]
    fn lor_h1_quad_bijection_and_vertex_ids_preserved() {
        for k in [2usize, 3usize] {
            let mesh = Mesh::<2>::unit_square_quad(3);
            let lor = LorH1::<2>::new(&mesh, k as u8).expect("LOR H1 quad");
            assert_eq!(lor.refinement(), k);
            assert_eq!(lor.lor_mesh().n_elems(), 9 * k * k);
            assert_perm_bijection(lor.perm(), lor.n_ho());
            // Original mesh vertices keep their dof ids (both numberings put
            // vertices first, in mesh order).
            for v in 0..mesh.n_nodes() as u32 {
                assert_eq!(lor.perm()[v as usize], v, "vertex {v} remapped at k={k}");
            }
        }
    }

    #[test]
    fn lor_h1_hex_bijection() {
        for k in [2usize, 3usize] {
            let mesh = Mesh::<3>::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
            let lor = LorH1::<3>::new(&mesh, k as u8).expect("LOR H1 hex");
            assert_eq!(lor.refinement(), k);
            assert_eq!(lor.lor_mesh().n_elems(), 2 * k * k * k);
            assert_perm_bijection(lor.perm(), lor.n_ho());
            for v in 0..mesh.n_nodes() as u32 {
                assert_eq!(lor.perm()[v as usize], v, "vertex {v} remapped at k={k}");
            }
        }
    }

    #[test]
    fn lor_h1_tri_and_tet_low_order() {
        let tri = Mesh::<2>::make_cartesian_2d_tri(2, 2, 1.0, 1.0);
        let lor_t = LorH1::<2>::new(&tri, 2).expect("LOR H1 tri (k=2)");
        assert_perm_bijection(lor_t.perm(), lor_t.n_ho());
        assert_eq!(lor_t.lor_mesh().n_elems(), 8 * 4); // 8 macro tris x nref^2

        let tet = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Tet4, 1.0, 1.0, 1.0, false);
        let lor_te = LorH1::<3>::new(&tet, 2).expect("LOR H1 tet (k=2)");
        assert_perm_bijection(lor_te.perm(), lor_te.n_ho());
        assert_eq!(lor_te.lor_mesh().n_elems(), 6 * 8); // 6 macro tets x 1->8
    }

    #[test]
    fn lor_h1_vector_perm_by_nodes() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let lor = LorH1::<2>::new(&mesh, 2).unwrap();
        let n_scalar = lor.n_ho();
        let n_lor = lor.n_lor();
        let vp = lor.vector_perm(n_scalar, 2);
        assert_eq!(vp.len(), 2 * n_lor);
        // Component block structure: block c maps onto HO block c.
        for c in 0..2 {
            for (i, &p) in lor.perm().iter().enumerate() {
                assert_eq!(vp[c * n_lor + i], (c * n_scalar + p as usize) as u32);
            }
        }
    }

    #[test]
    fn lor_h1_ho_numbering_preserves_symmetry_and_diagonal() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let lor = LorH1::<2>::new(&mesh, 2).unwrap();
        let n = lor.n_lor();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0 + (i % 3) as f64);
            if i + 1 < n {
                coo.add(i, i + 1, -0.5);
                coo.add(i + 1, i, -0.5);
            }
        }
        let a = coo.into_csr();
        let b = lor.ho_numbering(&a);
        for i in 0..b.nrows {
            for r in b.row_ptr[i]..b.row_ptr[i + 1] {
                let j = b.col_idx[r] as usize;
                assert!((b.values[r] - b.get(j, i)).abs() < 1e-14);
            }
        }
        // Prolongate/restrict round trip through the permutation.
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let mut y = vec![0.0; lor.n_ho()];
        lor.prolongate(&x, &mut y);
        let mut z = vec![0.0; n];
        lor.restrict(&y, &mut z);
        assert_eq!(x, z);
    }
}
