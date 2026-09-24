//! Raviart-Thomas hexahedral element `RT_k` on the reference hex `[0,1]^3`.
//!
//! 1:1 port of both MFEM `RT_HexahedronElement(p, GaussLobatto, ·)` open-basis
//! variants (MFEM order `p`; `RT_k` with `k = p`), in MFEM's own `[0,1]³`
//! reference frame (D721 — the pre-D721 code evaluated everything on the
//! historical fem-rs `[-1,1]³` frame and carried explicit pull-back factors):
//!
//! - [`HexRTk::new_gauss_legendre`] — `ob_type = GaussLegendre`, the element
//!   `RT_FECollection(p, dim)` builds **by default** (`fem/fe_coll.hpp`): the
//!   open modes are the degree-`k` Gauss-Legendre nodal Lagrange polynomials
//!   at the `k+1` open points.  This is the flavour every MFEM VisIt data
//!   collection / `get-values` / `Project_RT` exchange uses, and its
//!   contravariant pull-back to the physical hex is *exactly* MFEM's physical
//!   field point for point (bit for bit since D721: the reference basis, the
//!   Jacobian and the quadrature rule all live on `[0,1]`).
//! - [`HexRTk::new`] — `ob_type = IntegratedGLL`, the basis pair MFEM
//!   documents (and `fem/lor/lor.cpp` *requires*) for LOR-compatible RT
//!   spaces.  Open modes are the integrated (Gerritsma) edge functions
//!   `-Σ_{j<=i} c'_j` built from the degree-`k+1` GLL closed basis — the
//!   `[0,1]` derivatives of `ClosedBasis::new_01`, with no chain factor (the
//!   historical `1/2` was the `[-1,1]` pull-back) — i.e. exactly MFEM's
//!   `EvalIntegrated` modes.
//!
//! Both variants share the identical dof layout, dof positions and MFEM
//! `dof_map` orientation encoding (round 40: `probe_map_glvslgl.cpp` — the
//! maps are element-for-element identical), so the LOR permutation machinery
//! in `fem_space::lor` is variant independent.
//!
//! Tensor structure per component (`c` = closed GLL nodal mode of degree
//! `k+1` on `[0,1]`, `o` = open mode of the variant, two per axis count
//! `k+1`):
//!
//! ```text
//!     x-dofs: c(x)·o(y)·o(z)   y-dofs: o(x)·c(y)·o(z)   z-dofs: o(x)·o(y)·c(z)
//! ```
//!
//! Local DOF order (matches `HDivSpace::build_3d_hex`):
//! - six face blocks of `(k+1)^2` dofs in `HDivSpace::HEX_FACES` order —
//!   MFEM `Geometry::CUBE::FaceVert`: bottom z−, front y−, right x+, back y+,
//!   left x−, top z+ — so the positional (basis i ↔ element_dofs[i]) pairing
//!   used by the vector assembler is geometrically consistent across
//!   neighbouring hexes;
//! - three interior blocks of `k(k+1)^2` dofs (x/y/z components, closed
//!   interior index `1..=k` innermost per MFEM's loop order), with MFEM's
//!   reference orientation flips baked in (D225): interior dofs whose closed
//!   block index is `<= k/2` carry a negative sign (MFEM
//!   `dof_map = -1 - dof_map`), i.e. for `RT2` the first interior closed
//!   index of every component flips.
//!
//! Each face-block function carries the sign of its outward normal
//! (`s = -1` on the −x/−y/−z faces, `+1` on +x/+y/+z) so that the nominal
//! face mode has unit outward flux `∫_F φ·n dA = 1`; this is MFEM's `nk`
//! orientation convention for the positively oriented face dofs and makes
//! the flux dofs exact duals of the basis (matching
//! `HDivSpace`'s flux-dof interpretation).
//!
//! Within each face block the two free (open) indices run (i outer, j inner);
//! the closed factor is anchored at the face's endpoint GLL node.

use crate::gll_basis::{gll_nodes_01, ClosedBasis};
use crate::nedelec::hex_ndk::open_basis;
use crate::reference::VectorReferenceElement;

/// Per-DOF integrated (sub-cell face flux) functional of the hex
/// IntegratedGLL RT variant — the 3-D analogue of the quad
/// `IntegratedDofFunctional` (`crates/element/src/nedelec/quad_ndk_mfem.rs`).
pub struct IntegratedDofFunctional3D {
    /// `(reference sample point, weight)` pairs over the DOF's sub-cell face;
    /// the weight is the 2-D Gauss product `(w_u·h_1)·(w_v·h_2)` with the
    /// `[0,1]` Gauss-Legendre rule and the `[0,1]` GLL sub-cell widths.
    pub samples: Vec<([f64; 3], f64)>,
    /// Reference normal direction `±e_c` — the `dof_map` flip (MFEM `nk`).
    pub t: [f64; 3],
}

/// The 1-D open-factor kind of the tensor RT basis — the `ob_type` argument of
/// MFEM `RT_HexahedronElement(p, cb_type, ob_type)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HexRtOpen {
    /// MFEM `BasisType::GaussLegendre` — the `RT_FECollection(p, dim)` default
    /// (what [`HexRTk::new_gauss_legendre`] builds).  Open modes are the
    /// degree-`k` Gauss-Legendre point-value Lagrange polynomials at the `k+1`
    /// open points (the very points of `FE::Nodes`).
    GaussLegendre,
    /// MFEM `BasisType::IntegratedGLL` — the open half of the
    /// `(GaussLobatto, IntegratedGLL)` basis pair that MFEM documents — and
    /// `fem/lor/lor.cpp` enforces — for LOR discretizations (what
    /// [`HexRTk::new`] builds).  Open modes are the integrated (Gerritsma)
    /// edge functions `-Σ_{j<=i} c'_j` built from the degree-`k+1` GLL closed
    /// basis (`Poly_1D::Basis::EvalIntegrated`); the dof positions are
    /// unchanged (the same Gauss-Legendre points).
    IntegratedGLL,
}

/// The six hex faces in `HDivSpace::HEX_FACES` order — MFEM
/// `Geometry::Constants<Geometry::CUBE>::FaceVert`: bottom z−, front y−,
/// right x+, back y+, left x−, top z+.
///
/// Each entry is `(normal axis, normal endpoint is the max (`k+1`) one,
/// outward-normal sign, flip the first free-axis index, flip the second)`.
///
/// MFEM enumerates the `(k+1)^2` face dofs of `RT_HexahedronElement` in the
/// face frame `(u,v)` of `FaceVert`, which is chosen so that
/// `u × v = outward normal`.  For the bottom (z−), back (y+) and left (x−)
/// faces that frame is a *reflection* of the increasing-axis frame the
/// tensor basis is naturally written in, so those three faces enumerate one
/// (or both, for z−) of their free GLL indices in reverse: slot `t` holds the
/// basis function whose axis index is `k+1−t`, and the slot's node sits at
/// that same reversed axis index (see `RT_HexahedronElement::dof_map`, which
/// assigns e.g. the bottom face's slots to the z-block entries
/// `(i, p−j)`).  The remaining three faces (y−, x+, z+) enumerate in the
/// increasing-axis order.
pub const HEX_RT_FACES: [(usize, bool, f64, bool, bool); 6] = [
    (2, false, -1.0, false, true),  // z− : u=+x, v=−y → (i, k+1−j)
    (1, false, -1.0, false, false), // y− : u=+x, v=+z → (i, j)
    (0, true, 1.0, false, false),   // x+ : u=+y, v=+z → (i, j)
    (1, true, 1.0, true, false),    // y+ : u=−x, v=+z → (k+1−i, j)
    (0, false, -1.0, true, false),  // x− : u=−y, v=+z → (k+1−i, j)
    (2, true, 1.0, false, false),   // z+ : u=+x, v=+y → (i, j)
];

/// The two free axes of `normal`, in increasing axis order.
#[inline]
pub fn free_axes(normal: usize) -> (usize, usize) {
    match normal {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// Sign of the permutation `(a1, a2, normal)` — the `τ` of the curl identity
/// `(∇×w e_c)_{a1} = τ ∂_{a2} w`, `(∇×w e_c)_{a2} = −τ ∂_{a1} w`.
#[inline]
pub fn curl_tau(normal: usize) -> f64 {
    match normal {
        1 => -1.0,
        _ => 1.0,
    }
}

pub struct HexRTk {
    order: usize,
    open: HexRtOpen,
}

impl HexRTk {
    /// MFEM `RT_HexahedronElement(p, GaussLobatto, IntegratedGLL)` — the
    /// LOR-compatible basis pair (`fem/lor/lor.cpp` `CheckBasisType`).
    ///
    /// Note the deliberate asymmetry with the 2-D [`QuadRTk`](super::QuadRTk),
    /// whose `new` is the GaussLegendre default: the hex element predates the
    /// D236 variant split and is the element the LOR stack (the permutation
    /// legs in `fem_space::lor` and the `lor_factory` RT test legs) was built
    /// and calibrated against, so it keeps the IntegratedGLL flavour and the
    /// MFEM-default GaussLegendre variant is the opt-in
    /// [`HexRTk::new_gauss_legendre`].
    pub fn new(p: usize) -> Self {
        HexRTk { order: p, open: HexRtOpen::IntegratedGLL }
    }

    /// MFEM `RT_HexahedronElement(p, GaussLobatto, GaussLegendre)` — the
    /// element `RT_FECollection(p, dim)` builds by default (`fe_coll.hpp`):
    /// the flavour behind every MFEM-saved RT field (VisIt data collections,
    /// `get-values`, `Project_RT`), with the nodal Gauss-Legendre open modes.
    /// Same dof count/layout and dof positions as [`HexRTk::new`]; only the
    /// open modes differ.
    pub fn new_gauss_legendre(p: usize) -> Self {
        HexRTk { order: p, open: HexRtOpen::GaussLegendre }
    }

    /// The `k+1` open 1-D modes along one axis at `x` (reference `[0,1]`).
    ///
    /// Both variants are MFEM's native `[0,1]` modes (D721 dropped the
    /// historical hex pull-back scale): a tensor mode is MFEM's reference
    /// value exactly, so the physical Piola transform `det(J)^-1·J·Φ_ref`
    /// equals MFEM's point for point.
    fn open_modes(&self, x: f64) -> Vec<f64> {
        match self.open {
            HexRtOpen::GaussLegendre => open_basis(self.order + 1, x).0,
            HexRtOpen::IntegratedGLL => {
                partial_open(&ClosedBasis::new_01(self.order + 1).eval_mfem(x).dc)
            }
        }
    }

    /// [`open_modes`](Self::open_modes) together with the open-mode
    /// derivatives `o'_i` (the `EvalIntegrated` second chain for IntegratedGLL,
    /// the Lagrange derivative for GaussLegendre) — the factors the curl
    /// differentiates.
    fn open_modes_and_derivs(&self, x: f64) -> (Vec<f64>, Vec<f64>) {
        match self.open {
            HexRtOpen::GaussLegendre => open_basis(self.order + 1, x),
            HexRtOpen::IntegratedGLL => {
                let v = ClosedBasis::new_01(self.order + 1).eval_mfem(x);
                (partial_open(&v.dc), partial_open(&v.d2c))
            }
        }
    }

    /// Per-DOF data of MFEM `RT_HexahedronElement::ProjectIntegrated`
    /// (`fe_rt.cpp:661-711`) — the *integrated* sub-cell face-flux
    /// functionals of the IntegratedGLL variant ([`Self::new`]): each local
    /// DOF is the integral of `f·(adj(J)·n̂)` over one boundary-parallel
    /// sub-cell face `[cp_j, cp_j+1] × [cp_l, cp_l+1]` (GLL sub-cells of the
    /// `[-1,1]` frame), sampled with MFEM `IntRules.Get(SQUARE, order)` — the
    /// `(k+3)/2`-point `[0,1]` Gauss-Legendre rule per axis (4.10 rules live
    /// on `[0,1]`), weight `(w_u·h_1)·(w_v·h_2)`.
    ///
    /// The list is in the element's local DOF order (entry `n` = local DOF
    /// `n`, the face-block/interior-block enumeration of
    /// [`VectorReferenceElement::eval_basis_vec`], which is MFEM's dof
    /// numbering — pinned by `d577_hex_igll_functionals_mfem_truth`), and
    /// `t` carries the `dof_map` flip (`±e_c`, MFEM's `nk`): face dofs take
    /// their outward normal, interior dofs flip on the closed block index
    /// `<= k/2` (fe_rt.cpp:407-430 — no odd-p supplement in 3-D).  A consumer
    /// scatters `global = sign · Σ w·f(x(ξ))·(adj(J)·t)` exactly as the quad
    /// IGLL path in `HDivSpace::interpolate_vector`.
    pub fn integrated_functionals(&self) -> Vec<IntegratedDofFunctional3D> {
        assert!(
            self.open == HexRtOpen::IntegratedGLL,
            "HexRTk::integrated_functionals requires the IntegratedGLL variant"
        );
        let k = self.order;
        let m = k + 1;
        // Closed GLL nodes of the degree-`k+1` closed basis on the `[0,1]`
        // frame: `k+2` points (`gll_nodes_01` takes the degree), the sub-cell
        // edges of the integrated dofs — the same nodes
        // `ClosedBasis::new_01(k + 1)` is nodal at.
        let cp = gll_nodes_01(k + 1);
        // MFEM `IntRules.Get(Geometry::SQUARE, p+1)`: per-axis SEGMENT rule
        // of order `p+1` (bit-identical `gauss_legendre_01`, D74).
        let npts = (k + 3) / 2;
        let (gx, gw) = crate::quadrature::gauss_legendre_01(npts);
        let mut out = Vec::with_capacity(self.n_dofs());
        // One functional per (component c, closed index ic, free-axis
        // sub-cell indices s1, s2); `flip` is MFEM's `dof_map` negative,
        // i.e. `t = -e_c` when set (face dofs: the outward normal).
        let mut push = |c: usize, ic: usize, s1: usize, s2: usize, flip: bool| {
            let h1 = cp[s1 + 1] - cp[s1];
            let h2 = cp[s2 + 1] - cp[s2];
            let (a1, a2) = free_axes(c);
            let mut samples = Vec::with_capacity(npts * npts);
            for (&u, &wu) in gx.iter().zip(&gw) {
                let q1 = cp[s1] + h1 * u;
                for (&v, &wv) in gx.iter().zip(&gw) {
                    let q2 = cp[s2] + h2 * v;
                    let mut pt = [0.0_f64; 3];
                    pt[c] = cp[ic];
                    pt[a1] = q1;
                    pt[a2] = q2;
                    samples.push((pt, wu * h1 * wv * h2));
                }
            }
            let mut t = [0.0_f64; 3];
            t[c] = if flip { -1.0 } else { 1.0 };
            out.push(IntegratedDofFunctional3D { samples, t });
        };
        // Face blocks in HEX_RT_FACES order; the sub-cell indices are the
        // *basis* open-axis indices (the slot reversals relabel the slot, not
        // the mode), the closed index is the face endpoint.
        for &(nc, at_max, _s, f1, f2) in &HEX_RT_FACES {
            let ic = if at_max { k + 1 } else { 0 };
            let flip = ic <= k / 2;
            for j in 0..m {
                let q = if f2 { m - 1 - j } else { j };
                for i in 0..m {
                    let p = if f1 { m - 1 - i } else { i };
                    push(nc, ic, p, q, flip);
                }
            }
        }
        // Interior blocks, MFEM loop order (closed-interior index innermost).
        if k >= 1 {
            for l in 0..m {
                for j in 0..m {
                    for i in 1..=k {
                        push(0, i, j, l, i <= k / 2);
                    }
                }
            }
            for l in 0..m {
                for j in 1..=k {
                    for i in 0..m {
                        push(1, j, i, l, j <= k / 2);
                    }
                }
            }
            for l in 1..=k {
                for j in 0..m {
                    for i in 0..m {
                        push(2, l, i, j, l <= k / 2);
                    }
                }
            }
        }
        debug_assert_eq!(out.len(), self.n_dofs());
        out
    }
}

impl VectorReferenceElement for HexRTk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        3 * (self.order + 1) * (self.order + 1) * (self.order + 2)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        values.fill(0.0);

        // Closed basis degree k+1 (k+2 GLL modes); open modes of the variant
        // (k+1 modes).
        let vx = ClosedBasis::new_01(k + 1).val_mfem(x);
        let vy = ClosedBasis::new_01(k + 1).val_mfem(y);
        let vz = ClosedBasis::new_01(k + 1).val_mfem(z);
        let (cx, cy, cz) = (&vx, &vy, &vz);
        let ox = self.open_modes(x);
        let oy = self.open_modes(y);
        let oz = self.open_modes(z);
        let m = k + 1;

        let mut off = 0usize;
        // Face blocks in HEX_FACES order; see `HEX_RT_FACES` for the frame.
        for &(nc, at_max, s, f1, f2) in &HEX_RT_FACES {
            let idx = if at_max { k + 1 } else { 0 };
            let (a1, a2) = free_axes(nc);
            let o1: &[f64] = match a1 {
                0 => &ox,
                1 => &oy,
                _ => &oz,
            };
            let o2: &[f64] = match a2 {
                0 => &ox,
                1 => &oy,
                _ => &oz,
            };
            let closed = match nc {
                0 => cx[idx],
                1 => cy[idx],
                _ => cz[idx],
            };
            for j in 0..m {
                let q = if f2 { m - 1 - j } else { j };
                for i in 0..m {
                    let p = if f1 { m - 1 - i } else { i };
                    // MFEM `RT_HexahedronElement::CalcVShape` writes the
                    // closed factor in the position of its own axis inside the
                    // literal product — `s*cx(i)*oy(j)*oz(k)` for the
                    // x-component, `s*ox(i)*cy(j)*oz(k)` for y and
                    // `s*ox(i)*oy(j)*cz(k)` for z — so the *rounding* of the
                    // triple product depends on which axis the closed factor
                    // sits on.  Reproducing that association (rather than the
                    // generic `s*closed*o1*o2`) is what makes the reference
                    // values bit-identical; the earlier generic order left a
                    // 1-ulp residue on the z-normal face blocks (D742: 168 of
                    // the 6912 (q,dof,comp) entries at RT1, visible as 4
                    // structural + 276 junk misses in the d680 mass gate).
                    let val = match nc {
                        0 => s * closed * o1[p] * o2[q],
                        1 => s * o1[p] * closed * o2[q],
                        _ => s * o1[p] * o2[q] * closed,
                    };
                    values[off * 3 + nc] = val;
                    off += 1;
                }
            }
        }
        debug_assert_eq!(off, 6 * m * m);

        // Interior blocks (k >= 1), MFEM loop order (closed-interior index
        // innermost): x: c_i(x)·o_j(y)·o_l(z), i = 1..=k.  D225: MFEM's
        // reference orientation flips (`dof_map = -1 - dof_map` for the
        // closed block index `<= k/2`) are baked in.
        if k >= 1 {
            for l in 0..m {
                for j in 0..m {
                    for i in 1..=k {
                        let s = if i <= k / 2 { -1.0 } else { 1.0 };
                        values[off * 3] = s * cx[i] * oy[j] * oz[l];
                        off += 1;
                    }
                }
            }
            // y: o_i(x)·c_j(y)·o_l(z), j = 1..=k.
            for l in 0..m {
                for j in 1..=k {
                    let s = if j <= k / 2 { -1.0 } else { 1.0 };
                    for i in 0..m {
                        values[off * 3 + 1] = s * ox[i] * cy[j] * oz[l];
                        off += 1;
                    }
                }
            }
            // z: o_i(x)·o_j(y)·c_l(z), l = 1..=k.
            for l in 1..=k {
                let s = if l <= k / 2 { -1.0 } else { 1.0 };
                for j in 0..m {
                    for i in 0..m {
                        values[off * 3 + 2] = s * ox[i] * oy[j] * cz[l];
                        off += 1;
                    }
                }
            }
        }
        debug_assert_eq!(off, self.n_dofs());
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let k = self.order;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        div_vals.fill(0.0);

        let vx = ClosedBasis::new_01(k + 1).eval_mfem(x);
        let vy = ClosedBasis::new_01(k + 1).eval_mfem(y);
        let vz = ClosedBasis::new_01(k + 1).eval_mfem(z);
        let (dcx, dcy, dcz) = (&vx.dc, &vy.dc, &vz.dc);
        let ox = self.open_modes(x);
        let oy = self.open_modes(y);
        let oz = self.open_modes(z);
        let m = k + 1;

        // The divergence only differentiates the single CLOSED factor.
        // Face blocks first, same face frame as eval_basis_vec.
        let mut off = 0usize;
        for &(nc, at_max, s, f1, f2) in &HEX_RT_FACES {
            let idx = if at_max { k + 1 } else { 0 };
            let (a1, a2) = free_axes(nc);
            let o1: &[f64] = match a1 {
                0 => &ox,
                1 => &oy,
                _ => &oz,
            };
            let o2: &[f64] = match a2 {
                0 => &ox,
                1 => &oy,
                _ => &oz,
            };
            let dclosed = match nc {
                0 => dcx[idx],
                1 => dcy[idx],
                _ => dcz[idx],
            };
            for j in 0..m {
                let q = if f2 { m - 1 - j } else { j };
                for i in 0..m {
                    let p = if f1 { m - 1 - i } else { i };
                    // Same positional association as `eval_basis_vec` (and as
                    // MFEM `RT_HexahedronElement::CalcDivShape`, which also
                    // writes the differentiated closed factor in its own axis
                    // position inside the literal product).
                    let val = match nc {
                        0 => s * dclosed * o1[p] * o2[q],
                        1 => s * o1[p] * dclosed * o2[q],
                        _ => s * o1[p] * o2[q] * dclosed,
                    };
                    div_vals[off] = val;
                    off += 1;
                }
            }
        }
        if k >= 1 {
            for l in 0..m {
                for j in 0..m {
                    for i in 1..=k {
                        let s = if i <= k / 2 { -1.0 } else { 1.0 };
                        div_vals[off] = s * dcx[i] * oy[j] * oz[l];
                        off += 1;
                    }
                }
            }
            for l in 0..m {
                for j in 1..=k {
                    let s = if j <= k / 2 { -1.0 } else { 1.0 };
                    for i in 0..m {
                        div_vals[off] = s * ox[i] * dcy[j] * oz[l];
                        off += 1;
                    }
                }
            }
            for l in 1..=k {
                let s = if l <= k / 2 { -1.0 } else { 1.0 };
                for j in 0..m {
                    for i in 0..m {
                        div_vals[off] = s * ox[i] * oy[j] * dcz[l];
                        off += 1;
                    }
                }
            }
        }
        debug_assert_eq!(off, self.n_dofs());
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let k = self.order;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        curl_vals.fill(0.0);

        let vx = ClosedBasis::new_01(k + 1).eval_mfem(x);
        let vy = ClosedBasis::new_01(k + 1).eval_mfem(y);
        let vz = ClosedBasis::new_01(k + 1).eval_mfem(z);
        let (cx, cy, cz) = (&vx.c, &vy.c, &vz.c);
        // The curl of an RT tensor function differentiates its two OPEN
        // factors (IntegratedGLL: o'_i = -Σ_{t<=i} c''_t; GaussLegendre: the
        // Lagrange derivative).
        let (ox, dox) = self.open_modes_and_derivs(x);
        let (oy, doy) = self.open_modes_and_derivs(y);
        let (oz, doz) = self.open_modes_and_derivs(z);
        let m = k + 1;

        let mut off = 0usize;
        // Face dofs, same face frame as eval_basis_vec.  With
        // `Φ = s·c(nc)·o_p(a1)·o_q(a2)·e_nc` the curl is
        // `(∇×Φ)_{a1} = τ s c o_p o'_q` and `(∇×Φ)_{a2} = −τ s c o'_p o_q`,
        // `τ = ε_{a1 a2 nc}` (only the two OPEN factors differentiate).
        for &(nc, at_max, s, f1, f2) in &HEX_RT_FACES {
            let idx = if at_max { k + 1 } else { 0 };
            let (a1, a2) = free_axes(nc);
            let tau = curl_tau(nc);
            let o1: &[f64] = match a1 {
                0 => &ox,
                1 => &oy,
                _ => &oz,
            };
            let o2: &[f64] = match a2 {
                0 => &ox,
                1 => &oy,
                _ => &oz,
            };
            let do1: &[f64] = match a1 {
                0 => &dox,
                1 => &doy,
                _ => &doz,
            };
            let do2: &[f64] = match a2 {
                0 => &dox,
                1 => &doy,
                _ => &doz,
            };
            let closed = match nc {
                0 => cx[idx],
                1 => cy[idx],
                _ => cz[idx],
            };
            for j in 0..m {
                let q = if f2 { m - 1 - j } else { j };
                for i in 0..m {
                    let p = if f1 { m - 1 - i } else { i };
                    curl_vals[off * 3 + a1] = tau * s * closed * o1[p] * do2[q];
                    curl_vals[off * 3 + a2] = -tau * s * closed * do1[p] * o2[q];
                    off += 1;
                }
            }
        }
        if k >= 1 {
            // x-comp interior: Phi = s·(c_i(x)·o_j(y)·o_l(z), 0, 0)
            //   curl = s·(0, c_i·o_j·o'_l, -c_i·o'_j·o_l)
            for l in 0..m {
                for j in 0..m {
                    for i in 1..=k {
                        let s = if i <= k / 2 { -1.0 } else { 1.0 };
                        curl_vals[off * 3 + 1] = s * cx[i] * oy[j] * doz[l];
                        curl_vals[off * 3 + 2] = -s * cx[i] * doy[j] * oz[l];
                        off += 1;
                    }
                }
            }
            // y-comp interior: Phi = s·(0, o_i(x)·c_j(y)·o_l(z), 0)
            //   curl = s·(-o_i·c_j·o'_l, 0, o'_i·c_j·o_l)
            for l in 0..m {
                for j in 1..=k {
                    let s = if j <= k / 2 { -1.0 } else { 1.0 };
                    for i in 0..m {
                        curl_vals[off * 3] = -s * ox[i] * cy[j] * doz[l];
                        curl_vals[off * 3 + 2] = s * dox[i] * cy[j] * oz[l];
                        off += 1;
                    }
                }
            }
            // z-comp interior: Phi = s·(0, 0, o_i(x)·o_j(y)·c_l(z))
            //   curl = s·(o_i·o'_j·c_l, -o'_i·o_j·c_l, 0)
            for l in 1..=k {
                let s = if l <= k / 2 { -1.0 } else { 1.0 };
                for j in 0..m {
                    for i in 0..m {
                        curl_vals[off * 3] = s * ox[i] * doy[j] * cz[l];
                        curl_vals[off * 3 + 1] = -s * dox[i] * oy[j] * cz[l];
                        off += 1;
                    }
                }
            }
        }
        debug_assert_eq!(off, self.n_dofs());
    }

    fn quadrature(&self, order: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::hex_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let k = self.order;
        let m = k + 1;
        // D721: the face/interior nodes are MFEM's `[0,1]³` positions
        // (`Nodes.IntPoint(idx).Set3(cp[i], op[j], op[k])`), so the general
        // path below serves `k = 0` too (a single open point at 0.5 per axis,
        // the face centre) — the historical `[-1,1]` special case is gone.
        let gl = crate::gll_basis::gl_nodes_01(m);
        let n = self.n_dofs();
        let mut c = Vec::with_capacity(n);
        // Face DOFs in HEX_FACES order.  Free coordinates sit at the open
        // points, the normal coordinate on the face; the free-axis
        // enumeration follows `HEX_RT_FACES` (MFEM's node convention, which
        // relabels the reversed faces together with their basis function).
        for &(nc, at_max, _s, f1, f2) in &HEX_RT_FACES {
            let cnorm = if at_max { 1.0 } else { 0.0 };
            let (a1, a2) = free_axes(nc);
            for j in 0..m {
                let q = if f2 { m - 1 - j } else { j };
                for i in 0..m {
                    let p = if f1 { m - 1 - i } else { i };
                    let mut x = [0.0_f64; 3];
                    x[nc] = cnorm;
                    x[a1] = gl[p];
                    x[a2] = gl[q];
                    c.push(x.to_vec());
                }
            }
        }
        // Interior DOFs (k >= 1), mirroring the eval blocks: the closed mode
        // in one direction sits at the interior GLL nodes, the two open modes
        // at the Gauss-Legendre (open) points.
        //
        // D351 (pre-existing, found while working D342): the closed-direction
        // node must come from the *GaussLobatto* family of the `k+2`-point rule
        // — MFEM `RT_HexahedronElement`'s
        // `Nodes.IntPoint(idx).Set3(cp[i], op[j], op[k])` with
        // `cp = poly1d.ClosedPoints(p + 1, cb_type)` (`fem/fe/fe_rt.cpp:437`),
        // where `i` runs `0..=p+1` (`fe_rt.cpp:431-434`, one entry per slot of
        // the `p+2`-wide closed axis) so `cp` must supply `p+2` points
        // — because that is the node the closed GLL basis `ClosedBasis::new_01(k+1)`
        // is nodal at (and the node `HDivSpace::interp_rows` samples).  The
        // old code took the GaussLegendre node (`gl_nodes(k+2)[i]`) instead,
        // which is a *different* point for every k >= 2 while coinciding at
        // k <= 1; sibling elements do it the MFEM way (`HexNDk::dof_layout`
        // `gll_nodes`, `QuadRTk::dof_coords` `gll_nodes(k + 2)`).  No RT
        // assembly path reads these coordinates (the interpolation engine and
        // the space use their own tables), so this is a labelling fix.
        let glc = gll_nodes_01(m);
        for l in 0..m {
            for j in 0..m {
                for i in 1..=k {
                    c.push(vec![glc[i], gl[j], gl[l]]);
                }
            }
        }
        for l in 0..m {
            for j in 1..=k {
                for i in 0..m {
                    c.push(vec![gl[i], glc[j], gl[l]]);
                }
            }
        }
        for l in 1..=k {
            for j in 0..m {
                for i in 0..m {
                    c.push(vec![gl[i], gl[j], glc[l]]);
                }
            }
        }
        while c.len() < n {
            c.push(vec![0.0, 0.0, 0.0]);
        }
        c
    }
}

/// Integrated (Gerritsma) open modes `o_i = -Σ_{j<=i} c'_j` from the closed
/// basis derivative array `d` — the open half of the IntegratedGLL variant
/// ([`HexRTk::new`]).
///
/// Scaled by 1/2 (the `[-1,1]` pull-back of MFEM's `[0,1]` functions; the
/// `d/dξ = ½·d/dt` chain inside `EvalIntegrated` adds another ½, so a tensor
/// mode carries two factors of ¼ and sits at `V_mfem/16`, `div_mfem/32` —
/// the same frame every other fem-rs hex vector family pairs with the
/// `[-1,1]` isoparametric Jacobian).
fn partial_open(d: &[f64]) -> Vec<f64> {
    let n = d.len() - 1;
    let mut o = vec![0.0_f64; n];
    if n == 0 {
        return o;
    }
    o[0] = -d[0];
    for i in 1..n {
        o[i] = o[i - 1] - d[i];
    }
    o
}

#[cfg(test)]
mod mfem_gl_dump;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_dofs() {
        assert_eq!(HexRTk::new(0).n_dofs(), 6);
        assert_eq!(HexRTk::new(1).n_dofs(), 36);
        assert_eq!(HexRTk::new(2).n_dofs(), 108);
        // D342: `3(p+1)^2(p+2)` — MFEM `RT_HexahedronElement`'s
        // `3*(p+1)*(p+1)*(p+2)` ctor argument, orders 3..=6 (the range the
        // space cap now allows).
        assert_eq!(HexRTk::new(3).n_dofs(), 240);
        assert_eq!(HexRTk::new(4).n_dofs(), 450);
        assert_eq!(HexRTk::new(5).n_dofs(), 756);
        assert_eq!(HexRTk::new(6).n_dofs(), 1176);
    }

    #[test]
    fn finite() {
        // D342: every order the space cap allows, both open-basis variants.
        for k in 0..=6 {
            let e = HexRTk::new(k);
            let n = e.n_dofs();
            let mut v = vec![0.0; n * 3];
            let mut d = vec![0.0; n];
            let mut c = vec![0.0; n * 3];
            for p in &[(0.0, 0.0, 0.0), (0.3, -0.5, 0.7), (-0.2, 0.4, -0.6)] {
                e.eval_basis_vec(&[p.0, p.1, p.2], &mut v);
                e.eval_div(&[p.0, p.1, p.2], &mut d);
                e.eval_curl(&[p.0, p.1, p.2], &mut c);
                for val in v.iter().chain(d.iter()).chain(c.iter()) {
                    assert!(val.is_finite());
                }
            }
        }
    }

    /// RT0 must be the unit-flux Whitney form: each face mode has exactly
    /// unit outward flux through its own face and zero flux through the other
    /// five (MFEM RT0 with the IntegratedGLL open basis), with constant
    /// reference divergence 1 (D721: the reference frame *is* MFEM's `[0,1]³`,
    /// so the reference value is the physical one — the historical `1/8`
    /// reference / `×32` physical pair was the `[-1,1]` pull-back).
    #[test]
    fn rt0_unit_face_fluxes() {
        let e = HexRTk::new(0);
        assert_eq!(e.n_dofs(), 6);
        let (g1, w1) = crate::quadrature::gauss_legendre_01(3);
        // Faces in HEX_FACES order: bottom z-, front y-, right x+, back y+,
        // left x-, top z+.
        let faces: [(f64, usize); 6] = [
            (0.0, 2),
            (0.0, 1),
            (1.0, 0),
            (1.0, 1),
            (0.0, 0),
            (1.0, 2),
        ];
        let normals: [[f64; 3]; 6] = [
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut flux = vec![vec![0.0_f64; 6]; 6];
        for (f, &(fc, axis)) in faces.iter().enumerate() {
            for (fg, &ga) in g1.iter().enumerate() {
                for (fh, &gb) in g1.iter().enumerate() {
                    let xi = match axis {
                        0 => [fc, ga, gb],
                        1 => [ga, fc, gb],
                        _ => [ga, gb, fc],
                    };
                    let wq = w1[fg] * w1[fh];
                    let n = e.n_dofs();
                    let mut v = vec![0.0; n * 3];
                    e.eval_basis_vec(&xi, &mut v);
                    for dof in 0..n {
                        flux[dof][f] += wq
                            * (v[dof * 3] * normals[f][0]
                                + v[dof * 3 + 1] * normals[f][1]
                                + v[dof * 3 + 2] * normals[f][2]);
                    }
                }
            }
        }
        for dof in 0..6 {
            for f in 0..6 {
                let want = if dof == f { 1.0 } else { 0.0 };
                // D721: the reference-frame flux *is* MFEM's unit face flux
                // on the `[0,1]³` frame (the historical `×4` physical
                // conversion died with the `[-1,1]` pull-back).
                assert!(
                    (flux[dof][f] - want).abs() < 1e-13,
                    "dof {dof} flux through face {f}: {} (want {want})",
                    flux[dof][f],
                );
            }
        }
        let mut d = vec![0.0; 6];
        e.eval_div(&[0.63, 0.29, 0.785], &mut d);
        for di in &d {
            // D721: reference divergence 1 — MFEM's RT0 hex div is the
            // constant 1 on `[0,1]³` (the `[-1,1]` frame's 1/32 was
            // `1/8 · 1/4`).
            assert!((di - 1.0).abs() < 1e-14, "div {di}");
        }
    }

    /// Divergence consistency: analytic div vs central finite difference.
    #[test]
    fn div_matches_finite_difference() {
        // D342: the order-generic derivative chains are also exercised above
        // the old hex cap (k = 3..=6) — the `interior` blocks grow as `k(k+1)^2`
        // per component, so the closed-index flip rule `i <= k/2` and the
        // interior enumeration are covered at both parities.
        for k in 0..=6 {
            let e = HexRTk::new(k);
            let n = e.n_dofs();
            let eps = 1e-6;
            let pt = [0.5685, 0.2935, 0.8105];
            let eval = |xi: &[f64], out: &mut Vec<f64>| {
                out.clear();
                out.resize(n * 3, 0.0);
                e.eval_basis_vec(xi, out);
            };
            let mut vp = Vec::new();
            let mut vm = Vec::new();
            let mut fd = vec![0.0_f64; n];
            for d in 0..3 {
                let mut xp = pt;
                let mut xm = pt;
                xp[d] += eps;
                xm[d] -= eps;
                eval(&xp, &mut vp);
                eval(&xm, &mut vm);
                for i in 0..n {
                    fd[i] += (vp[i * 3 + d] - vm[i * 3 + d]) / (2.0 * eps);
                }
            }
            let mut dd = vec![0.0; n];
            e.eval_div(&pt, &mut dd);
            for i in 0..n {
                let scale = 1.0 + dd[i].abs();
                assert!(
                    (dd[i] - fd[i]).abs() < 1e-5 * scale,
                    "k={k} div[{i}]: analytic {} vs fd {}",
                    dd[i],
                    fd[i]
                );
            }
        }
    }

    /// Curl consistency: analytic curl vs central finite difference.
    #[test]
    fn curl_matches_finite_difference() {
        // D342: extended to the full newly-allowed order range (see
        // `div_matches_finite_difference`).
        for k in 0..=6 {
            let e = HexRTk::new(k);
            let n = e.n_dofs();
            let eps = 1e-6;
            let pt = [0.5685, 0.2935, 0.8105];
            let eval = |xi: &[f64], out: &mut Vec<f64>| {
                out.clear();
                out.resize(n * 3, 0.0);
                e.eval_basis_vec(xi, out);
            };
            let mut vp = Vec::new();
            let mut vm = Vec::new();
            let mut fd = vec![0.0_f64; n * 3];
            for d in 0..3 {
                let mut xp = pt;
                let mut xm = pt;
                xp[d] += eps;
                xm[d] -= eps;
                eval(&xp, &mut vp);
                eval(&xm, &mut vm);
                for i in 0..n {
                    fd[i * 3] += (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * eps)
                        * if d == 1 { 1.0 } else { 0.0 };
                    fd[i * 3] -= (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * eps)
                        * if d == 2 { 1.0 } else { 0.0 };
                    fd[i * 3 + 1] += (vp[i * 3] - vm[i * 3]) / (2.0 * eps)
                        * if d == 2 { 1.0 } else { 0.0 };
                    fd[i * 3 + 1] -= (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * eps)
                        * if d == 0 { 1.0 } else { 0.0 };
                    fd[i * 3 + 2] += (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * eps)
                        * if d == 0 { 1.0 } else { 0.0 };
                    fd[i * 3 + 2] -= (vp[i * 3] - vm[i * 3]) / (2.0 * eps)
                        * if d == 1 { 1.0 } else { 0.0 };
                }
            }
            let mut cc = vec![0.0; n * 3];
            e.eval_curl(&pt, &mut cc);
            for i in 0..n * 3 {
                let scale = 1.0 + cc[i].abs();
                assert!(
                    (cc[i] - fd[i]).abs() < 1e-5 * scale,
                    "k={k} curl[{i}]: analytic {} vs fd {}",
                    cc[i],
                    fd[i]
                );
            }
        }
    }

    /// D236: the GaussLegendre (nodal) variant must reproduce MFEM 4.10's
    /// `RT_HexahedronElement(p, GaussLobatto, GaussLegendre)` — the element
    /// `RT_FECollection(p, dim)` builds by default — **slot for slot**, at the
    /// generic sample points of the round-40 probe.  Both sides enumerate the
    /// identical dof_map (round 40: the maps of the two ob_type variants are
    /// element-for-element equal), and MFEM's reference orientation flips are
    /// baked into both, so this is a direct per-slot value comparison — and
    /// after D721 it is an **exact** one: both sides evaluate the same `[0,1]³`
    /// reference functions at the same points, so `V_femrs = V_mfem` and
    /// `div_femrs = div_mfem` entry by entry (the historical `×4`/`×8`
    /// pull-back conversions are gone with the `[-1,1]` frame).
    ///
    /// D342 extends the loop to `k = 3` (the first order the space cap used to
    /// reject); the table was regenerated with the identical probe/points.
    #[test]
    fn rt_gl_matches_mfem_nodal_dump() {
        // MFEM `[0,1]³` sample points == the fem-rs reference points (D721).
        let pts = [[0.65_f64, 0.7, 0.75], [0.36, 0.71, 0.44]];
        for k in 1..=3usize {
            let e = HexRTk::new_gauss_legendre(k);
            let n = e.n_dofs();
            let mut v = vec![0.0_f64; n * 3];
            let mut d = vec![0.0_f64; n];
            for (q, t) in pts.iter().enumerate() {
                let xi = [t[0], t[1], t[2]];
                e.eval_basis_vec(&xi, &mut v);
                e.eval_div(&xi, &mut d);
                for (i, row) in mfem_gl_dump::v(k, q).iter().enumerate() {
                    for c in 0..3 {
                        let got = v[i * 3 + c];
                        assert!(
                            (got - row[c]).abs() < 1e-12 * (1.0 + row[c].abs()),
                            "RT{k} v{q} slot {i} comp {c}: rust {got} vs mfem {}",
                            row[c]
                        );
                    }
                }
                for (i, want) in mfem_gl_dump::div(k, q).iter().enumerate() {
                    let got = d[i];
                    assert!(
                        (got - want).abs() < 1e-12 * (1.0 + want.abs()),
                        "RT{k} div{q} slot {i}: rust {got} vs mfem {want}"
                    );
                }
            }
        }
    }

    /// D342/D721: the unit-cube scaling ratios are *measured*, not assumed —
    /// for every slot/component whose fem-rs value is not negligible the
    /// quotient `V_mfem/V_femrs` (resp. `div_mfem/div_femrs`) must be exactly
    /// **1** to 1e-12 relative, because both sides now live on the same
    /// `[0,1]³` reference frame.  This is the independent check that no
    /// order-dependent or wrong global factor survived the D721 flip (the
    /// historical values were 4 and 8; a *reconstruction* test alone could not
    /// see a global factor because the dual matrix is built from the same
    /// basis).
    #[test]
    fn d342_rt_scaling_factors_measured() {
        let pts = [[0.65_f64, 0.7, 0.75], [0.36, 0.71, 0.44]];
        let mut worst_v: f64 = 0.0;
        let mut worst_x: f64 = 0.0;
        let mut n_cmp = 0usize;
        for k in 1..=3usize {
            let e = HexRTk::new_gauss_legendre(k);
            let n = e.n_dofs();
            let mut v = vec![0.0_f64; n * 3];
            let mut d = vec![0.0_f64; n];
            for (q, t) in pts.iter().enumerate() {
                let xi = [t[0], t[1], t[2]];
                e.eval_basis_vec(&xi, &mut v);
                e.eval_div(&xi, &mut d);
                for (i, row) in mfem_gl_dump::v(k, q).iter().enumerate() {
                    for c in 0..3 {
                        let rust = v[i * 3 + c];
                        if rust.abs() < 1e-9 {
                            continue;
                        }
                        worst_v = worst_v.max((row[c] / rust - 1.0).abs());
                        n_cmp += 1;
                    }
                }
                for (i, want) in mfem_gl_dump::div(k, q).iter().enumerate() {
                    if d[i].abs() < 1e-9 {
                        continue;
                    }
                    worst_x = worst_x.max((want / d[i] - 1.0).abs());
                    n_cmp += 1;
                }
            }
        }
        // Ratio deviations are limited only by the `%.17g` round-trip of the
        // dump (~1e-16); the count is printed so the sample cannot silently
        // thin out.
        println!(
            "D342/D721 measured unit-cube scaling: V ratio = 1 +- {worst_v:.3e}, \
             div ratio = 1 +- {worst_x:.3e} over {n_cmp} entries"
        );
        assert!(n_cmp > 1000, "sample too thin: {n_cmp} entries");
        assert!(worst_v < 1e-12, "V_mfem/V_femrs deviates from 1 by {worst_v}");
        assert!(worst_x < 1e-12, "div_mfem/div_femrs deviates from 1 by {worst_x}");
    }

    /// At order 0 both open-basis variants span the same MFEM tensor functions
    /// (`C_i(t)·1·1` — the single open mode is the constant 1 in MFEM's
    /// normalisation for GaussLegendre *and* IntegratedGLL), so on the `[0,1]³`
    /// frame the two reference bases are now **identical**, componentwise and
    /// slot for slot (the historical `4×` scale gap was the `[-1,1]`
    /// pull-back; D721 removed it).
    #[test]
    fn rt0_gl_equals_rt0_igll_reference() {
        let gl = HexRTk::new_gauss_legendre(0);
        let igll = HexRTk::new(0);
        let mut vg = vec![0.0_f64; 18];
        let mut vi = vec![0.0_f64; 18];
        let mut dg = vec![0.0_f64; 6];
        let mut di = vec![0.0_f64; 6];
        for p in &[(0.13, -0.42, 0.57), (-0.7, 0.2, 0.9)] {
            gl.eval_basis_vec(&[p.0, p.1, p.2], &mut vg);
            igll.eval_basis_vec(&[p.0, p.1, p.2], &mut vi);
            gl.eval_div(&[p.0, p.1, p.2], &mut dg);
            igll.eval_div(&[p.0, p.1, p.2], &mut di);
            for i in 0..18 {
                assert!(
                    (vg[i] - vi[i]).abs() < 1e-14,
                    "slot {i}: gl {} vs igll {}",
                    vg[i],
                    vi[i]
                );
            }
            for i in 0..6 {
                assert!((dg[i] - di[i]).abs() < 1e-14);
            }
        }
    }

    /// D236/D721: the GaussLegendre variant's face modes have MFEM's unit
    /// outward flux on the `[0,1]³` reference frame — the plain reference
    /// integral `∫∫ ψ_ref dξ dη` is ±1 per face mode, and both variants now
    /// integrate to 1 (the `4×` gap of the `[-1,1]` frame is gone).
    #[test]
    fn rt0_gl_unit_face_fluxes() {
        let e = HexRTk::new_gauss_legendre(0);
        let (g1, w1) = crate::quadrature::gauss_legendre_01(4);
        let faces: [(f64, usize); 6] = [
            (0.0, 2),
            (0.0, 1),
            (1.0, 0),
            (1.0, 1),
            (0.0, 0),
            (1.0, 2),
        ];
        let normals: [[f64; 3]; 6] = [
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut flux = vec![vec![0.0_f64; 6]; 6];
        for (f, &(fc, axis)) in faces.iter().enumerate() {
            for (fg, &ga) in g1.iter().enumerate() {
                for (fh, &gb) in g1.iter().enumerate() {
                    let xi = match axis {
                        0 => [fc, ga, gb],
                        1 => [ga, fc, gb],
                        _ => [ga, gb, fc],
                    };
                    let wq = w1[fg] * w1[fh];
                    let mut v = vec![0.0; 18];
                    e.eval_basis_vec(&xi, &mut v);
                    for dof in 0..6 {
                        flux[dof][f] += wq
                            * (v[dof * 3] * normals[f][0]
                                + v[dof * 3 + 1] * normals[f][1]
                                + v[dof * 3 + 2] * normals[f][2]);
                    }
                }
            }
        }
        for dof in 0..6 {
            for f in 0..6 {
                let want = if dof == f { 1.0 } else { 0.0 };
                assert!(
                    (flux[dof][f] - want).abs() < 1e-13,
                    "dof {dof} flux through face {f}: {} (want {want})",
                    flux[dof][f]
                );
            }
        }
    }

    /// Divergence/curl consistency of the GaussLegendre variant: analytic
    /// derivatives vs central finite differences of the basis values (the
    /// nodal open modes have a different derivative chain than the
    /// IntegratedGLL ones).
    #[test]
    fn rt_gl_div_curl_match_finite_difference() {
        // D342: extended to the full newly-allowed order range.
        for k in 0..=6usize {
            let e = HexRTk::new_gauss_legendre(k);
            let n = e.n_dofs();
            let eps = 1e-6;
            let pt = [0.5685, 0.2935, 0.8105];
            let eval = |xi: &[f64], out: &mut Vec<f64>| {
                out.clear();
                out.resize(n * 3, 0.0);
                e.eval_basis_vec(xi, out);
            };
            let mut vp = Vec::new();
            let mut vm = Vec::new();
            let mut fd_div = vec![0.0_f64; n];
            let mut fd_curl = vec![0.0_f64; n * 3];
            for d in 0..3 {
                let mut xp = pt;
                let mut xm = pt;
                xp[d] += eps;
                xm[d] -= eps;
                eval(&xp, &mut vp);
                eval(&xm, &mut vm);
                for i in 0..n {
                    fd_div[i] += (vp[i * 3 + d] - vm[i * 3 + d]) / (2.0 * eps);
                }
                for i in 0..n {
                    fd_curl[i * 3] += (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * eps)
                        * if d == 1 { 1.0 } else { 0.0 };
                    fd_curl[i * 3] -= (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * eps)
                        * if d == 2 { 1.0 } else { 0.0 };
                    fd_curl[i * 3 + 1] += (vp[i * 3] - vm[i * 3]) / (2.0 * eps)
                        * if d == 2 { 1.0 } else { 0.0 };
                    fd_curl[i * 3 + 1] -= (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * eps)
                        * if d == 0 { 1.0 } else { 0.0 };
                    fd_curl[i * 3 + 2] += (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * eps)
                        * if d == 0 { 1.0 } else { 0.0 };
                    fd_curl[i * 3 + 2] -= (vp[i * 3] - vm[i * 3]) / (2.0 * eps)
                        * if d == 1 { 1.0 } else { 0.0 };
                }
            }
            let mut dd = vec![0.0; n];
            e.eval_div(&pt, &mut dd);
            for i in 0..n {
                assert!(
                    (dd[i] - fd_div[i]).abs() < 1e-5 * (1.0 + dd[i].abs()),
                    "k={k} div[{i}]: analytic {} vs fd {}",
                    dd[i],
                    fd_div[i]
                );
            }
            let mut cc = vec![0.0; n * 3];
            e.eval_curl(&pt, &mut cc);
            for i in 0..n * 3 {
                assert!(
                    (cc[i] - fd_curl[i]).abs() < 1e-5 * (1.0 + cc[i].abs()),
                    "k={k} curl[{i}]: analytic {} vs fd {}",
                    cc[i],
                    fd_curl[i]
                );
            }
        }
    }
}
