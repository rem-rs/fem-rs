//! DoF (degree-of-freedom) transformation family — 1:1 port of MFEM
//! `fem/doftrans.hpp` / `fem/doftrans.cpp` (MFEM 4.10).
//!
//! Transformations map element-local DoFs to the global (mesh-aligned) DoF
//! convention so that basis functions in neighbouring elements align.  With
//! the primal operator `T` (local → global) the four actions are
//!
//! | vector       | transform            | matrix |
//! |--------------|----------------------|--------|
//! | primal (GridFunction) | `TransformPrimal`    | `T`     |
//! | primal inverse        | `InvTransformPrimal` | `T⁻¹`   |
//! | dual (LinearForm)     | `TransformDual`      | `T⁻ᵀ`   |
//! | dual inverse          | `InvTransformDual`   | `Tᵀ`    |
//!
//! and bilinear-form blocks transform as `A_t = T⁻ᵀ A T⁻¹` (dual rows and
//! dual columns); discrete-operator ranges use the primal on columns and the
//! dual on rows (`D_t = T D T⁻¹`).
//!
//! This module ports:
//!
//! * [`StatelessDofTransformation`] — orientation-parameterised transforms
//!   (MFEM `StatelessDofTransformation`),
//! * [`NdDofTransformation`] — Nedelec rotations: the 2×2 face-pair
//!   transforms for the 6 triangle-face orientations (MFEM
//!   `ND_DofTransformation::T_data` / `TInv_data`, verbatim), instantiated
//!   for tri / tet / wedge / pyramid element layouts (MFEM
//!   `ND_TriDofTransformation`, `ND_TetDofTransformation`,
//!   `ND_WedgeDofTransformation`, `ND_PyramidDofTransformation`),
//! * [`DofTransformation`] — the `vdim`/`ordering`-aware wrapper (MFEM
//!   `DofTransformation`) plus the mixed (row/column) transform free
//!   functions [`transform_primal_pairs`] / [`transform_dual_pairs`]
//!   (MFEM `TransformPrimal`/`TransformDual(ran, dom, elmat)`),
//! * [`rt_trace_face_sign`] — the RT/trace face-orientation sign.  MFEM has
//!   no `RT_DofTransformation`: RT (H(div)) face DoFs are normal moments,
//!   whose basis is orientation-invariant; the only cross-side semantics is
//!   the ±1 sign between the two elements' outward normals (applied inside
//!   MFEM's `TraceIntegrator` / `NormalTraceIntegrator` /
//!   `TangentTraceIntegrator` through the `Elem1`/`Elem2` check).  The DPG
//!   skeleton assembly in `fem-assembly` uses this helper for exactly that.

/// Face geometry tag for the transformation tables (MFEM `Geometry::Type`
/// restricted to the two face kinds ND spaces carry DoFs on).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceGeom {
    /// Triangular face (2-component DoF pairs, rotated by `T(ori)`).
    Triangle,
    /// Quadrilateral face (tensor DoFs, orientation-invariant).
    Quadrilateral,
}

/// Orientation-parameterised DoF transformation (MFEM
/// `StatelessDofTransformation`).  "Stateless" = no element-specific data
/// beyond the face orientation array `fo` passed to each call.
pub trait StatelessDofTransformation: Send + Sync {
    /// Number of local DoFs covered by the transformation.
    fn size(&self) -> usize;
    /// `true` when the transformation is the identity.
    fn is_identity(&self) -> bool;
    /// Local → global (MFEM `TransformPrimal(Fo, v)`).
    fn transform_primal(&self, fo: &[i32], v: &mut [f64]);
    /// Global → local (MFEM `InvTransformPrimal(Fo, v)`).
    fn inv_transform_primal(&self, fo: &[i32], v: &mut [f64]);
    /// Dual (LinearForm) transform `T⁻ᵀ` (MFEM `TransformDual(Fo, v)`).
    fn transform_dual(&self, fo: &[i32], v: &mut [f64]);
    /// Dual inverse `Tᵀ` (MFEM `InvTransformDual(Fo, v)`).
    fn inv_transform_dual(&self, fo: &[i32], v: &mut [f64]);
}

/// Nedelec DoF transformation (MFEM `ND_DofTransformation`).
///
/// ND face-interior DoFs on triangular faces come in pairs sharing an
/// interpolation point; the pair direction depends on the face orientation
/// and the mapping between the two conventions is a series of 2×2 transforms
/// selected by the orientation code `fo[f] ∈ 0..6`.
///
/// DoF layout (per element): `nedges × nedofs` edge DoFs, then per face (in
/// local face order) `ntdofs` (tri) or `nqdofs` (quad) face DoFs, then
/// interior DoFs (not touched by the face transforms).
pub struct NdDofTransformation {
    order: usize,
    /// DoFs per edge.
    nedofs: usize,
    /// DoFs per triangular face (`p(p−1)`).
    ntdofs: usize,
    /// DoFs per quadrilateral face (`2p(p−1)`).
    nqdofs: usize,
    nedges: usize,
    nfaces: usize,
    ftypes: &'static [FaceGeom],
    size_: usize,
}

/// MFEM `ND_DofTransformation::T_data` (verbatim, column-major 2×2 slices,
/// orientation `ori` occupies `T_data[4*ori..4*ori+4]`).
const T_DATA: [f64; 24] = [
    1.0, 0.0, 0.0, 1.0, //
    -1.0, -1.0, 0.0, 1.0, //
    0.0, 1.0, -1.0, -1.0, //
    1.0, 0.0, -1.0, -1.0, //
    -1.0, -1.0, 1.0, 0.0, //
    0.0, 1.0, 1.0, 0.0,
];

/// MFEM `ND_DofTransformation::TInv_data` (verbatim, column-major).
const TINV_DATA: [f64; 24] = [
    1.0, 0.0, 0.0, 1.0, //
    -1.0, -1.0, 0.0, 1.0, //
    -1.0, -1.0, 1.0, 0.0, //
    1.0, 0.0, -1.0, -1.0, //
    0.0, 1.0, -1.0, -1.0, //
    0.0, 1.0, 1.0, 0.0,
];

/// Extract the 2×2 transform for orientation `ori` as a row-major matrix.
fn face_matrix(data: &[f64; 24], ori: usize) -> [[f64; 2]; 2] {
    let d = &data[ori * 4..ori * 4 + 4];
    // column-major (i0j0, i1j0, i0j1, i1j1) → row-major [[a00, a01], [a10, a11]]
    [[d[0], d[2]], [d[1], d[3]]]
}

impl NdDofTransformation {
    fn build(size: usize, order: usize, nedges: usize, nfaces: usize, ftypes: &'static [FaceGeom]) -> Self {
        debug_assert_eq!(ftypes.len(), nfaces);
        Self {
            order,
            nedofs: order,
            ntdofs: order * (order.saturating_sub(1)),
            nqdofs: 2 * order * (order.saturating_sub(1)),
            nedges,
            nfaces,
            ftypes,
            size_: size,
        }
    }

    /// 2-D Nedelec on triangles (MFEM `ND_TriDofTransformation`).
    pub fn new_tri(order: usize) -> Self {
        Self::build(order * (order + 2), order, 3, 1, &[FaceGeom::Triangle])
    }

    /// Nedelec on tetrahedra (MFEM `ND_TetDofTransformation`).
    pub fn new_tet(order: usize) -> Self {
        Self::build(
            order * (order + 2) * (order + 3) / 2,
            order,
            6,
            4,
            &[FaceGeom::Triangle; 4],
        )
    }

    /// Nedelec on prisms (MFEM `ND_WedgeDofTransformation`); face types as
    /// MFEM `Geometry::Constants<Geometry::PRISM>::FaceTypes`
    /// (TRI, TRI, QUAD, QUAD, QUAD).
    pub fn new_wedge(order: usize) -> Self {
        Self::build(
            3 * order * ((order + 1) * (order + 2)) / 2,
            order,
            9,
            5,
            &[
                FaceGeom::Triangle,
                FaceGeom::Triangle,
                FaceGeom::Quadrilateral,
                FaceGeom::Quadrilateral,
                FaceGeom::Quadrilateral,
            ],
        )
    }

    /// Nedelec on pyramids (MFEM `ND_PyramidDofTransformation`); face types
    /// as MFEM `Geometry::Constants<Geometry::PYRAMID>::FaceTypes`
    /// (QUAD, TRI, TRI, TRI, TRI).
    pub fn new_pyramid(order: usize) -> Self {
        Self::build(
            2 * order * (order * (order + 1) + 2),
            order,
            8,
            5,
            &[
                FaceGeom::Quadrilateral,
                FaceGeom::Triangle,
                FaceGeom::Triangle,
                FaceGeom::Triangle,
                FaceGeom::Triangle,
            ],
        )
    }

    /// Basis order.
    pub fn order(&self) -> usize {
        self.order
    }

    /// 2×2 primal face transform for orientation `ori` (MFEM
    /// `GetFaceTransform`), row-major.
    pub fn face_transform(ori: usize) -> [[f64; 2]; 2] {
        face_matrix(&T_DATA, ori)
    }

    /// 2×2 inverse primal face transform (MFEM `GetFaceInverseTransform`).
    pub fn face_inverse_transform(ori: usize) -> [[f64; 2]; 2] {
        face_matrix(&TINV_DATA, ori)
    }

    /// Face-dof block offset of local face `f` (edge block precedes).
    fn face_offset(&self, f: usize) -> usize {
        let mut base = self.nedges * self.nedofs;
        for g in &self.ftypes[..f] {
            base += match g {
                FaceGeom::Triangle => self.ntdofs,
                FaceGeom::Quadrilateral => self.nqdofs,
            };
        }
        base
    }

    /// Apply `op(ori)` to every triangular-face DoF pair starting at `v[of]`.
    fn transform_face_pairs(&self, fo: &[i32], v: &mut [f64], data: &[f64; 24], transpose: bool) {
        debug_assert!(fo.len() >= self.nfaces);
        for f in 0..self.nfaces {
            if self.ftypes[f] != FaceGeom::Triangle {
                continue;
            }
            let of = self.face_offset(f);
            let m = face_matrix(data, fo[f] as usize);
            for i in 0..self.ntdofs / 2 {
                let (x, y) = (v[of + 2 * i], v[of + 2 * i + 1]);
                if transpose {
                    // v ← Mᵀ v
                    v[of + 2 * i] = m[0][0] * x + m[1][0] * y;
                    v[of + 2 * i + 1] = m[0][1] * x + m[1][1] * y;
                } else {
                    // v ← M v
                    v[of + 2 * i] = m[0][0] * x + m[0][1] * y;
                    v[of + 2 * i + 1] = m[1][0] * x + m[1][1] * y;
                }
            }
        }
    }
}

impl StatelessDofTransformation for NdDofTransformation {
    fn size(&self) -> usize {
        self.size_
    }

    fn is_identity(&self) -> bool {
        self.ntdofs < 2
    }

    fn transform_primal(&self, fo: &[i32], v: &mut [f64]) {
        if self.is_identity() {
            return;
        }
        self.transform_face_pairs(fo, v, &T_DATA, false);
    }

    fn inv_transform_primal(&self, fo: &[i32], v: &mut [f64]) {
        if self.is_identity() {
            return;
        }
        self.transform_face_pairs(fo, v, &TINV_DATA, false);
    }

    fn transform_dual(&self, fo: &[i32], v: &mut [f64]) {
        if self.is_identity() {
            return;
        }
        self.transform_face_pairs(fo, v, &TINV_DATA, true);
    }

    fn inv_transform_dual(&self, fo: &[i32], v: &mut [f64]) {
        if self.is_identity() {
            return;
        }
        self.transform_face_pairs(fo, v, &T_DATA, true);
    }
}

/// Component ordering of a vector-valued element block (MFEM `Ordering`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    /// Component-major blocks: `v[c*size + j]` (MFEM `Ordering::byNODES`).
    ByNodes,
    /// Interleaved: `v[j*vdim + c]` (MFEM `Ordering::byVDIM`).
    ByVdim,
}

/// Orientation-carrying DoF transformation wrapper (MFEM
/// `DofTransformation`): stores the face orientations of the current element
/// and applies the nested [`StatelessDofTransformation`] to `vdim`-expanded
/// vectors and dense blocks.
pub struct DofTransformation<'a> {
    fo: Vec<i32>,
    dof_trans: Option<&'a dyn StatelessDofTransformation>,
    vdim: usize,
    ordering: Ordering,
}

impl<'a> DofTransformation<'a> {
    /// Unconfigured wrapper (identity until
    /// [`Self::set_dof_transformation`]).
    pub fn new() -> Self {
        Self { fo: Vec::new(), dof_trans: None, vdim: 1, ordering: Ordering::ByNodes }
    }

    /// Wrapper with a known nested transformation (MFEM constructor).
    pub fn with_transformation(dof_trans: &'a dyn StatelessDofTransformation) -> Self {
        Self { fo: Vec::new(), dof_trans: Some(dof_trans), vdim: 1, ordering: Ordering::ByNodes }
    }

    /// Face orientations of the current element (MFEM `SetFaceOrientations`).
    pub fn set_face_orientations(&mut self, fo: Vec<i32>) {
        self.fo = fo;
    }

    /// Set / replace the nested transformation (MFEM
    /// `SetDofTransformation`).
    pub fn set_dof_transformation(&mut self, dof_trans: Option<&'a dyn StatelessDofTransformation>) {
        self.dof_trans = dof_trans;
    }

    /// Set vdim + ordering (MFEM `SetVDim`).
    pub fn set_vdim(&mut self, vdim: usize, ordering: Ordering) {
        self.vdim = vdim;
        self.ordering = ordering;
    }

    /// Nested transformation, if any.
    pub fn get_dof_transformation(&self) -> Option<&'a dyn StatelessDofTransformation> {
        self.dof_trans
    }

    /// `true` when no transformation is configured or it is the identity.
    pub fn is_identity(&self) -> bool {
        match self.dof_trans {
            None => true,
            Some(t) => t.is_identity(),
        }
    }

    /// Raw size of one component's DoF block.
    pub fn size(&self) -> usize {
        self.dof_trans.map_or(0, |t| t.size())
    }

    fn map_into_component(
        &self,
        v: &mut [f64],
        pick: fn(&dyn StatelessDofTransformation, &[i32], &mut [f64]),
    ) {
        let t = match self.dof_trans {
            Some(t) if !t.is_identity() => t,
            _ => return,
        };
        let size = t.size();
        if self.vdim == 1 || self.ordering == Ordering::ByNodes {
            for c in 0..self.vdim {
                pick(t, &self.fo, &mut v[c * size..(c + 1) * size]);
            }
        } else {
            let mut vec = vec![0.0_f64; size];
            for c in 0..self.vdim {
                for (j, slot) in vec.iter_mut().enumerate() {
                    *slot = v[j * self.vdim + c];
                }
                pick(t, &self.fo, &mut vec);
                for (j, &val) in vec.iter().enumerate() {
                    v[j * self.vdim + c] = val;
                }
            }
        }
    }

    /// Local → global over the `vdim`-expanded block (MFEM
    /// `DofTransformation::TransformPrimal`).
    pub fn transform_primal(&self, v: &mut [f64]) {
        self.map_into_component(v, |t, fo, x| t.transform_primal(fo, x));
    }

    /// Global → local (MFEM `DofTransformation::InvTransformPrimal`).
    pub fn inv_transform_primal(&self, v: &mut [f64]) {
        self.map_into_component(v, |t, fo, x| t.inv_transform_primal(fo, x));
    }

    /// Dual transform `T⁻ᵀ` (MFEM `DofTransformation::TransformDual`).
    pub fn transform_dual(&self, v: &mut [f64]) {
        self.map_into_component(v, |t, fo, x| t.transform_dual(fo, x));
    }

    /// Dual inverse `Tᵀ` (MFEM `DofTransformation::InvTransformDual`).
    pub fn inv_transform_dual(&self, v: &mut [f64]) {
        self.map_into_component(v, |t, fo, x| t.inv_transform_dual(fo, x));
    }

    /// Transform the rows of a row-major `height × width` block of dual DoFs
    /// (MFEM `TransformDualRows`); rows are in this space's convention.
    pub fn transform_dual_rows(&self, v: &mut [f64], height: usize, width: usize) {
        if self.is_identity() {
            return;
        }
        for r in 0..height {
            let mut row = v[r * width..(r + 1) * width].to_vec();
            self.transform_dual(&mut row);
            v[r * width..(r + 1) * width].copy_from_slice(&row);
        }
    }

    /// Transform the columns of a row-major `height × width` block of dual
    /// DoFs (MFEM `TransformDualCols`); columns are in this space's
    /// convention.
    pub fn transform_dual_cols(&self, v: &mut [f64], height: usize, width: usize) {
        if self.is_identity() {
            return;
        }
        let mut col = vec![0.0_f64; height];
        for c in 0..width {
            for (i, slot) in col.iter_mut().enumerate() {
                *slot = v[i * width + c];
            }
            self.transform_dual(&mut col);
            for (i, &val) in col.iter().enumerate() {
                v[i * width + c] = val;
            }
        }
    }

    /// Full block transform of dual DoFs (MFEM `DofTransformation::
    /// TransformDual(V)`): dual columns then dual rows.
    pub fn transform_dual_matrix(&self, v: &mut [f64], height: usize, width: usize) {
        self.transform_dual_cols(v, height, width);
        self.transform_dual_rows(v, height, width);
    }
}

impl Default for DofTransformation<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Mixed-space primal transform of an element block from a discrete
/// interpolator (MFEM `TransformPrimal(ran, dom, elmat)`): primal on the
/// range (columns, MFEM `TransformPrimalCols`) and dual on the domain
/// (rows, MFEM `TransformDualRows`).
pub fn transform_primal_pairs(
    ran: &DofTransformation,
    dom: &DofTransformation,
    elmat: &mut [f64],
    height: usize,
    width: usize,
) {
    if !ran.is_identity() {
        let mut col = vec![0.0_f64; height];
        for c in 0..width {
            for (i, slot) in col.iter_mut().enumerate() {
                *slot = elmat[i * width + c];
            }
            ran.transform_primal(&mut col);
            for (i, &val) in col.iter().enumerate() {
                elmat[i * width + c] = val;
            }
        }
    }
    if !dom.is_identity() {
        dom.transform_dual_rows(elmat, height, width);
    }
}

/// Mixed-space dual transform of an element block (MFEM
/// `TransformDual(ran, dom, elmat)`): dual on the range (columns, MFEM
/// `TransformDualCols`) and dual on the domain (rows, MFEM
/// `TransformDualRows`).
pub fn transform_dual_pairs(
    ran: &DofTransformation,
    dom: &DofTransformation,
    elmat: &mut [f64],
    height: usize,
    width: usize,
) {
    if !ran.is_identity() {
        let mut col = vec![0.0_f64; height];
        for c in 0..width {
            for (i, slot) in col.iter_mut().enumerate() {
                *slot = elmat[i * width + c];
            }
            ran.transform_dual(&mut col);
            for (i, &val) in col.iter().enumerate() {
                elmat[i * width + c] = val;
            }
        }
    }
    if !dom.is_identity() {
        dom.transform_dual_rows(elmat, height, width);
    }
}

/// RT / trace face-orientation sign (MFEM applies this inside
/// `TraceIntegrator`, `NormalTraceIntegrator` and `TangentTraceIntegrator`
/// through the `Elem1`/`Elem2` check).
///
/// MFEM has no `RT_DofTransformation`: RT (H(div)) face DoFs are normal
/// moments with an orientation-invariant basis, so the only cross-side
/// semantics is the sign between the two elements' outward normals.  The
/// canonical face orientation follows MFEM's global face storage — the face
/// is generated by (and directed along) its first adjacent element, MFEM
/// `Elem1` (`Mesh::FaceInfo`: "`Elem1No` always refers to the element that
/// generated the face").  An element contributes with `+1` when its local
/// face direction matches the canonical direction (`Elem1`) and `−1` when it
/// is reversed (`Elem2`).
#[must_use]
pub fn rt_trace_face_sign(face_orientation: i32) -> f64 {
    if face_orientation < 0 {
        -1.0
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T(ori) and TInv(ori) are exact inverses for all 6 orientations.
    #[test]
    fn t_and_tinv_are_inverse_pairs() {
        for ori in 0..6usize {
            for (a, b) in [(T_DATA, TINV_DATA), (TINV_DATA, T_DATA)] {
                let m = face_matrix(&a, ori);
                let minv = face_matrix(&b, ori);
                let prod = [
                    [
                        m[0][0] * minv[0][0] + m[0][1] * minv[1][0],
                        m[0][0] * minv[0][1] + m[0][1] * minv[1][1],
                    ],
                    [
                        m[1][0] * minv[0][0] + m[1][1] * minv[1][0],
                        m[1][0] * minv[0][1] + m[1][1] * minv[1][1],
                    ],
                ];
                assert!((prod[0][0] - 1.0).abs() < 1e-14 && (prod[0][1]).abs() < 1e-14, "ori {ori}");
                assert!((prod[1][0]).abs() < 1e-14 && (prod[1][1] - 1.0).abs() < 1e-14, "ori {ori}");
            }
        }
    }

    /// Known orientation tables: T(0) = I, T(1) column-major (−1,−1,0,1).
    #[test]
    fn t_tables_match_cpp() {
        let t0 = NdDofTransformation::face_transform(0);
        assert_eq!(t0, [[1.0, 0.0], [0.0, 1.0]]);
        let t1 = NdDofTransformation::face_transform(1);
        // column-major (-1,-1, 0,1) → [[-1, 0], [-1, 1]]
        assert_eq!(t1, [[-1.0, 0.0], [-1.0, 1.0]]);
        let t5 = NdDofTransformation::face_transform(5);
        // column-major (0,1, 1,0) → [[0, 1], [1, 0]]
        assert_eq!(t5, [[0.0, 1.0], [1.0, 0.0]]);
        let ti2 = NdDofTransformation::face_inverse_transform(2);
        // TInv(2) column-major (-1,-1, 1,0) → [[-1, 1], [-1, 0]]
        assert_eq!(ti2, [[-1.0, 1.0], [-1.0, 0.0]]);
    }

    /// Element sizes match the MFEM constructors (known ND dof counts).
    #[test]
    fn element_sizes_match_mfem() {
        for (p, tri, tet, wedge, pyr) in [
            (1usize, 3usize, 6usize, 9usize, 8usize),
            (2, 8, 20, 36, 32),
            (3, 15, 45, 90, 84),
        ] {
            assert_eq!(NdDofTransformation::new_tri(p).size(), tri, "tri p{p}");
            assert_eq!(NdDofTransformation::new_tet(p).size(), tet, "tet p{p}");
            assert_eq!(NdDofTransformation::new_wedge(p).size(), wedge, "wedge p{p}");
            assert_eq!(NdDofTransformation::new_pyramid(p).size(), pyr, "pyr p{p}");
        }
    }

    /// Primal/inv-primal round trip on a tet layout with all faces triangular.
    #[test]
    fn tet_primal_round_trip() {
        let p = 3usize;
        let t = NdDofTransformation::new_tet(p);
        assert!(!t.is_identity());
        let fo = [2i32, 0, 5, 1];
        let v0: Vec<f64> = (0..t.size()).map(|i| (i as f64 + 0.5).copysign(if i % 3 == 0 { 1.0 } else { -1.0 })).collect();
        let mut v = v0.clone();
        t.transform_primal(&fo, &mut v);
        assert_ne!(v, v0, "order-3 tet has tri-face pairs, transform must act");
        t.inv_transform_primal(&fo, &mut v);
        for (a, b) in v.iter().zip(v0.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    /// p=1 tet has no face pairs: transformation is the identity.
    #[test]
    fn tet_order1_is_identity() {
        let t = NdDofTransformation::new_tet(1);
        assert!(t.is_identity());
        let mut v = vec![1.0_f64; 6];
        let fo = [3i32; 4];
        t.transform_primal(&fo, &mut v);
        assert!(v.iter().all(|&x| x == 1.0));
    }

    /// Duality ⟨T⁻ᵀ f, w⟩ = ⟨f, T⁻¹ w⟩ on a wedge (mixed tri/quad faces).
    #[test]
    fn dual_duality_identity() {
        let t = NdDofTransformation::new_wedge(2);
        assert!(!t.is_identity());
        let fo = [1i32, 4, 0, 2, 3];
        let f: Vec<f64> = (0..t.size()).map(|i| 0.3 + i as f64 * 0.11).collect();
        let w: Vec<f64> = (0..t.size()).map(|i| -0.7 + i as f64 * 0.07).collect();
        let mut f_t = f.clone();
        t.transform_dual(&fo, &mut f_t);
        let mut w_it = w.clone();
        t.inv_transform_primal(&fo, &mut w_it);
        let lhs: f64 = f_t.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = f.iter().zip(w_it.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-10, "lhs {lhs} vs rhs {rhs}");
    }

    /// InvTransformDual ≙ transpose of TransformPrimal.
    #[test]
    fn inv_dual_is_primal_transpose() {
        let t = NdDofTransformation::new_pyramid(2);
        assert!(!t.is_identity());
        let fo = [0i32, 3, 1, 2, 4];
        let n = t.size();
        let f: Vec<f64> = (0..n).map(|i| 0.2 + i as f64 * 0.05).collect();
        let w: Vec<f64> = (0..n).map(|i| 1.1 - i as f64 * 0.03).collect();
        let mut f_t = f.clone();
        t.inv_transform_dual(&fo, &mut f_t);
        let mut w_p = w.clone();
        t.transform_primal(&fo, &mut w_p);
        let lhs: f64 = f_t.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = f.iter().zip(w_p.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-10);
    }

    /// The vdim/ordering wrapper: byVDIM interleaving round trip.
    #[test]
    fn wrapper_vdim_by_vdim_round_trip() {
        let nested = NdDofTransformation::new_tet(2);
        let mut dt = DofTransformation::with_transformation(&nested);
        dt.set_vdim(2, Ordering::ByVdim);
        dt.set_face_orientations(vec![5, 1, 3, 0]);
        let n = nested.size();
        let v0: Vec<f64> = (0..2 * n).map(|i| 0.5 + 0.01 * i as f64).collect();
        let mut v = v0.clone();
        dt.transform_primal(&mut v);
        dt.inv_transform_primal(&mut v);
        for (a, b) in v.iter().zip(v0.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    /// The wrapper's dual-matrix transform equals manual row/col application.
    #[test]
    fn wrapper_dual_matrix_matches_manual() {
        let nested = NdDofTransformation::new_tri(3);
        let mut dt2 = DofTransformation::with_transformation(&nested);
        let fo = [4i32];
        dt2.set_face_orientations(fo.to_vec());
        let n = nested.size();
        let mut a: Vec<f64> = (0..n * n).map(|i| ((i % 7) as f64) - 3.0).collect();
        let mut b = a.clone();
        dt2.transform_dual_matrix(&mut a, n, n);
        // manual: dual cols then dual rows
        let mut col = vec![0.0_f64; n];
        for c in 0..n {
            for (i, slot) in col.iter_mut().enumerate() {
                *slot = b[i * n + c];
            }
            nested.transform_dual(&fo, &mut col);
            for (i, &val) in col.iter().enumerate() {
                b[i * n + c] = val;
            }
        }
        for r in 0..n {
            let mut row = b[r * n..(r + 1) * n].to_vec();
            nested.transform_dual(&fo, &mut row);
            b[r * n..(r + 1) * n].copy_from_slice(&row);
        }
        assert_eq!(a, b);
    }

    /// Mixed primal/dual pair transform: `D_t = T_ran D T_dom⁻¹` acting on
    /// the identity block reproduces T_ran·T_dom⁻¹.
    #[test]
    fn mixed_pair_transform() {
        let ran_tr = NdDofTransformation::new_tet(2);
        let dom_tr = NdDofTransformation::new_tet(2);
        let mut ran = DofTransformation::with_transformation(&ran_tr);
        let mut dom = DofTransformation::with_transformation(&dom_tr);
        ran.set_face_orientations(vec![2, 2, 2, 2]);
        dom.set_face_orientations(vec![0, 0, 0, 0]);
        let n = ran_tr.size();
        let mut d = vec![0.0_f64; n * n];
        for i in 0..n {
            d[i * n + i] = 1.0;
        }
        transform_primal_pairs(&ran, &dom, &mut d, n, n);
        // dom = orientation 0 → T_dom = I, so D_t must equal T_ran applied to
        // columns of I = T_ran itself.
        let mut expected = vec![0.0_f64; n * n];
        for c in 0..n {
            let mut e = vec![0.0_f64; n];
            e[c] = 1.0;
            ran_tr.transform_primal(&[2, 2, 2, 2], &mut e);
            for (i, &val) in e.iter().enumerate() {
                expected[i * n + c] = val;
            }
        }
        for (a, b) in d.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    /// RT trace sign: canonical orientation (+, or any non-negative MFEM
    /// face-orientation code) → +1; reversed → −1.
    #[test]
    fn rt_trace_sign() {
        assert_eq!(rt_trace_face_sign(1), 1.0);
        assert_eq!(rt_trace_face_sign(0), 1.0);
        assert_eq!(rt_trace_face_sign(-1), -1.0);
    }
}
