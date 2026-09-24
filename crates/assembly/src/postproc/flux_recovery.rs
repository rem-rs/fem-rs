//! Trait for integrator-level flux recovery (MFEM `ComputeElementFlux` /
//! `ComputeFluxEnergy` equivalent).
//!
//! Enables coefficient-aware ZZ error estimation: the flux includes the
//! integrator's coefficient (e.g. κ in `-∇·(κ∇u)`), not just `∇u_h`.
//!
//! # Usage
//! ```rust,ignore
//! let integrator = DiffusionIntegrator { kappa: 1.0 };
//! let eta = zz_estimator_mfem(&gf, &integrator).eta;
//! ```

use fem_element::ReferenceElement;
use fem_mesh::{MeshTopology, element_type::ElementType};
use fem_space::fe_space::{FESpace, SpaceType};

use crate::postproc::error_estimate::ElementIndicators;
use crate::postproc::grid_function::GridFunction;

/// Trait for bilinear-form integrators that support ZZ-style flux recovery.
///
/// Mirrors MFEM's `BilinearFormIntegrator::ComputeElementFlux` and
/// `ComputeFluxEnergy` used by `ZienkiewiczZhuEstimator`.
pub trait FluxRecovery {
    /// Compute the raw (element-local) flux via L² projection onto the flux space.
    ///
    /// For `DiffusionIntegrator` with constant κ, this evaluates `κ·∇u_h` at
    /// quadrature points, forms the element mass matrix `M_{ij} = ∫ φ_i φ_j dΩ`
    /// and RHS `b_{i,d} = ∫ (κ·∇u_h)_d φ_i dΩ`, then solves `M·flux = b` for each
    /// dimension component. This matches MFEM's default `ComputeElementFlux` which
    /// L²-projects the flux rather than evaluating at DOF coordinates directly.
    ///
    /// Returns a flat array `[n_flux_dofs × dim]` where
    /// `flux[i * dim + d]` = d-th component at the i-th flux-space DOF.
    fn compute_element_flux<M: MeshTopology, S: FESpace<Mesh = M>>(
        &self,
        mesh: &M,
        space: &S,
        element: u32,
        solution_dofs: &[f64],
        flux_dof_coords: &[Vec<f64>],
    ) -> Vec<f64>;

    /// Compute the squared energy norm of a flux-difference vector on `element`.
    ///
    /// Returns `∫ (1/κ) |flux_diff|² dΩ` — the energy norm (not its square root),
    /// matching MFEM's `DiffusionIntegrator::ComputeFluxEnergy`.
    /// `flux_diff` has the same layout as `compute_element_flux` output.
    fn compute_flux_energy<M: MeshTopology>(
        &self,
        mesh: &M,
        element: u32,
        flux_diff: &[f64],
    ) -> f64;
}

// ─── Reference element helper ────────────────────────────────────────────────

/// D364: delegated to the single source of truth.  Arm-for-arm identical to
/// the historical local table: H¹-slot simplices (D185/D157/D202), the legacy
/// `[-1,1]²` `QuadQ1`/`QuadQ2` frames, the D353-sweep hex/prism arms where
/// basis and geometry share one frame (`HexQk`/`PrismPk`, `o.max(1)`), and —
/// D365 — the pyramid arm: the H¹ pyramid element of the default family
/// (Fuentes, entity slot order), the same slots `DofManager::build_pyramid_pk`
/// numbers the H¹ space's `element_dofs` in.  D614: the complete `Hex27` and
/// quadratic `Prism18` cell labels join their families' arms (the consumers'
/// families are label-independent — one CUBE / one wedge family each).
/// Same panic set.
fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    ref_elem_vol_with_pyramid_basis(
        elem_type,
        order,
        fem_element::lagrange::PyramidBasisType::default(),
    )
}

/// [`ref_elem_vol`] with an explicit pyramid H¹ basis family — the
/// flux-recovery analogue of the assembler's
/// `ref_elem_vol_h1_with_pyramid_basis` (MFEM `H1_FECollection`'s `pyr_type`
/// switch).  D462: callers that hold the *solution* space must thread
/// `space.pyramid_basis()` through here, so a Bergot pyramid space
/// (`pyr_type = 0`) is sampled in its own slot order; only pyramid cells at
/// order ≥ 2 depend on it, every other arm ignores `pyr`.
fn ref_elem_vol_with_pyramid_basis(
    elem_type: ElementType,
    order: u8,
    pyr: fem_element::lagrange::PyramidBasisType,
) -> Box<dyn ReferenceElement> {
    match elem_type {
        ElementType::Tri3 | ElementType::Tri6 | ElementType::Tet4 => {
            fem_space::ref_elem::h1_simplex_slots(elem_type, order)
        }
        ElementType::Quad4 => fem_space::ref_elem::fixed_order_tensor(elem_type, order),
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
            fem_space::ref_elem::gll_tensor(elem_type, order.max(1))
        }
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => {
            fem_space::ref_elem::equispaced_prism(order.max(1))
        }
        ElementType::Pyramid5 | ElementType::Pyramid13 => {
            fem_space::ref_elem::h1_pyramid_slots(order.max(1), pyr)
        }
        _ => panic!(
            "ref_elem_vol: unsupported (element_type={elem_type:?}, order={order})"
        ),
    }
}

fn is_simplex(elem_type: ElementType) -> bool {
    matches!(elem_type, ElementType::Tri3 | ElementType::Tri6 | ElementType::Tet4 | ElementType::Tet10)
}

/// Geometric-mapping Jacobian at reference point `xi` on `element`.
///
/// - **Simplex** (Tri3/Tri6/Tet4/Tet10): the P1 mapping, i.e. the constant
///   Jacobian built from `nodes[0..dim]`.
/// - **Quad4** (`dim == 2`): the analytic bilinear map on `[-1,1]²` — the frame
///   the `QuadQ1` basis in [`ref_elem_vol`] is evaluated in.
/// - **Hex8/Hex20**: the isoparametric `HexQk` geometry, i.e. the *same*
///   element family the solution basis uses, so the basis and the geometry
///   share one reference frame.
/// - **Prism6/Prism15**: the isoparametric `PrismPk` geometry (same family
///   argument as the hex).
/// - **Pyramid5/Pyramid13** (D365): delegated to
///   [`fem_mesh::transformation::element_jacobian_at`], the mesh crate's
///   single source of truth for geometry Jacobians — it applies the straight
///   pyramid's `PYR_P1_SLOT_VERTEX` slot permutation (D331: `PyramidPk(1)`'s
///   layer slots carry the shape functions of MFEM vertices `0,1,3,2,4`) and
///   evaluates a curved pyramid (`geom_order > 1`) with its own order-`g`
///   Fuentes element (D334), the same delegation
///   `postproc/grid_function.rs::element_jacobian` already makes.  The
///   straight-pyramid geometry and the Fuentes solution basis live on the same
///   unit-pyramid reference domain, so one `xi` means the same point for both.
/// - **Anything else**: the P1 corner-difference map.
///
/// D353 sweep.  The last arm used to be the only one for 3-D non-simplex
/// cells, and it is **singular on a hex**: `nodes[1]`, `nodes[2]`, `nodes[3]`
/// are the two base edges plus the base *diagonal*, so the three columns of
/// `J` are linearly dependent and `det J == 0`.  Both consumers then collapsed
/// to zero — `compute_element_flux`'s `jac.try_inverse().unwrap_or_default()`
/// produced an identically zero flux, and `compute_flux_energy` weighed every
/// quadrature point by `0`.  That is the same "silent zero" class as D353
/// itself (`grid_function::element_jacobian`); `ref_elem_vol` refused Hex8
/// outright, so no test had reached it before.
fn geom_jacobian<M: MeshTopology>(
    mesh: &M,
    element: u32,
    nodes: &[u32],
    xi: &[f64],
    dim: usize,
    elem_type: ElementType,
) -> (nalgebra::DMatrix<f64>, f64) {
    use nalgebra::DMatrix;
    if is_simplex(elem_type) {
        let x0 = mesh.node_coords(nodes[0]);
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for col in 0..dim {
            let xc = mesh.node_coords(nodes[col + 1]);
            for row in 0..dim { j[(row, col)] = xc[row] - x0[row]; }
        }
        (j.clone(), j.determinant())
    } else if dim == 2 && nodes.len() >= 4 {
        let (e, n) = (xi[0], xi[1]);
        let c = |i: usize| mesh.node_coords(nodes[i]);
        let j00 = 0.25 * (-(1.0 - n) * c(0)[0] + (1.0 - n) * c(1)[0] + (1.0 + n) * c(2)[0] - (1.0 + n) * c(3)[0]);
        let j01 = 0.25 * (-(1.0 - e) * c(0)[0] - (1.0 + e) * c(1)[0] + (1.0 + e) * c(2)[0] + (1.0 - e) * c(3)[0]);
        let j10 = 0.25 * (-(1.0 - n) * c(0)[1] + (1.0 - n) * c(1)[1] + (1.0 + n) * c(2)[1] - (1.0 + n) * c(3)[1]);
        let j11 = 0.25 * (-(1.0 - e) * c(0)[1] - (1.0 + e) * c(1)[1] + (1.0 + e) * c(2)[1] + (1.0 - e) * c(3)[1]);
        let det = j00 * j11 - j01 * j10;
        let jac = DMatrix::from_row_slice(2, 2, &[j00, j01, j10, j11]);
        (jac, det)
    } else if matches!(elem_type, ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27) {
        // A curved hex reads its own geometry table; a straight one the
        // vertex connectivity.  `HexQk` is the element `ref_elem_vol(Hex8, o)`
        // returns, so `xi` means the same thing for basis and geometry.
        // D614: the complete `Hex27` label shares the one CUBE family.
        let geo = fem_element::lagrange::HexQk::new(mesh.geom_order().max(1) as usize);
        let geo_nodes: &[u32] = if mesh.geom_order() > 1 {
            mesh.geometry_nodes(element)
        } else {
            nodes
        };
        let (j, det, _xp) =
            crate::vector_assembler::isoparametric_jacobian(mesh, geo_nodes, &geo, xi, 3);
        (j, det)
    } else if matches!(
        elem_type,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18
    ) {
        // Wedge: `PrismPk` is both the solution basis ([`ref_elem_vol`]) and
        // the geometry `geo_ref_elem_from_mesh` selects.  D614: `Prism18`
        // joins the one wedge family.
        let geo = fem_element::lagrange::PrismPk::new(mesh.geom_order().max(1) as usize);
        let geo_nodes: &[u32] = if mesh.geom_order() > 1 {
            mesh.geometry_nodes(element)
        } else {
            nodes
        };
        let (j, det, _xp) =
            crate::vector_assembler::isoparametric_jacobian(mesh, geo_nodes, &geo, xi, 3);
        (j, det)
    } else if matches!(elem_type, ElementType::Pyramid5 | ElementType::Pyramid13) {
        // D365: the pyramid geometry is *not* the solution basis family (the
        // straight-pyramid map is the `PyramidPk(1)` layer-slot frame, a
        // curved one its own order-`g` Fuentes element) — exactly the cases
        // the mesh crate's `element_jacobian_at` already encodes.  Reuse it
        // instead of a fourth hand-rolled pyramid Jacobian.
        let (j, _xp) = fem_mesh::transformation::element_jacobian_at(mesh, element, xi, dim);
        let det = j.determinant();
        (j, det)
    } else {
        let x0 = mesh.node_coords(nodes[0]);
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for col in 0..dim.min(nodes.len().saturating_sub(1)) {
            let xc = mesh.node_coords(nodes[col + 1]);
            for row in 0..dim { j[(row, col)] = xc[row] - x0[row]; }
        }
        (j.clone(), j.determinant())
    }
}

// ─── Implementation for DiffusionIntegrator ──────────────────────────────────

use crate::standard::DiffusionIntegrator;

/// The flux-vector length → FE order inference behind
/// [`FluxRecovery::compute_flux_energy`] (MFEM: `order =
/// 2*fluxelem.GetOrder(); IntRules.Get(geom, order)` — the flux space has the
/// same order as the solution space, so `n_flux_dofs = flux_diff.len()/dim`
/// pins it).
///
/// D366: the table used to stop at p = 3 and fall through to a **silent**
/// `_ => 1` — right by accident at p = 1 only, and for any higher-order flux
/// space it read an order-1 basis against an order-p flux vector (the D353
/// defect class).  The arms now cover every H¹ cell type through p = 5
/// (pyramids through p = 4, both families — see the pyramid arm), and
/// the fallback is no longer silent: a `debug_assert!` names the unmapped
/// `(element_type, n_flux_dofs)` pair in debug builds while release builds
/// keep the conservative order-1 quadrature.
fn infer_fe_order(elem_type: ElementType, n_flux_dofs: usize) -> u8 {
    match (elem_type, n_flux_dofs) {
        (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => 1,
        (ElementType::Tri3, 6) | (ElementType::Tri6, 6) => 2,
        (ElementType::Tri3, 10) | (ElementType::Tri6, 10) => 3,
        (ElementType::Tri3, 15) | (ElementType::Tri6, 15) => 4,
        (ElementType::Tri3, 21) | (ElementType::Tri6, 21) => 5,
        (ElementType::Quad4, 4) => 1,
        (ElementType::Quad4, 9) => 2,
        (ElementType::Quad4, 16) => 3,
        (ElementType::Quad4, 25) => 4,
        (ElementType::Quad4, 36) => 5,
        // D353 sweep: the `_ => 1` fallback happened to be right for the
        // hex only at p = 1 (HexQk(1) has 8 DOFs); at p >= 2 the estimator
        // would read an 8-DOF basis against a 27- (or 64-) DOF flux vector.
        // D614: the complete `Hex27` label shares the hex counts.
        (ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27, 8) => 1,
        (ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27, 27) => 2,
        (ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27, 64) => 3,
        (ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27, 125) => 4,
        (ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27, 216) => 5,
        // D614: `Prism18` shares the wedge counts ((p+1)²(p+2)/2) and the
        // p = 4/5 arms close the table.
        (ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18, 6) => 1,
        (ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18, 18) => 2,
        (ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18, 40) => 3,
        (ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18, 75) => 4,
        (ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18, 126) => 5,
        (ElementType::Tet4, 4) => 1,
        (ElementType::Tet4, 10) => 2,
        (ElementType::Tet4, 20) => 3,
        (ElementType::Tet4, 35) => 4,
        (ElementType::Tet4, 56) => 5,
        // D365/D366: the pyramid counts.  Two MFEM families live on pyramid
        // cells (`H1_FECollection`'s `pyr_type`), and at p ≥ 2 their counts
        // differ, so the flux-vector length pins the *order* either way:
        //   Fuentes (default, `p(p²+3)+1`): 5 / 15 / 37 / 77 at p = 1..4
        //     (`fem_element::lagrange::pyramid`'s `pyramid_basis_type_family_map`);
        //   Bergot (`(p+1)(p+2)(2p+3)/6`, the `pyr_type = 0` opt-out,
        //     D454): 5 / 14 / 30 / 55 at p = 1..4 — counts measured from
        //     `h1_pyramid_slots(p, PyramidBasisType::Bergot).n_dofs()` and
        //     pinned by `fem_element`'s `h1_pyramid_pk_counts`.
        // The 5 at p = 1 is shared and maps to the same order in both
        // families; 14/30/55 collide with no other arm of this table.
        (ElementType::Pyramid5 | ElementType::Pyramid13, 5) => 1,
        (ElementType::Pyramid5 | ElementType::Pyramid13, 15) => 2,
        (ElementType::Pyramid5 | ElementType::Pyramid13, 37) => 3,
        (ElementType::Pyramid5 | ElementType::Pyramid13, 77) => 4,
        (ElementType::Pyramid5 | ElementType::Pyramid13, 14) => 2,
        (ElementType::Pyramid5 | ElementType::Pyramid13, 30) => 3,
        (ElementType::Pyramid5 | ElementType::Pyramid13, 55) => 4,
        other => {
            debug_assert!(
                false,
                "infer_fe_order: unmapped (element_type, n_flux_dofs) = \
                 ({:?}, {}) — matches no H¹ count of any supported family \
                 (Fuentes *and* Bergot pyramids included since D454); add an \
                 arm; conservatively using quadrature order 2 (= 2·1)",
                other.0, other.1
            );
            1
        }
    }
}

/// D454: pick the pyramid flux basis **family** from the flux-vector length.
///
/// [`infer_fe_order`] resolves the length to the order for both pyramid
/// families, but [`ref_elem_vol`] only builds the Fuentes default — a Bergot
/// flux space (MFEM `H1_FECollection(.., pyr_type = 0)`) would then be read
/// against a 15/37/77-DOF Fuentes basis (the D353 defect class).  The caller
/// chain (`compute_flux_energy`) holds no space object, only the count — but
/// the count distinguishes the families at every p ≥ 2, so reconstruct the
/// element from it: Fuentes if the count matches, Bergot otherwise.
fn pyramid_flux_element(order: u8, n_flux_dofs: usize) -> Box<dyn ReferenceElement> {
    let fuentes = fem_space::ref_elem::h1_pyramid_slots(
        order,
        fem_element::lagrange::PyramidBasisType::Fuentes,
    );
    if fuentes.n_dofs() == n_flux_dofs {
        return fuentes;
    }
    let bergot = fem_space::ref_elem::h1_pyramid_slots(
        order,
        fem_element::lagrange::PyramidBasisType::Bergot,
    );
    debug_assert_eq!(
        bergot.n_dofs(),
        n_flux_dofs,
        "pyramid_flux_element: n_flux_dofs={n_flux_dofs} matches neither family \
         at order {order} (Fuentes {}, Bergot {})",
        fuentes.n_dofs(),
        bergot.n_dofs()
    );
    bergot
}

/// D637: the flux recovery implements MFEM's `ZZErrorEstimator` with an H¹
/// flux space (the solution's own `FECollection`): every element's flux
/// vector has exactly one slot per **that element's** H¹ flux DOF, and
/// `H1_FECollection` makes the dof transformations no-ops (D455), so the
/// recovery samples basis values directly against the first `n_ldofs` slots
/// of `element_dofs`.  On the D613 wildcard-numbered quadratic-connectivity
/// labels (Hex20/Hex27/…) the H¹ table legitimately carries *extra*
/// connectivity dofs beyond the flux family (20 ≥ 8) — the leading slots are
/// the vertex slots the family reads (pinned by
/// `d614_flux_recovery_affine_exact_on_high_order_cells`).
///
/// Everything else decouples the flux rows from the dof table and was
/// silently mis-indexed (or silently produced zero η): RT/HDiv (RT0 hex
/// carries 6 face dofs against the 8-slot H¹ hex family — and hex RT p ≥ 1
/// additionally needs the shared-quad-face canonical rotation
/// (`element_face_blocks`), which is unwired; d448 covered 2-D quad faces
/// only), HCurl/ND, L², vector spaces, variable-order tables shorter than
/// the family.  Those are refused loudly here; the hex-RT canonical
/// rotation itself remains future work.
fn check_h1_flux_layout<S: FESpace>(
    space: &S,
    element: u32,
    elem_type: ElementType,
    n_ldofs: usize,
) {
    let n_table = space.element_dofs(element).len();
    assert!(
        space.space_type() == SpaceType::H1 && n_table >= n_ldofs,
        "D637 flux recovery: element {element} ({elem_type:?}) has {n_table} \
         flux-space dofs but the H¹ flux family at order {} has {n_ldofs} \
         slots — this recovery only implements the H¹ flux space (MFEM \
         ZZErrorEstimator with the solution's H1_FECollection).  A non-H¹ \
         ({:?}) table here was silently mis-indexed; for hex RT p ≥ 1 the \
         shared-quad-face canonical rotation table is not wired yet (d448: \
         2-D only).",
        space.order(),
        space.space_type()
    );
    assert!(
        space.element_face_blocks(element).is_empty(),
        "D637 flux recovery: element {element} reports canonical face-block \
         rotations (element_face_blocks non-empty) that this recovery does \
         not apply — averaging would silently drop them"
    );
}

impl FluxRecovery for DiffusionIntegrator<f64> {
    fn compute_element_flux<M: MeshTopology, S: FESpace<Mesh = M>>(
        &self,
        mesh: &M,
        space: &S,
        element: u32,
        solution_dofs: &[f64],
        flux_dof_coords: &[Vec<f64>],
    ) -> Vec<f64> {
        let dim = mesh.dim() as usize;
        let n_flux_dofs = flux_dof_coords.len();
        let elem_type = mesh.element_type(element);
        let order = space.order();
        // D462: the sampler is the *solution* space's own H¹ basis — a Bergot
        // pyramid space (`pyr_type = 0`) numbers its `element_dofs` in Bergot
        // slot order, so reading the Fuentes default here desynchronises the
        // basis from the dof table.
        let ref_elem = ref_elem_vol_with_pyramid_basis(
            elem_type,
            order,
            space.pyramid_basis(),
        );
        let n_ldofs = ref_elem.n_dofs();
        let nodes = mesh.element_nodes(element);
        let elem_dofs = space.element_dofs(element);
        check_h1_flux_layout(space, element, elem_type, n_ldofs);

        // MFEM DiffusionIntegrator::ComputeElementFlux evaluates κ·∇u_h at the
        // flux-space DOF nodes (fluxelem.GetNodes()) directly — no L²
        // projection, no quadrature.  Per node:
        //   vec_j     = Σ_i u_i · ∂φ_i/∂ξ_j          (reference gradient,
        //                                             dshape.MultTranspose(u))
        //   vecdxt_i  = Σ_j (J^{-1})_{j,i} · vec_j   (invdfdx.MultTranspose)
        //   flux(i,d) = κ · vecdxt_d
        // Order of operations (combine first, then transform) matches MFEM
        // bit-for-bit; an L² projection (even though mathematically equal for
        // P1 gradients ⊂ P2) differs in the last ulps and flips elements near
        // the AMR error threshold.
        let mut grad_ref = vec![0.0; n_ldofs * dim];
        let mut vec = vec![0.0; dim];
        let mut flux = vec![0.0; n_flux_dofs * dim];
        for (i, xi) in flux_dof_coords.iter().enumerate() {
            ref_elem.eval_grad_basis(xi, &mut grad_ref);
            for j in 0..dim {
                let mut s = 0.0;
                for k in 0..n_ldofs {
                    s += solution_dofs[elem_dofs[k] as usize] * grad_ref[k * dim + j];
                }
                vec[j] = s;
            }
            let (jac, _) = geom_jacobian(mesh, element, nodes, xi, dim, elem_type);
            // D365: a pyramid's reference map collapses at its apex — `det J`
            // vanishes there (the base plane maps to a single point) — and the
            // Fuentes H¹ flux DOF sits exactly on that point.  MFEM's own
            // pyramid transformation has the same singularity; the physical
            // gradient, however, has a finite limit at the apex, so resample
            // once at a point pulled 10% toward the reference-domain centroid
            // (an interior point for every supported frame) and invert that —
            // exact for affine fields, first-order for the estimator.  The
            // simplex/tensor frames are regular everywhere, so this arm is
            // never reached for them.
            let j_inv = match jac.try_inverse() {
                Some(inv) => inv,
                None => {
                    let pull = 0.1;
                    let centroid = 1.0 / (dim as f64 + 1.0);
                    let xi_in: Vec<f64> =
                        xi.iter().map(|&c| (1.0 - pull) * c + pull * centroid).collect();
                    let (jac_in, _) = geom_jacobian(mesh, element, nodes, &xi_in, dim, elem_type);
                    let j_inv = jac_in.try_inverse().unwrap_or_else(|| {
                        panic!(
                            "compute_element_flux: singular geometry Jacobian at \
                             a flux dof and its pulled interior resample \
                             (element_type={elem_type:?}, xi={xi:?})"
                        )
                    });
                    // D462: the reference gradient must be sampled at the
                    // *same* point as the geometry Jacobian.  For an
                    // exactly-represented field ∇ξu_h(ξ) = Jᵀ(ξ)·∇xu, so the
                    // pulled-point Jacobian pairs with the pulled-point
                    // gradient — mixing the apex gradient with the pulled
                    // Jacobian broke affine exactness for the Bergot pyramid
                    // spaces whose flux dof sits on the apex.
                    ref_elem.eval_grad_basis(&xi_in, &mut grad_ref);
                    for j in 0..dim {
                        let mut s = 0.0;
                        for k in 0..n_ldofs {
                            s += solution_dofs[elem_dofs[k] as usize] * grad_ref[k * dim + j];
                        }
                        vec[j] = s;
                    }
                    j_inv
                }
            };
            for d in 0..dim {
                let mut s = 0.0;
                for j in 0..dim {
                    s += j_inv[(j, d)] * vec[j];
                }
                // MFEM's ZienkiewiczZhuEstimator defaults to with_coeff=false,
                // so ComputeElementFlux returns the raw gradient (no κ factor);
                // ComputeFluxEnergy applies the coefficient (Q->Eval) instead.
                flux[i * dim + d] = s;
            }
        }
        flux
    }

    fn compute_flux_energy<M: MeshTopology>(
        &self,
        mesh: &M,
        element: u32,
        flux_diff: &[f64],
    ) -> f64 {
        let dim = mesh.dim() as usize;
        let elem_type = mesh.element_type(element);
        // The flux space has the same order as the solution space (ex15: the
        // estimator's flux FES is built from the same H1_FECollection).  The
        // FE order is inferred from the flux_diff layout (n_dofs per component).
        let n_flux_dofs = if dim > 0 { flux_diff.len() / dim } else { 0 };
        let fe_order = infer_fe_order(elem_type, n_flux_dofs);
        // MFEM: order = 2 * fluxelem.GetOrder(); IntRules.Get(geom, order).
        let quad_order = (fe_order as u8) * 2;
        // D454: on pyramid cells the flux basis family is reconstructed from
        // the flux-vector length (the Fuentes default that `ref_elem_vol`
        // builds would misread a Bergot flux space).
        let ref_elem = if matches!(elem_type, ElementType::Pyramid5 | ElementType::Pyramid13) {
            pyramid_flux_element(fe_order as u8, n_flux_dofs)
        } else {
            ref_elem_vol(elem_type, fe_order as u8)
        };
        let n_ldofs = ref_elem.n_dofs();
        let nodes = mesh.element_nodes(element);
        let quad = ref_elem.quadrature(quad_order);

        // MFEM ComputeFluxEnergy: for each quadrature point
        //   pointflux_d = Σ_j flux_diff(j,d) · φ_j(ip)      (CalcPhysShape:
        //                reference basis evaluated at the physical point,
        //                i.e. the reference basis for VALUE elements)
        //   energy += Trans.Weight()·ip.weight · (pointflux·pointflux)
        //   (with coeff Q: ·Q->Eval; ex15 uses ConstantCoefficient(1.0))
        let mut phi = vec![0.0; n_ldofs];
        let mut pointflux = vec![0.0; dim];
        let mut energy = 0.0;
        for (q, xi) in quad.points.iter().enumerate() {
            let (_, det_j) = geom_jacobian(mesh, element, nodes, xi, dim, elem_type);
            // D696 batch 3 adjudication (revised on the d365 red light):
            // MFEM ComputeFluxEnergy weights with `Trans.Weight()·ip.weight`
            // and the MFEM probe `tmp/d696b/bipyramid_probe.cpp` proves
            // Weight() is +1 at every point of a valid apex-DOWN pyramid
            // (GetElementVolume = +1/3), while the fem-rs straight-pyramid
            // frame (D235/D339/D340 axes-permuted collapsed map) carries
            // det < 0 on such cells.  The measure therefore keeps |det| on
            // the PYRAMID frames (a frame-normalization that RESTORES MFEM
            // parity, not a semantic deviation); simplex/quad/hex frames
            // share MFEM's orientation, where the signed det reproduces
            // MFEM's Weight bitwise — including the negative signature of
            // inverted cells (d696_signed_det_pins_batch3).
            let dj = if matches!(elem_type, ElementType::Pyramid5 | ElementType::Pyramid13) {
                det_j.abs()
            } else {
                det_j
            };
            let w = quad.weights[q] * dj;
            ref_elem.eval_basis(xi, &mut phi);
            for d in 0..dim {
                let mut s = 0.0;
                for j in 0..n_ldofs {
                    s += flux_diff[j * dim + d] * phi[j];
                }
                pointflux[d] = s;
            }
            let mut e = 0.0;
            for d in 0..dim {
                e += pointflux[d] * pointflux[d];
            }
            // Q->Eval == kappa for DiffusionIntegrator (ConstantCoefficient
            // one(1.0) in ex15) — multiply by kappa like MFEM's `e *= Q`.
            energy += w * self.kappa * e;
        }
        energy
    }
}

// ─── MFEM-style ZZ estimator ─────────────────────────────────────────────────

/// ZZ error estimator using MFEM-style `FluxRecovery` trait.
///
/// Algorithm (matches MFEM `SumFluxAndCount` + `ComputeFluxEnergy`):
/// 1. For each element, compute raw flux at all solution-space DOF coordinates
///    via `integrator.compute_element_flux`.
/// 2. Average fluxes at shared global DOFs (sum / count).
/// 3. For each element, compute `raw - averaged`, then
///    `η_K = √(integrator.compute_flux_energy(diff))`.
pub fn zz_estimator_mfem<'a, M, S, F>(
    gf: &GridFunction<'a, S>,
    integrator: &F,
) -> ElementIndicators
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
    F: FluxRecovery,
{
    zz_estimator_mfem_impl(gf, integrator, "ZZ(MFEM)")
}

/// MFEM-style ZZ estimator with hanging-node constraint support.
///
/// For constrained DOFs, the averaged flux is recovered from parent DOFs via
/// the constraint relationship, matching MFEM's flux-space handling.
///
/// D455: `constraints` is currently **unused** — see the NOTE on the averaging
/// step below (MFEM's `H1_FECollection` flux spaces make the primal
/// `TransformPrimal`/`InvTransformPrimal` calls no-ops, so applying hanging-node
/// recovery here biased the estimator).  The parameter is kept because the
/// caller (`postproc::amr_refiner`) threads its AMR constraint table through
/// this API surface, and an MFEM-parity recovery (a flux space whose collection
/// *does* transform slaves) would need exactly this input.
pub fn zz_estimator_mfem_nc<'a, M, S, F>(
    gf: &GridFunction<'a, S>,
    integrator: &F,
    _constraints: &[fem_mesh::amr::HangingNodeConstraint],
) -> ElementIndicators
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
    F: FluxRecovery,
{
    zz_estimator_mfem_impl(gf, integrator, "ZZ(MFEM-NC)")
}

/// Shared body of [`zz_estimator_mfem`] / [`zz_estimator_mfem_nc`] — D499: the
/// two entry points were literal copies (round-53 ③路钉位恒等) and are now
/// thin delegates that differ only in the indicator label and the D455
/// `constraints` parameter.  The single body keeps their outputs identical:
/// averaging over all DOFs with **no** hanging-node recovery (MFEM
/// `H1_FECollection` flux spaces make `TransformPrimal`/`InvTransformPrimal`
/// no-ops), exactly the NOTE semantics below.
fn zz_estimator_mfem_impl<'a, M, S, F>(
    gf: &GridFunction<'a, S>,
    integrator: &F,
    label: &'static str,
) -> ElementIndicators
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
    F: FluxRecovery,
{
    let mesh: &M = gf.space().mesh();
    let ne = mesh.n_elements();
    let nd = gf.space().n_dofs();
    let dim = mesh.dim() as usize;
    let order = gf.space().order();
    let dofs_vec = gf.dofs();

    // ── Step 1-2: SumFluxAndCount ────────────────────────────────────────────
    let mut flux_sum = vec![vec![0.0; dim]; nd];
    let mut flux_count = vec![0usize; nd];

    // D637: each element is sampled in its OWN family, at its OWN H¹ dof
    // coordinates — element 0's slot count used to size every element's flux
    // vector, which silently mis-sampled (or indexed out of bounds) every
    // cell of a different family on a mixed mesh.
    for e in 0..ne as u32 {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_vol_with_pyramid_basis(
            elem_type,
            order,
            gf.space().pyramid_basis(),
        );
        let n_ldofs = ref_elem.n_dofs();
        check_h1_flux_layout(gf.space(), e, elem_type, n_ldofs);
        let dof_coords = ref_elem.dof_coords();
        let raw = integrator.compute_element_flux(mesh, gf.space(), e, &dofs_vec, &dof_coords);
        let elem_dofs = gf.space().element_dofs(e);
        for (i, &gdof) in elem_dofs.iter().enumerate() {
            let idx = gdof as usize;
            for d in 0..dim {
                flux_sum[idx][d] += raw[i * dim + d];
            }
            flux_count[idx] += 1;
        }
    }

    // Average (over unconstrained DOFs the average is the plain sum/count; on
    // constrained DOFs see the NOTE below — no hanging-node recovery).
    let mut flux_avg = vec![vec![0.0; dim]; nd];
    for i in 0..nd {
        let c = flux_count[i] as f64;
        if c > 0.0 {
            for d in 0..dim {
                flux_avg[i][d] = flux_sum[i][d] / c;
            }
        }
    }
    // NOTE: no hanging-node recovery on the averaged flux.  MFEM's flux space
    // here is the same H1 order-2 space as the solution (ex15.cpp uses `fec`);
    // H1_FECollection does NOT override DofTransformationForGeometry, so
    // fdoftrans.TransformPrimal / InvTransformPrimal in ZZErrorEstimator /
    // SumFluxAndCount are no-ops — the averaged flux keeps its directly
    // averaged slave-DOF values (a slave DOF shared by one element simply
    // keeps that element's raw flux).  Applying recover_hanging_values here
    // overwrote slaves with the master interpolation and systematically
    // biased the estimator (ex15: ~3.7x err on hanging clusters).

    // ── Step 3: per-element error ────────────────────────────────────────────
    let mut eta = vec![0.0; ne];
    for e in 0..ne as u32 {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_vol_with_pyramid_basis(
            elem_type,
            order,
            gf.space().pyramid_basis(),
        );
        let n_ldofs = ref_elem.n_dofs();
        check_h1_flux_layout(gf.space(), e, elem_type, n_ldofs);
        let dof_coords = ref_elem.dof_coords();
        let raw = integrator.compute_element_flux(mesh, gf.space(), e, &dofs_vec, &dof_coords);
        let elem_dofs = gf.space().element_dofs(e);

        let mut diff = vec![0.0; n_ldofs * dim];
        for (i, &gdof) in elem_dofs.iter().enumerate() {
            let idx = gdof as usize;
            for d in 0..dim {
                diff[i * dim + d] = raw[i * dim + d] - flux_avg[idx][d];
            }
        }

        let eng = integrator.compute_flux_energy(mesh, e, &diff);
        eta[e as usize] = eng.sqrt();
    }

    ElementIndicators::new(eta, label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;
    use crate::postproc::grid_function::GridFunction;
    use crate::standard::DiffusionIntegrator;

    #[test]
    fn mfem_zz_linear_exact() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0] + x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let int = DiffusionIntegrator { kappa: 1.0 };
        for &e in &zz_estimator_mfem(&gf, &int).eta {
            assert!(e < 1e-12, "MFEM ZZ should be exact for linear fns");
        }
    }

    #[test]
    fn mfem_zz_quadratic_nonzero() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]*x[0] + x[1]*x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let int = DiffusionIntegrator { kappa: 1.0 };
        let eta = zz_estimator_mfem(&gf, &int).eta;
        assert!(eta.iter().sum::<f64>() > 0.0, "should be > 0 for quadratic");
    }

    #[test]
    fn mfem_zz_kappa_scales() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]*x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let int1 = DiffusionIntegrator { kappa: 1.0 };
        let int2 = DiffusionIntegrator { kappa: 4.0 };
        let eta1 = zz_estimator_mfem(&gf, &int1).eta;
        let eta2 = zz_estimator_mfem(&gf, &int2).eta;
        // Flux diff = κ·(∇u* - ∇u_h), energy = ∫ (1/κ)·|flux_diff|² → η ∝ √κ
        // For κ=4 vs κ=1: η_ratio = √4 = 2
        for (e1, e2) in eta1.iter().zip(eta2.iter()) {
            assert!((e2 / e1 - 2.0).abs() < 1e-10, "eta should scale as √κ, got ratio {:.6}", e2 / e1);
        }
    }
}

// ─── D202: high-order tri/tet table arms ────────────────────────────────────
#[cfg(test)]
mod d202_high_order_tables {
    //! The local table must hand back, at every order, the element whose slots
    //! pair with the H¹ space's DOF numbering — i.e. the assembler's space
    //! dispatch choice (`ref_elem_vol_for_space`'s H1 branch,
    //! `assembler.rs::ref_elem_vol_h1`).  Before D202, orders >= 4 fell into
    //! the fail-fast panic arm.

    use super::ref_elem_vol;
    use crate::assembler::ref_elem_vol_h1;
    use fem_mesh::element_type::ElementType;

    #[test]
    fn tri_tet_o4_to_o6_match_space_slots() {
        for o in 4..=6u8 {
            let local = ref_elem_vol(ElementType::Tri3, o);
            let space = ref_elem_vol_h1(ElementType::Tri3, o);
            assert_eq!(local.n_dofs(), space.n_dofs(), "tri n_dofs at order {o}");
            assert_eq!(
                local.dof_coords(),
                space.dof_coords(),
                "tri dof_coords at order {o}"
            );
            let mut phi = vec![0.0; local.n_dofs()];
            local.eval_basis(&[0.3, 0.2], &mut phi);
            let sum: f64 = phi.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "tri partition of unity at order {o}: {sum}");

            let local = ref_elem_vol(ElementType::Tet4, o);
            let space = ref_elem_vol_h1(ElementType::Tet4, o);
            assert_eq!(local.n_dofs(), space.n_dofs(), "tet n_dofs at order {o}");
            assert_eq!(
                local.dof_coords(),
                space.dof_coords(),
                "tet dof_coords at order {o}"
            );
            let mut phi = vec![0.0; local.n_dofs()];
            local.eval_basis(&[0.25, 0.3, 0.2], &mut phi);
            let sum: f64 = phi.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "tet partition of unity at order {o}: {sum}");
        }
    }
}

// ─── D366: fe_order inference covers every cell type × order ────────────────
#[cfg(test)]
mod d366_fe_order_table {
    //! `compute_flux_energy` infers the FE order from the flux-vector length.
    //! For every H¹ cell type and every order the flux space can hand over
    //! (p ≤ 5; pyramids p ≤ 3, the Fuentes counts measured through
    //! `h1_pyramid_element`), the inference must resolve that length back to
    //! the same order — otherwise the energy integral reads an order-1 basis
    //! against an order-p flux vector (the D353 defect class).

    use super::{infer_fe_order, pyramid_flux_element, ref_elem_vol, ref_elem_vol_with_pyramid_basis};
    use fem_element::lagrange::PyramidBasisType;
    use fem_mesh::element_type::ElementType;
    use fem_space::ref_elem::{h1_field_element, h1_pyramid_slots};

    #[test]
    fn every_cell_type_order_resolves_back_from_its_n_dofs() {
        let pyr = PyramidBasisType::default();
        // (element type, orders mapped by the table)
        let cases: &[(ElementType, std::ops::RangeInclusive<u8>)] = &[
            (ElementType::Tri3, 1..=5),
            (ElementType::Tri6, 1..=5),
            (ElementType::Quad4, 1..=5),
            (ElementType::Hex8, 1..=5),
            (ElementType::Prism6, 1..=3),
            (ElementType::Prism15, 1..=3),
            (ElementType::Tet4, 1..=5),
            (ElementType::Pyramid5, 1..=4),
            (ElementType::Pyramid13, 1..=4),
        ];
        for (et, orders) in cases {
            for p in orders.clone() {
                let n = h1_field_element(*et, p, pyr).n_dofs();
                assert_eq!(
                    infer_fe_order(*et, n),
                    p,
                    "{et:?}: n_dofs({p}) = {n} did not resolve back to order {p}"
                );
            }
        }
    }

    /// D454: the Bergot opt-out family (`pyr_type = 0`,
    /// `(p+1)(p+2)(2p+3)/6` DOFs — 5/14/30/55 at p = 1..4) resolves back to
    /// the same order, and `pyramid_flux_element` reconstructs a Bergot basis
    /// whose DOF count matches the flux vector exactly.
    #[test]
    fn bergot_pyramid_counts_resolve_back_and_pick_the_bergot_family() {
        for p in 1..=4u8 {
            let n = h1_pyramid_slots(p, PyramidBasisType::Bergot).n_dofs();
            assert_eq!(
                infer_fe_order(ElementType::Pyramid5, n),
                p,
                "Bergot p={p}: n_dofs={n} did not resolve back to order {p}"
            );
            let e = pyramid_flux_element(p, n);
            assert_eq!(
                e.n_dofs(),
                n,
                "Bergot p={p}: reconstructed flux basis must hold exactly {n} dofs"
            );
            if p >= 2 {
                // The families are distinguishable from p = 2 on: the
                // reconstructed element must be the Bergot one, not the
                // Fuentes default.
                let fuentes =
                    h1_pyramid_slots(p, PyramidBasisType::Fuentes);
                assert_ne!(
                    e.n_dofs(),
                    fuentes.n_dofs(),
                    "Bergot p={p}: family reconstruction collapsed onto Fuentes"
                );
            }
        }
    }

    // ── D614: the high-order-cell labels in the flux table ──────────────────

    /// D614: `Hex27` and `Prism18` flow through the flux-recovery table on
    /// the same families as their sibling labels (one CUBE / one wedge
    /// family), the flux-order inference covers them at every tabulated p,
    /// and the pyramid threading still honours the explicit family.
    #[test]
    fn d614_flux_ref_elem_vol_high_order_labels() {
        for et in [ElementType::Hex8, ElementType::Hex20, ElementType::Hex27] {
            assert_eq!(ref_elem_vol(et, 2).n_dofs(), 27, "{et:?}");
            assert_eq!(ref_elem_vol(et, 1).n_dofs(), 8, "{et:?}");
        }
        for et in [ElementType::Prism6, ElementType::Prism15, ElementType::Prism18] {
            assert_eq!(ref_elem_vol(et, 2).n_dofs(), 18, "{et:?}");
            assert_eq!(ref_elem_vol(et, 1).n_dofs(), 6, "{et:?}");
        }
        for et in [ElementType::Pyramid5, ElementType::Pyramid13] {
            assert_eq!(ref_elem_vol(et, 2).n_dofs(), 15, "{et:?}");
            assert_eq!(
                ref_elem_vol_with_pyramid_basis(et, 2, fem_element::lagrange::PyramidBasisType::Bergot)
                    .n_dofs(),
                14,
                "{et:?} Bergot opt-out"
            );
        }
        for (n, p) in [(8usize, 1u8), (27, 2), (64, 3), (125, 4), (216, 5)] {
            assert_eq!(infer_fe_order(ElementType::Hex27, n), p, "Hex27 n={n}");
        }
        for (n, p) in [(6usize, 1u8), (18, 2), (40, 3), (75, 4), (126, 5)] {
            assert_eq!(infer_fe_order(ElementType::Prism18, n), p, "Prism18 n={n}");
        }
    }
}
