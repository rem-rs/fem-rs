use fem_element::{iga::{BsplineBasis, KnotVector, NurbsBasis}, quadrature::gauss_legendre_01};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_space::iga::{IgaSpace1D, IgaSpace2D};

fn quadrature_points_01(order: u8) -> Result<(Vec<f64>, Vec<f64>), String> {
    // n-point Gauss-Legendre integrates degree (2n-1) exactly; choose minimal n.
    let n = ((order as usize + 2) / 2).max(1);
    if n > 4 {
        return Err(format!(
            "quad_order={order} requires n={n} Gauss points, but gauss_legendre_01 currently supports n<=4 (exactness up to order 7)"
        ));
    }
    Ok(gauss_legendre_01(n))
}

fn build_basis_from_space(space: &IgaSpace1D) -> Result<BsplineBasis, String> {
    let kv = KnotVector::new_clamped(space.knot_slice().to_vec())?;
    BsplineBasis::new(space.degree(), kv)
}

/// `dx/du` for the 1D isogeometric map `x = Σ c_i R_i` at a quadrature point, using local
/// derivatives from the same B-spline / NURBS basis.
fn dx_du_iga_1d(
    active: &[usize],
    ders_local: &[f64],
    ctrl_x: &[f64],
) -> Result<f64, String> {
    if active.len() != ders_local.len() {
        return Err("dx_du_iga_1d: active/ders len mismatch".to_string());
    }
    let mut s = 0.0_f64;
    for (k, &dof) in active.iter().enumerate() {
        let c = *ctrl_x
            .get(dof)
            .ok_or_else(|| format!("ctrl_x index {dof} out of range"))?;
        s += c * ders_local[k];
    }
    Ok(s)
}

fn local_values_for_active(
    active: &[usize],
    values: &[(usize, f64)],
    derivs: &[(usize, f64)],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut vals = Vec::with_capacity(active.len());
    let mut ders = Vec::with_capacity(active.len());
    for &dof in active {
        let n = values
            .iter()
            .find_map(|(i, v)| (*i == dof).then_some(*v))
            .ok_or_else(|| format!("basis values missing active dof {dof}"))?;
        let dn = derivs
            .iter()
            .find_map(|(i, v)| (*i == dof).then_some(*v))
            .ok_or_else(|| format!("basis derivatives missing active dof {dof}"))?;
        vals.push(n);
        ders.push(dn);
    }
    Ok((vals, ders))
}

fn assemble_bilinear_iga_1d_impl(
    space: &IgaSpace1D,
    quad_order: u8,
    integrand: impl Fn(f64, f64) -> f64,
) -> Result<CsrMatrix<f64>, String> {
    let basis = build_basis_from_space(space)?;
    let nurbs = space
        .weights()
        .map(|w| NurbsBasis::new(basis.clone(), w.to_vec()))
        .transpose()?;
    let spans = space.non_empty_spans();
    if spans.is_empty() {
        return Err("IgaSpace1D has no non-empty spans".to_string());
    }
    let knots = space.knot_slice();
    let (qpts, qwts) = quadrature_points_01(quad_order)?;
    let n_dofs = space.n_dofs();

    let mut coo = CooMatrix::new(n_dofs, n_dofs);
    for span in spans {
        let active = space.active_dofs_for_span(span)?;
        let ua = knots[span];
        let ub = knots[span + 1];
        let jac = ub - ua;

        for (&q, &w) in qpts.iter().zip(&qwts) {
            let u = ua + jac * q;
            let values = if let Some(nurbs_basis) = &nurbs {
                nurbs_basis.nonzero_values(u)?
            } else {
                basis.nonzero_values(u)?
            };
            let derivs = if let Some(nurbs_basis) = &nurbs {
                nurbs_basis.nonzero_derivatives(u)?
            } else {
                basis.nonzero_derivatives(u)?
            };
            let (vals_local, ders_local) = local_values_for_active(&active, &values, &derivs)?;

            for (a_loc, &i) in active.iter().enumerate() {
                let ni = vals_local[a_loc];
                let dni = ders_local[a_loc];
                for (b_loc, &j) in active.iter().enumerate() {
                    let nj = vals_local[b_loc];
                    let dnj = ders_local[b_loc];
                    coo.add(i, j, w * jac * integrand(ni * nj, dni * dnj));
                }
            }
        }
    }
    Ok(coo.into_csr())
}

fn build_bases_from_space_2d(space: &IgaSpace2D) -> Result<(BsplineBasis, BsplineBasis), String> {
    let ku = KnotVector::new_clamped(space.knot_slice_u().to_vec())?;
    let kv = KnotVector::new_clamped(space.knot_slice_v().to_vec())?;
    let bu = BsplineBasis::new(space.degree_u(), ku)?;
    let bv = BsplineBasis::new(space.degree_v(), kv)?;
    Ok((bu, bv))
}

fn local_values_for_active_indices(
    active: &[usize],
    values: &[(usize, f64)],
) -> Result<Vec<f64>, String> {
    let mut vals = Vec::with_capacity(active.len());
    for &dof in active {
        let n = values
            .iter()
            .find_map(|(i, v)| (*i == dof).then_some(*v))
            .ok_or_else(|| format!("basis values missing active dof {dof}"))?;
        vals.push(n);
    }
    Ok(vals)
}

fn evaluate_surface_shape(
    active: &[usize],
    active_u: &[usize],
    active_v: &[usize],
    vals_u: &[f64],
    ders_u: &[f64],
    vals_v: &[f64],
    ders_v: &[f64],
    weights: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let nu_loc = active_u.len();
    let nv_loc = active_v.len();
    let n_loc = nu_loc * nv_loc;
    if active.len() != n_loc {
        return Err(format!(
            "active size mismatch: active={}, expected={}",
            active.len(),
            n_loc
        ));
    }

    let mut phi = vec![0.0_f64; n_loc];
    let mut dphi_du = vec![0.0_f64; n_loc];
    let mut dphi_dv = vec![0.0_f64; n_loc];

    // B-spline tensor-product path.
    if weights.is_none() {
        for jv in 0..nv_loc {
            for iu in 0..nu_loc {
                let a = jv * nu_loc + iu;
                phi[a] = vals_u[iu] * vals_v[jv];
                dphi_du[a] = ders_u[iu] * vals_v[jv];
                dphi_dv[a] = vals_u[iu] * ders_v[jv];
            }
        }
        return Ok((phi, dphi_du, dphi_dv));
    }

    // Full tensor-product NURBS surface:
    // R_a = B_a / W, B_a = N_i(u) M_j(v) w_a
    // dR_a = (dB_a * W - B_a * dW) / W^2
    let weights = weights.unwrap();
    let mut b = vec![0.0_f64; n_loc];
    let mut db_du = vec![0.0_f64; n_loc];
    let mut db_dv = vec![0.0_f64; n_loc];
    let mut w_sum = 0.0_f64;
    let mut dw_du = 0.0_f64;
    let mut dw_dv = 0.0_f64;

    for jv in 0..nv_loc {
        for iu in 0..nu_loc {
            let a = jv * nu_loc + iu;
            let g = active[a];
            let wg = *weights
                .get(g)
                .ok_or_else(|| format!("weights missing global dof index {g}"))?;
            let bi = vals_u[iu] * vals_v[jv] * wg;
            let dbi_du = ders_u[iu] * vals_v[jv] * wg;
            let dbi_dv = vals_u[iu] * ders_v[jv] * wg;
            b[a] = bi;
            db_du[a] = dbi_du;
            db_dv[a] = dbi_dv;
            w_sum += bi;
            dw_du += dbi_du;
            dw_dv += dbi_dv;
        }
    }

    if w_sum.abs() <= 1e-30 {
        return Err("NURBS denominator W(u,v) is near zero".to_string());
    }
    let inv_w = 1.0 / w_sum;
    let inv_w2 = inv_w * inv_w;
    for a in 0..n_loc {
        phi[a] = b[a] * inv_w;
        dphi_du[a] = (db_du[a] * w_sum - b[a] * dw_du) * inv_w2;
        dphi_dv[a] = (db_dv[a] * w_sum - b[a] * dw_dv) * inv_w2;
    }
    Ok((phi, dphi_du, dphi_dv))
}

fn eval_surface_geometry(
    active: &[usize],
    ctrl_points: &[[f64; 2]],
    phi: &[f64],
    dphi_du: &[f64],
    dphi_dv: &[f64],
) -> Result<([f64; 2], [[f64; 2]; 2], f64), String> {
    let mut x = [0.0_f64; 2];
    let mut dx_du = [0.0_f64; 2];
    let mut dx_dv = [0.0_f64; 2];
    for (a, &g) in active.iter().enumerate() {
        let p = *ctrl_points
            .get(g)
            .ok_or_else(|| format!("missing control point for global dof {g}"))?;
        x[0] += phi[a] * p[0];
        x[1] += phi[a] * p[1];
        dx_du[0] += dphi_du[a] * p[0];
        dx_du[1] += dphi_du[a] * p[1];
        dx_dv[0] += dphi_dv[a] * p[0];
        dx_dv[1] += dphi_dv[a] * p[1];
    }

    let det_j = dx_du[0] * dx_dv[1] - dx_du[1] * dx_dv[0];
    if det_j.abs() <= 1e-20 {
        return Err("geometry Jacobian is near-singular at quadrature point".to_string());
    }
    Ok((x, [dx_du, dx_dv], det_j))
}

fn assemble_bilinear_iga_2d_impl(
    space: &IgaSpace2D,
    quad_order: u8,
    integrand: impl Fn(f64, [f64; 2], [f64; 2]) -> f64,
) -> Result<CsrMatrix<f64>, String> {
    let (basis_u, basis_v) = build_bases_from_space_2d(space)?;
    let spans = space.non_empty_spans();
    if spans.is_empty() {
        return Err("IgaSpace2D has no non-empty spans".to_string());
    }
    let knots_u = space.knot_slice_u();
    let knots_v = space.knot_slice_v();
    let ctrl_points = space.control_points();
    let (qpts, qwts) = quadrature_points_01(quad_order)?;
    let n_dofs = space.n_dofs();
    let mut coo = CooMatrix::new(n_dofs, n_dofs);

    for (span_u, span_v) in spans {
        let active = space.active_dofs_for_span(span_u, span_v)?;
        let active_u = basis_u.active_basis_indices(span_u);
        let active_v = basis_v.active_basis_indices(span_v);
        let nu_loc = active_u.len();
        let nv_loc = active_v.len();
        let n_loc = nu_loc * nv_loc;

        let ua = knots_u[span_u];
        let ub = knots_u[span_u + 1];
        let va = knots_v[span_v];
        let vb = knots_v[span_v + 1];
        let jac_u = ub - ua;
        let jac_v = vb - va;
        let cell_jac = jac_u * jac_v;

        for (&qu, &wu) in qpts.iter().zip(&qwts) {
            let u = ua + jac_u * qu;
            let vals_u = basis_u.nonzero_values(u)?;
            let ders_u = basis_u.nonzero_derivatives(u)?;
            let vals_u_local = local_values_for_active_indices(&active_u, &vals_u)?;
            let ders_u_local = local_values_for_active_indices(&active_u, &ders_u)?;

            for (&qv, &wv) in qpts.iter().zip(&qwts) {
                let v = va + jac_v * qv;
                let vals_v = basis_v.nonzero_values(v)?;
                let ders_v = basis_v.nonzero_derivatives(v)?;
                let vals_v_local = local_values_for_active_indices(&active_v, &vals_v)?;
                let ders_v_local = local_values_for_active_indices(&active_v, &ders_v)?;
                let wq_param = wu * wv * cell_jac;
                let (phi, dphi_du, dphi_dv) = evaluate_surface_shape(
                    &active,
                    &active_u,
                    &active_v,
                    &vals_u_local,
                    &ders_u_local,
                    &vals_v_local,
                    &ders_v_local,
                    space.weights(),
                )?;
                let (_x, jac, det_j) =
                    eval_surface_geometry(&active, ctrl_points, &phi, &dphi_du, &dphi_dv)?;
                let wq = wq_param * det_j.abs();

                // grad_x = J^{-T} grad_{u,v}
                let inv_t = [
                    [jac[1][1] / det_j, -jac[1][0] / det_j],
                    [-jac[0][1] / det_j, jac[0][0] / det_j],
                ];

                for a_loc in 0..n_loc {
                    let i = active[a_loc];
                    let na = phi[a_loc];
                    let ga = [
                        inv_t[0][0] * dphi_du[a_loc] + inv_t[0][1] * dphi_dv[a_loc],
                        inv_t[1][0] * dphi_du[a_loc] + inv_t[1][1] * dphi_dv[a_loc],
                    ];
                    for b_loc in 0..n_loc {
                        let j = active[b_loc];
                        let nb = phi[b_loc];
                        let gb = [
                            inv_t[0][0] * dphi_du[b_loc] + inv_t[0][1] * dphi_dv[b_loc],
                            inv_t[1][0] * dphi_du[b_loc] + inv_t[1][1] * dphi_dv[b_loc],
                        ];
                        coo.add(i, j, wq * integrand(na * nb, ga, gb));
                    }
                }
            }
        }
    }
    Ok(coo.into_csr())
}

pub fn assemble_bilinear_diffusion_iga_1d(
    space: &IgaSpace1D,
    quad_order: u8,
) -> Result<CsrMatrix<f64>, String> {
    // Parametric-domain operator assembly in u (geometry map/Jacobian factors deferred).
    assemble_bilinear_iga_1d_impl(space, quad_order, |_mass_term, diff_term| diff_term)
}

/// Physical Laplacian stiffness in 1D: `∫_Ω u' v' dx` on an interval/curve
/// with isogeometric map `x(u) = Σ_i c_i R_i(u)`.
///
/// `ctrl_x[i]` is the physical coordinate of control point `i` (same B-spline / NURBS basis
/// for geometry and the scalar trial/test space). The integrand uses
/// `dR_i/dx = (dR_i/du) / (dx/du)` with `dx/du = Σ_i c_i dR_i/du` (requires `dx/du > 0` at
/// all quadrature points: monotone `u ↦ x` on the patch).
pub fn assemble_bilinear_diffusion_iga_1d_physical(
    space: &IgaSpace1D,
    ctrl_x: &[f64],
    quad_order: u8,
) -> Result<CsrMatrix<f64>, String> {
    if ctrl_x.len() != space.n_dofs() {
        return Err(format!(
            "assemble_bilinear_diffusion_iga_1d_physical: ctrl len {} != n_dofs {}",
            ctrl_x.len(),
            space.n_dofs()
        ));
    }
    let basis = build_basis_from_space(space)?;
    let nurbs = space
        .weights()
        .map(|w| NurbsBasis::new(basis.clone(), w.to_vec()))
        .transpose()?;
    let spans = space.non_empty_spans();
    if spans.is_empty() {
        return Err("IgaSpace1D has no non-empty spans".to_string());
    }
    let knots = space.knot_slice();
    let (qpts, qwts) = quadrature_points_01(quad_order)?;
    let n_dofs = space.n_dofs();
    let mut coo = CooMatrix::new(n_dofs, n_dofs);

    for span in spans {
        let active = space.active_dofs_for_span(span)?;
        let ua = knots[span];
        let ub = knots[span + 1];
        let jac = ub - ua;

        for (&q, &w) in qpts.iter().zip(&qwts) {
            let u = ua + jac * q;
            let values = if let Some(nurbs_basis) = &nurbs {
                nurbs_basis.nonzero_values(u)?
            } else {
                basis.nonzero_values(u)?
            };
            let derivs = if let Some(nurbs_basis) = &nurbs {
                nurbs_basis.nonzero_derivatives(u)?
            } else {
                basis.nonzero_derivatives(u)?
            };
            let (_vals_local, ders_local) = local_values_for_active(&active, &values, &derivs)?;

            let dx_du = dx_du_iga_1d(&active, &ders_local, ctrl_x)?;
            if dx_du <= 1e-20 {
                return Err(format!(
                    "dx/du = {dx_du} is not positive (non-monotone or singular geometry at u={u})"
                ));
            }
            let inv = 1.0 / dx_du;

            for (a_loc, &i) in active.iter().enumerate() {
                let dni = ders_local[a_loc] * inv;
                for (b_loc, &j) in active.iter().enumerate() {
                    let dnj = ders_local[b_loc] * inv;
                    coo.add(i, j, w * jac * dni * dnj);
                }
            }
        }
    }
    Ok(coo.into_csr())
}

pub fn assemble_bilinear_mass_iga_1d(
    space: &IgaSpace1D,
    quad_order: u8,
) -> Result<CsrMatrix<f64>, String> {
    assemble_bilinear_iga_1d_impl(space, quad_order, |mass_term, _diff_term| mass_term)
}

/// Parametric 1D Helmholtz: `κ ∫ u' v' dû + ρ ∫ u v dû` in reference parameter `û` (same
/// B-spline / NURBS measure as [`assemble_bilinear_diffusion_iga_1d`] and
/// [`assemble_bilinear_mass_iga_1d`], one quadrature pass).
pub fn assemble_bilinear_helmholtz_iga_1d(
    space: &IgaSpace1D,
    kappa: f64,
    rho: f64,
    quad_order: u8,
) -> Result<CsrMatrix<f64>, String> {
    let has_k = kappa != 0.0;
    let has_m = rho != 0.0;
    if !has_k && !has_m {
        return Err("assemble_bilinear_helmholtz_iga_1d: kappa and rho are both zero".to_string());
    }
    assemble_bilinear_iga_1d_impl(space, quad_order, |ni_nj, duiduj| {
        let mut s = 0.0_f64;
        if has_k {
            s += kappa * duiduj;
        }
        if has_m {
            s += rho * ni_nj;
        }
        s
    })
}

/// Physical L² mass in 1D: `∫_Ω u v dx` with `x(u) = Σ_i c_i R_i(u)` and measure `dx = (dx/du) du`.
pub fn assemble_bilinear_mass_iga_1d_physical(
    space: &IgaSpace1D,
    ctrl_x: &[f64],
    quad_order: u8,
) -> Result<CsrMatrix<f64>, String> {
    if ctrl_x.len() != space.n_dofs() {
        return Err(format!(
            "assemble_bilinear_mass_iga_1d_physical: ctrl len {} != n_dofs {}",
            ctrl_x.len(),
            space.n_dofs()
        ));
    }
    let basis = build_basis_from_space(space)?;
    let nurbs = space
        .weights()
        .map(|w| NurbsBasis::new(basis.clone(), w.to_vec()))
        .transpose()?;
    let spans = space.non_empty_spans();
    if spans.is_empty() {
        return Err("IgaSpace1D has no non-empty spans".to_string());
    }
    let knots = space.knot_slice();
    let (qpts, qwts) = quadrature_points_01(quad_order)?;
    let n_dofs = space.n_dofs();
    let mut coo = CooMatrix::new(n_dofs, n_dofs);

    for span in spans {
        let active = space.active_dofs_for_span(span)?;
        let ua = knots[span];
        let ub = knots[span + 1];
        let jac = ub - ua;

        for (&q, &w) in qpts.iter().zip(&qwts) {
            let u = ua + jac * q;
            let values = if let Some(nurbs_basis) = &nurbs {
                nurbs_basis.nonzero_values(u)?
            } else {
                basis.nonzero_values(u)?
            };
            let derivs = if let Some(nurbs_basis) = &nurbs {
                nurbs_basis.nonzero_derivatives(u)?
            } else {
                basis.nonzero_derivatives(u)?
            };
            let (vals_local, ders_local) = local_values_for_active(&active, &values, &derivs)?;
            let dx_du = dx_du_iga_1d(&active, &ders_local, ctrl_x)?;
            if dx_du <= 1e-20 {
                return Err(format!(
                    "dx/du = {dx_du} is not positive (geometry at u={u})"
                ));
            }
            for (a_loc, &i) in active.iter().enumerate() {
                let ni = vals_local[a_loc];
                for (b_loc, &j) in active.iter().enumerate() {
                    let nj = vals_local[b_loc];
                    coo.add(i, j, w * jac * ni * nj * dx_du);
                }
            }
        }
    }
    Ok(coo.into_csr())
}

/// Fused Helmholtz-type matrix in physical 1D:
/// `κ ∫ u' v' dx + ρ ∫ u v dx` with the same isogeometric map as
/// [`assemble_bilinear_diffusion_iga_1d_physical`] / [`assemble_bilinear_mass_iga_1d_physical`].
///
/// One span/quadrature pass; numerically equivalent to
/// `κ * K_diff + ρ * M_mass` from those assemblers but cheaper when both terms are needed.
/// If `κ = 0` or `ρ = 0`, the corresponding block is skipped (the other must be non-zero).
pub fn assemble_bilinear_helmholtz_iga_1d_physical(
    space: &IgaSpace1D,
    ctrl_x: &[f64],
    kappa: f64,
    rho: f64,
    quad_order: u8,
) -> Result<CsrMatrix<f64>, String> {
    if ctrl_x.len() != space.n_dofs() {
        return Err(format!(
            "assemble_bilinear_helmholtz_iga_1d_physical: ctrl len {} != n_dofs {}",
            ctrl_x.len(),
            space.n_dofs()
        ));
    }
    let has_k = kappa != 0.0;
    let has_m = rho != 0.0;
    if !has_k && !has_m {
        return Err("assemble_bilinear_helmholtz_iga_1d_physical: kappa and rho are both zero"
            .to_string());
    }

    let basis = build_basis_from_space(space)?;
    let nurbs = space
        .weights()
        .map(|w| NurbsBasis::new(basis.clone(), w.to_vec()))
        .transpose()?;
    let spans = space.non_empty_spans();
    if spans.is_empty() {
        return Err("IgaSpace1D has no non-empty spans".to_string());
    }
    let knots = space.knot_slice();
    let (qpts, qwts) = quadrature_points_01(quad_order)?;
    let n_dofs = space.n_dofs();
    let mut coo = CooMatrix::new(n_dofs, n_dofs);

    for span in spans {
        let active = space.active_dofs_for_span(span)?;
        let ua = knots[span];
        let ub = knots[span + 1];
        let jac = ub - ua;

        for (&q, &w) in qpts.iter().zip(&qwts) {
            let u = ua + jac * q;
            let values = if let Some(nurbs_basis) = &nurbs {
                nurbs_basis.nonzero_values(u)?
            } else {
                basis.nonzero_values(u)?
            };
            let derivs = if let Some(nurbs_basis) = &nurbs {
                nurbs_basis.nonzero_derivatives(u)?
            } else {
                basis.nonzero_derivatives(u)?
            };
            let (vals_local, ders_local) = local_values_for_active(&active, &values, &derivs)?;
            let dx_du = dx_du_iga_1d(&active, &ders_local, ctrl_x)?;
            if dx_du <= 1e-20 {
                return Err(format!(
                    "dx/du = {dx_du} is not positive (geometry at u={u})"
                ));
            }
            let inv = 1.0 / dx_du;

            for (a_loc, &i) in active.iter().enumerate() {
                let ni = vals_local[a_loc];
                let dni = ders_local[a_loc];
                for (b_loc, &j) in active.iter().enumerate() {
                    let nj = vals_local[b_loc];
                    let dnj = ders_local[b_loc];
                    let mut s = 0.0_f64;
                    if has_k {
                        s += kappa * (dni * inv) * (dnj * inv);
                    }
                    if has_m {
                        s += rho * ni * nj * dx_du;
                    }
                    coo.add(i, j, w * jac * s);
                }
            }
        }
    }
    Ok(coo.into_csr())
}

/// Assemble the single-patch 2D diffusion bilinear form in physical coordinates.
///
/// Uses the isogeometric map `x(u,v)` from the patch control net and integrates
/// `\int_\Omega \nabla N_a \cdot \nabla N_b \, dx`.
/// Supports both B-spline and weighted NURBS patch bases.
pub fn assemble_bilinear_diffusion_iga_2d(
    space: &IgaSpace2D,
    quad_order: u8,
) -> Result<CsrMatrix<f64>, String> {
    // Physical-domain diffusion via isogeometric geometry map x(u,v).
    assemble_bilinear_iga_2d_impl(space, quad_order, |_mass_term, ga, gb| {
        ga[0] * gb[0] + ga[1] * gb[1]
    })
}

/// Assemble the single-patch 2D mass bilinear form in physical coordinates.
///
/// Uses the isogeometric map `x(u,v)` from the patch control net and integrates
/// `\int_\Omega N_a N_b \, dx`.
/// Supports both B-spline and weighted NURBS patch bases.
pub fn assemble_bilinear_mass_iga_2d(
    space: &IgaSpace2D,
    quad_order: u8,
) -> Result<CsrMatrix<f64>, String> {
    // Physical-domain mass via isogeometric geometry map x(u,v).
    assemble_bilinear_iga_2d_impl(space, quad_order, |mass_term, _ga, _gb| mass_term)
}

/// Fused Helmholtz-type operator in physical 2D: `κ ∫ ∇u·∇v dx + ρ ∫ u v dx` on a single
/// IGA / NURBS patch (same path as [`assemble_bilinear_diffusion_iga_2d`] and
/// [`assemble_bilinear_mass_iga_2d`], one tensor-product quadrature pass).
pub fn assemble_bilinear_helmholtz_iga_2d(
    space: &IgaSpace2D,
    kappa: f64,
    rho: f64,
    quad_order: u8,
) -> Result<CsrMatrix<f64>, String> {
    let has_k = kappa != 0.0;
    let has_m = rho != 0.0;
    if !has_k && !has_m {
        return Err("assemble_bilinear_helmholtz_iga_2d: kappa and rho are both zero".to_string());
    }
    assemble_bilinear_iga_2d_impl(space, quad_order, |mass_term, ga, gb| {
        let mut s = 0.0_f64;
        if has_k {
            s += kappa * (ga[0] * gb[0] + ga[1] * gb[1]);
        }
        if has_m {
            s += rho * mass_term;
        }
        s
    })
}

pub fn assemble_linear_source_iga_1d<F: Fn(f64) -> f64>(
    space: &IgaSpace1D,
    source: F,
    quad_order: u8,
) -> Result<Vec<f64>, String> {
    // Parametric-domain load assembly in u (geometry map/Jacobian factors deferred).
    let basis = build_basis_from_space(space)?;
    let nurbs = space
        .weights()
        .map(|w| NurbsBasis::new(basis.clone(), w.to_vec()))
        .transpose()?;
    let spans = space.non_empty_spans();
    if spans.is_empty() {
        return Err("IgaSpace1D has no non-empty spans".to_string());
    }
    let knots = space.knot_slice();
    let (qpts, qwts) = quadrature_points_01(quad_order)?;
    let mut rhs = vec![0.0; space.n_dofs()];

    for span in spans {
        let active = space.active_dofs_for_span(span)?;
        let ua = knots[span];
        let ub = knots[span + 1];
        let jac = ub - ua;
        for (&q, &w) in qpts.iter().zip(&qwts) {
            let u = ua + jac * q;
            let values = if let Some(nurbs_basis) = &nurbs {
                nurbs_basis.nonzero_values(u)?
            } else {
                basis.nonzero_values(u)?
            };
            let derivs = if let Some(nurbs_basis) = &nurbs {
                nurbs_basis.nonzero_derivatives(u)?
            } else {
                basis.nonzero_derivatives(u)?
            };
            let (vals_local, _) = local_values_for_active(&active, &values, &derivs)?;
            let f = source(u);
            for (a_loc, &i) in active.iter().enumerate() {
                rhs[i] += w * jac * vals_local[a_loc] * f;
            }
        }
    }
    Ok(rhs)
}

/// `∫_Ω f(x) v dx` with `f` evaluated at physical `x` from the 1D isogeometric map
/// `x(u) = Σ_i c_i R_i(u)`.
pub fn assemble_linear_source_iga_1d_physical<F: Fn(f64) -> f64>(
    space: &IgaSpace1D,
    ctrl_x: &[f64],
    source: F,
    quad_order: u8,
) -> Result<Vec<f64>, String> {
    if ctrl_x.len() != space.n_dofs() {
        return Err(format!(
            "assemble_linear_source_iga_1d_physical: ctrl len {} != n_dofs {}",
            ctrl_x.len(),
            space.n_dofs()
        ));
    }
    let basis = build_basis_from_space(space)?;
    let nurbs = space
        .weights()
        .map(|w| NurbsBasis::new(basis.clone(), w.to_vec()))
        .transpose()?;
    let spans = space.non_empty_spans();
    if spans.is_empty() {
        return Err("IgaSpace1D has no non-empty spans".to_string());
    }
    let knots = space.knot_slice();
    let (qpts, qwts) = quadrature_points_01(quad_order)?;
    let mut rhs = vec![0.0; space.n_dofs()];

    for span in spans {
        let active = space.active_dofs_for_span(span)?;
        let ua = knots[span];
        let ub = knots[span + 1];
        let jac = ub - ua;
        for (&q, &w) in qpts.iter().zip(&qwts) {
            let u = ua + jac * q;
            let values = if let Some(nurbs_basis) = &nurbs {
                nurbs_basis.nonzero_values(u)?
            } else {
                basis.nonzero_values(u)?
            };
            let derivs = if let Some(nurbs_basis) = &nurbs {
                nurbs_basis.nonzero_derivatives(u)?
            } else {
                basis.nonzero_derivatives(u)?
            };
            let (vals_local, ders_local) = local_values_for_active(&active, &values, &derivs)?;
            let mut x_phys = 0.0_f64;
            for &(i, v) in &values {
                let c = *ctrl_x
                    .get(i)
                    .ok_or_else(|| format!("ctrl_x index {i} out of range"))?;
                x_phys += c * v;
            }
            let dx_du = dx_du_iga_1d(&active, &ders_local, ctrl_x)?;
            if dx_du <= 1e-20 {
                return Err(format!(
                    "dx/du = {dx_du} is not positive (geometry at u={u})"
                ));
            }
            let f = source(x_phys);
            for (a_loc, &i) in active.iter().enumerate() {
                rhs[i] += w * jac * vals_local[a_loc] * f * dx_du;
            }
        }
    }
    Ok(rhs)
}

/// Assemble the single-patch 2D source linear form in physical coordinates.
///
/// The callback is evaluated at physical points `x(u,v)` from the patch geometry.
/// Supports both B-spline and weighted NURBS patch bases.
pub fn assemble_linear_source_iga_2d<F: Fn([f64; 2]) -> f64>(
    space: &IgaSpace2D,
    source: F,
    quad_order: u8,
) -> Result<Vec<f64>, String> {
    let (basis_u, basis_v) = build_bases_from_space_2d(space)?;
    let spans = space.non_empty_spans();
    if spans.is_empty() {
        return Err("IgaSpace2D has no non-empty spans".to_string());
    }
    let knots_u = space.knot_slice_u();
    let knots_v = space.knot_slice_v();
    let ctrl_points = space.control_points();
    let (qpts, qwts) = quadrature_points_01(quad_order)?;
    let mut rhs = vec![0.0; space.n_dofs()];

    for (span_u, span_v) in spans {
        let active = space.active_dofs_for_span(span_u, span_v)?;
        let active_u = basis_u.active_basis_indices(span_u);
        let active_v = basis_v.active_basis_indices(span_v);
        let nu_loc = active_u.len();
        let nv_loc = active_v.len();
        let n_loc = nu_loc * nv_loc;

        let ua = knots_u[span_u];
        let ub = knots_u[span_u + 1];
        let va = knots_v[span_v];
        let vb = knots_v[span_v + 1];
        let jac_u = ub - ua;
        let jac_v = vb - va;
        let cell_jac = jac_u * jac_v;

        for (&qu, &wu) in qpts.iter().zip(&qwts) {
            let u = ua + jac_u * qu;
            let vals_u = basis_u.nonzero_values(u)?;
            let ders_u = basis_u.nonzero_derivatives(u)?;
            let vals_u_local = local_values_for_active_indices(&active_u, &vals_u)?;
            let ders_u_local = local_values_for_active_indices(&active_u, &ders_u)?;

            for (&qv, &wv) in qpts.iter().zip(&qwts) {
                let v = va + jac_v * qv;
                let vals_v = basis_v.nonzero_values(v)?;
                let ders_v = basis_v.nonzero_derivatives(v)?;
                let vals_v_local = local_values_for_active_indices(&active_v, &vals_v)?;
                let ders_v_local = local_values_for_active_indices(&active_v, &ders_v)?;
                let wq_param = wu * wv * cell_jac;
                let (phi, dphi_du, dphi_dv) = evaluate_surface_shape(
                    &active,
                    &active_u,
                    &active_v,
                    &vals_u_local,
                    &ders_u_local,
                    &vals_v_local,
                    &ders_v_local,
                    space.weights(),
                )?;
                let (x, _jac, det_j) =
                    eval_surface_geometry(&active, ctrl_points, &phi, &dphi_du, &dphi_dv)?;
                let wq = wq_param * det_j.abs();
                let f = source(x);
                for a_loc in 0..n_loc {
                    rhs[active[a_loc]] += wq * phi[a_loc] * f;
                }
            }
        }
    }

    Ok(rhs)
}

#[cfg(test)]
mod tests {
    use super::{
        assemble_bilinear_diffusion_iga_1d, assemble_bilinear_diffusion_iga_1d_physical,
        assemble_bilinear_helmholtz_iga_1d, assemble_bilinear_helmholtz_iga_1d_physical,
        assemble_bilinear_mass_iga_1d, assemble_bilinear_mass_iga_1d_physical,
        assemble_linear_source_iga_1d, assemble_linear_source_iga_1d_physical,
        assemble_bilinear_diffusion_iga_2d, assemble_bilinear_helmholtz_iga_2d,
        assemble_bilinear_mass_iga_2d, assemble_linear_source_iga_2d,
    };
    use fem_linalg::CsrMatrix;
    use fem_space::iga::IgaSpace1D;
    use fem_space::iga::IgaSpace2D;

    fn assert_symmetric(mat: &CsrMatrix<f64>, tol: f64) {
        for i in 0..mat.nrows {
            for j in 0..mat.ncols {
                let aij = mat.get(i, j);
                let aji = mat.get(j, i);
                assert!(
                    (aij - aji).abs() <= tol,
                    "matrix not symmetric at ({i},{j}): aij={aij}, aji={aji}"
                );
            }
        }
    }

    #[test]
    fn iga_1d_mass_and_diffusion_shapes_match_dofs() {
        let space = IgaSpace1D::new_uniform_clamped(2, 6).unwrap();
        let n = space.n_dofs();
        let k = assemble_bilinear_diffusion_iga_1d(&space, 4).unwrap();
        let m = assemble_bilinear_mass_iga_1d(&space, 4).unwrap();
        assert_eq!(k.nrows, n);
        assert_eq!(k.ncols, n);
        assert_eq!(m.nrows, n);
        assert_eq!(m.ncols, n);
    }

    #[test]
    fn iga_1d_mass_and_diffusion_are_symmetric() {
        let space = IgaSpace1D::new_uniform_clamped(3, 8).unwrap();
        let k = assemble_bilinear_diffusion_iga_1d(&space, 4).unwrap();
        let m = assemble_bilinear_mass_iga_1d(&space, 4).unwrap();
        assert_symmetric(&k, 1e-10);
        assert_symmetric(&m, 1e-10);
    }

    #[test]
    fn iga_1d_diffusion_diagonal_is_nonnegative_with_positive_entry() {
        let space = IgaSpace1D::new_uniform_clamped(2, 6).unwrap();
        let k = assemble_bilinear_diffusion_iga_1d(&space, 4).unwrap();
        let diag = k.diagonal();
        assert!(diag.iter().all(|d| *d >= -1e-12));
        assert!(diag.iter().any(|d| *d > 1e-12));
    }

    #[test]
    fn iga_1d_source_vector_has_expected_length_and_finite_entries() {
        let space = IgaSpace1D::new_uniform_clamped(2, 6).unwrap();
        let rhs = assemble_linear_source_iga_1d(&space, |x| 1.0 + x, 4).unwrap();
        assert_eq!(rhs.len(), space.n_dofs());
        assert!(rhs.iter().all(|v| v.is_finite()));
    }

    /// Identity map: linear B-splines on `[0,1]` with $c_i = i/2$ gives $x(u)=u$ and
    /// physical diffusion should match the existing parametric (`u`-only) operator.
    #[test]
    fn iga_1d_physical_diffusion_matches_parametric_for_identity_map() {
        let space = IgaSpace1D::new_uniform_clamped(1, 3).expect("1d");
        let ctrl = vec![0.0_f64, 0.5, 1.0];
        let k_par = assemble_bilinear_diffusion_iga_1d(&space, 4).expect("k");
        let k_phys = assemble_bilinear_diffusion_iga_1d_physical(&space, &ctrl, 4).expect("kp");
        for i in 0..k_par.nrows {
            for j in 0..k_par.ncols {
                let a = k_par.get(i, j);
                let b = k_phys.get(i, j);
                assert!((a - b).abs() < 1e-9, "({i},{j}): {a} vs {b}");
            }
        }
    }

    /// Constant unit load: with $x(u)=u$, `∫ 1·N_i dx` matches parametric `∫ 1·N_i du`.
    #[test]
    fn iga_1d_physical_source_constant_matches_parametric() {
        let space = IgaSpace1D::new_uniform_clamped(1, 3).expect("1d");
        let ctrl = vec![0.0_f64, 0.5, 1.0];
        let a = assemble_linear_source_iga_1d(&space, |_| 1.0_f64, 4).expect("a");
        let b = assemble_linear_source_iga_1d_physical(&space, &ctrl, |_| 1.0_f64, 4)
            .expect("b");
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() < 1e-9, "i={i}: {} vs {}", a[i], b[i]);
        }
    }

    /// With $x(u)=u$, `∫ N_i N_j dx` should match the parametric `∫ N_i N_j du`.
    #[test]
    fn iga_1d_physical_mass_matches_parametric_for_identity_map() {
        let space = IgaSpace1D::new_uniform_clamped(1, 3).expect("1d");
        let ctrl = vec![0.0_f64, 0.5, 1.0];
        let m_par = assemble_bilinear_mass_iga_1d(&space, 4).expect("m");
        let m_phys = assemble_bilinear_mass_iga_1d_physical(&space, &ctrl, 4).expect("mp");
        for i in 0..m_par.nrows {
            for j in 0..m_par.ncols {
                let a = m_par.get(i, j);
                let b = m_phys.get(i, j);
                assert!((a - b).abs() < 1e-9, "({i},{j}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn iga_1d_helmholtz_physical_matches_scaled_sum() {
        let space = IgaSpace1D::new_uniform_clamped(2, 8).expect("1d");
        let n = space.n_dofs();
        let ctrl: Vec<f64> = (0..n)
            .map(|i| i as f64 / (n.saturating_sub(1).max(1) as f64))
            .collect();
        let kappa = 1.7_f64;
        let rho = 0.23_f64;
        let q = 4_u8;
        let h = assemble_bilinear_helmholtz_iga_1d_physical(&space, &ctrl, kappa, rho, q)
            .expect("h");
        let k = assemble_bilinear_diffusion_iga_1d_physical(&space, &ctrl, q).expect("k");
        let m = assemble_bilinear_mass_iga_1d_physical(&space, &ctrl, q).expect("m");
        let sum = k.axpby(kappa, &m, rho);
        for i in 0..h.nrows {
            for j in 0..h.ncols {
                let a = h.get(i, j);
                let b = sum.get(i, j);
                assert!((a - b).abs() < 1e-9, "({i},{j}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn iga_1d_helmholtz_parametric_matches_scaled_sum() {
        let space = IgaSpace1D::new_uniform_clamped(2, 7).expect("1d");
        let kappa = 0.8_f64;
        let rho = 0.15_f64;
        let q = 4_u8;
        let h = assemble_bilinear_helmholtz_iga_1d(&space, kappa, rho, q).expect("h");
        let k = assemble_bilinear_diffusion_iga_1d(&space, q).expect("k");
        let m = assemble_bilinear_mass_iga_1d(&space, q).expect("m");
        let sum = k.axpby(kappa, &m, rho);
        for i in 0..h.nrows {
            for j in 0..h.ncols {
                let a = h.get(i, j);
                let b = sum.get(i, j);
                assert!((a - b).abs() < 1e-9, "({i},{j}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn iga_1d_nonuniform_knots_preserve_symmetry_and_shapes() {
        let space = IgaSpace1D::new(
            2,
            vec![0.0, 0.0, 0.0, 0.2, 0.65, 1.0, 1.0, 1.0],
            5,
            None,
        )
        .unwrap();
        let n = space.n_dofs();
        let k = assemble_bilinear_diffusion_iga_1d(&space, 4).unwrap();
        let m = assemble_bilinear_mass_iga_1d(&space, 4).unwrap();
        assert_eq!(k.nrows, n);
        assert_eq!(k.ncols, n);
        assert_eq!(m.nrows, n);
        assert_eq!(m.ncols, n);
        assert_symmetric(&k, 1e-10);
        assert_symmetric(&m, 1e-10);
    }

    #[test]
    fn iga_2d_mass_and_diffusion_shapes_match_dofs() {
        let space = IgaSpace2D::new_uniform_clamped(2, 2, 6, 5).unwrap();
        let n = space.n_dofs();
        let k = assemble_bilinear_diffusion_iga_2d(&space, 4).unwrap();
        let m = assemble_bilinear_mass_iga_2d(&space, 4).unwrap();
        assert_eq!(k.nrows, n);
        assert_eq!(k.ncols, n);
        assert_eq!(m.nrows, n);
        assert_eq!(m.ncols, n);
    }

    #[test]
    fn iga_2d_mass_and_diffusion_are_symmetric() {
        let space = IgaSpace2D::new_uniform_clamped(2, 2, 6, 5).unwrap();
        let k = assemble_bilinear_diffusion_iga_2d(&space, 4).unwrap();
        let m = assemble_bilinear_mass_iga_2d(&space, 4).unwrap();
        assert_symmetric(&k, 1e-10);
        assert_symmetric(&m, 1e-10);
    }

    #[test]
    fn iga_2d_helmholtz_matches_scaled_sum() {
        let space = IgaSpace2D::new_uniform_clamped(1, 1, 3, 3).expect("2d");
        let kappa = 1.4_f64;
        let rho = 0.11_f64;
        let q = 3_u8;
        let h = assemble_bilinear_helmholtz_iga_2d(&space, kappa, rho, q).expect("h");
        let k = assemble_bilinear_diffusion_iga_2d(&space, q).expect("k");
        let m = assemble_bilinear_mass_iga_2d(&space, q).expect("m");
        let sum = k.axpby(kappa, &m, rho);
        for i in 0..h.nrows {
            for j in 0..h.ncols {
                let a = h.get(i, j);
                let b = sum.get(i, j);
                assert!((a - b).abs() < 1e-9, "({i},{j}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn iga_2d_diffusion_diagonal_is_nonnegative_with_positive_entry() {
        let space = IgaSpace2D::new_uniform_clamped(2, 2, 6, 5).unwrap();
        let k = assemble_bilinear_diffusion_iga_2d(&space, 4).unwrap();
        let diag = k.diagonal();
        assert!(diag.iter().all(|d| *d >= -1e-12));
        assert!(diag.iter().any(|d| *d > 1e-12));
    }

    #[test]
    fn iga_2d_source_vector_has_expected_length_and_finite_entries() {
        let space = IgaSpace2D::new_uniform_clamped(2, 2, 6, 5).unwrap();
        let rhs = assemble_linear_source_iga_2d(&space, |x| 1.0 + x[0] + x[1], 4).unwrap();
        assert_eq!(rhs.len(), space.n_dofs());
        assert!(rhs.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn iga_2d_mass_and_source_are_consistent_for_constant_one() {
        let space = IgaSpace2D::new_uniform_clamped(2, 2, 6, 5).unwrap();
        let m = assemble_bilinear_mass_iga_2d(&space, 4).unwrap();
        let rhs = assemble_linear_source_iga_2d(&space, |_x| 1.0, 4).unwrap();
        let ones = vec![1.0; space.n_dofs()];
        let mut m_ones = vec![0.0; space.n_dofs()];
        m.spmv(&ones, &mut m_ones);

        // For default unit-square control net, geometry is identity and area is 1.
        let mass_integral = ones
            .iter()
            .zip(m_ones.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let source_integral = rhs.iter().sum::<f64>();
        assert!((mass_integral - 1.0).abs() <= 1e-11, "mass integral={mass_integral}");
        assert!(
            (source_integral - 1.0).abs() <= 1e-11,
            "source integral={source_integral}"
        );
    }

    #[test]
    fn iga_2d_weighted_nurbs_assembles_and_is_symmetric() {
        let mut weights = vec![1.0_f64; 6 * 5];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = 1.0 + 0.05 * (i as f64);
        }
        let space = IgaSpace2D::new(
            2,
            2,
            vec![0.0, 0.0, 0.0, 0.2, 0.55, 0.8, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.35, 0.7, 1.0, 1.0, 1.0],
            6,
            5,
            Some(weights),
        )
        .unwrap();

        let k = assemble_bilinear_diffusion_iga_2d(&space, 4).unwrap();
        let m = assemble_bilinear_mass_iga_2d(&space, 4).unwrap();
        let rhs = assemble_linear_source_iga_2d(&space, |_x| 1.0, 4).unwrap();

        assert_eq!(k.nrows, space.n_dofs());
        assert_eq!(m.nrows, space.n_dofs());
        assert!(rhs.iter().all(|v| v.is_finite()));
        assert_symmetric(&k, 1e-10);
        assert_symmetric(&m, 1e-10);
    }
}
