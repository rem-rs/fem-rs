use fem_element::iga::{BsplineBasis, KnotVector};

#[derive(Clone, Debug)]
pub struct IgaSpace1D {
    basis: BsplineBasis,
    knots: Vec<f64>,
    n_ctrl: usize,
    weights: Option<Vec<f64>>,
}

impl IgaSpace1D {
    pub fn new(
        degree: usize,
        knots: Vec<f64>,
        n_ctrl: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self, String> {
        if n_ctrl == 0 {
            return Err("IgaSpace1D: n_ctrl must be > 0".to_string());
        }
        let kv = KnotVector::new_clamped(knots.clone())?;
        let basis = BsplineBasis::new(degree, kv)?;
        if basis.n_basis() != n_ctrl {
            return Err(format!(
                "IgaSpace1D: n_ctrl ({n_ctrl}) does not match basis count ({})",
                basis.n_basis()
            ));
        }
        validate_weights(weights.as_deref(), n_ctrl, "IgaSpace1D")?;
        Ok(Self {
            basis,
            knots,
            n_ctrl,
            weights,
        })
    }

    pub fn new_uniform_clamped(degree: usize, n_ctrl: usize) -> Result<Self, String> {
        let knots = uniform_clamped_knots(degree, n_ctrl)?;
        Self::new(degree, knots, n_ctrl, None)
    }

    pub fn n_dofs(&self) -> usize {
        self.n_ctrl
    }

    pub fn degree(&self) -> usize {
        self.basis.degree()
    }

    pub fn knot_slice(&self) -> &[f64] {
        &self.knots
    }

    pub fn non_empty_spans(&self) -> Vec<usize> {
        let p = self.basis.degree();
        let n = self.basis.n_basis() - 1;
        (p..=n)
            .filter(|&span| self.knots[span] < self.knots[span + 1])
            .collect()
    }

    pub fn active_dofs_for_span(&self, span: usize) -> Result<Vec<usize>, String> {
        let p = self.basis.degree();
        let n = self.basis.n_basis() - 1;
        if span < p || span > n {
            return Err(format!(
                "IgaSpace1D: span {span} out of valid range [{p}, {n}]"
            ));
        }
        if self.knots[span] >= self.knots[span + 1] {
            return Err(format!(
                "IgaSpace1D: span {span} is empty (knot[{span}] == knot[{}] == {})",
                span + 1,
                self.knots[span]
            ));
        }
        Ok(self.basis.active_basis_indices(span))
    }

    pub fn boundary_dofs(&self) -> (usize, usize) {
        (0, self.n_ctrl - 1)
    }

    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// Greville abscissae in parametric form (one per basis / control point).
    ///
    /// For degree `p > 0`: `ξ_i = (1/p) * Σ_{j=1}^{p} t_{i+j}` with knot vector `t` from
    /// this space. Used e.g. for [`crate::iga_fe_space::IgaFESpace1D`] node coordinates
    /// when no separate geometric map is stored.
    pub fn greville_param_coords(&self) -> Result<Vec<f64>, String> {
        let p = self.basis.degree();
        if p == 0 {
            return Err("IgaSpace1D: greville for degree-0 (discontinuous) is not defined here".to_string());
        }
        let n = self.n_ctrl;
        let m = self.knots.len();
        if m < n + p + 1 {
            return Err("IgaSpace1D: knot length incompatible with greville abscissae".to_string());
        }
        let mut g = vec![0.0_f64; n];
        for i in 0..n {
            let s: f64 = (1..=p).map(|j| self.knots[i + j]).sum();
            g[i] = s / p as f64;
        }
        Ok(g)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IgaBoundary2D {
    UMin,
    UMax,
    VMin,
    VMax,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IgaBoundary3D {
    UMin,
    UMax,
    VMin,
    VMax,
    WMin,
    WMax,
}

/// Minimal single-patch tensor-product IGA space in 2D.
///
/// Global control-point DOFs use row-major indexing:
/// `global = j * nu + i`, where `i` is the u-index and `j` is the v-index.
///
/// Boundary DOF ordering conventions:
/// - `UMin`, `UMax`: iterate in increasing `j` (v direction),
/// - `VMin`, `VMax`: iterate in increasing `i` (u direction).
#[derive(Clone, Debug)]
pub struct IgaSpace2D {
    basis_u: BsplineBasis,
    basis_v: BsplineBasis,
    knots_u: Vec<f64>,
    knots_v: Vec<f64>,
    nu: usize,
    nv: usize,
    weights: Option<Vec<f64>>,
    ctrl_points: Vec<[f64; 2]>,
}

impl IgaSpace2D {
    pub fn new(
        p: usize,
        q: usize,
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
        nu: usize,
        nv: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self, String> {
        let ctrl_points = default_unit_square_ctrl_points(nu, nv)?;
        Self::new_with_ctrl_points(p, q, knots_u, knots_v, nu, nv, weights, ctrl_points)
    }

    pub fn new_with_ctrl_points(
        p: usize,
        q: usize,
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
        nu: usize,
        nv: usize,
        weights: Option<Vec<f64>>,
        ctrl_points: Vec<[f64; 2]>,
    ) -> Result<Self, String> {
        if nu == 0 || nv == 0 {
            return Err("IgaSpace2D: nu and nv must be > 0".to_string());
        }
        let ku = KnotVector::new_clamped(knots_u.clone())?;
        let kv = KnotVector::new_clamped(knots_v.clone())?;
        let basis_u = BsplineBasis::new(p, ku)?;
        let basis_v = BsplineBasis::new(q, kv)?;
        if basis_u.n_basis() != nu {
            return Err(format!(
                "IgaSpace2D: nu ({nu}) does not match u basis count ({})",
                basis_u.n_basis()
            ));
        }
        if basis_v.n_basis() != nv {
            return Err(format!(
                "IgaSpace2D: nv ({nv}) does not match v basis count ({})",
                basis_v.n_basis()
            ));
        }
        validate_weights(weights.as_deref(), nu * nv, "IgaSpace2D")?;
        if ctrl_points.len() != nu * nv {
            return Err(format!(
                "IgaSpace2D: ctrl_points length ({}) does not match expected count ({})",
                ctrl_points.len(),
                nu * nv
            ));
        }
        Ok(Self {
            basis_u,
            basis_v,
            knots_u,
            knots_v,
            nu,
            nv,
            weights,
            ctrl_points,
        })
    }

    pub fn new_uniform_clamped(p: usize, q: usize, nu: usize, nv: usize) -> Result<Self, String> {
        let knots_u = uniform_clamped_knots(p, nu)?;
        let knots_v = uniform_clamped_knots(q, nv)?;
        Self::new(p, q, knots_u, knots_v, nu, nv, None)
    }

    pub fn n_dofs(&self) -> usize {
        self.nu * self.nv
    }

    pub fn degree_u(&self) -> usize {
        self.basis_u.degree()
    }

    pub fn degree_v(&self) -> usize {
        self.basis_v.degree()
    }

    pub fn knot_slice_u(&self) -> &[f64] {
        &self.knots_u
    }

    pub fn knot_slice_v(&self) -> &[f64] {
        &self.knots_v
    }

    pub fn non_empty_spans(&self) -> Vec<(usize, usize)> {
        let p = self.basis_u.degree();
        let nu_last = self.basis_u.n_basis() - 1;
        let q = self.basis_v.degree();
        let nv_last = self.basis_v.n_basis() - 1;

        let u_spans: Vec<usize> = (p..=nu_last)
            .filter(|&span| self.knots_u[span] < self.knots_u[span + 1])
            .collect();
        let v_spans: Vec<usize> = (q..=nv_last)
            .filter(|&span| self.knots_v[span] < self.knots_v[span + 1])
            .collect();

        let mut spans = Vec::with_capacity(u_spans.len() * v_spans.len());
        for span_v in v_spans {
            for &span_u in &u_spans {
                spans.push((span_u, span_v));
            }
        }
        spans
    }

    pub fn active_dofs_for_span(
        &self,
        span_u: usize,
        span_v: usize,
    ) -> Result<Vec<usize>, String> {
        let p = self.basis_u.degree();
        let nu_last = self.basis_u.n_basis() - 1;
        let q = self.basis_v.degree();
        let nv_last = self.basis_v.n_basis() - 1;
        if span_u < p || span_u > nu_last {
            return Err(format!(
                "IgaSpace2D: span_u {span_u} out of valid range [{p}, {nu_last}]"
            ));
        }
        if span_v < q || span_v > nv_last {
            return Err(format!(
                "IgaSpace2D: span_v {span_v} out of valid range [{q}, {nv_last}]"
            ));
        }
        if self.knots_u[span_u] >= self.knots_u[span_u + 1] {
            return Err(format!(
                "IgaSpace2D: span_u {span_u} is empty (knot_u[{span_u}] == knot_u[{}] == {})",
                span_u + 1,
                self.knots_u[span_u]
            ));
        }
        if self.knots_v[span_v] >= self.knots_v[span_v + 1] {
            return Err(format!(
                "IgaSpace2D: span_v {span_v} is empty (knot_v[{span_v}] == knot_v[{}] == {})",
                span_v + 1,
                self.knots_v[span_v]
            ));
        }

        let active_u = self.basis_u.active_basis_indices(span_u);
        let active_v = self.basis_v.active_basis_indices(span_v);
        let mut active = Vec::with_capacity(active_u.len() * active_v.len());
        for j in active_v {
            for &i in &active_u {
                active.push(j * self.nu + i);
            }
        }
        Ok(active)
    }

    pub fn boundary_dofs(&self, side: IgaBoundary2D) -> Vec<usize> {
        match side {
            IgaBoundary2D::UMin => (0..self.nv).map(|j| j * self.nu).collect(),
            IgaBoundary2D::UMax => (0..self.nv).map(|j| j * self.nu + (self.nu - 1)).collect(),
            IgaBoundary2D::VMin => (0..self.nu).collect(),
            IgaBoundary2D::VMax => {
                let offset = (self.nv - 1) * self.nu;
                (0..self.nu).map(|i| offset + i).collect()
            }
        }
    }

    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    pub fn control_points(&self) -> &[[f64; 2]] {
        &self.ctrl_points
    }
}

// ─── 3D tri-variate IGA space ─────────────────────────────────────────────

/// Minimal single-patch tri-variate tensor-product IGA space in 3D.
///
/// Global control-point DOFs use standard lexicographic indexing:
/// `global = k * nu * nv + j * nu + i`, where `k` is the w-index,
/// `j` the v-index, and `i` the u-index.
#[derive(Clone, Debug)]
pub struct IgaSpace3D {
    basis_u: BsplineBasis,
    basis_v: BsplineBasis,
    basis_w: BsplineBasis,
    knots_u: Vec<f64>,
    knots_v: Vec<f64>,
    knots_w: Vec<f64>,
    nu: usize,
    nv: usize,
    nw: usize,
    weights: Option<Vec<f64>>,
    ctrl_points: Vec<[f64; 3]>,
}

impl IgaSpace3D {
    pub fn new(
        p: usize, q: usize, r: usize,
        knots_u: Vec<f64>, knots_v: Vec<f64>, knots_w: Vec<f64>,
        nu: usize, nv: usize, nw: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self, String> {
        let ctrl_points = default_unit_cube_ctrl_points(nu, nv, nw)?;
        Self::new_with_ctrl_points(p, q, r, knots_u, knots_v, knots_w, nu, nv, nw, weights, ctrl_points)
    }

    pub fn new_with_ctrl_points(
        p: usize, q: usize, r: usize,
        knots_u: Vec<f64>, knots_v: Vec<f64>, knots_w: Vec<f64>,
        nu: usize, nv: usize, nw: usize,
        weights: Option<Vec<f64>>,
        ctrl_points: Vec<[f64; 3]>,
    ) -> Result<Self, String> {
        if nu == 0 || nv == 0 || nw == 0 {
            return Err("IgaSpace3D: nu, nv, nw must be > 0".to_string());
        }
        let ku = KnotVector::new_clamped(knots_u.clone())?;
        let kv = KnotVector::new_clamped(knots_v.clone())?;
        let kw = KnotVector::new_clamped(knots_w.clone())?;
        let basis_u = BsplineBasis::new(p, ku)?;
        let basis_v = BsplineBasis::new(q, kv)?;
        let basis_w = BsplineBasis::new(r, kw)?;
        if basis_u.n_basis() != nu {
            return Err(format!("IgaSpace3D: nu ({nu}) does not match u basis count ({})", basis_u.n_basis()));
        }
        if basis_v.n_basis() != nv {
            return Err(format!("IgaSpace3D: nv ({nv}) does not match v basis count ({})", basis_v.n_basis()));
        }
        if basis_w.n_basis() != nw {
            return Err(format!("IgaSpace3D: nw ({nw}) does not match w basis count ({})", basis_w.n_basis()));
        }
        validate_weights(weights.as_deref(), nu * nv * nw, "IgaSpace3D")?;
        if ctrl_points.len() != nu * nv * nw {
            return Err(format!("IgaSpace3D: ctrl_points length ({}) does not match expected count ({})", ctrl_points.len(), nu * nv * nw));
        }
        Ok(Self { basis_u, basis_v, basis_w, knots_u, knots_v, knots_w, nu, nv, nw, weights, ctrl_points })
    }

    pub fn new_uniform_clamped(p: usize, q: usize, r: usize, nu: usize, nv: usize, nw: usize) -> Result<Self, String> {
        let knots_u = uniform_clamped_knots(p, nu)?;
        let knots_v = uniform_clamped_knots(q, nv)?;
        let knots_w = uniform_clamped_knots(r, nw)?;
        Self::new(p, q, r, knots_u, knots_v, knots_w, nu, nv, nw, None)
    }

    pub fn n_dofs(&self) -> usize { self.nu * self.nv * self.nw }
    pub fn degree_u(&self) -> usize { self.basis_u.degree() }
    pub fn degree_v(&self) -> usize { self.basis_v.degree() }
    pub fn degree_w(&self) -> usize { self.basis_w.degree() }
    pub fn knot_slice_u(&self) -> &[f64] { &self.knots_u }
    pub fn knot_slice_v(&self) -> &[f64] { &self.knots_v }
    pub fn knot_slice_w(&self) -> &[f64] { &self.knots_w }
    pub fn weights(&self) -> Option<&[f64]> { self.weights.as_deref() }
    pub fn control_points(&self) -> &[[f64; 3]] { &self.ctrl_points }

    pub fn non_empty_spans(&self) -> Vec<(usize, usize, usize)> {
        let p = self.basis_u.degree();
        let nu_last = self.basis_u.n_basis() - 1;
        let q = self.basis_v.degree();
        let nv_last = self.basis_v.n_basis() - 1;
        let r = self.basis_w.degree();
        let nw_last = self.basis_w.n_basis() - 1;
        let u_spans: Vec<usize> = (p..=nu_last).filter(|&s| self.knots_u[s] < self.knots_u[s + 1]).collect();
        let v_spans: Vec<usize> = (q..=nv_last).filter(|&s| self.knots_v[s] < self.knots_v[s + 1]).collect();
        let w_spans: Vec<usize> = (r..=nw_last).filter(|&s| self.knots_w[s] < self.knots_w[s + 1]).collect();
        let mut spans = Vec::with_capacity(u_spans.len() * v_spans.len() * w_spans.len());
        for sw in &w_spans { for sv in &v_spans { for su in &u_spans { spans.push((*su, *sv, *sw)); } } }
        spans
    }

    pub fn active_dofs_for_span(&self, span_u: usize, span_v: usize, span_w: usize) -> Result<Vec<usize>, String> {
        let p = self.basis_u.degree(); let nu_last = self.basis_u.n_basis() - 1;
        let q = self.basis_v.degree(); let nv_last = self.basis_v.n_basis() - 1;
        let r = self.basis_w.degree(); let nw_last = self.basis_w.n_basis() - 1;
        if span_u < p || span_u > nu_last { return Err(format!("span_u {span_u} out of range [{p},{nu_last}]")); }
        if span_v < q || span_v > nv_last { return Err(format!("span_v {span_v} out of range [{q},{nv_last}]")); }
        if span_w < r || span_w > nw_last { return Err(format!("span_w {span_w} out of range [{r},{nw_last}]")); }
        if self.knots_u[span_u] >= self.knots_u[span_u + 1] { return Err(format!("span_u {span_u} is empty")); }
        if self.knots_v[span_v] >= self.knots_v[span_v + 1] { return Err(format!("span_v {span_v} is empty")); }
        if self.knots_w[span_w] >= self.knots_w[span_w + 1] { return Err(format!("span_w {span_w} is empty")); }
        let au = self.basis_u.active_basis_indices(span_u);
        let av = self.basis_v.active_basis_indices(span_v);
        let aw = self.basis_w.active_basis_indices(span_w);
        let mut active = Vec::with_capacity(au.len() * av.len() * aw.len());
        for k in aw { for j in &av { for &i in &au { active.push(k * self.nu * self.nv + j * self.nu + i); } } }
        Ok(active)
    }
}

fn default_unit_cube_ctrl_points(nu: usize, nv: usize, nw: usize) -> Result<Vec<[f64; 3]>, String> {
    if nu < 2 || nv < 2 || nw < 2 {
        return Err(format!("IgaSpace3D: default control net requires nu,nv,nw >= 2, got ({nu},{nv},{nw})"));
    }
    let mut pts = Vec::with_capacity(nu * nv * nw);
    for k in 0..nw {
        for j in 0..nv {
            for i in 0..nu {
                let u = i as f64 / (nu - 1) as f64;
                let v = j as f64 / (nv - 1) as f64;
                let w = k as f64 / (nw - 1) as f64;
                pts.push([u, v, w]);
            }
        }
    }
    Ok(pts)
}

fn default_unit_square_ctrl_points(nu: usize, nv: usize) -> Result<Vec<[f64; 2]>, String> {
    if nu < 2 || nv < 2 {
        return Err(format!(
            "IgaSpace2D: default control net requires nu,nv >= 2, got nu={nu}, nv={nv}"
        ));
    }
    let mut pts = Vec::with_capacity(nu * nv);
    for j in 0..nv {
        for i in 0..nu {
            let u = i as f64 / (nu - 1) as f64;
            let v = j as f64 / (nv - 1) as f64;
            pts.push([u, v]);
        }
    }
    Ok(pts)
}

fn uniform_clamped_knots(degree: usize, n_ctrl: usize) -> Result<Vec<f64>, String> {
    if n_ctrl <= degree {
        return Err(format!(
            "uniform clamped knots require n_ctrl > degree, got n_ctrl={n_ctrl}, degree={degree}"
        ));
    }
    let n_spans = n_ctrl - degree;
    let mut knots = Vec::with_capacity(n_ctrl + degree + 1);
    knots.extend(std::iter::repeat_n(0.0, degree + 1));
    for i in 1..n_spans {
        knots.push((i as f64) / (n_spans as f64));
    }
    knots.extend(std::iter::repeat_n(1.0, degree + 1));
    Ok(knots)
}

fn validate_weights(weights: Option<&[f64]>, expected: usize, who: &str) -> Result<(), String> {
    let Some(w) = weights else {
        return Ok(());
    };
    if w.len() != expected {
        return Err(format!(
            "{who}: weights length ({}) does not match expected count ({expected})",
            w.len()
        ));
    }
    if w.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err(format!(
            "{who}: weights must be finite and strictly positive"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{IgaBoundary2D, IgaSpace1D, IgaSpace2D};
    use std::collections::HashSet;

    #[test]
    fn iga_1d_dof_count_equals_ctrl_count() {
        let space = IgaSpace1D::new_uniform_clamped(2, 5).unwrap();
        assert_eq!(space.n_dofs(), 5);
    }

    #[test]
    fn iga_1d_non_empty_spans_have_p_plus_1_active_dofs() {
        let space = IgaSpace1D::new_uniform_clamped(2, 6).unwrap();
        for span in space.non_empty_spans() {
            let active = space.active_dofs_for_span(span).unwrap();
            assert_eq!(active.len(), 3);
        }
    }

    #[test]
    fn iga_1d_boundary_dofs_are_first_and_last() {
        let space = IgaSpace1D::new_uniform_clamped(3, 8).unwrap();
        assert_eq!(space.boundary_dofs(), (0, 7));
    }

    #[test]
    fn iga_2d_dof_count_equals_nu_times_nv() {
        let space = IgaSpace2D::new_uniform_clamped(2, 1, 5, 4).unwrap();
        assert_eq!(space.n_dofs(), 20);
    }

    #[test]
    fn iga_2d_non_empty_spans_have_tensor_active_count() {
        let space = IgaSpace2D::new_uniform_clamped(2, 1, 5, 4).unwrap();
        for (span_u, span_v) in space.non_empty_spans() {
            let active = space.active_dofs_for_span(span_u, span_v).unwrap();
            assert_eq!(active.len(), 6);
        }
    }

    #[test]
    fn iga_2d_boundary_dof_sets_have_expected_sizes_and_shared_corners() {
        let space = IgaSpace2D::new_uniform_clamped(2, 1, 5, 4).unwrap();

        let u_min = space.boundary_dofs(IgaBoundary2D::UMin);
        let u_max = space.boundary_dofs(IgaBoundary2D::UMax);
        let v_min = space.boundary_dofs(IgaBoundary2D::VMin);
        let v_max = space.boundary_dofs(IgaBoundary2D::VMax);

        assert_eq!(u_min.len(), 4);
        assert_eq!(u_max.len(), 4);
        assert_eq!(v_min.len(), 5);
        assert_eq!(v_max.len(), 5);

        assert!(u_min.contains(&0));
        assert!(u_min.contains(&(3 * 5)));
        assert!(u_max.contains(&4));
        assert!(u_max.contains(&(3 * 5 + 4)));
        assert!(v_min.contains(&0));
        assert!(v_min.contains(&4));
        assert!(v_max.contains(&(3 * 5)));
        assert!(v_max.contains(&(3 * 5 + 4)));
    }

    #[test]
    fn iga_rejects_invalid_sizes_and_weights() {
        let err_1d = IgaSpace1D::new_uniform_clamped(3, 3).unwrap_err();
        assert!(err_1d.contains("n_ctrl > degree"));

        let err_2d = IgaSpace2D::new(
            2,
            1,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            3,
            2,
            Some(vec![1.0; 3]),
        )
        .unwrap_err();
        assert!(err_2d.contains("weights length"));
    }

    #[test]
    fn iga_1d_rejects_empty_and_out_of_range_spans() {
        let space = IgaSpace1D::new(2, vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0], 5, None)
            .unwrap();

        let out_of_range = space.active_dofs_for_span(1).unwrap_err();
        assert!(out_of_range.contains("out of valid range"));

        let empty_span = space.active_dofs_for_span(3).unwrap_err();
        assert!(empty_span.contains("is empty"));
    }

    #[test]
    fn iga_2d_rejects_empty_and_out_of_range_spans() {
        let space = IgaSpace2D::new(
            1,
            1,
            vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
            vec![0.0, 0.0, 0.5, 1.0, 1.0],
            4,
            3,
            None,
        )
        .unwrap();

        let out_of_range_u = space.active_dofs_for_span(0, 1).unwrap_err();
        assert!(out_of_range_u.contains("span_u 0 out of valid range"));

        let out_of_range_v = space.active_dofs_for_span(1, 9).unwrap_err();
        assert!(out_of_range_v.contains("out of valid range"));

        let empty_u = space.active_dofs_for_span(2, 1).unwrap_err();
        assert!(empty_u.contains("span_u 2 is empty"));
    }

    #[test]
    fn iga_constructors_reject_non_positive_or_non_finite_weights() {
        let bad_values = [0.0, -1.0, f64::NAN, f64::INFINITY];

        for bad in bad_values {
            let err_1d = IgaSpace1D::new(
                2,
                vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
                4,
                Some(vec![1.0, bad, 1.0, 1.0]),
            )
            .unwrap_err();
            assert!(err_1d.contains("weights must be finite and strictly positive"));
        }

        for bad in bad_values {
            let err_2d = IgaSpace2D::new(
                1,
                1,
                vec![0.0, 0.0, 0.5, 1.0, 1.0],
                vec![0.0, 0.0, 0.5, 1.0, 1.0],
                3,
                3,
                Some(vec![1.0, 1.0, bad, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            )
            .unwrap_err();
            assert!(err_2d.contains("weights must be finite and strictly positive"));
        }
    }

    #[test]
    fn iga_2d_active_dof_ordering_matches_row_major_and_boundaries() {
        let space = IgaSpace2D::new_uniform_clamped(2, 1, 5, 4).unwrap();
        let active = space.active_dofs_for_span(2, 1).unwrap();
        assert_eq!(active, vec![0, 1, 2, 5, 6, 7]);

        let u_min: HashSet<_> = space.boundary_dofs(IgaBoundary2D::UMin).into_iter().collect();
        let v_min: HashSet<_> = space.boundary_dofs(IgaBoundary2D::VMin).into_iter().collect();
        let v_max: HashSet<_> = space.boundary_dofs(IgaBoundary2D::VMax).into_iter().collect();

        assert!(u_min.contains(&active[0]));
        assert!(v_min.contains(&active[0]));
        assert!(!v_max.contains(&active[0]));
    }
}
