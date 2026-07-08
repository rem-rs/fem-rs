use std::collections::HashMap;
use fem_core::{NodeId, ElemId};
use crate::{element_type::ElementType, simplex::Mesh};

/// Compute element-wise Zienkiewicz–Zhu (ZZ) gradient-recovery error indicators.
///
/// Uses simple nodal averaging of element gradients to recover a smoothed
/// gradient `G(u)`, then computes
/// `η_K = ‖∇u_h|_K − G(u)|_K‖_{L²(K)}`
/// for each element `K`.
///
/// # Arguments
/// - `mesh`     — the mesh.
/// - `u`        — solution vector (one value per node, length = `n_nodes`).
///
/// # Returns
/// Vector of `η_K` for each element (length = `n_elems`).
pub fn zz_estimator(mesh: &Mesh<2>, u: &[f64]) -> Vec<f64> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    let is_quad = mesh.element_type_at(0) == ElementType::Quad4;

    // ── 1. Compute element gradients ──────────────────────────────────────────
    let mut elem_grads: Vec<[f64; 2]> = Vec::with_capacity(n_elems);

    if is_quad {
        // Quad4: bilinear gradient at centroid (ξ=0, η=0).
        // Bilinear shape functions: Nᵢ = ¼(1±ξ)(1±η)
        // Reference gradient at centroid: ∂u/∂ξ = ¼(-u₀+u₁+u₂-u₃), ∂u/∂η similarly.
        // Physical gradient via J^{-T} at centroid.
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            let c = |i: usize| mesh.coords_of(ns[i]);
            let uu = |i: usize| u[ns[i] as usize];

            let dxi  = 0.25 * (-uu(0) + uu(1) + uu(2) - uu(3));
            let deta = 0.25 * (-uu(0) - uu(1) + uu(2) + uu(3));

            let j00 = 0.25 * (-c(0)[0] + c(1)[0] + c(2)[0] - c(3)[0]);
            let j01 = 0.25 * (-c(0)[0] - c(1)[0] + c(2)[0] + c(3)[0]);
            let j10 = 0.25 * (-c(0)[1] + c(1)[1] + c(2)[1] - c(3)[1]);
            let j11 = 0.25 * (-c(0)[1] - c(1)[1] + c(2)[1] + c(3)[1]);
            let det_j = j00 * j11 - j01 * j10;

            // J^{-T} = (1/det) * [[j11, -j10], [-j01, j00]]
            let gx = ( j11 * dxi - j10 * deta) / det_j;
            let gy = (-j01 * dxi + j00 * deta) / det_j;
            elem_grads.push([gx, gy]);
        }
    } else {
        // Tri3: ∇u is constant over each element (P1).
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];
            let [x0, y0] = mesh.coords_of(n0);
            let [x1, y1] = mesh.coords_of(n1);
            let [x2, y2] = mesh.coords_of(n2);
            let u0 = u[n0 as usize]; let u1 = u[n1 as usize]; let u2 = u[n2 as usize];

            // Jacobian of mapping from reference triangle to physical:
            // J = [[x1-x0, x2-x0], [y1-y0, y2-y0]]
            let j00 = x1 - x0; let j01 = x2 - x0;
            let j10 = y1 - y0; let j11 = y2 - y0;
            let det = j00 * j11 - j01 * j10;

            // Reference gradients of Lagrange basis: ∇ψ₀ = (-1,-1), ∇ψ₁ = (1,0), ∇ψ₂ = (0,1)
            // Physical grad = J^{-T} * ref_grad
            // J^{-T} = (1/det) * [[j11, -j10], [-j01, j00]]
            let g_ref = [
                [-1.0_f64, -1.0],
                [ 1.0,  0.0],
                [ 0.0,  1.0],
            ];
            let uh = [u0, u1, u2];
            let mut gx = 0.0_f64; let mut gy = 0.0_f64;
            for k in 0..3 {
                let gpx = ( j11 * g_ref[k][0] - j10 * g_ref[k][1]) / det;
                let gpy = (-j01 * g_ref[k][0] + j00 * g_ref[k][1]) / det;
                gx += uh[k] * gpx;
                gy += uh[k] * gpy;
            }
            elem_grads.push([gx, gy]);
        }
    }

    // ── 2. Nodal gradient recovery (simple averaging) ─────────────────────────
    let mut nodal_grad = vec![[0.0_f64; 2]; n_nodes];
    let mut nodal_count = vec![0usize; n_nodes];

    for (e, &grad) in elem_grads.iter().enumerate() {
        let ns = mesh.elem_nodes(e as ElemId);
        for &n in ns {
            nodal_grad[n as usize][0] += grad[0];
            nodal_grad[n as usize][1] += grad[1];
            nodal_count[n as usize] += 1;
        }
    }
    for n in 0..n_nodes {
        let c = nodal_count[n] as f64;
        if c > 0.0 {
            nodal_grad[n][0] /= c;
            nodal_grad[n][1] /= c;
        }
    }

    // ── 3. Element error indicator ────────────────────────────────────────────
    let mut eta = Vec::with_capacity(n_elems);

    if is_quad {
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            let c = |i: usize| mesh.coords_of(ns[i]);

            // Shoelace formula for quadrilateral area.
            let area = 0.5 * (
                c(0)[0]*c(1)[1] + c(1)[0]*c(2)[1] + c(2)[0]*c(3)[1] + c(3)[0]*c(0)[1]
              - c(1)[0]*c(0)[1] - c(2)[0]*c(1)[1] - c(3)[0]*c(2)[1] - c(0)[0]*c(3)[1]
            ).abs();

            let grx = ns.iter().map(|&n| nodal_grad[n as usize][0]).sum::<f64>() / 4.0;
            let gry = ns.iter().map(|&n| nodal_grad[n as usize][1]).sum::<f64>() / 4.0;
            let eg = &elem_grads[e as usize];
            let dx = eg[0] - grx;
            let dy = eg[1] - gry;
            eta.push(area.sqrt() * (dx*dx + dy*dy).sqrt());
        }
    } else {
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            let [x0, y0] = mesh.coords_of(ns[0]);
            let [x1, y1] = mesh.coords_of(ns[1]);
            let [x2, y2] = mesh.coords_of(ns[2]);
            let area = 0.5 * ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)).abs();

            // Recovered gradient at centroid = average of nodal recovered gradients
            let grx: f64 = ns.iter().map(|&n| nodal_grad[n as usize][0]).sum::<f64>() / 3.0;
            let gry: f64 = ns.iter().map(|&n| nodal_grad[n as usize][1]).sum::<f64>() / 3.0;
            let eg = &elem_grads[e as usize];

            let dx = eg[0] - grx;
            let dy = eg[1] - gry;
            // η_K = ‖(∇u_h − G(u_h))‖ * sqrt(area)
            eta.push(area.sqrt() * (dx*dx + dy*dy).sqrt());
        }
    }
    eta
}

/// 3-D ZZ error indicator for Tet4 meshes.
pub fn zz_estimator_3d(mesh: &Mesh<3>, u: &[f64]) -> Vec<f64> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    let mut elem_grads: Vec<[f64; 3]> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        let uu = |i| u[ns[i] as usize];
        let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
        let det = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
        let idet = if det.abs() > 1e-30 { 1.0/det } else { 0.0 };
        // J^{-T} via adjugate: adj(J)^T / det
        let jit = |r: usize, c: usize| -> f64 {
            let a = (r+1)%3; let b = (r+2)%3;
            let d = (c+1)%3; let e = (c+2)%3;
            (j[a][d]*j[b][e] - j[a][e]*j[b][d]) * idet
        };
        let gref = [[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        let uh = [uu(0),uu(1),uu(2),uu(3)];
        let mut g = [0.0_f64;3];
        for k in 0..4 { for i in 0..3 { for j in 0..3 { g[i] += uh[k] * jit(j,i) * gref[k][j]; } } }
        elem_grads.push(g);
    }
    let mut ng = vec![[0.0_f64;3]; n_nodes]; let mut nc = vec![0usize; n_nodes];
    for (e, &g) in elem_grads.iter().enumerate() {
        for &n in mesh.elem_nodes(e as ElemId) {
            for d in 0..3 { ng[n as usize][d] += g[d]; } nc[n as usize] += 1;
        }
    }
    for n in 0..n_nodes { let c = nc[n] as f64; if c>0.0 { for d in 0..3 { ng[n][d] /= c; } } }
    let mut eta = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
        let vol = (j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                 - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                 + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs() / 6.0;
        let gr = [ns.iter().map(|&n| ng[n as usize][0]).sum::<f64>()/4.0,
                  ns.iter().map(|&n| ng[n as usize][1]).sum::<f64>()/4.0,
                  ns.iter().map(|&n| ng[n as usize][2]).sum::<f64>()/4.0];
        let eg = elem_grads[e as usize];
        let d = [(eg[0]-gr[0]), (eg[1]-gr[1]), (eg[2]-gr[2])];
        eta.push(vol.powf(1.0/3.0) * (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]).sqrt());
    }
    eta
}

/// 3-D Kelly (face-jump) error indicator for Tet4 meshes.
pub fn kelly_estimator_3d(mesh: &Mesh<3>, u: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();

    // Face adjacency
    let fkey = |a: u32, b: u32, c: u32| -> (u32, u32, u32) {
        let mut v = [a, b, c]; v.sort_unstable(); (v[0], v[1], v[2])
    };
    use std::collections::HashMap;
    let mut fem: HashMap<(u32,u32,u32), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a,b,c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
            fem.entry(fkey(a,b,c)).or_default().push(e);
        }
    }

    // Scale ZZ gradients back to raw form for Kelly (undo volume scaling in ZZ)
    let mut raw_grads: Vec<[f64;3]> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
        let det = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
        let idet = if det.abs() > 1e-30 { 1.0/det } else { 0.0 };
        let jit = |r: usize, c: usize| -> f64 {
            let a=(r+1)%3; let b=(r+2)%3; let d=(c+1)%3; let e=(c+2)%3;
            (j[a][d]*j[b][e] - j[a][e]*j[b][d]) * idet
        };
        let gref = [[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        let uu = |i| u[ns[i] as usize]; let uh = [uu(0),uu(1),uu(2),uu(3)];
        let mut g = [0.0_f64;3];
        for k in 0..4 { for i in 0..3 { for jj in 0..3 { g[i] += uh[k]*jit(jj,i)*gref[k][jj]; } } }
        raw_grads.push(g);
    }
    let elem_grads = raw_grads;

    let mut eta_sq = vec![0.0_f64; n_elems];
    for (fk, elems) in fem.iter().filter(|(_,v)| v.len() == 2) {
        let e0 = elems[0] as usize;
        let e1 = elems[1] as usize;
        let (na, nb, nc) = *fk;
        let ca = mesh.coords_of(na); let cb = mesh.coords_of(nb); let cc = mesh.coords_of(nc);
        let ex = cb[0]-ca[0]; let ey = cb[1]-ca[1]; let ez = cb[2]-ca[2];
        let fx = cc[0]-ca[0]; let fy = cc[1]-ca[1]; let fz = cc[2]-ca[2];
        let nx = ey*fz - ez*fy; let ny = ez*fx - ex*fz; let nz = ex*fy - ey*fx;
        let face_area = 0.5 * (nx*nx+ny*ny+nz*nz).sqrt();
        if face_area < 1e-30 { continue; }
        let inv = 1.0/(nx*nx+ny*ny+nz*nz).sqrt();
        let jump = (elem_grads[e0][0]-elem_grads[e1][0])*nx*inv
                 + (elem_grads[e0][1]-elem_grads[e1][1])*ny*inv
                 + (elem_grads[e0][2]-elem_grads[e1][2])*nz*inv;
        let contrib = face_area * jump * jump;
        eta_sq[e0] += contrib; eta_sq[e1] += contrib;
    }
    eta_sq.iter().map(|v| v.sqrt()).collect()
}

/// Compute element-wise Kelly (face-jump) error indicators for 2-D Tri3 or Quad4.
pub fn kelly_estimator(mesh: &Mesh<2>, u: &[f64]) -> Vec<f64> {
    use std::collections::HashMap;
    let n_elems = mesh.n_elems();
    let is_quad = mesh.element_type_at(0) == ElementType::Quad4;

    // ── 1. Element gradients at centroid ───────────────────────────────────────
    let mut elem_grads: Vec<[f64; 2]> = Vec::with_capacity(n_elems);
    if is_quad {
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            let c = |i: usize| mesh.coords_of(ns[i]);
            let uu = |i: usize| u[ns[i] as usize];
            let dxi  = 0.25 * (-uu(0) + uu(1) + uu(2) - uu(3));
            let deta = 0.25 * (-uu(0) - uu(1) + uu(2) + uu(3));
            let j00 = 0.25 * (-c(0)[0] + c(1)[0] + c(2)[0] - c(3)[0]);
            let j01 = 0.25 * (-c(0)[0] - c(1)[0] + c(2)[0] + c(3)[0]);
            let j10 = 0.25 * (-c(0)[1] + c(1)[1] + c(2)[1] - c(3)[1]);
            let j11 = 0.25 * (-c(0)[1] - c(1)[1] + c(2)[1] + c(3)[1]);
            let det_j = j00 * j11 - j01 * j10;
            let gx = ( j11 * dxi - j10 * deta) / det_j;
            let gy = (-j01 * dxi + j00 * deta) / det_j;
            elem_grads.push([gx, gy]);
        }
    } else {
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            let c = |i| mesh.coords_of(ns[i]); let uu = |i| u[ns[i] as usize];
            let j00 = c(1)[0]-c(0)[0]; let j01 = c(2)[0]-c(0)[0];
            let j10 = c(1)[1]-c(0)[1]; let j11 = c(2)[1]-c(0)[1];
            let det = j00*j11 - j01*j10;
            let gref = [[-1.0,-1.0],[1.0,0.0],[0.0,1.0]];
            let uh = [uu(0),uu(1),uu(2)];
            let mut gx=0.0; let mut gy=0.0;
            for k in 0..3 {
                let gpx = ( j11*gref[k][0] - j10*gref[k][1])/det;
                let gpy = (-j01*gref[k][0] + j00*gref[k][1])/det;
                gx += uh[k]*gpx; gy += uh[k]*gpy;
            }
            elem_grads.push([gx,gy]);
        }
    }

    // ── 2. Edge jumps ─────────────────────────────────────────────────────────
    type Edge = (NodeId,NodeId);
    fn ek(a: NodeId,b: NodeId) -> Edge { if a<b {(a,b)} else {(b,a)} }
    let mut ee: HashMap<Edge,Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if is_quad {
            for &(a,b) in &[(ns[0],ns[1]),(ns[1],ns[2]),(ns[2],ns[3]),(ns[3],ns[0])] {
                ee.entry(ek(a,b)).or_default().push(e);
            }
        } else {
            for &(a,b) in &[(ns[0],ns[1]),(ns[1],ns[2]),(ns[0],ns[2])] {
                ee.entry(ek(a,b)).or_default().push(e);
            }
        }
    }
    let mut eta_sq = vec![0.0_f64; n_elems];
    for (&(na,nb), elems) in &ee {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize; let e1 = elems[1] as usize;
        let ca = mesh.coords_of(na); let cb = mesh.coords_of(nb);
        let h = ((cb[0]-ca[0]).powi(2)+(cb[1]-ca[1]).powi(2)).sqrt();
        if h < 1e-30 { continue; }
        let nx = -(cb[1]-ca[1])/h; let ny = (cb[0]-ca[0])/h;
        let jump = (elem_grads[e0][0]-elem_grads[e1][0])*nx
                 + (elem_grads[e0][1]-elem_grads[e1][1])*ny;
        eta_sq[e0] += h*jump*jump; eta_sq[e1] += h*jump*jump;
    }
    eta_sq.iter().map(|v| v.sqrt()).collect()
}

/// Mark elements for refinement using Dörfler bulk criterion.
pub fn dorfler_mark(eta: &[f64], theta: f64) -> Vec<ElemId> {
    let total: f64 = eta.iter().sum();
    let threshold = theta.clamp(0.0,1.0) * total;
    let mut indices: Vec<usize> = (0..eta.len()).collect();
    indices.sort_unstable_by(|&a,&b| eta[b].partial_cmp(&eta[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut marked = Vec::new(); let mut acc = 0.0_f64;
    for &i in &indices { if acc >= threshold { break; } acc += eta[i]; marked.push(i as ElemId); }
    marked.sort_unstable(); marked
}

/// Mark low-error elements for derefinement.
pub fn mark_for_derefinement(eta: &[f64], theta: f64) -> Vec<ElemId> {
    if eta.is_empty() { return Vec::new(); }
    let max_eta = eta.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = theta.clamp(0.0,1.0) * max_eta;
    eta.iter().enumerate().filter(|(_,&e)| e <= cutoff).map(|(i,_)| i as ElemId).collect()
}

/// Compute element-wise DWR (Dual Weighted Residual) error indicators for
/// 2-D Tri3 meshes and the Poisson equation `−∇·(κ∇u) = f`.
///
/// # DWR principle
///
/// For a goal functional `J(u)`, the error `J(u) − J(u_h)` can be estimated
/// element-wise using the primal residual weighted by the dual solution:
///
/// ```text
/// η_K ≈ |∫_K f · ω dΩ  +  ½ ∫_{∂K} j · ω dS|
/// ```
///
/// where `ω = z_h − I₀(z_h)` is the dual solution fluctuation (cell-wise
/// deviation from the cell-constant average), and `j = [κ·∂u_h/∂n]` is the
/// jump in the normal flux across interior element edges.
///
/// For P1 elements on Tri3 meshes with constant κ:
/// - The interior residual `r_K = f + ∇·(κ∇u_h)` reduces to `f` (P1 gradient
///   is element-wise constant, so `∇·∇u_h = 0`).
/// - The edge jump captures the discontinuity of the recovered flux.
///
/// # Arguments
/// - `mesh` — triangular 2-D mesh (Tri3).
/// - `u`    — primal solution (nodal values, length = `n_nodes`).
/// - `z`    — dual solution (nodal values, length = `n_nodes`).
/// - `f`    — source term evaluated at each node (length = `n_nodes`).
///
/// # Returns
/// Vector of `η_K` for each element (length = `n_elems`).
pub fn dwr_estimator(mesh: &Mesh<2>, u: &[f64], z: &[f64], f: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    use std::collections::HashMap;

    // ── 1. Element gradients (primal) ──────────────────────────────────────────
    let mut elem_grad: Vec<[f64; 2]> = Vec::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);
        let u0 = u[ns[0] as usize]; let u1 = u[ns[1] as usize]; let u2 = u[ns[2] as usize];

        let j00 = x1 - x0; let j01 = x2 - x0;
        let j10 = y1 - y0; let j11 = y2 - y0;
        let det = j00 * j11 - j01 * j10;
        let inv_det = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };

        let g_ref = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
        let uh = [u0, u1, u2];
        let mut gx = 0.0; let mut gy = 0.0;
        for k in 0..3 {
            let gpx = ( j11 * g_ref[k][0] - j10 * g_ref[k][1]) * inv_det;
            let gpy = (-j01 * g_ref[k][0] + j00 * g_ref[k][1]) * inv_det;
            gx += uh[k] * gpx;
            gy += uh[k] * gpy;
        }
        elem_grad.push([gx, gy]);
    }

    // ── 2. Edge adjacency ──────────────────────────────────────────────────────
    type Edge = (NodeId, NodeId);
    fn edge_key(a: NodeId, b: NodeId) -> Edge {
        if a < b { (a, b) } else { (b, a) }
    }

    let mut edge_elems: HashMap<Edge, Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let edges = [
            edge_key(ns[0], ns[1]),
            edge_key(ns[1], ns[2]),
            edge_key(ns[0], ns[2]),
        ];
        for ek in &edges {
            edge_elems.entry(*ek).or_default().push(e);
        }
    }

    // ── 3. Element dual fluctuation ω_K = z_h − avg(z_h) ──────────────────────
    let mut elem_omega: Vec<f64> = Vec::with_capacity(n_elems);
    let mut elem_centroid_f: Vec<f64> = Vec::with_capacity(n_elems);
    let mut elem_area: Vec<f64> = Vec::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let z_avg = (z[ns[0] as usize] + z[ns[1] as usize] + z[ns[2] as usize]) / 3.0;
        let f_avg = (f[ns[0] as usize] + f[ns[1] as usize] + f[ns[2] as usize]) / 3.0;
        elem_omega.push(z_avg);
        elem_centroid_f.push(f_avg);

        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);
        let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs();
        elem_area.push(area);
    }

    // ── 4. Assemble DWR indicator per element ─────────────────────────────────
    let mut eta = vec![0.0_f64; n_elems];

    // 4a. Interior contribution: ∫_K f · ω ≈ f_centroid · ω · |K|
    for e in 0..n_elems {
        eta[e] += (elem_centroid_f[e] * elem_omega[e]).abs() * elem_area[e];
    }

    // 4b. Edge jump contribution: ½ ∫_E [[κ·∂u_h/∂n]] · ω dS
    for (&(na, _nb), elems) in &edge_elems {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize;
        let e1 = elems[1] as usize;

        let [xa, ya] = mesh.coords_of(na);
        let [xb, yb] = mesh.coords_of(_nb);
        let h_edge = ((xb - xa).powi(2) + (yb - ya).powi(2)).sqrt();
        if h_edge < 1e-30 { continue; }

        let nx = -(yb - ya) / h_edge;
        let ny =  (xb - xa) / h_edge;

        let j0 = elem_grad[e0][0] * nx + elem_grad[e0][1] * ny;
        let j1 = elem_grad[e1][0] * nx + elem_grad[e1][1] * ny;
        let jump = (j0 - j1).abs();
        if jump < 1e-30 { continue; }

        let w_mid = (elem_omega[e0] + elem_omega[e1]) * 0.5;
        let edge_contrib = 0.5 * h_edge * jump * w_mid.abs();
        eta[e0] += edge_contrib;
        eta[e1] += edge_contrib;
    }

    eta
}

/// Element-wise residual error indicator for Poisson on Tri3 meshes.
///
/// Standard residual estimator (Verfürth, 1996):
///
/// ```text
/// η_K² = h_K² · ‖f‖²_L²(K)  +  ½ Σ_{E⊂∂K} h_E · ‖[[∇u_h·n]]‖²_L²(E)
/// ```
///
/// For P1 elements the interior residual reduces to `f` since ∇·∇u_h = 0
/// element-wise. The edge jump term is identical to the Kelly indicator.
///
/// # Arguments
/// - `mesh` — Tri3 mesh.
/// - `u`    — P1 solution at nodes.
/// - `f`    — source term at nodes (or element centroid values).
///
/// # Returns
/// Element-wise error indicators η_K.
pub fn residual_estimator(mesh: &Mesh<2>, u: &[f64], f: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    let elem_grads: Vec<[f64; 2]> = {
        let mut g = Vec::with_capacity(n_elems);
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            let [x0, y0] = mesh.coords_of(ns[0]);
            let [x1, y1] = mesh.coords_of(ns[1]);
            let [x2, y2] = mesh.coords_of(ns[2]);
            let u0 = u[ns[0] as usize]; let u1 = u[ns[1] as usize]; let u2 = u[ns[2] as usize];
            let j00 = x1 - x0; let j01 = x2 - x0;
            let j10 = y1 - y0; let j11 = y2 - y0;
            let det = j00 * j11 - j01 * j10;
            let inv_det = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };
            let gref = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
            let mut gx = 0.0; let mut gy = 0.0;
            for k in 0..3 {
                let gpx = ( j11 * gref[k][0] - j10 * gref[k][1]) * inv_det;
                let gpy = (-j01 * gref[k][0] + j00 * gref[k][1]) * inv_det;
                gx += [u0, u1, u2][k] * gpx;
                gy += [u0, u1, u2][k] * gpy;
            }
            g.push([gx, gy]);
        }
        g
    };

    // Element diameters (h_K = sqrt(2 * area) for triangles)
    let mut elem_h: Vec<f64> = Vec::with_capacity(n_elems);
    let mut elem_area: Vec<f64> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);
        let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs();
        let h = (2.0 * area).sqrt();
        elem_h.push(h);
        elem_area.push(area);
    }

    let mut ee: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &[(ns[0], ns[1]), (ns[1], ns[2]), (ns[0], ns[2])] {
            let k = if a < b { (a, b) } else { (b, a) };
            ee.entry(k).or_default().push(e);
        }
    }

    let mut eta_sq = vec![0.0_f64; n_elems];

    for e in 0..n_elems {
        let ns = mesh.elem_nodes(e as ElemId);
        let favg = (f[ns[0] as usize] + f[ns[1] as usize] + f[ns[2] as usize]) / 3.0;
        eta_sq[e] += elem_h[e] * elem_h[e] * elem_area[e] * favg * favg;
    }

    for (&(na, nb), elems) in &ee {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize;
        let e1 = elems[1] as usize;
        let [xa, ya] = mesh.coords_of(na);
        let [xb, yb] = mesh.coords_of(nb);
        let h_edge = ((xb - xa).powi(2) + (yb - ya).powi(2)).sqrt();
        if h_edge < 1e-30 { continue; }
        let nx = -(yb - ya) / h_edge;
        let ny =  (xb - xa) / h_edge;
        let jump = (elem_grads[e0][0] - elem_grads[e1][0]) * nx
                 + (elem_grads[e0][1] - elem_grads[e1][1]) * ny;
        let contrib = 0.5 * h_edge * jump * jump;
        eta_sq[e0] += contrib;
        eta_sq[e1] += contrib;
    }

    eta_sq.iter().map(|v| v.sqrt()).collect()
}

/// Element-wise residual error indicator for Poisson on Tet4 meshes.
///
/// ```text
/// η_K² = h_K² · ‖f‖²_L²(K)  +  ½ Σ_{F⊂∂K} h_F · ‖[[∇u_h·n]]‖²_L²(F)
/// ```
///
/// For P1 elements the interior residual vanishes, leaving the face-jump term.
/// The face jump contribution is computed similarly to the 3-D Kelly estimator
/// but weighted by the face diameter rather than the face area.
pub fn residual_estimator_3d(mesh: &Mesh<3>, u: &[f64], f: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    let mut elem_grads: Vec<[f64; 3]> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        let uu = |i| u[ns[i] as usize];
        let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
        let det = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
        let idet = if det.abs() > 1e-30 { 1.0/det } else { 0.0 };
        let jit = |r: usize, c: usize| -> f64 {
            let a = (r+1)%3; let b = (r+2)%3;
            let d = (c+1)%3; let e = (c+2)%3;
            (j[a][d]*j[b][e] - j[a][e]*j[b][d]) * idet
        };
        let gref = [[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        let uh = [uu(0),uu(1),uu(2),uu(3)];
        let mut g = [0.0_f64;3];
        for k in 0..4 { for i in 0..3 { for jj in 0..3 { g[i] += uh[k] * jit(jj,i) * gref[k][jj]; } } }
        elem_grads.push(g);
    }

    let mut elem_vol: Vec<f64> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
        let vol = (j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                 - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                 + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs() / 6.0;
        elem_vol.push(vol);
    }

    type Face3 = (NodeId, NodeId, NodeId);
    let fkey = |a: NodeId, b: NodeId, c: NodeId| -> Face3 {
        let mut v = [a, b, c]; v.sort_unstable(); (v[0], v[1], v[2])
    };
    let mut fem: HashMap<Face3, Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a,b,c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
            fem.entry(fkey(a,b,c)).or_default().push(e);
        }
    }

    let mut eta_sq = vec![0.0_f64; n_elems];

    for e in 0..n_elems {
        let ns = mesh.elem_nodes(e as ElemId);
        let favg = (f[ns[0] as usize] + f[ns[1] as usize] + f[ns[2] as usize] + f[ns[3] as usize]) / 4.0;
        let h2 = elem_vol[e].powf(2.0 / 3.0);
        eta_sq[e] += h2 * elem_vol[e] * favg * favg;
    }

    for (fk, elems) in &fem {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize;
        let e1 = elems[1] as usize;
        let (na, nb, nc) = *fk;
        let ca = mesh.coords_of(na); let cb = mesh.coords_of(nb); let cc = mesh.coords_of(nc);
        let ex = cb[0]-ca[0]; let ey = cb[1]-ca[1]; let ez = cb[2]-ca[2];
        let fx = cc[0]-ca[0]; let fy = cc[1]-ca[1]; let fz = cc[2]-ca[2];
        let nx = ey*fz - ez*fy; let ny = ez*fx - ex*fz; let nz = ex*fy - ey*fx;
        let face_area = 0.5 * (nx*nx+ny*ny+nz*nz).sqrt();
        if face_area < 1e-30 { continue; }
        let inv = 1.0/(nx*nx+ny*ny+nz*nz).sqrt();
        let jump = (elem_grads[e0][0]-elem_grads[e1][0])*nx*inv
                 + (elem_grads[e0][1]-elem_grads[e1][1])*ny*inv
                 + (elem_grads[e0][2]-elem_grads[e1][2])*nz*inv;
        let h_face = (2.0 * face_area).sqrt();
        let contrib = 0.5 * h_face * jump * jump;
        eta_sq[e0] += contrib;
        eta_sq[e1] += contrib;
    }

    eta_sq.iter().map(|v| v.sqrt()).collect()
}

/// Mark elements for p-refinement based on a smoothness indicator.
///
/// Elements with high estimated curvature (ratio of recovered gradient
/// variation to element size above a threshold) are marked for order
/// elevation.  The smoothness indicator is derived from the ZZ error
/// estimator: elements where the local error exceeds `theta * max(eta)`
/// are candidates for p-refinement rather than h-refinement.
///
/// # Returns
/// Indices of elements recommended for p-refinement.
pub fn mark_for_p_refinement(eta: &[f64], theta: f64) -> Vec<ElemId> {
    if eta.is_empty() { return Vec::new(); }
    let max_eta = eta.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = theta.clamp(0.0, 1.0) * max_eta;
    eta.iter().enumerate()
        .filter(|(_, &e)| e >= cutoff)
        .map(|(i, _)| i as ElemId)
        .collect()
}

/// Element-wise ZZ estimator for 3-D meshes of any linear element type.
///
/// Computes the gradient recovery-based error indicator:
/// η_K = ‖∇u_h|_K − G(u_h)|_K‖ · |K|^{1/3}
/// where G is the nodal-averaged recovered gradient.
///
/// Supports Tet4, Hex8, Prism6, Pyramid5.
pub fn zz_estimator_3d_general(mesh: &Mesh<3>, u: &[f64]) -> Vec<f64> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    let _dim = 3usize;

    let mut elem_grads: Vec<[f64;3]> = Vec::with_capacity(n_elems);
    let mut elem_vols: Vec<f64> = Vec::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let npe = ns.len();
        let c = |i| mesh.coords_of(ns[i]);

        let (jac, vol) = match npe {
            4 => {
                let j = [[c(1)[0]-c(0)[0],c(2)[0]-c(0)[0],c(3)[0]-c(0)[0]],
                         [c(1)[1]-c(0)[1],c(2)[1]-c(0)[1],c(3)[1]-c(0)[1]],
                         [c(1)[2]-c(0)[2],c(2)[2]-c(0)[2],c(3)[2]-c(0)[2]]];
                let d = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
                (j, d.abs()/6.0)
            }
            8 => {
                let j = [[0.125*( -c(0)[0]+c(1)[0]+c(2)[0]-c(3)[0]-c(4)[0]+c(5)[0]+c(6)[0]-c(7)[0]),
                          0.125*( -c(0)[0]-c(1)[0]+c(2)[0]+c(3)[0]-c(4)[0]-c(5)[0]+c(6)[0]+c(7)[0]),
                          0.125*( -c(0)[0]-c(1)[0]-c(2)[0]-c(3)[0]+c(4)[0]+c(5)[0]+c(6)[0]+c(7)[0])],
                         [0.125*( -c(0)[1]+c(1)[1]+c(2)[1]-c(3)[1]-c(4)[1]+c(5)[1]+c(6)[1]-c(7)[1]),
                          0.125*( -c(0)[1]-c(1)[1]+c(2)[1]+c(3)[1]-c(4)[1]-c(5)[1]+c(6)[1]+c(7)[1]),
                          0.125*( -c(0)[1]-c(1)[1]-c(2)[1]-c(3)[1]+c(4)[1]+c(5)[1]+c(6)[1]+c(7)[1])],
                         [0.125*( -c(0)[2]+c(1)[2]+c(2)[2]-c(3)[2]-c(4)[2]+c(5)[2]+c(6)[2]-c(7)[2]),
                          0.125*( -c(0)[2]-c(1)[2]+c(2)[2]+c(3)[2]-c(4)[2]-c(5)[2]+c(6)[2]+c(7)[2]),
                          0.125*( -c(0)[2]-c(1)[2]-c(2)[2]-c(3)[2]+c(4)[2]+c(5)[2]+c(6)[2]+c(7)[2])]];
                let d = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
                (j, d.abs())
            }
            6 => {
                let j = [[ (c(1)[0]-c(0)[0])/2.0, (c(2)[0]-c(0)[0])/2.0, (c(3)[0]-c(0)[0])/2.0],
                         [ (c(1)[1]-c(0)[1])/2.0, (c(2)[1]-c(0)[1])/2.0, (c(3)[1]-c(0)[1])/2.0],
                         [ (c(1)[2]-c(0)[2])/2.0, (c(2)[2]-c(0)[2])/2.0, (c(3)[2]-c(0)[2])/2.0]];
                let d = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
                (j, d.abs()/2.0)
            }
            5 => {
                let v = |i| c(i);
                let t1 = (v(0),v(1),v(2),v(4)); let t2 = (v(2),v(3),v(0),v(4));
                let d1 = |a:[f64;3],b:[f64;3],c:[f64;3],d:[f64;3]| -> f64 {
                    (b[0]-a[0])*(c[1]-a[1])*(d[2]-a[2])+(b[1]-a[1])*(c[2]-a[2])*(d[0]-a[0])+(b[2]-a[2])*(c[0]-a[0])*(d[1]-a[1])
                    -(b[2]-a[2])*(c[1]-a[1])*(d[0]-a[0])-(b[1]-a[1])*(c[0]-a[0])*(d[2]-a[2])-(b[0]-a[0])*(c[2]-a[2])*(d[1]-a[1])
                };
                let vol = (d1(t1.0,t1.1,t1.2,t1.3).abs()+d1(t2.0,t2.1,t2.2,t2.3).abs())/6.0;
                let j = [[c(1)[0]-c(0)[0],c(2)[0]-c(0)[0],c(4)[0]-c(0)[0]],
                         [c(1)[1]-c(0)[1],c(2)[1]-c(0)[1],c(4)[1]-c(0)[1]],
                         [c(1)[2]-c(0)[2],c(2)[2]-c(0)[2],c(4)[2]-c(0)[2]]];
                (j, vol)
            }
            _ => panic!("zz_estimator_3d_general: unsupported npe={}", npe),
        };

        let det = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])
                - jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])
                + jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
        let idet = if det.abs() > 1e-30 { 1.0/det } else { 0.0 };
        let jit = |r: usize, c: usize| -> f64 {
            let a=(r+1)%3;let b=(r+2)%3;let d=(c+1)%3;let e=(c+2)%3;
            (jac[a][d]*jac[b][e]-jac[a][e]*jac[b][d])*idet
        };

        let gref: Vec<[f64;3]> = match npe {
            4 => vec![[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
            8 => vec![[-0.125,-0.125,-0.125],[0.125,-0.125,-0.125],[0.125,0.125,-0.125],[-0.125,0.125,-0.125],
                      [-0.125,-0.125,0.125],[0.125,-0.125,0.125],[0.125,0.125,0.125],[-0.125,0.125,0.125]],
            6 => vec![[-0.5,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],
                      [-0.5,0.0,0.5],[0.5,0.0,0.5],[0.0,0.5,0.5]],
            5 => vec![[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[-1.0,0.0,0.0],[0.0,0.0,1.0]],
            _ => unreachable!(),
        };

        let uh: Vec<f64> = ns.iter().map(|&n| u[n as usize]).collect();
        let mut g = [0.0_f64;3];
        for k in 0..npe {
            for i in 0..3 {
                for j in 0..3 {
                    g[i] += uh[k] * jit(j,i) * gref[k][j];
                }
            }
        }
        elem_grads.push(g);
        elem_vols.push(vol);
    }

    let mut ng = vec![[0.0_f64;3]; n_nodes];
    let mut nc = vec![0usize; n_nodes];
    for (e, &g) in elem_grads.iter().enumerate() {
        for &n in mesh.elem_nodes(e as ElemId) {
            for d in 0..3 { ng[n as usize][d] += g[d]; }
            nc[n as usize] += 1;
        }
    }
    for n in 0..n_nodes {
        let c = nc[n] as f64;
        if c > 0.0 { for d in 0..3 { ng[n][d] /= c; } }
    }

    let mut eta = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let gr = [ns.iter().map(|&n| ng[n as usize][0]).sum::<f64>() / ns.len() as f64,
                  ns.iter().map(|&n| ng[n as usize][1]).sum::<f64>() / ns.len() as f64,
                  ns.iter().map(|&n| ng[n as usize][2]).sum::<f64>() / ns.len() as f64];
        let eg = elem_grads[e as usize];
        let d = [eg[0]-gr[0], eg[1]-gr[1], eg[2]-gr[2]];
        eta.push(elem_vols[e as usize].powf(1.0/3.0) * (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]).sqrt());
    }
    eta
}

/// Element-wise Kelly (face-jump) estimator for 3-D meshes.
///
/// η_K² = ½ Σ_{F⊂∂K} h_F · ‖[[∇u_h · n]]‖²_L²(F)
///
/// Supports Tet4, Hex8, Prism6, Pyramid5.
pub fn kelly_estimator_3d_general(mesh: &Mesh<3>, u: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    let (elem_grads, _) = zz_gradients_and_volumes_3d(mesh, u);

    use std::collections::HashMap;
    let mut tri_face_elems: HashMap<(u32,u32,u32), Vec<ElemId>> = HashMap::new();
    let mut quad_face_elems: HashMap<[u32;4], Vec<ElemId>> = HashMap::new();

    match mesh.elem_type {
        ElementType::Tet4 => {
            for e in 0..n_elems as ElemId {
                let ns = mesh.elem_nodes(e);
                for &(a,b,c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
                    let mut v = [a,b,c]; v.sort_unstable();
                    tri_face_elems.entry((v[0],v[1],v[2])).or_default().push(e);
                }
            }
        }
        ElementType::Hex8 => {
            for e in 0..n_elems as ElemId {
                let ns = mesh.elem_nodes(e);
                for face in crate::amr::amr_inner::local_faces_hex() {
                    let fns = [ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];
                    let mut k = fns; k.sort_unstable();
                    quad_face_elems.entry(k).or_default().push(e);
                }
            }
        }
        ElementType::Prism6 => {
            for e in 0..n_elems as ElemId {
                let ns = mesh.elem_nodes(e);
                for (a,b,c) in crate::amr::amr_inner::local_faces_prism_tri() {
                    let mut v = [ns[a],ns[b],ns[c]]; v.sort_unstable();
                    tri_face_elems.entry((v[0],v[1],v[2])).or_default().push(e);
                }
                for face in crate::amr::amr_inner::local_faces_prism_quad() {
                    let fns = [ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];
                    let mut k = fns; k.sort_unstable();
                    quad_face_elems.entry(k).or_default().push(e);
                }
            }
        }
        ElementType::Pyramid5 => {
            for e in 0..n_elems as ElemId {
                let ns = mesh.elem_nodes(e);
                for (a,b,c) in crate::amr::amr_inner::local_faces_pyramid_tri() {
                    let mut v = [ns[a],ns[b],ns[c]]; v.sort_unstable();
                    tri_face_elems.entry((v[0],v[1],v[2])).or_default().push(e);
                }
                let qf = crate::amr::amr_inner::local_faces_pyramid_quad()[0];
                let fns = [ns[qf[0]],ns[qf[1]],ns[qf[2]],ns[qf[3]]];
                let mut k = fns; k.sort_unstable();
                quad_face_elems.entry(k).or_default().push(e);
            }
        }
        _ => panic!("kelly_estimator_3d_general: unsupported {:?}", mesh.elem_type),
    }

    let mut eta_sq = vec![0.0_f64; n_elems];

    for (&(na,nb,nc), elems) in &tri_face_elems {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize; let e1 = elems[1] as usize;
        let ca=mesh.coords_of(na);let cb=mesh.coords_of(nb);let cc=mesh.coords_of(nc);
        let ex=cb[0]-ca[0];let ey=cb[1]-ca[1];let ez=cb[2]-ca[2];
        let fx=cc[0]-ca[0];let fy=cc[1]-ca[1];let fz=cc[2]-ca[2];
        let nx=ey*fz-ez*fy;let ny=ez*fx-ex*fz;let nz=ex*fy-ey*fx;
        let fa=0.5*(nx*nx+ny*ny+nz*nz).sqrt();
        if fa < 1e-30 { continue; }
        let inv=1.0/(nx*nx+ny*ny+nz*nz).sqrt();
        let jump=(elem_grads[e0][0]-elem_grads[e1][0])*nx*inv
                +(elem_grads[e0][1]-elem_grads[e1][1])*ny*inv
                +(elem_grads[e0][2]-elem_grads[e1][2])*nz*inv;
        let hf=(2.0*fa).sqrt();
        let contrib=0.5*hf*jump*jump;
        eta_sq[e0]+=contrib; eta_sq[e1]+=contrib;
    }

    for (fns, elems) in &quad_face_elems {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize; let e1 = elems[1] as usize;
        let [a,b,c,d] = *fns;
        let ca=mesh.coords_of(a);let cb=mesh.coords_of(b);let cc=mesh.coords_of(c);let cd=mesh.coords_of(d);
        let ex1=cb[0]-ca[0];let ey1=cb[1]-ca[1];let ez1=cb[2]-ca[2];
        let fx1=cc[0]-ca[0];let fy1=cc[1]-ca[1];let fz1=cc[2]-ca[2];
        let nx1=ey1*fz1-ez1*fy1;let ny1=ez1*fx1-ex1*fz1;let nz1=ex1*fy1-ey1*fx1;
        let area1=0.5*(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let ex2=cd[0]-ca[0];let ey2=cd[1]-ca[1];let ez2=cd[2]-ca[2];
        let fx2=cc[0]-ca[0];let fy2=cc[1]-ca[1];let fz2=cc[2]-ca[2];
        let nx2=ey2*fz2-ez2*fy2;let ny2=ez2*fx2-ex2*fz2;let nz2=ex2*fy2-ey2*fx2;
        let area2=0.5*(nx2*nx2+ny2*ny2+nz2*nz2).sqrt();
        let fa=area1+area2;
        if fa < 1e-30 { continue; }
        let inv=1.0/(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let jump=(elem_grads[e0][0]-elem_grads[e1][0])*nx1*inv
                +(elem_grads[e0][1]-elem_grads[e1][1])*ny1*inv
                +(elem_grads[e0][2]-elem_grads[e1][2])*nz1*inv;
        let hf=(2.0*fa).sqrt();
        let contrib=0.5*hf*jump*jump;
        eta_sq[e0]+=contrib; eta_sq[e1]+=contrib;
    }

    eta_sq.iter().map(|v| v.sqrt()).collect()
}

/// Compute element gradients and volumes for the generalized 3-D estimators.
fn zz_gradients_and_volumes_3d(mesh: &Mesh<3>, u: &[f64]) -> (Vec<[f64;3]>, Vec<f64>) {
    let n_elems = mesh.n_elems();
    let mut g = Vec::with_capacity(n_elems);
    let mut v = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e); let npe = ns.len();
        let c = |i| mesh.coords_of(ns[i]);
        let j = match npe {
            4 => [[c(1)[0]-c(0)[0],c(2)[0]-c(0)[0],c(3)[0]-c(0)[0]],
                  [c(1)[1]-c(0)[1],c(2)[1]-c(0)[1],c(3)[1]-c(0)[1]],
                  [c(1)[2]-c(0)[2],c(2)[2]-c(0)[2],c(3)[2]-c(0)[2]]],
            8 => [[0.125*(-c(0)[0]+c(1)[0]+c(2)[0]-c(3)[0]-c(4)[0]+c(5)[0]+c(6)[0]-c(7)[0]),
                   0.125*(-c(0)[0]-c(1)[0]+c(2)[0]+c(3)[0]-c(4)[0]-c(5)[0]+c(6)[0]+c(7)[0]),
                   0.125*(-c(0)[0]-c(1)[0]-c(2)[0]-c(3)[0]+c(4)[0]+c(5)[0]+c(6)[0]+c(7)[0])],
                  [0.125*(-c(0)[1]+c(1)[1]+c(2)[1]-c(3)[1]-c(4)[1]+c(5)[1]+c(6)[1]-c(7)[1]),
                   0.125*(-c(0)[1]-c(1)[1]+c(2)[1]+c(3)[1]-c(4)[1]-c(5)[1]+c(6)[1]+c(7)[1]),
                   0.125*(-c(0)[1]-c(1)[1]-c(2)[1]-c(3)[1]+c(4)[1]+c(5)[1]+c(6)[1]+c(7)[1])],
                  [0.125*(-c(0)[2]+c(1)[2]+c(2)[2]-c(3)[2]-c(4)[2]+c(5)[2]+c(6)[2]-c(7)[2]),
                   0.125*(-c(0)[2]-c(1)[2]+c(2)[2]+c(3)[2]-c(4)[2]-c(5)[2]+c(6)[2]+c(7)[2]),
                   0.125*(-c(0)[2]-c(1)[2]-c(2)[2]-c(3)[2]+c(4)[2]+c(5)[2]+c(6)[2]+c(7)[2])]],
            6 => [[(c(1)[0]-c(0)[0])/2.0,(c(2)[0]-c(0)[0])/2.0,(c(3)[0]-c(0)[0])/2.0],
                  [(c(1)[1]-c(0)[1])/2.0,(c(2)[1]-c(0)[1])/2.0,(c(3)[1]-c(0)[1])/2.0],
                  [(c(1)[2]-c(0)[2])/2.0,(c(2)[2]-c(0)[2])/2.0,(c(3)[2]-c(0)[2])/2.0]],
            5 => [[c(1)[0]-c(0)[0],c(2)[0]-c(0)[0],c(4)[0]-c(0)[0]],
                  [c(1)[1]-c(0)[1],c(2)[1]-c(0)[1],c(4)[1]-c(0)[1]],
                  [c(1)[2]-c(0)[2],c(2)[2]-c(0)[2],c(4)[2]-c(0)[2]]],
            _ => panic!("zz_gradients_and_volumes_3d: unsupported npe={}", npe),
        };
        let idet = { let d=j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]); if d.abs()>1e-30{1.0/d}else{0.0} };
        let jit=|r:usize,c:usize|->f64{let a=(r+1)%3;let b=(r+2)%3;let d=(c+1)%3;let e=(c+2)%3;(j[a][d]*j[b][e]-j[a][e]*j[b][d])*idet};
        let gref:Vec<[f64;3]>=match npe{
            4=>vec![[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
            8=>vec![[-0.125,-0.125,-0.125],[0.125,-0.125,-0.125],[0.125,0.125,-0.125],[-0.125,0.125,-0.125],[-0.125,-0.125,0.125],[0.125,-0.125,0.125],[0.125,0.125,0.125],[-0.125,0.125,0.125]],
            6=>vec![[-0.5,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[-0.5,0.0,0.5],[0.5,0.0,0.5],[0.0,0.5,0.5]],
            5=>vec![[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[-1.0,0.0,0.0],[0.0,0.0,1.0]],
            _=>unreachable!()};
        let uh:Vec<f64>=ns.iter().map(|&n|u[n as usize]).collect();
        let mut grad=[0.0_f64;3];
        for k in 0..npe{for i in 0..3{for j in 0..3{grad[i]+=uh[k]*jit(j,i)*gref[k][j];}}}
        g.push(grad);
        let vol=match npe{
            4=>(j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs()/6.0,
            8=>(j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs(),
            6=>(j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs()/2.0,
            5=>{let v=|i|c(i);let d=|a:[f64;3],b:[f64;3],c:[f64;3],d:[f64;3]|->f64{(b[0]-a[0])*(c[1]-a[1])*(d[2]-a[2])+(b[1]-a[1])*(c[2]-a[2])*(d[0]-a[0])+(b[2]-a[2])*(c[0]-a[0])*(d[1]-a[1])-(b[2]-a[2])*(c[1]-a[1])*(d[0]-a[0])-(b[1]-a[1])*(c[0]-a[0])*(d[2]-a[2])-(b[0]-a[0])*(c[2]-a[2])*(d[1]-a[1])};(d(v(0),v(1),v(2),v(4)).abs()+d(v(2),v(3),v(0),v(4)).abs())/6.0}
            ,_=>unreachable!()};
        v.push(vol);
    }
    (g, v)
}

/// Element-wise residual error indicator for 3-D meshes of any type.
///
/// ```text
/// η_K² = h_K² · ‖f‖²_L²(K)  +  ½ Σ_{F⊂∂K} h_F · ‖[[∇u_h·n]]‖²_L²(F)
/// ```
///
/// For P1 elements the interior residual ∇·(κ∇u_h) vanishes element-wise,
/// reducing the formula to the face-jump estimator weighted by h_F.
///
/// Supports Tet4, Hex8, Prism6, Pyramid5.
pub fn residual_estimator_3d_general(mesh: &Mesh<3>, u: &[f64], f: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    let (elem_grads, elem_vols) = zz_gradients_and_volumes_3d(mesh, u);

    use std::collections::HashMap;
    let mut tri_faces: HashMap<(u32,u32,u32), Vec<ElemId>> = HashMap::new();
    let mut quad_faces: HashMap<[u32;4], Vec<ElemId>> = HashMap::new();

    match mesh.elem_type {
        ElementType::Tet4 => {
            for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
                for &(a,b,c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
                    let mut v=[a,b,c];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}}
        }
        ElementType::Hex8 => {
            for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
                for face in crate::amr::amr_inner::local_faces_hex() { let fns=[ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];
                    let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}}
        }
        ElementType::Prism6 => {
            for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
                for (a,b,c) in crate::amr::amr_inner::local_faces_prism_tri() {let mut v=[ns[a],ns[b],ns[c]];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}
                for face in crate::amr::amr_inner::local_faces_prism_quad() {let fns=[ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}}
        }
        ElementType::Pyramid5 => {
            for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
                for (a,b,c) in crate::amr::amr_inner::local_faces_pyramid_tri() {let mut v=[ns[a],ns[b],ns[c]];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}
                let qf=crate::amr::amr_inner::local_faces_pyramid_quad()[0];let fns=[ns[qf[0]],ns[qf[1]],ns[qf[2]],ns[qf[3]]];let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}
        }
        _ => panic!("residual_estimator_3d_general: unsupported {:?}", mesh.elem_type),
    }

    let mut eta_sq = vec![0.0_f64; n_elems];

    for e in 0..n_elems {
        let ns = mesh.elem_nodes(e as ElemId);
        let favg: f64 = ns.iter().map(|&n| f[n as usize]).sum::<f64>() / ns.len() as f64;
        let h2 = elem_vols[e].powf(2.0 / 3.0);
        eta_sq[e] += h2 * elem_vols[e] * favg * favg;
    }

    for (&(na,nb,nc), elems) in &tri_faces {
        if elems.len() != 2 { continue; }
        let e0=elems[0]as usize;let e1=elems[1]as usize;
        let ca=mesh.coords_of(na);let cb=mesh.coords_of(nb);let cc=mesh.coords_of(nc);
        let ex=cb[0]-ca[0];let ey=cb[1]-ca[1];let ez=cb[2]-ca[2];
        let fx=cc[0]-ca[0];let fy=cc[1]-ca[1];let fz=cc[2]-ca[2];
        let nx=ey*fz-ez*fy;let ny=ez*fx-ex*fz;let nz=ex*fy-ey*fx;
        let fa=0.5*(nx*nx+ny*ny+nz*nz).sqrt();
        if fa<1e-30{continue;}
        let inv=1.0/(nx*nx+ny*ny+nz*nz).sqrt();
        let jump=(elem_grads[e0][0]-elem_grads[e1][0])*nx*inv
                +(elem_grads[e0][1]-elem_grads[e1][1])*ny*inv
                +(elem_grads[e0][2]-elem_grads[e1][2])*nz*inv;
        let hf=(2.0*fa).sqrt();
        eta_sq[e0]+=0.5*hf*jump*jump;eta_sq[e1]+=0.5*hf*jump*jump;
    }

    for (fns, elems) in &quad_faces {
        if elems.len()!=2{continue;}
        let e0=elems[0]as usize;let e1=elems[1]as usize;
        let[a,b,c,d]=*fns;
        let ca=mesh.coords_of(a);let cb=mesh.coords_of(b);let cc=mesh.coords_of(c);let cd=mesh.coords_of(d);
        let ex1=cb[0]-ca[0];let ey1=cb[1]-ca[1];let ez1=cb[2]-ca[2];
        let fx1=cc[0]-ca[0];let fy1=cc[1]-ca[1];let fz1=cc[2]-ca[2];
        let nx1=ey1*fz1-ez1*fy1;let ny1=ez1*fx1-ex1*fz1;let nz1=ex1*fy1-ey1*fx1;
        let area1=0.5*(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let ex2=cd[0]-ca[0];let ey2=cd[1]-ca[1];let ez2=cd[2]-ca[2];
        let fx2=cc[0]-ca[0];let fy2=cc[1]-ca[1];let fz2=cc[2]-ca[2];
        let nx2=ey2*fz2-ez2*fy2;let ny2=ez2*fx2-ex2*fz2;let nz2=ex2*fy2-ey2*fx2;
        let area2=0.5*(nx2*nx2+ny2*ny2+nz2*nz2).sqrt();
        let fa=area1+area2;
        if fa<1e-30{continue;}
        let inv=1.0/(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let jump=(elem_grads[e0][0]-elem_grads[e1][0])*nx1*inv
                +(elem_grads[e0][1]-elem_grads[e1][1])*ny1*inv
                +(elem_grads[e0][2]-elem_grads[e1][2])*nz1*inv;
        let hf=(2.0*fa).sqrt();
        eta_sq[e0]+=0.5*hf*jump*jump;eta_sq[e1]+=0.5*hf*jump*jump;
    }

    eta_sq.iter().map(|v| v.sqrt()).collect()
}

/// Element-wise DWR (Dual Weighted Residual) indicator for 3-D meshes.
///
/// ```text
/// η_K = |∫_K f · ω dΩ  +  ½ Σ_{F⊂∂K} ∫_F [[∇u_h·n]] · ω dS|
/// ```
///
/// where ω = z − avg(z) is the dual solution fluctuation (element-wise).
/// For P1 elements, the interior residual ∇·(κ∇u_h) vanishes element-wise,
/// leaving only the source term and face jumps.
///
/// Supports Tet4, Hex8, Prism6, Pyramid5.
///
/// # Arguments
/// * `mesh` — the mesh
/// * `u` — primal solution (nodal values)
/// * `z` — dual solution (nodal values)
/// * `f` — source term (nodal values)
pub fn dwr_estimator_3d_general(mesh: &Mesh<3>, u: &[f64], z: &[f64], f: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    let (elem_grads, elem_vols) = zz_gradients_and_volumes_3d(mesh, u);

    use std::collections::HashMap;
    let mut tri_faces: HashMap<(u32,u32,u32), Vec<ElemId>> = HashMap::new();
    let mut quad_faces: HashMap<[u32;4], Vec<ElemId>> = HashMap::new();

    match mesh.elem_type {
        ElementType::Tet4 => { for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
            for &(a,b,c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
                let mut v=[a,b,c];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}}
        }
        ElementType::Hex8 => { for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
            for face in crate::amr::amr_inner::local_faces_hex() { let fns=[ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];
                let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}}
        }
        ElementType::Prism6 => { for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
            for (a,b,c) in crate::amr::amr_inner::local_faces_prism_tri() {let mut v=[ns[a],ns[b],ns[c]];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}
            for face in crate::amr::amr_inner::local_faces_prism_quad() {let fns=[ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}}
        }
        ElementType::Pyramid5 => { for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
            for (a,b,c) in crate::amr::amr_inner::local_faces_pyramid_tri() {let mut v=[ns[a],ns[b],ns[c]];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}
            let qf=crate::amr::amr_inner::local_faces_pyramid_quad()[0];let fns=[ns[qf[0]],ns[qf[1]],ns[qf[2]],ns[qf[3]]];let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}
        }
        _ => panic!("dwr_estimator_3d_general: unsupported {:?}", mesh.elem_type),
    }

    let mut elem_omega: Vec<f64> = Vec::with_capacity(n_elems);
    let mut elem_f_avg: Vec<f64> = Vec::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let npe = ns.len();
        let z_avg: f64 = ns.iter().map(|&n| z[n as usize]).sum::<f64>() / npe as f64;
        let f_avg: f64 = ns.iter().map(|&n| f[n as usize]).sum::<f64>() / npe as f64;
        elem_omega.push(z_avg);
        elem_f_avg.push(f_avg);
    }

    let mut eta = vec![0.0_f64; n_elems];

    for e in 0..n_elems {
        eta[e] += (elem_f_avg[e] * elem_omega[e]).abs() * elem_vols[e];
    }

    for (&(na,nb,nc), elems) in &tri_faces {
        if elems.len() != 2 { continue; }
        let e0=elems[0]as usize;let e1=elems[1]as usize;
        let ca=mesh.coords_of(na);let cb=mesh.coords_of(nb);let cc=mesh.coords_of(nc);
        let ex=cb[0]-ca[0];let ey=cb[1]-ca[1];let ez=cb[2]-ca[2];
        let fx=cc[0]-ca[0];let fy=cc[1]-ca[1];let fz=cc[2]-ca[2];
        let nx=ey*fz-ez*fy;let ny=ez*fx-ex*fz;let nz=ex*fy-ey*fx;
        let fa=0.5*(nx*nx+ny*ny+nz*nz).sqrt();
        if fa<1e-30{continue;}
        let inv=1.0/(nx*nx+ny*ny+nz*nz).sqrt();
        let j0=elem_grads[e0][0]*nx*inv+elem_grads[e0][1]*ny*inv+elem_grads[e0][2]*nz*inv;
        let j1=elem_grads[e1][0]*nx*inv+elem_grads[e1][1]*ny*inv+elem_grads[e1][2]*nz*inv;
        let jump=(j0-j1).abs();
        if jump<1e-30{continue;}
        let w_mid=(elem_omega[e0]+elem_omega[e1])*0.5;
        let hf=(2.0*fa).sqrt();
        let contrib=0.5*hf*jump*w_mid.abs();
        eta[e0]+=contrib;eta[e1]+=contrib;
    }

    for (fns, elems) in &quad_faces {
        if elems.len()!=2{continue;}
        let e0=elems[0]as usize;let e1=elems[1]as usize;
        let[a,b,c,d]=*fns;
        let ca=mesh.coords_of(a);let cb=mesh.coords_of(b);let cc=mesh.coords_of(c);let cd=mesh.coords_of(d);
        let ex1=cb[0]-ca[0];let ey1=cb[1]-ca[1];let ez1=cb[2]-ca[2];
        let fx1=cc[0]-ca[0];let fy1=cc[1]-ca[1];let fz1=cc[2]-ca[2];
        let nx1=ey1*fz1-ez1*fy1;let ny1=ez1*fx1-ex1*fz1;let nz1=ex1*fy1-ey1*fx1;
        let area1=0.5*(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let ex2=cd[0]-ca[0];let ey2=cd[1]-ca[1];let ez2=cd[2]-ca[2];
        let fx2=cc[0]-ca[0];let fy2=cc[1]-ca[1];let fz2=cc[2]-ca[2];
        let nx2=ey2*fz2-ez2*fy2;let ny2=ez2*fx2-ex2*fz2;let nz2=ex2*fy2-ey2*fx2;
        let area2=0.5*(nx2*nx2+ny2*ny2+nz2*nz2).sqrt();
        let fa=area1+area2;
        if fa<1e-30{continue;}
        let inv=1.0/(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let j0=elem_grads[e0][0]*nx1*inv+elem_grads[e0][1]*ny1*inv+elem_grads[e0][2]*nz1*inv;
        let j1=elem_grads[e1][0]*nx1*inv+elem_grads[e1][1]*ny1*inv+elem_grads[e1][2]*nz1*inv;
        let jump=(j0-j1).abs();
        if jump<1e-30{continue;}
        let w_mid=(elem_omega[e0]+elem_omega[e1])*0.5;
        let hf=(2.0*fa).sqrt();
        let contrib=0.5*hf*jump*w_mid.abs();
        eta[e0]+=contrib;eta[e1]+=contrib;
    }

    eta
}
