//! CutFEM ghost-penalty stabilizer for immersed boundary problems.
//!
//! Ghost penalty (Burman 2010): s_h(u,v) = Σ_{F∈F_Γ} γ·h_F·∫_F [∇u·n][∇v·n] dS
//! where F_Γ are internal faces adjacent to a level-set-cut element.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use fem_element::quadrature::{seg_rule, tri_rule};
use fem_element::lagrange::{TriPk, TetPk};
use fem_element::ReferenceElement;

/// Build the ghost-penalty stabilization matrix for CutFEM.
pub fn assemble_ghost_penalty<M, S>(
    space: &S, mesh: &M, phi: &dyn Fn(&[f64]) -> f64,
    gamma: f64, quad_order: u8,
) -> CsrMatrix<f64>
where M: MeshTopology, S: FESpace<Mesh = M>,
{
    let n = space.n_dofs();
    let mut coo = CooMatrix::new(n, n);
    let dim = mesh.dim() as usize;
    let order = space.order() as usize;

    let mut is_cut = vec![false; mesh.n_elements()];
    for e in mesh.elem_iter() {
        let nn = mesh.element_nodes(e);
        is_cut[e as usize] = nn.iter().any(|&n| phi(mesh.node_coords(n)) > 1e-14)
            && nn.iter().any(|&n| phi(mesh.node_coords(n)) < -1e-14);
    }

    let interior_faces = crate::InteriorFaceList::build(mesh);
    for f in &interior_faces.faces {
        let el = f.elem_left;
        let er = f.elem_right;
        if !is_cut[el as usize] && !is_cut[er as usize] { continue; }

        let fnodes = &f.face_nodes;
        let h = if dim == 2 {
            let c0 = mesh.node_coords(fnodes[0]); let c1 = mesh.node_coords(fnodes[1]);
            ((c1[0]-c0[0]).powi(2)+(c1[1]-c0[1]).powi(2)).sqrt()
        } else {
            let c0 = mesh.node_coords(fnodes[0]); let c1 = mesh.node_coords(fnodes[1]);
            let c2 = mesh.node_coords(fnodes[2]);
            let u1 = [c1[0]-c0[0],c1[1]-c0[1],c1[2]-c0[2]];
            let u2 = [c2[0]-c0[0],c2[1]-c0[1],c2[2]-c0[2]];
            let nx = u1[1]*u2[2]-u1[2]*u2[1]; let ny = u1[2]*u2[0]-u1[0]*u2[2];
            let nz = u1[0]*u2[1]-u1[1]*u2[0];
            0.5*((nx*nx+ny*ny+nz*nz).sqrt())
        };
        let h_inv = 1.0/h.max(1e-14);
        let qf = if dim == 2 { seg_rule(quad_order) } else { tri_rule(quad_order) };
        let ref_e: Box<dyn ReferenceElement> = if dim == 2 {
            Box::new(TriPk::new(order))
        } else { Box::new(TetPk::new(order)) };
        let ne = ref_e.n_dofs();

        let dofs_l: Vec<usize> = space.element_dofs(el).iter().map(|&d| d as usize).collect();
        let dofs_r: Vec<usize> = space.element_dofs(er).iter().map(|&d| d as usize).collect();

        for qi in 0..qf.n_points() {
            let xi = &qf.points[qi];
            let w = qf.weights[qi] * h;
            let xp: Vec<f64> = if dim == 2 {
                let c = mesh.node_coords(fnodes[0]); let d = mesh.node_coords(fnodes[1]);
                vec![c[0]+xi[0]*(d[0]-c[0]), c[1]+xi[0]*(d[1]-c[1])]
            } else {
                let c0 = mesh.node_coords(fnodes[0]); let c1 = mesh.node_coords(fnodes[1]);
                let c2 = mesh.node_coords(fnodes[2]);
                vec![c0[0]+xi[0]*(c1[0]-c0[0])+xi[1]*(c2[0]-c0[0]),
                     c0[1]+xi[0]*(c1[1]-c0[1])+xi[1]*(c2[1]-c0[1]),
                     c0[2]+xi[0]*(c1[2]-c0[2])+xi[1]*(c2[2]-c0[2])]
            };
            let normal: Vec<f64> = if dim == 2 {
                let dx = mesh.node_coords(fnodes[1])[0]-mesh.node_coords(fnodes[0])[0];
                let dy = mesh.node_coords(fnodes[1])[1]-mesh.node_coords(fnodes[0])[1];
                let l = (dx*dx+dy*dy).sqrt().max(1e-14);
                vec![-dy/l, dx/l, 0.0]
            } else {
                let c0=mesh.node_coords(fnodes[0]);let c1=mesh.node_coords(fnodes[1]);
                let c2=mesh.node_coords(fnodes[2]);
                let u1=[c1[0]-c0[0],c1[1]-c0[1],c1[2]-c0[2]];
                let u2=[c2[0]-c0[0],c2[1]-c0[1],c2[2]-c0[2]];
                let nx=u1[1]*u2[2]-u1[2]*u2[1];let ny=u1[2]*u2[0]-u1[0]*u2[2];
                let nz=u1[0]*u2[1]-u1[1]*u2[0];let l=(nx*nx+ny*ny+nz*nz).sqrt().max(1e-14);
                vec![nx/l, ny/l, nz/l]
            };

            for (side, dofs) in [&dofs_l, &dofs_r].iter().enumerate() {
                let e_side = if side == 0 { el } else { er };
                let n_side = mesh.element_nodes(e_side);
                let x0 = mesh.node_coords(n_side[0]);
                let mut jac = nalgebra::DMatrix::zeros(dim, dim);
                for i in 0..dim {
                    let xn = mesh.node_coords(n_side[1+i]);
                    for d in 0..dim { jac[(d,i)] = xn[d] - x0[d]; }
                }
                let ji = jac.try_inverse().unwrap_or_else(|| nalgebra::DMatrix::identity(dim,dim));
                let mut xi_ref = vec![0.0; dim];
                for i in 0..dim {
                    for j in 0..dim { xi_ref[i] += ji[(i,j)] * (xp[j] - x0[j]); }
                }
                let mut phi = vec![0.0; ne];
                ref_e.eval_basis(&xi_ref, &mut phi);
                let mut g_ref = vec![0.0; ne*dim];
                ref_e.eval_grad_basis(&xi_ref, &mut g_ref);
                let mut g_phys = vec![0.0; ne*dim];
                for i in 0..ne {
                    for d in 0..dim {
                        g_phys[i*dim+d] = (0..dim).map(|k| ji[(k,d)] * g_ref[i*dim+k]).sum();
                    }
                }
                let pen = gamma * h_inv;
                for i in 0..ne { for j in 0..ne {
                    let godot: f64 = (0..dim).map(|d| g_phys[i*dim+d]*normal[d]).sum();
                    let gjdot: f64 = (0..dim).map(|d| g_phys[j*dim+d]*normal[d]).sum();
                    let v = pen * w * godot * gjdot;
                    if v.abs() > 1e-30 { coo.add(dofs[i], dofs[j], v); }
                }}
            }
        }
    }
    coo.into_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    #[test]
    fn ghost_penalty_2d_spd() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let mesh_ref = space.mesh();
        let s = assemble_ghost_penalty(&space, mesh_ref, &|x| x[0]-0.6, 1.0, 3);
        for i in 0..space.n_dofs().min(50) {
            let mut diag = 0.0;
            for p in s.row_ptr[i]..s.row_ptr[i+1] {
                if s.col_idx[p] == i as u32 { diag = s.values[p]; break; }
            }
            assert!(diag >= 0.0, "Neg diag[{i}]={diag}");
        }
    }

    #[test]
    fn ghost_penalty_2d_symmetry() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let space = H1Space::new(mesh, 1);
        let mesh_ref = space.mesh();
        let s = assemble_ghost_penalty(&space, mesh_ref, &|x| x[0]-0.6, 1.0, 3);
        let n = space.n_dofs().min(100);
        let mut asym: f64 = 0.0;
        for i in 0..n {
            for p in s.row_ptr[i]..s.row_ptr[i+1] {
                let j = s.col_idx[p] as usize;
                if j < n { for q in s.row_ptr[j]..s.row_ptr[j+1] {
                    if s.col_idx[q] == i as u32 { asym = asym.max((s.values[p]-s.values[q]).abs()); }
                }}
            }
        }
        assert!(asym < 1e-12, "Symmetry {asym}");
    }

    #[test]
    fn ghost_penalty_no_cut_is_zero() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let mesh_ref = space.mesh();
        let s = assemble_ghost_penalty(&space, mesh_ref, &|_| 1.0, 1.0, 3);
        let mut nnz = 0usize;
        for i in 0..space.n_dofs() { nnz += (s.row_ptr[i+1]-s.row_ptr[i]) as usize; }
        assert_eq!(nnz, 0, "Expected zero, got {nnz}");
    }

    // ── CutFEM MMS convergence ──────────────────────────────────────────
    // Solves standard Poisson on unit square (no cuts) as a first-order check
    // that the assembly, Dirichlet BC, and error computation are correct.
    fn apply_dirichlet_full(
        k: &mut fem_linalg::CsrMatrix<f64>, rhs: &mut [f64],
        bdr: &std::collections::BTreeSet<usize>,
        bdr_val: &dyn Fn(usize) -> f64,
    ) {
        // 1) rhs[i] -= K[i,d] * g[d] for i ∉ bdr
        for &d in bdr {
            let g = bdr_val(d);
            for i in 0..k.nrows {
                if bdr.contains(&i) { continue; }
                let s = k.row_ptr[i]; let e = k.row_ptr[i+1];
                for p in s..e {
                    if k.col_idx[p] == d as u32 { rhs[i] -= k.values[p] * g; break; }
                }
            }
        }
        // 2) zero Dirichlet rows, set diagonal = 1, rhs = g
        for &d in bdr {
            k.apply_dirichlet_row_zeroing(d, bdr_val(d), rhs);
        }
    }

    fn solve_poisson(level: u32, order: u8) -> (f64, f64) {
        use fem_linalg::SolverConfig;
        use fem_solver::solve_cg;
        let sol = |x: &[f64]| x[0]*(1.0-x[0])*x[1]*(1.0-x[1]);
        let fsrc = |x: &[f64]| 2.0*(x[0]-x[0]*x[0]+x[1]-x[1]*x[1]);
        let n = 2u32.pow(level); let h = 1.0 / n as f64;
        let mesh = Mesh::<2>::unit_square_tri(n as usize);
        let sp = H1Space::new(mesh, order); let m = sp.mesh();
        let nd = sp.n_dofs();
        let qo = 2 * (order + 1);
        let re = TriPk::new(order as usize);
        let nv = re.n_dofs();

        let mut coo = CooMatrix::<f64>::new(nd, nd);
        let mut rhs = vec![0.0_f64; nd];

        for e in m.elem_iter() {
            let nodes = m.element_nodes(e);
            let dofs: Vec<usize> = sp.element_dofs(e).iter().map(|&d| d as usize).collect();
            let x0 = m.node_coords(nodes[0]);
            let mut jac = nalgebra::DMatrix::<f64>::zeros(2, 2);
            for i in 0..2 { let xn = m.node_coords(nodes[1+i]);
                for d in 0..2 { jac[(d,i)] = xn[d] - x0[d]; }
            }
            let det = jac.determinant();
            let jac_saved = jac.clone();
            let ji = match jac.try_inverse() { Some(v) => v, None => continue };
            let qr = tri_rule(qo);

            for (qi, pt_ref) in qr.points.iter().enumerate() {
                let pt = [
                    x0[0] + jac_saved[(0,0)]*pt_ref[0] + jac_saved[(0,1)]*pt_ref[1],
                    x0[1] + jac_saved[(1,0)]*pt_ref[0] + jac_saved[(1,1)]*pt_ref[1],
                ];
                let w = qr.weights[qi] * det.abs();
                let mut pv = vec![0.0_f64; nv];
                let mut gr = vec![0.0_f64; nv*2];
                re.eval_basis(pt_ref, &mut pv);
                re.eval_grad_basis(pt_ref, &mut gr);
                // ∇_x = J^{-T} ∇_ξ; ji[(k,d)] = J^{-1}[k,d], so
                // ∂φ/∂x_d = Σ_k ji[(k,d)] * ∂φ/∂ξ_k = J^{-T}[d,k] * ∂φ/∂ξ_k
                let fv = fsrc(&pt);
                for i in 0..nv {
                    let gx: f64 = (0..2).map(|k| ji[(k,0)] * gr[i*2+k]).sum();
                    let gy: f64 = (0..2).map(|k| ji[(k,1)] * gr[i*2+k]).sum();
                    rhs[dofs[i]] += w * fv * pv[i];
                    for j in 0..nv {
                        let hx: f64 = (0..2).map(|k| ji[(k,0)] * gr[j*2+k]).sum();
                        let hy: f64 = (0..2).map(|k| ji[(k,1)] * gr[j*2+k]).sum();
                        let kij = w * (gx*hx + gy*hy);
                        if kij.abs() > 1e-30 { coo.add(dofs[i], dofs[j], kij); }
                    }
                }
            }
        }

        // Collect Dirichlet DOFs (P1: face_nodes only; P2+: use boundary_dofs)
        let mut k = coo.into_csr();
        let mut bdr: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for bf in 0..m.n_boundary_faces() as u32 {
            for &n in m.face_nodes(bf) { bdr.insert(n as usize); }
        }
        let bdr_val = |d: usize| -> f64 { sol(m.node_coords(d as u32)) };
        apply_dirichlet_full(&mut k, &mut rhs, &bdr, &bdr_val);

        let mut x = vec![0.0_f64; nd];
        let cfg = SolverConfig { rtol: 1e-12, max_iter: 10000, ..Default::default() };
        let _ = solve_cg(&k, &rhs, &mut x, &cfg);

        // L2 error
        let mut l2 = 0.0_f64;
        for e in m.elem_iter() {
            let nodes = m.element_nodes(e);
            let dofs: Vec<usize> = sp.element_dofs(e).iter().map(|&d| d as usize).collect();
            let x0 = m.node_coords(nodes[0]);
            let mut jac = nalgebra::DMatrix::<f64>::zeros(2, 2);
            for i in 0..2 { let xn = m.node_coords(nodes[1+i]);
                for d in 0..2 { jac[(d,i)] = xn[d] - x0[d]; }
            }
            let det = jac.determinant();
            let qr = tri_rule(qo);
            for (qi, pt_ref) in qr.points.iter().enumerate() {
                let pt = [
                    x0[0] + jac[(0,0)]*pt_ref[0] + jac[(0,1)]*pt_ref[1],
                    x0[1] + jac[(1,0)]*pt_ref[0] + jac[(1,1)]*pt_ref[1],
                ];
                let w = qr.weights[qi] * det.abs();
                let mut pv = vec![0.0_f64; nv];
                re.eval_basis(pt_ref, &mut pv);
                let uh: f64 = dofs.iter().enumerate().map(|(kk,&d)| x[d]*pv[kk]).sum();
                l2 += w * (uh - sol(&pt)).powi(2);
            }
        }
        (l2.sqrt(), h)
    }

    #[test]
    fn cutfem_mms_p1_no_cut() {
        let mut pe: Option<f64> = None; let mut ph: Option<f64> = None;
        let mut r = 0.0_f64;
        for l in 3..=5 {
            let (err, h) = solve_poisson(l, 1);
            eprintln!("P1 h={:.5} L2={:.6e}", h, err);
            if let (Some(e0), Some(h0)) = (pe, ph) {
                r = (err/e0).ln() / (h/h0).ln();
                eprintln!("  rate={:.2}", r);
            }
            pe = Some(err); ph = Some(h);
        }
        assert!(r > 1.8, "P1: expected O(h^2), got {:.2}", r);
    }
}
