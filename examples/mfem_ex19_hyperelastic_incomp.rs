//! Example 19 — 1:1 translation of MFEM ex19
//! Quasi-static incompressible neo-Hookean hyperelasticity (mixed u/p).
//!
//! Solves H(x) = 0 via Newton's method with block-preconditioned GMRES.
//!
//! BCs (matching MFEM ex19):
//!   Boundary attribute 1: u = 0 (fixed)
//!   Boundary attribute 2: u_x = 0, u_y = 0.25·x (prescribed shear)
//!
//! Usage:
//!   cargo run --example mfem_ex19_hyperelastic_incomp
//!   cargo run --example mfem_ex19_hyperelastic_incomp -- -m data/beam-quad.mesh -o 2 -r 0
//!   cargo run --example mfem_ex19_hyperelastic_incomp -- -mu 1.0 -rel 1e-4 -abs 1e-6 -it 500

#![allow(non_snake_case)]

use std::fs::File;
use std::io::Write;
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{BlockMatrix, CooMatrix, CsrMatrix, SolverConfig};
use fem_solver::block_operator::right_preconditioned_gmres;
use fem_solver::{solve_gmres_gssmoother, solve_pcg_gssmoother};
use fem_mesh::{refine_uniform, MeshTopology};
use fem_space::{constraints::boundary_dofs, fe_space::FESpace, H1Space, VectorH1Space};
use fem_element::lagrange::{TriP1, TriP2, TriP3, QuadQ1, QuadQ2, QuadQ3, QuadQ4};
use fem_element::lagrange::tet::{TetP1, TetP2, TetP3};
use fem_element::lagrange::hex::{HexQ1, HexQ2, HexQ3};
use fem_element::lagrange::prism::PrismPk;
use fem_mesh::element_type::ElementType;
use nalgebra::DMatrix;

// ─── Reference element helpers ─────────────────────────────────────────

/// Factory: return a reference element for the given type and order.
fn re(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Quad4, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        (ElementType::Quad4, 3) => Box::new(QuadQ3),
        (ElementType::Quad4, 4) => Box::new(QuadQ4),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        (ElementType::Hex8, 1) => Box::new(HexQ1),
        (ElementType::Hex8, 2) => Box::new(HexQ2),
        (ElementType::Hex8, 3) => Box::new(HexQ3),
        (ElementType::Prism6, 1) => Box::new(PrismPk::new(1)),
        (ElementType::Prism6, 2) => Box::new(PrismPk::new(2)),
        (ElementType::Prism6, 3) => Box::new(PrismPk::new(3)),
        _ => panic!("unsupported element type {et:?} x order {order}"),
    }
}

/// Compute element Jacobian determinant and inverse-transpose at a reference point.
/// Returns (detJ, J^{-T}) where J = ∂x/∂ξ.
fn jacf(m: &impl MeshTopology, elem: u32, xi: &[f64], dim: usize) -> (f64, DMatrix<f64>) {
    let et = m.element_type(elem);
    let nd = m.element_nodes(elem);
    let n_ldofs = nd.len();
    // MFEM: uses FE collection for the mesh, here we use order=1 (linear geometry)
    let re_geom = re(et, 1);
    let mut grad = vec![0.0_f64; n_ldofs * dim];
    re_geom.eval_grad_basis(xi, &mut grad);
    let mut jac = DMatrix::<f64>::zeros(dim, dim);
    for k in 0..n_ldofs {
        let x = m.node_coords(nd[k]);
        for i in 0..dim {
            for j in 0..dim {
                jac[(i, j)] += x[i] * grad[k * dim + j];
            }
        }
    }
    let det = jac.determinant();
    let inv = jac.try_inverse().expect("singular Jacobian");
    (det, inv.transpose()) // return J^{-T} for covariant gradient transform
}

/// Transform reference-element gradients to physical space:
///   gp[a*dim + j] = Σ_k J^{-T}_{(j,k)} * gr[a*dim + k]
fn xform_grads(ji: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for a in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += ji[(j, k)] * gr[a * dim + k];
            }
            gp[a * dim + j] = s;
        }
    }
}

/// Euclidean norm of a slice.
fn nr(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// ─── Pressure mass matrix ──────────────────────────────────────────────

fn build_pressure_mass(
    mesh: impl MeshTopology + Clone,
    p_order: u8,
    quad_order: u8,
    np: usize,
) -> CsrMatrix<f64> {
    use fem_space::fe_space::FESpace;
    let space = H1Space::new(mesh.clone(), p_order);
    let mut coo = CooMatrix::<f64>::new(np, np);
    let ne = mesh.n_elements() as usize;
    for e in 0..ne {
        let et = mesh.element_type(e as u32);
        let ref_elem = re(et, p_order);
        let n_ldofs = ref_elem.n_dofs();
        let edofs: Vec<usize> = space.element_dofs(e as u32)
            .iter().map(|&d| d as usize).collect();
        let q = ref_elem.quadrature(quad_order);
        let mut phi = vec![0.0_f64; n_ldofs];
        let mut me = vec![0.0_f64; n_ldofs * n_ldofs];
        for (qi, xi) in q.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let (det_j, _ji) = jacf(&mesh, e as u32, xi, mesh.dim() as usize);
            let w = q.weights[qi] * det_j.abs();
            for i in 0..n_ldofs {
                for j in 0..n_ldofs {
                    me[i * n_ldofs + j] += w * phi[i] * phi[j];
                }
            }
        }
        for a in 0..n_ldofs {
            for b in 0..n_ldofs {
                coo.add(edofs[a], edofs[b], me[a * n_ldofs + b]);
            }
        }
    }
    coo.into_csr()
}

// ─── Residual assembly ─────────────────────────────────────────────────

/// Compute the residual vector [R_u; R_p] for the mixed u/p formulation.
///
/// R_u(i,a) = ∫ (μ·F_{iJ} - p·F^{-T}_{iJ}) · ∂φ_a/∂X_J  dx
/// R_p(m)   = ∫ (J - 1) · ψ_m                              dx
fn residual(
    mesh: &impl MeshTopology,
    dim: usize,
    order: u8,
    p_order: u8,
    quad_order: u8,
    mu: f64,
    u: &[f64],
    p: &[f64],
    elem_dofs_u: &[Vec<usize>],
    elem_dofs_p: &[Vec<usize>],
    ru: &mut [f64],
    rp: &mut [f64],
) {
    ru.fill(0.0);
    rp.fill(0.0);
    let ne = mesh.n_elements() as usize;
    for e in 0..ne {
        let et = mesh.element_type(e as u32);
        let ru_ref = re(et, order);
        let rp_ref = re(et, p_order);
        let n_du = ru_ref.n_dofs();       // scalar shape functions (displacement)
        let n_dp = rp_ref.n_dofs();       // scalar shape functions (pressure)
        let n_vd = n_du * dim;            // vector DOFs per element

        let eu: &[usize] = &elem_dofs_u[e];
        let ep: &[usize] = &elem_dofs_p[e];

        let mut ue = vec![0.0_f64; n_vd];
        for (k, &g) in eu.iter().enumerate() {
            ue[k] = u[g];
        }
        let mut pe = vec![0.0_f64; n_dp];
        for (k, &g) in ep.iter().enumerate() {
            pe[k] = p[g];
        }

        let q = ru_ref.quadrature(quad_order);
        let mut phi_u = vec![0.0_f64; n_du];
        let mut gr_u = vec![0.0_f64; n_du * dim];
        let mut gp_u = vec![0.0_f64; n_du * dim];
        let mut phi_p = vec![0.0_f64; n_dp];

        let mut fu_e = vec![0.0_f64; n_vd];
        let mut fp_e = vec![0.0_f64; n_dp];

        for (qi, xi) in q.points.iter().enumerate() {
            ru_ref.eval_basis(xi, &mut phi_u);
            ru_ref.eval_grad_basis(xi, &mut gr_u);
            rp_ref.eval_basis(xi, &mut phi_p);

            let (det_j, ji) = jacf(mesh, e as u32, xi, dim);
            xform_grads(&ji, &gr_u, &mut gp_u, n_du, dim);
            let w = q.weights[qi] * det_j.abs();

            // Deformation gradient F = I + ∇u
            let mut F = DMatrix::<f64>::identity(dim, dim);
            for k in 0..n_du {
                for i in 0..dim {
                    for j in 0..dim {
                        // ue[k*dim + i] = u_i component at scalar DOF k
                        // gp_u[k*dim + j] = ∂φ_k/∂X_j
                        F[(i, j)] += ue[k * dim + i] * gp_u[k * dim + j];
                    }
                }
            }
            let dJ = F.determinant();
            let iF = F.clone().try_inverse().unwrap_or_else(|| DMatrix::<f64>::identity(dim, dim));
            let FT = iF.transpose(); // F^{-T}

            // Pressure at this QP
            let mut pres = 0.0;
            for k in 0..n_dp {
                pres += pe[k] * phi_p[k];
            }

            // First Piola-Kirchhoff stress: P = μ·F - p·F^{-T}
            let mut P = DMatrix::<f64>::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    P[(i, j)] = mu * F[(i, j)] - pres * FT[(i, j)];
                }
            }

            // R_u contribution: P : ∇v  (v = φ_a · e_i)
            for a in 0..n_du {
                for i in 0..dim {
                    let row = a * dim + i;
                    let mut s = 0.0;
                    for j in 0..dim {
                        s += P[(i, j)] * gp_u[a * dim + j];
                    }
                    fu_e[row] += w * s;
                }
            }

            // R_p contribution: (J - 1) · ψ
            for m in 0..n_dp {
                fp_e[m] += w * (dJ - 1.0) * phi_p[m];
            }
        }

        // Scatter to global
        for (k, &g) in eu.iter().enumerate() {
            ru[g] += fu_e[k];
        }
        for (k, &g) in ep.iter().enumerate() {
            rp[g] += fp_e[k];
        }
    }
}

// ─── Jacobian assembly ─────────────────────────────────────────────────

/// Assemble the block Jacobian J = [K_uu, K_up; K_pu, 0] and
/// apply Dirichlet row-zeroing.
///
/// K_uu[(a,i),(b,j)] = ∫ C_{iIjJ} · ∂φ_a/∂X_I · ∂φ_b/∂X_J  dx
///   C_{iIjJ} = μ·δ_{ij}·δ_{IJ} + p·F^{-T}_{jI}·F^{-T}_{iJ}
///
/// K_up[(a,i),m] = -∫ ψ_m · F^{-T}_{iJ} · ∂φ_a/∂X_J  dx
///
/// K_pu[m,(b,j)] =  ∫ J · F^{-T}_{jJ} · ∂φ_b/∂X_J · ψ_m  dx
fn jacobian(
    mesh: &impl MeshTopology,
    dim: usize,
    order: u8,
    p_order: u8,
    quad_order: u8,
    mu: f64,
    u: &[f64],
    p: &[f64],
    elem_dofs_u: &[Vec<usize>],
    elem_dofs_p: &[Vec<usize>],
    nu: usize,
    np: usize,
    du: &[(usize, f64)],
) -> (Vec<usize>, BlockMatrix) {
    let nt = nu + np;
    let mut coo = CooMatrix::<f64>::new(nt, nt);

    let ne = mesh.n_elements() as usize;
    for e in 0..ne {
        let et = mesh.element_type(e as u32);
        let ru_ref = re(et, order);
        let rp_ref = re(et, p_order);
        let n_du = ru_ref.n_dofs();
        let n_dp = rp_ref.n_dofs();
        let n_vd = n_du * dim;

        let eu: &[usize] = &elem_dofs_u[e];
        let ep: &[usize] = &elem_dofs_p[e];

        let mut ue = vec![0.0_f64; n_vd];
        for (k, &g) in eu.iter().enumerate() {
            ue[k] = u[g];
        }
        let mut pe = vec![0.0_f64; n_dp];
        for (k, &g) in ep.iter().enumerate() {
            pe[k] = p[g];
        }

        let q = ru_ref.quadrature(quad_order);
        let mut phi_u = vec![0.0_f64; n_du];
        let mut gr_u = vec![0.0_f64; n_du * dim];
        let mut gp_u = vec![0.0_f64; n_du * dim];
        let mut phi_p = vec![0.0_f64; n_dp];

        // Element stiffness blocks
        let mut kuu = vec![0.0_f64; n_vd * n_vd];
        let mut kup = vec![0.0_f64; n_vd * n_dp];
        let mut kpu = vec![0.0_f64; n_dp * n_vd];

        for (qi, xi) in q.points.iter().enumerate() {
            ru_ref.eval_basis(xi, &mut phi_u);
            ru_ref.eval_grad_basis(xi, &mut gr_u);
            rp_ref.eval_basis(xi, &mut phi_p);

            let (det_j, ji) = jacf(mesh, e as u32, xi, dim);
            xform_grads(&ji, &gr_u, &mut gp_u, n_du, dim);
            let w = q.weights[qi] * det_j.abs();

            // Deformation gradient
            let mut F = DMatrix::<f64>::identity(dim, dim);
            for k in 0..n_du {
                for i in 0..dim {
                    for j in 0..dim {
                        F[(i, j)] += ue[k * dim + i] * gp_u[k * dim + j];
                    }
                }
            }
            let dJ = F.determinant();
            let iF = F.try_inverse().unwrap_or_else(|| DMatrix::<f64>::identity(dim, dim));
            let FT = iF.transpose();

            let mut pres = 0.0;
            for k in 0..n_dp {
                pres += pe[k] * phi_p[k];
            }

            // K_uu: tangent stiffness
            // C_{iIjJ} = μ·δ_{ij}·δ_{IJ} + p·F^{-T}_{jI}·F^{-T}_{iJ}
            for a in 0..n_du {
                for i in 0..dim {
                    let row = a * dim + i;
                    for b in 0..n_du {
                        for j in 0..dim {
                            let col = b * dim + j;

                            // First term: μ·δ_{ij}·∇φ_a·∇φ_b
                            let mut v = 0.0;
                            if i == j {
                                for l in 0..dim {
                                    v += mu * gp_u[a * dim + l] * gp_u[b * dim + l];
                                }
                            }

                            // Second term: p·F^{-T}_{jI}·F^{-T}_{iJ} · ∂φ_a/∂X_I · ∂φ_b/∂X_J
                            // = p·(F^{-T}_{j,·}·∇φ_b) · (F^{-T}_{i,·}·∇φ_a)
                            let ftn: f64 = (0..dim).map(|l| FT[(i, l)] * gp_u[b * dim + l]).sum();
                            let ftl: f64 = (0..dim).map(|l| FT[(j, l)] * gp_u[a * dim + l]).sum();
                            v += pres * ftn * ftl;

                            kuu[row * n_vd + col] += v * w;
                        }
                    }
                }
            }

            // K_up: ∂R_u/∂p
            // = -∫ ψ_m · F^{-T}_{iJ} · ∂φ_a/∂X_J  dx
            for a in 0..n_du {
                for i in 0..dim {
                    let row = a * dim + i;
                    let ft_gp: f64 = (0..dim).map(|l| FT[(i, l)] * gp_u[a * dim + l]).sum();
                    for m in 0..n_dp {
                        kup[row * n_dp + m] -= w * ft_gp * phi_p[m];
                    }
                }
            }

            // K_pu: ∂R_p/∂u
            // = ∫ J · F^{-T}_{jJ} · ∂φ_b/∂X_J · ψ_m  dx
            for m in 0..n_dp {
                for b in 0..n_du {
                    for j in 0..dim {
                        let col = b * dim + j;
                        let ft_gp: f64 = (0..dim).map(|l| FT[(j, l)] * gp_u[b * dim + l]).sum();
                        kpu[m * n_vd + col] += w * dJ * ft_gp * phi_p[m];
                    }
                }
            }
        }

        // Scatter element blocks to global COO
        for a in 0..n_vd {
            let gi = eu[a] as usize;
            for b in 0..n_vd {
                let gj = eu[b] as usize;
                coo.add(gi, gj, kuu[a * n_vd + b]);
            }
        }
        for a in 0..n_vd {
            let gi = eu[a] as usize;
            for m in 0..n_dp {
                let gj = ep[m] as usize;
                coo.add(gi, nu + gj, kup[a * n_dp + m]);
            }
        }
        for m in 0..n_dp {
            let gi = ep[m] as usize;
            for b in 0..n_vd {
                let gj = eu[b] as usize;
                coo.add(nu + gi, gj, kpu[m * n_vd + b]);
            }
        }
    }

    let mut flat = coo.into_csr();

    // Apply Dirichlet row-zeroing to enforce essential BCs
    let mut diag_scratch = vec![0.0_f64; nt];
    for &(dof, _) in du {
        flat.apply_dirichlet_row_zeroing(dof, 0.0, &mut diag_scratch);
    }

    // Wrap into BlockMatrix
    let block_sizes = vec![nu, np];
    let mut bm = BlockMatrix::new_square(block_sizes.clone());

    // Extract K_uu block (rows 0..nu, cols 0..nu)
    let mut coo_uu = CooMatrix::new(nu, nu);
    for i in 0..nu {
        for p in flat.row_ptr[i]..flat.row_ptr[i + 1] {
            let c = flat.col_idx[p] as usize;
            if c < nu {
                coo_uu.add(i, c, flat.values[p]);
            }
        }
    }
    bm.set(0, 0, coo_uu.into_csr());

    // Extract K_up block (rows 0..nu, cols nu..nu+np)
    let mut coo_up = CooMatrix::new(nu, np);
    for i in 0..nu {
        for p in flat.row_ptr[i]..flat.row_ptr[i + 1] {
            let c = flat.col_idx[p] as usize;
            if c >= nu && c < nu + np {
                coo_up.add(i, c - nu, flat.values[p]);
            }
        }
    }
    bm.set(0, 1, coo_up.into_csr());

    // Extract K_pu block (rows nu..nu+np, cols 0..nu)
    let mut coo_pu = CooMatrix::new(np, nu);
    for i in nu..nu + np {
        for p in flat.row_ptr[i]..flat.row_ptr[i + 1] {
            let c = flat.col_idx[p] as usize;
            if c < nu {
                coo_pu.add(i - nu, c, flat.values[p]);
            }
        }
    }
    bm.set(1, 0, coo_pu.into_csr());

    (block_sizes, bm)
}

// ─── Output ────────────────────────────────────────────────────────────

/// Write deformed mesh (MFEM v1.0 format) with displaced node coordinates.
///
/// Matches C++: mesh->SwapNodes(nodes) → mesh->Print(mesh_ofs).
///
/// Caveat: assumes 1:1 correspondence between mesh nodes and displacement
/// DOFs.  Correct for linear (P1) elements; for higher-order elements only
/// corner node positions will be correct.
fn write_deformed_mesh(
    mesh: &impl MeshTopology,
    u: &[f64],
    dim: usize,
    ns: usize,
    path: &str,
) {
    let nn = mesh.n_nodes() as usize;

    // Current position = reference + displacement.
    // Displacement at node i: u[i] (x-comp), u[ns + i] (y-comp).
    let mut coords = Vec::with_capacity(nn * dim);
    for n in 0..nn {
        let ref_pt = mesh.node_coords(n as u32);
        let ux = if n < ns { u[n] } else { 0.0 };
        let uy = if n + ns < u.len() { u[ns + n] } else { 0.0 };
        coords.push(ref_pt[0] + ux);
        coords.push(ref_pt[1] + uy);
    }

    let mut f = File::create(path).expect("cannot create deformed.mesh");
    writeln!(f, "MFEM mesh v1.0\n").ok();
    writeln!(f, "dimension\n{dim}\n").ok();

    // Elements
    let ne = mesh.n_elements() as usize;
    writeln!(f, "elements\n{ne}").ok();
    for e in 0..ne {
        let et = mesh.element_type(e as u32);
        let nd = mesh.element_nodes(e as u32);
        let code = match et {
            ElementType::Tri3 => 2,
            ElementType::Quad4 => 3,
            ElementType::Tet4 => 4,
            ElementType::Hex8 => 5,
            _ => panic!("unsupported element type for MFEM export"),
        };
        let tag = mesh.element_tag(e as u32);
        write!(f, "{tag} {code}").ok();
        for &n in nd {
            write!(f, " {}", n + 1).ok();
        }
        writeln!(f).ok();
    }

    // Boundary
    let nb = mesh.n_boundary_faces() as usize;
    writeln!(f, "\nboundary\n{nb}").ok();
    for b in 0..nb {
        let fnodes = mesh.face_nodes(b as u32);
        let attr = mesh.face_tag(b as u32);
        write!(f, "{attr} 1").ok(); // type 1 = SEGMENT
        for &n in fnodes {
            write!(f, " {}", n + 1).ok();
        }
        writeln!(f).ok();
    }

    // Vertices (deformed)
    writeln!(f, "\nvertices\n{nn}\n{dim}").ok();
    for i in 0..nn {
        for d in 0..dim {
            write!(f, " {:.8}", coords[i * dim + d]).ok();
        }
        writeln!(f).ok();
    }
}

/// Write a solution vector as a simple ASCII file.
/// Format: first line = n_dofs, then one value per line.
fn write_solution(values: &[f64], path: &str) {
    let mut f = File::create(path).expect("cannot create solution file");
    writeln!(f, "{}", values.len()).ok();
    for &v in values {
        writeln!(f, "{:.14e}", v).ok();
    }
}

fn main() {
    let args = Args::parse();
    println!("=== MFEM ex19: Incompressible neo-Hookean hyperelasticity ===");

    // 1. Read mesh
    let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");
    let mesh2d = mfem.mesh2d.expect("expected 2D mesh");
    let mut mesh = mesh2d;
    for _ in 0..args.refine {
        mesh = refine_uniform(&mesh);
    }
    let dim = mesh.dim() as u8;
    let order = args.order;
    let p_order = if order > 1 { order - 1 } else { 1 };

    // 2. FE spaces (Taylor-Hood: VectorH1^dim for u, H1 for p)
    let u_space = VectorH1Space::new(mesh.clone(), order, dim);
    let p_space = H1Space::new(mesh.clone(), p_order);
    let nu = u_space.n_dofs();
    let np = p_space.n_dofs();
    let ns = u_space.n_scalar_dofs(); // scalar DOFs per component
    println!("dim(u) = {nu}");
    println!("dim(p) = {np}");
    println!("dim(u+p) = {}", nu + np);

    // 3. Dirichlet BCs (matching MFEM ex19)
    //    Attr 1: fixed (u=0). Attr 2: u_x=0, u_y=0.25*x
    let dm = u_space.scalar_dof_manager();
    let attr1 = boundary_dofs(u_space.mesh(), dm, &[1]);
    let attr2 = boundary_dofs(u_space.mesh(), dm, &[2]);
    let mut du: Vec<(usize, f64)> = Vec::new();
    for &d in &attr1 {
        // Both components zero
        du.push((d as usize, 0.0));
        du.push((d as usize + ns, 0.0));
    }
    for &d in &attr2 {
        let x = dm.dof_coord(d as u32)[0]; // x-coordinate
        du.push((d as usize, 0.0));         // u_x = 0
        du.push((d as usize + ns, 0.25 * x)); // u_y = 0.25*x
    }

    // 4. Initial guess: InitialDeformation = ReferenceConfiguration + shear
    //    u(x) = x_def - x_ref  ->  u_x = 0, u_y = 0.25*x[0]
    let mut u = vec![0.0_f64; nu];
    let mut p = vec![0.0_f64; np];
    for s in 0..ns {
        let xc = dm.dof_coord(s as u32);
        let x = xc[0];
        // Component-major: idx = comp * ns + s
        u[0 * ns + s] = 0.0;         // u_x = 0 (no offset from reference)
        u[1 * ns + s] = 0.25 * x;    // u_y = 0.25*x
    }
    // Apply BC values to the DOF vector (essential BC elimination)
    for &(dof, val) in &du {
        u[dof] = val;
    }

    println!("Initial guess set. DOFs: displacement={nu}, pressure={np}");

    // 5. Pre-compute element DOF tables
    let ne = mesh.n_elements() as usize;
    let elem_dofs_u: Vec<Vec<usize>> = (0..ne)
        .map(|e| u_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
        .collect();
    let elem_dofs_p: Vec<Vec<usize>> = (0..ne)
        .map(|e| p_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
        .collect();

    // 6. Quadrature order
    let quad_order = 2 * order + 3;

    // 7. Pressure mass matrix (built once, used in preconditioner)
    let p_mass = build_pressure_mass(mesh.clone(), p_order, quad_order, np);

    // 8. Initial residual
    let mut ru = vec![0.0_f64; nu];
    let mut rp = vec![0.0_f64; np];
    residual(&mesh, dim as usize, order, p_order, quad_order, args.mu,
             &u, &p, &elem_dofs_u, &elem_dofs_p, &mut ru, &mut rp);
    for &(dof, _) in &du { ru[dof] = 0.0; }

    let r0 = nr(&[ru.as_slice(), rp.as_slice()].concat());
    println!("Newton 0 ||r|| = {r0:.5e}");
    if r0 < args.abs_tol {
        println!("Initial residual below absolute tolerance, skipping Newton.");
        return;
    }

    // 9. Newton loop
    // MFEM: J_gmres rtol=1e-12, atol=1e-12, max_iter=300
    let inner_cfg = SolverConfig {
        rtol: 1e-12,
        atol: 1e-12,
        max_iter: 300,
        verbose: false,
        ..SolverConfig::default()
    };
    // MFEM: stiff_pcg rel/abs tol=1e-8, max_iter=200
    let k_cfg = SolverConfig {
        rtol: 1e-8,
        atol: 1e-8,
        max_iter: 200,
        verbose: false,
        ..SolverConfig::default()
    };
    // MFEM: mass_pcg rel/abs tol=1e-12, max_iter=200
    let s_cfg = SolverConfig {
        rtol: 1e-12,
        atol: 1e-12,
        max_iter: 200,
        verbose: false,
        ..SolverConfig::default()
    };
    let gamma = 1e-5;

    let mut converged = false;
    for it in 1..=args.max_iter {
        // Assemble Jacobian
        let (_sizes, jac) = jacobian(
            &mesh, dim as usize, order, p_order, quad_order, args.mu,
            &u, &p, &elem_dofs_u, &elem_dofs_p, nu, np, &du,
        );

        // Build flat system matrix (full CSR for GMRES)
        let mut coo_flat = CooMatrix::new(nu + np, nu + np);
        for bi in 0..2 {
            for bj in 0..2 {
                if let Some(mat) = jac.get(bi, bj) {
                    let row_off = if bi == 0 { 0 } else { nu };
                    let col_off = if bj == 0 { 0 } else { nu };
                    for i in 0..mat.nrows {
                        for p in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                            coo_flat.add(
                                row_off + i,
                                col_off + mat.col_idx[p] as usize,
                                mat.values[p],
                            );
                        }
                    }
                }
            }
        }
        let flat_mat = coo_flat.into_csr();

        // RHS = -residual
        let mut rhs = vec![0.0_f64; nu + np];
        for i in 0..nu { rhs[i] = -ru[i]; }
        for i in 0..np { rhs[nu + i] = -rp[i]; }

        // Block preconditioner (matching MFEM JacobianPreconditioner):
        //   z_p =  gamma * M_p^{-1} * r_p
        //   z_u = K_uu^{-1} * (r_u - K_up * z_p)
        let kuu = jac.get(0, 0).cloned().unwrap_or_else(|| {
            CooMatrix::new(nu, nu).into_csr()
        });
        let kup = jac.get(0, 1).cloned().unwrap_or_else(|| {
            CooMatrix::new(nu, np).into_csr()
        });
        let mp = p_mass.clone();

        let s_cfg_inner = s_cfg.clone();
        let k_cfg_inner = k_cfg.clone();
        let precond = move |r: &[f64], z: &mut [f64]| {
            // Pressure block: z_p = gamma * M_p^{-1} * r_p
            let mut zp = vec![0.0_f64; np];
            let _ = solve_pcg_gssmoother(&mp, &r[nu..], &mut zp, &s_cfg_inner);
            for i in 0..np {
                z[nu + i] = gamma * zp[i];
            }

            // Displacement block: z_u = K_uu^{-1} * (r_u - K_up * z_p)
            let mut kup_zp = vec![0.0_f64; nu];
            kup.spmv(&z[nu..], &mut kup_zp);
            let mut rhs_u = vec![0.0_f64; nu];
            for i in 0..nu {
                rhs_u[i] = r[i] - kup_zp[i];
            }

            let mut zu = vec![0.0_f64; nu];
            let _ = solve_gmres_gssmoother(&kuu, &rhs_u, &mut zu, 200, &k_cfg_inner);
            for i in 0..nu {
                z[i] = zu[i];
            }
        };

        // Solve: J * dx = -R
        let mut dx = vec![0.0_f64; nu + np];
        let result = right_preconditioned_gmres(
            &flat_mat, &rhs, &mut dx, 30, &inner_cfg, &precond,
        );

        match &result {
            Ok(r) => println!("  GMRES: {} its, res={:.3e}", r.iterations, r.final_residual),
            Err(e) => eprintln!("  GMRES error: {e}"),
        }

        // Damped Newton with backtracking line search
        let r_norm0 = nr(&[ru.as_slice(), rp.as_slice()].concat());
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ls in 0..8 {
            let mut u_trial = u.clone();
            let mut p_trial = p.clone();
            for i in 0..nu { u_trial[i] += alpha * dx[i]; }
            for i in 0..np { p_trial[i] += alpha * dx[nu + i]; }

            // Re-apply BCs (essential BC enforcement)
            for &(dof, val) in &du { u_trial[dof] = val; }

            let mut ru_t = vec![0.0_f64; nu];
            let mut rp_t = vec![0.0_f64; np];
            residual(&mesh, dim as usize, order, p_order, quad_order, args.mu,
                     &u_trial, &p_trial, &elem_dofs_u, &elem_dofs_p,
                     &mut ru_t, &mut rp_t);
            for &(dof, _) in &du { ru_t[dof] = 0.0; }

            let r_new = nr(&[ru_t.as_slice(), rp_t.as_slice()].concat());
            if r_new < r_norm0 * (1.0 - 1e-4 * alpha) {
                // Accept step
                u.copy_from_slice(&u_trial);
                p.copy_from_slice(&p_trial);
                ru.copy_from_slice(&ru_t);
                rp.copy_from_slice(&rp_t);
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if !accepted {
            // Even with alpha=1/128, no decrease — accept the full Newton step anyway
            for i in 0..nu { u[i] += dx[i]; }
            for i in 0..np { p[i] += dx[nu + i]; }
            for &(dof, val) in &du { u[dof] = val; }
            // Recompute residual
            residual(&mesh, dim as usize, order, p_order, quad_order, args.mu,
                     &u, &p, &elem_dofs_u, &elem_dofs_p, &mut ru, &mut rp);
            for &(dof, _) in &du { ru[dof] = 0.0; }
        }

        let r_norm = nr(&[ru.as_slice(), rp.as_slice()].concat());
        println!("Newton {it:2} ||r|| = {r_norm:.5e}  r/r0 = {:.6}", r_norm / r0);

        if r_norm < args.abs_tol || r_norm < r0 * args.rel_tol {
            println!("Newton converged in {it} iterations.");
            converged = true;
            break;
        }
    }
    // MFEM: MFEM_VERIFY(newton_solver.GetConverged(), ...)
    assert!(converged, "Newton solver did not converge in {} iterations (final ||r||={:.3e}, rtol={}, atol={})",
            args.max_iter, nr(&[ru.as_slice(), rp.as_slice()].concat()),
            args.rel_tol, args.abs_tol);

    // 10. Save output
    // Deformed mesh
    write_deformed_mesh(&mesh, &u, dim as usize, ns, "deformed.mesh");
    println!("  Wrote deformed.mesh");

    // Pressure solution
    write_solution(&p, "pressure.sol");
    println!("  Wrote pressure.sol");

    // Deformation (relative displacement = u - u_ref, where u_ref = 0)
    let mut u_def = vec![0.0_f64; nu];
    for i in 0..nu {
        u_def[i] = u[i];
    }
    write_solution(&u_def, "deformation.sol");
    println!("  Wrote deformation.sol");
}

struct Args {
    mesh: String,
    refine: usize,
    order: u8,
    mu: f64,
    rel_tol: f64,
    abs_tol: f64,
    max_iter: usize,
    #[allow(dead_code)]
    visualization: bool,
}

impl Args {
    fn parse() -> Self {
        let mut a = Self {
            mesh: "data/beam-quad.mesh".into(),
            refine: 0,
            order: 2,
            mu: 1.0,
            rel_tol: 1e-4,
            abs_tol: 1e-6,
            max_iter: 500,
            visualization: true,  // MFEM defaults: enabled
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" => a.mesh = it.next().unwrap_or_default(),
                "-r" => a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
                "-o" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2),
                "-mu" => a.mu = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
                "-rel" => a.rel_tol = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-4),
                "-abs" => a.abs_tol = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-6),
                "-it" => a.max_iter = it.next().and_then(|v| v.parse().ok()).unwrap_or(500),
                "-vis" => a.visualization = true,
                "-no-vis" => a.visualization = false,
                _ => {}
            }
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn zero_state_zero_residual() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let dim = 2;
        let order = 1;
        let p_order = 1;
        let mu = 1.0;

        let u_space = VectorH1Space::new(mesh.clone(), order, dim);
        let p_space = H1Space::new(mesh.clone(), p_order);
        let nu = u_space.n_dofs();
        let np = p_space.n_dofs();
        let ns = u_space.n_scalar_dofs();

        // BC: all boundaries fixed (u=0) so zero displacement is equilibrium
        let dm = u_space.scalar_dof_manager();
        let all_bdr = boundary_dofs(u_space.mesh(), dm, &[1, 2, 3, 4]);
        let mut du: Vec<(usize, f64)> = Vec::new();
        for &d in &all_bdr {
            du.push((d as usize, 0.0));
            du.push((d as usize + ns, 0.0));
        }

        // Initial guess: zero
        let mut u = vec![0.0_f64; nu];
        let mut p = vec![0.0_f64; np];
        for &(dof, val) in &du { u[dof] = val; }

        let ne = mesh.n_elements() as usize;
        let elem_dofs_u: Vec<Vec<usize>> = (0..ne)
            .map(|e| u_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
            .collect();
        let elem_dofs_p: Vec<Vec<usize>> = (0..ne)
            .map(|e| p_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
            .collect();
        let quad_order = 5;

        let mut ru = vec![0.0_f64; nu];
        let mut rp = vec![0.0_f64; np];
        residual(&mesh, dim as usize, order, p_order, quad_order, mu,
                 &u, &p, &elem_dofs_u, &elem_dofs_p, &mut ru, &mut rp);
        // Zero residual at Dirichlet DOFs (reaction forces not included)
        for &(dof, _) in &du { ru[dof] = 0.0; }
        let r0 = nr(&[ru.as_slice(), rp.as_slice()].concat());
        // With all boundaries fixed and zero displacement+pressure, the internal
        // residual should be exactly zero (F=I, J-1=0)
        assert!(r0 < 1e-14, "zero state should have zero residual, got {r0}");
    }
}
