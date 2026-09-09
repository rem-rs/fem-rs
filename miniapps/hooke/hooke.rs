//! 1:1 serial port of MFEM's `miniapps/hooke` (a PAR miniapp) at `-np 1`.
//!
//! Solves a quasistatic solid mechanics problem `∇·σ(∇u) = 0` (no body
//! forces) on `Mesh::MakeCartesian3D(8, 2, 2, HEXAHEDRON, 8, 1, 1)` with a
//! `NeoHookeanMaterial` (D1 = 100, C1 = 50): the left end (attribute 5) is
//! fixed, the right end (attribute 3) gets a static displacement of 1e-2 in
//! every component.
//!
//! Ported structure (C++ file → this file):
//!   * `kernels/elasticity_kernels.hpp` + `kernel_helpers.hpp` → the
//!     per-element kernels in [`ElasticityOperator`] (`apply_kernel`,
//!     `assemble_gradient_diagonal`).  The PA/smem sum-factorization kernels
//!     are replaced by direct contractions over the same quadrature rule —
//!     identical math (same `B`/`G`/`J`/`detJ`/`w` per point).
//!   * `materials/neohookean.hpp` → [`NeoHookeanMaterial`] with
//!     `stress<T>` generic over the `fem_assembly::ad` scalar type,
//!     `gradient` (analytic tangent, used by the `use_cache` branch exactly
//!     like the C++ compiled default), and `action_of_gradient` dispatched on
//!     [`GradientType`] (`Symbolic` / `InternalFwd` dual-number AD via
//!     `fem_assembly::ad::Dual` / `FiniteDiff`).
//!   * `operators/…` → [`ElasticityOperator`] + [`ElasticityGradientOperator`].
//!   * `preconditioners/diagonal_preconditioner.hpp` → [`DiagonalPC`]
//!     (0: Diagonal / 1: BlockDiagonal Jacobi of the 3×3 nodal blocks).
//!   * MFEM `CGSolver` (relTol 1e-1, maxIter 10000, printLevel 2) inside
//!     MFEM `NewtonSolver` (relTol 1e-6, maxIter 10, printLevel 1) → local
//!     ports with byte-identical iteration output.
//!
//! Clipped vs the C++ miniapp (message + `exit(3)` where a run would differ):
//!   * `-d` device other than `cpu` (no GPU backends);
//!   * GLVis 3-D socket output and ParaView output (not ported; message only,
//!     they are post-solve outputs).
//!
//! Sample runs:
//! ```text
//! cargo run --release --example hooke -- -no-vis
//! cargo run --release --example hooke -- -pc 1 -no-vis
//! cargo run --release --example hooke -- -rs 1 -no-vis
//! ```

use fem_assembly::ad::{AdScalar, Dual};
use fem_element::lagrange::HexQ1;
use fem_element::lagrange::HexQk;
use fem_element::quadrature::hex_rule;
use fem_element::{QuadratureRule, ReferenceElement};
use fem_mesh::{refine_uniform_3d, ElementType, Mesh, MeshTopology};
use fem_space::constraints::boundary_dofs;
use fem_space::{FESpace, VectorH1Space};
use std::cell::RefCell;

const DIM: usize = 3;

type Mat3 = [[f64; DIM]; DIM];
type Mat3T<T> = [[T; DIM]; DIM];
type Tensor4 = [[[[f64; DIM]; DIM]; DIM]; DIM];

// ─── materials/gradient_type.hpp ─────────────────────────────────────────────

/// Port of MFEM `enum class GradientType` (the Enzyme variants are unavailable
/// outside C++ and are omitted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GradientType {
    /// Manually derived gradient.
    Symbolic,
    /// Native dual number forward mode (`fem_assembly::ad::Dual`).
    InternalFwd,
    /// Finite differences.
    FiniteDiff,
}

// ─── 3×3 tensor helpers (MFEM `future::tensor` equivalents) ──────────────────

fn ident<T: AdScalar>() -> Mat3T<T> {
    let mut a = [[T::zero(); DIM]; DIM];
    for i in 0..DIM {
        a[i][i] = T::of_f64(1.0);
    }
    a
}

fn t_det<T: AdScalar>(a: &Mat3T<T>) -> T {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

fn t_transpose<T: AdScalar>(a: &Mat3T<T>) -> Mat3T<T> {
    let mut r = [[T::zero(); DIM]; DIM];
    for i in 0..DIM {
        for j in 0..DIM {
            r[i][j] = a[j][i];
        }
    }
    r
}

fn t_dot(a: &Mat3, b: &Mat3) -> f64 {
    let mut s = 0.0;
    for i in 0..DIM {
        for j in 0..DIM {
            s += a[i][j] * b[i][j];
        }
    }
    s
}

fn t_matmul<T: AdScalar>(a: &Mat3T<T>, b: &Mat3T<T>) -> Mat3T<T> {
    let mut r = [[T::zero(); DIM]; DIM];
    for i in 0..DIM {
        for j in 0..DIM {
            let mut s = T::zero();
            for k in 0..DIM {
                s = s + a[i][k] * b[k][j];
            }
            r[i][j] = s;
        }
    }
    r
}

fn t_add<T: AdScalar>(a: &Mat3T<T>, b: &Mat3T<T>) -> Mat3T<T> {
    let mut r = [[T::zero(); DIM]; DIM];
    for i in 0..DIM {
        for j in 0..DIM {
            r[i][j] = a[i][j] + b[i][j];
        }
    }
    r
}

/// MFEM `dev(A) = A − tr(A)/3 · I`.
fn t_dev<T: AdScalar>(a: &Mat3T<T>) -> Mat3T<T> {
    let mut tr = T::zero();
    for i in 0..DIM {
        tr = tr + a[i][i];
    }
    let third = tr / T::of_f64(3.0);
    let mut r = [[T::zero(); DIM]; DIM];
    for i in 0..DIM {
        for j in 0..DIM {
            r[i][j] = a[i][j] - third * T::of_f64((i == j) as u8 as f64);
        }
    }
    r
}

/// Closed-form inverse of a plain 3×3 (MFEM `inv` on `tensor<real_t,3,3>`).
fn inv3(a: &Mat3) -> Mat3 {
    let c = [
        [
            a[1][1] * a[2][2] - a[1][2] * a[2][1],
            a[0][2] * a[2][1] - a[0][1] * a[2][2],
            a[0][1] * a[1][2] - a[0][2] * a[1][1],
        ],
        [
            a[1][2] * a[2][0] - a[1][0] * a[2][2],
            a[0][0] * a[2][2] - a[0][2] * a[2][0],
            a[0][2] * a[1][0] - a[0][0] * a[1][2],
        ],
        [
            a[1][0] * a[2][1] - a[1][1] * a[2][0],
            a[0][1] * a[2][0] - a[0][0] * a[2][1],
            a[0][0] * a[1][1] - a[0][1] * a[1][0],
        ],
    ];
    let det = a[0][0] * c[0][0] + a[0][1] * c[1][0] + a[0][2] * c[2][0];
    let inv_det = 1.0 / det;
    let mut r = [[0.0; DIM]; DIM];
    for i in 0..DIM {
        for j in 0..DIM {
            r[i][j] = c[i][j] * inv_det;
        }
    }
    r
}

/// Determinant of a plain 3×3.
fn det3(a: &Mat3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

// ─── materials/neohookean.hpp ────────────────────────────────────────────────

/// Port of MFEM `NeoHookeanMaterial<dim = 3, gradient_type>` (D1 = 100,
/// C1 = 50).
#[derive(Debug, Clone, Copy)]
struct NeoHookeanMaterial {
    d1: f64,
    c1: f64,
    gradient_type: GradientType,
}

impl NeoHookeanMaterial {
    /// MFEM `NeoHookeanMaterial::stress<T>` (Cauchy stress), generic so it can
    /// be evaluated with `f64` (residual / FiniteDiff) or `Dual`
    /// (InternalFwd action) — this is where the AD infrastructure is exercised.
    fn stress<T: AdScalar>(&self, dudx: &Mat3T<T>) -> Mat3T<T> {
        let i3 = ident::<T>();
        // T J = det(I + dudx);
        let jac = t_det(&t_add(dudx, &i3));
        // T p = -2.0 * D1 * J * (J - 1);
        let p = T::of_f64(-2.0) * T::of_f64(self.d1) * jac * (jac - T::of_f64(1.0));
        // auto devB = dev(dudx + transpose(dudx) + dot(dudx, transpose(dudx)));
        let b = t_add(
            &t_add(dudx, &t_transpose(dudx)),
            &t_matmul(dudx, &t_transpose(dudx)),
        );
        let dev_b = t_dev(&b);
        // auto sigma = -(p / J) * I + 2 * (C1 / pow(J, 5.0/3.0)) * devB;
        let coef = T::of_f64(self.c1) / T::powf_s(jac, 5.0 / 3.0);
        let mut sigma = [[T::zero(); DIM]; DIM];
        for i in 0..DIM {
            for j in 0..DIM {
                sigma[i][j] =
                    (T::zero() - p) / jac * i3[i][j] + T::of_f64(2.0) * coef * dev_b[i][j];
            }
        }
        sigma
    }

    /// MFEM `NeoHookeanMaterial::gradient` — the analytic material tangent
    /// `C_{ijkl}` used by the `use_cache` branch of the gradient kernel and by
    /// the diagonal assembly (independent of `gradient_type` in C++ too).
    fn gradient(&self, dudx: &Mat3) -> Tensor4 {
        let i3 = ident::<f64>();
        let f = t_add(dudx, &i3);
        let inv_f = inv3(&f);
        let dev_b = t_dev(&t_add(
            &t_add(dudx, &t_transpose(dudx)),
            &t_matmul(dudx, &t_transpose(dudx)),
        ));
        let jac = t_det(&f);
        let coef = self.c1 / jac.powf(5.0 / 3.0);
        let mut c = [[[[0.0; DIM]; DIM]; DIM]; DIM];
        for i in 0..DIM {
            for j in 0..DIM {
                for k in 0..DIM {
                    for l in 0..DIM {
                        c[i][j][k][l] = (2.0 * self.d1 * jac * (i == j) as u8 as f64
                            - (10.0 / 3.0) * coef * dev_b[i][j])
                            * inv_f[l][k]
                            + 2.0 * coef
                                * ((i == k) as u8 as f64 * f[j][l]
                                    + f[i][l] * (j == k) as u8 as f64
                                    - (2.0 / 3.0) * (i == j) as u8 as f64 * f[k][l]);
                    }
                }
            }
        }
        c
    }

    /// MFEM `action_of_gradient(dudx, ddudx) = (dσ/d∇u) : d∇u`, dispatched on
    /// the [`GradientType`] exactly like the C++ member.
    fn action_of_gradient(&self, dudx: &Mat3, ddudx: &Mat3) -> Mat3 {
        match self.gradient_type {
            GradientType::Symbolic => self.action_of_gradient_symbolic(dudx, ddudx),
            GradientType::InternalFwd => self.action_of_gradient_dual(dudx, ddudx),
            GradientType::FiniteDiff => self.action_of_gradient_finite_diff(dudx, ddudx),
        }
    }

    /// MFEM `action_of_gradient_dual`: seed the stress evaluation with the
    /// dual numbers `{dudx, ddudx}` — `fem_assembly::ad` forward mode.
    fn action_of_gradient_dual(&self, dudx: &Mat3, ddudx: &Mat3) -> Mat3 {
        let mut seeded = [[Dual::new(0.0, 0.0); DIM]; DIM];
        for i in 0..DIM {
            for j in 0..DIM {
                seeded[i][j] = Dual::new(dudx[i][j], ddudx[i][j]);
            }
        }
        let sigma = self.stress::<Dual>(&seeded);
        let mut r = [[0.0; DIM]; DIM];
        for i in 0..DIM {
            for j in 0..DIM {
                r[i][j] = sigma[i][j].gradient;
            }
        }
        r
    }

    /// MFEM `action_of_gradient_finite_diff`.
    fn action_of_gradient_finite_diff(&self, dudx: &Mat3, ddudx: &Mat3) -> Mat3 {
        let mut dp = [[0.0; DIM]; DIM];
        let mut dm = [[0.0; DIM]; DIM];
        for i in 0..DIM {
            for j in 0..DIM {
                dp[i][j] = dudx[i][j] + 1.0e-8 * ddudx[i][j];
                dm[i][j] = dudx[i][j] - 1.0e-8 * ddudx[i][j];
            }
        }
        let sp = self.stress::<f64>(&dp);
        let sm = self.stress::<f64>(&dm);
        let mut r = [[0.0; DIM]; DIM];
        for i in 0..DIM {
            for j in 0..DIM {
                r[i][j] = (sp[i][j] - sm[i][j]) / 2.0e-8;
            }
        }
        r
    }

    /// MFEM `action_of_gradient_symbolic`.
    fn action_of_gradient_symbolic(&self, du_dx: &Mat3, ddu_dx: &Mat3) -> Mat3 {
        let i3 = ident::<f64>();
        let f = t_add(du_dx, &i3);
        let inv_ft = t_transpose(&inv3(&f));
        let dev_b = t_dev(&t_add(
            &t_add(du_dx, &t_transpose(du_dx)),
            &t_matmul(du_dx, &t_transpose(du_dx)),
        ));
        let jac = t_det(&f);
        let coef = self.c1 / jac.powf(5.0 / 3.0);
        let a1 = t_dot(&inv_ft, ddu_dx);
        let a2 = t_dot(&f, ddu_dx);
        // dot(ddu_dx, transpose(F)) + dot(F, transpose(ddu_dx))
        let mut dft = [[0.0; DIM]; DIM];
        for a in 0..DIM {
            for b in 0..DIM {
                dft[a][b] = ddu_dx[a][0] * f[b][0]
                    + ddu_dx[a][1] * f[b][1]
                    + ddu_dx[a][2] * f[b][2]
                    + f[a][0] * ddu_dx[b][0]
                    + f[a][1] * ddu_dx[b][1]
                    + f[a][2] * ddu_dx[b][2];
            }
        }
        let mut r = [[0.0; DIM]; DIM];
        for i in 0..DIM {
            for j in 0..DIM {
                r[i][j] = (2.0 * self.d1 * jac * a1 - (4.0 / 3.0) * coef * a2) * i3[i][j]
                    - (10.0 / 3.0) * coef * a1 * dev_b[i][j]
                    + 2.0 * coef * dft[i][j];
            }
        }
        r
    }
}

// ─── operators/elasticity_operator.hpp + elasticity_gradient_operator.hpp ────

/// Trait matching MFEM `Operator::Mult`.
trait Operator {
    fn height(&self) -> usize;
    fn mult(&self, x: &[f64], y: &mut [f64]);
}

/// Serial replica of `ElasticityOperator` (matrix-free; serial T-vector ==
/// L-vector so the restriction/prolongation chain is the identity).
struct ElasticityOperator {
    mesh: Mesh<3>,
    space: VectorH1Space<Mesh<3>>,
    order: usize,
    /// Solution-space reference element ([`HexQk`], MFEM H1 basis).
    ref_elem: HexQk,
    /// MFEM `IntRules.Get(CUBE, 2*order+1)`.
    quad: QuadratureRule,
    material: NeoHookeanMaterial,
    ess_tdof_list: Vec<usize>,
    displaced_tdof_list: Vec<usize>,
    /// Cached state for the Newton linearization (`current_state`).
    current_state: RefCell<Vec<f64>>,
}

impl ElasticityOperator {
    fn new(mesh: Mesh<3>, order: usize) -> Self {
        let space = VectorH1Space::new(mesh.clone(), order as u8, DIM as u8);
        println!("#dofs: {}", space.n_dofs());
        let ref_elem = HexQk::new(order);
        let quad = hex_rule((2 * order + 1) as u8);
        ElasticityOperator {
            mesh,
            space,
            order,
            ref_elem,
            quad,
            material: NeoHookeanMaterial {
                d1: 100.0,
                c1: 50.0,
                gradient_type: GradientType::InternalFwd,
            },
            ess_tdof_list: Vec::new(),
            displaced_tdof_list: Vec::new(),
            current_state: RefCell::new(Vec::new()),
        }
    }

    /// MFEM `SetMaterial` (the kernel is instantiated with the material).
    fn set_material(&mut self, material: NeoHookeanMaterial) {
        self.material = material;
    }

    /// MFEM `SetEssentialAttributes` (byNODES true-dofs of the tagged
    /// boundary entities).
    fn set_essential_attributes(&mut self, tags: &[i32]) {
        let n = self.space.n_scalar_dofs();
        let dofs = boundary_dofs(&self.mesh, self.space.scalar_dof_manager(), tags);
        self.ess_tdof_list = dofs
            .iter()
            .flat_map(|&d| (0..DIM).map(move |c| c * n + d as usize))
            .collect();
    }

    /// MFEM `SetPrescribedDisplacement`.
    fn set_prescribed_displacement(&mut self, tags: &[i32]) {
        let n = self.space.n_scalar_dofs();
        let dofs = boundary_dofs(&self.mesh, self.space.scalar_dof_manager(), tags);
        self.displaced_tdof_list = dofs
            .iter()
            .flat_map(|&d| (0..DIM).map(move |c| c * n + d as usize))
            .collect();
    }

    /// Shared per-element kernel (replaces `Apply3D` / `ApplyGradient3D`).
    ///
    /// `state` is the current iterate; `dfield` (if given) is the perturbation
    /// direction.  With `use_cache = true` (MFEM's compiled default) the
    /// tangent action is computed as `ddot(C(dudx), ddudx)` from the analytic
    /// material gradient; otherwise the material's `action_of_gradient`
    /// (the AD path selected by `GradientType`) is used.
    fn apply_kernel(&self, state: &[f64], dfield: Option<&[f64]>, use_cache: bool, out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        let nq = self.quad.points.len();
        let mut grads_ref = vec![0.0_f64; self.ref_elem.n_dofs() * DIM];
        let mut geo_grads = vec![0.0_f64; 8 * DIM];
        let geo = HexQ1;

        for e in 0..self.mesh.n_elements() as u32 {
            let vdofs = self.space.element_dofs(e);
            let nd = vdofs.len() / DIM;
            let mut elx = vec![0.0_f64; nd * DIM];
            for (k, &d) in vdofs.iter().enumerate() {
                elx[k] = state[d as usize];
            }
            let eld: Vec<f64> = match dfield {
                Some(df) => vdofs.iter().map(|&d| df[d as usize]).collect(),
                None => Vec::new(),
            };
            let nodes = self.mesh.element_nodes(e);

            for q in 0..nq {
                let xi = &self.quad.points[q];
                let w = self.quad.weights[q];
                self.ref_elem.eval_grad_basis(xi, &mut grads_ref);
                geo.eval_grad_basis(xi, &mut geo_grads);
                let mut jac = [[0.0_f64; DIM]; DIM];
                for k in 0..8 {
                    let xk = self.mesh.geom_coords_of(nodes[k]);
                    for i in 0..DIM {
                        for d in 0..DIM {
                            jac[i][d] += xk[i] * geo_grads[k * DIM + d];
                        }
                    }
                }
                let det_j = det3(&jac);
                let inv_j = inv3(&jac);

                // dudxi[c][j] = Σ_i elx[i][c] · dφ_i/dξ_j  (element vdofs are
                // interleaved (x0,y0,z0,x1,…), matching VectorH1Space).
                let mut dudxi = [[0.0_f64; DIM]; DIM];
                for c in 0..DIM {
                    for j in 0..DIM {
                        let mut s = 0.0;
                        for i in 0..nd {
                            s += elx[i * DIM + c] * grads_ref[i * DIM + j];
                        }
                        dudxi[c][j] = s;
                    }
                }
                // dudx = dudxi * invJ
                let mut dudx = [[0.0_f64; DIM]; DIM];
                for c in 0..DIM {
                    for l in 0..DIM {
                        let mut s = 0.0;
                        for k in 0..DIM {
                            s += dudxi[c][k] * inv_j[k][l];
                        }
                        dudx[c][l] = s;
                    }
                }

                // The quadrature function operation A_Q(x):
                //   residual:  f = invJ·σ(dudx)·detJ·w
                //   gradient:  f = invJ·(dσ/d∇u : ddudx)·detJ·w
                let mut f = [[0.0_f64; DIM]; DIM];
                if dfield.is_none() {
                    let sigma = self.material.stress::<f64>(&dudx);
                    for i in 0..DIM {
                        for l in 0..DIM {
                            let mut s = 0.0;
                            for k in 0..DIM {
                                s += inv_j[i][k] * sigma[k][l];
                            }
                            f[i][l] = s * det_j * w;
                        }
                    }
                } else {
                    let mut ddudxi = [[0.0_f64; DIM]; DIM];
                    for c in 0..DIM {
                        for j in 0..DIM {
                            let mut s = 0.0;
                            for i in 0..nd {
                                s += eld[i * DIM + c] * grads_ref[i * DIM + j];
                            }
                            ddudxi[c][j] = s;
                        }
                    }
                    let mut ddudx = [[0.0_f64; DIM]; DIM];
                    for c in 0..DIM {
                        for l in 0..DIM {
                            let mut s = 0.0;
                            for k in 0..DIM {
                                s += ddudxi[c][k] * inv_j[k][l];
                            }
                            ddudx[c][l] = s;
                        }
                    }
                    let dsigma = if use_cache {
                        // C = dσ/d∇u; dsigma = ddot(C, ddudx)
                        let c4 = self.material.gradient(&dudx);
                        let mut ds = [[0.0_f64; DIM]; DIM];
                        for i in 0..DIM {
                            for j in 0..DIM {
                                let mut s = 0.0;
                                for k in 0..DIM {
                                    for l in 0..DIM {
                                        s += c4[k][l][i][j] * ddudx[k][l];
                                    }
                                }
                                ds[i][j] = s;
                            }
                        }
                        ds
                    } else {
                        self.material.action_of_gradient(&dudx, &ddudx)
                    };
                    for i in 0..DIM {
                        for l in 0..DIM {
                            let mut s = 0.0;
                            for k in 0..DIM {
                                s += inv_j[i][k] * dsigma[k][l];
                            }
                            f[i][l] = s * det_j * w;
                        }
                    }
                }

                // ely[i][c] += Σ_j f[c][j] · dφ_i/dξ_j
                for i in 0..nd {
                    for c in 0..DIM {
                        let mut s = 0.0;
                        for j in 0..DIM {
                            s += f[c][j] * grads_ref[i * DIM + j];
                        }
                        out[vdofs[i * DIM + c] as usize] += s;
                    }
                }
            }
        }
    }

    /// MFEM `ElasticityOperator::Mult` — residual with essential dofs zeroed.
    fn residual(&self, x: &[f64], y: &mut [f64]) {
        self.apply_kernel(x, None, false, y);
        for &d in &self.ess_tdof_list {
            y[d] = 0.0;
        }
    }

    /// MFEM `ElasticityOperator::GradientMult` — `K(U)·dX` with essential
    /// column elimination.
    fn gradient_mult(&self, dx: &[f64], y: &mut [f64]) {
        let mut dx_ess = dx.to_vec();
        for &d in &self.ess_tdof_list {
            dx_ess[d] = 0.0;
        }
        // use_cache = true (MFEM compiled default): analytic C tensor.
        self.apply_kernel(&self.current_state.borrow(), Some(&dx_ess), true, y);
        // Re-assign the essential degrees of freedom.
        for &d in &self.ess_tdof_list {
            y[d] = dx[d];
        }
    }

    /// MFEM `ElasticityOperator::AssembleGradientDiagonal` — 3×3 nodal blocks
    /// with layout `(s*DIM + l)*DIM + m`; essential blocks set to identity.
    fn assemble_gradient_diagonal(&self, k_diag: &mut Vec<f64>) {
        let ns = self.space.n_dofs() / DIM;
        k_diag.clear();
        k_diag.resize(ns * DIM * DIM, 0.0);
        let nq = self.quad.points.len();
        let mut grads_ref = vec![0.0_f64; self.ref_elem.n_dofs() * DIM];
        let mut geo_grads = vec![0.0_f64; 8 * DIM];
        let geo = HexQ1;

        for e in 0..self.mesh.n_elements() as u32 {
            let vdofs = self.space.element_dofs(e);
            let nd = vdofs.len() / DIM;
            let scalar_dofs = self.space.scalar_dof_manager().element_dofs(e);
            let state = self.current_state.borrow();
            let mut elx = vec![0.0_f64; nd * DIM];
            for (k, &d) in vdofs.iter().enumerate() {
                elx[k] = state[d as usize];
            }
            let nodes = self.mesh.element_nodes(e);

            for q in 0..nq {
                let xi = &self.quad.points[q];
                let w = self.quad.weights[q];
                self.ref_elem.eval_grad_basis(xi, &mut grads_ref);
                geo.eval_grad_basis(xi, &mut geo_grads);
                let mut jac = [[0.0_f64; DIM]; DIM];
                for k in 0..8 {
                    let xk = self.mesh.geom_coords_of(nodes[k]);
                    for i in 0..DIM {
                        for d in 0..DIM {
                            jac[i][d] += xk[i] * geo_grads[k * DIM + d];
                        }
                    }
                }
                let det_j = det3(&jac);
                let inv_j = inv3(&jac);
                let jxw = w * det_j;

                let mut dudxi = [[0.0_f64; DIM]; DIM];
                for c in 0..DIM {
                    for j in 0..DIM {
                        let mut s = 0.0;
                        for i in 0..nd {
                            s += elx[i * DIM + c] * grads_ref[i * DIM + j];
                        }
                        dudxi[c][j] = s;
                    }
                }
                let mut dudx = [[0.0_f64; DIM]; DIM];
                for c in 0..DIM {
                    for l in 0..DIM {
                        let mut s = 0.0;
                        for k in 0..DIM {
                            s += dudxi[c][k] * inv_j[k][l];
                        }
                        dudx[c][l] = s;
                    }
                }
                // AssembleGradientDiagonal3D: dsigma_ddudx = material.gradient(dudx)
                let c4 = self.material.gradient(&dudx);
                for i in 0..nd {
                    // dphidx(i) = physical gradient of φ_i (dshape·invJ row i)
                    for l in 0..DIM {
                        for m in 0..DIM {
                            let mut s = 0.0;
                            for a in 0..DIM {
                                // dφ_i/dx_a
                                let mut gp = 0.0;
                                for d in 0..DIM {
                                    gp += grads_ref[i * DIM + d] * inv_j[d][a];
                                }
                                for b in 0..DIM {
                                    let mut gm = 0.0;
                                    for d in 0..DIM {
                                        gm += grads_ref[i * DIM + d] * inv_j[d][b];
                                    }
                                    s += gp * c4[a][b][l][m] * gm;
                                }
                            }
                            let sd = scalar_dofs[i] as usize;
                            k_diag[(sd * DIM + l) * DIM + m] += jxw * s;
                        }
                    }
                }
            }
        }

        // Essential dofs: identity block, zero row/column.
        for &idx in &self.ess_tdof_list {
            let submat = idx % ns;
            let row = idx / ns;
            for j in 0..DIM {
                if row == j {
                    k_diag[(submat * DIM + row) * DIM + j] = 1.0;
                } else {
                    k_diag[(submat * DIM + row) * DIM + j] = 0.0;
                    k_diag[(submat * DIM + j) * DIM + row] = 0.0;
                }
            }
        }
    }

    /// MFEM `GetGradient` — cache the state, return the gradient operator.
    fn get_gradient(&self, x: &[f64]) -> ElasticityGradientOperator<'_> {
        let mut st = self.current_state.borrow_mut();
        if st.len() != x.len() {
            *st = x.to_vec();
        } else {
            st.copy_from_slice(x);
        }
        drop(st);
        ElasticityGradientOperator { op: self }
    }
}

impl Operator for ElasticityOperator {
    fn height(&self) -> usize {
        self.space.n_dofs()
    }
    fn mult(&self, x: &[f64], y: &mut [f64]) {
        self.residual(x, y);
    }
}

/// Port of `ElasticityGradientOperator` (passes `GradientMult` through
/// `NewtonSolver` to the preconditioner / CG).
struct ElasticityGradientOperator<'a> {
    op: &'a ElasticityOperator,
}

impl Operator for ElasticityGradientOperator<'_> {
    fn height(&self) -> usize {
        self.op.space.n_dofs()
    }
    fn mult(&self, x: &[f64], y: &mut [f64]) {
        self.op.gradient_mult(x, y);
    }
}

// ─── preconditioners/diagonal_preconditioner.hpp ─────────────────────────────

/// Port of `ElasticityDiagonalPreconditioner`.
struct DiagonalPC {
    k_diag: Vec<f64>,
    ns: usize,
    block: bool,
}

impl DiagonalPC {
    fn new(block: bool) -> Self {
        DiagonalPC { k_diag: Vec::new(), ns: 0, block }
    }

    /// MFEM `SetOperator`: assemble the gradient diagonal (3×3 blocks).
    fn set_operator(&mut self, grad: &ElasticityGradientOperator) {
        grad.op.assemble_gradient_diagonal(&mut self.k_diag);
        self.ns = grad.height() / DIM;
    }

    fn mult(&self, x: &[f64], y: &mut [f64]) {
        let ns = self.ns;
        if !self.block {
            for s in 0..ns {
                for i in 0..DIM {
                    y[s + i * ns] = x[s + i * ns] / self.k_diag[(s * DIM + i) * DIM + i];
                }
            }
        } else {
            for s in 0..ns {
                let mut submat = [[0.0_f64; DIM]; DIM];
                for i in 0..DIM {
                    for j in 0..DIM {
                        submat[i][j] = self.k_diag[(s * DIM + i) * DIM + j];
                    }
                }
                let subinv = inv3(&submat);
                for i in 0..DIM {
                    let mut acc = 0.0;
                    for j in 0..DIM {
                        acc += subinv[i][j] * x[s + j * ns];
                    }
                    y[s + i * ns] = acc;
                }
            }
        }
    }
}

// ─── MFEM IterativeSolver / CGSolver / NewtonSolver (serial ports) ───────────

/// MFEM `PrintLevel` flags derived from the legacy print level.
struct PrintLevel {
    warnings: bool,
    iterations: bool,
    summary: bool,
    first_and_last: bool,
}

/// MFEM `FromLegacyPrintLevel`.
fn from_legacy_print_level(lvl: i32) -> PrintLevel {
    match lvl {
        -1 => PrintLevel { warnings: false, iterations: false, summary: false, first_and_last: false },
        0 => PrintLevel { warnings: true, iterations: false, summary: false, first_and_last: false },
        1 => PrintLevel { warnings: true, iterations: true, summary: false, first_and_last: false },
        2 => PrintLevel { warnings: true, iterations: false, summary: true, first_and_last: false },
        3 => PrintLevel { warnings: true, iterations: false, summary: false, first_and_last: true },
        _ => PrintLevel { warnings: true, iterations: false, summary: false, first_and_last: false },
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// CG solver parameters (`CGSolver` state).
struct CgParams {
    rtol: f64,
    atol: f64,
    max_iter: usize,
}

/// Port of MFEM `CGSolver::Mult` (serial, `iterative_mode = false`).
/// Returns `(converged, final_iter, sqrt(betanom))`.
// `unused_assignments` is silenced because MFEM's control flow lets `betanom`
// keep its last loop value across every break/return path, so the seeding
// assignment is deliberately never read.
#[allow(unused_assignments)]
#[allow(clippy::too_many_arguments)]
fn cg_mult(
    a: &dyn Operator,
    b: &[f64],
    x: &mut [f64],
    prec: &DiagonalPC,
    rtol: f64,
    atol: f64,
    max_iter: usize,
    pl: &PrintLevel,
) -> (bool, usize, f64) {
    let n = a.height();
    // iterative_mode == false: r = b; x = 0.0;
    let mut r = b.to_vec();
    for v in x.iter_mut() {
        *v = 0.0;
    }
    let mut z = vec![0.0_f64; n];
    prec.mult(&r, &mut z); // z = B r
    let mut d = z.clone();
    let mut nom = dot(&d, &r);
    let nom0 = nom;
    if pl.iterations || pl.first_and_last {
        print!(
            "   Iteration : {:>3}  (B r, r) = {}{}",
            0,
            fmt_g6(nom),
            if pl.first_and_last { " ...\n" } else { "\n" }
        );
    }

    if nom < 0.0 {
        if pl.warnings {
            println!(
                "PCG: The preconditioner is not positive definite. (Br, r) = {}",
                fmt_g6(nom)
            );
        }
        return (false, 0, nom);
    }
    let r0 = (nom * rtol * rtol).max(atol * atol);
    if nom <= r0 {
        // MFEM returns here without printing the summary.
        return (true, 0, nom.sqrt());
    }

    // z = A d
    let mut den;
    a.mult(&d, &mut z);
    den = dot(&z, &d);

    let mut converged = false;
    let mut final_iter = max_iter;
    let mut betanom = nom;
    let mut i = 1_usize;
    loop {
        let alpha = nom / den;
        for k in 0..n {
            x[k] += alpha * d[k];
            r[k] -= alpha * z[k];
        }
        prec.mult(&r, &mut z);
        betanom = dot(&r, &z);
        if betanom < 0.0 {
            if pl.warnings {
                println!(
                    "PCG: The preconditioner is not positive definite. (Br, r) = {}",
                    fmt_g6(betanom)
                );
            }
            final_iter = i;
            break;
        }
        if pl.iterations {
            println!("   Iteration : {:>3}  (B r, r) = {}", i, fmt_g6(betanom));
        }
        if betanom <= r0 {
            converged = true;
            final_iter = i;
            break;
        }
        i += 1;
        if i > max_iter {
            break;
        }
        let beta = betanom / nom;
        for k in 0..n {
            d[k] = z[k] + beta * d[k];
        }
        a.mult(&d, &mut z);
        den = dot(&d, &z);
        if den == 0.0 {
            final_iter = i;
            break;
        }
        nom = betanom;
    }

    if pl.first_and_last && !pl.iterations {
        println!("   Iteration : {:>3}  (B r, r) = {}", final_iter, fmt_g6(betanom));
    }
    if pl.summary || (pl.warnings && !converged) {
        println!("PCG: Number of iterations: {}", final_iter);
    }
    if pl.summary || pl.iterations || pl.first_and_last {
        let arf = (betanom / nom0).powf(0.5 / final_iter as f64);
        println!("Average reduction factor = {}", fmt_g6(arf));
    }
    if pl.warnings && !converged {
        println!("PCG: No convergence!");
    }
    (converged, final_iter, betanom.sqrt())
}

/// Port of MFEM `NewtonSolver::Mult` (serial, `iterative_mode = true`,
/// `ComputeScalingFactor == 1`).  `b` is the empty "zero" vector in both
/// miniapps, so `have_b == false`.
#[allow(clippy::too_many_arguments)]
fn newton_mult(
    oper: &ElasticityOperator,
    x: &mut [f64],
    pc: &mut DiagonalPC,
    rel_tol: f64,
    abs_tol: f64,
    max_iter: usize,
    pl: &PrintLevel,
    cg: (&CgParams, &PrintLevel),
) -> (bool, usize, f64) {
    let n = oper.height();
    let mut r = vec![0.0_f64; n];
    oper.mult(x, &mut r);
    let norm0 = norm2(&r);
    if pl.first_and_last && !pl.iterations {
        println!("Newton iteration {:>2} : ||r|| = {}...\n", 0, fmt_g6(norm0));
    }
    let norm_goal = (rel_tol * norm0).max(abs_tol);

    let mut norm = norm0;
    let mut converged = false;
    let mut it = 0_usize;
    loop {
        if pl.iterations {
            print!("Newton iteration {:>2} : ||r|| = {}", it, fmt_g6(norm));
            if it > 0 {
                print!(", ||r||/||r_0|| = {}", fmt_g6(norm / norm0));
            }
            println!();
        }
        if norm <= norm_goal {
            converged = true;
            break;
        }
        if it >= max_iter {
            break;
        }

        let grad = oper.get_gradient(x);
        pc.set_operator(&grad);
        let mut c = vec![0.0_f64; n];
        cg_mult(&grad, &r, &mut c, pc, cg.0.rtol, cg.0.atol, cg.0.max_iter, cg.1);
        // c_scale = 1.0; x -= c
        for k in 0..n {
            x[k] -= c[k];
        }
        oper.mult(x, &mut r);
        norm = norm2(&r);
        it += 1;
    }

    if pl.summary || (!converged && pl.warnings) || pl.first_and_last {
        println!("Newton: Number of iterations: {}", it);
        println!(
            "   ||r|| = {},  ||r||/||r_0|| = {}",
            fmt_g6(norm),
            fmt_g6(norm / norm0)
        );
    }
    if !converged && (pl.summary || pl.warnings) {
        println!("Newton: No convergence!");
    }
    (converged, it, norm)
}

// ─── C++ std::ostream default formatting (%.6g) ──────────────────────────────

/// `std::ostream` with default precision 6: `%g` semantics.
fn fmt_g6(v: f64) -> String {
    fmt_g(v, 6)
}

fn fmt_g(x: f64, sig: u32) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    let exp = x.abs().log10().floor() as i32;
    if exp < -4 || exp >= sig as i32 {
        let mut e = exp;
        let mut mantissa = x / 10_f64.powi(e);
        // rounding may carry the mantissa to ±10
        if mantissa.abs() >= 10.0 {
            mantissa /= 10.0;
            e += 1;
        }
        let mut s = format!("{:.*}", (sig - 1) as usize, mantissa);
        trim_trailing_zeros(&mut s);
        format!("{s}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else {
        let decimals = ((sig as i32 - 1) - exp).max(0) as usize;
        let mut s = format!("{:.*}", decimals, x);
        trim_trailing_zeros(&mut s);
        s
    }
}

fn trim_trailing_zeros(s: &mut String) {
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
}

// ─── CLI (MFEM OptionsParser) ────────────────────────────────────────────────

struct Args {
    order: i32,
    device: String,
    diagpc_type: i32,
    serial_refinement_levels: i32,
    visualization: bool,
    paraview: bool,
}

impl Args {
    fn parse() -> Args {
        let mut a = Args {
            order: 1,
            device: "cpu".to_string(),
            diagpc_type: 0,
            serial_refinement_levels: 0,
            visualization: true,
            paraview: false,
        };
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let arg = argv[i].clone();
            let val = |i: &mut usize| -> String {
                *i += 1;
                argv.get(*i).cloned().unwrap_or_else(|| {
                    eprintln!("hooke: missing value for {arg}");
                    std::process::exit(3);
                })
            };
            match arg.as_str() {
                "-o" | "--order" => a.order = val(&mut i).parse().unwrap_or(a.order),
                "-d" | "--device" => a.device = val(&mut i),
                "-pc" | "--pctype" => a.diagpc_type = val(&mut i).parse().unwrap_or(a.diagpc_type),
                "-rs" | "--ref-serial" => {
                    a.serial_refinement_levels =
                        val(&mut i).parse().unwrap_or(a.serial_refinement_levels)
                }
                "-vis" | "--visualization" => a.visualization = true,
                "-no-vis" | "--no-visualization" => a.visualization = false,
                "-pv" | "--paraview" => a.paraview = true,
                "-no-pv" | "--no-paraview" => a.paraview = false,
                other => {
                    eprintln!("hooke (Rust port): Unrecognized option: {other}");
                    std::process::exit(3);
                }
            }
            i += 1;
        }
        a
    }

    /// MFEM `OptionsParser::PrintOptions`.
    fn print_options(&self) {
        println!("Options used:");
        println!("   --order {}", self.order);
        println!("   --device {}", self.device);
        println!("   --pctype {}", self.diagpc_type);
        println!("   --ref-serial {}", self.serial_refinement_levels);
        if self.visualization {
            println!("   --visualization");
        } else {
            println!("   --no-visualization");
        }
        if self.paraview {
            println!("   --paraview");
        } else {
            println!("   --no-paraview");
        }
    }
}

fn display_banner() {
    print!(
        r#"
         ___ ___ ________   ________   ____  __.___________
        /   |   \\_____  \  \_____  \ |    |/ _|\_   _____/
       /    ~    \/   |   \  /   |   \|      <   |    __)_
       \    Y    /    |    \/    |    \    |  \  |        \
        \___|_  /\_______  /\_______  /\____|__ \/_______  /
              \/         \/         \/        \/        \/ 
      "#
    );
    println!();
}

fn main() {
    display_banner();
    let args = Args::parse();
    args.print_options();

    // Device(device_config) + device.Print(): only the CPU path is ported.
    if args.device != "cpu" {
        eprintln!(
            "hooke (Rust port): device {:?} is not ported (CPU only).",
            args.device
        );
        std::process::exit(3);
    }
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");

    // Mesh::MakeCartesian3D(8, 2, 2, Element::HEXAHEDRON, 8.0, 1.0, 1.0)
    let mut mesh = Mesh::<3>::make_cartesian_3d(8, 2, 2, ElementType::Hex8, 8.0, 1.0, 1.0, true);
    for _ in 0..args.serial_refinement_levels {
        mesh = refine_uniform_3d(&mesh);
    }

    let mut op = ElasticityOperator::new(mesh, args.order.max(1) as usize);
    // NeoHookeanMaterial<3, GradientType::InternalFwd> (D1 = 100, C1 = 50)
    op.set_material(NeoHookeanMaterial {
        d1: 100.0,
        c1: 50.0,
        gradient_type: GradientType::InternalFwd,
    });

    // Essential boundaries: attr 5 (x = 0, fixed) and attr 3 (x = sx,
    // prescribed displacement); displaced dofs on attr 3.
    op.set_essential_attributes(&[5, 3]);
    op.set_prescribed_displacement(&[3]);

    // optional AD cross-check (extension; not part of the C++ output)
    if std::env::var("HOOKE_CHECK_AD").is_ok() {
        check_ad(&op.material);
    }

    let nd = op.height();
    let mut u = vec![0.0_f64; nd];
    for &d in &op.displaced_tdof_list {
        u[d] = 1.0e-2;
    }

    let mut pc = DiagonalPC::new(args.diagpc_type == 1);

    let cg = CgParams { rtol: 1e-1, atol: 0.0, max_iter: 10000 };
    let cg_pl = from_legacy_print_level(2);

    let (converged, it, final_norm) = newton_mult(
        &op,
        &mut u,
        &mut pc,
        1e-6,
        0.0,
        10,
        &from_legacy_print_level(1),
        (&cg, &cg_pl),
    );

    println!("[check] converged = {converged} iters = {it}");
    println!("[check] final ||r|| = {:.16e}", final_norm);
    let umax = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let umin = u.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "[check] ||U|| = {:.16e} Umax = {umax:.16e} Umin = {umin:.16e}",
        norm2(&u)
    );

    // GLVis (3D) and ParaView outputs are not ported.
    if args.visualization {
        eprintln!("  GLVis 3D send is not ported (solution not visualized).");
    }
    if args.paraview {
        eprintln!("  ParaViewDataCollection output is not ported.");
    }
}

/// Extension (env `HOOKE_CHECK_AD=1`): verify the three `action_of_gradient`
/// paths (analytic Symbolic, dual-number AD InternalFwd via
/// `fem_assembly::ad`, FiniteDiff) agree at a representative quadrature state.
fn check_ad(material: &NeoHookeanMaterial) {
    let mut dudx = [[0.0_f64; DIM]; DIM];
    let mut ddudx = [[0.0_f64; DIM]; DIM];
    for (k, cell) in dudx.iter_mut().flatten().enumerate() {
        *cell = 0.01 * ((k as f64) * 0.37 - 0.9).sin();
    }
    for (k, cell) in ddudx.iter_mut().flatten().enumerate() {
        *cell = ((k as f64) * 0.61 - 0.4).cos();
    }
    let sym = material.action_of_gradient_symbolic(&dudx, &ddudx);
    let ad = material.action_of_gradient_dual(&dudx, &ddudx);
    let fd = material.action_of_gradient_finite_diff(&dudx, &ddudx);
    let mut e_ad = 0.0_f64;
    let mut e_fd = 0.0_f64;
    for i in 0..DIM {
        for j in 0..DIM {
            e_ad = e_ad.max((sym[i][j] - ad[i][j]).abs());
            e_fd = e_fd.max((sym[i][j] - fd[i][j]).abs());
        }
    }
    println!(
        "[check] AD cross-check: |symbolic - dual_ad| = {e_ad:.3e}, |symbolic - fd| = {e_fd:.3e}"
    );
}
