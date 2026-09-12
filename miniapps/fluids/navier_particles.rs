//! Tracer particles in a 2D channel flow — 1:1 port of MFEM 4.10
//! `miniapps/fluids/navier/navier_particles.{hpp,cpp}` (`mfem::navier::NavierParticles`)
//! for the 2D case, used by `navier_bifurcation`.
//!
//! The particles are one-way coupled Lagrangian tracers integrated with the
//! BDFk/EXTk scheme of Dutta & Som (2017):
//!
//! ```text
//! dv/dt = κ(u − v) − γ ê + ζ (u − v) × ω
//! dx/dt = v
//! ```
//!
//! where `x`/`v` are the particle position/velocity, `u`/`ω` the fluid
//! velocity/vorticity interpolated at the particle location (via
//! [`GslibFindPoints`], the port of `FindPointsGSLIB`), `κ` the drag, `ζ` the
//! lift and `γ ê` the body force.  Wall reflection boundary conditions
//! ([`NavierParticles::add_2d_reflection_bc`]) reflect particles that cross a
//! wall segment during a time step, and particles that leave the domain
//! (`FindPointsGSLIB` reports them as `CODE_NOT_FOUND`) are moved to the
//! inactive set.
//!
//! # Port notes
//!
//! * **2-D only**: the C++ `Step` dispatches on `ParticleSet::GetDim()` and
//!   calls `MFEM_ABORT("3D particles not yet implemented.")` otherwise; the
//!   port mirrors the 2D path of `ParticleStep2D` alone (the miniapp driving
//!   it, `navier_bifurcation`, is a 2D problem).
//! * **Data layout**: `ParticleSet` stores every field of every particle in one
//!   `ParticleVector` (`Ordering::byVDIM` for the fields added by
//!   `NavierParticles`, i.e. one particle's `(x, y)` contiguous in memory).
//!   This port keeps one `Vec<[f64; 2]>` per field, which is the same
//!   byVDIM layout with the indexing spelled out — `U(nm)[p] == U(nm)(p, :)`.
//!   The `X(0)`/`U(0)`/`V(0)`/`W(0)` "current" slots of `Step` are rotated into
//!   the history slots at the start of the step (`std::move` semantics: the
//!   rotated-from slot is left empty and resized), which is what
//!   [`NavierParticles::step`] reproduces with `std::mem::take`.
//! * **Field/tag bookkeeping**: the C++ class registers `kappa`, `zeta`,
//!   `gamma` (scalars), `u`/`v`/`w` (`N_HIST = 4` vector fields each), `x`
//!   (`N_HIST - 1 = 3` vector fields) and the `Order` tag.  Those are exactly
//!   the fields this module stores, so `PrintCSV` gets the same column order
//!   (`id,X,Y,kappa,Order` for the miniapp's `print_field_idxs = {0}`,
//!   `print_tag_idxs = {0}`).
//! * **`GslibFindPoints`** (`fem_mesh::findpts`) replaces `FindPointsGSLIB`.
//!   Only its *code* is used for the active/inactive decision
//!   (`GetPointsNotFoundIndices()` filters `code == 2`, i.e. points that were
//!   not found at all — border points, `code == 1`, stay active), and the
//!   interpolation evaluates the H¹ FE basis at the located reference
//!   coordinates, which is what gslib's `findpts_eval` does with the element's
//!   nodal values.
//! * **The `β` index of the reflection correction** (see
//!   [`NavierParticles::apply_2d_reflection_bc`]) is an out-of-bounds read in
//!   the C++ (`beta_k[Order()[i]]` with `Order()[i] == 3` on a 3-element
//!   `std::array`); the port uses the k=3 entry the expression evidently means.

use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElem};
use fem_element::ReferenceElement;
use fem_mesh::findpts::{GslibFindPoints, CODE_NOT_FOUND};
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::VectorH1Space;

/// MFEM `FindPointsGSLIB::default_interp_value` — the value
/// `findpts_eval` writes for the points it could not locate (they are moved to
/// the inactive set in the same step, right after the interpolation).
const DEFAULT_INTERP_VALUE: f64 = f64::NEG_INFINITY;

/// `NavierParticles::N_HIST` — number of time levels stored (current + 3).
const N_HIST: usize = 4;

// ─── ParticleSet ────────────────────────────────────────────────────────────

/// MFEM `ParticleSet` restricted to the fields and operations
/// `NavierParticles` uses (2-D, serial, `Ordering::byVDIM`).
///
/// `ids` is MFEM's ID bookkeeping: a serial `ParticleSet` has
/// `id_stride = 1`, `id_counter = 0`, and `AddParticles(n)` hands out the next
/// `n` consecutive IDs (they are never reused when particles are removed).
pub struct ParticleSet {
    /// `Coords()` — `X(0)`, the current particle positions.
    x: Vec<[f64; 2]>,
    /// `X(nm)`, `nm = 1..3` — the position history.
    x_hist: [Vec<[f64; 2]>; N_HIST - 1],
    /// `U(nm)` — the fluid velocity interpolated at `X(nm)`.
    u: [Vec<[f64; 2]>; N_HIST],
    /// `V(nm)` — the particle velocity at `t - nm·dt`.
    v: [Vec<[f64; 2]>; N_HIST],
    /// `W(nm)` — the fluid vorticity interpolated at `X(nm)`.
    w: [Vec<[f64; 2]>; N_HIST],
    /// Fields `kappa`, `zeta`, `gamma`.
    kappa: Vec<f64>,
    zeta: Vec<f64>,
    gamma: Vec<f64>,
    /// Tag `Order`.
    order: Vec<i32>,
    /// Particle IDs.
    ids: Vec<u64>,
    /// `id_counter` (serial: incremented by `id_stride = 1`).
    id_counter: u64,
}

impl ParticleSet {
    /// Serial `ParticleSet(num_particles, dim)` for `dim == 2` with the three
    /// scalar fields (`kappa`, `zeta`, `gamma`), the `N_HIST` `u`/`v`/`w`
    /// vector fields, the `N_HIST - 1` `x` history fields and the `Order` tag.
    pub fn new(num_particles: usize) -> Self {
        let mut ps = ParticleSet {
            x: Vec::new(),
            x_hist: Default::default(),
            u: Default::default(),
            v: Default::default(),
            w: Default::default(),
            kappa: Vec::new(),
            zeta: Vec::new(),
            gamma: Vec::new(),
            order: Vec::new(),
            ids: Vec::new(),
            id_counter: 0,
        };
        ps.reserve(num_particles);
        ps.add_particles(num_particles);
        ps
    }

    /// `ParticleSet::Reserve(n)`.
    pub fn reserve(&mut self, n: usize) {
        self.x.reserve(n);
        for h in self.x_hist.iter_mut() {
            h.reserve(n);
        }
        for j in 0..N_HIST {
            self.u[j].reserve(n);
            self.v[j].reserve(n);
            self.w[j].reserve(n);
        }
        self.kappa.reserve(n);
        self.zeta.reserve(n);
        self.gamma.reserve(n);
        self.order.reserve(n);
        self.ids.reserve(n);
    }

    /// `ParticleSet::GetNParticles()`.
    pub fn n_particles(&self) -> usize {
        self.x.len()
    }

    /// `ParticleSet::GetGlobalNParticles()` — the serial version is the local
    /// count (`MPI_Allreduce` is a no-op on one rank).
    pub fn global_n_particles(&self) -> u64 {
        self.n_particles() as u64
    }

    /// `ParticleSet::AddParticles(n, &new_indices)`: append `n` particles with
    /// fresh consecutive IDs and zero-initialised data, returning the indices
    /// of the new particles (MFEM's `new_indices`).
    ///
    /// MFEM leaves the freshly grown `ParticleVector` entries uninitialised;
    /// the caller (`SetInjectedParticles`) overwrites every field, and
    /// `NavierParticles::step` overwrites `X(0)`/`U(0)`/`V(0)`/`W(0)` of every
    /// particle, so zeros here are equivalent (and defined).
    pub fn add_particles(&mut self, n: usize) -> Vec<usize> {
        let old = self.n_particles();
        let new_np = old + n;
        let idxs: Vec<usize> = (old..new_np).collect();
        self.x.resize(new_np, [0.0; 2]);
        for h in self.x_hist.iter_mut() {
            h.resize(new_np, [0.0; 2]);
        }
        for j in 0..N_HIST {
            self.u[j].resize(new_np, [0.0; 2]);
            self.v[j].resize(new_np, [0.0; 2]);
            self.w[j].resize(new_np, [0.0; 2]);
        }
        self.kappa.resize(new_np, 0.0);
        self.zeta.resize(new_np, 0.0);
        self.gamma.resize(new_np, 0.0);
        self.order.resize(new_np, 0);
        for _ in 0..n {
            self.ids.push(self.id_counter);
            self.id_counter += 1;
        }
        idxs
    }

    /// `ParticleSet::RemoveParticles(list)`: drop the listed particles from
    /// every field and tag, keeping the relative order of the survivors.
    pub fn remove_particles(&mut self, list: &[usize]) {
        let mut drop = vec![false; self.n_particles()];
        for &i in list {
            drop[i] = true;
        }
        let keep: Vec<usize> = (0..drop.len()).filter(|&i| !drop[i]).collect();
        retain_by_index(&mut self.x, &keep);
        for h in self.x_hist.iter_mut() {
            retain_by_index(h, &keep);
        }
        for j in 0..N_HIST {
            retain_by_index(&mut self.u[j], &keep);
            retain_by_index(&mut self.v[j], &keep);
            retain_by_index(&mut self.w[j], &keep);
        }
        retain_by_index(&mut self.kappa, &keep);
        retain_by_index(&mut self.zeta, &keep);
        retain_by_index(&mut self.gamma, &keep);
        retain_by_index(&mut self.order, &keep);
        retain_by_index(&mut self.ids, &keep);
    }

    /// `Coords()` / `X(0)`.
    pub fn coords(&self) -> &[[f64; 2]] {
        &self.x
    }
    /// `Coords()` / `X(0)`, mutable.
    pub fn coords_mut(&mut self) -> &mut [[f64; 2]] {
        &mut self.x
    }
    /// `X(nm)`, `nm >= 1`.
    pub fn x_hist(&self, nm: usize) -> &[[f64; 2]] {
        &self.x_hist[nm - 1]
    }
    /// `X(nm)`, `nm >= 1`, mutable.
    pub fn x_hist_mut(&mut self, nm: usize) -> &mut [[f64; 2]] {
        &mut self.x_hist[nm - 1]
    }
    /// `U(nm)`.
    pub fn u(&self, nm: usize) -> &[[f64; 2]] {
        &self.u[nm]
    }
    /// `U(nm)`, mutable.
    pub fn u_mut(&mut self, nm: usize) -> &mut [[f64; 2]] {
        &mut self.u[nm]
    }
    /// `V(nm)`.
    pub fn v(&self, nm: usize) -> &[[f64; 2]] {
        &self.v[nm]
    }
    /// `V(nm)`, mutable.
    pub fn v_mut(&mut self, nm: usize) -> &mut [[f64; 2]] {
        &mut self.v[nm]
    }
    /// `W(nm)`.
    pub fn w(&self, nm: usize) -> &[[f64; 2]] {
        &self.w[nm]
    }
    /// `W(nm)`, mutable.
    pub fn w_mut(&mut self, nm: usize) -> &mut [[f64; 2]] {
        &mut self.w[nm]
    }
    /// `Kappa()`.
    pub fn kappa(&self) -> &[f64] {
        &self.kappa
    }
    /// `Kappa()`, mutable.
    pub fn kappa_mut(&mut self) -> &mut [f64] {
        &mut self.kappa
    }
    /// `Zeta()`.
    pub fn zeta(&self) -> &[f64] {
        &self.zeta
    }
    /// `Zeta()`, mutable.
    pub fn zeta_mut(&mut self) -> &mut [f64] {
        &mut self.zeta
    }
    /// `Gamma()`.
    pub fn gamma(&self) -> &[f64] {
        &self.gamma
    }
    /// `Gamma()`, mutable.
    pub fn gamma_mut(&mut self) -> &mut [f64] {
        &mut self.gamma
    }
    /// `Order()`.
    pub fn order(&self) -> &[i32] {
        &self.order
    }
    /// `Order()`, mutable.
    pub fn order_mut(&mut self) -> &mut [i32] {
        &mut self.order
    }

    /// `ParticleSet::PrintCSV(fname, field_idxs, tag_idxs)` with MFEM's
    /// default `precision = 16` (the serial `ParticleSet` has no `rank`
    /// column).  The caller writes the returned text to `fname`
    /// (`WriteToFile`).
    ///
    /// The column order is `id`, the coordinate components `X`,`Y`, the listed
    /// fields (`field_idxs`: 0 = `kappa`, 1 = `zeta`, 2 = `gamma`) and the
    /// listed tags (`tag_idxs`: 0 = `Order`).
    pub fn print_csv(&self, field_idxs: &[usize], tag_idxs: &[usize]) -> String {
        const FIELD_NAMES: [&str; 3] = ["kappa", "zeta", "gamma"];
        const TAG_NAMES: [&str; 1] = ["Order"];
        let mut out = String::from("id");
        for ax in ["X", "Y"] {
            out.push(',');
            out.push_str(ax);
        }
        for &f in field_idxs {
            out.push(',');
            out.push_str(FIELD_NAMES[f]);
        }
        for &t in tag_idxs {
            out.push(',');
            out.push_str(TAG_NAMES[t]);
        }
        out.push('\n');
        for i in 0..self.n_particles() {
            out.push_str(&self.ids[i].to_string());
            out.push(',');
            out.push_str(&fmt_g16(self.x[i][0]));
            out.push(',');
            out.push_str(&fmt_g16(self.x[i][1]));
            let fields = [&self.kappa, &self.zeta, &self.gamma];
            for &f in field_idxs {
                out.push(',');
                out.push_str(&fmt_g16(fields[f][i]));
            }
            let tags = [&self.order];
            for &t in tag_idxs {
                out.push(',');
                out.push_str(&tags[t][i].to_string());
            }
            out.push('\n');
        }
        out
    }
}

/// Keep only the entries of `v` listed in `keep` (MFEM `Array::DeleteAt` of
/// the complement list).
fn retain_by_index<T: Copy>(v: &mut Vec<T>, keep: &[usize]) {
    let kept: Vec<T> = keep.iter().map(|&i| v[i]).collect();
    *v = kept;
}

impl ParticleSet {
    /// `for i = N_HIST-1 … 1: U(i) = std::move(U(i-1))` for the `u`, `v` and
    /// `w` fields.  The move leaves `U(0)` empty (MFEM reallocates it with
    /// `SetSize` right after the loop; here [`Self::resize_current`] does).
    fn rotate_histories(&mut self) {
        for arr in [&mut self.u, &mut self.v, &mut self.w] {
            arr[3] = std::mem::take(&mut arr[2]);
            arr[2] = std::mem::take(&mut arr[1]);
            arr[1] = std::mem::take(&mut arr[0]);
        }
    }

    /// `for i = N_HIST-1 … 1: X(i) = std::move(X(i-1))` — `X(3) = X(2)`,
    /// `X(2) = X(1)`, `X(1) = X(0)`.  The old `X(3)` is released and `X(0)`
    /// (the coordinates) is left empty: the particle step recomputes it from
    /// the history and `V(0)`.
    fn rotate_positions(&mut self) {
        std::mem::take(&mut self.x_hist[2]);
        self.x_hist[2] = std::mem::take(&mut self.x_hist[1]);
        self.x_hist[1] = std::mem::take(&mut self.x_hist[0]);
        self.x_hist[0] = std::mem::take(&mut self.x);
    }

    /// `U(0).SetSize(U(1).Size())` etc. after [`Self::rotate_histories`] /
    /// [`Self::rotate_positions`]: re-establish the size of the current time
    /// level.  Every entry is written before it is read (`V(0)` and `X(0)` by
    /// the particle step, `U(0)`/`W(0)` by the interpolation), so zero-filling
    /// is equivalent to MFEM's uninitialised `SetSize`.
    fn resize_current(&mut self, np: usize) {
        self.x.resize(np, [0.0; 2]);
        self.u[0].resize(np, [0.0; 2]);
        self.v[0].resize(np, [0.0; 2]);
        self.w[0].resize(np, [0.0; 2]);
    }
}
// ─── NavierParticles ────────────────────────────────────────────────────────

/// 2D wall reflection boundary condition (`ReflectionBC_2D`).
#[derive(Debug, Clone, Copy)]
pub struct ReflectionBc2D {
    /// Wall segment start point.
    pub line_start: [f64; 2],
    /// Wall segment end point.
    pub line_end: [f64; 2],
    /// Boundary collision restitution constant (`1` = perfectly elastic).
    pub e: f64,
    /// `true` if the left normal points out of the domain.
    pub invert_normal: bool,
}

/// MFEM `navier::NavierParticles` (2-D).
///
/// The `finder` borrows the fluid mesh, so the particle solver must be created
/// from a mesh that outlives it (the miniapp keeps the refined mesh alive and
/// hands a clone to the flow discretization).
pub struct NavierParticles<'a> {
    particles: ParticleSet,
    inactive: ParticleSet,
    finder: GslibFindPoints<'a, 2>,
    /// `dthist` — the particle time-step history.
    dthist: [f64; 3],
    /// `beta_k` BDFk coefficients, `k = 1, 2, 3` (`beta_k[k-1][0]` is the
    /// implicit coefficient of `V(0)`).
    beta_k: [[f64; 4]; 3],
    /// `alpha_k` EXTk coefficients, `k = 1, 2, 3`.
    alpha_k: [[f64; 3]; 3],
    /// `bcs` — the boundary conditions, in the order they were added.
    bcs: Vec<ReflectionBc2D>,
    /// `FindPointsGSLIB::gsl_code` of the last search — [`Self::interpolate_uw`]
    /// leaves it in place and `DeactivateLostParticles(false)` reads it
    /// instead of searching again (the positions did not change in between).
    last_codes: Vec<u32>,
}

impl<'a> NavierParticles<'a> {
    /// `NavierParticles::NavierParticles(comm, num_particles, mesh)` (serial:
    /// no communicator; the finder is set up on `mesh`).
    pub fn new(mesh: &'a Mesh<2>, num_particles: usize) -> Self {
        NavierParticles {
            particles: ParticleSet::new(num_particles),
            inactive: ParticleSet::new(0),
            finder: GslibFindPoints::new(mesh),
            dthist: [0.0; 3],
            beta_k: [[0.0; 4]; 3],
            alpha_k: [[0.0; 3]; 3],
            bcs: Vec::new(),
            last_codes: Vec::new(),
        }
    }

    /// `NavierParticles::Setup(dt)` — `dthist[0] = dt`.
    pub fn setup(&mut self, dt: f64) {
        self.dthist[0] = dt;
    }

    /// `GetParticles()`.
    pub fn particles(&self) -> &ParticleSet {
        &self.particles
    }
    /// `GetParticles()`, mutable.
    pub fn particles_mut(&mut self) -> &mut ParticleSet {
        &mut self.particles
    }
    /// `GetInactiveParticles()`.
    pub fn inactive_particles(&self) -> &ParticleSet {
        &self.inactive
    }

    /// `Add2DReflectionBC(line_start, line_end, e, invert_normal)`.
    pub fn add_2d_reflection_bc(
        &mut self,
        line_start: [f64; 2],
        line_end: [f64; 2],
        e: f64,
        invert_normal: bool,
    ) {
        self.bcs.push(ReflectionBc2D {
            line_start,
            line_end,
            e,
            invert_normal,
        });
    }

    /// `SetTimeIntegrationCoefficients()` — the BDFk/EXTk coefficients from the
    /// current `dthist`.
    fn set_time_integration_coefficients(&mut self) {
        let rho1 = self.dthist[0] / self.dthist[1];
        let rho2 = self.dthist[1] / self.dthist[2];

        for o in 0..3 {
            if o == 0 {
                // k = 1
                self.beta_k[o] = [1.0, -1.0, 0.0, 0.0];
                self.alpha_k[o] = [1.0, 0.0, 0.0];
            } else if o == 1 {
                // k = 2
                self.beta_k[o][0] = (1.0 + 2.0 * rho1) / (1.0 + rho1);
                self.beta_k[o][1] = -(1.0 + rho1);
                self.beta_k[o][2] = rho1.powi(2) / (1.0 + rho1);
                self.beta_k[o][3] = 0.0;
                self.alpha_k[o][0] = 1.0 + rho1;
                self.alpha_k[o][1] = -rho1;
                self.alpha_k[o][2] = 0.0;
            } else {
                // k = 3
                self.beta_k[o][0] =
                    1.0 + rho1 / (1.0 + rho1) + (rho2 * rho1) / (1.0 + rho2 * (1.0 + rho1));
                self.beta_k[o][1] =
                    -1.0 - rho1 - (rho2 * rho1 * (1.0 + rho1)) / (1.0 + rho2);
                self.beta_k[o][2] = rho1.powi(2) * (rho2 + 1.0 / (1.0 + rho1));
                self.beta_k[o][3] = -(rho2.powi(3) * rho1.powi(2) * (1.0 + rho1))
                    / ((1.0 + rho2) * (1.0 + rho2 + rho2 * rho1));
                self.alpha_k[o][0] =
                    ((1.0 + rho1) * (1.0 + rho2 * (1.0 + rho1))) / (1.0 + rho2);
                self.alpha_k[o][1] = -rho1 * (1.0 + rho2 * (1.0 + rho1));
                self.alpha_k[o][2] = (rho2.powi(2) * rho1 * (1.0 + rho1)) / (1.0 + rho2);
            }
        }
    }

    /// `ParticleStep2D(dt, p)`.
    fn particle_step_2d(&mut self, dt: f64, p: usize) {
        // `int order_idx = Order()[p] - 1` (the order was incremented by the
        // caller), then the `beta`/`alpha` arrays of that BDF/EXT order.
        let order_idx = (self.particles.order()[p] - 1) as usize;
        let beta = self.beta_k[order_idx];
        let alpha = self.alpha_k[order_idx];

        let kappa = self.particles.kappa()[p];
        let zeta = self.particles.zeta()[p];
        let gamma = self.particles.gamma()[p];

        // Extrapolate the particle vorticity with EXTk:
        // `w_n_ext = alpha1*w_nm1 + alpha2*w_nm2 + alpha3*w_nm3`.
        let mut w_n_ext = 0.0;
        for j in 1..=3 {
            w_n_ext += alpha[j - 1] * self.particles.w(j)[p][0];
        }

        // The implicit 2x2 matrix B (row-major, as MFEM's DenseMatrix initializer).
        let b = [
            [beta[0] + dt * kappa, zeta * dt * w_n_ext],
            [-zeta * dt * w_n_ext, beta[0] + dt * kappa],
        ];
        let det = b[0][0] * b[1][1] - b[0][1] * b[1][0];
        let binv = [
            [b[1][1] / det, -b[0][1] / det],
            [-b[1][0] / det, b[0][0] / det],
        ];

        // RHS: `r = -Σ_j beta[j]·V(j) + dt·Σ_j alpha[j-1]·C(j)` with
        // `C = κ·U(j) - γ·ê + ζ·w_n_ext·(U(j)_y, -U(j)_x)`.
        let mut r = [0.0_f64; 2];
        for j in 1..=3 {
            let up = self.particles.u(j)[p];
            let vp = self.particles.v(j)[p];
            for c in 0..2 {
                r[c] += -beta[j] * vp[c];
            }
            let mut cc = [kappa * up[0], kappa * up[1]];
            cc[1] += -gamma * 1.0;
            let z = zeta * w_n_ext;
            cc[0] += z * up[1];
            cc[1] += z * -up[0];
            for c in 0..2 {
                r[c] += dt * alpha[j - 1] * cc[c];
            }
        }

        // `B_inv.Mult(r, vp)`: the particle velocity `V(0)`.
        let v0 = [
            binv[0][0] * r[0] + binv[0][1] * r[1],
            binv[1][0] * r[0] + binv[1][1] * r[1],
        ];
        self.particles.v_mut(0)[p] = v0;

        // `xpn = -Σ_j beta[j]·X(j) + dt·V(0)`, scaled by `1/beta[0]`.  In the
        // C++ `xpn` aliases `X(0)(p, :)` and is zeroed before the sum, so the
        // current position only enters through `X(1)`.
        let mut xpn = [0.0_f64; 2];
        for j in 1..=3 {
            let xp = self.particles.x_hist(j)[p];
            for c in 0..2 {
                xpn[c] += -beta[j] * xp[c];
            }
        }
        for c in 0..2 {
            xpn[c] += dt * v0[c];
            xpn[c] *= 1.0 / beta[0];
        }
        self.particles.coords_mut()[p] = xpn;
    }

    /// `Get2DNormal(p1, p2, inv_normal, normal)`.
    fn get_2d_normal(p1: [f64; 2], p2: [f64; 2], inv_normal: bool) -> [f64; 2] {
        let diff = [p2[0] - p1[0], p2[1] - p1[1]];
        let n = if inv_normal {
            [diff[1], -diff[0]]
        } else {
            [-diff[1], diff[0]]
        };
        let len = (n[0] * n[0] + n[1] * n[1]).sqrt();
        [n[0] / len, n[1] / len]
    }

    /// `Get2DSegmentIntersection(s1, s2, x_int, t1, t2)` — `Some((x_int, t1,
    /// t2))` when the two segments cross.
    fn get_2d_segment_intersection(
        s1_start: [f64; 2],
        s1_end: [f64; 2],
        s2_start: [f64; 2],
        s2_end: [f64; 2],
    ) -> Option<([f64; 2], f64, f64)> {
        // `r_1 = s1_start + t1*[s1_end - s1_start]`,
        // `r_2 = s2_start + t2*[s2_end - s2_start]`.
        let denom = (s1_end[0] - s1_start[0]) * (s2_start[1] - s2_end[1])
            - (s1_end[1] - s1_start[1]) * (s2_start[0] - s2_end[0]);

        // Nearly-parallel segments are not well-posed.
        let rho = denom.abs() / (dist(s1_start, s2_end) * dist(s2_start, s2_end));
        if rho < 1e-12 {
            return None;
        }

        let t1 = ((s2_start[0] - s1_start[0]) * (s2_start[1] - s2_end[1])
            - (s2_start[1] - s1_start[1]) * (s2_start[0] - s2_end[0]))
            / denom;
        let t2 = ((s1_end[0] - s1_start[0]) * (s2_start[1] - s1_start[1])
            - (s1_end[1] - s1_start[1]) * (s2_start[0] - s1_start[0]))
            / denom;

        if (0.0..=1.0).contains(&t1) && (0.0..=1.0).contains(&t2) {
            let x_int = [
                s2_start[0] + (s2_end[0] - s2_start[0]) * t2,
                s2_start[1] + (s2_end[1] - s2_start[1]) * t2,
            ];
            Some((x_int, t1, t2))
        } else {
            None
        }
    }

    /// `Apply2DReflectionBC(bc)`.
    fn apply_2d_reflection_bc(&mut self, bc: &ReflectionBc2D) {
        let normal = Self::get_2d_normal(bc.line_start, bc.line_end, bc.invert_normal);
        for i in 0..self.particles.n_particles() {
            let p_xn = self.particles.coords()[i];
            let p_xnm1 = self.particles.x_hist_mut(1)[i];
            let mut p_vn = self.particles.v(0)[i];

            if let Some((x_int, _, _)) =
                Self::get_2d_segment_intersection(bc.line_start, bc.line_end, p_xnm1, p_xn)
            {
                // Only particles that moved *into* the wall reflect (a
                // particle whose `x_nm1` sits on the wall within machine
                // precision must not bounce).
                let p_xdiff = [p_xn[0] - p_xnm1[0], p_xn[1] - p_xnm1[1]];
                if p_xdiff[0] * normal[0] + p_xdiff[1] * normal[1] > 0.0 {
                    continue;
                }

                let dt_c = dist(p_xnm1, x_int) / norm2(p_vn);

                // Correct the velocity: `v -= (1+e)(v·n) n`.
                let p_vdiff = p_vn;
                let vn = p_vn[0] * normal[0] + p_vn[1] * normal[1];
                p_vn[0] -= (1.0 + bc.e) * vn * normal[0];
                p_vn[1] -= (1.0 + bc.e) * vn * normal[1];
                let p_vdiff = [p_vdiff[0] - p_vn[0], p_vdiff[1] - p_vn[1]];
                self.particles.v_mut(0)[i] = p_vn;

                // Correct the position with the (now bounded) velocity change.
                // The C++ indexes `beta_k[Order()[i]]`; for `Order()[i] == 3`
                // that is past the end of the 3-element array, so the port uses
                // the k=3 entry `beta_k[2]` it means (`o - 1`).
                let o = self.particles.order()[i] as usize;
                let beta0 = self.beta_k[(o - 1).min(2)][0];
                let scale = (1.0 / beta0) * (dt_c - self.dthist[0]);
                let xn = &mut self.particles.coords_mut()[i];
                xn[0] += scale * p_vdiff[0];
                xn[1] += scale * p_vdiff[1];

                // Back to BDF1 on the next step.
                self.particles.order_mut()[i] = 0;
            }
        }
    }

    /// `ApplyBCs()`.
    fn apply_bcs(&mut self) {
        for bc in self.bcs.clone() {
            self.apply_2d_reflection_bc(&bc);
        }
    }

    /// `InterpolateUW(u_gf, w_gf)`: locate the particles and interpolate the
    /// fluid velocity/vorticity at their new position.
    fn interpolate_uw(
        &mut self,
        space: &VectorH1Space<Mesh<2>>,
        ref_elem: &dyn ReferenceElement,
        u_gf: &[f64],
        w_gf: &[f64],
    ) {
        let pts = self.particles.coords().to_vec();
        let found = self.finder.find_points(&pts);
        self.last_codes = found.iter().map(|f| f.code).collect();
        let n_ldofs = ref_elem.n_dofs();
        let mut phi = vec![0.0_f64; n_ldofs];
        for (p, fp) in found.iter().enumerate() {
            let e = fp.elem;
            let dofs = space.element_dofs(e);
            let mut uu = [0.0_f64; 2];
            let mut ww = [0.0_f64; 2];
            if fp.code == CODE_NOT_FOUND {
                uu = [DEFAULT_INTERP_VALUE; 2];
                ww = [DEFAULT_INTERP_VALUE; 2];
            } else {
                ref_elem.eval_basis(&fp.xi, &mut phi);
                for (k, _) in phi.iter().enumerate() {
                    uu[0] += u_gf[dofs[k * 2] as usize] * phi[k];
                    uu[1] += u_gf[dofs[k * 2 + 1] as usize] * phi[k];
                    ww[0] += w_gf[dofs[k * 2] as usize] * phi[k];
                    ww[1] += w_gf[dofs[k * 2 + 1] as usize] * phi[k];
                }
            }
            self.particles.u_mut(0)[p] = uu;
            self.particles.w_mut(0)[p] = ww;
        }
    }

    /// `DeactivateLostParticles(findpts)`: move every particle whose
    /// `FindPointsGSLIB` code is `CODE_NOT_FOUND` (2) to the inactive set.
    ///
    /// The miniapp calls it with `findpts = false`: the codes are the ones the
    /// interpolation search of [`Self::interpolate_uw`] left in
    /// [`Self::last_codes`].  (`FindPointsGSLIB::GetPointsNotFoundIndices`
    /// filters `code == 2`, so border points — `code == 1`, e.g. a particle
    /// sitting exactly on the inlet — stay active.)
    fn deactivate_lost_particles(&mut self) {
        let lost: Vec<usize> = self
            .last_codes
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == CODE_NOT_FOUND)
            .map(|(i, _)| i)
            .collect();
        let new_idxs = self.inactive.add_particles(lost.len());
        for (k, &i) in lost.iter().enumerate() {
            let coords = self.particles.coords()[i];
            self.inactive.coords_mut()[new_idxs[k]] = coords;
            self.inactive.kappa_mut()[new_idxs[k]] = self.particles.kappa()[i];
            self.inactive.zeta_mut()[new_idxs[k]] = self.particles.zeta()[i];
            self.inactive.gamma_mut()[new_idxs[k]] = self.particles.gamma()[i];
        }
        self.particles.remove_particles(&lost);
    }

    /// `NavierParticles::Step(dt, u_gf, w_gf)`.
    pub fn step(
        &mut self,
        dt: f64,
        space: &VectorH1Space<Mesh<2>>,
        order: u8,
        u_gf: &[f64],
        w_gf: &[f64],
    ) {
        // Shift the fluid velocity, fluid vorticity, particle velocity and
        // particle position histories (`std::move`, so the shifted-from slots
        // are reallocated below — every entry is written before it is read).
        let np = self.particles.n_particles();
        self.particles.rotate_histories();
        self.particles.rotate_positions();
        self.particles.resize_current(np);

        self.set_time_integration_coefficients();

        for i in 0..np {
            // Increment the particle order (capped at 3) …
            let o = &mut self.particles.order_mut()[i];
            if *o < 3 {
                *o += 1;
            }
            // … then take the 2D step.
            self.particle_step_2d(dt, i);
        }

        self.apply_bcs();

        let ref_elem = factory_ref_elem(FactoryElem::Quad, order);
        self.interpolate_uw(space, &*ref_elem, u_gf, w_gf);

        // Lost particles are already flagged by `InterpolateUW`'s search.
        self.deactivate_lost_particles();

        // Rotate the particle time-step history.
        self.dthist[2] = self.dthist[1];
        self.dthist[1] = self.dthist[0];
        self.dthist[0] = dt;
    }
}

/// `sqrt((a-b)·(a-b))`.
fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// `v.Norml2()`.
fn norm2(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

// ─── `%g`-with-16-significant-digits formatting (ParticleSet::PrintCSV) ─────

/// `std::stringstream` with `precision(16)` in default float notation — the
/// formatting `ParticleSet::PrintCSV` uses for coordinates, fields and tags
/// (`id` is printed as an integer).
///
/// This is the `%g` rule: use scientific notation when the decimal exponent is
/// `< -4` or `>= 16`, fixed notation otherwise, always with 16 significant
/// digits and with trailing zeros removed.
///
/// Implementation note: `fem_io` already has this formatter
/// (`c_printf_g16` in `crates/io/src/mfem.rs`), but it is private and
/// `crates/io` is not modifiable in this round, so the 40 lines are repeated
/// here (pinned by the unit tests below against the C++ output).
pub fn fmt_g16(x: f64) -> String {
    if x == 0.0 {
        // `printf("%.16g", -0.0)` prints "-0".
        return if x.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        };
    }
    if !x.is_finite() {
        return x.to_string();
    }
    // 16 significant digits via scientific notation with 15 decimals.
    let s = format!("{:.15e}", x);
    let (mant, exp_str) = s.split_once('e').expect("scientific format");
    let exp: i32 = exp_str.parse().expect("exponent");
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    if (-4..16).contains(&exp) {
        let mut out = String::new();
        if neg && digits != "0" {
            out.push('-');
        }
        let dot = 1 + exp; // 0-based position of the decimal point
        if dot <= 0 {
            out.push_str("0.");
            for _ in 0..-dot {
                out.push('0');
            }
            out.push_str(digits);
        } else if dot as usize >= digits.len() {
            out.push_str(digits);
            for _ in 0..(dot as usize - digits.len()) {
                out.push('0');
            }
        } else {
            out.push_str(&digits[..dot as usize]);
            out.push('.');
            out.push_str(&digits[dot as usize..]);
        }
        out
    } else {
        let mut m = digits.to_string();
        if m.len() > 1 {
            m.insert(1, '.');
        }
        let sign = if exp >= 0 { "+" } else { "-" };
        format!(
            "{}{}e{}{:02}",
            if neg { "-" } else { "" },
            m,
            sign,
            exp.abs()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `Order` tag and the three scalar fields of a fresh particle are
    /// zero, and `AddParticles` hands out consecutive IDs like MFEM's serial
    /// `ParticleSet` (`id_stride = 1`, `id_counter = 0`).
    #[test]
    fn particle_set_ids_and_growth() {
        let mut ps = ParticleSet::new(0);
        assert_eq!(ps.n_particles(), 0);
        let idxs = ps.add_particles(3);
        assert_eq!(idxs, vec![0, 1, 2]);
        let idxs = ps.add_particles(2);
        assert_eq!(idxs, vec![3, 4]);
        assert_eq!(ps.global_n_particles(), 5);
        assert_eq!(ps.print_csv(&[0], &[0]).lines().count(), 6);
        // Removing particles keeps the IDs of the survivors.
        ps.remove_particles(&[1, 3]);
        assert_eq!(ps.n_particles(), 3);
        let csv = ps.print_csv(&[0], &[0]);
        assert!(csv.starts_with("id,X,Y,kappa,Order\n0,"));
        assert!(csv.contains("\n2,"));
        assert!(csv.contains("\n4,"));
    }

    /// `%g` at precision 16, pinned against the C++ `PrintCSV` output of the
    /// reference run (see the `navier_bifurcation` verification table).
    #[test]
    fn g16_matches_the_cpp_csv_formatting() {
        assert_eq!(fmt_g16(0.0), "0");
        assert_eq!(fmt_g16(2.984007413745882), "2.984007413745882");
        assert_eq!(fmt_g16(0.4958833291565495), "0.4958833291565495");
        assert_eq!(fmt_g16(5.731353361718071e-05), "5.731353361718071e-05");
        assert_eq!(fmt_g16(1.0), "1");
        assert_eq!(fmt_g16(-1.0), "-1");
        assert_eq!(fmt_g16(1234567.0), "1234567");
        assert_eq!(fmt_g16(1e-4), "0.0001");
        assert_eq!(fmt_g16(1e-5), "1e-05");
        assert_eq!(fmt_g16(1e16), "1e+16");
        assert_eq!(fmt_g16(-0.0), "-0");
    }

    /// The 2D wall normal helper: for a wall segment `(0,1) → (8,1)` with
    /// `invert_normal = true` the left normal points *down* into the domain
    /// (`(0,-1)`), which is the convention `Add2DReflectionBC` documents.
    #[test]
    fn normal_points_into_the_domain() {
        let n = NavierParticles::get_2d_normal([0.0, 1.0], [8.0, 1.0], true);
        assert!((n[0] - 0.0).abs() < 1e-15 && (n[1] + 1.0).abs() < 1e-15);
        let n = NavierParticles::get_2d_normal([0.0, 0.0], [17.0, 0.0], false);
        assert!((n[0] - 0.0).abs() < 1e-15 && (n[1] - 1.0).abs() < 1e-15);
    }

    /// Segment crossing: the unit square's diagonal crossings, the
    /// parallel-segment rejection and the `0 <= t1, t2 <= 1` bounds.
    #[test]
    fn segment_intersection() {
        let hit = NavierParticles::get_2d_segment_intersection(
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        )
        .expect("crossing");
        assert!(dist(hit.0, [0.0, 0.0]) < 1e-15);
        assert!((hit.1 - 0.5).abs() < 1e-15 && (hit.2 - 0.5).abs() < 1e-15);
        // Parallel.
        assert!(NavierParticles::get_2d_segment_intersection(
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        )
        .is_none());
        // The crossing lies outside the first segment.
        assert!(NavierParticles::get_2d_segment_intersection(
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0]
        )
        .is_none());
    }
}
