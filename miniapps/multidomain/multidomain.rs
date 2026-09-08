//! multidomain miniapp (H1 version) — serial port of MFEM
//! `miniapps/multidomain/multidomain.cpp` (397 lines, PAR-only upstream).
//!
//! Physics (unchanged from C++):
//! - outer box (domain attribute 2): heat equation  dT/dt = κΔT, κ = 1,
//!   Dirichlet T = 1 on the four outside walls (bdr attrs 1-4), natural
//!   (insulated) on the inner cylinder wall (bdr attr 9).
//! - inner cylinder (domain attribute 1): convection-diffusion
//!   dT/dt = κΔT - α∇·(b T), κ = 0.1, α = 1, prescribed pipe velocity profile
//!   `b` (no-slip-ish at the wall), Dirichlet T = 0 on the inflow cap
//!   (bdr attr 8) and T = T_block on the interface (bdr attr 9, taken from the
//!   block solution → first-order one-way coupling).
//! - Both domains use order-2 H1 elements; time integration is SSP-RK3
//!   (RK3SSPSolver) with an explicit mass-matrix inversion (CG + Jacobi,
//!   rtol 1e-8, max_iter 100) per Mult call.
//!
//! Serial mechanism mapping (C++ PAR → this port):
//!
//! | MFEM (parallel)                             | this port                                     |
//! |---------------------------------------------|-----------------------------------------------|
//! | ParSubMesh::CreateFromDomain(parent, attrs) | `fem_mesh::submesh::extract_submesh_3d`       |
//! | ParFiniteElementSpace / true dofs           | `H1Space<Mesh<3>>` (conforming: vdofs=tdofs)  |
//! | ParSubMesh::CreateTransferMap + .Transfer   | interface dof copy by dof-coordinate match    |
//! | CGSolver + HypreSmoother(Jacobi)            | `solve_pcg_dsmoother` (CG + Jacobi diag)      |
//! | GetEssentialTrueDofs(bdr_attrs)             | local `hex_boundary_dofs` (see note below)    |
//! | ProjectBdrCoefficient(ConstantCoefficient)  | direct dof assignment (constant coefficient)  |
//! | RK3SSPSolver                                | hand-coded SSP-RK3 (identical stage coeffs)   |
//! | GLVis socketstream output                   | trimmed (silent runs, upstream `-no-vis`)     |
//!
//! The `SubMesh::CreateTransferMap` equivalence: MFEM's SubMesh-to-SubMesh
//! transfer moves both fields to the common root parent and back; because the
//! two domains partition the parent, the net effect is exactly
//! `T_cyl[i] = T_blk[match(i)]` on interface dofs and `T_cyl[i] = T_cyl[i]`
//! elsewhere (dst_to_parent carries the cylinder's own values through the
//! parent). Both submeshes are extracted from the same refined parent mesh, so
//! interface dofs coincide bit-for-bit in physical coordinates and the
//! coordinate match is exact. H1 order 2 entity dofs (edge midpoints, quad
//! face centers) are orientation-symmetric, so matching by coordinates is
//! orientation-independent.
//!
//! Usage (after wiring the `[[example]]` block in `examples/Cargo.toml`):
//! ```text
//! cargo run --example multidomain --                      # upstream defaults (dt=1e-5, tf=5)
//! cargo run --example multidomain -- -tf 0.5 -dt 2e-4     # short stable run
//! cargo run --example multidomain -- -m <mesh> -nr 1 -qp 20   # verification aids
//! ```
//!
//! Options: `-o` order (2), `-tf` final time (5.0), `-dt` time step (1e-5),
//! `-vs` print stride (10), `-m` mesh (data/multidomain-hex.mesh),
//! `-nr 1` skip uniform refinement (verification: file already refined),
//! `-qp N` force one shared quadrature order on all forms (verification:
//! default -1 uses MFEM-default-equivalent rules: mass 2p, K 2p-1),
//! `-dump 1` write final dof dumps rust_{block,cyl}_dofs.txt (verification).
//!
//! Note: dt must respect the explicit diffusion CFL limit; upstream defaults
//! (dt = 1e-5) are far inside the stable region, dt ≳ 5e-4 blows up.

use std::collections::{HashMap, HashSet};

use fem_assembly::postproc::coefficient::{CoeffCtx, VectorCoeff};
use fem_assembly::standard::{ConvectionIntegrator, DiffusionIntegrator, MassIntegrator};
use fem_assembly::Assembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::submesh::extract_submesh_3d;
use fem_mesh::{refine_hex8_uniform, Mesh};
use fem_solver::{solve_pcg_dsmoother, SolverConfig};
use fem_space::{fe_space::FESpace, dof_manager::{EdgeKey, QuadFaceKey}, H1Space};

/// Prescribed velocity profile for the convection-diffusion equation inside
/// the cylinder (C++ multidomain.cpp:52-70). Approximates no-slip (v=0) at
/// the cylinder wall r = 0.25.
fn velocity_profile(x: &[f64], out: &mut [f64]) {
    let a = 1.0;
    let px = x[0];
    let py = x[1];
    let r = (px * px + py * py).sqrt();

    out[0] = 0.0;
    out[1] = 0.0;

    if r.abs() >= 0.25 - 1e-8 {
        out[2] = 0.0;
    } else {
        out[2] = a * (-(px * px / 2.0 + py * py / 2.0)).exp();
    }
}

/// Velocity coefficient folded with MFEM's ConvectionIntegrator factor
/// `-alpha` (C++ passes `ConvectionIntegrator(*q, -alpha)`; fem-rs'
/// ConvectionIntegrator has no scalar factor, so the factor is folded into
/// the velocity field: `-alpha·(b·∇u) ≡ ((-alpha·b)·∇u)`).
struct ScaledVelocity {
    alpha: f64,
}

impl VectorCoeff for ScaledVelocity {
    fn eval(&self, ctx: &CoeffCtx<'_>, out: &mut [f64]) {
        velocity_profile(ctx.x, out);
        for v in out.iter_mut() {
            *v *= -self.alpha;
        }
    }
}

/// Serial equivalent of MFEM `FiniteElementSpace::GetEssentialTrueDofs` for
/// Hex8 submeshes: all H1 dofs (vertices, edge dofs, quad-face-interior dofs)
/// on the boundary faces whose tag is listed in `tags`.
///
/// Kernel gap note: fem-space's `boundary_dofs` enumerates face edges through
/// the stored boundary-face vertex order. Submeshes extracted by
/// `fem_mesh::submesh::extract_submesh_3d` store quad boundary faces in
/// *sorted-vertex* order (a canonical dedup key, not a cyclic Quad4
/// connectivity), so the derived edge pairs include face diagonals and about
/// half of the edge-dof lookups fail. This equivalent walks the submesh
/// *elements'* hex face table (proper cyclic vertex order) instead; MFEM
/// avoids the same problem by recording face orientations in the SubMesh
/// (`GetParentFaceOrientations`).
fn hex_boundary_dofs(mesh: &Mesh<3>, dm: &fem_space::DofManager, tags: &[i32]) -> Vec<u32> {
    use fem_mesh::topology::MeshTopology;

    // Sorted quad vertex set -> boundary tag of the stored submesh faces.
    let mut face_tag_by_set: HashMap<[u32; 4], i32> = HashMap::new();
    for f in 0..mesh.n_boundary_faces() as u32 {
        let ns = mesh.face_nodes(f);
        let mut s = [ns[0], ns[1], ns[2], ns[3]];
        s.sort_unstable();
        face_tag_by_set.insert(s, mesh.face_tag(f));
    }

    // Hex element face table (cyclic vertex order per face).
    const HEX_FACES: [[usize; 4]; 6] = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ];

    let mut out: HashSet<u32> = HashSet::new();
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        for lf in &HEX_FACES {
            let fns = [ns[lf[0]], ns[lf[1]], ns[lf[2]], ns[lf[3]]];
            let mut key = fns;
            key.sort_unstable();
            let Some(&tag) = face_tag_by_set.get(&key) else {
                continue;
            };
            if !tags.contains(&tag) {
                continue;
            }
            // Vertex dofs (conforming hex mesh: dof id == node id, as the
            // `phys_to_vertex_dof` fallback in fem-space's boundary_dofs).
            for &n in &fns {
                let d = dm.phys_to_vertex_dof.get(&n).copied().unwrap_or(n);
                out.insert(d);
            }
            // Edge dofs along the proper cyclic face order.
            for i in 0..4 {
                let ek = EdgeKey::new(fns[i], fns[(i + 1) % 4]);
                if let Some(ds) = dm.edge_pk_map.get(&ek) {
                    out.extend(ds.iter().copied());
                }
                if let Some(&d) = dm.edge_dof_map.get(&ek) {
                    out.insert(d);
                }
                if let Some(ds) = dm.edge_dof2_map.get(&ek) {
                    out.extend(ds.iter().copied());
                }
            }
            // Quad-face-interior dofs.
            let qk = QuadFaceKey::new(fns[0], fns[1], fns[2], fns[3]);
            if let Some(ds) = dm.quad_face_pk_map.get(&qk) {
                out.extend(ds.iter().copied());
            }
        }
    }
    let mut v: Vec<u32> = out.into_iter().collect();
    v.sort_unstable();
    v
}

/// Convection-diffusion time dependent operator
///
/// ```text
/// dT/dt = κΔT - α∇·(b T)
/// ```
///
/// (C++ multidomain.cpp:80-202). Can also be used to create a diffusion or
/// convection only operator by setting α or κ to zero. MFEM folds the minus
/// sign of the heat equation into the diffusion coefficient
/// (`DiffusionIntegrator(ConstantCoefficient(-kappa))`), which is kept here.
struct ConvectionDiffusionTDO {
    /// Mass form, essential dofs eliminated (C++ `Mform.FormSystemMatrix`).
    m_mat: CsrMatrix<f64>,
    /// Stiffness operator: convection + diffusion (C++ `Kform`).
    k_mat: CsrMatrix<f64>,
    /// RHS vector (zero for the H1 branch; C++ assembles an empty bform).
    b: Vec<f64>,
    /// Essential dof array (du_dt is zeroed there in `Mult`).
    ess_tdofs: Vec<u32>,
    /// Mass matrix solver config: CG + Jacobi, reltol 1e-8, max_iter 100
    /// (C++ M_solver settings, print level 0).
    solve_cfg: SolverConfig,
    /// Auxiliary vector (C++ t1).
    t1: Vec<f64>,
}

impl ConvectionDiffusionTDO {
    /// Assemble M and K (C++ constructor, multidomain.cpp:92-144).
    ///
    /// `qp_override > 0` forces one high-order quadrature rule on every form
    /// (verification aid matching the C++ harness `-qp`; 0 keeps the MFEM
    /// default-equivalent rules: mass 2·order, K 2·order-1).
    fn new(
        space: &H1Space<Mesh<3>>,
        ess_tdofs: Vec<u32>,
        alpha: f64,
        kappa: f64,
        order: u8,
        qp_override: i32,
    ) -> Self {
        // Mass form: MassIntegrator, exact for degree 2·order → quad_order 2·order.
        let mass = MassIntegrator { rho: 1.0 };
        let mass_qp = if qp_override > 0 { qp_override as u8 } else { 2 * order };
        let mut m_mat = Assembler::assemble_bilinear(space, &[&mass], mass_qp);

        // Mform.FormSystemMatrix(ess_tdofs_): eliminate essential rows/cols
        // (DIAG_KEEP, rhs values zero — values are re-imposed by the transfer).
        let n = m_mat.nrows;
        let mut zero_rhs = vec![0.0_f64; n];
        let zero_vals = vec![0.0_f64; ess_tdofs.len()];
        fem_space::apply_dirichlet(&mut m_mat, &mut zero_rhs, &ess_tdofs, &zero_vals);

        // Kform: ConvectionIntegrator(*q, -alpha) + DiffusionIntegrator(-kappa).
        let vel = ScaledVelocity { alpha };
        let conv = ConvectionIntegrator { velocity: vel };
        let diff = DiffusionIntegrator { kappa: -kappa };
        // Exact for degree max(2p-1, 2p-2) → quad_order 2p-1 (MFEM defaults).
        let k_qp = if qp_override > 0 { qp_override as u8 } else { 2 * order - 1 };
        let k_mat = Assembler::assemble_bilinear(space, &[&conv, &diff], k_qp);

        // bform.Assemble() with no integrators on the H1 branch → b = 0.
        let b = vec![0.0_f64; n];

        let solve_cfg = SolverConfig {
            rtol: 1e-8,
            atol: 0.0,
            max_iter: 100,
            verbose: false,
            ..SolverConfig::default()
        };

        ConvectionDiffusionTDO {
            m_mat,
            k_mat,
            b,
            ess_tdofs,
            solve_cfg,
            t1: vec![0.0; n],
        }
    }

    /// du_dt = M⁻¹(K u + b), zeroed at essential dofs
    /// (C++ Mult, multidomain.cpp:146-152).
    fn mult(&mut self, u: &[f64], du_dt: &mut [f64]) {
        self.k_mat.spmv(u, &mut self.t1);
        for (t, &bv) in self.t1.iter_mut().zip(self.b.iter()) {
            *t += bv;
        }
        solve_pcg_dsmoother(&self.m_mat, &self.t1, du_dt, &self.solve_cfg)
            .expect("ConvectionDiffusionTDO::Mult: mass CG solve failed");
        for &d in &self.ess_tdofs {
            du_dt[d as usize] = 0.0;
        }
    }
}

/// SSP-RK3 step (MFEM linalg/ode.cpp:263-285, RK3SSPSolver::Step) with the
/// same fused expression forms. As in MFEM, `t` is advanced by reference at
/// the end of the step:
///
/// ```text
/// k  = f(t, x);          y  = x + dt·k
/// k  = f(t+dt, y);       y  = 3/4·x + 1/4·(y + dt·k)
/// k  = f(t+dt/2, y);     x  = 1/3·x + 2/3·(y + dt·k)
/// t += dt
/// ```
fn rk3ssp_step(f: &mut ConvectionDiffusionTDO, x: &mut [f64], t: &mut f64, dt: f64) {
    let n = x.len();
    let mut k = vec![0.0_f64; n];
    let mut y = vec![0.0_f64; n];

    // k = f(t, x); y = x + dt·k
    f.mult(x, &mut k);
    for i in 0..n {
        y[i] = x[i] + dt * k[i];
    }

    // k = f(t+dt, y); y += dt·k; y = 3/4·x + 1/4·y
    f.mult(&y, &mut k);
    for i in 0..n {
        y[i] += dt * k[i];
    }
    for i in 0..n {
        y[i] = 3.0_f64 / 4.0 * x[i] + 1.0_f64 / 4.0 * y[i];
    }

    // k = f(t+dt/2, y); y += dt·k; x = 1/3·x + 2/3·y
    f.mult(&y, &mut k);
    for i in 0..n {
        y[i] += dt * k[i];
    }
    for i in 0..n {
        x[i] = 1.0_f64 / 3.0 * x[i] + 2.0_f64 / 3.0 * y[i];
    }

    *t += dt;
}

/// Serial equivalent of `ParSubMesh::CreateTransferMap(block → cylinder)`
/// (C++ multidomain.cpp:344-347): copies the block values onto the cylinder
/// dofs lying on the shared interface (bdr attribute 9), leaving the
/// cylinder-interior dofs untouched (see module docs).
struct BlockToCylinderMap {
    /// (cylinder dof, block dof) pairs on the interface.
    pairs: Vec<(u32, u32)>,
}

impl BlockToCylinderMap {
    fn new(
        cyl_space: &H1Space<Mesh<3>>,
        blk_space: &H1Space<Mesh<3>>,
        interface_dofs_cyl: &[u32],
        interface_dofs_blk: &[u32],
    ) -> Self {
        // Coordinates are quantized to a 1e-9 grid: the two submeshes compute
        // a shared dof's position by interpolating from *different* element
        // neighborhoods (cylinder-side vs block-side hexes), so quad-face
        // center coordinates can differ in the last ulp between the two
        // spaces. Distinct mesh dofs are O(1e-2) apart, so the 1e-9 grid
        // cannot merge distinct dofs.
        fn coord_key(c: &[f64]) -> [i64; 3] {
            [
                (c[0] * 1e9).round() as i64,
                (c[1] * 1e9).round() as i64,
                (c[2] * 1e9).round() as i64,
            ]
        }
        let dm_blk = blk_space.dof_manager();
        let mut by_coord: HashMap<[i64; 3], u32> = HashMap::new();
        for &d in interface_dofs_blk {
            by_coord.insert(coord_key(dm_blk.dof_coord(d)), d);
        }
        let dm_cyl = cyl_space.dof_manager();
        let mut pairs = Vec::with_capacity(interface_dofs_cyl.len());
        for &c in interface_dofs_cyl {
            let key = coord_key(dm_cyl.dof_coord(c));
            let &b = by_coord.get(&key).unwrap_or_else(|| {
                panic!("BlockToCylinderMap: no block dof at cylinder interface dof {c}");
            });
            pairs.push((c, b));
        }
        BlockToCylinderMap { pairs }
    }

    /// C++ `temperature_block_to_cylinder_map.Transfer(block_gf, cyl_gf)`
    /// followed by `cyl_gf.GetTrueDofs` (multidomain.cpp:365-368).
    fn transfer(&self, src: &[f64], dst: &mut [f64]) {
        for &(c, b) in &self.pairs {
            dst[c as usize] = src[b as usize];
        }
    }
}

/// Dump all H1 dofs as "x y z value" lines (verification against the C++
/// serial harness; not part of the upstream miniapp).
fn dump_dofs(space: &H1Space<Mesh<3>>, sol: &[f64], fname: &str) {
    let dm = space.dof_manager();
    let mut lines: Vec<String> = Vec::with_capacity(sol.len());
    for (d, &v) in sol.iter().enumerate() {
        let c = dm.dof_coord(d as u32);
        lines.push(format!(
            "{:.17e} {:.17e} {:.17e} {:.17e}",
            c[0], c[1], c[2], v
        ));
    }
    lines.sort();
    std::fs::write(fname, lines.join("\n") + "\n").expect("dump write failed");
}

fn parse_f64(args: &[String], flag: &str, default: f64) -> f64 {
    args.iter()
        .position(|a| a == flag)
        .map(|i| args[i + 1].parse().expect("bad float arg"))
        .unwrap_or(default)
}

fn parse_u32(args: &[String], flag: &str, default: u32) -> u32 {
    args.iter()
        .position(|a| a == flag)
        .map(|i| args[i + 1].parse().expect("bad int arg"))
        .unwrap_or(default)
}

fn parse_i32(args: &[String], flag: &str, default: i32) -> i32 {
    args.iter()
        .position(|a| a == flag)
        .map(|i| args[i + 1].parse().expect("bad int arg"))
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // C++ multidomain.cpp:211-230 — options (visualization is trimmed: this
    // port always runs silent, i.e. as if `-no-vis` was given).
    let order = parse_u32(&args, "-o", 2) as u8;
    let t_final = parse_f64(&args, "-tf", 5.0);
    let dt = parse_f64(&args, "-dt", 1.0e-5);
    let vis_steps = parse_u32(&args, "-vs", 10) as usize;
    let qp = parse_i32(&args, "-qp", -1);
    let mesh_file = args
        .iter()
        .position(|a| a == "-m")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| "data/multidomain-hex.mesh".to_string());

    // C++ multidomain.cpp:232-236 — load parent mesh and refine once
    // (`parent_mesh.UniformRefinement()`).
    //
    // Kernel gap note: `refine_uniform_3d` (the generic dispatcher) panics on
    // Hex8 meshes with quad boundary faces — its `rebuild_3d_boundary` step
    // looks up the new quad-face centers by exact-bit coordinates computed as
    // (sum of boundary-face vertex order)/4, while `refine_nonconforming_hex`
    // creates those centers summing in MFEM *element-face* vertex order; the
    // two accumulation orders differ in the last ulp for generic coordinates.
    // `refine_hex8_uniform` emits the refined boundary faces from its own
    // face-center ids (no coordinate re-lookup) and is used instead. The
    // resulting mesh is the same octasection MFEM produces.
    let parent = read_mfem_file(&mesh_file).expect("failed to read parent mesh");
    let parent_mesh: Mesh<3> = parent.mesh3d.expect("multidomain-hex.mesh must be 3D");
    assert!(
        parent_mesh.elem_type == fem_mesh::ElementType::Hex8,
        "multidomain-hex.mesh must be an Hex8 mesh"
    );
    // -nr 1: the mesh file is already refined (verification path — e.g. feed
    // the MFEM-refined mesh to isolate refinement differences).
    let no_refine = parse_i32(&args, "-nr", 0) != 0;
    let parent_mesh = if no_refine {
        parent_mesh
    } else {
        let all_elems: Vec<u32> = (0..parent_mesh.n_elems() as u32).collect();
        refine_hex8_uniform(&parent_mesh, &all_elems).0
    };
    println!(
        "Parent mesh: {} elements, {} vertices",
        parent_mesh.n_elems(),
        parent_mesh.n_nodes()
    );

    // C++ multidomain.cpp:242-249 — cylinder submesh (domain attribute 1) and
    // its H1 space.
    let cylinder_submesh = extract_submesh_3d(&parent_mesh, &[1]);
    let fes_cylinder = H1Space::new(cylinder_submesh.mesh.clone(), order);
    {
        let m = &cylinder_submesh.mesh;
        let min = m.face_tags.iter().copied().min().unwrap_or(0);
        let max = m.face_tags.iter().copied().max().unwrap_or(0);
        println!(
            "Cylinder submesh: NE={} NV={} ndofs={} bdr_attrs={min}..{max}",
            m.n_elems(),
            m.n_nodes(),
            fes_cylinder.n_dofs()
        );
    }

    // C++ multidomain.cpp:251-270 — essential dofs of the cylinder: inflow
    // (bdr attr 8) and interface (bdr attr 9). C++ passes zero-based *marker*
    // arrays (`bdr_attr_is_ess`, index = attribute-1) to GetEssentialTrueDofs;
    // fem-rs' boundary_dofs takes the explicit tag list, so the markers are
    // expanded into tags first.
    fn marker_to_tags(marker: &[i32]) -> Vec<i32> {
        (1..=marker.len() as i32)
            .filter(|&a| marker[(a - 1) as usize] != 0)
            .collect()
    }
    let dm_cyl = fes_cylinder.dof_manager();
    let mesh_cyl = fes_cylinder.mesh();
    let mut inflow_attributes = vec![0_i32; 9];
    inflow_attributes[7] = 1;
    let mut inner_cylinder_wall_attributes = vec![0_i32; 9];
    inner_cylinder_wall_attributes[8] = 1;
    let mut ess_tdofs = hex_boundary_dofs(mesh_cyl, dm_cyl, &marker_to_tags(&inflow_attributes));
    ess_tdofs.extend(hex_boundary_dofs(
        mesh_cyl,
        dm_cyl,
        &marker_to_tags(&inner_cylinder_wall_attributes),
    ));
    ess_tdofs.sort_unstable();
    ess_tdofs.dedup();
    println!("Cylinder ess vdofs: {}", ess_tdofs.len());
    let mut cd_tdo =
        ConvectionDiffusionTDO::new(&fes_cylinder, ess_tdofs, 0.0, 1.0e-1, order, qp);

    // C++ multidomain.cpp:273-280 — zero initial condition in the cylinder.
    let mut temperature_cylinder = vec![0.0_f64; fes_cylinder.n_dofs()];

    // C++ multidomain.cpp:282-304 — block submesh (domain attribute 2), heat
    // equation with κ = 1, α = 0, essential walls = bdr attrs 1-4.
    let block_submesh = extract_submesh_3d(&parent_mesh, &[2]);
    let fes_block = H1Space::new(block_submesh.mesh.clone(), order);
    {
        let m = &block_submesh.mesh;
        let min = m.face_tags.iter().copied().min().unwrap_or(0);
        let max = m.face_tags.iter().copied().max().unwrap_or(0);
        println!(
            "Block submesh: NE={} NV={} ndofs={} bdr_attrs={min}..{max}",
            m.n_elems(),
            m.n_nodes(),
            fes_block.n_dofs()
        );
    }

    let dm_blk = fes_block.dof_manager();
    let mesh_blk = fes_block.mesh();
    let mut block_wall_attributes = vec![0_i32; 9];
    block_wall_attributes[0] = 1;
    block_wall_attributes[1] = 1;
    block_wall_attributes[2] = 1;
    block_wall_attributes[3] = 1;
    let block_ess_tdofs = hex_boundary_dofs(mesh_blk, dm_blk, &marker_to_tags(&block_wall_attributes));
    println!("Block ess vdofs (walls 1-4): {}", block_ess_tdofs.len());

    let mut d_tdo = ConvectionDiffusionTDO::new(&fes_block, block_ess_tdofs, 0.0, 1.0, order, qp);

    // C++ multidomain.cpp:306-313 — T = 1 on the outside walls via
    // ProjectBdrCoefficient(ConstantCoefficient(1.0)); interior stays 0.
    let mut temperature_block = vec![0.0_f64; fes_block.n_dofs()];
    for &d in &hex_boundary_dofs(mesh_blk, dm_blk, &marker_to_tags(&block_wall_attributes)) {
        temperature_block[d as usize] = 1.0;
    }
    println!(
        "Block initial BC dofs (nonzero): {}",
        temperature_block.iter().filter(|&&v| v != 0.0).count()
    );

    // C++ multidomain.cpp:318-322 creates a surface SubMesh of the interface
    // (bdr attr 9) that is never used afterwards — trimmed here.

    // C++ multidomain.cpp:344-347 — block → cylinder transfer map.
    let interface_dofs_cyl = hex_boundary_dofs(
        mesh_cyl,
        dm_cyl,
        &marker_to_tags(&inner_cylinder_wall_attributes),
    );
    let interface_dofs_blk = hex_boundary_dofs(
        mesh_blk,
        dm_blk,
        &marker_to_tags(&inner_cylinder_wall_attributes),
    );
    let temperature_block_to_cylinder_map =
        BlockToCylinderMap::new(&fes_cylinder, &fes_block, &interface_dofs_cyl, &interface_dofs_blk);

    // C++ multidomain.cpp:349-394 — segregated coupling time loop.
    let mut t = 0.0_f64;
    let mut last_step = false;
    let mut ti = 1_usize;
    while !last_step {
        if t + dt >= t_final - dt / 2.0 {
            last_step = true;
        }

        // Advance the diffusion equation on the outer block (RK3SSP). As in
        // C++, both Step calls share `t` by reference, so `t` advances by
        // 2·dt per loop iteration (upstream multidomain.cpp:359-372).
        rk3ssp_step(&mut d_tdo, &mut temperature_block, &mut t, dt);

        // Transfer the block solution onto the cylinder interface dofs to act
        // as the Dirichlet boundary condition of the convection-diffusion
        // equation (one-way coupling).
        temperature_block_to_cylinder_map
            .transfer(&temperature_block, &mut temperature_cylinder);

        // Advance the convection-diffusion equation inside the cylinder.
        rk3ssp_step(&mut cd_tdo, &mut temperature_cylinder, &mut t, dt);

        if last_step || ti % vis_steps == 0 {
            let bsum: f64 = temperature_block.iter().sum();
            let csum: f64 = temperature_cylinder.iter().sum();
            let bmin = temperature_block.iter().cloned().fold(f64::INFINITY, f64::min);
            let bmax = temperature_block
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let cmin = temperature_cylinder
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let cmax = temperature_cylinder
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            println!(
                "step {ti}, t = {t}  block: sum={bsum:.6e} min={bmin:.6e} max={bmax:.6e}  cyl: sum={csum:.6e} min={cmin:.6e} max={cmax:.6e}"
            );
        }

        ti += 1;
    }

    // -dump 1: write final fields as "x y z value" dof dumps (verification aid
    // for cross-diffing against the C++ serial harness; off by default —
    // upstream writes nothing in -no-vis mode).
    if parse_i32(&args, "-dump", 0) != 0 {
        dump_dofs(&fes_block, &temperature_block, "rust_block_dofs.txt");
        dump_dofs(&fes_cylinder, &temperature_cylinder, "rust_cyl_dofs.txt");
        println!("Dumps written.");
    }
}
