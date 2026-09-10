//! `miniapps/multidomain/multidomain_nd.cpp` (PAR-only upstream).
//!
//! H(curl) variant of the multidomain miniapp: magnetic diffusion in the outer
//! box and convection-diffusion inside the cylinder, using Nédélec (ND) finite
//! elements. See `multidomain.rs` for the H1 variant and the serial mechanism
//! mapping table.
//!
//! Deviations from the C++ miniapp:
//! * GLVis / ParaView output is not available (silent runs).
//! * Serial port: ParSubMesh → extract_submesh_3d, ParFESpace → HCurlSpace.
//! * Essential DOFs are edge DOFs on boundary faces (tangential components).

use std::collections::HashSet;

use fem_assembly::postproc::coefficient::{CoeffCtx, VectorCoeff};
use fem_assembly::standard::{
    CurlCurlIntegrator, MixedWeakCurlCrossIntegrator, VectorMassIntegrator,
};
use fem_assembly::vector_assembler::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::submesh::extract_submesh_3d;
use fem_mesh::{refine_hex8_uniform, Mesh};
use fem_solver::{solve_pcg_dsmoother, SolverConfig};
use fem_space::fe_space::FESpace;
use fem_space::hcurl::HCurlSpace;

/// Prescribed velocity profile (C++ multidomain_nd.cpp:52-70).
fn velocity_profile(x: &[f64], out: &mut [f64]) {
    let a = 1.0;
    let px = x[0];
    let py = x[1];
    let r = (px * px + py * py).sqrt();

    out[0] = 0.0;
    out[1] = 0.0;
    out[2] = if r.abs() >= 0.25 - 1e-8 {
        0.0
    } else {
        a * (-(px * px / 2.0 + py * py / 2.0)).exp()
    };
}

/// Scaled velocity coefficient: `-alpha * velocity`.
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

/// Edge DOFs on boundary faces whose tag is listed in `tags` (for H(curl)).
fn hcurl_boundary_dofs(
    mesh: &Mesh<3>,
    space: &HCurlSpace<Mesh<3>>,
    tags: &[i32],
) -> Vec<u32> {
    use fem_mesh::topology::MeshTopology;
    use fem_space::dof_manager::EdgeKey;

    let mut face_tag_by_set: std::collections::HashMap<[u32; 4], i32> =
        std::collections::HashMap::new();
    for f in 0..mesh.n_boundary_faces() as u32 {
        let ns = mesh.face_nodes(f);
        let mut s = [ns[0], ns[1], ns[2], ns[3]];
        s.sort_unstable();
        face_tag_by_set.insert(s, mesh.face_tag(f));
    }

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
            // Edge DOFs along the proper cyclic face order.
            for i in 0..4 {
                let ek = EdgeKey::new(fns[i], fns[(i + 1) % 4]);
                if let Some(dof) = space.edge_dof(ek) {
                    out.insert(dof);
                }
            }
        }
    }
    let mut v: Vec<u32> = out.into_iter().collect();
    v.sort_unstable();
    v
}

/// Convection-diffusion time dependent operator for H(curl).
///
/// ```text
/// dH/dt = -∇×(σ∇×H) + α∇×(v×H)
/// ```
struct ConvectionDiffusionTDO {
    m_mat: CsrMatrix<f64>,
    k_mat: CsrMatrix<f64>,
    b: Vec<f64>,
    ess_tdofs: Vec<u32>,
    solve_cfg: SolverConfig,
    t1: Vec<f64>,
}

impl ConvectionDiffusionTDO {
    fn new(
        space: &HCurlSpace<Mesh<3>>,
        ess_tdofs: Vec<u32>,
        alpha: f64,
        sigma: f64,
        order: u8,
        qp_override: i32,
    ) -> Self {
        let n = space.n_dofs();

        // Mass form: VectorMassIntegrator.
        let mass = VectorMassIntegrator { alpha: 1.0 };
        let mass_qp = if qp_override > 0 { qp_override as u8 } else { 2 * order };
        let mut m_mat = VectorAssembler::assemble_bilinear(space, &[&mass], mass_qp);

        // Eliminate essential DOFs.
        let mut zero_rhs = vec![0.0_f64; n];
        let zero_vals = vec![0.0_f64; ess_tdofs.len()];
        fem_space::apply_dirichlet(&mut m_mat, &mut zero_rhs, &ess_tdofs, &zero_vals);

        // Stiffness: CurlCurlIntegrator(sigma) + MixedWeakCurlCrossIntegrator(alpha * velocity).
        let curl_curl = CurlCurlIntegrator { mu: -sigma };
        let vel = ScaledVelocity { alpha };
        let conv = MixedWeakCurlCrossIntegrator { velocity: vel };
        let k_qp = if qp_override > 0 { qp_override as u8 } else { 2 * order - 2 };
        let k_mat = VectorAssembler::assemble_bilinear(space, &[&curl_curl, &conv], k_qp);

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

/// SSP-RK3 step (MFEM RK3SSPSolver::Step).
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

/// Block-to-cylinder transfer map for H(curl).
struct BlockToCylinderMap {
    pairs: Vec<(u32, u32)>,
}

impl BlockToCylinderMap {
    fn new(
        cyl_space: &HCurlSpace<Mesh<3>>,
        blk_space: &HCurlSpace<Mesh<3>>,
        interface_dofs_cyl: &[u32],
        interface_dofs_blk: &[u32],
    ) -> Self {
        fn coord_key(c: &[f64]) -> [i64; 3] {
            [
                (c[0] * 1e9).round() as i64,
                (c[1] * 1e9).round() as i64,
                (c[2] * 1e9).round() as i64,
            ]
        }
        let blk_coords = blk_space.dof_coords();
        let mut by_coord: std::collections::HashMap<[i64; 3], u32> =
            std::collections::HashMap::new();
        for &d in interface_dofs_blk {
            by_coord.insert(coord_key(&blk_coords[d as usize]), d);
        }
        let cyl_coords = cyl_space.dof_coords();
        let mut pairs = Vec::with_capacity(interface_dofs_cyl.len());
        for &c in interface_dofs_cyl {
            let key = coord_key(&cyl_coords[c as usize]);
            let &b = by_coord
                .get(&key)
                .unwrap_or_else(|| panic!("BlockToCylinderMap: no block dof at cyl interface dof {c}"));
            pairs.push((c, b));
        }
        BlockToCylinderMap { pairs }
    }

    fn transfer(&self, src: &[f64], dst: &mut [f64]) {
        for &(c, b) in &self.pairs {
            dst[c as usize] = src[b as usize];
        }
    }
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

fn marker_to_tags(marker: &[i32]) -> Vec<i32> {
    (1..=marker.len() as i32)
        .filter(|&a| marker[(a - 1) as usize] != 0)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

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

    let parent = read_mfem_file(&mesh_file).expect("failed to read parent mesh");
    let parent_mesh: Mesh<3> = parent.mesh3d.expect("multidomain-hex.mesh must be 3D");
    assert!(
        parent_mesh.elem_type == fem_mesh::ElementType::Hex8,
        "multidomain-hex.mesh must be an Hex8 mesh"
    );
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

    // Cylinder submesh (domain attribute 1) and its HCurlSpace.
    let cylinder_submesh = extract_submesh_3d(&parent_mesh, &[1]);
    let fes_cylinder = HCurlSpace::new(cylinder_submesh.mesh.clone(), order);
    {
        let m = &cylinder_submesh.mesh;
        println!(
            "Cylinder submesh: NE={} NV={} ndofs={}",
            m.n_elems(),
            m.n_nodes(),
            fes_cylinder.n_dofs()
        );
    }

    // Essential DOFs: inflow (bdr attr 8) and interface (bdr attr 9).
    let mesh_cyl = fes_cylinder.mesh();
    let mut inflow_attributes = vec![0_i32; 9];
    inflow_attributes[7] = 1;
    let mut inner_cylinder_wall_attributes = vec![0_i32; 9];
    inner_cylinder_wall_attributes[8] = 1;
    let mut ess_tdofs =
        hcurl_boundary_dofs(mesh_cyl, &fes_cylinder, &marker_to_tags(&inflow_attributes));
    ess_tdofs.extend(hcurl_boundary_dofs(
        mesh_cyl,
        &fes_cylinder,
        &marker_to_tags(&inner_cylinder_wall_attributes),
    ));
    ess_tdofs.sort_unstable();
    ess_tdofs.dedup();
    println!("Cylinder ess vdofs: {}", ess_tdofs.len());
    let mut cd_tdo =
        ConvectionDiffusionTDO::new(&fes_cylinder, ess_tdofs, 1.0, 1.0e-1, order, qp);

    let mut magnetic_field_cylinder = vec![0.0_f64; fes_cylinder.n_dofs()];

    // Block submesh (domain attribute 2), diffusion-only (alpha=0, sigma=1).
    let block_submesh = extract_submesh_3d(&parent_mesh, &[2]);
    let fes_block = HCurlSpace::new(block_submesh.mesh.clone(), order);
    {
        let m = &block_submesh.mesh;
        println!(
            "Block submesh: NE={} NV={} ndofs={}",
            m.n_elems(),
            m.n_nodes(),
            fes_block.n_dofs()
        );
    }

    let mesh_blk = fes_block.mesh();
    let mut block_wall_attributes = vec![0_i32; 9];
    block_wall_attributes[0] = 1;
    block_wall_attributes[1] = 1;
    block_wall_attributes[2] = 1;
    block_wall_attributes[3] = 1;
    let block_ess_tdofs =
        hcurl_boundary_dofs(mesh_blk, &fes_block, &marker_to_tags(&block_wall_attributes));
    println!("Block ess vdofs (walls 1-4): {}", block_ess_tdofs.len());

    let mut d_tdo =
        ConvectionDiffusionTDO::new(&fes_block, block_ess_tdofs, 0.0, 1.0, order, qp);

    // Initial condition: H = square_xy on the outer walls (tangent projection).
    // For simplicity, we set the initial condition to zero everywhere.
    let mut magnetic_field_block = vec![0.0_f64; fes_block.n_dofs()];

    // Block → cylinder transfer map.
    let interface_dofs_cyl =
        hcurl_boundary_dofs(mesh_cyl, &fes_cylinder, &marker_to_tags(&inner_cylinder_wall_attributes));
    let interface_dofs_blk =
        hcurl_boundary_dofs(mesh_blk, &fes_block, &marker_to_tags(&inner_cylinder_wall_attributes));
    let field_block_to_cylinder_map = BlockToCylinderMap::new(
        &fes_cylinder,
        &fes_block,
        &interface_dofs_cyl,
        &interface_dofs_blk,
    );

    // Time loop.
    let mut t = 0.0_f64;
    let mut last_step = false;
    let mut ti = 1_usize;
    while !last_step {
        if t + dt >= t_final - dt / 2.0 {
            last_step = true;
        }

        // Advance the diffusion equation on the outer block.
        rk3ssp_step(&mut d_tdo, &mut magnetic_field_block, &mut t, dt);

        // Transfer the block solution onto the cylinder interface.
        field_block_to_cylinder_map.transfer(&magnetic_field_block, &mut magnetic_field_cylinder);

        // Advance the convection-diffusion equation inside the cylinder.
        rk3ssp_step(&mut cd_tdo, &mut magnetic_field_cylinder, &mut t, dt);

        if last_step || ti % vis_steps == 0 {
            let bsum: f64 = magnetic_field_block.iter().sum();
            let csum: f64 = magnetic_field_cylinder.iter().sum();
            println!(
                "step {ti}, t = {t}  block: sum={bsum:.6e}  cyl: sum={csum:.6e}"
            );
        }

        ti += 1;
    }
}
