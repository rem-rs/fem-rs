//! D708 — `fem_space::hdiv` RT1-hex on **curved** hexes (P2 geometry):
//! ordered slot tables, local-mass scatter, submesh transfer pairing and the
//! multidomain_rt dataflow, pinned against MFEM 4.10 gold.
//!
//! Gold provenance (probes `tmp/d708/probe_hex.cpp`, `rt_ser_dump.cpp`, WSL
//! MFEM 4.10 serial tree, mesh `tests/data/d708_multidomain_hex.mesh` after
//! one `UniformRefinement`, cylinder = domain attr 1):
//!
//! * `d708_curved_hex.mesh` — one wall-adjacent hex of the refined cylinder,
//!   extracted with its H1_3D_P2 node row by MFEM itself (`Mesh::Save`).
//! * `d708_curved_hex_rt1_gold.txt` — its ordered `GetElementVDofs` (MFEM
//!   sign coding: `-d-1` = negative orientation sign) and the
//!   `VectorFEMassIntegrator` local matrix (significant entries, default
//!   rule = `Trans.OrderW() + 2·GetOrder` = 9 on the curved map).
//! * `d708_cyl_vdofs_gold.txt` — the ordered `GetElementVDofs` stream of all
//!   288 refined-cylinder elements (10368 slots).
//! * `d708_cyl_transfer_map_gold.txt` — the composed block→cylinder
//!   `SubMesh::CreateTransferMap`, probed pair-by-pair with unit vectors
//!   (384 pairs = 96 interface faces × 4 slots).
//!
//! Findings locked here (round 68):
//! 1. The `HDivSpace` slot/sign tables and the assembled mass are already
//!    MFEM-exact on curved hexes — the round-67 hypothesis "RT1-hex interior
//!    slot/sign correspondence" is disproven, and these pins keep it so.
//! 2. The multidomain_rt cylinder divergence was a **dataflow** defect in the
//!    serial port, not a space defect: MFEM's `Transfer` reads *and* writes
//!    the destination GridFunction, so with the upstream loop (no
//!    `SetFromTrueDofs` between `Step` and `Transfer`) each step's transfer
//!    discards the cylinder's RK3 evolution and restarts it from the previous
//!    transfer output.  `d708_cylinder_dataflow_2step` pins the corrected
//!    semantics (the miniapp must keep it green).

use std::collections::{HashMap, HashSet};

use fem_assembly::postproc::coefficient::{CoeffCtx, VectorCoeff};
use fem_assembly::standard::{
    DivDivIntegrator, MixedWeakGradDotIntegrator, VectorMassIntegrator,
    mfem_div_div_quad_order_rt_hex, mfem_vector_mass_quad_order_rt_hex,
    mfem_weak_grad_dot_quad_order_rt_hex,
};
use fem_assembly::vector_assembler::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::submesh::extract_submesh_3d;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_hex8_uniform, Mesh};
use fem_solver::{solve_pcg_dsmoother, SolverConfig};
use fem_space::dof_manager::FaceKey;
use fem_space::fe_space::FESpace;
use fem_space::hdiv::HDivSpace;

fn data(rel: &str) -> String {
    format!(
        "{}/tests/data/{}",
        env!("CARGO_MANIFEST_DIR"),
        rel
    )
}

/// MFEM vdof coding: negative sign ⇒ `-(dof)-1`.
fn coded(dof: u32, sign: f64) -> i64 {
    if sign < 0.0 {
        -(dof as i64) - 1
    } else {
        dof as i64
    }
}

// ─── 1. Curved single hex: ordered table + local mass ──────────────────────

#[test]
fn d708_curved_single_hex_ordered_table_and_local_mass() {
    let mfem = read_mfem_file(data("d708_curved_hex.mesh")).expect("read curved hex");
    let mesh: Mesh<3> = mfem.mesh3d.expect("3-D");
    assert_eq!(mesh.geom_order(), 2, "fixture must carry P2 geometry");
    let space = HDivSpace::new(mesh.clone(), 1);
    assert_eq!(space.n_dofs(), 36, "RT1 hex: 24 face + 12 interior");

    let dofs = space.element_dofs(0);
    let signs = space.element_signs(0);
    let got: Vec<String> = (0..dofs.len())
        .map(|i| format!("VDOFS {} {}", i, coded(dofs[i], signs[i])))
        .collect();

    let gold = std::fs::read_to_string(data("d708_curved_hex_rt1_gold.txt")).unwrap();
    let mut gold_vdofs: Vec<String> = Vec::new();
    let mut gold_mass: Vec<(usize, usize, f64)> = Vec::new();
    for line in gold.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("VDOFS") => {
                gold_vdofs.push(format!("VDOFS {} {}", it.next().unwrap(), it.next().unwrap()));
            }
            Some("LMASS") => gold_mass.push((
                it.next().unwrap().parse().unwrap(),
                it.next().unwrap().parse().unwrap(),
                it.next().unwrap().parse().unwrap(),
            )),
            _ => {}
        }
    }
    assert_eq!(got.len(), gold_vdofs.len(), "slot count");
    for (i, (g, e)) in got.iter().zip(gold_vdofs.iter()).enumerate() {
        assert_eq!(g, e, "ordered slot table differs at slot {i}");
    }

    // Local mass: on a 1-element mesh the assembled global matrix IS the
    // element matrix scattered with signs[i]·signs[j] — compare entrywise
    // against MFEM's VectorFEMassIntegrator on the same curved map.
    let mass = VectorMassIntegrator { alpha: 1.0 };
    let qp = mfem_vector_mass_quad_order_rt_hex(1, mesh.geom_order());
    assert_eq!(qp, 9, "MFEM default rule on the curved hex");
    let m = VectorAssembler::assemble_bilinear(&space, &[&mass], qp);
    for (i, j, v) in gold_mass {
        let gv = m.get(i, j);
        let gi = dofs[i] as usize;
        let gj = dofs[j] as usize;
        let assembled = m.get(gi, gj) * signs[i] * signs[j];
        assert!(
            (assembled - v).abs() <= 1e-9 * (v.abs() + 1e-300),
            "local mass ({i},{j}): {assembled} vs MFEM {v} (raw get {gv})"
        );
    }
}

// ─── 2. Refined cylinder: full ordered table ────────────────────────────────

#[test]
fn d708_cylinder_ordered_vdofs_table() {
    let mfem = read_mfem_file(data("d708_multidomain_hex.mesh")).expect("read parent mesh");
    let parent: Mesh<3> = mfem.mesh3d.expect("3-D");
    let all: Vec<u32> = (0..parent.n_elems() as u32).collect();
    let parent = refine_hex8_uniform(&parent, &all).0;
    let sub = extract_submesh_3d(&parent, &[1]);
    let mesh = sub.mesh;
    assert_eq!(mesh.n_elems(), 288, "refined cylinder element count");
    let space = HDivSpace::new(mesh.clone(), 1);
    assert_eq!(space.n_dofs(), 7296);

    let gold = std::fs::read_to_string(data("d708_cyl_vdofs_gold.txt")).unwrap();
    let mut lines = gold.lines();
    for e in 0..mesh.n_elems() as u32 {
        let dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        for (i, (&d, &s)) in dofs.iter().zip(signs).enumerate() {
            let line = lines.next().unwrap_or_else(|| panic!("gold exhausted at e{e}"));
            let expect = format!("CVDOFS {e} {i} {}", coded(d, s));
            assert_eq!(line, expect, "ordered cylinder vdofs diverge");
        }
    }
    assert!(lines.next().is_none(), "gold table has leftover lines");
}

// ─── 3./4. Transfer pairing + 2-step dataflow (miniapp replica) ────────────

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

struct ScaledVelocity {
    alpha: f64,
}

impl VectorCoeff for ScaledVelocity {
    fn eval(&self, ctx: &CoeffCtx<'_>, out: &mut [f64]) {
        velocity_profile(ctx.x, out);
        for v in out.iter_mut() {
            *v *= self.alpha;
        }
    }
}

fn square_xy(x: &[f64]) -> Vec<f64> {
    vec![-2.0 * x[1], 2.0 * x[0], 0.0]
}

/// Cyclic-corner-ordered hex faces with a boundary tag in `tags`
/// (verbatim miniapp helper).
fn boundary_hex_faces(mesh: &Mesh<3>, tags: &[i32]) -> Vec<[u32; 4]> {
    let mut face_tag_by_set: HashMap<[u32; 4], i32> = HashMap::new();
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

    let mut seen: HashSet<[u32; 4]> = HashSet::new();
    let mut out: Vec<[u32; 4]> = Vec::new();
    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        for lf in &HEX_FACES {
            let fns = [ns[lf[0]], ns[lf[1]], ns[lf[2]], ns[lf[3]]];
            let mut key = fns;
            key.sort_unstable();
            if seen.contains(&key) {
                continue;
            }
            let Some(&tag) = face_tag_by_set.get(&key) else {
                continue;
            };
            if !tags.contains(&tag) {
                continue;
            }
            seen.insert(key);
            out.push(fns);
        }
    }
    out
}

/// Full face blocks of `faces` (miniapp helper).
fn hdiv_boundary_dofs(space: &HDivSpace<Mesh<3>>, faces: &[[u32; 4]]) -> Vec<u32> {
    let mut out: HashSet<u32> = HashSet::new();
    for f in faces {
        let mut s4 = *f;
        s4.sort_unstable();
        let fk = FaceKey::new(s4[0], s4[1], s4[2]);
        if let Some(ds) = space.face_dofs(fk) {
            out.extend(ds);
        }
    }
    let mut v: Vec<u32> = out.into_iter().collect();
    v.sort_unstable();
    v
}

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
        space: &HDivSpace<Mesh<3>>,
        ess_tdofs: Vec<u32>,
        alpha: f64,
        kappa: f64,
        order: u8,
    ) -> Self {
        let n = space.n_dofs();

        let mass = VectorMassIntegrator { alpha: 1.0 };
        let geom_order = space.mesh().geom_order();
        let mass_qp = mfem_vector_mass_quad_order_rt_hex(order, geom_order);
        let mut m_mat = VectorAssembler::assemble_bilinear(space, &[&mass], mass_qp);

        let mut zero_rhs = vec![0.0_f64; n];
        let zero_vals = vec![0.0_f64; ess_tdofs.len()];
        fem_space::apply_dirichlet(&mut m_mat, &mut zero_rhs, &ess_tdofs, &zero_vals);

        let div_div = DivDivIntegrator { kappa: -kappa };
        let vel = ScaledVelocity { alpha };
        let conv = MixedWeakGradDotIntegrator { velocity: vel };
        let div_qp = mfem_div_div_quad_order_rt_hex(order);
        let conv_qp = mfem_weak_grad_dot_quad_order_rt_hex(order, geom_order);
        let div_mat = VectorAssembler::assemble_bilinear(space, &[&div_div], div_qp);
        let conv_mat = VectorAssembler::assemble_bilinear(space, &[&conv], conv_qp);
        let k_mat = div_mat.add(&conv_mat);

        ConvectionDiffusionTDO {
            m_mat,
            k_mat,
            b: vec![0.0_f64; n],
            ess_tdofs,
            solve_cfg: SolverConfig {
                rtol: 1e-8,
                atol: 0.0,
                max_iter: 100,
                verbose: false,
                ..SolverConfig::default()
            },
            t1: vec![0.0; n],
        }
    }

    fn mult(&mut self, u: &[f64], du_dt: &mut [f64]) {
        self.k_mat.spmv(u, &mut self.t1);
        for (t, &bv) in self.t1.iter_mut().zip(self.b.iter()) {
            *t += bv;
        }
        solve_pcg_dsmoother(&self.m_mat, &self.t1, du_dt, &self.solve_cfg)
            .expect("mass CG solve failed");
        for &d in &self.ess_tdofs {
            du_dt[d as usize] = 0.0;
        }
    }
}

fn rk3ssp_step(f: &mut ConvectionDiffusionTDO, x: &mut [f64], t: &mut f64, dt: f64) {
    let n = x.len();
    let mut k = vec![0.0_f64; n];
    let mut y = vec![0.0_f64; n];

    f.mult(x, &mut k);
    for i in 0..n {
        y[i] = x[i] + dt * k[i];
    }

    f.mult(&y, &mut k);
    for i in 0..n {
        y[i] += dt * k[i];
    }
    for i in 0..n {
        y[i] = 3.0_f64 / 4.0 * x[i] + 1.0_f64 / 4.0 * y[i];
    }

    f.mult(&y, &mut k);
    for i in 0..n {
        y[i] += dt * k[i];
    }
    for i in 0..n {
        x[i] = 1.0_f64 / 3.0 * x[i] + 2.0_f64 / 3.0 * y[i];
    }

    *t += dt;
}

fn coord_key(c: &[f64]) -> [i64; 3] {
    [
        (c[0] * 1e9).round() as i64,
        (c[1] * 1e9).round() as i64,
        (c[2] * 1e9).round() as i64,
    ]
}

/// `(cylinder dof, block dof, orientation sign)` interface pairing — the
/// miniapp's `BlockToCylinderMap` (anchor-key pairing per physical face).
type FaceDofs = HashMap<
    ([i64; 3], [i64; 3], [i64; 3], [i64; 3]),
    HashMap<[i64; 3], (u32, [f64; 3])>,
>;

fn geometric_factors(space: &HDivSpace<Mesh<3>>) -> Vec<[f64; 3]> {
    let mut g = vec![[0.0_f64; 3]; space.n_dofs()];
    for i in 0..3 {
        let v = space
            .interpolate_vector(&|_| {
                let mut e = vec![0.0_f64; 3];
                e[i] = 1.0;
                e
            })
            .as_slice()
            .to_vec();
        for (d, gd) in g.iter_mut().enumerate() {
            gd[i] = v[d];
        }
    }
    g
}

fn face_inventory(
    space: &HDivSpace<Mesh<3>>,
    faces: &[[u32; 4]],
    g: &[[f64; 3]],
) -> FaceDofs {
    let mesh = space.mesh();
    let coords = space.dof_coords();
    let mut inv: FaceDofs = HashMap::new();
    for f in faces {
        let mut s4 = *f;
        s4.sort_unstable();
        let fk = FaceKey::new(s4[0], s4[1], s4[2]);
        let Some(ds) = space.face_dofs(fk) else {
            continue;
        };
        let mut cs: Vec<[i64; 3]> =
            f.iter().map(|n| coord_key(&mesh.node_coords(*n))).collect();
        cs.sort_unstable();
        let key = (cs[0], cs[1], cs[2], cs[3]);
        let entry = inv.entry(key).or_default();
        for d in ds {
            entry.insert(coord_key(&coords[d as usize]), (d, g[d as usize]));
        }
    }
    inv
}

fn block_to_cylinder_pairs(
    cyl_space: &HDivSpace<Mesh<3>>,
    cyl_faces: &[[u32; 4]],
    blk_space: &HDivSpace<Mesh<3>>,
    blk_faces: &[[u32; 4]],
) -> Vec<(u32, u32, f64)> {
    let cyl_g = geometric_factors(cyl_space);
    let blk_g = geometric_factors(blk_space);
    let blk_inv = face_inventory(blk_space, blk_faces, &blk_g);
    let cyl_inv = face_inventory(cyl_space, cyl_faces, &cyl_g);

    let mut pairs: Vec<(u32, u32, f64)> = Vec::new();
    for (key, cyl_map) in &cyl_inv {
        let Some(blk_map) = blk_inv.get(key) else {
            panic!("block side missing interface face {key:?}");
        };
        assert_eq!(
            cyl_map.len(),
            blk_map.len(),
            "face {key:?} dof count mismatch"
        );
        for (&pos, &(c, gc)) in cyl_map {
            let Some(&(b, gb)) = blk_map.get(&pos) else {
                panic!("no block dof on face {key:?} at cyl dof {c}");
            };
            let dot = gc[0] * gb[0] + gc[1] * gb[1] + gc[2] * gb[2];
            pairs.push((c, b, if dot > 0.0 { 1.0 } else { -1.0 }));
        }
    }
    pairs.sort_by_key(|&(c, b, _)| (c, b));
    pairs
}

/// The full refined-cylinder pipeline state shared by tests 3 and 4.
struct Pipeline {
    cyl_ess: Vec<u32>,
    pairs: Vec<(u32, u32, f64)>,
    cd_tdo: ConvectionDiffusionTDO,
    d_tdo: ConvectionDiffusionTDO,
    field_block: Vec<f64>,
    n_cyl_dofs: usize,
}

fn build_pipeline() -> Pipeline {
    let mfem = read_mfem_file(data("d708_multidomain_hex.mesh")).expect("read parent mesh");
    let parent: Mesh<3> = mfem.mesh3d.expect("3-D");
    let all: Vec<u32> = (0..parent.n_elems() as u32).collect();
    let parent = refine_hex8_uniform(&parent, &all).0;

    let cylinder_submesh = extract_submesh_3d(&parent, &[1]);
    let fes_cylinder = HDivSpace::new(cylinder_submesh.mesh.clone(), 1);

    let mesh_cyl = fes_cylinder.mesh();
    let cyl_inflow_faces = boundary_hex_faces(mesh_cyl, &[8_i32]);
    let cyl_interface_faces = boundary_hex_faces(mesh_cyl, &[9_i32]);
    let mut cyl_ess = hdiv_boundary_dofs(&fes_cylinder, &cyl_inflow_faces);
    cyl_ess.extend(hdiv_boundary_dofs(&fes_cylinder, &cyl_interface_faces));
    cyl_ess.sort_unstable();
    cyl_ess.dedup();

    let block_submesh = extract_submesh_3d(&parent, &[2]);
    let fes_block = HDivSpace::new(block_submesh.mesh.clone(), 1);
    let mesh_blk = fes_block.mesh();
    let block_wall_faces = boundary_hex_faces(mesh_blk, &(1..=8).collect::<Vec<i32>>());
    let blk_ess = hdiv_boundary_dofs(&fes_block, &block_wall_faces);

    let cd_tdo = ConvectionDiffusionTDO::new(&fes_cylinder, cyl_ess.clone(), 1.0, 1.0e-1, 1);
    let blk_ic = fes_block.interpolate_vector(&square_xy).as_slice().to_vec();
    let d_tdo = ConvectionDiffusionTDO::new(&fes_block, blk_ess.clone(), 0.0, 1.0, 1);

    // IC: ProjectBdrCoefficientNormal(square_xy) on the block walls.
    let mut field_block = vec![0.0_f64; fes_block.n_dofs()];
    for &d in &blk_ess {
        field_block[d as usize] = blk_ic[d as usize];
    }

    let blk_interface_faces = boundary_hex_faces(mesh_blk, &[9_i32]);
    let pairs = block_to_cylinder_pairs(
        &fes_cylinder,
        &cyl_interface_faces,
        &fes_block,
        &blk_interface_faces,
    );
    let n_cyl_dofs = fes_cylinder.n_dofs();

    Pipeline {
        cyl_ess,
        pairs,
        cd_tdo,
        d_tdo,
        field_block,
        n_cyl_dofs,
    }
}

#[test]
fn d708_cylinder_transfer_map_matches_mfem() {
    let p = build_pipeline();
    let got: HashSet<(u32, u32, i64)> = p
        .pairs
        .iter()
        .map(|&(c, b, s)| (c, b, s as i64))
        .collect();

    let gold = std::fs::read_to_string(data("d708_cyl_transfer_map_gold.txt")).unwrap();
    let mut n = 0_usize;
    for line in gold.lines() {
        let mut it = line.split_whitespace();
        assert_eq!(it.next(), Some("PAIR"));
        let c: u32 = it.next().unwrap().parse().unwrap();
        let b: u32 = it.next().unwrap().parse().unwrap();
        let s: f64 = it.next().unwrap().parse().unwrap();
        assert!(
            got.contains(&(c, b, s as i64)),
            "pair (cyl {c} <- blk {b}, sign {s:+}) missing/wrong in fem-rs map"
        );
        n += 1;
    }
    assert_eq!(n, 384, "interface pair count");
    assert_eq!(got.len(), 384, "fem-rs map must be exactly the gold set");
}

#[test]
fn d708_cylinder_dataflow_2step_free_sums() {
    // MFEM 4.10 free-dof sums (tmp/d690/cpp_decomp_early.txt, full
    // precision): the cylinder trajectory of multidomain_rt under the
    // upstream dataflow (transfer uploads the STALE pressure_cylinder_gf,
    // discarding the RK3 evolution of the previous step).
    const STEP1_FREE_SUM: f64 = -4.6111574118297711e-07;
    const STEP2_FREE_SUM: f64 = -9.0597541046775525e-07;
    const STEP2_FREE_SSQ: f64 = 2.0719607940123502e-13;

    let mut p = build_pipeline();
    let dt = 1.0e-5;
    let t_final = 2.0e-5;

    let mut field_cylinder = vec![0.0_f64; p.n_cyl_dofs];
    // Shadow of MFEM's `pressure_cylinder_gf`: the transfer's dst.  The RK3
    // result is never SetFromTrueDofs back — each transfer restarts from the
    // PREVIOUS transfer output (interiors of the previous upload).
    let mut gf_state = vec![0.0_f64; p.n_cyl_dofs];

    let mut t = 0.0_f64;
    let mut last_step = false;
    let mut ti = 1_usize;
    while !last_step {
        if t + dt >= t_final - dt / 2.0 {
            last_step = true;
        }

        rk3ssp_step(&mut p.d_tdo, &mut p.field_block, &mut t, dt);

        let mut transferred = gf_state.clone();
        for &(c, b, s) in &p.pairs {
            transferred[c as usize] = s * p.field_block[b as usize];
        }
        gf_state = transferred;
        field_cylinder.copy_from_slice(&gf_state);

        rk3ssp_step(&mut p.cd_tdo, &mut field_cylinder, &mut t, dt);

        let ess: HashSet<u32> = p.cyl_ess.iter().copied().collect();
        let mut free_sum = 0.0_f64;
        let mut free_ssq = 0.0_f64;
        for (d, v) in field_cylinder.iter().enumerate() {
            if !ess.contains(&(d as u32)) {
                free_sum += v;
                free_ssq += v * v;
            }
        }
        let rel = |a: f64, b: f64| (a - b).abs() / (b.abs() + 1e-300);
        match ti {
            1 => {
                assert!(rel(free_sum, STEP1_FREE_SUM) < 1e-9, "step 1 free_sum {free_sum:e}");
            }
            2 => {
                assert!(rel(free_sum, STEP2_FREE_SUM) < 1e-9, "step 2 free_sum {free_sum:e}");
                assert!(rel(free_ssq, STEP2_FREE_SSQ) < 1e-9, "step 2 free_ssq {free_ssq:e}");
            }
            _ => unreachable!(),
        }
        ti += 1;
    }
}
