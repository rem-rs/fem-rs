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
//!
//! Serial mechanism mapping (additions w.r.t. `multidomain.rs`):
//!
//! | MFEM (parallel)                             | this port                                     |
//! |---------------------------------------------|-----------------------------------------------|
//! | `GetEssentialTrueDofs(bdr_attrs)` (ND)      | full tangential blocks on bdr faces: all `order` edge dofs + all `2·order·(order−1)` face dofs (`hcurl_boundary_dofs`) |
//! | `ProjectBdrCoefficientTangent(square_xy)`   | `HCurlSpace::interpolate_vector(square_xy)` restricted to the wall dofs (exact for this linear field: the tangential trace lies in the trace space, so interpolation == L2 projection) |
//! | `ParSubMesh::CreateTransferMap` (ND)        | entity-matched, orientation-corrected dof pairing (`BlockToCylinderMap`) |
//!
//! The ND transfer map cannot match dofs by coordinates alone (the H1 trick):
//! an ND edge carries `order` dofs at Gauss-Legendre points along the
//! *canonical* (min→max node id) direction, and the two submeshes number the
//! same physical edge endpoints differently, so on ~half the edges the first
//! dofs sit at mirrored points; ND face dofs take the face-creating element's
//! anchor points, which differ per submesh for non-parallelogram faces.
//! Instead, dofs are paired *per physical entity* (edge / quad face, keyed by
//! sorted corner coordinates) with the index-free "geometric factor"
//! calibration `g(d)` of each dof (see `BlockToCylinderMap::new`): two dofs on
//! the same physical entity carry the same functional up to orientation, i.e.
//! `g_cyl = ±g_blk`, and the value map is `dst[c] = sign(g_c·g_b)·src[b]`.

use std::collections::{HashMap, HashSet};

use fem_assembly::postproc::coefficient::{CoeffCtx, VectorCoeff};
use fem_assembly::standard::{
    CurlCurlIntegrator, MixedWeakCurlCrossIntegrator, VectorMassIntegrator,
    mfem_vector_mass_quad_order_nd_hex, mfem_weak_curl_cross_quad_order_nd_hex,
};
use fem_assembly::vector_assembler::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::submesh::extract_submesh_3d;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_hex8_uniform, Mesh};
use fem_solver::{solve_pcg_dsmoother, SolverConfig};
use fem_space::dof_manager::{EdgeKey, QuadFaceKey};
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

/// Scaled velocity coefficient: `alpha * velocity`, matching the C++
/// `aq = ScalarVectorProductCoefficient(alpha, *q)` (multidomain_nd.cpp:106).
/// (The C++ folds the minus sign into `CurlCurlIntegrator(-sigma)` only.)
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

/// C++ `square_xy` (multidomain_nd.cpp:74-82): a vector field parallel to the
/// xy plane wrapping counter-clockwise, used for the initial condition.
fn square_xy(x: &[f64]) -> Vec<f64> {
    vec![-2.0 * x[1], 2.0 * x[0], 0.0]
}

/// Cyclic-corner-ordered hex faces of `mesh` whose boundary tag is listed in
/// `tags`, deduplicated by sorted vertex set.
///
/// The element face table (not the stored boundary-face vertex order) provides
/// the proper cyclic corner order — submeshes store quad boundary faces in
/// sorted-vertex order, which derives wrong edge keys (see the note on
/// `hex_boundary_dofs` in `multidomain.rs`).
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

/// All H(curl) dofs lying on `faces`: the full edge blocks (`order` dofs per
/// edge, Gauss-Legendre points for `order >= 2`) plus the full quad-face
/// blocks (`2·order·(order−1)` dofs per face) — MFEM's
/// `GetEssentialTrueDofs` constrains every tangential trace dof of the
/// boundary faces, not just the first of each block.
fn hcurl_boundary_dofs(space: &HCurlSpace<Mesh<3>>, faces: &[[u32; 4]]) -> Vec<u32> {
    let mut out: HashSet<u32> = HashSet::new();
    for f in faces {
        for i in 0..4 {
            let ek = EdgeKey::new(f[i], f[(i + 1) % 4]);
            if let Some(ds) = space.edge_dofs(ek) {
                out.extend(ds);
            }
        }
        let qk = QuadFaceKey::new(f[0], f[1], f[2], f[3]);
        if let Some(ds) = space.quad_face_dofs(qk) {
            out.extend(ds);
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

        // Mass form: VectorMassIntegrator. D678: MFEM's true default rule is
        // `Trans.OrderW() + 2·GetOrder()` with `GetOrder() = k` for
        // ND_HexahedronElement (fe_nd.cpp:28) and `OrderW() = 3g − 1` for the
        // Qk hex map (affine 2k + 2 / P2-curved 2k + 5) — the round-64
        // hardcode `2p + 2` dropped OrderW (D667 probe).
        let mass = VectorMassIntegrator { alpha: 1.0 };
        let geom_order = space.mesh().geom_order();
        let mass_qp = if qp_override > 0 {
            qp_override as u8
        } else {
            mfem_vector_mass_quad_order_nd_hex(order, geom_order)
        };
        let mut m_mat = VectorAssembler::assemble_bilinear(space, &[&mass], mass_qp);

        // Eliminate essential DOFs.
        let mut zero_rhs = vec![0.0_f64; n];
        let zero_vals = vec![0.0_f64; ess_tdofs.len()];
        fem_space::apply_dirichlet(&mut m_mat, &mut zero_rhs, &ess_tdofs, &zero_vals);

        // Stiffness: CurlCurlIntegrator(-sigma) + MixedWeakCurlCrossIntegrator(alpha * velocity)
        // (C++ multidomain_nd.cpp:107-109). D678: MFEM rules — CurlCurl on a
        // Qk ND element selects its own `2·GetOrder = 2k` order internally
        // (the integrator's `integration_order_for`, never the fallback
        // below); MixedWeakCurlCross derives the base
        // `MixedVectorIntegrator::GetIntegrationOrder = trial + test + OrderW
        // = 2k + 3g − 1` (ND2: affine 6 / curved 9) — replacing the
        // `2·order + 2` hardcode that mis-served both on curved maps.
        let curl_curl = CurlCurlIntegrator { mu: -sigma };
        let vel = ScaledVelocity { alpha };
        let conv = MixedWeakCurlCrossIntegrator { velocity: vel };
        let k_qp = if qp_override > 0 {
            qp_override as u8
        } else {
            mfem_weak_curl_cross_quad_order_nd_hex(order, geom_order)
        };
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

/// Block-to-cylinder transfer map for H(curl): the serial equivalent of
/// `ParSubMesh::CreateTransferMap(magnetic_field_block_gf,
/// magnetic_field_cylinder_gf)` restricted to the interface dofs (the
/// cylinder keeps its own values on non-interface dofs, as in C++).
///
/// ND dofs are canonical functionals tied to oriented mesh entities, and the
/// two submeshes orient the shared interface entities differently (node ids
/// are unrelated). The map therefore pairs dofs per *physical entity* (edge /
/// quad face keyed by sorted corner coordinates) and corrects the orientation
/// through each dof's geometric factor `g(d)` — the vector-valued density of
/// its functional, recovered from three constant-field interpolations:
/// `interpolate_vector(e_i)[d] = e_i · g(d)`. Two dofs on the same physical
/// entity carry the same functional up to orientation, so `g_cyl = ±g_blk`
/// (equal magnitude, parallel) and `value_cyl = sign(g_c·g_b) · value_blk`.
///
/// `new` verifies this relation per pair with an interpolated linear field
/// before accepting the map.
struct BlockToCylinderMap {
    /// (cylinder dof, block dof, orientation sign) triples on the interface.
    pairs: Vec<(u32, u32, f64)>,
}

/// Physical entity key: sorted quantized endpoint coordinates (edge) or corner
/// coordinates (quad face).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum EntityKey {
    Edge([i64; 3], [i64; 3]),
    Quad([i64; 3], [i64; 3], [i64; 3], [i64; 3]),
}

fn coord_key(c: &[f64]) -> [i64; 3] {
    [
        (c[0] * 1e9).round() as i64,
        (c[1] * 1e9).round() as i64,
        (c[2] * 1e9).round() as i64,
    ]
}

/// (dof position key) → dof, plus each dof's geometric factor, per entity.
type EntityDofs = HashMap<EntityKey, HashMap<[i64; 3], (u32, [f64; 3])>>;

impl BlockToCylinderMap {
    fn new(
        cyl_space: &HCurlSpace<Mesh<3>>,
        cyl_faces: &[[u32; 4]],
        blk_space: &HCurlSpace<Mesh<3>>,
        blk_faces: &[[u32; 4]],
    ) -> Self {
        /// Geometric factor `g(d)` of every dof (see struct docs).
        fn factors(space: &HCurlSpace<Mesh<3>>) -> Vec<[f64; 3]> {
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

        /// Entity → (dof position → (dof, geometric factor)) inventory of the
        /// interface: the four edges and the quad itself of every face. Entity
        /// keys use the *node* coordinates of the corners; dof positions use
        /// the dof-coordinate table.
        fn inventory(space: &HCurlSpace<Mesh<3>>, faces: &[[u32; 4]], g: &[[f64; 3]]) -> EntityDofs {
            let mesh = space.mesh();
            let coords = space.dof_coords();
            let mut inv: EntityDofs = HashMap::new();
            let put = |key: EntityKey, pos: [f64; 3], dof: u32, inv: &mut EntityDofs| {
                inv.entry(key)
                    .or_default()
                    .insert(coord_key(&pos), (dof, g[dof as usize]));
            };
            for f in faces {
                for i in 0..4 {
                    let ek = EdgeKey::new(f[i], f[(i + 1) % 4]);
                    if let Some(ds) = space.edge_dofs(ek) {
                        let mut ends = [
                            coord_key(&mesh.node_coords(f[i])),
                            coord_key(&mesh.node_coords(f[(i + 1) % 4])),
                        ];
                        ends.sort_unstable();
                        let key = EntityKey::Edge(ends[0], ends[1]);
                        for d in ds {
                            let p = coords[d as usize];
                            put(key, p, d, &mut inv);
                        }
                    }
                }
                let qk = QuadFaceKey::new(f[0], f[1], f[2], f[3]);
                if let Some(ds) = space.quad_face_dofs(qk) {
                    let mut cs: Vec<[i64; 3]> =
                        f.iter().map(|n| coord_key(&mesh.node_coords(*n))).collect();
                    cs.sort_unstable();
                    let key = EntityKey::Quad(cs[0], cs[1], cs[2], cs[3]);
                    for d in ds {
                        let p = coords[d as usize];
                        put(key, p, d, &mut inv);
                    }
                }
            }
            inv
        }

        let cyl_g = factors(cyl_space);
        let blk_g = factors(blk_space);
        let blk_inv = inventory(blk_space, blk_faces, &blk_g);
        let cyl_inv = inventory(cyl_space, cyl_faces, &cyl_g);

        // Self-verification field: interpolation of `square_xy` in both spaces
        // yields the exact trace coefficients (linear field), so a correct
        // pairing must satisfy `proj_c[c] == sign · proj_b[b]` per pair.
        let proj_c = cyl_space.interpolate_vector(&square_xy).as_slice().to_vec();
        let proj_b = blk_space.interpolate_vector(&square_xy).as_slice().to_vec();

        let mut pairs: Vec<(u32, u32, f64)> = Vec::new();
        for (key, cyl_map) in &cyl_inv {
            let Some(blk_map) = blk_inv.get(key) else {
                panic!("BlockToCylinderMap: block side missing interface entity {key:?}");
            };
            if cyl_map.len() != blk_map.len() {
                panic!(
                    "BlockToCylinderMap: entity {key:?} dof count mismatch: cyl {} vs blk {}",
                    cyl_map.len(),
                    blk_map.len()
                );
            }
            for (&pos, &(c, gc)) in cyl_map {
                let Some(&(b, gb)) = blk_map.get(&pos) else {
                    panic!(
                        "BlockToCylinderMap: no block dof on entity {key:?} at cyl dof {c} pos {pos:?}"
                    );
                };
                let dot = gc[0] * gb[0] + gc[1] * gb[1] + gc[2] * gb[2];
                let nc = (gc[0] * gc[0] + gc[1] * gc[1] + gc[2] * gc[2]).sqrt();
                let nb = (gb[0] * gb[0] + gb[1] * gb[1] + gb[2] * gb[2]).sqrt();
                assert!(
                    dot.abs() > 0.999_999 * nc * nb && nc > 0.0 && nb > 0.0,
                    "BlockToCylinderMap: geometric factors not parallel at cyl dof {c} / blk dof {b}: g_c={gc:?} g_b={gb:?}"
                );
                let sign = if dot > 0.0 { 1.0 } else { -1.0 };
                let want = sign * proj_b[b as usize];
                let got = proj_c[c as usize];
                assert!(
                    (got - want).abs() <= 1e-9 * (1.0 + got.abs() + want.abs()),
                    "BlockToCylinderMap: transfer verification failed at cyl dof {c} / blk dof {b}: {got} vs {want}"
                );
                pairs.push((c, b, sign));
            }
        }
        pairs.sort_by_key(|&(c, b, _)| (c, b));
        BlockToCylinderMap { pairs }
    }

    /// C++ `map.Transfer(block_gf, cyl_gf)` followed by
    /// `cyl_gf.GetTrueDofs` (multidomain_nd.cpp:394-401): copies the block
    /// values onto the cylinder interface dofs (orientation-corrected),
    /// leaving the cylinder-interior dofs untouched.
    fn transfer(&self, src: &[f64], dst: &mut [f64]) {
        for &(c, b, s) in &self.pairs {
            dst[c as usize] = s * src[b as usize];
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
    let inflow_tags = vec![8_i32];
    let inner_cylinder_wall_tags = vec![9_i32];
    let cyl_inflow_faces = boundary_hex_faces(mesh_cyl, &inflow_tags);
    let cyl_interface_faces = boundary_hex_faces(mesh_cyl, &inner_cylinder_wall_tags);
    let mut ess_tdofs = hcurl_boundary_dofs(&fes_cylinder, &cyl_inflow_faces);
    ess_tdofs.extend(hcurl_boundary_dofs(&fes_cylinder, &cyl_interface_faces));
    ess_tdofs.sort_unstable();
    ess_tdofs.dedup();
    println!("Cylinder ess vdofs: {}", ess_tdofs.len());
    let mut cd_tdo =
        ConvectionDiffusionTDO::new(&fes_cylinder, ess_tdofs, 1.0, 1.0e-1, order, qp);

    let mut magnetic_field_cylinder = vec![0.0_f64; fes_cylinder.n_dofs()];
    // D708/D716: shadow of C++ `magnetic_field_cylinder_gf` — same data flow
    // as multidomain_rt (see the D708 note there): MFEM's Transfer reads and
    // overwrites the destination, and the upstream loop never feeds the RK3
    // result back, so each step's transfer discards the cylinder evolution.
    let mut gf_state = vec![0.0_f64; fes_cylinder.n_dofs()];

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
    let block_wall_tags = vec![1_i32, 2, 3, 4];
    let block_wall_faces = boundary_hex_faces(mesh_blk, &block_wall_tags);
    let block_ess_tdofs = hcurl_boundary_dofs(&fes_block, &block_wall_faces);
    println!("Block ess vdofs (walls 1-4): {}", block_ess_tdofs.len());

    let mut d_tdo =
        ConvectionDiffusionTDO::new(&fes_block, block_ess_tdofs.clone(), 0.0, 1.0, order, qp);

    // Initial condition: H = square_xy on the outer walls, tangential
    // projection (C++ `ProjectBdrCoefficientTangent(one, block_wall_attributes)`,
    // multidomain_nd.cpp:323-325). The interpolation of `square_xy` equals the
    // L2 projection because the tangential trace of this linear field lies in
    // the trace space; the interior stays zero as in C++.
    let mut magnetic_field_block = vec![0.0_f64; fes_block.n_dofs()];
    {
        let proj = fes_block.interpolate_vector(&square_xy).as_slice().to_vec();
        for &d in &block_ess_tdofs {
            magnetic_field_block[d as usize] = proj[d as usize];
        }
    }
    let nonzero = magnetic_field_block.iter().filter(|&&v| v != 0.0).count();
    let ic_sum: f64 = magnetic_field_block.iter().sum();
    println!("Block initial BC dofs (nonzero): {nonzero}  IC sum: {ic_sum:.6e}");

    // Block → cylinder transfer map (C++ multidomain_nd.cpp:383-386).
    let field_block_to_cylinder_map = BlockToCylinderMap::new(
        &fes_cylinder,
        &cyl_interface_faces,
        &fes_block,
        &boundary_hex_faces(mesh_blk, &inner_cylinder_wall_tags),
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

        // Transfer the block solution onto the cylinder interface — into the
        // gf shadow (D708/D716); the RK3 state is copied FROM the shadow.
        let mut transferred = gf_state.clone();
        field_block_to_cylinder_map.transfer(&magnetic_field_block, &mut transferred);
        gf_state = transferred;
        magnetic_field_cylinder.copy_from_slice(&gf_state);

        // Advance the convection-diffusion equation inside the cylinder.
        rk3ssp_step(&mut cd_tdo, &mut magnetic_field_cylinder, &mut t, dt);

        if last_step || ti % vis_steps == 0 {
            let bsum: f64 = magnetic_field_block.iter().sum();
            let csum: f64 = magnetic_field_cylinder.iter().sum();
            let bsq: f64 = magnetic_field_block.iter().map(|v| v * v).sum();
            let csq: f64 = magnetic_field_cylinder.iter().map(|v| v * v).sum();
            println!(
                "step {ti}, t = {t}  block: sum={bsum:.6e} ssq={bsq:.6e}  cyl: sum={csum:.6e} ssq={csq:.6e}"
            );
        }

        ti += 1;
    }
}
