//! # Parallel Example 34 — Magnetostatics with SubMesh current density
//! (1:1 with MFEM ex34p / ex34p.cpp)
//!
//! Solves `curl curl A = J` where the current density `J = -σ∇φ` is computed
//! on a (parallel) SubMesh representing the conducting region.  Nédélec
//! H(curl) elements for the vector potential, Lagrange H¹ for the scalar
//! potential, Raviart-Thomas H(div) for the current density.
//!
//! Flow (mirrors ex34p):
//! 1. Build the fully-refined serial mesh (mixed `fichera-mixed.mesh` by
//!    default, or `-hex` → `fichera.mesh`).  All refinement is done serially
//!    before partitioning (par_uniform_refine is 2-D only; the final global
//!    mesh is identical to C++'s rs+rp refinement).
//! 2. Partition the full mesh; each rank extracts its *local* SubMesh (the
//!    local elements with the submesh attribute — owned + ghost, exactly
//!    like MFEM `ParSubMesh::CreateFromDomain` whose `AddElementsToMesh`
//!    walks the local elements).  The SubMesh partition reuses the parent
//!    global element/node ids, so the RT0 face keys and H1 node ids are
//!    cross-rank consistent.
//! 3. Solve for the electric potential φ: `-∇·(σ∇φ) = 0` on the SubMesh with
//!    Dirichlet φ=0 (attr 2) / φ=1 (attr 23), CG + AMG.
//! 4. Solve for the current density J in H(div) on the SubMesh:
//!    `(J, v) = -(σ∇φ, v)` with J·n = 0 on attrs {25, 9..16}, CG + diag-scale.
//! 5. Assemble the full-mesh H(curl) RHS `b_i = ∫ J·W_i` directly from the
//!    SubMesh RT0 dofs over the local submesh elements (mathematically equal
//!    to C++'s `Transfer` + `VectorFEDomainLFIntegrator`, see the docs), with
//!    a reverse ghost exchange so owned ND rows get the complete load.
//! 6. Solve `(curl curl + δI) A = b` with PEC BCs (all boundaries except the
//!    symmetry planes 9..16), PCG + AMS (block-diagonal parallel AMS).
//! 7. Recover `B = curl A` with the parallel discrete curl operator.
//!
//! ## Known difference vs C++
//! The recovered `B = curl A` on *tetrahedral* faces differs from C++ by a
//! factor of 2: MFEM's `CurlInterpolator` evaluates RT0 dof-transformed normals
//! directly at face centers (tet normal `nk = {1,1,1}` = 2× the face-area
//! normal, so the triangular-face flux is 2× the true value), while fem-rs
//! `DiscreteLinearOperator::curl_3d` uses the Stokes-circulation (mathematically
//! correct, and required by pex24's de Rham commuting checks).  Hex meshes have
//! no triangular faces, so `-hex` B matches C++ exactly.  This is a deliberate
//! core-library choice (do NOT "fix" curl_3d — it would break pex24).

//! ## Usage
//! ```bash
//! cargo run --release --example mfem_pex34_magnetostatics -- --ranks 1 -no-vis
//! cargo run --release --example mfem_pex34_magnetostatics -- --ranks 4 -no-vis
//! cargo run --release --example mfem_pex34_magnetostatics -- --ranks 2 -hex -no-vis
//! ```

use std::collections::HashMap;
use std::io::Write;

use fem_core::NodeId;
use fem_assembly::{
    GridFunction,
    geo_ref_elem_from_mesh,
    isoparametric_jacobian,
    mixed::{MixedVectorGradientIntegrator, assemble_h1_hdiv_mixed, ref_elem_vec},
    standard::{CurlCurlIntegrator, DiffusionIntegrator, VectorMassIntegrator},
};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::{
    ElementTransformation, ElementType, Mesh,
    extract_submesh_3d, refine_uniform_3d,
    topology::MeshTopology,
};
use fem_parallel::{
    DofPartition, ParAmsPrecond, ParAssembler, ParCsrMatrix, ParDiscreteLinearOperator,
    ParVector, ParVectorAssembler, ParallelFESpace, ParallelMesh, SmootherType,
    WorkerConfig, par_assembler::permute_vec, par_partition::partition_mesh,
    par_solve_pcg_amg, par_solve_pcg_jacobi, par_solve_pcg_precond,
};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_amg::ParAmgConfig;
use fem_solver::SolverConfig;
use fem_space::{
    H1Space, HCurlSpace, HDivSpace, SpaceType,
    constraints::{boundary_dofs, boundary_dofs_hcurl},
    fe_space::FESpace,
};
use linlvo::precond::{AmsConfig, AmsCycle, AmsEdgeSmoother};

// ─── CLI ───────────────────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    order: u8,
    ser_ref_levels: i32,
    par_ref_levels: i32,
    delta: f64,
    static_cond: bool,
    mixed: bool,
    visualization: bool,
    ranks: usize,
}

fn default_args() -> Args {
    Args {
        mesh_file: "data/fichera-mixed.mesh".into(),
        order: 1,
        ser_ref_levels: 1,
        par_ref_levels: 1,
        delta: 1e-6,
        static_cond: false,
        mixed: true,
        visualization: false,
        ranks: 1,
    }
}

fn parse_args() -> Args {
    let mut a = default_args();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = it.next().unwrap_or_default(),
            "-rs" | "--refine-serial" => a.ser_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-rp" | "--refine-parallel" => a.par_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-mc" | "--magnetic-cond" => a.delta = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-6),
            "-sc" | "--static-condensation" => a.static_cond = true,
            "-no-sc" | "--no-static-condensation" => a.static_cond = false,
            "-mixed" | "--mixed-mesh" => a.mixed = true,
            "-hex" | "--hex-mesh" => a.mixed = false,
            "-pa" | "--partial-assembly" => {}
            "-no-pa" | "--no-partial-assembly" => {}
            "-d" | "--device" => { it.next(); }
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "--ranks" => a.ranks = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            _ => {}
        }
    }
    a
}

// ─── Hard-coded element selections (matching MFEM ex34p.cpp) ────────────────

const SUBELEMS_MIXED: &[u32] = &[0, 2, 3, 4, 9];
const SUBELEMS_HEX: &[u32] = &[10, 14, 34, 36, 37, 38, 39];
const SYM_PLANE_ATTRS: &[i32] = &[9, 10, 11, 12, 13, 14, 15, 16];
const PHI0_ATTR: i32 = 2;
const PHI1_ATTR: i32 = 23;

/// Build the fully-refined serial mesh with the submesh attribute set,
/// replicating ex34p's refinement arithmetic (the Mesh ctor `refine=1` only
/// *marks* triangles, it does not refine).  Returns the mesh and the submesh
/// attribute used to mark the conducting region.
fn build_serial_mesh(args: &Args) -> (Mesh<3>, i32) {
    let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
    let mut mesh: Mesh<3> = mfem.mesh3d.expect("MFEM mesh must be 3D");

    let mut ser_ref = args.ser_ref_levels;
    let mut par_ref = args.par_ref_levels;

    if !args.mixed {
        // ex34p: `mesh_file = fichera.mesh; mesh->UniformRefinement(); ...`
        let mfem2 = read_mfem_file("data/fichera.mesh").expect("failed to read fichera.mesh");
        mesh = mfem2.mesh3d.expect("fichera.mesh must be 3D");
        mesh = refine_uniform_3d(&mesh);
        if ser_ref > 0 { ser_ref -= 1; } else { par_ref -= 1; }
    }

    // SubMesh attribute on the (possibly once-refined) mesh.
    let max_attr = *mesh.elem_tags.iter().max().unwrap_or(&0);
    let submesh_attr = max_attr + 1;
    let submesh_elems = if args.mixed { SUBELEMS_MIXED } else { SUBELEMS_HEX };
    for &ei in submesh_elems {
        mesh.elem_tags[ei as usize] = submesh_attr;
    }

    // Serial refinements, then the "parallel" refinements done serially too.
    for _ in 0..ser_ref {
        mesh = refine_uniform_3d(&mesh);
    }
    for _ in 0..par_ref {
        mesh = refine_uniform_3d(&mesh);
    }
    (mesh, submesh_attr)
}

/// Face-key → boundary-tag map of the full serial mesh (keys = sorted global
/// vertex ids).  Used to assign the *parent* boundary attribute to the
/// SubMesh boundary faces (the local partition's face table may not carry
/// every boundary face of the local elements).
fn full_face_tag_map(mesh: &Mesh<3>) -> (HashMap<Vec<u32>, i32>, i32) {
    let mut map: HashMap<Vec<u32>, i32> = HashMap::new();
    let mut max_tag = 0i32;
    for f in 0..mesh.n_faces() as u32 {
        let nodes = mesh.face_nodes(f);
        let tag = mesh.face_tag(f);
        max_tag = max_tag.max(tag);
        let mut key: Vec<u32> = nodes.to_vec();
        key.sort_unstable();
        map.insert(key, tag);
    }
    (map, max_tag)
}

/// Partition-order vector → space (DofManager) order, applying the per-DOF
/// sign correction (pex24 helper).
fn to_dm_signed(v_par: &ParVector, dp: &DofPartition) -> Vec<f64> {
    let n_total = dp.n_total_dofs();
    let mut dm = vec![0.0; n_total];
    for p in 0..n_total {
        let d = dp.unpermute_dof(p as u32) as usize;
        let s = dp.sign_correction(d as u32);
        dm[d] = s * v_par.as_slice()[p];
    }
    dm
}

/// Exchange locally-detected (global dof id, value) pairs across ranks and
/// return the union map.  Used for essential-DOF clamping: a boundary DOF may
/// be detected only on the rank holding its face while its owner is a
/// different rank (pex39 lesson).  The merged value is the **max** over the
/// contributors: for the φ BC (attr 2 → 0, attr 23 → 1, projected in that
/// order) a node on the attr-2/attr-23 intersection is detected with value 1
/// by the rank(s) holding its attr-23 face and 0 by the rank(s) holding only
/// its attr-2 face — the globally-consistent value is 1 (max).
fn exchange_global_dof_values(
    comm: &fem_parallel::Comm,
    local: &[(u32, f64)],
) -> HashMap<u32, f64> {
    let rank = comm.rank();
    let mut sends: Vec<(i32, Vec<u8>)> = Vec::new();
    for r in 0..comm.size() as i32 {
        if r == rank { continue; }
        let mut bytes = Vec::with_capacity(local.len() * 12);
        for &(g, v) in local {
            bytes.extend_from_slice(&g.to_le_bytes());
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        sends.push((r, bytes));
    }
    let incoming = comm.alltoallv_bytes(&sends);
    let mut out: HashMap<u32, f64> = HashMap::new();
    for &(g, v) in local {
        out.entry(g).and_modify(|e| *e = e.max(v)).or_insert(v);
    }
    for (_, bytes) in incoming {
        for chunk in bytes.chunks_exact(12) {
            let g = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let v = f64::from_le_bytes(chunk[4..12].try_into().unwrap());
            out.entry(g).and_modify(|e| *e = e.max(v)).or_insert(v);
        }
    }
    out
}

/// Compute the global set of J·n = 0 RT0 faces of the SubMesh from the *full*
/// serial mesh (deterministic on every rank): faces of SubMesh elements that
/// belong to exactly one SubMesh element (the SubMesh boundary), with tag 25
/// (shared with the rest of the domain) or a symmetry-plane attribute 9..16.
/// Face keys = the 3 smallest global vertex ids (RT0 FaceKey convention).
///
/// The per-rank local boundary-face detection is NOT used here: after
/// partitioning, the local element sets differ and a face shared by two
/// SubMesh elements on different ranks can be miscounted as a boundary face
/// locally, producing an inconsistent essential set (pex34 np2 found 229 vs
/// the correct 160).
fn global_jn_zero_face_keys(
    full_mesh: &Mesh<3>,
    submesh_attr: i32,
    full_face_map: &HashMap<Vec<u32>, i32>,
    global_max_tag: i32,
) -> std::collections::HashSet<[u32; 3]> {
    use fem_mesh::ElementType;
    // Local face tables (vertex index lists), same as extract_submesh_3d.
    let faces_of = |et: ElementType| -> Vec<Vec<usize>> {
        match et {
            ElementType::Tet4 | ElementType::Tet10 => vec![
                vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2],
            ],
            ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => vec![
                vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![0, 1, 5, 4],
                vec![2, 3, 7, 6], vec![0, 3, 7, 4], vec![1, 2, 6, 5],
            ],
            ElementType::Prism6 | ElementType::Prism15 => vec![
                vec![0, 1, 2], vec![3, 4, 5], vec![0, 1, 4, 3],
                vec![1, 2, 5, 4], vec![0, 2, 5, 3],
            ],
            _ => vec![],
        }
    };
    // Count SubMesh elements per face (by 3-min global vertex ids).
    let mut counts: std::collections::HashMap<[u32; 3], (usize, Vec<u32>)> =
        std::collections::HashMap::new();
    for e in 0..full_mesh.n_elems() as u32 {
        if full_mesh.elem_tags[e as usize] != submesh_attr { continue; }
        let et = full_mesh.element_type(e);
        let ns = full_mesh.elem_nodes(e);
        for fv in faces_of(et) {
            let mut verts: Vec<u32> = fv.iter().map(|&i| ns[i]).collect();
            verts.sort_unstable();
            let key = [verts[0], verts[1], verts[2]];
            let entry = counts.entry(key).or_insert((0, verts.clone()));
            entry.0 += 1;
        }
    }
    let mut ess = std::collections::HashSet::new();
    for (key, (cnt, verts)) in counts {
        if cnt != 1 { continue; }
        // tag: parent boundary attr if the face is on the domain boundary,
        // else 25 (the SubMesh's internal boundary).
        let tag = full_face_map.get(&verts).copied().unwrap_or(global_max_tag + 1);
        if tag == 25 || (9..=16).contains(&tag) {
            ess.insert(key);
        }
    }
    ess
}

/// Build the per-rank local SubMesh (mesh + partition) from the local SubMesh
/// mesh, **reordering the nodes to `[owned | ghost]`** (the convention
/// `DofPartition::from_mesh_partition` / `ParVector` assume: the P1 DOFs are
/// the mesh nodes, so local ids `< n_owned_nodes` must be exactly the owned
/// nodes).  `extract_submesh_3d` numbers nodes by first occurrence, which
/// interleaves owned and ghost nodes and misclassifies early ghost nodes as
/// owned.
///
/// `sub` was extracted from the *local* mesh, so its `parent_elem_ids` /
/// `parent_node_of_sub` refer to local parent ids.  The SubMesh partition
/// reuses the parent's *global* element/node ids, so face keys and node ids
/// are cross-rank consistent.  A SubMesh node is owned by the minimum-rank
/// SubMesh element containing it (the parent node owner may own no SubMesh
/// element and must not own the node here).
///
/// Returns the reordered SubMesh, its partition, and the reordered
/// `parent_node_of_sub`.
fn build_submesh_partition(
    sub_mesh: &Mesh<3>,
    parent_elem_ids: &[u32],
    parent_node_of_sub: &[NodeId],
    parent_part: &fem_parallel::MeshPartition,
    comm: &fem_parallel::Comm,
) -> (Mesh<3>, fem_parallel::MeshPartition, Vec<NodeId>) {
    use fem_parallel::MeshPartition;
    let rank = comm.rank();
    let n_sub_nodes = sub_mesh.n_nodes();
    let n_sub_elems = sub_mesh.n_elems();

    // SubMesh element ownership (parent element owner).
    let mut elem_owner = vec![0i32; n_sub_elems];
    let mut global_elem_ids = vec![0u32; n_sub_elems];
    let mut n_owned_elems = 0usize;
    for (i, &pe) in parent_elem_ids.iter().enumerate() {
        let owner = if (pe as usize) < parent_part.n_owned_elems {
            rank
        } else {
            parent_part.elem_owner[pe as usize]
        };
        elem_owner[i] = owner;
        global_elem_ids[i] = parent_part.global_elem(pe);
        if owner == rank { n_owned_elems += 1; }
    }

    // Node ownership: min rank over local SubMesh elements containing the node.
    let mut node_owner = vec![i32::MAX; n_sub_nodes];
    for sn in 0..n_sub_nodes {
        let mut min_owner = i32::MAX;
        for i in 0..n_sub_elems {
            let contains = sub_mesh.element_nodes(i as u32).iter().any(|&s| s as usize == sn);
            if contains {
                min_owner = min_owner.min(elem_owner[i]);
            }
        }
        node_owner[sn] = min_owner;
    }

    // Reorder nodes: owned first (in current order), then ghosts.
    let mut old_to_new = vec![0u32; n_sub_nodes];
    let mut new_order: Vec<usize> = Vec::with_capacity(n_sub_nodes);
    for sn in 0..n_sub_nodes {
        if node_owner[sn] == rank {
            old_to_new[sn] = new_order.len() as u32;
            new_order.push(sn);
        }
    }
    let n_owned_nodes = new_order.len();
    for sn in 0..n_sub_nodes {
        if node_owner[sn] != rank {
            old_to_new[sn] = new_order.len() as u32;
            new_order.push(sn);
        }
    }

    // Rebuild the mesh with the new node order.
    let mut new_coords = vec![0.0f64; n_sub_nodes * 3];
    let mut new_parent_node = vec![0u32; n_sub_nodes];
    let mut new_global_ids = vec![0u32; n_sub_nodes];
    let mut new_owners = vec![0i32; n_sub_nodes];
    for (k, &sn) in new_order.iter().enumerate() {
        let base = sn * 3;
        new_coords[k * 3..k * 3 + 3].copy_from_slice(&sub_mesh.coords[base..base + 3]);
        new_parent_node[k] = parent_node_of_sub[sn];
        new_global_ids[k] = parent_part.global_node(parent_node_of_sub[sn]);
        new_owners[k] = node_owner[sn];
    }
    let remap = |nodes: &[NodeId]| -> Vec<NodeId> {
        nodes.iter().map(|&n| old_to_new[n as usize]).collect()
    };
    let new_conn: Vec<NodeId> = if let Some(offs) = &sub_mesh.elem_offsets {
        let mut c = Vec::with_capacity(sub_mesh.conn.len());
        for e in 0..n_sub_elems {
            c.extend_from_slice(&remap(&sub_mesh.conn[offs[e]..offs[e + 1]]));
        }
        c
    } else {
        remap(&sub_mesh.conn)
    };
    let new_face_conn: Vec<NodeId> = if let Some(offs) = &sub_mesh.face_offsets {
        let mut c = Vec::with_capacity(sub_mesh.face_conn.len());
        for f in 0..sub_mesh.n_faces() {
            c.extend_from_slice(&remap(&sub_mesh.face_conn[offs[f]..offs[f + 1]]));
        }
        c
    } else {
        remap(&sub_mesh.face_conn)
    };
    let mut reordered = sub_mesh.clone();
    reordered.coords = new_coords;
    reordered.conn = new_conn;
    reordered.face_conn = new_face_conn;

    let mut part = MeshPartition::from_partitioner(
        &new_global_ids,
        &[],
        &global_elem_ids,
        &[],
        rank,
    );
    // Override ownership (from_partitioner assumed everything owned).
    part.n_owned_nodes = n_owned_nodes;
    part.n_ghost_nodes = n_sub_nodes - n_owned_nodes;
    part.node_owner = new_owners;
    part.n_owned_elems = n_owned_elems;
    part.n_ghost_elems = n_sub_elems - n_owned_elems;
    part.elem_owner = elem_owner;
    part.build_lookup();

    (reordered, part, new_parent_node)
}

/// Assemble the full-mesh H(curl) RHS `b_i = ∫_Ω J·W_i dx` directly from the
/// SubMesh RT0 dofs.  Iterates the local full-mesh elements that belong to
/// the SubMesh (identified by tag), evaluates J from the SubMesh RT dofs and
/// tests against the full-mesh ND basis (pex34: the direct form is
/// mathematically identical to C++'s `Transfer` + `VectorFEDomainLFIntegrator`,
/// and at np1 reduces exactly to the serial ex34 assembly).
fn assemble_nd_rhs_from_submesh(
    nd_space: &HCurlSpace<Mesh<3>>,
    sub_rt_space: &HDivSpace<Mesh<3>>,
    sub_parent_elem_of: &HashMap<u32, u32>,
    submesh_attr: i32,
    j_sub_dm: &[f64],
    quad_order: u8,
) -> Vec<f64> {
    let mesh = nd_space.mesh();
    let dim = 3;
    let n_dofs = nd_space.n_dofs();
    let mut rhs = vec![0.0_f64; n_dofs];

    for e in mesh.elem_iter() {
        if mesh.elem_tags[e as usize] != submesh_attr { continue; }
        let e_sub = sub_parent_elem_of[&e];
        let elem_type = mesh.element_type(e);
        let nd_ref = ref_elem_vec(elem_type, nd_space.order(), SpaceType::HCurl).unwrap();
        let rt_ref = ref_elem_vec(elem_type, sub_rt_space.order(), SpaceType::HDiv).unwrap();
        let quad = nd_ref.quadrature(quad_order);

        let nd_dofs: Vec<u32> = nd_space.element_dofs(e).iter().copied().collect();
        let rt_dofs: Vec<u32> = sub_rt_space.element_dofs(e_sub).iter().copied().collect();
        let nd_s = nd_space.element_signs(e);
        let rt_s = sub_rt_space.element_signs(e_sub);
        let n_nd = nd_dofs.len();
        let n_rt = rt_dofs.len();

        let nodes = mesh.element_nodes(e);
        let use_iso = !matches!(elem_type, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2);
        let geo_elem = if use_iso { geo_ref_elem_from_mesh(mesh, e) } else { None };

        let mut nd_phi = vec![0.0; n_nd * dim];
        let mut rt_phi = vec![0.0; n_rt * dim];
        let mut nd_phys = vec![0.0; n_nd * dim];
        let mut rt_phys = vec![0.0; n_rt * dim];
        let mut f_elem = vec![0.0; n_nd];

        for (q, xi) in quad.points.iter().enumerate() {
            let (jit, jac, det_j) = if use_iso {
                let ge = geo_elem.as_ref().expect("geo_ref_elem");
                let (jac, dj, _x) = isoparametric_jacobian(mesh, nodes, ge.as_ref(), xi, dim);
                let jit = jac.clone().try_inverse().expect("invertible Jacobian").transpose();
                (jit, jac, dj)
            } else {
                let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
                (tr.jacobian_inv_t().clone(), tr.jacobian().clone(), tr.det_j())
            };
            let w = quad.weights[q] * det_j.abs();

            nd_ref.eval_basis_vec(xi, &mut nd_phi);
            rt_ref.eval_basis_vec(xi, &mut rt_phi);

            // ND:  w_i = sign_i · J^{-T} ŵ_i ;  RT: φ_j = sign_j · (1/det J) J φ̂_j
            for i in 0..n_nd {
                let s = nd_s.get(i).copied().unwrap_or(1.0);
                for r in 0..dim {
                    let mut acc = 0.0;
                    for c in 0..dim { acc += jit[(r, c)] * nd_phi[i * dim + c]; }
                    nd_phys[i * dim + r] = s * acc;
                }
            }
            for j in 0..n_rt {
                let s = rt_s.get(j).copied().unwrap_or(1.0);
                for r in 0..dim {
                    let mut acc = 0.0;
                    for c in 0..dim { acc += jac[(r, c)] * rt_phi[j * dim + c]; }
                    rt_phys[j * dim + r] = s * acc / det_j;
                }
            }

            let mut j_at_q = [0.0_f64; 3];
            for j in 0..n_rt {
                let jv = j_sub_dm[rt_dofs[j] as usize];
                for c in 0..3 { j_at_q[c] += jv * rt_phys[j * 3 + c]; }
            }
            for i in 0..n_nd {
                let dot = (0..3).map(|c| nd_phys[i * 3 + c] * j_at_q[c]).sum::<f64>();
                f_elem[i] += w * dot;
            }
        }
        for (li, &gi) in nd_dofs.iter().enumerate() {
            rhs[gi as usize] += f_elem[li];
        }
    }
    rhs
}

/// Solve the essential-BC-clamped system.  Returns the clamped owned dofs and
/// the full union map of (global dof id → value) for all locally-known
/// essential DOFs (owned and ghost).
fn clamp_owned_ess(
    dp: &DofPartition,
    comm: &fem_parallel::Comm,
    local_dm_dofs: &[u32],
    values: &[f64],
) -> (Vec<(usize, f64)>, HashMap<u32, f64>) {
    let local_pairs: Vec<(u32, f64)> = local_dm_dofs
        .iter()
        .zip(values.iter())
        .map(|(&d, &v)| (dp.global_dof(dp.permute_dof(d)), v))
        .collect();
    let union = exchange_global_dof_values(comm, &local_pairs);
    let n_owned = dp.n_owned_dofs;
    let clamped = (0..n_owned)
        .filter_map(|pid| {
            let g = dp.global_dof(pid as u32);
            union.get(&g).map(|&v| (pid, v))
        })
        .collect();
    (clamped, union)
}

/// Finish the MFEM DIAG_KEEP elimination for essential DOFs whose ghost-column
/// couplings cross ranks (non-homogeneous values need the `-A[j, d]·x_bc`
/// RHS contributions from other ranks' ghost columns; homogeneous ones only
/// need the columns zeroed for symmetry).  Delegates to
/// [`ParCsrMatrix::apply_ghost_ess_columns`].
fn apply_offd_ess_elimination(
    a: &mut ParCsrMatrix,
    rhs: &mut ParVector,
    dp: &DofPartition,
    _comm: &fem_parallel::Comm,
    ess_union: &HashMap<u32, f64>,
) {
    let n_owned = dp.n_owned_dofs;
    let n_ghost = dp.n_ghost_dofs;
    if n_ghost == 0 {
        return;
    }
    let ghost_ess: Vec<(usize, f64)> = (0..n_ghost)
        .filter_map(|g| {
            let gid = dp.global_dof((n_owned + g) as u32);
            ess_union.get(&gid).map(|&v| (g, v))
        })
        .collect();
    if !ghost_ess.is_empty() {
        a.apply_ghost_ess_columns(&ghost_ess, rhs);
    }
}

fn main() {
    let args = parse_args();
    let (full_mesh, submesh_attr) = build_serial_mesh(&args);
    let order = args.order;
    let rt_order = if order > 0 { order - 1 } else { 0 };
    let qo = (2 * order + 1) as u8;
    let delta = args.delta;

    let (full_face_map, global_max_tag) = full_face_tag_map(&full_mesh);

    let full_mesh_arc = std::sync::Arc::new(full_mesh);

    ThreadLauncher::new(WorkerConfig::new(args.ranks)).launch(move |comm| {
        let rank = comm.rank();
        let is_root = rank == 0;
        let mut out = String::new();

        if is_root {
            out.push_str("Options used:\n");
            out.push_str(&format!("   --refine-serial {}\n", args.ser_ref_levels));
            out.push_str(&format!("   --refine-parallel {}\n", args.par_ref_levels));
            out.push_str(&format!("   --order {}\n", args.order));
            out.push_str(&format!("   --magnetic-cond {}\n", args.delta));
            out.push_str(&format!("   --{}static-condensation\n", if args.static_cond { "" } else { "no-" }));
            out.push_str(&format!("   --{}mixed-mesh\n", if args.mixed { "" } else { "no-" }));
            out.push_str("   --no-partial-assembly\n   --device cpu\n   --no-visualization\n");
        }

        // ── 1. Partition the full mesh (all refinement already serial) ────────
        let pmesh = partition_mesh(&*full_mesh_arc, &comm);
        let local_mesh = pmesh.local_mesh().clone();
        let part = pmesh.partition().clone();

        // ── 2. Full-mesh parallel spaces: ND (H(curl)) and RT (H(div)) ────────
        let nd_par = ParallelFESpace::new_for_edge_space(
            HCurlSpace::new(local_mesh.clone(), order), &pmesh, comm.clone());
        let rt_full_part = DofPartition::from_face_space(
            &HDivSpace::new(local_mesh.clone(), rt_order), &part, &comm);
        let rt_full_par = ParallelFESpace::new_with_dof_partition(
            HDivSpace::new(local_mesh.clone(), rt_order), rt_full_part, comm.clone());
        let nd_dp = nd_par.dof_partition();
        let rt_full_dp = rt_full_par.dof_partition();

        // Full H1 space (P1) — needed by the AMS gradient.
        let h1_full_space = H1Space::new(local_mesh.clone(), 1);
        let h1_full_par = ParallelFESpace::new(h1_full_space, &pmesh, comm.clone());

        // ── 3. Local SubMesh: extract + partition (mirrors ParSubMesh) ────────
        let sub = extract_submesh_3d(&local_mesh, &[submesh_attr]);
        let mut sub_mesh = sub.mesh.clone();
        // Fix the SubMesh boundary face tags from the full serial mesh map
        // (the local partition's face table may lack boundary faces assigned
        // to other ranks; the global max tag fixes the ghost-boundary attr).
        {
            let face_conn = sub_mesh.face_conn.clone();
            let foffs = sub_mesh.face_offsets.clone();
            let npe = sub_mesh.face_type.nodes_per_element();
            let mut new_tags = Vec::with_capacity(sub_mesh.face_tags.len());
            for f in 0..sub_mesh.n_faces() {
                let (s, e) = match &foffs {
                    Some(o) => (o[f], o[f + 1]),
                    None => (f * npe, (f + 1) * npe),
                };
                let mut key: Vec<u32> = face_conn[s..e]
                    .iter()
                    .map(|&sn| part.global_node(sub.parent_node_of_sub[sn as usize]))
                    .collect();
                key.sort_unstable();
                let tag = full_face_map.get(&key).copied().unwrap_or(global_max_tag + 1);
                new_tags.push(tag);
            }
            sub_mesh.face_tags = new_tags;
        }
        // Partition + reorder the SubMesh nodes to [owned | ghost] (the
        // DofPartition/ParVector layout convention).
        let (sub_mesh, sub_partition, sub_parent_node_of_sub) = build_submesh_partition(
            &sub_mesh,
            &sub.parent_elem_ids,
            &sub.parent_node_of_sub,
            &part,
            &comm,
        );
        let pmesh_cond = ParallelMesh::new(sub_mesh.clone(), comm.clone(), sub_partition);

        // SubMesh parallel spaces: H1 (P1) and RT (RT0).
        let sub_h1_space = H1Space::new(sub_mesh.clone(), order);
        let sub_rt_space = HDivSpace::new(sub_mesh.clone(), rt_order);
        let sub_h1_par = ParallelFESpace::new(sub_h1_space, &pmesh_cond, comm.clone());
        let sub_rt_part = DofPartition::from_face_space(&sub_rt_space, pmesh_cond.partition(), &comm);
        let sub_rt_par = ParallelFESpace::new_with_dof_partition(sub_rt_space, sub_rt_part, comm.clone());
        let sub_h1_dp = sub_h1_par.dof_partition();
        let sub_rt_dp = sub_rt_par.dof_partition();
        // Fresh serial-space instances for the local (ghost-row-complete)
        // mixed assembly and the RHS element integrals (spaces are not Clone).
        let sub_h1_space_local = H1Space::new(sub_mesh.clone(), order);
        let sub_rt_space_local = HDivSpace::new(sub_mesh.clone(), rt_order);

        if is_root {
            out.push_str(&format!("  SubMesh: {} elements, {} nodes, {} boundary faces\n",
                sub_mesh.n_elems(), sub_mesh.n_nodes(), sub_mesh.n_faces()));
            out.push_str(&format!("  SubMesh H1 DOFs: {}, RT DOFs: {}\n",
                sub_h1_par.n_global_dofs(), sub_rt_par.n_global_dofs()));
        }

        // ── 4. Solve for φ: -∇·(σ∇φ) = 0 on the SubMesh ───────────────────────
        let mut a_h1 = ParAssembler::assemble_bilinear(
            &sub_h1_par, &[&DiffusionIntegrator { kappa: 1.0 }], qo,
        );

        // ProjectBdrCoefficient: φ = 0 on attr 2, then φ = 1 on attr 23
        // (second overwrites the intersection, as in ex34p).
        let n_h1 = sub_h1_space_local.n_dofs();
        let dm_h1 = sub_h1_space_local.dof_manager();
        let mut phi_dm = vec![0.0_f64; n_h1];
        {
            let mut gf = GridFunction::new(&sub_h1_space_local, phi_dm.clone());
            gf.project_bdr_coefficient(&|_| 0.0, &[PHI0_ATTR], dm_h1);
            gf.project_bdr_coefficient(&|_| 1.0, &[PHI1_ATTR], dm_h1);
            phi_dm = gf.dofs().to_vec();
        }

        let ess_phi_local = boundary_dofs(&sub_mesh, dm_h1, &[PHI0_ATTR, PHI1_ATTR]);
        let ess_phi_vals: Vec<f64> = ess_phi_local.iter().map(|&d| phi_dm[d as usize]).collect();
        let (clamped_phi, ess_phi_union) = clamp_owned_ess(sub_h1_dp, &comm, &ess_phi_local, &ess_phi_vals);

        let mut rhs_h1 = ParVector::zeros(&sub_h1_par);
        let mut phi = ParVector::from_local_raw(
            permute_vec(&phi_dm, sub_h1_dp),
            sub_h1_dp.n_owned_dofs,
            sub_h1_par.dof_ghost_exchange_arc(),
            comm.clone(),
        );
        for &(pid, v) in &clamped_phi {
            a_h1.apply_dirichlet_par_keep_diag(pid, v, &mut rhs_h1);
        }
        apply_offd_ess_elimination(&mut a_h1, &mut rhs_h1, sub_h1_dp, &comm, &ess_phi_union);

        if is_root {
            out.push_str("\nSolving for electric potential using CG with AMG\n");
        }
        let h1_cfg = SolverConfig {
            rtol: 1e-12, atol: 0.0, max_iter: 2000,
            verbose: false, ..Default::default()
        };
        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            smoothed_prolongation: true,
            ..Default::default()
        };
        let res_h1 = par_solve_pcg_amg(&a_h1, &rhs_h1, &mut phi, &amg_cfg, &h1_cfg)
            .expect("phi PCG+AMG failed");
        if is_root {
            out.push_str(&format!("  phi PCG: {} iters, residual = {:.3e}, converged = {}\n",
                res_h1.iterations, res_h1.final_residual, res_h1.converged));
        }

        // ── 5. Solve for J = -σ∇φ in H(div) on the SubMesh ────────────────────
        let mut m_rt = ParVectorAssembler::assemble_bilinear(
            &sub_rt_par, &[&VectorMassIntegrator { alpha: 1.0 }], qo,
        );

        // G = assemble_h1_hdiv_mixed: rows = H¹, cols = H(div) — assembled on
        // the local (serial) SubMesh spaces so ghost-row couplings are kept.
        let grad_mat = assemble_h1_hdiv_mixed(
            &sub_h1_space_local, &sub_rt_space_local,
            &[&MixedVectorGradientIntegrator { sigma: 1.0 }],
            qo,
        );
        phi.update_ghosts();
        let phi_dm_solved = to_dm_signed(&phi, sub_h1_dp);
        let n_rt = sub_rt_space_local.n_dofs();
        let mut b_rt_dm = vec![0.0_f64; n_rt];
        for h1_row in 0..grad_mat.nrows {
            for k in grad_mat.row_ptr[h1_row]..grad_mat.row_ptr[h1_row + 1] {
                let hdiv_col = grad_mat.col_idx[k] as usize;
                b_rt_dm[hdiv_col] -= grad_mat.values[k] * phi_dm_solved[h1_row];
            }
        }
        let mut b_rt = ParVector::from_local_raw(
            permute_vec(&b_rt_dm, sub_rt_dp),
            sub_rt_dp.n_owned_dofs,
            sub_rt_par.dof_ghost_exchange_arc(),
            comm.clone(),
        );

        // J·n = 0 on walls (symmetry planes + attr 25).  The essential face
        // set is computed from the FULL serial mesh (deterministic on every
        // rank); the local boundary-face detection is unreliable after
        // partitioning (a face shared by two SubMesh elements on different
        // ranks can be miscounted as a local boundary face).
        let jn_zero = global_jn_zero_face_keys(&full_mesh_arc, submesh_attr, &full_face_map, global_max_tag);
        // Local face tables in the **HDivSpace element-dof order** (the serial
        // space's element_dofs[e][fi] is the dof of the fi-th face of this
        // table): tet = TET_FACES, hex = HEX_FACES, prism = PRISM_FACES.
        let sub_faces_of = |et: ElementType| -> Vec<Vec<usize>> {
            match et {
                ElementType::Tet4 | ElementType::Tet10 => vec![
                    vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2],
                ],
                ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => vec![
                    vec![3, 2, 1, 0], vec![0, 1, 5, 4], vec![1, 2, 6, 5],
                    vec![2, 3, 7, 6], vec![3, 0, 4, 7], vec![4, 5, 6, 7],
                ],
                ElementType::Prism6 | ElementType::Prism15 => vec![
                    vec![0, 1, 2], vec![3, 4, 5], vec![0, 1, 4, 3],
                    vec![1, 2, 5, 4], vec![0, 2, 5, 3],
                ],
                _ => vec![],
            }
        };
        // (dm dof, value) for all local SubMesh elements (owned + ghost).
        let mut ess_rt_local: Vec<(u32, f64)> = Vec::new();
        for e in 0..sub_mesh.n_elems() as u32 {
            let et = sub_mesh.element_type(e);
            let ns = sub_mesh.element_nodes(e);
            let dofs = sub_rt_space_local.element_dofs(e);
            for (fi, fv) in sub_faces_of(et).iter().enumerate() {
                let mut verts: Vec<u32> = fv.iter()
                    .map(|&i| part.global_node(sub_parent_node_of_sub[ns[i] as usize]))
                    .collect();
                verts.sort_unstable();
                let key = [verts[0], verts[1], verts[2]];
                if jn_zero.contains(&key) {
                    let dm_dof = dofs[fi];
                    let pid = sub_rt_dp.permute_dof(dm_dof);
                    if (pid as usize) < sub_rt_dp.n_owned_dofs {
                        ess_rt_local.push((dm_dof, 0.0));
                    }
                }
            }
        }
        let ess_rt_dm: Vec<u32> = ess_rt_local.iter().map(|&(d, _)| d).collect();
        let (clamped_rt, ess_rt_union) = clamp_owned_ess(sub_rt_dp, &comm, &ess_rt_dm, &vec![0.0; ess_rt_dm.len()]);
        for &(pid, _) in &clamped_rt {
            m_rt.apply_dirichlet_par_keep_diag(pid, 0.0, &mut b_rt);
        }
        apply_offd_ess_elimination(&mut m_rt, &mut b_rt, sub_rt_dp, &comm, &ess_rt_union);

        if is_root {
            out.push_str("\nSolving for current density in H(Div) using diagonally scaled CG\n");
            out.push_str(&format!("Size of linear system: {}\n", sub_rt_par.n_global_dofs()));
        }
        let rt_cfg = SolverConfig {
            rtol: 1e-12, atol: 0.0, max_iter: 2000,
            verbose: false, ..Default::default()
        };
        let mut j_cond = ParVector::zeros(&sub_rt_par);
        let res_rt = par_solve_pcg_jacobi(&m_rt, &b_rt, &mut j_cond, &rt_cfg)
            .expect("RT CG failed");
        if is_root {
            out.push_str(&format!("  J CG: {} iters, residual = {:.3e}, converged = {}\n",
                res_rt.iterations, res_rt.final_residual, res_rt.converged));
        }

        // ── 6. Full-mesh ND RHS: b_i = ∫ J·W_i (direct from SubMesh RT dofs) ──
        j_cond.update_ghosts();
        let j_sub_dm = to_dm_signed(&j_cond, sub_rt_dp);
        let sub_parent_elem_of: HashMap<u32, u32> = sub.parent_elem_ids
            .iter()
            .enumerate()
            .map(|(i, &pe)| (pe, i as u32))
            .collect();
        let nd_space_rhs = HCurlSpace::new(local_mesh.clone(), order);
        let rhs_nd_dm = assemble_nd_rhs_from_submesh(
            &nd_space_rhs, &sub_rt_space_local, &sub_parent_elem_of,
            submesh_attr, &j_sub_dm, qo,
        );
        // No reverse exchange needed: the local mesh's ghost layer contains
        // every element (owned + ghost) that contributes to an owned ND DOF,
        // so the owned rows of the permuted RHS are already complete (same
        // reasoning as ParAssembler::assemble_linear).
        let mut rhs_nd = ParVector::from_local_raw(
            permute_vec(&rhs_nd_dm, nd_dp),
            nd_dp.n_owned_dofs,
            nd_par.dof_ghost_exchange_arc(),
            comm.clone(),
        );

        // ── 7. ND system: curl curl + δ·I, PEC on all boundaries but the
        //     symmetry planes ──────────────────────────────────────────────────
        let mut a_nd = ParVectorAssembler::assemble_bilinear(
            &nd_par,
            &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: delta }],
            qo,
        );

        let nd_mesh = nd_space_rhs.mesh();
        let all_tags: Vec<i32> = nd_mesh.unique_boundary_tags();
        let pec_tags: Vec<i32> = all_tags.into_iter()
            .filter(|t| !SYM_PLANE_ATTRS.contains(t))
            .collect();
        let ess_nd_local = if pec_tags.is_empty() {
            vec![]
        } else {
            boundary_dofs_hcurl(nd_mesh, &nd_space_rhs, &pec_tags)
        };
        let (clamped_nd, ess_nd_union) = clamp_owned_ess(nd_dp, &comm, &ess_nd_local, &vec![0.0; ess_nd_local.len()]);
        for &(pid, _) in &clamped_nd {
            a_nd.apply_dirichlet_par_keep_diag(pid, 0.0, &mut rhs_nd);
        }
        apply_offd_ess_elimination(&mut a_nd, &mut rhs_nd, nd_dp, &comm, &ess_nd_union);

        if is_root {
            out.push_str("\nSolving for magnetic vector potential using CG with AMS\n");
            out.push_str(&format!("Size of linear system: {}\n", nd_par.n_global_dofs()));
        }

        let mut a_sol = ParVector::zeros(&nd_par);
        let nd_cfg = SolverConfig {
            rtol: 1e-12, atol: 0.0, max_iter: 2000,
            verbose: false, ..Default::default()
        };

        // PCG + block-diagonal parallel AMS (H(curl) auxiliary-space solver).
        // Multiplicative V(1,1) + symmetric GS matches HYPRE AMS's default
        // cycle structure and is the configuration pex8 found PCG-usable.
        // The small singularity regularization anchors the nodal GᵀAG problem
        // on small per-rank diagonal blocks (np4 -hex without it stalled).
        let grad = ParDiscreteLinearOperator::gradient(&h1_full_par, &nd_par);
        let ams = ParAmsPrecond::new(&a_nd, &grad, AmsConfig {
            smoother_omega: 1.0,
            smoother_sweeps: 2,
            edge_smoother: AmsEdgeSmoother::SymmetricGaussSeidel,
            cycle: AmsCycle::MultiplicativeV11,
            singularity_regularization: 1e-10,
            ..Default::default()
        });
        let res_nd = par_solve_pcg_precond(&a_nd, &rhs_nd, &mut a_sol, &|r, z| ams.apply(r, z), &nd_cfg)
            .expect("ND PCG failed");
        if is_root {
            out.push_str(&format!("  A PCG: {} iters, residual = {:.3e}, converged = {}\n",
                res_nd.iterations, res_nd.final_residual, res_nd.converged));
        }

        // ── 8. B = curl A ──────────────────────────────────────────────────────
        let curl = ParDiscreteLinearOperator::curl_3d(&nd_par, &rt_full_par);
        let mut a_full = a_sol.clone_vec();
        a_full.update_ghosts();
        let n_rt_owned = rt_full_dp.n_owned_dofs;
        let mut b_vec = vec![0.0_f64; n_rt_owned];
        curl.spmv(a_full.as_slice(), &mut b_vec);

        // ── 9. Solution metrics (norm/sum/checksum — np1-4 must agree) ────────
        a_sol.update_ghosts();
        let n_owned_nd = nd_dp.n_owned_dofs;
        let solution_norm = a_sol.global_norm();
        let solution_sum = comm.allreduce_sum_f64(a_sol.as_slice()[..n_owned_nd].iter().sum::<f64>());
        // Checksum with a *physical* edge id (the ND1 dof of the edge with
        // the smallest global vertex id) so it is partition-independent and
        // matches C++'s `(i+1)·x_i` at np1 (the full-mesh edge numbering =
        // element×local-edge first-seen, same as MFEM's ND dof numbering).
        let local_nd = HCurlSpace::new(local_mesh.clone(), order);
        let edges_for_elem = |et: ElementType| -> &'static [(usize, usize)] {
            match et {
                ElementType::Tet4 | ElementType::Tet10 => &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                ElementType::Hex8 | ElementType::Hex20 => &[
                    (0, 1), (1, 2), (3, 2), (0, 3), (4, 5), (5, 6), (7, 6), (4, 7),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ],
                ElementType::Prism6 => &[
                    (0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5),
                ],
                _ => &[],
            }
        };
        // Global edge ids from the FULL serial mesh (deterministic on all ranks).
        let mut key_to_edge: HashMap<(u32, u32), u64> = HashMap::new();
        let mut next_edge = 0u64;
        for e in full_mesh_arc.elem_iter() {
            let et = full_mesh_arc.element_type(e);
            let ns = full_mesh_arc.element_nodes(e);
            for &(a, b) in edges_for_elem(et) {
                let (ga, gb) = (ns[a], ns[b]);
                let key = if ga < gb { (ga, gb) } else { (gb, ga) };
                if !key_to_edge.contains_key(&key) {
                    key_to_edge.insert(key, next_edge);
                    next_edge += 1;
                }
            }
        }
        // Local dm dof → edge key (from the local space; the parent global
        // node ids make the key cross-rank consistent), then → global edge id.
        let mut dm_to_key: HashMap<u32, (u32, u32)> = HashMap::new();
        for e in local_nd.mesh().elem_iter() {
            let et = local_nd.mesh().element_type(e);
            let ns = local_nd.mesh().element_nodes(e);
            let dofs = local_nd.element_dofs(e);
            for (k, &(a, b)) in edges_for_elem(et).iter().enumerate() {
                let (ga, gb) = (part.global_node(ns[a]), part.global_node(ns[b]));
                dm_to_key.entry(dofs[k]).or_insert((ga.min(gb), ga.max(gb)));
            }
        }
        let local_checksum: f64 = (0..n_owned_nd)
            .map(|pid| {
                let dm = nd_dp.unpermute_dof(pid as u32);
                let id = dm_to_key
                    .get(&dm)
                    .and_then(|&k| key_to_edge.get(&k))
                    .copied()
                    .unwrap_or(0);
                (id as f64 + 1.0) * a_sol.as_slice()[pid]
            })
            .sum();
        let solution_checksum = comm.allreduce_sum_f64(local_checksum);
        let b_norm = comm.allreduce_sum_f64(b_vec.iter().map(|x| x * x).sum::<f64>()).sqrt();
        let b_sum = comm.allreduce_sum_f64(b_vec.iter().sum::<f64>());
        let j_norm = comm.allreduce_sum_f64(
            j_cond.owned_slice().iter().map(|x| x * x).sum::<f64>()).sqrt();
        if is_root {
            out.push_str(&format!("  ||A||_2 = {:.8e}, sum = {:.8e}, checksum = {:.8e}\n",
                solution_norm, solution_sum, solution_checksum));
            out.push_str(&format!("  ||B||_2 = {:.8e}, sum = {:.8e}\n", b_norm, b_sum));
            out.push_str(&format!("  ||J||_2 = {:.8e}\n", j_norm));
        }

        // ── 10. Save outputs per rank (C++ file names) ────────────────────────
        {
            let dummy2d = Mesh::<2>::unit_square_tri(1);
            let cond_mesh_name = format!("cond_mesh.{:06}", rank);
            let mut f = std::fs::File::create(&cond_mesh_name).expect("cond_mesh");
            write_mfem(&mut f, &dummy2d, Some(&sub_mesh)).expect("write cond_mesh");
            let cond_name = format!("cond_j.{:06}", rank);
            let mut f = std::fs::File::create(&cond_name).expect("cond_j");
            for &v in j_cond.owned_slice() {
                writeln!(f, "{:.14e}", v).expect("cond_j write");
            }
            let mesh_name = format!("mesh.{:06}", rank);
            let mut f = std::fs::File::create(&mesh_name).expect("mesh");
            write_mfem(&mut f, &dummy2d, Some(&local_mesh)).expect("write mesh");
            let sol_name = format!("sol.{:06}", rank);
            let mut f = std::fs::File::create(&sol_name).expect("sol");
            for &v in a_sol.owned_slice() {
                writeln!(f, "{:.14e}", v).expect("sol write");
            }
            let dsol_name = format!("dsol.{:06}", rank);
            let mut f = std::fs::File::create(&dsol_name).expect("dsol");
            for &v in &b_vec {
                writeln!(f, "{:.14e}", v).expect("dsol write");
            }
        }
        if is_root {
            out.push_str("  Wrote cond_mesh/cond_j/mesh/sol/dsol per rank\n\nFinished.\n");
            println!("{out}");
        }
    });
}
