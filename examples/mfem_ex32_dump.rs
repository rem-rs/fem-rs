//! ex32 dump — Rust side A/M matrix dump for 1:1 comparison with C++ ex32p
//! (tools/ex32_cpp_helper/ex32_dump.cpp).  Dumps the same file set:
//!   rust_ndof.txt / rust_rtdof.txt   H(Curl) / H(Div) unknowns
//!   rust_ess.txt                     essential dofs (boundary_dofs_hcurl)
//!   rust_A_elim.txt                  A after eliminate_essential_bc_diag(1.0)
//!   rust_M_elim.txt                  M after eliminate_essential_bc_diag(min)
//!
//! Run: cargo run --example mfem_ex32_dump -- -m data/fichera.mesh -rs 2 -o 1
//! (run from the tools/ex32_cpp_helper dir or copy the txt files there)

use std::f64::consts::SQRT_2;
use std::fs::File;
use std::io::{Write, BufWriter};

use fem_assembly::{
    VectorAssembler, ConstantMatrixCoeff,
    standard::{CurlCurlIntegrator, VectorMassTensorIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology, refine_uniform_3d};
use fem_space::{
    H1Space, HCurlSpace, HDivSpace,
    constraints::boundary_dofs_hcurl,
    fe_space::FESpace,
};

struct Args { mesh_file: String, ser_ref_levels: usize, order: u8 }
fn parse_args() -> Args {
    let mut a = Args { mesh_file: "data/fichera.mesh".into(), ser_ref_levels: 2, order: 1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = it.next().unwrap_or("data/fichera.mesh".into()),
            "-rs" | "--refine-serial" => a.ser_ref_levels = it.next().unwrap_or("2".into()).parse().unwrap_or(2),
            "-o" | "--order" => a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            _ => {}
        }
    }
    a
}

fn dump_sparse(path: &str, m: &fem_linalg::CsrMatrix<f64>) {
    let f = File::create(path).expect(path);
    let mut w = BufWriter::new(f);
    for r in 0..m.nrows {
        write!(w, "[row {}]", r).unwrap();
        for k in m.row_ptr[r]..m.row_ptr[r + 1] {
            write!(w, " ({},{})", m.col_idx[k], m.values[k]).unwrap();
        }
        write!(w, "\n").unwrap();
    }
}

fn main() {
    let args = parse_args();

    let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
    let mut mesh: Mesh<3> = mfem.mesh3d.expect("ex32 requires a 3D mesh");
    eprintln!("DIAG: initial: n_elems={} n_nodes={} bfaces={} face_type={:?} face_offsets={:?} face_conn_len={}",
        mesh.n_elems(), mesh.n_nodes(), mesh.n_boundary_faces(),
        mesh.face_type, mesh.face_offsets.as_ref().map(|v| v.len()), mesh.face_conn.len());
    for lev in 0..args.ser_ref_levels {
        mesh = refine_uniform_3d(&mesh);
        eprintln!("DIAG: after refine[{}]: n_elems={} n_nodes={} bfaces={} face_type={:?} face_conn_len={}",
            lev, mesh.n_elems(), mesh.n_nodes(), mesh.n_boundary_faces(),
            mesh.face_type, mesh.face_conn.len());
    }

    let order = args.order;
    let rt_order = if order > 0 { order - 1 } else { 0 };
    let fec_nd = HCurlSpace::new(mesh.clone(), order);
    let fec_rt = HDivSpace::new(mesh.clone(), rt_order);
    let n_nd = fec_nd.n_dofs();
    let n_rt = fec_rt.n_dofs();

    // Dump unknown counts.
    {
        let mut f = File::create("rust_ndof.txt").unwrap();
        write!(f, "{}\n", n_nd).unwrap();
        let mut g = File::create("rust_rtdof.txt").unwrap();
        write!(g, "{}\n", n_rt).unwrap();
    }

    let inv_sqrt2 = 1.0 / SQRT_2;
    let epsilon_coeff = ConstantMatrixCoeff(vec![
        2.0, inv_sqrt2, 0.0,
        inv_sqrt2, 2.0, inv_sqrt2,
        0.0, inv_sqrt2, 2.0,
    ]);

    let quad_order = 2 * order as u8 + 1;

    let mut a_mat = VectorAssembler::assemble_bilinear(
        &fec_nd, &[&CurlCurlIntegrator { mu: 1.0 }], quad_order,
    );
    let mut m_mat = VectorAssembler::assemble_bilinear(
        &fec_nd, &[&VectorMassTensorIntegrator { alpha: ConstantMatrixCoeff(vec![
            2.0, inv_sqrt2, 0.0,
            inv_sqrt2, 2.0, inv_sqrt2,
            0.0, inv_sqrt2, 2.0,
        ]) }], quad_order,
    );

    let nd_mesh = fec_nd.mesh();
    let all_tags: Vec<i32> = nd_mesh.unique_boundary_tags();
    let ess_bdr_nd = if all_tags.is_empty() { vec![] }
        else { boundary_dofs_hcurl(nd_mesh, &fec_nd, &all_tags) };

    // Dump essential dofs (sorted, same as C++ GetEssentialTrueDofs order is
    // ascending dof index for a conforming serial space).
    {
        let mut mut_ess = ess_bdr_nd.clone();
        mut_ess.sort_unstable();
        let mut f = File::create("rust_ess.txt").unwrap();
        for &d in &mut_ess { write!(f, "{}\n", d).unwrap(); }
    }

    // Dump per-dof physical coords (edge midpoint for ND1 3D) — anchor for the
    // dof permutation used in matrix comparison (cf. cpp_dofpos.txt).
    {
        let dp = fec_nd.dof_coords();
        let mut f = File::create("rust_dofpos.txt").unwrap();
        for p in &dp { write!(f, "{} {} {}\n", p[0], p[1], p[2]).unwrap(); }
    }

    // Element-0 vertex coords (topology anchor, cf. cpp_elem0_verts.txt).
    {
        let nodes = mesh.element_nodes(0);
        let mut f = File::create("rust_elem0_verts.txt").unwrap();
        for &n in nodes.iter() {
            let c = mesh.node_coords(n);
            write!(f, "{} {} {}\n", c[0], c[1], c[2]).unwrap();
        }
        let mut g = File::create("rust_elem0_dofs.txt").unwrap();
        for &d in fec_nd.element_dofs(0).iter() { write!(g, "{}\n", d).unwrap(); }
    }

    // Element-0 elmats BEFORE BC elimination (12x12 ND1 hex), same quadrature
    // orders as MFEM: CurlCurlIntegrator -> IntRules order 2p-2=0,
    // VectorFEMassIntegrator -> order 2p=2.
    {
        use fem_assembly::vector_assembler::accumulate_vector_bilinear_element;
        use fem_linalg::CooMatrix;
        let mut dump_elmat = |name: &str, integ: &dyn fem_assembly::vector_integrator::VectorBilinearIntegrator, qo: u8| {
            let mut coo = CooMatrix::<f64>::new(n_nd, n_nd);
            accumulate_vector_bilinear_element(&fec_nd, 0, &[integ], qo, &mut coo);
            let csr = coo.into_csr();
            let mut f = File::create(name).unwrap();
            write!(f, "dofs:").unwrap();
            for &d in fec_nd.element_dofs(0).iter() { write!(f, " {}", d).unwrap(); }
            write!(f, "\n").unwrap();
            for i in 0..12 {
                write!(f, "[row {}]", i).unwrap();
                for k in csr.row_ptr[i]..csr.row_ptr[i + 1] {
                    write!(f, " ({},{})", csr.col_idx[k], csr.values[k]).unwrap();
                }
                write!(f, "\n").unwrap();
            }
        };
        dump_elmat("rust_elmat_A_0.txt", &CurlCurlIntegrator { mu: 1.0 }, 0);
        let eps2 = ConstantMatrixCoeff(vec![
            2.0, inv_sqrt2, 0.0,
            inv_sqrt2, 2.0, inv_sqrt2,
            0.0, inv_sqrt2, 2.0,
        ]);
        dump_elmat("rust_elmat_M_0.txt",
            &fem_assembly::standard::VectorMassTensorIntegrator { alpha: eps2 }, 2);

        // DIAGNOSTIC: element-0 assembly intermediates at first quad point.
        use fem_element::{ReferenceElement, VectorReferenceElement, lagrange::HexQ1, nedelec::HexNDk};
        use fem_assembly::isoparametric_jacobian;
        let nodes = mesh.element_nodes(0);
        let geo = HexQ1;
        let quad = HexQ1.quadrature(2);
        for (q, xi) in quad.points.iter().enumerate().take(2) {
            let (jac, det, _xp) = isoparametric_jacobian(&mesh, nodes, &geo, xi, 3);
            let jit = jac.clone().try_inverse().unwrap().transpose();
            let mut rp = vec![0.0_f64; 36];
            HexNDk::new(1).eval_basis_vec(xi, &mut rp);
            let mut pp = vec![0.0_f64; 36];
            for i in 0..12 {
                for r in 0..3 {
                    let mut s = 0.0;
                    for c in 0..3 { s += jit[(r, c)] * rp[i * 3 + c]; }
                    pp[i * 3 + r] = s;
                }
            }
            eprintln!("DIAG el0 q={q}: xi={xi:?} det_j={det:.10} w={:.6} ref_phi0={:?} phys_phi0={:?}",
                quad.weights[q], &rp[0..3], &pp[0..3]);
        }
        // Manual M_00 accumulation (match integrator formula).
        let mut m00 = 0.0_f64;
        for (q, xi) in quad.points.iter().enumerate() {
            let (jac, det, _xp) = isoparametric_jacobian(&mesh, nodes, &geo, xi, 3);
            let jit = jac.clone().try_inverse().unwrap().transpose();
            let mut rp = vec![0.0_f64; 36];
            HexNDk::new(1).eval_basis_vec(xi, &mut rp);
            let mut pp = vec![0.0_f64; 36];
            for i in 0..12 {
                for r in 0..3 {
                    let mut s = 0.0;
                    for c in 0..3 { s += jit[(r, c)] * rp[i * 3 + c]; }
                    pp[i * 3 + r] = s;
                }
            }
            let au0 = 2.0 * pp[0] + inv_sqrt2 * pp[1];
            let au1 = inv_sqrt2 * pp[0] + 2.0 * pp[1] + inv_sqrt2 * pp[2];
            let au2 = inv_sqrt2 * pp[1] + 2.0 * pp[2];
            let dot = au0 * pp[0] + au1 * pp[1] + au2 * pp[2];
            m00 += quad.weights[q] * det.abs() * dot;
        }
        eprintln!("DIAG el0: manual M_00 = {m00:.12}  (expect 1/18 = {})", 1.0 / 18.0);
    }

    // Diagnostic: boundary edge accounting.
    {
        use std::collections::HashSet;
        let mut be: HashSet<(u32, u32)> = HashSet::new();
        let mut n_face_nodes_total = 0usize;
        for f in 0..mesh.n_boundary_faces() as u32 {
            let nodes = mesh.face_nodes(f);
            n_face_nodes_total += nodes.len();
            for i in 0..nodes.len() {
                let (a, b) = (nodes[i], nodes[(i + 1) % nodes.len()]);
                be.insert(if a < b { (a, b) } else { (b, a) });
            }
        }
        let mut hit = 0usize;
        let mut miss = 0usize;
        let mut miss_keys: Vec<String> = Vec::new();
        for &(a, b) in &be {
            if fec_nd.edge_dofs(fem_space::dof_manager::EdgeKey(a, b)).is_some() {
                hit += 1;
            } else {
                miss += 1;
                if miss_keys.len() < 8 { miss_keys.push(format!("({a},{b})")); }
            }
        }
        eprintln!("DIAG: boundary faces={} face_nodes_total={} unique_bdry_edges={} edge_dof_hit={} miss={}",
            mesh.n_boundary_faces(), n_face_nodes_total, be.len(), hit, miss);
        if !miss_keys.is_empty() { eprintln!("DIAG: miss samples: {}", miss_keys.join(" ")); }
        eprintln!("DIAG: n_edges(total mesh)={} n_dofs(nd)={}", fec_nd.n_edges(), fec_nd.n_dofs());
        {
            let m = &mesh;
            eprintln!("DIAG: face_type={:?} face_types={:?} face_offsets={:?} face_conn_len={}",
                m.face_type,
                m.face_types.as_ref().map(|v| v.len()),
                m.face_offsets.as_ref().map(|v| v.len()),
                m.face_conn.len());
            if let Some(offs) = &m.face_offsets {
                eprintln!("DIAG: face_offsets[0..5]={:?}", &offs[0..6.min(offs.len())]);
            }
            let n = m.bface_nodes(0).len();
            eprintln!("DIAG: bface_nodes(0) len={} {:?}", n, m.bface_nodes(0));
        }
    }

    for &d in &ess_bdr_nd {
        a_mat.eliminate_essential_bc_diag_symmetric(d as usize, 1.0);
        m_mat.eliminate_essential_bc_diag_symmetric(d as usize, f64::MIN_POSITIVE);
    }

    dump_sparse("rust_A_elim.txt", &a_mat);
    dump_sparse("rust_M_elim.txt", &m_mat);

    // Verify A·G ≈ 0 (curl-curl nullspace = range of discrete gradient G).
    // If G disagrees with the A/M DOF semantics, the AME/DivFree projector
    // cannot filter the nullspace and LOBPCG stalls.
    {
        use fem_assembly::{DiscreteLinearOperator};
        let fec_h1 = H1Space::new(mesh.clone(), 1);
        let grad = DiscreteLinearOperator::gradient(&fec_h1, &fec_nd)
            .expect("gradient assembly failed");
        let grad_t = grad.transpose(); // n_h1 × n_nd: row j = vertex-j gradient column
        let mut col = vec![0.0_f64; n_nd];
        let mut out = vec![0.0_f64; n_nd];
        let mut fro = 0.0_f64;
        let mut maxabs = 0.0_f64;
        for j in 0..fec_h1.n_dofs() {
            col.fill(0.0);
            for k in grad_t.row_ptr[j]..grad_t.row_ptr[j + 1] {
                col[grad_t.col_idx[k] as usize] = grad_t.values[k];
            }
            out.fill(0.0);
            for r in 0..n_nd {
                for k in a_mat.row_ptr[r]..a_mat.row_ptr[r + 1] {
                    out[r] += a_mat.values[k] * col[a_mat.col_idx[k] as usize];
                }
            }
            for r in 0..n_nd {
                fro += out[r] * out[r];
                maxabs = maxabs.max(out[r].abs());
            }
        }
        eprintln!("DIAG: ||A·G||_F = {:.3e}  max|A·G| = {:.3e}  (should be ≈ 0)",
            fro.sqrt(), maxabs);
        // Verify: A_elim · G_eff ≈ 0 where G_eff drops the boundary H1 columns
        // (PEC: φ|Γ=0 ⇒ ∇φ has zero tangential trace on boundary edges, so the
        // BC rows of G_eff vanish and G_eff spans the nullspace of the
        // *eliminated* A).  This mirrors MFEM HypreAMS which builds G from the
        // FE space with its essential dofs.
        {
            use fem_space::constraints::boundary_dofs;
            let h1_bdr = boundary_dofs(&mesh, fec_h1.dof_manager(),
                &mesh.unique_boundary_tags());
            let keep: Vec<usize> = (0..fec_h1.n_dofs())
                .filter(|&j| !h1_bdr.contains(&(j as u32))).collect();
            let mut fro2 = 0.0_f64;
            let mut max2 = 0.0_f64;
            for &j in &keep {
                col.fill(0.0);
                for k in grad_t.row_ptr[j]..grad_t.row_ptr[j + 1] {
                    col[grad_t.col_idx[k] as usize] = grad_t.values[k];
                }
                out.fill(0.0);
                for r in 0..n_nd {
                    for k in a_mat.row_ptr[r]..a_mat.row_ptr[r + 1] {
                        out[r] += a_mat.values[k] * col[a_mat.col_idx[k] as usize];
                    }
                }
                for r in 0..n_nd {
                    fro2 += out[r] * out[r];
                    max2 = max2.max(out[r].abs());
                }
            }
            eprintln!("DIAG: ||A_elim·G_eff||_F = {:.3e}  max = {:.3e}  (h1_bdr={} cols kept {}/{})",
                fro2.sqrt(), max2, h1_bdr.len(), keep.len(), fec_h1.n_dofs());
        }
        // De Rham check: CurlInterpolator · G should be exactly 0.
        {
            use fem_assembly::DiscreteLinearOperator;
            let fec_rt_local = HDivSpace::new(mesh.clone(), 0);
            let curl_op = DiscreteLinearOperator::curl_3d(&fec_nd, &fec_rt_local)
                .expect("curl_3d");
            let mut fro3 = 0.0_f64;
            let mut max3 = 0.0_f64;
            let mut col2 = vec![0.0_f64; n_nd];
            let mut out2 = vec![0.0_f64; fec_rt_local.n_dofs()];
            for j in 0..fec_h1.n_dofs() {
                col2.fill(0.0);
                for k in grad_t.row_ptr[j]..grad_t.row_ptr[j + 1] {
                    col2[grad_t.col_idx[k] as usize] = grad_t.values[k];
                }
                out2.fill(0.0);
                curl_op.spmv(&col2, &mut out2);
                for r in 0..fec_rt_local.n_dofs() {
                    fro3 += out2[r] * out2[r];
                    max3 = max3.max(out2[r].abs());
                }
            }
            eprintln!("DIAG: ||CurlInterpolator·G||_F = {:.3e}  max = {:.3e}  (de Rham, should be 0)",
                fro3.sqrt(), max3);
            // Dump C for analysis.
            let mut cf = File::create("rust_C.txt").unwrap();
            for r in 0..curl_op.nrows {
                write!(cf, "[row {}]", r).unwrap();
                for k in curl_op.row_ptr[r]..curl_op.row_ptr[r + 1] {
                    write!(cf, " ({},{})", curl_op.col_idx[k], curl_op.values[k]).unwrap();
                }
                write!(cf, "\n").unwrap();
            }
        }
        // Dump G for analysis.
        let mut gf = File::create("rust_G.txt").unwrap();
        for r in 0..grad.nrows {
            write!(gf, "[row {}]", r).unwrap();
            for k in grad.row_ptr[r]..grad.row_ptr[r + 1] {
                write!(gf, " ({},{})", grad.col_idx[k], grad.values[k]).unwrap();
            }
            write!(gf, "\n").unwrap();
        }
    }

    eprintln!("dumped: rust_ndof/rtdof/ess/A_elim/M_elim (n_nd={} n_rt={} ess={})",
              n_nd, n_rt, ess_bdr_nd.len());
}
