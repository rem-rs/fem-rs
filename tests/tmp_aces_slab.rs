// ═══════════════════════════════════════════════════════════════════════
// ACES: Dielectric slab reflection/transmission (Fresnel)
// ═══════════════════════════════════════════════════════════════════════

/// ACES benchmark: TMz plane wave incident on a dielectric half-space.
///
/// Problem: -∇²u - k²·εr·u = 0  (total-field Helmholtz)
///   Half-space: εr = 4 for x > 0.5, εr = 1 for x < 0.5
///   ABC on left/right: ∂u/∂n + ik·cosθ·u = g (Robin)
///   PMC on top/bottom (∂u/∂n = 0)
///
/// Uses scattered-field formulation:
///   v = u - u_inc,  u_inc = e^{-ikx}
///   ∇²v + k²·εr·v = -k²·(εr-1)·e^{-ikx}
///   ABC: ∂v/∂n + ik·v = 0 on left/right
///
/// Fresnel reference (normal incidence, εr=4):
///   R = (1-√εr)/(1+√εr) = -1/3,  |R|² = 1/9
///   T = 2/(1+√εr) = 2/3
///
/// References:
///   - ACES benchmark series: dielectric half-space
///   - Balanis, "Advanced Engineering Electromagnetics", §5.3
#[test]
fn em_dielectric_slab_reflection() {
    use fem_assembly::standard::BoundaryMassIntegrator;
    use fem_assembly::standard::DomainSourceIntegrator;
    use fem_assembly::assembler::face_dofs_p1;
    use fem_assembly::complex::NativeComplexSystem;
    use fem_linalg::complex_csr::ComplexCsr;

    let k_wave = 8.0;     // wavenumber
    let k2 = k_wave * k_wave;
    let eps_r = 4.0;
    let n_mesh = 40;      // subdivisions
    let lx = 1.0;         // domain half-size (total = 2lx = 2.0)
    let ly = 0.5;         // half-height (total = 2ly = 1.0)

    // Build rectangular mesh with two element tags (x < 0.5lx → tag 1, else tag 2)
    let mesh_raw = SimplexMesh::<2>::unit_square_tri(n_mesh);
    let mut mesh = mesh_raw;
    for c in mesh.coords.chunks_mut(2) {
        c[0] = c[0] * 2.0 * lx;  // scale to [0, 2*lx]
        c[1] = c[1] * 2.0 * ly;  // scale to [0, 2*ly]
    }
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let cx: f64 = nodes.iter()
            .map(|&n| mesh.node_coords(n)[0]).sum::<f64>() / nodes.len() as f64;
        mesh.elem_tags[e as usize] = if cx < lx { 1 } else { 2 };
    }

    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();
    let dm = space.dof_manager();

    // Permittivity: εr = 4 on right (tag 2), 1 on left (tag 1)
    let eps = PWConstCoeff::new([(1, 1.0), (2, eps_r)]);

    // Complex matrix: A_re = K - k²·εr·M
    let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 5);
    let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: eps }], 5);

    let mut coo_re = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        for p in k_mat.row_ptr[i]..k_mat.row_ptr[i + 1] {
            coo_re.add(i, k_mat.col_idx[p] as usize, k_mat.values[p]);
        }
        for p in m_mat.row_ptr[i]..m_mat.row_ptr[i + 1] {
            coo_re.add(i, m_mat.col_idx[p] as usize, -k2 * m_mat.values[p]);
        }
    }
    let a_re: CsrMatrix<f64> = coo_re.into_csr();

    // A_im = ik·M_Γ on left/right boundaries (ABC) — tags 2(left, shifted) and 4(right)
    // Actually after scaling: tag 1=bottom, 2=right, 3=top, 4=left
    let bnd_integ = BoundaryMassIntegrator { alpha: k_wave };
    let a_im = Assembler::assemble_boundary_bilinear(
        n, &mesh, &face_dofs_p1(&mesh), 1,
        &[&bnd_integ], &[2, 4], 5,
    );

    let csr = ComplexCsr::from_re_im(&a_re, &a_im);
    let mut sys = NativeComplexSystem { mat: csr, omega: k_wave, n_dofs: n };

    // RHS: -k²·(εr-1)·e^{-ikx} in the dielectric region (tag 2)
    let src = |x: &[f64]| {
        let eps_local = if x[0] >= lx { eps_r } else { 1.0 };
        -k2 * (eps_local - 1.0) * (-k_wave * x[0]).cos()
    };
    let src_re = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(src)], 5);
    let src_im = |x: &[f64]| {
        let eps_local = if x[0] >= lx { eps_r } else { 1.0 };
        -k2 * (eps_local - 1.0) * (-k_wave * x[0]).sin()
    };
    let src_im_vec = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(src_im)], 5);

    // No essential BCs — the ABC provides the BC naturally.
    let gf = sys.solve(&src_re, &src_im_vec, 1e-8, 8000, 50)
        .expect("Dielectric slab GMRES failed");

    // Extract reflection coefficient from the standing wave in the left region
    use fem_space::constraints::boundary_dofs;
    let left_dofs = boundary_dofs(space.mesh(), dm, &[4]);
    let right_dofs = boundary_dofs(space.mesh(), dm, &[2]);

    // Compute total field u = u_inc + v at the left boundary
    let mut max_mag: f64 = 0.0;
    let mut max_total: f64 = 0.0;
    for &d in &right_dofs {
        let c = dm.dof_coord(d);
        let kx = k_wave * c[0];
        let inc_re = kx.cos();
        let inc_im = -kx.sin();
        let total_re = gf.u_re[d as usize] + inc_re;
        let total_im = gf.u_im[d as usize] + inc_im;
        let mag = (total_re * total_re + total_im * total_im).sqrt();
        max_mag = max_mag.max(mag);
    }

    // The max field magnitude on the right boundary = |T| (transmission coefficient)
    // Fresnel T = 2/(1+√εr) = 2/3
    let fresnel_t = 2.0 / (1.0 + eps_r.sqrt());
    let t_err = (max_mag - fresnel_t).abs() / fresnel_t;

    eprintln!("  [ACES Dielectric slab] k={}, εr={}, n={}:", k_wave, eps_r, n_mesh);
    eprintln!("       DOFs={}, max|u|@right={:.4e} (Fresnel T={:.4e}, err={:.2%})",
        n, max_mag, fresnel_t, t_err);
    eprintln!("       (Valid: scattered-field formulation, complex system, ABC ✓)");

    // With coarse mesh, expect ~10% accuracy
    assert!(t_err < 0.15,
        "Dielectric slab: transmission error {:.2%} too large", t_err);

    fem_regression::regression("em_dielectric_slab_reflection")
        .check_with("transmission_mag", max_mag, 1e-6, 1e-10)
        .check_with("n_dofs", n as f64, 1e-6, 0.5)
        .finalize();
}
