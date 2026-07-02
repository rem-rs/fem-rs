use std::fs;
use std::time::Instant;

use fem_assembly::{
    assemble_degraded_stiffness, assemble_phase_field_system,
    assemble_miehe_stiffness_and_force, build_elem_dof_cache,
    compute_psi_plus, update_history_field,
};
use fem_assembly::phasefield::apply_dirichlet as pf_apply_dirichlet;
use fem_io::vtk::{DataArray, VtkWriter};
use fem_linalg::CsrMatrix;
use fem_mesh::{SimplexMesh, element_type::ElementType};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{H1Space, VectorH1Space, constraints::boundary_dofs, fe_space::FESpace};

/// Build a notched mesh for the SEN tension test.
/// Domain: [0,1]×[0,1], horizontal notch from (0,0.5) to (0.5,0.5).
/// `n` elements per side of each half (total ~8n² triangles).
fn build_sen_mesh(n: usize) -> SimplexMesh<2> {
    let h = 0.5 / n as f64;
    let mut coords = Vec::new();
    let mut conn = Vec::new();
    let mut elem_tags = Vec::new();
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();

    // Layout: domain split into 4 blocks:
    //  A: [0,0.5]×[0.5,1] with separate bottom nodes (crack top face)
    //  C: [0,0.5]×[0,0.5] with separate top nodes (crack bottom face)
    //  B: [0.5,1]×[0.5,1]
    //  D: [0.5,1]×[0,0.5]
    // Blocks meet at x=0.5, y∈[0,1] where nodes are shared.

    // Use a grid-based node numbering to avoid hash maps.
    let nn = n;
    // Total grid: [0..2n] in x, [0..2n] in y (including crack duplication)
    // Nodes at x=0..0.5 (indices 0..n) are duplicated at y=0.5:
    //   bottom row of A (yi=0, xi=0..n) = crack top edge
    //   top row of C (yi=n-1, but now we need a separate row) = crack bottom edge

    // Simpler: generate all nodes in a 4-block approach.

    // We need the crack tip at (0.5, 0.5) to have a single shared node.
    // Blocks: A (top-left), C (bottom-left, separate crack face), B (top-right), D (bottom-right)
    // A and C are separate at y=0.5 EXCEPT at the crack tip x=0.5 where they share.

    // Block A: [0,0.5]×[0.5,1], nodes (n+1)×(n+1)
    let nx_a = nn + 1;
    let ny_a = nn + 1;
    let mut a = vec![0u32; nx_a * ny_a];
    for yi in 0..ny_a {
        for xi in 0..nx_a {
            a[yi * nx_a + xi] = coords.len() as u32 / 2;
            coords.push(xi as f64 * h);
            coords.push(0.5 + yi as f64 * h);
        }
    }

    // Block C: [0,0.5]×[0,0.5], nodes (n+1)×(n+1)
    // The TOP row of C (yi=nn) = crack bottom face.
    // At the crack tip (xi=nn, yi=nn), we share node with A's bottom-right corner.
    let nx_c = nn + 1;
    let ny_c = nn + 1;
    let mut c = vec![0u32; nx_c * ny_c];
    for yi in 0..ny_c {
        for xi in 0..nx_c {
            if xi == nn && yi == nn {
                // Crack tip: share with block A's (nn, 0) node
                c[yi * nx_c + xi] = a[0 * nx_a + nn];
            } else {
                c[yi * nx_c + xi] = coords.len() as u32 / 2;
                coords.push(xi as f64 * h);
                coords.push(yi as f64 * h);
            }
        }
    }

    // Block B: [0.5,1]×[0.5,1], nodes (n+1)×(n+1) sharing with A along x=0.5
    let nx_b = nn + 1;
    let ny_b = nn + 1;
    let mut b = vec![0u32; nx_b * ny_b];
    for yi in 0..ny_b {
        for xi in 0..nx_b {
            let x = 0.5 + xi as f64 * h;
            let y = 0.5 + yi as f64 * h;
            if xi == 0 {
                b[yi * nx_b + xi] = a[yi * nx_a + nn];
            } else {
                b[yi * nx_b + xi] = coords.len() as u32 / 2;
                coords.push(x);
                coords.push(y);
            }
        }
    }

    // Block D: [0.5,1]×[0,0.5], nodes (n+1)×(n+1) sharing with C along x=0.5
    // At the crack tip (xi=0, yi=nn): share node at (0.5, 0.5) which is c[nn][nn] = a[0][nn]
    let nx_d = nn + 1;
    let ny_d = nn + 1;
    let mut d = vec![0u32; nx_d * ny_d];
    for yi in 0..ny_d {
        for xi in 0..nx_d {
            let x = 0.5 + xi as f64 * h;
            let y = yi as f64 * h;
            if xi == 0 && yi == nn {
                // Crack tip: share with block A/B/C tip node
                d[yi * nx_d + xi] = a[0 * nx_a + nn];
            } else if xi == 0 {
                d[yi * nx_d + xi] = c[yi * nx_c + nn];
            } else {
                d[yi * nx_d + xi] = coords.len() as u32 / 2;
                coords.push(x);
                coords.push(y);
            }
        }
    }

    // Helper to make 2 triangles from a quad (CCW: nw→ne→se→sw)
    let mut mk_quad = |nw: u32, ne: u32, se: u32, sw: u32| {
        conn.push(nw); conn.push(ne); conn.push(se); elem_tags.push(1);
        conn.push(nw); conn.push(se); conn.push(sw); elem_tags.push(1);
    };

    // Block A elements: bottom row of A (yi=0) = crack top edge
    for yi in 0..nn {
        for xi in 0..nn {
            mk_quad(
                a[(yi+1)*nx_a + xi],     a[(yi+1)*nx_a + xi+1],
                a[yi*nx_a + xi+1],        a[yi*nx_a + xi],
            );
        }
    }

    // Block C elements: top row of C (yi=nn) = crack bottom edge
    for yi in 0..nn {
        for xi in 0..nn {
            mk_quad(
                c[(yi+1)*nx_c + xi],     c[(yi+1)*nx_c + xi+1],
                c[yi*nx_c + xi+1],        c[yi*nx_c + xi],
            );
        }
    }

    // Block B elements
    for yi in 0..nn {
        for xi in 0..nn {
            mk_quad(
                b[(yi+1)*nx_b + xi],     b[(yi+1)*nx_b + xi+1],
                b[yi*nx_b + xi+1],        b[yi*nx_b + xi],
            );
        }
    }

    // Block D elements
    for yi in 0..nn {
        for xi in 0..nn {
            mk_quad(
                d[(yi+1)*nx_d + xi],     d[(yi+1)*nx_d + xi+1],
                d[yi*nx_d + xi+1],        d[yi*nx_d + xi],
            );
        }
    }

    // --- Boundary faces ---
    // Top: top row of A (yi=nn, xi=0..n)
    let top_yi = nn;
    for xi in 0..nn {
        face_conn.push(a[top_yi*nx_a + xi]);
        face_conn.push(a[top_yi*nx_a + xi + 1]);
        face_tags.push(2);
    }
    // Top: top row of B (yi=nn, xi=0..n)
    if nx_b > 0 {
        for xi in 0..nn {
            face_conn.push(b[top_yi*nx_b + xi]);
            face_conn.push(b[top_yi*nx_b + xi + 1]);
            face_tags.push(2);
        }
    }

    // Bottom: bottom row of D (yi=0, xi=0..n)
    for xi in 0..nn {
        face_conn.push(d[0*nx_d + xi]);
        face_conn.push(d[0*nx_d + xi + 1]);
        face_tags.push(1);
    }
    // Bottom: bottom row of C (yi=0, xi=0..n)
    for xi in 0..nn {
        face_conn.push(c[0*nx_c + xi]);
        face_conn.push(c[0*nx_c + xi + 1]);
        face_tags.push(1);
    }

    // Left: left column of A (xi=0, yi=0..n)
    for yi in 0..nn {
        face_conn.push(a[yi*nx_a + 0]);
        face_conn.push(a[(yi+1)*nx_a + 0]);
        face_tags.push(3);
    }
    // Left: left column of C (xi=0, yi=0..n)
    for yi in 0..nn {
        face_conn.push(c[yi*nx_c + 0]);
        face_conn.push(c[(yi+1)*nx_c + 0]);
        face_tags.push(3);
    }

    // Right: right column of B (xi=nn, yi=0..n)
    for yi in 0..nn {
        face_conn.push(b[yi*nx_b + nn]);
        face_conn.push(b[(yi+1)*nx_b + nn]);
        face_tags.push(4);
    }
    // Right: right column of D (xi=nn, yi=0..n)
    for yi in 0..nn {
        face_conn.push(d[yi*nx_d + nn]);
        face_conn.push(d[(yi+1)*nx_d + nn]);
        face_tags.push(4);
    }

    SimplexMesh {
        coords,
        conn,
        elem_tags,
        elem_type: ElementType::Tri3,
        face_conn,
        face_tags,
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = if args.len() > 1 { args[1].parse::<usize>().unwrap_or(40) } else { 40 };
    let n_load_steps = if args.len() > 2 { args[2].parse::<usize>().unwrap_or(50) } else { 50 };
    let max_disp = if args.len() > 3 { args[3].parse::<f64>().unwrap_or(0.01) } else { 0.01 };
    let csv_path = if args.len() > 4 { args[4].clone() } else { "output/phasefield_force_disp.csv".to_string() };
    let vtk_path = if args.len() > 5 { args[5].clone() } else { "output/phasefield_final.vtu".to_string() };

    let e_mod = 2.1e5;
    let nu = 0.3;
    let g_c = 2.7e-3;
    let l = 0.015;
    let kappa_eps = 1e-6;

    let lambda = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e_mod / (2.0 * (1.0 + nu));

    println!("Phase-field fracture: SEN tension test");
    println!("  n={n}, load_steps={n_load_steps}, max_disp={max_disp}");
    println!("  E={e_mod}, nu={nu}, G_c={g_c}, l={l}, kappa_eps={kappa_eps:.1e}");

    println!("Building mesh...");
    let mesh = build_sen_mesh(n);
    let n_elems = mesh.n_elems();
    let n_nodes = mesh.n_nodes();
    println!("  nodes={n_nodes}, elems={n_elems}");

    let order: u8 = 1;
    let quad_order: u8 = 2;
    let space_d = H1Space::new(mesh.clone(), order);
    let space_u = VectorH1Space::new(mesh.clone(), order, 2);

    let (d_elem_dofs, d_n_ldofs) = build_elem_dof_cache(&space_d);
    let (u_elem_dofs, u_n_ldofs) = build_elem_dof_cache(&space_u);

    let n_u = space_u.n_dofs();
    let n_d = space_d.n_dofs();

    let mut d = vec![0.0; n_d];
    let mut u = vec![0.0; n_u];

    // Initialize history field using ψ⁺ (Miehe split) for anisotropic degradation
    let (psi_init, n_qp_per_elem) = compute_psi_plus(
        &mesh, &space_u, &u_elem_dofs, u_n_ldofs, &u, lambda, mu, quad_order,
    );
    let mut h_field = psi_init.clone();

    let dt = max_disp / n_load_steps as f64;

    // Dirichlet BCs for phase field: d=0 on all boundaries (far-field undamaged)
    let bd_d = boundary_dofs(&mesh, space_d.dof_manager(), &[1, 2, 3, 4]);
    let bd_d_usize: Vec<usize> = bd_d.iter().map(|&d| d as usize).collect();
    let bd_d_vals = vec![0.0; bd_d_usize.len()];

    let solver_cfg = SolverConfig {
        rtol: 1e-8, atol: 0.0, max_iter: 10000, verbose: false,
        ..SolverConfig::default()
    };

    println!("Starting staggered load-stepping...");
    let t0 = Instant::now();
    let mut csv = String::new();
    csv.push_str("step,disp,reaction_y,max_d,max_dof\n");

    for step in 1..=n_load_steps {
        let disp = step as f64 * dt;

            // Build mechanics BC: top edge y-dir = disp, bottom clamped
        let bd_top = boundary_dofs(&mesh, space_u.scalar_dof_manager(), &[2]);
        let bd_bot = boundary_dofs(&mesh, space_u.scalar_dof_manager(), &[1]);
        // For VectorH1Space: x-DOFs are 0..n_scalar, y-DOFs are n_scalar..2*n_scalar
        let n_scalar = space_u.n_scalar_dofs();
        let mut bc_dofs_u = Vec::new();
        let mut bc_vals_u = Vec::new();
        // Bottom: fix both x and y
        for &d in &bd_bot {
            bc_dofs_u.push(d as usize);       // x-component
            bc_vals_u.push(0.0);
            bc_dofs_u.push(d as usize + n_scalar); // y-component
            bc_vals_u.push(0.0);
        }
        // Top: fix x, prescribe y = disp
        for &d in &bd_top {
            bc_dofs_u.push(d as usize);
            bc_vals_u.push(0.0);
            bc_dofs_u.push(d as usize + n_scalar);
            bc_vals_u.push(disp);
        }

        // Staggered coupling with Miehe spectral split (anisotropic degradation)
        // Mechanics: Newton iterations with split tangent K_T = [(1-d)²+κ]·C⁺ + C⁻
        // History: ψ⁺ only (compressive strain energy does NOT drive fracture)
        let pf_solver_cfg = SolverConfig {
            rtol: 1e-10, atol: 0.0, max_iter: 20000, verbose: false,
            ..SolverConfig::default()
        };
        for _stag in 0..10 {
            // 1. Solve degraded elasticity with Miehe split (Newton)
            let newton_tol = 1e-10;
            for _newton in 0..5 {
                let (k_uu, f_int) = assemble_miehe_stiffness_and_force(
                    &mesh, &space_u, &u_elem_dofs, u_n_ldofs,
                    &u, &d, &space_d, &d_elem_dofs, d_n_ldofs,
                    lambda, mu, kappa_eps, quad_order,
                );

                // Residual: r = f_int(u) - f_ext, where f_ext = 0
                let mut rhs = vec![0.0; n_u];
                for i in 0..n_u { rhs[i] = -f_int[i]; }

                // BC for Δu: Δu_i = prescribed_val - u_curr_i
                let mut k_bc = k_uu;
                let mut bc_delta = vec![0.0; bc_dofs_u.len()];
                for i in 0..bc_dofs_u.len() {
                    bc_delta[i] = bc_vals_u[i] - u[bc_dofs_u[i]];
                }
                pf_apply_dirichlet(&mut k_bc, &mut rhs, &bc_dofs_u, &bc_delta);

                let mut du = vec![0.0; n_u];
                let res = solve_pcg_jacobi(&k_bc, &rhs, &mut du, &solver_cfg)
                    .expect("mechanics Newton PCG failed");
                let _ = res;

                // Line search: u += du
                let du_norm: f64 = du.iter().map(|v| v * v).sum::<f64>().sqrt();
                let u_norm: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-32);
                for i in 0..n_u { u[i] += du[i]; }

                if du_norm < newton_tol * u_norm {
                    break;
                }
            }

            // 2. Update history field with ψ⁺ (tensile part only)
            let (psi_plus, _) = compute_psi_plus(
                &mesh, &space_u, &u_elem_dofs, u_n_ldofs, &u, lambda, mu, quad_order,
            );
            update_history_field(&mut h_field, &psi_plus);

            // 3. Solve phase field
            let (k_dd, rhs_d) = assemble_phase_field_system(
                &mesh, &space_d, &d_elem_dofs, d_n_ldofs,
                &h_field, n_qp_per_elem, g_c, l, quad_order,
            );

            let mut kd_bc = k_dd.clone();
            let mut rd_bc = rhs_d.clone();
            pf_apply_dirichlet(&mut kd_bc, &mut rd_bc, &bd_d_usize, &bd_d_vals);

            // Use previous d as initial guess for faster convergence
            let mut d_new = d.clone();
            let res = solve_pcg_jacobi(&kd_bc, &rd_bc, &mut d_new, &pf_solver_cfg)
                .expect("phase field PCG failed");
            let _ = res;

            // Clamp d to [0, 1] (AT2 model bound)
            for di in d_new.iter_mut() {
                *di = di.clamp(0.0, 1.0);
            }

            let d_diff: f64 = d.iter().zip(d_new.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>() / n_d as f64;
            d = d_new;

            if d_diff < 1e-10 {
                break;
            }
        }

        // Compute reaction force on bottom (y-component)
        let k_uu = assemble_degraded_stiffness(
            &mesh, &space_u, &u_elem_dofs, u_n_ldofs,
            &d, &space_d, &d_elem_dofs, d_n_ldofs,
            lambda, mu, kappa_eps, quad_order,
        );
        let reaction = compute_reaction_y(&k_uu, &u, &bd_bot, n_scalar);

        let max_d = d.iter().cloned().fold(0.0, f64::max);
        let n_damaged = d.iter().filter(|&&v| v > 0.5).count();

        println!("  step {step}: disp={disp:.6e}, reaction_y={reaction:.6e}, max_d={max_d:.6e}, n_damaged={n_damaged}");
        csv.push_str(&format!("{step},{disp:.12e},{reaction:.12e},{max_d:.12e},{n_damaged}\n"));

        if step % 10 == 0 {
            let elapsed = t0.elapsed();
            println!("    elapsed={elapsed:.1?}");
        }
    }

    fs::write(&csv_path, &csv).unwrap_or_else(|e| panic!("failed to write CSV: {e}"));
    println!("Force-displacement data written to {csv_path}");

    // VTK output
    fs::create_dir_all("output").ok();
    let mut writer = VtkWriter::new(&mesh);
    writer.add_point_data(DataArray::scalars("damage", d.clone()));
    writer.add_point_data(DataArray::vectors("displacement", 2, u.clone()));
    writer.write_file(&vtk_path).unwrap_or_else(|e| panic!("failed to write VTK: {e}"));
    println!("VTK output written to {vtk_path}");

    let elapsed = t0.elapsed();
    println!("Total elapsed: {elapsed:.1?}");
}

/// Compute y-reaction force on the bottom edge.
fn compute_reaction_y(k: &CsrMatrix<f64>, u: &[f64], bot_dofs: &[u32], n_scalar: usize) -> f64 {
    let mut ku = vec![0.0; u.len()];
    k.spmv(u, &mut ku);
    bot_dofs.iter().map(|&d| ku[d as usize + n_scalar]).sum()
}
