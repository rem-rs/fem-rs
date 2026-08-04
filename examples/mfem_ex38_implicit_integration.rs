//                                MFEM Example 38 — Rust port (1:1)
//
// Integration over implicit interfaces and subdomains bounded by implicit
// interfaces using moment-fitting quadrature rules (Mueller–Kummer–Oberlack
// 2013). 1:1 port of MFEM `examples/ex38.cpp` + `fem/intrules_cut.cpp`
// (`MomentFittingIntRules`).
//
// Sample runs:
//   cargo run --example mfem_ex38_implicit_integration -- -i surface2d -r 3 -o 2 -m 0 -no-vis
//   cargo run --example mfem_ex38_implicit_integration -- -i volumetric1d -r 3 -o 2 -m 0 -no-vis
//   cargo run --example mfem_ex38_implicit_integration -- -i volumetric2d -r 3 -o 2 -m 0 -no-vis
//   cargo run --example mfem_ex38_implicit_integration -- -i surface3d  -r 3 -o 2 -m 0 -no-vis
//   cargo run --example mfem_ex38_implicit_integration -- -i volumetric3d -r 3 -o 2 -m 0 -no-vis

use fem_assembly::cut::{CutGeom, CutRule, MomentFitting};
use fem_element::ReferenceElement;
use fem_mesh::{Mesh, refine_uniform, refine_uniform_3d};
use fem_space::{FESpace, H1Space};

#[derive(Clone, Copy, PartialEq, Debug)]
enum IntegrationType {
    Volumetric1D,
    Surface2D,
    Volumetric2D,
    Surface3D,
    Volumetric3D,
}

struct Args {
    ref_levels: i32,
    order: i32,
    itype: IntegrationType,
}

fn parse_args() -> Args {
    let mut args = Args {
        ref_levels: 3,
        order: 2,
        itype: IntegrationType::Surface2D,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" | "--order" => args.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(2),
            "-r" | "--refine" => args.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(3),
            "-i" | "--integration-type" => {
                let v = it.next().unwrap_or_default();
                args.itype = match v.as_str() {
                    "volumetric1d" | "Volumetric1D" => IntegrationType::Volumetric1D,
                    "volumetric2d" | "Volumetric2D" => IntegrationType::Volumetric2D,
                    "surface3d" | "Surface3d" => IntegrationType::Surface3D,
                    "volumetric3d" | "Volumetric3d" => IntegrationType::Volumetric3D,
                    _ => IntegrationType::Surface2D,
                };
            }
            "-m" | "--method" => {} // method 0 only (moments); 1 = Algoim
            "-vis" | "-no-vis" | "-pv" | "-no-pv" => {}
            _ => {}
        }
    }
    args
}

// ── Level-set, integrand and analytic values (ex38.cpp) ──────────────────────

fn lvlset(itype: IntegrationType, x: &[f64]) -> f64 {
    match itype {
        IntegrationType::Volumetric1D => 0.55 - x[0],
        IntegrationType::Surface2D => 1.0 - (x[0] * x[0] + x[1] * x[1]),
        IntegrationType::Volumetric2D => 1.0 - ((x[0] / 1.5).powi(2) + (x[1] / 0.75).powi(2)),
        IntegrationType::Surface3D => 1.0 - (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]),
        IntegrationType::Volumetric3D => 1.0
            - ((x[0] / 1.5).powi(2) + (x[1] / 0.75).powi(2) + (x[2] / 0.5).powi(2)),
    }
}

fn integrand(itype: IntegrationType, x: &[f64]) -> f64 {
    match itype {
        IntegrationType::Volumetric1D => x[0] * x[0],
        IntegrationType::Surface2D => 3.0 * x[0] * x[0] - x[1] * x[1],
        IntegrationType::Volumetric2D => 1.0,
        IntegrationType::Surface3D => 4.0 - 3.0 * x[0] * x[0] + 2.0 * x[1] * x[1] - x[2] * x[2],
        IntegrationType::Volumetric3D => 1.0,
    }
}

fn surface_true(itype: IntegrationType) -> f64 {
    match itype {
        IntegrationType::Volumetric1D => 0.3025,
        IntegrationType::Surface2D => 2.0 * std::f64::consts::PI,
        IntegrationType::Volumetric2D => 7.26633616541076,
        IntegrationType::Surface3D => 40.0 / 3.0 * std::f64::consts::PI,
        IntegrationType::Volumetric3D => 9.90182151329315,
    }
}

fn volume_true(itype: IntegrationType) -> f64 {
    match itype {
        IntegrationType::Volumetric1D => 0.55_f64.powi(3) / 3.0,
        IntegrationType::Surface2D | IntegrationType::Surface3D => f64::NAN,
        IntegrationType::Volumetric2D => 9.0 / 8.0 * std::f64::consts::PI,
        IntegrationType::Volumetric3D => 3.0 / 4.0 * std::f64::consts::PI,
    }
}

fn has_volume(itype: IntegrationType) -> bool {
    matches!(
        itype,
        IntegrationType::Volumetric1D | IntegrationType::Volumetric2D | IntegrationType::Volumetric3D
    )
}

// ── Assemble a linear form with a custom per-element integration rule ────────
//    (equivalent to MFEM `SurfaceLFIntegrator` / `SubdomainLFIntegrator` +
//    `LinearForm::Sum`; the H1(order=1) shape functions form a partition of
//    unity, so the sum equals Σ_e Σ_ip w·|J|·Q — we keep the full assembly to
//    match MFEM's accumulation order).

fn assemble_surface(
    mf: &mut MomentFitting<'_>,
    geom: &CutGeom,
    ref_elem: &dyn ReferenceElement,
    space: &dyn FESpace<Mesh = Mesh<2>>,
    itype: IntegrationType,
) -> f64 {
    let n_dofs = space.n_dofs();
    let mut lf = vec![0.0; n_dofs];
    let mut phi = vec![0.0; ref_elem.n_dofs()];
    for e in 0..geom.elem_verts.len() as u32 {
        let sir = mf.get_surface_integration_rule(e);
        let sw = mf.get_surface_weights(e, &sir);
        let sdofs = space.element_dofs(e);
        for ip in 0..sir.n_points() {
            let xi = &sir.points[ip];
            let p = geom.map_phys(e, xi);
            let val = geom.det_j(e) * integrand(itype, &p);
            ref_elem.eval_basis(xi, &mut phi);
            let w = sir.weights[ip] * sw[ip] * val;
            for (i, &d) in sdofs.iter().enumerate() {
                lf[d as usize] += w * phi[i];
            }
        }
    }
    lf.iter().sum()
}

fn assemble_surface_3d(
    mf: &mut MomentFitting<'_>,
    geom: &CutGeom,
    ref_elem: &dyn ReferenceElement,
    space: &dyn FESpace<Mesh = Mesh<3>>,
    itype: IntegrationType,
) -> f64 {
    let n_dofs = space.n_dofs();
    let mut lf = vec![0.0; n_dofs];
    let mut phi = vec![0.0; ref_elem.n_dofs()];
    for e in 0..geom.elem_verts.len() as u32 {
        let sir = mf.get_surface_integration_rule(e);
        let sw = mf.get_surface_weights(e, &sir);
        let sdofs = space.element_dofs(e);
        for ip in 0..sir.n_points() {
            let xi = &sir.points[ip];
            let p = geom.map_phys(e, xi);
            let val = geom.det_j(e) * integrand(itype, &p);
            ref_elem.eval_basis(xi, &mut phi);
            let w = sir.weights[ip] * sw[ip] * val;
            for (i, &d) in sdofs.iter().enumerate() {
                lf[d as usize] += w * phi[i];
            }
        }
    }
    lf.iter().sum()
}

fn assemble_volume(
    geom: &CutGeom,
    ref_elem: &dyn ReferenceElement,
    space: &dyn FESpace<Mesh = Mesh<2>>,
    itype: IntegrationType,
    rules: &[CutRule],
) -> f64 {
    let n_dofs = space.n_dofs();
    let mut lf = vec![0.0; n_dofs];
    let mut phi = vec![0.0; ref_elem.n_dofs()];
    for (e, rule) in rules.iter().enumerate() {
        let e = e as u32;
        let sdofs = space.element_dofs(e);
        for ip in 0..rule.n_points() {
            let xi = &rule.points[ip];
            let p = geom.map_phys(e, xi);
            let val = geom.det_j(e) * integrand(itype, &p);
            ref_elem.eval_basis(xi, &mut phi);
            let w = rule.weights[ip] * val;
            for (i, &d) in sdofs.iter().enumerate() {
                lf[d as usize] += w * phi[i];
            }
        }
    }
    lf.iter().sum()
}

fn assemble_volume_3d(
    geom: &CutGeom,
    ref_elem: &dyn ReferenceElement,
    space: &dyn FESpace<Mesh = Mesh<3>>,
    itype: IntegrationType,
    rules: &[CutRule],
) -> f64 {
    let n_dofs = space.n_dofs();
    let mut lf = vec![0.0; n_dofs];
    let mut phi = vec![0.0; ref_elem.n_dofs()];
    for (e, rule) in rules.iter().enumerate() {
        let e = e as u32;
        let sdofs = space.element_dofs(e);
        for ip in 0..rule.n_points() {
            let xi = &rule.points[ip];
            let p = geom.map_phys(e, xi);
            let val = geom.det_j(e) * integrand(itype, &p);
            ref_elem.eval_basis(xi, &mut phi);
            let w = rule.weights[ip] * val;
            for (i, &d) in sdofs.iter().enumerate() {
                lf[d as usize] += w * phi[i];
            }
        }
    }
    lf.iter().sum()
}

/// Direct quadrature sum (used in 1-D, where the S/C IntegrationRule weights
/// already contain all the geometry factors).
fn direct_sum(mf: &mut MomentFitting<'_>, geom: &CutGeom, itype: IntegrationType) -> (f64, f64) {
    let ne = geom.elem_verts.len() as u32;
    let mut s_sum = 0.0;
    for e in 0..ne {
        let sir = mf.get_surface_integration_rule(e);
        for ip in 0..sir.n_points() {
            let xi = &sir.points[ip];
            let p = geom.map_phys(e, xi);
            s_sum += sir.weights[ip] * geom.det_j(e) * integrand(itype, &p);
        }
    }
    let mut v_sum = 0.0;
    for e in 0..ne {
        let cir = mf.get_volume_integration_rule(e, None);
        for ip in 0..cir.n_points() {
            let xi = &cir.points[ip];
            let p = geom.map_phys(e, xi);
            v_sum += cir.weights[ip] * geom.det_j(e) * integrand(itype, &p);
        }
    }
    (s_sum, v_sum)
}

fn build_mesh_2d(ref_levels: i32) -> Mesh<2> {
    let mut mesh: Mesh<2> = Mesh::make_cartesian_2d(1, 1, 3.2, 3.2);
    mesh.translate([-1.6, -1.6]);
    for _ in 0..ref_levels {
        mesh = refine_uniform(&mesh);
    }
    mesh
}

fn build_mesh_3d(ref_levels: i32) -> Mesh<3> {
    let coords = vec![
        -1.6, -1.6, -1.6, 1.6, -1.6, -1.6, 1.6, 1.6, -1.6, -1.6, 1.6, -1.6, //
        -1.6, -1.6, 1.6, 1.6, -1.6, 1.6, 1.6, 1.6, 1.6, -1.6, 1.6, 1.6,
    ];
    let mut mesh: Mesh<3> = Mesh::uniform(
        coords,
        vec![0, 1, 2, 3, 4, 5, 6, 7],
        vec![1],
        fem_mesh::ElementType::Hex8,
        vec![],
        vec![],
        fem_mesh::ElementType::Quad4,
    );
    for _ in 0..ref_levels {
        mesh = refine_uniform_3d(&mesh);
    }
    mesh
}

fn main() {
    let args = parse_args();
    let order = args.order as usize;
    let ref_levels = args.ref_levels;
    let ls_order = 2usize;
    let itype = args.itype;
    let lvl_fn = move |x: &[f64]| lvlset(itype, x);

    let (surface_sum, volume_sum) = match itype {
        IntegrationType::Volumetric1D => {
            // inline-segment.mesh: [0,1], 4 segments (dx = .25), refined.
            let n = 4usize << ref_levels;
            let coords: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
            let geom = CutGeom::from_1d(&coords);
            let mut mf = MomentFitting::new_1d(&coords, order, ls_order, &lvl_fn);
            direct_sum(&mut mf, &geom, itype)
        }
        IntegrationType::Surface2D | IntegrationType::Volumetric2D => {
            let mesh = build_mesh_2d(ref_levels);
            let geom = CutGeom::from_mesh2(&mesh);
            let mut mf = MomentFitting::new_2d(&mesh, order, ls_order, &lvl_fn);
            let space = H1Space::new(mesh.clone(), 1);
            let ref_elem: Box<dyn ReferenceElement> =
                Box::new(fem_element::lagrange::QuadQk::new(1));
            let s = assemble_surface(&mut mf, &geom, ref_elem.as_ref(), &space, itype);
            let v = if has_volume(itype) {
                let ne = geom.elem_verts.len() as u32;
                let rules: Vec<CutRule> =
                    (0..ne).map(|e| mf.get_volume_integration_rule(e, None)).collect();
                assemble_volume(&geom, ref_elem.as_ref(), &space, itype, &rules)
            } else {
                0.0
            };
            (s, v)
        }
        IntegrationType::Surface3D | IntegrationType::Volumetric3D => {
            let mesh = build_mesh_3d(ref_levels);
            let geom = CutGeom::from_mesh3(&mesh);
            let mut mf = MomentFitting::new_3d(&mesh, order, ls_order, &lvl_fn);
            let space = H1Space::new(mesh.clone(), 1);
            let ref_elem: Box<dyn ReferenceElement> =
                Box::new(fem_element::lagrange::HexQk::new(1));
            let s = assemble_surface_3d(&mut mf, &geom, ref_elem.as_ref(), &space, itype);
            let v = if has_volume(itype) {
                let ne = geom.elem_verts.len() as u32;
                let rules: Vec<CutRule> =
                    (0..ne).map(|e| mf.get_volume_integration_rule(e, None)).collect();
                assemble_volume_3d(&geom, ref_elem.as_ref(), &space, itype, &rules)
            } else {
                0.0
            };
            (s, v)
        }
    };

    // ── Print information (ex38.cpp step 7) ────────────────────────────────
    let nbasis = 2 * (order + 1) + order * (order + 1) / 2;
    let mut qorder = 0usize;
    let mut npts = fem_assembly::cut::rule_npts(2, qorder);
    while npts <= nbasis {
        qorder += 1;
        npts = fem_assembly::cut::rule_npts(2, qorder);
    }
    let dim = match itype {
        IntegrationType::Volumetric1D => 1,
        IntegrationType::Surface2D | IntegrationType::Volumetric2D => 2,
        _ => 3,
    };
    println!("============================================");
    if dim != 1 {
        println!("Mesh size dx:                       {}", 3.2 / 2f64.powi(ref_levels));
    } else {
        println!("Mesh size dx:                       {}", 0.25 / 2f64.powi(ref_levels));
    }
    if dim == 2 {
        println!("Number of div free basis functions: {}", nbasis);
        println!("Number of quadrature points:        {}", npts);
    }
    println!("============================================");
    println!("Computed value of surface integral: {:.10e}", surface_sum);
    println!("True value of surface integral:     {:.10e}", surface_true(itype));
    println!(
        "Absolute Error (Surface):           {:.10e}",
        (surface_sum - surface_true(itype)).abs()
    );
    println!(
        "Relative Error (Surface):           {:.10e}",
        (surface_sum - surface_true(itype)).abs() / surface_true(itype)
    );
    if has_volume(itype) {
        println!("--------------------------------------------");
        println!("Computed value of volume integral:  {:.10e}", volume_sum);
        println!("True value of volume integral:      {:.10e}", volume_true(itype));
        println!(
            "Absolute Error (Volume):            {:.10e}",
            (volume_sum - volume_true(itype)).abs()
        );
        println!(
            "Relative Error (Volume):            {:.10e}",
            (volume_sum - volume_true(itype)).abs() / volume_true(itype)
        );
    }
    println!("============================================");
}
