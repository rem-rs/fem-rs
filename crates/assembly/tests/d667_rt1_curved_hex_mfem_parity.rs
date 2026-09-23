//! D667: multidomain numeric residual family — RT1-hex element-matrix parity
//! against MFEM 4.10 on a **curved** (`H1_3D_P2`) hexahedron extracted from the
//! refined `multidomain-hex.mesh` cylinder (the multidomain_rt/nd production
//! geometry).
//!
//! # What is compared
//!
//! The raw slot-order element matrices (36×36) of the three integrators the
//! multidomain_rt TDO assembles, on the single-hex fixtures
//! `data/d667_curved_hex{,_w2,_w6}.mesh` (real / 2× / 6× amplified P2 warp):
//!
//! - `VectorMassIntegrator`      (MFEM `VectorFEMassIntegrator`, Q = 1)
//! - `DivDivIntegrator`          (κ = −0.1, the miniapp's `−kappa` fold)
//! - `MixedWeakGradDotIntegrator` (v = (0.3, −1.7, 2.1))
//!
//! Each at MFEM's per-element *default* rule and at the explicit lower rules
//! the round-64 miniapp passes (`2p+2`, `2p−1`): MFEM's RT1 hex reports
//! `GetOrder() = 2` (not 1) and the curved-P2 transformation `OrderW() = 5`,
//! so the true MFEM defaults are mass `OrderW+2·order = 9`, divdiv
//! `2·order−2 = 2`, wgrad `order+order+OrderW = 9` — the miniapp's trilinear
//! assumptions (4 / 0 / 1) under-integrate every curved element.
//!
//! # Truth provenance
//!
//! `tmp/d667/d667_probe.cpp` (MFEM 4.10 serial tree, compiled
//! `g++ -std=c++17 -O2 -I$HOME/mfem410_ser d667_probe.cpp
//! $HOME/mfem410_ser/libmfem.a`), which also writes the three fixtures with
//! `Mesh::Print` after copying the picked parent hex's 27 P2 geometry dofs.
//! The fixture `data/d667_rt1_curved_hex_mfem.txt` carries the probe's raw
//! `AssembleElementMatrix` dumps (slot order, `dof_map` signs baked) plus the
//! reference/physical div at the rule centers.
//!
//! # Comparison convention (D615/D235 discipline)
//!
//! MFEM's fe_rt bakes the slot orientation signs into its shapes; fem-rs bakes
//! the same flips into `HexRTk` and applies the *space* interface signs on top,
//! so the fem-rs element matrix is unmangled as
//! `M_nosign[i][j] = signs[i]·signs[j]·M_local[i][j]` (single-element mesh ⇒
//! `element_dofs(0)[i] = i`) before the entry-wise comparison.

use fem_assembly::standard::{DivDivIntegrator, MixedWeakGradDotIntegrator, VectorMassIntegrator};
use fem_assembly::vector_assembler::accumulate_vector_bilinear_element;
use fem_assembly::postproc::coefficient::ConstantVectorCoeff;
use fem_assembly::VectorBilinearIntegrator;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CooMatrix;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::HDivSpace;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/d667_rt1_curved_hex_mfem.txt"
));

/// `(mesh name, integrator, rule tag) → 36×36 row-major matrix` from the probe dump.
fn parse_fixture() -> std::collections::HashMap<String, Vec<Vec<f64>>> {
    let mut out = std::collections::HashMap::new();
    let mut lines = FIXTURE.lines().peekable();
    while let Some(line) = lines.next() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.first() != Some(&"D667RT") {
            continue;
        }
        let mut mesh = "";
        let mut integ = "";
        let mut rule = "";
        let mut n = 0usize;
        for t in &tokens[1..] {
            if let Some(v) = t.strip_prefix("mesh=") {
                mesh = v;
            } else if let Some(v) = t.strip_prefix("INTEG=") {
                integ = v;
            } else if let Some(v) = t.strip_prefix("RULE=") {
                rule = v;
            } else if let Some(v) = t.strip_prefix("n=") {
                n = v.parse().unwrap();
            }
        }
        let mut mat = Vec::with_capacity(n * n);
        for _ in 0..n {
            let row: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
            assert_eq!(row.len(), n, "row width for {mesh}/{integ}/{rule}");
            for r in row {
                mat.push(r.parse::<f64>().unwrap());
            }
        }
        out.insert(format!("{mesh}|{integ}|{rule}"), {
            let mut rows = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    rows[i][j] = mat[i * n + j];
                }
            }
            rows
        });
    }
    out
}

/// The probe's `CalcDivShape` reference div at the `[0,1]³` center, per mesh.
fn parse_divref() -> std::collections::HashMap<String, Vec<f64>> {
    let mut out = std::collections::HashMap::new();
    let mut lines = FIXTURE.lines();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("PROBE divref_center") {
            let mesh = rest.trim_start_matches("PROBE divref_center");
            // the mesh name is on the "PROBE mesh=..." line above; simpler:
            // divref lines appear directly under their mesh's block header.
            let _ = mesh;
            let vals: Vec<f64> = lines.next().unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
            // mesh name from the preceding PROBE mesh= line — re-scan
            out.insert(vals.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"), vals);
        }
    }
    out
}

fn load_mesh(name: &str) -> Mesh<3> {
    let path = format!("{}/../../data/{}", env!("CARGO_MANIFEST_DIR"), name);
    let parent = read_mfem_file(&path).expect("read fixture mesh");
    parent.mesh3d.expect("3D fixture")
}

/// fem-rs element matrix, unmangled per D615 (`signs[i]·signs[j]·M[i][j]`).
fn femrs_element_matrix(
    space: &HDivSpace<Mesh<3>>,
    integ: &dyn VectorBilinearIntegrator,
    quad_order: u8,
) -> Vec<Vec<f64>> {
    let n = space.element_dofs(0).len();
    let mut coo = CooMatrix::new(space.n_dofs(), space.n_dofs());
    accumulate_vector_bilinear_element(space, 0, &[integ], quad_order, &mut coo);
    let csr = coo.into_csr();
    let signs = space.element_signs(0).to_vec();
    let mut mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let v = csr.get(i, j);
            mat[i][j] = signs[i] * signs[j] * v;
        }
    }
    mat
}

fn max_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn rel_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flatten().map(|x| x * x).sum::<f64>().sqrt()
}

/// The (integrator, rule) grid: tag → fem-rs quad order. `default` mirrors the
/// MFEM per-element defaults (mass 9, divdiv 2, wgrad 9 on the curved hex).
const MESHES: [&str; 3] =
    ["d667_curved_hex.mesh", "d667_curved_hex_w2.mesh", "d667_curved_hex_w6.mesh"];

fn run_case(mesh_name: &str, integ_name: &str, rule: &str, quad_order: u8, truth: &Vec<Vec<f64>>) {
    let mesh = load_mesh(mesh_name);
    assert_eq!(mesh.geom_order(), 2, "fixture must carry P2 geometry");
    let space = HDivSpace::new(mesh, 1);
    let mat: Vec<Vec<f64>> = match integ_name {
        "mass" => femrs_element_matrix(&space, &VectorMassIntegrator { alpha: 1.0 }, quad_order),
        "divdiv" => {
            femrs_element_matrix(&space, &DivDivIntegrator { kappa: -0.1 }, quad_order)
        }
        "wgrad" | "wgradneg" => femrs_element_matrix(
            &space,
            &MixedWeakGradDotIntegrator { velocity: ConstantVectorCoeff(vec![0.3, -1.7, 2.1]) },
            quad_order,
        ),
        other => panic!("unknown integrator {other}"),
    };
    let d = max_diff(&mat, truth);
    let rel = d / rel_norm(truth);
    if d > 1e-11 * rel_norm(truth).max(1e-30) {
        // report the top-8 diverging entries with their slot indices
        let mut entries: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..mat.len() {
            for j in 0..mat.len() {
                entries.push(((mat[i][j] - truth[i][j]).abs(), i, j));
            }
        }
        entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for &(d, i, j) in entries.iter().take(8) {
            println!(
                "  [{i:2}][{j:2}]: fem-rs = {:.17e}, mfem = {:.17e}, diff = {d:.6e}",
                mat[i][j], truth[i][j]
            );
        }
    }
    assert!(
        d <= 1e-11 * rel_norm(truth).max(1e-30),
        "RED {mesh_name} {integ_name} {rule} (quad {quad_order}): max|diff| = {d:.6e} (rel {rel:.3e})"
    );
}

/// The probe's `CalcVShape` reference/physical V-shapes at the rule center.
fn parse_vshape(tag: &str) -> Vec<[f64; 3]> {
    let mut lines = FIXTURE.lines();
    parse_vshape_from(&mut lines, tag, "")
}

fn parse_vshape_from(
    lines: &mut std::str::Lines<'_>,
    tag: &str,
    want: &str,
) -> Vec<[f64; 3]> {
    while let Some(line) = lines.next() {
        if line.starts_with(tag) {
            if !want.is_empty() {
                assert_eq!(line.trim(), want, "quadrature point identity for {tag}");
            }
            let mut out = Vec::with_capacity(36);
            for _ in 0..36 {
                let row: Vec<f64> =
                    lines.next().unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
                out.push([row[0], row[1], row[2]]);
            }
            return out;
        }
    }
    panic!("fixture section {tag} not found");
}

#[test]
fn d667_geometry_diagnostics() {
    let mesh = load_mesh("d667_curved_hex.mesh");
    println!("geom_order = {}", mesh.geom_order());
    let space = HDivSpace::new(mesh.clone(), 1);
    let dofs: Vec<u32> = space.element_dofs(0).to_vec();
    assert_eq!(dofs, (0..36).collect::<Vec<_>>(), "single-element slot map");
    assert!(space.element_signs(0).iter().all(|&s| s == 1.0), "signs all +1");

    // reference basis at the [-1,1] center vs MFEM CalcVShape at [0,1] center
    use fem_element::reference::VectorReferenceElement;
    use fem_element::raviart_thomas::HexRTk;
    let elem = HexRTk::new_gauss_legendre(1);
    let mut phi = vec![0.0_f64; 36 * 3];
    elem.eval_basis_vec(&[0.0, 0.0, 0.0], &mut phi);
    let vref = parse_vshape("PROBE vref_center");
    let mut bad = 0usize;
    for i in 0..36 {
        for c in 0..3 {
            let got = phi[i * 3 + c] * 4.0;
            if ((got - vref[i][c]) as f64).abs() > 1e-14 * (vref[i][c].abs().max(1e-12)) {
                if bad < 12 {
                    println!(
                        "VREF mismatch slot {i} comp {c}: fem-rs*4 = {got:.17e}, mfem = {:.17e}",
                        vref[i][c]
                    );
                }
                bad += 1;
            }
        }
    }
    println!("vref mismatches: {bad}/108");
    assert_eq!(bad, 0, "reference V-shape mismatch vs MFEM");

    // reference V-shapes at the 8 order-2 CUBE rule points (the [0,1] rule
    // points map to [-1,1] via xi -> 2*xi - 1; matched by VALUE since the
    // rule point orderings differ between the two codes)
    let mut mfem_q: Vec<([f64; 3], Vec<[f64; 3]>)> = Vec::new();
    {
        let mut lines = FIXTURE.lines();
        while let Some(line) = lines.next() {
            if line.starts_with("PROBE vref_q") && line.contains("ip=") {
                let vals: Vec<f64> = line[line.find("ip=").unwrap() + 3..]
                    .split(',')
                    .map(|v| v.parse().unwrap())
                    .collect();
                let mut shapes = Vec::with_capacity(36);
                for _ in 0..36 {
                    let row: Vec<f64> =
                        lines.next().unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
                    shapes.push([row[0], row[1], row[2]]);
                }
                mfem_q.push(([vals[0], vals[1], vals[2]], shapes));
                if mfem_q.len() == 8 {
                    break; // first mesh block only
                }
            }
        }
    }
    assert_eq!(mfem_q.len(), 8);
    let q2 = fem_element::quadrature::hex_rule(2);
    let mut mismatch_pts = Vec::new();
    for xi_rs in q2.points.iter() {
        let ip = [
            (xi_rs[0] + 1.0) / 2.0,
            (xi_rs[1] + 1.0) / 2.0,
            (xi_rs[2] + 1.0) / 2.0,
        ];
        let (_, vq) = mfem_q
            .iter()
            .find(|(p, _)| (p[0] - ip[0]).abs() < 1e-13 && (p[1] - ip[1]).abs() < 1e-13 && (p[2] - ip[2]).abs() < 1e-13)
            .unwrap_or_else(|| panic!("no MFEM q block at {ip:?}"));
        elem.eval_basis_vec(xi_rs, &mut phi);
        let mut badq = 0usize;
        for i in 0..36 {
            for c in 0..3 {
                let got = phi[i * 3 + c] * 4.0;
                if (got - vq[i][c]).abs() > 1e-13 * (vq[i][c].abs().max(1e-12)) {
                    if badq < 2 {
                        println!(
                            "qpoint {ip:?} slot {i} comp {c}: fem-rs*4 = {got:.17e}, mfem = {:.17e}",
                            vq[i][c]
                        );
                    }
                    badq += 1;
                }
            }
        }
        if badq > 0 {
            mismatch_pts.push((ip, badq));
        }
    }
    println!("vref@q2 mismatching points: {}", mismatch_pts.len());
    for (ip, badq) in &mismatch_pts {
        println!("  point {ip:?}: {badq} mismatching slots");
    }

    // physical shapes at the center through the P2 geometry (re-evaluate the
    // reference basis at the center first — the q2 loop overwrote `phi`)
    elem.eval_basis_vec(&[0.0, 0.0, 0.0], &mut phi);
    use fem_element::lagrange::factory::{ref_elem, ElemType};
    let geo27 = ref_elem(ElemType::Hex, 2);
    let (j, det, _xp) = fem_assembly::isoparametric_jacobian(
        &mesh,
        mesh.geometry_nodes(0),
        geo27.as_ref(),
        &[0.0, 0.0, 0.0],
        3,
    );
    let vphys = parse_vshape("PROBE vphys_center");
    let mut bad = 0usize;
    for i in 0..36 {
        for c in 0..3 {
            let v = phi[i * 3] * 4.0;
            let _ = v;
            let got = (j[(c, 0)] * phi[i * 3] + j[(c, 1)] * phi[i * 3 + 1]
                + j[(c, 2)] * phi[i * 3 + 2])
                / det;
            if got.abs() > 1e-14 * (vphys[i][c].abs().max(1e-12))
                && ((got - vphys[i][c]) as f64).abs() > 1e-12 * (vphys[i][c].abs().max(1e-12))
            {
                if bad < 12 {
                    println!(
                        "VPHY mismatch slot {i} comp {c}: fem-rs*4 = {got:.17e}, mfem = {:.17e}",
                        vphys[i][c]
                    );
                }
                bad += 1;
            }
        }
    }
    println!("vphys mismatches: {bad}/108");
}

/// The `-nr 1` validation mesh (MFEM-refined parent, 480 hexes) must keep its
/// P2 curvature through the reader, and fem-rs's per-element geometry must
/// reproduce MFEM's own Weight survey on the same file exactly: no folded
/// hexes at the order-9 points and the identical minimum Weight
/// (`det_femrs · 8 = det_mfem` for the two reference frames).
#[test]
fn d667_refined_mesh_curvature_and_folds() {
    use fem_element::lagrange::factory::{ref_elem, ElemType};
    let mesh = load_mesh("d667_refined_curved.mesh");
    assert_eq!(mesh.geom_order(), 2, "refined mesh must carry P2 geometry");
    let geo = ref_elem(ElemType::Hex, 2);
    // MFEM's own Weight survey samples the order-9 rule (the mass default on
    // the curved hex); the fold lives in the corner-adjacent shell, so the
    // coarse rules (center / order 4) see det > 0 everywhere.
    let q9 = fem_element::quadrature::hex_rule(9);
    let mut folded = 0u32;
    let mut det_min = f64::MAX;
    for e in 0..mesh.n_elems() as u32 {
        let nodes = mesh.geometry_nodes(e);
        let mut elem_det_min = f64::MAX;
        for xi in &q9.points {
            let (_j, det, _xp) =
                fem_assembly::isoparametric_jacobian(&mesh, nodes, geo.as_ref(), xi, 3);
            elem_det_min = elem_det_min.min(det);
        }
        det_min = det_min.min(elem_det_min);
        if elem_det_min < 0.0 {
            folded += 1;
        }
    }
    println!("refined mesh: folded hexes (order-9 survey) = {folded}, det_min = {det_min:.6e}");
    // MFEM survey (probe tmp/d667/d667_refined_survey.cpp, MFEM 4.10):
    //   REFINED SURVEY: folded=0 gmin=0.00040277375351502911 NE=480
    // The weight scales by 8 between the [-1,1] and [0,1] frames, so
    // fem-rs det_min * 8 must reproduce MFEM's gmin exactly.
    assert_eq!(folded, 0, "MFEM agrees: no folded hexes in the refined mesh");
    let gmin_mfem = 4.0277375351502911e-4_f64;
    let gmin_femrs_scaled = det_min * 8.0;
    println!("gmin: fem-rs*8 = {gmin_femrs_scaled:.17e}, mfem = {gmin_mfem:.17e}");
    assert!(
        (gmin_femrs_scaled - gmin_mfem).abs() <= 1e-10 * gmin_mfem,
        "refined-mesh minimum Weight mismatch vs MFEM survey"
    );
}

#[test]
fn d667_rt1_curved_hex_parity() {
    let truth = parse_fixture();

    // ── mass ──
    // MFEM default on the curved P2 hex: OrderW + 2·GetOrder = 5 + 4 = 9.
    for m in MESHES {
        let t = truth
            .get(&format!("{m}|mass|default"))
            .unwrap_or_else(|| panic!("no truth for {m} mass default"));
        run_case(m, "mass", "default", 9, t);
    }
    // miniapp order 2p+2 = 4.
    for m in MESHES {
        let t = truth.get(&format!("{m}|mass|o4")).unwrap();
        run_case(m, "mass", "o4", 4, t);
    }

    // ── divdiv ──
    // MFEM default: 2·GetOrder − 2 = 2 (an 8-point rule — NOT the round-64
    // assumed 1-point order-0 rule).
    for m in MESHES {
        let t = truth.get(&format!("{m}|divdiv|default")).unwrap();
        run_case(m, "divdiv", "default", 2, t);
    }

    // ── wgrad ──
    // MFEM default: order + order + OrderW = 2 + 2 + 5 = 9.  The raw-MFEM
    // block (internal `shape *= -1` test negation) must equal the fem-rs
    // integrator with the NEGATED coefficient — asserted via `wgradneg`
    // (the C++ production fold `-alpha·q`), which by construction is
    // `-1 × wgrad` here.
    for m in MESHES {
        let t = truth.get(&format!("{m}|wgradneg|default")).unwrap();
        run_case(m, "wgradneg", "default", 9, t);
    }
    // miniapp order 2p−1 = 1 and the shared -qp override order 4.
    for m in MESHES {
        let t = truth.get(&format!("{m}|wgradneg|o1")).unwrap();
        run_case(m, "wgradneg", "o1", 1, t);
        let t = truth.get(&format!("{m}|wgradneg|o4")).unwrap();
        run_case(m, "wgradneg", "o4", 4, t);
    }
}

/// The reference div of the GaussLegendre RT1 hex at the rule center: fem-rs
/// `HexRTk::new_gauss_legendre(1).eval_div` on the `[-1,1]` hex equals MFEM's
/// `RT_HexahedronElement::CalcDivShape` at `[0,1]` center divided by 8 (the
/// `[0,1] → [-1,1]` frame factor: `dc/2 · o/2 · o/2`).
#[test]
fn d667_rt1_hex_divref_center_matches_mfem() {
    use fem_element::reference::VectorReferenceElement;
    use fem_element::raviart_thomas::HexRTk;

    let truth = parse_divref();
    assert!(!truth.is_empty(), "no divref truth parsed");
    let mfem: Vec<f64> = truth.keys().next().unwrap().split('|').map(|v| v.parse().unwrap()).collect();
    assert_eq!(mfem.len(), 36);

    let elem = HexRTk::new_gauss_legendre(1);
    let mut div = vec![0.0_f64; 36];
    elem.eval_div(&[0.0, 0.0, 0.0], &mut div);
    for i in 0..36 {
        let got = div[i] * 8.0;
        assert!(
            (got - mfem[i]).abs() <= 1e-14 * (mfem[i].abs().max(1e-12)),
            "divref[{i}]: fem-rs*8 = {got:.17e}, mfem = {:.17e}",
            mfem[i]
        );
    }
}
