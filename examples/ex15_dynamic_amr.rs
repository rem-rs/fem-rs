//! ex15_dynamic_amr - minimal dynamic AMR loop with refine + derefine.
//!
//! Demonstrates the Phase-57 core cycle on Tri3:
//! 1) build synthetic nodal field
//! 2) estimate element errors (ZZ)
//! 3) mark+refine using refine_marked_with_tree
//! 4) prolongate P1 solution
//! 5) derefine selected parents and restrict back

use fem_core::NodeId;
use fem_mesh::{
    SimplexMesh,
    zz_estimator,
    dorfler_mark,
    mark_for_derefinement,
    refine_marked_with_tree,
    derefine_marked,
    prolongate_p1,
    restrict_to_coarse_p1,
};

fn main() {
    let args = parse_args();

    println!("=== ex15_dynamic_amr: refine + derefine demo ===");
    println!("  n={}, theta_refine={}, theta_derefine={}", args.n, args.theta_refine, args.theta_derefine);

    let mesh0 = SimplexMesh::<2>::unit_square_tri(args.n);
    let n0 = mesh0.n_nodes();
    let u0 = synthetic_field(&mesh0, 0.35, 0.45, 0.08);

    let eta0 = zz_estimator(&mesh0, &u0);
    let marked_refine = dorfler_mark(&eta0, args.theta_refine);
    println!("  coarse: elems={}, nodes={}, refine_marked={}", mesh0.n_elems(), mesh0.n_nodes(), marked_refine.len());

    let (mesh1, tree) = refine_marked_with_tree(&mesh0, &marked_refine);

    // Prolongate from coarse to refined mesh by midpoint interpolation.
    let u1_from_prolong = prolongate_p1(&u0, mesh1.n_nodes(), &tree.midpoint_map);

    // Simulate moved feature on refined mesh.
    let u1 = synthetic_field(&mesh1, 0.70, 0.55, 0.08);
    let eta1 = zz_estimator(&mesh1, &u1);
    let _low_error_children = mark_for_derefinement(&eta1, args.theta_derefine);

    // In this one-level API, derefine by parent ids recorded in tree.
    let parents = tree.parents();
    let mesh2 = derefine_marked(&mesh1, &tree, &parents);

    // Restrict fine solution back to coarse node set.
    let u2 = restrict_to_coarse_p1(&u1_from_prolong, n0);

    println!("  refined: elems={}, nodes={}", mesh1.n_elems(), mesh1.n_nodes());
    println!("  derefined: elems={}, nodes={}", mesh2.n_elems(), mesh2.n_nodes());
    println!("  restricted size={} (coarse nodes={})", u2.len(), n0);

    assert_eq!(mesh2.n_elems(), mesh0.n_elems(), "element count should roundtrip");
    assert_eq!(u2.len(), n0, "restriction size mismatch");

    println!("  PASS");
}

fn synthetic_field(mesh: &SimplexMesh<2>, cx: f64, cy: f64, sigma: f64) -> Vec<f64> {
    (0..mesh.n_nodes())
        .map(|i| {
            let [x, y] = mesh.coords_of(i as NodeId);
            let r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            (-r2 / (sigma * sigma)).exp()
        })
        .collect()
}

struct Args {
    n: usize,
    theta_refine: f64,
    theta_derefine: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 8,
        theta_refine: 0.5,
        theta_derefine: 0.2,
    };

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("8".into()).parse().unwrap_or(8),
            "--theta-refine" => a.theta_refine = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5),
            "--theta-derefine" => a.theta_derefine = it.next().unwrap_or("0.2".into()).parse().unwrap_or(0.2),
            _ => {}
        }
    }
    a
}
