//! mfem_ex15_dynamic_amr - minimal dynamic AMR loop with refine + derefine.
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

    println!("=== mfem_ex15_dynamic_amr: refine + derefine demo ===");
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

#[cfg(test)]
fn run_roundtrip(n: usize, theta_refine: f64, field: &[f64]) -> (SimplexMesh<2>, SimplexMesh<2>, SimplexMesh<2>, Vec<f64>) {
    let mesh0 = SimplexMesh::<2>::unit_square_tri(n);
    let eta0 = zz_estimator(&mesh0, field);
    let marked_refine = dorfler_mark(&eta0, theta_refine);
    let (mesh1, tree) = refine_marked_with_tree(&mesh0, &marked_refine);
    let u1 = prolongate_p1(field, mesh1.n_nodes(), &tree.midpoint_map);
    let parents = tree.parents();
    let mesh2 = derefine_marked(&mesh1, &tree, &parents);
    let u2 = restrict_to_coarse_p1(&u1, mesh0.n_nodes());
    (mesh0, mesh1, mesh2, u2)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex15_refine_marking_grows_with_theta() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let field = synthetic_field(&mesh, 0.35, 0.45, 0.08);
        let eta = zz_estimator(&mesh, &field);

        let marked_low = dorfler_mark(&eta, 0.2);
        let marked_mid = dorfler_mark(&eta, 0.5);
        let marked_high = dorfler_mark(&eta, 0.8);

        assert!(!marked_low.is_empty(), "expected non-empty refine set for localized feature");
        assert!(
            marked_low.len() <= marked_mid.len() && marked_mid.len() <= marked_high.len(),
            "expected Dörfler marking to be monotone in theta, got low={} mid={} high={}",
            marked_low.len(),
            marked_mid.len(),
            marked_high.len()
        );
    }

    #[test]
    fn ex15_constant_field_survives_roundtrip() {
        let mesh0 = SimplexMesh::<2>::unit_square_tri(8);
        let u0 = vec![1.0; mesh0.n_nodes()];
        let (mesh_coarse, mesh_refined, mesh_roundtrip, u2) = run_roundtrip(8, 0.5, &u0);

        assert!(mesh_refined.n_elems() >= mesh_coarse.n_elems());
        assert!(mesh_refined.n_nodes() >= mesh_coarse.n_nodes());
        assert_eq!(mesh_roundtrip.n_elems(), mesh_coarse.n_elems());
        assert_eq!(mesh_roundtrip.n_nodes(), mesh_coarse.n_nodes());

        let max_error = u0
            .iter()
            .zip(&u2)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_error < 1.0e-12, "constant field roundtrip error = {}", max_error);
    }

    #[test]
    fn ex15_zero_field_roundtrip_stays_zero() {
        let mesh0 = SimplexMesh::<2>::unit_square_tri(8);
        let u0 = vec![0.0; mesh0.n_nodes()];
        let (_, _, mesh_roundtrip, u2) = run_roundtrip(8, 0.5, &u0);

        assert_eq!(u2.len(), mesh0.n_nodes());
        assert_eq!(mesh_roundtrip.n_elems(), mesh0.n_elems());

        let max_value = u2.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        assert!(max_value < 1.0e-14, "zero field should remain zero after roundtrip, got max {}", max_value);
    }

    /// Refining a mesh should strictly increase the element count.
    #[test]
    fn ex15_mesh_refinement_increases_element_count() {
        let mesh0 = SimplexMesh::<2>::unit_square_tri(6);
        let field = synthetic_field(&mesh0, 0.5, 0.5, 0.15);
        let eta = zz_estimator(&mesh0, &field);
        let marked = dorfler_mark(&eta, 0.7);

        assert!(!marked.is_empty(), "expected non-empty refinement set");
        
        let (mesh1, _tree) = refine_marked_with_tree(&mesh0, &marked);
        assert!(mesh1.n_elems() > mesh0.n_elems(),
            "refined mesh should have more elements: {} vs {}",
            mesh1.n_elems(), mesh0.n_elems());
        assert!(mesh1.n_nodes() > mesh0.n_nodes(),
            "refined mesh should have more nodes: {} vs {}",
            mesh1.n_nodes(), mesh0.n_nodes());
    }

    /// Multiple refinement passes should further increase element count.
    #[test]
    fn ex15_multiple_refinement_passes_grow_mesh() {
        let mut mesh = SimplexMesh::<2>::unit_square_tri(6);
        let mut field = synthetic_field(&mesh, 0.5, 0.5, 0.15);
        let mut elem_counts = vec![mesh.n_elems()];

        for _ in 0..2 {
            let eta = zz_estimator(&mesh, &field);
            let marked = dorfler_mark(&eta, 0.6);
            if marked.is_empty() {
                break;
            }
            let (new_mesh, tree) = refine_marked_with_tree(&mesh, &marked);
            field = prolongate_p1(&field, new_mesh.n_nodes(), &tree.midpoint_map);
            mesh = new_mesh;
            elem_counts.push(mesh.n_elems());
        }

        for i in 1..elem_counts.len() {
            assert!(elem_counts[i] > elem_counts[i-1],
                "element count should increase monotonically: {} vs {}",
                elem_counts[i-1], elem_counts[i]);
        }
    }

    /// Error estimator should be positive and finite everywhere.
    #[test]
    fn ex15_error_estimator_is_positive_finite() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let field = synthetic_field(&mesh, 0.3, 0.7, 0.1);
        let eta = zz_estimator(&mesh, &field);

        assert_eq!(eta.len(), mesh.n_elems());
        for (i, &err) in eta.iter().enumerate() {
            assert!(err > 0.0, "error indicator at element {} should be > 0, got {}", i, err);
            assert!(err.is_finite(), "error indicator at element {} should be finite, got {}", i, err);
        }
    }

    /// Stronger Dörfler marking parameter should mark more elements.
    #[test]
    fn ex15_dorfler_marking_is_monotone_in_threshold() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let field = synthetic_field(&mesh, 0.4, 0.6, 0.12);
        let eta = zz_estimator(&mesh, &field);

        let marked_01 = dorfler_mark(&eta, 0.1);
        let marked_05 = dorfler_mark(&eta, 0.5);
        let marked_09 = dorfler_mark(&eta, 0.9);

        assert!(marked_01.len() <= marked_05.len(),
            "marking with θ=0.1 should give ≤ elements than θ=0.5: {} vs {}",
            marked_01.len(), marked_05.len());
        assert!(marked_05.len() <= marked_09.len(),
            "marking with θ=0.5 should give ≤ elements than θ=0.9: {} vs {}",
            marked_05.len(), marked_09.len());
    }

    /// After P1 prolongation the new nodal values must stay within the
    /// original field's min/max range (linear interpolation is bounded).
    #[test]
    fn ex15_prolongated_field_range_bounded_by_original() {
        let mesh = SimplexMesh::<2>::unit_square_tri(6);
        let field = synthetic_field(&mesh, 0.5, 0.5, 0.15);
        let eta = zz_estimator(&mesh, &field);
        let marked = dorfler_mark(&eta, 0.6);
        let (new_mesh, tree) = refine_marked_with_tree(&mesh, &marked);
        let field2 = prolongate_p1(&field, new_mesh.n_nodes(), &tree.midpoint_map);
        let fmin = field.iter().cloned().fold(f64::INFINITY, f64::min);
        let fmax = field.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let f2min = field2.iter().cloned().fold(f64::INFINITY, f64::min);
        let f2max = field2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(f2min >= fmin - 1.0e-12,
            "prolongated field min {} underflows original min {}", f2min, fmin);
        assert!(f2max <= fmax + 1.0e-12,
            "prolongated field max {} overflows original max {}", f2max, fmax);
    }
}

