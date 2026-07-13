//! Fuzz test: random refine/derefine sequences on AMR meshes.
//!
//! Verifies that 1000+ steps of random non-conforming refinement and
//! coarsening complete without panic or invalid mesh state.

use fem_mesh::{Mesh, amr::NCStateQuad};
use fem_core::{ElemId, NodeId};

// ─── Deterministic pseudo-random generator (LCG) ──────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self { Self(seed) }

    /// Returns a pseudo-random u64 in [0, 2^64).
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005)
                       .wrapping_add(1442695040888963407);
        self.0
    }

    /// Returns a pseudo-random index in [0, max).
    fn next_idx(&mut self, max: usize) -> usize {
        if max == 0 { return 0; }
        (self.next_u64() % max as u64) as usize
    }

    /// Returns a pseudo-random bool with given probability (0.0 – 1.0).
    fn next_bool(&mut self, p: f64) -> bool {
        (self.next_u64() as f64 / u64::MAX as f64) < p
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Fuzz test: 1000 random refine/derefine steps on Quad4 mesh
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn fuzz_refine_derefine_1000_steps_quad4() {
    let mut nc = NCStateQuad::new();
    let mut mesh = Mesh::<2>::unit_square_quad(4);  // 16 elements
    let max_elems = 500;  // keep mesh size manageable
    let mut rng = Lcg::new(42);
    let mut total_steps = 0u32;

    for iteration in 0..1000 {
        let n_elems = mesh.n_elems();
        if n_elems == 0 { break; }

        // 70 % refine, 30 % derefine (when history exists)
        let do_refine = if nc.can_derefine() {
            rng.next_bool(0.70)
        } else {
            true  // must refine if no history for derefine
        };

        if do_refine && n_elems < max_elems {
            // Refine 1-3 random elements (distinct)
            let n_mark = (rng.next_u64() % 3) as usize + 1;
            let mut marked_set = std::collections::HashSet::new();
            for _ in 0..n_mark.saturating_mul(2) {
                // Try up to 2×n_mark times to get distinct elements
                marked_set.insert(rng.next_idx(n_elems) as ElemId);
                if marked_set.len() >= n_mark { break; }
            }
            let marked: Vec<ElemId> = marked_set.into_iter().collect();
            if !marked.is_empty() {
                let (new_mesh, _constraints, _midpoint_map) = nc.refine(&mesh, &marked);
                mesh = new_mesh;
                total_steps += 1;
            }
        } else if nc.can_derefine() {
            if let Some((m, _)) = nc.derefine_last() {
                mesh = m;
                total_steps += 1;
            }
        }

        // ── Invariants ─────────────────────────────────────────────────
        assert!(
            mesh.n_elems() > 0,
            "iteration {iteration}: mesh lost all elements"
        );
        for n in 0..mesh.n_nodes() as NodeId {
            let c = mesh.coords_of(n);
            assert!(
                c[0].is_finite() && c[1].is_finite(),
                "iteration {iteration}: node {n} has non-finite coords ({}, {})",
                c[0], c[1],
            );
        }

        // Periodically reset to keep test fast
        if iteration % 200 == 199 && iteration < 999 {
            mesh = Mesh::<2>::unit_square_quad(4);
            nc = NCStateQuad::new();
        }
    }

    println!("  fuzz test: {total_steps} refine/derefine steps completed");
    assert!(total_steps > 100, "fuzz test should complete >100 steps, got {total_steps}");
}

// ═════════════════════════════════════════════════════════════════════════════
//  Fuzz test: 500 random refine/derefine steps on Tri3 mesh
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn fuzz_refine_derefine_500_steps_tri3() {
    use fem_mesh::amr::NCState;  // 2-D Tri3 version

    let mut nc = NCState::new();
    let mut mesh = Mesh::<2>::unit_square_tri(4);  // 32 elements
    let max_elems = 500;
    let mut rng = Lcg::new(123);
    let mut total_steps = 0u32;

    for iteration in 0..500 {
        let n_elems = mesh.n_elems();
        if n_elems == 0 { break; }

        let do_refine = if nc.can_derefine() {
            rng.next_bool(0.70)
        } else {
            true
        };

        if do_refine && n_elems < max_elems {
            let n_mark = (rng.next_u64() % 3) as usize + 1;
            let mut marked_set = std::collections::HashSet::new();
            for _ in 0..n_mark.saturating_mul(2) {
                marked_set.insert(rng.next_idx(n_elems) as ElemId);
                if marked_set.len() >= n_mark { break; }
            }
            let marked: Vec<ElemId> = marked_set.into_iter().collect();
            if !marked.is_empty() {
                let (new_mesh, _constraints, _midpoint_map) = nc.refine(&mesh, &marked);
                mesh = new_mesh;
                total_steps += 1;
            }
        } else if nc.can_derefine() {
            if let Some((m, _)) = nc.derefine_last() {
                mesh = m;
                total_steps += 1;
            }
        }

        // Invariants
        assert!(mesh.n_elems() > 0, "iteration {iteration}: mesh lost all elements");
        for n in 0..mesh.n_nodes() as NodeId {
            let c = mesh.coords_of(n);
            assert!(
                c[0].is_finite() && c[1].is_finite(),
                "iteration {iteration}: node {n} has non-finite coords ({}, {})",
                c[0], c[1],
            );
        }

        if iteration % 200 == 199 && iteration < 499 {
            mesh = Mesh::<2>::unit_square_tri(4);
            nc = NCState::new();
        }
    }

    println!("  Tri3 fuzz: {total_steps} refine/derefine steps completed");
    assert!(total_steps > 50, "Tri3 fuzz should complete >50 steps, got {total_steps}");
}
