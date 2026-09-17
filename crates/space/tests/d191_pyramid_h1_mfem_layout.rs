//! D191 — pyramid H1 field slots vs MFEM 4.10 ground truth.
//!
//! Ground truth: MFEM 4.10 `H1_FECollection(p, 3, GaussLobatto, pyr_type=0)`
//! (the Bergot pyramid — the family fem-rs' `PyramidPk` implements; MFEM's
//! *default* `pyr_type=1` Fuentes pyramid has 15/37/77 dofs at p=2/3/4 and is
//! a documented family gap, see `tmp/d191/EVIDENCE.md`).
//!
//! The probe `tmp/d191/pyr_h1_probe.cpp` (dumps
//! `$HOME/work/d299/probe_p{2,3,4,5}.txt`) projected the XYZ coefficient
//! through the space on a single straight unit pyramid
//! `(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1)` and printed, per
//! `GetElementDofs` slot, the entity classification and the position.  On the
//! straight unit pyramid the linear map is the identity, so the projected
//! positions ARE the reference positions of the slots.
//!
//! MFEM's reference lattice uses 1-D Gauss–Lobatto points (and barycentric
//! combinations of them on tri faces / interior), while fem-rs' `PyramidPk`
//! uses the equispaced collapsed lattice — the two agree in *arrangement*
//! (same slot → entity/lattice-index map, provably identical at p ≤ 2 where
//! the GLL points of degree ≤ 2 coincide with the equispaced ones) but not in
//! position for p ≥ 3.  The tests below therefore pin
//!
//! 1. p = 2: full position parity with the MFEM dump (both lattices coincide);
//! 2. p = 2..5: the slot→lattice-index arrangement against an independent
//!    reimplementation of the MFEM-encoded layout, and the slot positions
//!    against the linear-pyramid image `(i/p, j/p, k/p)`;
//! 3. the MFEM edge directions for blocks 2/3 (`(3,2)`, `(0,3)`);
//! 4. dof sharing (same global ids) on a 2-pyramid mesh sharing edges and a
//!    slanted face.

use fem_mesh::{ElementType, Mesh};
use fem_space::DofManager;

fn unit_pyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1.],
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Independent reimplementation of the MFEM slot table (must stay in sync
/// with the probe dumps, not with `DofManager`).
fn mfem_slot_grid(p: usize) -> Vec<[usize; 3]> {
    let mut s = Vec::new();
    s.push([0, 0, 0]);
    s.push([p, 0, 0]);
    s.push([p, p, 0]);
    s.push([0, p, 0]);
    s.push([0, 0, p]);
    if p >= 2 {
        let c = [[0, 0, 0], [p, 0, 0], [p, p, 0], [0, p, 0], [0, 0, p]];
        for [a, b] in [
            [0usize, 1], [1, 2], [3, 2], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4],
        ] {
            for q in 1..p {
                let pt = [0usize, 1, 2].map(|d| {
                    let e = (c[a][d] as isize) * (p - q) as isize + (c[b][d] as isize) * q as isize;
                    (e / p as isize) as usize
                });
                s.push(pt);
            }
        }
        for j in 1..p {
            for i in 1..p {
                s.push([i, j, 0]);
            }
        }
    }
    if p >= 3 {
        for k in 1..=p - 2 {
            for i in 1..=p - 1 - k {
                s.push([i, 0, k]);
            }
        }
        for k in 1..=p - 2 {
            for j in 1..=p - 1 - k {
                s.push([p - k, j, k]);
            }
        }
        for i in 1..=p - 2 {
            for k in 1..=p - 1 - i {
                s.push([i, p - k, k]);
            }
        }
        for j in 1..=p - 2 {
            for k in 1..=p - 1 - j {
                s.push([0, j, k]);
            }
        }
        for k in 1..=p - 2 {
            for j in 1..=p - 1 - k {
                for i in 1..=p - 1 - k {
                    s.push([i, j, k]);
                }
            }
        }
    }
    s
}

/// D191 pin 1: at p = 2 the fem-rs and MFEM lattices coincide pointwise, so
/// the slot positions must match the probe dump bit-for-bit-pattern
/// (`probe_p2.txt`, `== bergot p=2`, 14 slots).
#[test]
fn d191_p2_positions_match_mfem_probe() {
    let mesh = unit_pyramid();
    let dm = DofManager::new(&mesh, 2);
    assert_eq!(dm.n_dofs, 14);
    let expected: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0], // edge 0 (0,1)
        [1.0, 0.5, 0.0], // edge 1 (1,2)
        [0.5, 1.0, 0.0], // edge 2 (3,2)
        [0.0, 0.5, 0.0], // edge 3 (0,3)
        [0.0, 0.0, 0.5], // edge 4 (0,4)
        [0.5, 0.0, 0.5], // edge 5 (1,4)
        [0.5, 0.5, 0.5], // edge 6 (2,4)
        [0.0, 0.5, 0.5], // edge 7 (3,4)
        [0.5, 0.5, 0.0], // quad face 0
    ];
    let dofs = dm.element_dofs(0);
    for (s, g) in expected.iter().enumerate() {
        let d = dofs[s] as usize;
        for k in 0..3 {
            assert!(
                (dm.dof_coords[d * 3 + k] - g[k]).abs() < 1e-14,
                "p2 slot {s}: got {:?}, want {g:?}",
                &dm.dof_coords[d * 3..d * 3 + 3],
            );
        }
    }
}

/// D191 pin 2: for p = 2..5 the slot arrangement (which lattice index each
/// element-dof slot carries) matches the MFEM-encoded table, and each slot's
/// coordinate equals the linear-pyramid image of `(i/p, j/p, k/p)` — on the
/// unit pyramid exactly that point.
#[test]
fn d191_slot_arrangement_and_positions_p2_to_p5() {
    for p in 2..=5usize {
        let mesh = unit_pyramid();
        let dm = DofManager::new(&mesh, p as u8);
        let n_expected = (p + 1) * (p + 2) * (2 * p + 3) / 6;
        assert_eq!(dm.element_dofs(0).len(), n_expected, "p{p} dofs_per_elem");
        assert_eq!(dm.n_dofs, n_expected, "p{p} n_dofs");
        let grid = mfem_slot_grid(p);
        assert_eq!(grid.len(), n_expected);
        let dofs = dm.element_dofs(0);
        assert_eq!(dofs.len(), n_expected);
        // All dofs distinct on a single element.
        let mut seen = dofs.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), n_expected, "p{p}: duplicate dofs on element");
        for (s, g) in grid.iter().enumerate() {
            let d = dofs[s] as usize;
            for k in 0..3 {
                let want = g[k] as f64 / p as f64;
                assert!(
                    (dm.dof_coords[d * 3 + k] - want).abs() < 1e-13,
                    "p{p} slot {s} (grid {g:?}): got {}, want {want}",
                    dm.dof_coords[d * 3 + k],
                );
            }
        }
    }
}

/// D191 pin 3: MFEM's edge blocks 2/3 run `v3→v2` and `v0→v3` (probe: at
/// p = 4 slots 11,12,13 climb x = 0.173 → 0.5 → 0.827 along y = 1).
#[test]
fn d191_edge_block_directions_match_mfem() {
    for p in 3..=5usize {
        let mesh = unit_pyramid();
        let dm = DofManager::new(&mesh, p as u8);
        let dofs = dm.element_dofs(0);
        let edp = p - 1;
        // Block 2 occupies slots 5 + 2*edp .. +edp (edge (3,2), the base
        // edge at y = 1 running from v3=(0,1,0) to v2=(1,1,0)).
        let b2 = 5 + 2 * edp;
        for q in 0..edp - 1 {
            let x0 = dm.dof_coords[dofs[b2 + q] as usize * 3];
            let x1 = dm.dof_coords[dofs[b2 + q + 1] as usize * 3];
            assert!(
                x0 < x1,
                "p{p}: edge block 2 (3→2) must climb x: slot {q} {x0} !< {x1}",
            );
            let y0 = dm.dof_coords[dofs[b2 + q] as usize * 3 + 1];
            assert!((y0 - 1.0).abs() < 1e-13, "p{p}: edge block 2 y={y0}");
        }
        // Block 3 occupies slots 5 + 3*edp (edge (0,3): x = 0, y climbing).
        let b3 = 5 + 3 * edp;
        for q in 0..edp - 1 {
            let y0 = dm.dof_coords[dofs[b3 + q] as usize * 3 + 1];
            let y1 = dm.dof_coords[dofs[b3 + q + 1] as usize * 3 + 1];
            assert!(y0 < y1, "p{p}: edge block 3 (0→3) must climb y");
        }
    }
}

/// D191 pin 4: a second pyramid glued to the slanted face (1,2,4) of the
/// unit one must share that face's dofs and the shared edges' dofs (same
/// global ids) with the first element.
#[test]
fn d191_two_pyramids_share_face_and_edge_dofs() {
    // B is the mirror of A across the plane x+z=1: base (1,0,0),(1,0,1),
    // (1,1,1),(1,1,0), apex (0,0,1).  Shared: edges (1,2),(2,4),(1,4) and
    // the slanted face (1,2,4).
    let mesh = Mesh::<3>::uniform(
        vec![
            0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., // A
            1., 0., 1., 1., 1., 1., // B new verts 5, 6
        ],
        vec![0, 1, 2, 3, 4, 1, 5, 6, 2, 4],
        vec![1, 1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    );
    assert_eq!(mesh.n_nodes(), 7);
    for p in [2u8, 3, 4] {
        let dm = DofManager::new(&mesh, p);
        let edp = (p as usize).saturating_sub(1).max(1);
        let tdp = if p >= 3 { (p as usize - 1) * (p as usize - 2) / 2 } else { 0 };
        let qdp = if p >= 2 { (p as usize - 1) * (p as usize - 1) } else { 0 };
        let _ = (edp, tdp, qdp);
        let a = dm.element_dofs(0);
        let b = dm.element_dofs(1);
        // Shared edge (1,2): A block 1, B block 3 (pair (0,3) = (v1,v2)).
        let a_e = &a[5 + edp..5 + 2 * edp];
        let b_e = &b[5 + 3 * edp..5 + 4 * edp];
        let mut s1 = a_e.to_vec();
        s1.sort_unstable();
        let mut s2 = b_e.to_vec();
        s2.sort_unstable();
        assert_eq!(s1, s2, "p{p}: edge (1,2) dofs shared");
        // Shared edge (2,4): A block 6, B block 7 (pair (3,4) = (v2,v4)).
        let a_e = &a[5 + 6 * edp..5 + 7 * edp];
        let b_e = &b[5 + 7 * edp..5 + 8 * edp];
        let mut s1 = a_e.to_vec();
        s1.sort_unstable();
        let mut s2 = b_e.to_vec();
        s2.sort_unstable();
        assert_eq!(s1, s2, "p{p}: edge (2,4) dofs shared");
        // Shared face (1,2,4): A tri block 1 (faces (0,1,4),(1,2,4),(2,3,4),
        // (3,0,4)), B tri block 3 = local face (3,0,4) = (v2,v1,apex).
        let abase = 5 + 8 * edp + qdp;
        let a_f = &a[abase + tdp..abase + 2 * tdp];
        let b_f = &b[abase + 3 * tdp..abase + 4 * tdp];
        let mut s1 = a_f.to_vec();
        s1.sort_unstable();
        let mut s2 = b_f.to_vec();
        s2.sort_unstable();
        assert_eq!(s1, s2, "p{p}: face (1,2,4) dofs shared");
    }
}
