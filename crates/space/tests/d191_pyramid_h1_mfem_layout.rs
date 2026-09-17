//! D191 — pyramid H1 field slots vs MFEM 4.10 ground truth.
//!
//! Ground truth: MFEM 4.10 `H1_FECollection(p, 3, GaussLobatto, pyr_type=0)`
//! (the Bergot pyramid — the family fem-rs numbers pyramid H¹ spaces with;
//! MFEM's *default* `pyr_type=1` Fuentes pyramid has 15/37/77 dofs at p=2/3/4
//! and is a documented family gap, see `tmp/d191/EVIDENCE.md`).
//!
//! The probe `tmp/d191/pyr_h1_probe.cpp` (dumps
//! `$HOME/work/d299/probe_p{2,3,4,5}.txt`) projected the XYZ coefficient
//! through the space on a single straight unit pyramid
//! `(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1)` and printed, per
//! `GetElementDofs` slot, the entity classification and the position.  On the
//! straight unit pyramid the linear map is the identity, so the projected
//! positions ARE the reference positions of the slots.
//!
//! MFEM's reference lattice uses 1-D Gauss–Lobatto points (barycentric
//! combinations of them on tri faces / interior); D299 moved the slot
//! *coordinates* from the equispaced collapsed lattice `(i/p, j/p, k/p)` to
//! those GLL-barycentric positions (`H1PyramidPk::dof_coords`, the element
//! the assembler now pairs with these slots).  The tests below pin
//!
//! 1. p = 2: full position parity with the MFEM dump (the GLL points of
//!    degree 2 coincide with the equispaced ones);
//! 2. p = 2..5: the slot→lattice-index arrangement against an independent
//!    reimplementation of the MFEM-encoded layout, and the slot positions
//!    against an independent implementation of MFEM's GLL-barycentric node
//!    placement (exact for every p — the equispaced `(i/p, j/p, k/p)` at
//!    p = 2 is a special case);
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

/// Closed Gauss–Lobatto points on `[0,1]`: endpoints plus the roots of
/// `P'_p` (Newton on the Legendre recurrence, independent of `fem-element`).
fn gll_closed(p: usize) -> Vec<f64> {
    const PIF: f64 = std::f64::consts::PI;
    // Legendre P_n(x) and P'_n(x) on [-1,1] (three-term recurrence, and
    // `(x²−1)P'_n = n(x P_n − P_{n−1})` for the derivative).
    fn pleg(n: usize, x: f64) -> (f64, f64) {
        if n == 0 {
            return (1.0, 0.0);
        }
        if n == 1 {
            return (x, 1.0);
        }
        let (mut p0, mut p1) = (1.0_f64, x);
        for k in 1..n {
            let t = ((2 * k + 1) as f64 * x * p1 - k as f64 * p0) / (k + 1) as f64;
            p0 = p1;
            p1 = t;
        }
        let d = n as f64 * (x * p1 - p0) / (x * x - 1.0);
        (p1, d)
    }
    let mut pts = vec![0.0; p + 1];
    pts[p] = 1.0;
    let pf = p as f64;
    for i in 1..p {
        // Chebyshev-like interior guess; Newton on P'_p (second derivative
        // via `P''_n = (2x P'_n − n(n+1) P_n)/(x²−1)`).
        let mut x = (PIF * (i as f64 + 0.5) / (pf + 0.5)).cos();
        let (mut f, mut df) = {
            let (pn, dpn) = pleg(p, x);
            (dpn, (2.0 * x * dpn - pf * (pf + 1.0) * pn) / (1.0 - x * x))
        };
        for _ in 0..100 {
            let step = f / df;
            x -= step;
            if step.abs() < 1e-16 {
                break;
            }
            let (pn, dpn) = pleg(p, x);
            f = dpn;
            df = (2.0 * x * dpn - pf * (pf + 1.0) * pn) / (1.0 - x * x);
        }
        pts[i] = 0.5 * (x + 1.0);
    }
    // The interior roots were found right-to-left; MFEM's `cp` increases.
    pts[1..p].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    pts
}

/// Independent implementation of MFEM `H1_BergotPyramidElement`'s
/// GLL-barycentric node placement for a collapsed-lattice label
/// (`fe/fe_h1.cpp` constructor — mirrors the probe's XYZ projection on the
/// straight unit pyramid; must stay in sync with the probe dumps, not with
/// `H1PyramidPk`).
fn mfem_slot_position(p: usize, l: [usize; 3]) -> [f64; 3] {
    let cp = gll_closed(p);
    let [i, j, k] = l;
    if k == 0 {
        return [cp[i], cp[j], cp[0]];
    }
    if (i, j, k) == (0, 0, p) {
        return [cp[0], cp[0], cp[p]];
    }
    if i == 0 && j == 0 {
        return [cp[0], cp[0], cp[k]];
    }
    if j == 0 && i + k == p {
        return [cp[i], cp[0], cp[k]];
    }
    if i == j && i + k == p {
        return [cp[i], cp[j], cp[k]];
    }
    if i == 0 && j + k == p {
        return [cp[0], cp[j], cp[k]];
    }
    if j == 0 {
        let w = cp[i] + cp[k] + cp[p - i - k];
        return [cp[i] / w, cp[0], cp[k] / w];
    }
    if i == p - k {
        let w = cp[j] + cp[k] + cp[p - j - k];
        return [1.0 - cp[k] / w, cp[j] / w, cp[k] / w];
    }
    if j == p - k {
        let w = cp[i] + cp[k] + cp[p - i - k];
        return [cp[i] / w, 1.0 - cp[k] / w, cp[k] / w];
    }
    if i == 0 {
        let w = cp[j] + cp[k] + cp[p - j - k];
        return [cp[0], cp[j] / w, cp[k] / w];
    }
    let wjk = cp[j] + cp[k] + cp[p - j - k];
    let wik = cp[i] + cp[k] + cp[p - i - k];
    let w = wik * wjk * cp[p - k];
    [
        cp[i] * (cp[j] + cp[p - j - k]) / w,
        cp[j] * (cp[i] + cp[p - i - k]) / w,
        cp[k] * cp[p - k] / w,
    ]
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

/// D191 pin 2 (D299-updated): for p = 2..5 the slot arrangement (which
/// lattice index each element-dof slot carries) matches the MFEM-encoded
/// table, and each slot's coordinate equals MFEM's GLL-barycentric reference
/// position for that label — on the unit pyramid exactly that point.
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
            let want = mfem_slot_position(p, *g);
            for k in 0..3 {
                assert!(
                    (dm.dof_coords[d * 3 + k] - want[k]).abs() < 1e-13,
                    "p{p} slot {s} (grid {g:?}): got {}, want {}",
                    dm.dof_coords[d * 3 + k],
                    want[k],
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
