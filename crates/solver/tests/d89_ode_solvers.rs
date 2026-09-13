//! D89-B: the MFEM `ODESolver` family (`crates/solver/src/ode/mfem_ode.rs`)
//! against per-step values dumped from MFEM 4.10 itself.
//!
//! The model is a stiff, non-diagonal 2×2 system — `du/dt = A u` with
//! `A = [[-1000, 1], [0, -0.5]]` — integrated from `u = (1, 2)` with `dt = 0.1`
//! for four steps:
//!
//! ```text
//!   BackwardEuler step 0 t 0.10000000000000001 u 0.011786892975011787 1.9047619047619047
//!   ...
//! ```
//!
//! The dump was produced by `tmp/d89_probe.cpp` (compiled against
//! `$HOME/mfem410/libmfem.a`, `g++ -std=c++17 -O2`), which drives MFEM's own
//! `BackwardEulerSolver`, `SDIRK23Solver(1)`, `SDIRK23Solver(3)`,
//! `SDIRK33Solver`, `SDIRK34Solver`, `ImplicitMidpointSolver` and
//! `SIAVSolver(1..4)` over that operator, printing every step with
//! `setprecision(17)`.  The values below are transcribed verbatim.
//!
//! Agreement is asserted at `≤ 1e-13` relative — the C++ side inverts the 2×2
//! stage matrix (`DenseMatrix::Invert`), the Rust side eliminates it, so the
//! last two bits of the mantissa are allowed to differ.

use fem_solver::{
    BackwardEulerSolver, DenseOperator, ImplicitMidpointSolver, LinearOp, OdeSolver, Sdirk23Gamma,
    Sdirk23Solver, Sdirk33Solver, Sdirk34Solver, SiavSolver, TimeDependentOperator,
};

/// `A = [[-1000, 1], [0, -0.5]]`.
fn stiff_operator() -> DenseOperator {
    DenseOperator::from_rows(&[&[-1000.0, 1.0], &[0.0, -0.5]])
}

const DT: f64 = 0.1;
const NSTEPS: usize = 4;
const U0: [f64; 2] = [1.0, 2.0];

/// MFEM 4.10, `BackwardEulerSolver` — `u0 u1` per step.
const BE: [[f64; 2]; 4] = [
    [0.011786892975011787, 1.9047619047619047],
    [0.0019127998877883357, 1.8140589569160996],
    [0.0017295081147929063, 1.727675197062952],
    [0.001646237654189795, 1.6454049495837639],
];

/// MFEM 4.10, `SDIRK23Solver(1)` (A-stable, 3rd order).
const SDIRK23: [[f64; 2]; 4] = [
    [-0.70131275416871985, 1.9024578366628508],
    [0.49731505581004015, 1.8096729101399471],
    [-0.34742312362063688, 1.7214132048461048],
    [0.24765524456663107, 1.6374580208471927],
];

/// MFEM 4.10, `SDIRK23Solver(3)` (L-stable, 2nd order, `gamma > 1`).
const SDIRK23_G3: [[f64; 2]; 4] = [
    [0.010109217615775878, 1.9027407162753474],
    [0.0018785823152022811, 1.8102111166860109],
    [0.0017235974207038534, 1.7221811983863682],
    [0.001639256330110138, 1.6384321434868072],
];

/// MFEM 4.10, `SDIRK33Solver`.
const SDIRK33: [[f64; 2]; 4] = [
    [-0.024498175674090295, 1.9024585499058884],
    [0.0025090208778390591, 1.8096742670550077],
    [0.0017037993481903244, 1.7214151409517358],
    [0.0016387684145933146, 1.63746047642054],
];

/// MFEM 4.10, `SDIRK34Solver`.
const SDIRK34: [[f64; 2]; 4] = [
    [-0.6040105591751076, 1.9024587633724985],
    [0.36967842230911202, 1.8096746731664082],
    [-0.22162089760510212, 1.7214157204093476],
    [0.13723636161223646, 1.6374612113499729],
];

/// MFEM 4.10, `ImplicitMidpointSolver`.
const MIDPOINT: [[f64; 2]; 4] = [
    [-0.95695839311334274, 1.9024390243902438],
    [0.92306990332963501, 1.8096371207614514],
    [-0.88340931961080393, 1.7213621392608929],
    [0.85205871432575275, 1.637393254418898],
];

/// MFEM 4.10, `SIAVSolver(order)`: `q` then `p` per step, `q = (1, 2)`,
/// `p = (0.5, -0.25)`, `P = [[0, 1], [-1, 0]]`, `F = A`.
const SIAV_Q: [[[f64; 2]; 4]; 4] = [
    // order 1
    [
        [0.96499999999999997, 11.930000000000001],
        [0.87034999999999996, 31.390700000000002],
        [0.61874649999999987, 59.240993000000003],
        [0.070938034999999733, 92.686341069999997],
    ],
    // order 2
    [
        [0.97006250000000005, 6.8776250000000001],
        [0.90573687500000011, 21.38709875],
        [0.73447575625000017, 44.740070262500005],
        [0.33951428618750013, 74.990398634875007],
    ],
    // order 3
    [
        [0.96659776215277771, 6.8825977233796314],
        [0.89479263659444419, 21.323745934830196],
        [0.7127526920585765, 44.401732753987673],
        [0.30596560884170787, 73.971391276446312],
    ],
    // order 4
    [
        [0.96787423626022728, 6.9216322936433237],
        [0.89685001377319173, 21.477595614469337],
        [0.71403836184229164, 44.745678272618918],
        [0.30321752423254589, 74.555025363656711],
    ],
];

const SIAV_P: [[[f64; 2]; 4]; 4] = [
    // order 1
    [
        [-99.300000000000011, -0.34999999999999998],
        [-194.60700000000003, -0.94650000000000012],
        [-278.50292999999999, -2.5160350000000005],
        [-334.4534807, -5.4780846500000013],
    ],
    // order 2
    [
        [-98.052500000000009, -0.34875],
        [-192.13697500000001, -0.93776250000000005],
        [-274.92245525000004, -2.4874598749999999],
        [-330.08411219750008, -5.4117695262500005],
    ],
    // order 3
    [
        [-97.700060567129654, -0.43170293402777782],
        [-190.20468732367539, -1.0996024534394711],
        [-269.03246401914333, -2.7129611992315099],
        [-316.9507581823957, -5.6587908737942554],
    ],
    // order 4
    [
        [-98.189894231907374, -0.43233820857396049],
        [-191.1291605212127, -1.1065601046233726],
        [-270.20973755040734, -2.7371249417402845],
        [-317.90430971260844, -5.7147617755224447],
    ],
];

const TOL: f64 = 1e-13;

/// Relative difference, largest over the components (values up to ~1e2, so this
/// is the meaningful measure; `≤ 1e-13` is ~1 ulp of a 17-digit value).
fn rel_err(got: &[f64], want: &[f64]) -> f64 {
    got.iter()
        .zip(want.iter())
        .map(|(g, w)| (g - w).abs() / w.abs().max(1e-30))
        .fold(0.0_f64, f64::max)
}

/// Run `solver` for `NSTEPS` steps and return every state.
fn integrate<S: OdeSolver<LinearOp>>(mut solver: S, op: &mut LinearOp) -> Vec<[f64; 2]> {
    solver.init(op);
    let mut u = U0;
    let mut t = 0.0;
    let mut out = Vec::with_capacity(NSTEPS);
    for _ in 0..NSTEPS {
        solver.step(op, &mut u, &mut t, DT);
        out.push(u);
    }
    // `t` advanced by exactly one dt per step.
    assert!((t - DT * NSTEPS as f64).abs() < 1e-15, "t = {t}");
    out
}

fn cpp_operator() -> LinearOp {
    LinearOp::new(stiff_operator())
}

#[test]
fn backward_euler_matches_mfem() {
    let mut op = cpp_operator();
    for (k, step) in integrate(BackwardEulerSolver::new(), &mut op).iter().enumerate() {
        let e = rel_err(step, &BE[k]);
        assert!(e <= TOL, "BackwardEuler step {k}: {step:?} vs {:?} (rel {e:.3e})", BE[k]);
    }
}

#[test]
fn sdirk23_matches_mfem() {
    for (opt, want) in [
        (Sdirk23Gamma::AStable, SDIRK23),
        (Sdirk23Gamma::LStableOutside, SDIRK23_G3),
    ] {
        let mut op = cpp_operator();
        let solver = Sdirk23Solver::new(opt);
        assert!(solver.gamma() > 0.0);
        for (k, step) in integrate(solver, &mut op).iter().enumerate() {
            let e = rel_err(step, &want[k]);
            assert!(e <= TOL, "SDIRK23 {opt:?} step {k}: {step:?} vs {:?} (rel {e:.3e})", want[k]);
        }
    }
}

#[test]
fn sdirk33_matches_mfem() {
    let mut op = cpp_operator();
    for (k, step) in integrate(Sdirk33Solver::new(), &mut op).iter().enumerate() {
        let e = rel_err(step, &SDIRK33[k]);
        assert!(e <= TOL, "SDIRK33 step {k}: {step:?} vs {:?} (rel {e:.3e})", SDIRK33[k]);
    }
}

#[test]
fn sdirk34_matches_mfem() {
    let mut op = cpp_operator();
    for (k, step) in integrate(Sdirk34Solver::new(), &mut op).iter().enumerate() {
        let e = rel_err(step, &SDIRK34[k]);
        assert!(e <= TOL, "SDIRK34 step {k}: {step:?} vs {:?} (rel {e:.3e})", SDIRK34[k]);
    }
}

#[test]
fn implicit_midpoint_matches_mfem() {
    let mut op = cpp_operator();
    for (k, step) in integrate(ImplicitMidpointSolver::new(), &mut op).iter().enumerate() {
        let e = rel_err(step, &MIDPOINT[k]);
        assert!(e <= TOL, "ImplicitMidpoint step {k}: {step:?} vs {:?} (rel {e:.3e})", MIDPOINT[k]);
    }
}

/// `run` must be the `while (t < tf) step(...)` of `ODESolver::Run`.
#[test]
fn run_matches_repeated_step() {
    let mut a = cpp_operator();
    let mut b = cpp_operator();
    let mut solver = BackwardEulerSolver::new();
    solver.init(&a);
    let mut u_step = U0;
    let mut t_step = 0.0;
    for _ in 0..NSTEPS {
        solver.step(&mut a, &mut u_step, &mut t_step, DT);
    }

    let mut solver = BackwardEulerSolver::new();
    solver.init(&b);
    let mut u_run = U0;
    let mut t_run = 0.0;
    solver.run(&mut b, &mut u_run, &mut t_run, DT, DT * NSTEPS as f64);
    assert_eq!(u_run, u_step);
    assert_eq!(t_run, t_step);
}

/// The `ImplicitVariableType::STATE` branch of `ODESolver::Step`
/// (`f->ImplicitVarTypeIsState()`): with a state-returning `ImplicitSolve` the
/// solvers call `ComputeSlopeFromState` and must land on the same iterates as
/// the slope convention (MFEM `ode.cpp:688` vs `:694`).
#[test]
fn state_implicit_variable_type_matches_slope_convention() {
    let mut slope_op = cpp_operator();
    let mut state_op = LinearOp::new(stiff_operator()).with_state_implicit(true);
    assert!(state_op.implicit_var_is_state());
    assert!(!slope_op.implicit_var_is_state());

    let u_slope = integrate(BackwardEulerSolver::new(), &mut slope_op);
    let u_state = integrate(BackwardEulerSolver::new(), &mut state_op);
    for (k, (s, st)) in u_slope.iter().zip(u_state.iter()).enumerate() {
        let e = rel_err(st, s);
        assert!(e <= TOL, "BackwardEuler(STATE) step {k}: {st:?} vs {s:?} (rel {e:.3e})");
    }

    // Implicit midpoint: x.Neg(); x.Add(2.0, k) for the state convention.
    let mut slope_op = cpp_operator();
    let mut state_op = LinearOp::new(stiff_operator()).with_state_implicit(true);
    let u_slope = integrate(ImplicitMidpointSolver::new(), &mut slope_op);
    let u_state = integrate(ImplicitMidpointSolver::new(), &mut state_op);
    for (k, (s, st)) in u_slope.iter().zip(u_state.iter()).enumerate() {
        let e = rel_err(st, s);
        assert!(e <= TOL, "Midpoint(STATE) step {k}: {st:?} vs {s:?} (rel {e:.3e})");
    }
}

/// `SIAVSolver::Step` against MFEM, orders 1–4, explicit `F_` branch
/// (`F_->Mult(q, dp_)`).
#[test]
fn siav_matches_mfem() {
    let p_mat = DenseOperator::from_rows(&[&[0.0, 1.0], &[-1.0, 0.0]]);
    for order in 1..=4 {
        let siav = SiavSolver::new(order);
        assert_eq!(siav.order(), order);
        assert!((siav.a().iter().sum::<f64>() - 1.0).abs() < 1e-15, "sum(a) != 1");

        // `MaxwellSolver`-style explicit F: `isExplicit()` is true, so the step
        // calls `Mult` and never `ImplicitSolve`.
        let mut f = LinearOp::new(stiff_operator()).as_explicit();
        assert!(f.is_explicit());
        let mut q = vec![U0[0], U0[1]];
        let mut p = vec![0.5, -0.25];
        let mut t = 0.0;
        for s in 0..NSTEPS {
            siav.step_vecs(&mut f, |x, y| p_mat.mult(x, y), &mut q, &mut p, &mut t, DT);
            let (eq, ep) = (
                rel_err(&q, &SIAV_Q[order - 1][s]),
                rel_err(&p, &SIAV_P[order - 1][s]),
            );
            assert!(
                eq <= TOL && ep <= TOL,
                "SIAV{order} step {s}: q {q:?} (rel {eq:.3e}), p {p:?} (rel {ep:.3e})"
            );
        }
    }
}

/// `isExplicit()` decides which of the two `F_` branches runs: an implicit
/// operator must be driven through `ImplicitSolve`.  (`ImplicitDenseOp` returns
/// `A x` from `implicit_solve`, so the two runs must agree exactly.)
#[test]
fn siav_implicit_branch_is_taken_when_the_operator_is_implicit() {
    struct ImplicitDenseOp {
        a: DenseOperator,
        calls: std::rc::Rc<std::cell::Cell<usize>>,
    }
    impl TimeDependentOperator for ImplicitDenseOp {
        fn size(&self) -> usize { self.a.size() }
        fn set_time(&mut self, _t: f64) {}
        fn mult(&self, _x: &[f64], _y: &mut [f64]) {
            panic!("SIAV must not call Mult on an implicit operator");
        }
        fn implicit_solve(&mut self, _dt: f64, x: &[f64], k: &mut [f64]) {
            self.calls.set(self.calls.get() + 1);
            self.a.mult(x, k);
        }
        fn is_explicit(&self) -> bool { false }
    }

    let p_mat = DenseOperator::from_rows(&[&[0.0, 1.0], &[-1.0, 0.0]]);
    let siav = SiavSolver::new(2);

    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    let mut f = ImplicitDenseOp { a: stiff_operator(), calls: calls.clone() };
    let mut q = vec![U0[0], U0[1]];
    let mut p = vec![0.5, -0.25];
    let mut t = 0.0;
    siav.step_vecs(&mut f, |x, y| p_mat.mult(x, y), &mut q, &mut p, &mut t, DT);
    // order 2 has exactly one nonzero b_i, so one implicit solve per step.
    assert_eq!(calls.get(), 1);

    let mut f_exp = LinearOp::new(stiff_operator()).as_explicit();
    let mut q_exp = vec![U0[0], U0[1]];
    let mut p_exp = vec![0.5, -0.25];
    let mut t_exp = 0.0;
    siav.step_vecs(&mut f_exp, |x, y| p_mat.mult(x, y), &mut q_exp, &mut p_exp, &mut t_exp, DT);
    assert_eq!(q, q_exp);
    assert_eq!(p, p_exp);
}

/// `a`/`b` tables of `ode.cpp:1109-1149` (they are what makes the scheme
/// symplectic; a typo here is invisible in a single step).
#[test]
fn siav_tables_match_ode_cpp() {
    let expect: [(Vec<f64>, Vec<f64>); 4] = [
        (vec![1.0], vec![1.0]),
        (vec![0.5, 0.5], vec![0.0, 1.0]),
        (vec![2.0 / 3.0, -2.0 / 3.0, 1.0], vec![7.0 / 24.0, 0.75, -1.0 / 24.0]),
        (
            {
                let c = 2.0_f64.powf(1.0 / 3.0);
                vec![(2.0 + c + 1.0 / c) / 6.0, (1.0 - c - 1.0 / c) / 6.0,
                     (1.0 - c - 1.0 / c) / 6.0, (2.0 + c + 1.0 / c) / 6.0]
            },
            {
                let c = 2.0_f64.powf(1.0 / 3.0);
                vec![0.0, 1.0 / (2.0 - c), 1.0 / (1.0 - c * c), 1.0 / (2.0 - c)]
            },
        ),
    ];
    for (i, (a, b)) in expect.iter().enumerate() {
        let s = SiavSolver::new(i + 1);
        assert_eq!(s.a(), a.as_slice(), "a table of order {}", i + 1);
        assert_eq!(s.b(), b.as_slice(), "b table of order {}", i + 1);
    }
}
