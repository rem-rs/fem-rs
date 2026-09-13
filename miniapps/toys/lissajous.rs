//! # Lissajous Miniapp — Spinning Optical Illusion (partial delivery, exit 3)
//!
//! Port target: MFEM `miniapps/toys/lissajous.cpp` (MFEM 4.10).
//!
//! Generates two Lissajous curves in 3D which appear to spin vertically and/or
//! horizontally, even though the net motion is the same (the 2019 Illusion of
//! the Year "Dual Axis Illusion", <http://illusionoftheyear.com/2019/12/dual-axis-illusion>).
//!
//! ## Gap list (round 32, D130) — why this is not a 1:1 port
//!
//! The C++ miniapp builds a **2-D surface mesh embedded in 3-D**:
//! `Mesh::MakeCartesian2D(nx, ny, QUADRILATERAL, 1, 2π, 2π)` →
//! `SetCurvature(order, true, 3, Ordering::byVDIM)` →
//! `Transform(lissajous_trans_{v,h})`, then writes
//! `lissajous.mesh` (29,968 B) and `lissajous.gf` = the H¹ grid function
//! `u = x[2]` (4,829 B) for the *horizontal* mesh, measured with the C++ binary.
//!
//! `fem_mesh::Mesh<D>` stores exactly `D` coordinate components (`D` is both the
//! topological and the spatial dimension — `crates/io/src/mfem.rs` documents
//! "the two agree for every mesh this writer accepts"), so fem-rs cannot
//! represent a `dim = 2, sdim = 3` surface mesh, cannot set its 3-D curvature,
//! and has no writer for such a `.mesh`/`.gf` pair.
//!
//! The previous revision of this file silently wrote two **all-zero**
//! placeholders under non-C++ names (`lissajous-v.gf`, `lissajous-h.gf`,
//! 5,534 B each) plus invented `Vertical curve sample at (π,π): …` lines; both
//! are removed — nothing is written now and the process exits with code 3 (the
//! project's honest partial-delivery code).
//!
//! Usage:
//!   cargo run --release --example toys_lissajous -- -no-vis
//!   cargo run --release --example toys_lissajous -- -a 5 -b 4
//!   cargo run --release --example toys_lissajous -- -a 11 -b 10 -o 4

/// Default Lissajous curve parameters (matching the C++ globals).
const A_DEFAULT: f64 = 3.0;
const B_DEFAULT: f64 = 2.0;
const DELTA_DEFAULT: f64 = 90.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut nx: usize = 32;
    let mut ny: usize = 3;
    let mut order: u8 = 2;
    let mut a: f64 = A_DEFAULT;
    let mut b: f64 = B_DEFAULT;
    let mut delta: f64 = DELTA_DEFAULT;
    let mut visualization = true;
    let mut visport: i32 = 19916;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-nx" | "--num-elements-x" => {
                nx = it.next().and_then(|v| v.parse().ok()).unwrap_or(32);
            }
            "-ny" | "--num-elements-y" => {
                ny = it.next().and_then(|v| v.parse().ok()).unwrap_or(3);
            }
            "-o" | "--mesh-order" => {
                order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2);
            }
            "-a" | "--x-frequency" => {
                a = it.next().and_then(|v| v.parse().ok()).unwrap_or(A_DEFAULT);
            }
            "-b" | "--y-frequency" => {
                b = it.next().and_then(|v| v.parse().ok()).unwrap_or(B_DEFAULT);
            }
            "-delta" | "--x-phase" => {
                delta = it.next().and_then(|v| v.parse().ok()).unwrap_or(DELTA_DEFAULT);
            }
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            "-p" | "--send-port" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { visport = val; } }
            }
            _ => {}
        }
    }

    // C++ `args.PrintOptions(cout)`.
    println!("Options used:");
    println!("   --num-elements-x {nx}");
    println!("   --num-elements-y {ny}");
    println!("   --mesh-order {order}");
    println!("   --x-frequency {}", fem_solver::fmt_g(a));
    println!("   --y-frequency {}", fem_solver::fmt_g(b));
    println!("   --x-phase {}", fem_solver::fmt_g(delta));
    println!("   --{}", if visualization { "visualization" } else { "no-visualization" });
    println!("   --send-port {visport}");

    eprintln!(
        "lissajous (Rust port): partial delivery, exit 3. The C++ miniapp builds a 2-D surface \
         mesh embedded in 3-D (`Mesh::MakeCartesian2D` + `SetCurvature(order, true, 3, byVDIM)` + \
         `Transform`) and writes `lissajous.mesh` (29968 B) plus `lissajous.gf` = the H1 \
         GridFunction `u = x[2]` (4829 B). `fem_mesh::Mesh<D>` has `sdim == dim` (no surface-in-3D \
         meshes), so neither the curved mesh nor the grid function can be built or written; no \
         output files are produced and no GLVis socket is opened."
    );
    std::process::exit(3);
}
