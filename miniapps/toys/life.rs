//! # Life Miniapp — Conway's Game of Life
//!
//! 1:1 port of MFEM `miniapps/toys/life.cpp`.
//!
//! Implements Conway's Game of Life. A few simple starting positions are
//! available as well as a random initial state. The game will terminate only
//! if two successive iterations are identical.
//!
//! Sample runs:
//!   cargo run --release --example toys_life
//!   cargo run --release --example toys_life -- -nx 30
//!   cargo run --release --example toys_life -- -nx 100 -ny 100 -r 0.3
//!   cargo run --release --example toys_life -- -g "2 3 0"
//!   cargo run --release --example toys_life -- -b "10 10 0" -g "2 2 0"
//!   cargo run --release --example toys_life -- -sp "8 10 0 1 1 1 2 1 1 1"
//!   cargo run --release --example toys_life -- -nx 30 -sp "11 11 1 1 1 1 1 1 1 1 2 1 0 1 1 1 1 0 1 2 1 1 1 1 1 1 1 1"

use fem_io::mfem::{write_mfem_file, write_mfem_gf_file};
use fem_mesh::Mesh;
use fem_space::L2Space;

/// Periodic index into an `nx × ny` grid.
fn index(i: i32, j: i32, nx: i32, ny: i32) -> usize {
    ((j + ny).rem_euclid(ny)) as usize * nx as usize + ((i + nx).rem_euclid(nx)) as usize
}

/// Apply one step of the Game of Life.  Returns true if the state is stable
/// (identical to the previous state).
fn game_step(cur: &mut [bool], next: &mut [bool], nx: i32, ny: i32) -> bool {
    let mut stable = true;
    for j in 0..ny {
        for i in 0..nx {
            let c = cur[index(i + 0, j + 0, nx, nx)] as i32
                + cur[index(i + 1, j + 0, nx, ny)] as i32
                + cur[index(i + 1, j + 1, nx, ny)] as i32
                + cur[index(i + 0, j + 1, nx, ny)] as i32
                + cur[index(i - 1, j + 1, nx, ny)] as i32
                + cur[index(i - 1, j + 0, nx, ny)] as i32
                + cur[index(i - 1, j - 1, nx, ny)] as i32
                + cur[index(i + 0, j - 1, nx, ny)] as i32
                + cur[index(i + 1, j - 1, nx, ny)] as i32;
            next[index(i, j, nx, ny)] = match c {
                3 => true,
                4 => cur[index(i, j, nx, ny)],
                _ => false,
            };
            stable &= next[index(i, j, nx, ny)] == cur[index(i, j, nx, ny)];
        }
    }
    stable
}

/// Copy the bit vector into the grid function DOFs.
fn project_step(b: &[bool], v: &mut [f64], n: usize) {
    for i in 0..n {
        v[i] = if b[i] { 1.0 } else { 0.0 };
    }
}

/// Initialize a "blinker" oscillator centered at `(cx, cy)` with orientation
/// `ornt` (0 = vertical, 1 = horizontal).
fn init_blinker(b: &mut [bool], nx: i32, ny: i32, params: &[i32]) {
    let n = params.len() / 3;
    for k in 0..n {
        let cx = params[3 * k];
        let cy = params[3 * k + 1];
        let ornt = params[3 * k + 2];
        match ornt % 2 {
            0 => {
                b[index(cx + 0, cy + 1, nx, ny)] = true;
                b[index(cx + 0, cy + 0, nx, ny)] = true;
                b[index(cx + 0, cy - 1, nx, ny)] = true;
            }
            _ => {
                b[index(cx + 1, cy + 0, nx, ny)] = true;
                b[index(cx + 0, cy + 0, nx, ny)] = true;
                b[index(cx - 1, cy + 0, nx, ny)] = true;
            }
        }
    }
}

/// Initialize a "glider" centered at `(cx, cy)` with orientation `ornt` (0..3).
fn init_glider(b: &mut [bool], nx: i32, ny: i32, params: &[i32]) {
    let n = params.len() / 3;
    for k in 0..n {
        let cx = params[3 * k];
        let cy = params[3 * k + 1];
        let ornt = params[3 * k + 2];
        match ornt % 4 {
            0 => {
                b[index(cx - 1, cy + 0, nx, ny)] = true;
                b[index(cx + 0, cy + 1, nx, ny)] = true;
                b[index(cx + 1, cy - 1, nx, ny)] = true;
                b[index(cx + 1, cy + 0, nx, ny)] = true;
                b[index(cx + 1, cy + 1, nx, ny)] = true;
            }
            1 => {
                b[index(cx + 0, cy - 1, nx, ny)] = true;
                b[index(cx - 1, cy + 0, nx, ny)] = true;
                b[index(cx - 1, cy + 1, nx, ny)] = true;
                b[index(cx + 0, cy + 1, nx, ny)] = true;
                b[index(cx + 1, cy + 1, nx, ny)] = true;
            }
            2 => {
                b[index(cx + 1, cy + 0, nx, ny)] = true;
                b[index(cx + 0, cy - 1, nx, ny)] = true;
                b[index(cx - 1, cy - 1, nx, ny)] = true;
                b[index(cx - 1, cy + 0, nx, ny)] = true;
                b[index(cx - 1, cy + 1, nx, ny)] = true;
            }
            _ => {
                b[index(cx + 0, cy + 1, nx, ny)] = true;
                b[index(cx + 1, cy + 0, nx, ny)] = true;
                b[index(cx - 1, cy - 1, nx, ny)] = true;
                b[index(cx + 0, cy - 1, nx, ny)] = true;
                b[index(cx + 1, cy - 1, nx, ny)] = true;
            }
        }
    }
}

/// Initialize a sketch pad from a flat array.  Value `2` means newline
/// (reset ox, decrement oy).
fn init_sketch_pad(b: &mut [bool], nx: i32, ny: i32, params: &[i32]) {
    let cx = params[0];
    let cy = params[1];
    let mut ox = 0i32;
    let mut oy = 0i32;
    for &p in &params[2..] {
        if p / 2 == 1 {
            ox = 0;
            oy -= 1;
        } else {
            b[index(cx + ox, cy + oy, nx, ny)] = p != 0;
            ox += 1;
        }
    }
}

/// Draw "MFEM" (or a single "M" if the grid is too small) into the bit vector.
fn init_mfem(b: &mut [bool], nx: i32, ny: i32) {
    let wx = if nx >= 23 { 23 } else { 5 };
    let hy = if ny >= 7 { 7 } else { 5 };

    if wx == 23 {
        let ox = (nx - 23) / 2;
        let oy = (ny - hy) / 2;
        for j in 0..hy {
            b[index(ox + 0, oy + j, nx, ny)] = true;
            b[index(ox + 4, oy + j, nx, ny)] = true;
            b[index(ox + 6, oy + j, nx, ny)] = true;
            b[index(ox + 12, oy + j, nx, ny)] = true;
            b[index(ox + 18, oy + j, nx, ny)] = true;
            b[index(ox + 22, oy + j, nx, ny)] = true;
        }
        for i in 1..5 {
            b[index(ox + 6 + i, oy + hy - 1, nx, ny)] = true;
            b[index(ox + 12 + i, oy + 0, nx, ny)] = true;
            b[index(ox + 12 + i, oy + hy - 1, nx, ny)] = true;
        }
        for i in 1..4 {
            b[index(ox + 6 + i, oy + hy / 2, nx, ny)] = true;
            b[index(ox + 12 + i, oy + hy / 2, nx, ny)] = true;
        }
        b[index(ox + 1, oy + hy - 2, nx, ny)] = true;
        b[index(ox + 2, oy + hy - 3, nx, ny)] = true;
        b[index(ox + 3, oy + hy - 2, nx, ny)] = true;

        b[index(ox + 19, oy + hy - 2, nx, ny)] = true;
        b[index(ox + 20, oy + hy - 3, nx, ny)] = true;
        b[index(ox + 21, oy + hy - 2, nx, ny)] = true;
    } else if wx == 5 {
        let ox = (nx - 5) / 2;
        let oy = (ny - hy) / 2;
        for j in 0..hy {
            b[index(ox + 0, oy + j, nx, ny)] = true;
            b[index(ox + 4, oy + j, nx, ny)] = true;
        }
        b[index(ox + 1, oy + hy - 2, nx, ny)] = true;
        b[index(ox + 2, oy + hy - 3, nx, ny)] = true;
        b[index(ox + 3, oy + hy - 2, nx, ny)] = true;
    } else {
        b[index(nx / 2, ny / 2, nx, ny)] = true;
    }
}

/// Parse a space-separated list of integers from a CLI token.
fn parse_i32_list(token: &str) -> Vec<i32> {
    token.split_whitespace().filter_map(|s| s.parse().ok()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut nx: i32 = 20;
    let mut ny: i32 = 20;
    let mut r: f64 = -1.0;
    let mut rs: i32 = -1;
    let mut sketch_pad_params: Vec<i32> = Vec::new();
    let mut blinker_params: Vec<i32> = Vec::new();
    let mut glider_params: Vec<i32> = Vec::new();
    let mut visualization = false;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-nx" | "--num-elems-x" => {
                nx = it.next().and_then(|v| v.parse().ok()).unwrap_or(20);
            }
            "-ny" | "--num-elems-y" => {
                ny = it.next().and_then(|v| v.parse().ok()).unwrap_or(20);
            }
            "-r" | "--random-fraction" => {
                r = it.next().and_then(|v| v.parse().ok()).unwrap_or(-1.0);
            }
            "-rs" | "--random-seed" => {
                rs = it.next().and_then(|v| v.parse().ok()).unwrap_or(-1);
            }
            "-sp" | "--sketch-pad" => {
                sketch_pad_params = it.next().map(|v| parse_i32_list(v)).unwrap_or_default();
            }
            "-b" | "--blinker" => {
                blinker_params = it.next().map(|v| parse_i32_list(v)).unwrap_or_default();
            }
            "-g" | "--glider" => {
                glider_params = it.next().map(|v| parse_i32_list(v)).unwrap_or_default();
            }
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            _ => {}
        }
    }

    // 2. Build a rectangular mesh of quadrilateral elements.
    let mesh: Mesh<2> = Mesh::make_cartesian_2d(nx as usize, ny as usize, nx as f64, ny as f64);

    // 3. P0 L2 space.
    let fespace = L2Space::new(mesh.clone(), 0);
    let len = (nx as usize) * (ny as usize);

    // 4. Two bit arrays for double-buffering.
    let mut vb0 = vec![false; len];
    let mut vb1 = vec![false; len];

    // 5. Initialize state.
    if r > 0.0 {
        let seed = if rs < 0 {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as u32
        } else {
            rs as u32
        };
        println!("Using random seed:  {}", seed);
        // Simple xorshift PRNG matching MFEM behavior.
        let mut state: u32 = seed;
        for i in 0..len {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let rv = state as f64 / u32::MAX as f64;
            vb0[i] = rv <= r;
            vb1[i] = false;
        }
    } else {
        for v in &mut vb0 {
            *v = false;
        }
    }

    let mut initialized = false;
    if sketch_pad_params.len() > 2 {
        init_sketch_pad(&mut vb0, nx, ny, &sketch_pad_params);
        initialized = true;
    }
    if !blinker_params.is_empty() && blinker_params.len() % 3 == 0 {
        init_blinker(&mut vb0, nx, ny, &blinker_params);
        initialized = true;
    }
    if !glider_params.is_empty() && glider_params.len() % 3 == 0 {
        init_glider(&mut vb0, nx, ny, &glider_params);
        initialized = true;
    }
    if !initialized {
        init_mfem(&mut vb0, nx, ny);
    }

    // 5b. Grid function and initial projection.
    let mut v: Vec<f64> = vec![0.0; len];
    project_step(&vb0, &mut v, len);

    // 6. Run the game (with visualization = false, run until stable).
    println!();
    println!("Running the Game of Life...");
    loop {
        let stable = game_step(&mut vb0, &mut vb1, nx, ny);
        project_step(&vb1, &mut v, len);
        std::mem::swap(&mut vb0, &mut vb1);
        if stable {
            break;
        }
    }
    println!("done.");

    // 7. Save output.
    write_mfem_file("life.mesh", &mesh).expect("write mesh");
    write_mfem_gf_file("life.gf", 2, &v, "L2", 0, 1, 8).expect("write sol.gf");
    println!("Wrote life.mesh and life.gf");
}
