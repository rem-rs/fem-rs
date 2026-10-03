//! round-106 (D1046-D1050) byte-parity pins: toys (mandel/mondrian/lissajous)
//! and miniapp joule, against the MFEM 4.10 oracles.
//!
//! Every hash below was recorded from outputs verified **byte-identical**
//! against fresh `g++ -O2` MFEM 4.10 oracle binaries (WSL `$HOME/mfem410_ser`
//! for the toys, `$HOME/mfem410_mpi` + `mpirun -np 1` for joule) during
//! round 106; the evidence log is `tmp/d106resid/REPORT.md`.  The pins
//! rebuild each example, run it in a scratch directory laid out exactly like
//! the oracle run (relative `data/...` paths, so the `Options used:` dump
//! matches too), and md5-compare stdout and every written file.
//!
//! Round 107 (D1049 closed, D1066 dead filter removed): the former stray
//! line `Elements with wrong orientation: 70 / 252 (not fixed)` is gone —
//! `crates/mesh` `check_element_orientation` now judges wedge/pyramid/hex
//! through the MFEM center-trilinear Jacobian, and the joule stdout pins
//! below compare the **unfiltered** stdout (hashes re-recorded in
//! `3957857c`; pin labels kept md5-equal, only the filtering is history).
//!
//! Residual registered gaps (not pinned): mandel/mondrian `-vis` (no GLVis
//! client; refusal), lissajous `-vis`/`-o != 2`, the joule coupled time loop,
//! `-amr 1` (3-D NCMesh, D1047), `-sc 1`/`-debug 1`/`-print 1`/`-vis`, and
//! `-visit` with `-rs > 0` (refined-bdr parity, D1048) or `--ranks > 1`.

use std::path::{Path, PathBuf};
use std::process::Command;

// ─── MD5 (RFC 1321) — self-contained so no dependency changes are needed ────

fn md5(data: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, //
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, //
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, //
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, //
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, //
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, //
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, //
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, //
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8, //
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, //
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, //
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, //
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, //
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, //
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, //
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, //
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1, //
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, //
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
    ];
    let (mut a0, mut b0, mut c0, mut d0) =
        (0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476);
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_le_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, w) in m.iter_mut().enumerate() {
            *w = u32::from_le_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g): (u32, usize) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let tmp = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(tmp.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&a0.to_le_bytes());
    out[4..8].copy_from_slice(&b0.to_le_bytes());
    out[8..12].copy_from_slice(&c0.to_le_bytes());
    out[12..16].copy_from_slice(&d0.to_le_bytes());
    out
}

fn md5_hex(data: &[u8]) -> String {
    md5(data).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn d106_md5_selfcheck() {
    assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
}

// ─── Harness ────────────────────────────────────────────────────────────────

fn repo_root() -> PathBuf {
    // examples/ -> fem-rs root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("manifest parent")
        .to_path_buf()
}

fn build_example(name: &str) -> PathBuf {
    let root = repo_root();
    let status = Command::new(env!("CARGO"))
        .args(["build", "--release", "--example", name])
        .current_dir(&root)
        .status()
        .expect("spawn cargo build");
    assert!(status.success(), "cargo build --example {name} failed");
    root.join("target")
        .join("release")
        .join("examples")
        .join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

/// Run `exe` with `args` in `dir`; returns (stdout, exit code).
fn run(exe: &Path, dir: &Path, args: &[&str]) -> (Vec<u8>, i32) {
    let out = Command::new(exe)
        .args(args)
        .current_dir(dir)
        .output()
        .expect("spawn example");
    (out.stdout, out.status.code().unwrap_or(-1))
}

fn assert_md5(label: &str, data: &[u8], expected: &str) {
    let got = md5_hex(data);
    assert_eq!(
        got,
        expected,
        "{label}: md5 {got} != oracle {expected} ({} bytes)",
        data.len()
    );
}

fn read_file(dir: &Path, rel: &str) -> Vec<u8> {
    std::fs::read(dir.join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}


// ─── Pins: toys ──────────────────────────────────────────────────────────────

/// Scratch layout shared by the toys runs: `<tmp>/data` holds the inputs and
/// the examples run from `<tmp>/a/b`, matching the oracle invocation layout
/// (`../../data/inline-quad.mesh`).
fn toys_scratch(base: &Path) -> PathBuf {
    let dir = base.join("a").join("b");
    std::fs::create_dir_all(&dir).expect("mkdir scratch");
    let data = base.join("data");
    std::fs::create_dir_all(&data).expect("mkdir data");
    for f in ["inline-quad.mesh", "australia.pgm"] {
        std::fs::copy(repo_root().join("data").join(f), data.join(f))
            .unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    dir
}

#[test]
fn d106_mandel_byteparity() {
    let base = std::env::temp_dir().join("d106_pin_mandel");
    let _ = std::fs::remove_dir_all(&base);
    let dir = toys_scratch(&base);
    let exe = build_example("toys_mandel");

    let (out, code) = run(&exe, &dir, &["-no-vis"]);
    assert_eq!(code, 0, "mandel -no-vis must exit 0 (1:1 quad path)");
    assert_md5("mandel -no-vis stdout", &out, "d041218776e8d5beb3ae9a51dbfd6671");
    assert_md5(
        "mandel -no-vis mesh",
        &read_file(&dir, "mandel.mesh"),
        "30ed1ad82bb01b9a6e17b010826dca7e",
    );

    let (out, code) = run(&exe, &dir, &["-no-vis", "-a"]);
    assert_eq!(code, 0);
    assert_md5("mandel -a stdout", &out, "d22459f28030464b9a972eabd56c2c6c");
    assert_md5(
        "mandel -a mesh",
        &read_file(&dir, "mandel.mesh"),
        "3dd627c1056fb8b1893b61d3df166e51",
    );
}

#[test]
fn d106_mondrian_byteparity() {
    let base = std::env::temp_dir().join("d106_pin_mondrian");
    let _ = std::fs::remove_dir_all(&base);
    let dir = toys_scratch(&base);
    let exe = build_example("toys_mondrian");
    let common = [
        "-m",
        "../../data/inline-quad.mesh",
        "-i",
        "../../data/australia.pgm",
        "-no-vis",
    ];

    let (out, code) = run(&exe, &dir, &common);
    assert_eq!(code, 0);
    assert_md5("mondrian stdout", &out, "ba8c657c454af429121932eeda3e80d6");
    assert_md5(
        "mondrian mesh",
        &read_file(&dir, "mondrian.mesh"),
        "fb2da67f6ddaf271e87b3e616b376a75",
    );

    let mut args = common.to_vec();
    args.push("-a");
    let (out, code) = run(&exe, &dir, &args);
    assert_eq!(code, 0);
    assert_md5("mondrian -a stdout", &out, "716d275d6b74eba0dfbb97e34f6e2f1f");
    assert_md5(
        "mondrian -a mesh",
        &read_file(&dir, "mondrian.mesh"),
        "f482e58f7942b69d02952bf3ccc372de",
    );
}

#[test]
fn d106_lissajous_byteparity() {
    let base = std::env::temp_dir().join("d106_pin_lissajous");
    let _ = std::fs::remove_dir_all(&base);
    let dir = base.join("a").join("b");
    std::fs::create_dir_all(&dir).expect("mkdir scratch");
    let exe = build_example("toys_lissajous");

    let (out, code) = run(&exe, &dir, &["-no-vis"]);
    assert_eq!(code, 0);
    assert_md5("lissajous stdout", &out, "64de6e051ae0dba5b0725ef44b34fcd6");
    assert_md5(
        "lissajous.mesh",
        &read_file(&dir, "lissajous.mesh"),
        "a60dda042095fd1ceb11736471eb5075",
    );
    assert_md5(
        "lissajous.gf",
        &read_file(&dir, "lissajous.gf"),
        "491971c06fe1465ecc14f5b77766f1e5",
    );
}

// ─── Pins: joule ─────────────────────────────────────────────────────────────

fn joule_scratch(base: &Path) -> PathBuf {
    std::fs::create_dir_all(base.join("data")).expect("mkdir joule scratch");
    std::fs::copy(
        repo_root().join("data").join("cylinder-hex.mesh"),
        base.join("data").join("cylinder-hex.mesh"),
    )
    .expect("copy cylinder-hex.mesh");
    base.to_path_buf()
}

#[test]
fn d106_joule_stdout_and_visit_dc_byteparity() {
    let base = std::env::temp_dir().join("d106_pin_joule");
    let _ = std::fs::remove_dir_all(&base);
    let dir = joule_scratch(&base);
    let exe = build_example("miniapp_joule");

    // default run
    let (out, code) = run(
        &exe,
        &dir,
        &[
            "-m",
            "data/cylinder-hex.mesh",
            "-tf",
            "1.0",
            "-dt",
            "0.5",
            "-no-vis",
            "-no-visit",
        ],
    );
    assert_eq!(code, 3, "joule stops with 3 before the (unported) time loop");
    assert_md5(
        "joule default stdout (unfiltered)",
        &out,
        "bf6ff61808a8f38e08e23d6295a6a56d",
    );

    // -rs 1 run
    let (out, code) = run(
        &exe,
        &dir,
        &[
            "-m",
            "data/cylinder-hex.mesh",
            "-rs",
            "1",
            "-tf",
            "1.0",
            "-dt",
            "0.5",
            "-no-vis",
            "-no-visit",
        ],
    );
    assert_eq!(code, 3);
    assert_md5(
        "joule -rs 1 stdout (unfiltered)",
        &out,
        "41e3407d37032e084f6464c1578f62b1",
    );

    // -visit cycle-0 dc (default mesh)
    let (out, code) = run(
        &exe,
        &dir,
        &[
            "-m",
            "data/cylinder-hex.mesh",
            "-tf",
            "1.0",
            "-dt",
            "0.5",
            "-no-vis",
            "-visit",
        ],
    );
    assert_eq!(code, 3);
    assert_md5(
        "joule visit stdout (unfiltered)",
        &out,
        "87d8d2f601f139a1516161478aa0b255",
    );
    let dc = dir.join("Joule_000000");
    for (rel, hash) in [
        ("mesh.000000", "b20dcf4c7abe7edbbd6b815d73e3657e"),
        ("Phi.000000", "da06f542ddbb02e120e67d08d951a4e5"),
        ("E.000000", "b10e2422d7e1f35ff07e3a0394222541"),
        ("B.000000", "63b03e79f0c61e07d2867f4568dde65e"),
        ("T.000000", "8f5ae563e4f94370f253c52eef3b615b"),
        ("w.000000", "8f5ae563e4f94370f253c52eef3b615b"),
        ("F.000000", "63b03e79f0c61e07d2867f4568dde65e"),
    ] {
        assert_md5(&format!("dc {rel}"), &read_file(&dc, rel), hash);
    }
    assert_md5(
        "Joule_000000.mfem_root",
        &read_file(&dir, "Joule_000000.mfem_root"),
        "5eed9dabd0d6ecbcbfbc42f87b5e93b8",
    );
}
