#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz the GMSH .msh parser with arbitrary bytes.
    // The parser must not panic on any input.
    let _ = fem_io::gmsh::read_msh(data);
});
