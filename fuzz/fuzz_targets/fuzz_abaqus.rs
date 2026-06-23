#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz the Abaqus .inp parser with arbitrary bytes.
    let _ = fem_io::abaqus::read_abaqus_inp(data);
});
