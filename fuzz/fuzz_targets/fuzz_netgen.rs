#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz the Netgen .vol parser with arbitrary bytes.
    let _ = fem_io::netgen::read_netgen_vol(data);
});
