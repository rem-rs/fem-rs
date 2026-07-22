/// Build script for fem-linalg-gpu.
///
/// Reads the WGSL shader sources from wgsl/*.wgsl and generates f64 variants
/// into OUT_DIR. The f64 variant is produced by replacing `f32` → `f64`
/// at build time, so there is zero runtime overhead for the f64 PA apply path.
///
/// In pa_apply.rs, the f64 shaders are loaded via include_str! from OUT_DIR.
use std::path::Path;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let wgsl_dir = Path::new("wgsl");

    // Only regenerate when the WGSL files change.
    println!("cargo::rerun-if-changed=wgsl/");

    if let Ok(entries) = std::fs::read_dir(wgsl_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "wgsl") {
                let src = std::fs::read_to_string(&path).unwrap();
                let stem = path.file_stem().unwrap().to_str().unwrap();
                let f64_src = src.replace("f32", "f64");
                let out_path = Path::new(&out_dir).join(format!("{stem}_f64.wgsl"));
                std::fs::write(&out_path, f64_src).unwrap();
                println!("cargo::rerun-if-changed={}", path.display());
            }
        }
    }
}
