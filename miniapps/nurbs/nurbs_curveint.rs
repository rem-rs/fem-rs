//! # NURBS Curve Interpolation Miniapp — port of MFEM `miniapps/nurbs/nurbs_curveint.cpp`
//!
//! ⚠ Declared stub — under active porting (D374).  The C++ miniapp builds a
//! `NURBSPatch` control-point object, degree-elevates and knot-inserts it,
//! interpolates a sine through `KnotVector::GetInterpolant`/`GetDemko`, and
//! writes a `patches`-flavour NURBS mesh (`sin-fit.mesh`).  This placeholder
//! keeps the Cargo registration compiling while the `NURBSPatch` object layer
//! and the driver are written; it refuses to run rather than emit a wrong file.

fn main() {
    eprintln!(
        "nurbs_curveint: NOT PORTED YET — D374 in progress; this stub only keeps the Cargo \
         registration compiling.  C++ reference: MFEM 4.10 miniapps/nurbs/nurbs_curveint.cpp. \
         Exiting with status 3."
    );
    std::process::exit(3);
}
