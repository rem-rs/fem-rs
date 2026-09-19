//! # MG Abs-L1 Jacobi Miniapp — port of MFEM `miniapps/diag-smoothers/mg-abs-l1-jacobi.cpp`
//!
//! ⚠ Declared stub — under active porting (D376).  The C++ miniapp drives the
//! `|A|`-L1 Jacobi smoother through a geometric multigrid hierarchy
//! (`AbsL1GeometricMultigrid` + `ParFiniteElementSpaceHierarchy`, serial at one
//! rank).  This placeholder keeps the Cargo registration compiling while the
//! space-hierarchy multigrid builder and the driver are written; it refuses to
//! run rather than emit a wrong number.

fn main() {
    eprintln!(
        "diag_mg_abs_l1_jacobi: NOT PORTED YET — D376 in progress; this stub only keeps the \
         Cargo registration compiling.  C++ reference: MFEM 4.10 \
         miniapps/diag-smoothers/mg-abs-l1-jacobi.cpp.  Exiting with status 3."
    );
    std::process::exit(3);
}
