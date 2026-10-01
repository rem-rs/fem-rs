//! TMOP (Target-Matrix Optimization Paradigm) quality metrics.
//!
//! Ported from MFEM's `fem/tmop.hpp` and `fem/tmop.cpp`.
//!
//! This module provides:
//! - [`InvariantsEvaluator2D`] / [`InvariantsEvaluator3D`] — invariant evaluators
//! - [`ad`] — the native-AD machinery (MFEM `future::dual`, `ADGrad`,
//!   `ADHessian`, `DefaultAssembleH`) used by the AD-based metrics
//! - [`TmopQualityMetric`] trait — interface for quality metrics
//! - Concrete metric implementations (2D: 000/001/002/004/007/009/014/022/050/
//!   055/056/058/066/077/080/085/090/094/098/211/252, A-metrics:
//!   011/014/036/049/050/051/107/126, 3D: 301/302/303/304/311/313/315/316/318/
//!   321/322/323/328/332/333/334/338/342/347/352/360)
//! - [`tmop_check_metric`] — verification routine matching MFEM's tmop-check-metric

pub mod ad;
pub mod invariants;
pub mod metrics;
pub mod target;
pub mod check;
pub mod integrator;

pub use ad::{ad_grad, ad_hessian, default_assemble_h, Ad1, Ad2, AdScalar};
pub use invariants::{InvariantsEvaluator2D, InvariantsEvaluator3D};
pub use metrics::{
    TmopQualityMetric, TmopMetric000, TmopMetric001, TmopMetric002, TmopMetric004,
    TmopMetric007, TmopMetric009, TmopMetric014, TmopMetric022, TmopMetric050, TmopMetric055,
    TmopMetric056, TmopMetric058, TmopMetric066, TmopMetric077, TmopMetric080, TmopMetric085,
    TmopMetric090, TmopMetric094, TmopMetric098, TmopMetric211, TmopMetric252,
    TmopAMetric011, TmopAMetric014, TmopAMetric036, TmopAMetric049, TmopAMetric050,
    TmopAMetric051, TmopAMetric107, TmopAMetric126,
    TmopMetric301, TmopMetric302, TmopMetric303, TmopMetric304, TmopMetric311, TmopMetric313,
    TmopMetric315, TmopMetric316, TmopMetric318, TmopMetric321, TmopMetric322, TmopMetric323,
    TmopMetric328, TmopMetric332, TmopMetric333, TmopMetric334, TmopMetric338, TmopMetric342,
    TmopMetric347, TmopMetric352, TmopMetric360,
    TmopQualityMetric3D,
};
