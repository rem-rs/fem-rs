//! TMOP (Target-Matrix Optimization Paradigm) quality metrics.
//!
//! Ported from MFEM's `fem/tmop.hpp` and `fem/tmop.cpp`.
//!
//! This module provides:
//! - [`InvariantsEvaluator2D`] / [`InvariantsEvaluator3D`] — invariant evaluators
//! - [`TmopQualityMetric`] trait — interface for quality metrics
//! - Concrete metric implementations (2D: 001/002/007/009/014/022/050/055/056/058/077,
//!   A-metrics: 014/050, 3D: 301/302/303/304/315/316/318/321/323/360)
//! - [`tmop_check_metric`] — verification routine matching MFEM's tmop-check-metric

pub mod invariants;
pub mod metrics;
pub mod target;
pub mod check;
pub mod integrator;

pub use invariants::{InvariantsEvaluator2D, InvariantsEvaluator3D};
pub use metrics::{
    TmopQualityMetric, TmopMetric001, TmopMetric002, TmopMetric007, TmopMetric009,
    TmopMetric014, TmopMetric022, TmopMetric050, TmopMetric055, TmopMetric056,
    TmopMetric058, TmopMetric077, TmopAMetric014, TmopAMetric050,
    TmopMetric301, TmopMetric302, TmopMetric303, TmopMetric304,
    TmopMetric315, TmopMetric316, TmopMetric318, TmopMetric321, TmopMetric323, TmopMetric360,
    TmopQualityMetric3D,
};
