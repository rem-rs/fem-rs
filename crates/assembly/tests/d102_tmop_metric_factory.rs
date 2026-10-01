//! Round-102: the `-mid` factory covers the full MFEM 4.10 mesh-optimizer
//! metric zoo and the shared min-det binding reaches the delta metrics.

use fem_assembly::tmop_form::{metric_from_id_2d, metric_from_id_3d, SharedMinDet, TmopMetric};

/// All 2D ids the MFEM drivers accept (mesh-optimizer + tmop-check-metric).
const IDS_2D: &[i32] = &[
    0, 1, 2, 4, 7, 9, 11, 14, 22, 36, 49, 50, 51, 55, 56, 58, 66, 77, 80, 85, 90, 94, 98, 107,
    126, 211, 252,
];

/// All 3D ids the MFEM drivers accept.
const IDS_3D: &[i32] = &[
    301, 302, 303, 304, 311, 313, 315, 316, 318, 321, 322, 323, 328, 332, 333, 334, 338, 342,
    347, 352, 360,
];

#[test]
fn d102_factory_covers_full_zoo() {
    let min_det = SharedMinDet::new(-0.1);
    for id in IDS_2D {
        assert!(
            matches!(metric_from_id_2d(*id, &min_det), Some(TmopMetric::D2(_))),
            "2D factory missing id {id}"
        );
    }
    for id in IDS_3D {
        assert!(
            matches!(metric_from_id_3d(*id, &min_det), Some(TmopMetric::D3(_))),
            "3D factory missing id {id}"
        );
    }
    // Unknown ids stay unknown (mesh-optimizer aborts on them in C++).
    assert!(metric_from_id_2d(999, &min_det).is_none());
    assert!(metric_from_id_3d(999, &min_det).is_none());
}

#[test]
fn d102_factory_shared_min_det_reaches_delta_metrics() {
    // 252/313/352 bind tau0/min_detT by reference in MFEM; the solver updates
    // the shared cell and the metric must observe the update live.
    let min_det = SharedMinDet::new(-0.1);
    let m252 = metric_from_id_2d(252, &min_det);
    let m352 = metric_from_id_3d(352, &min_det);

    let jpt2 = [[1.2, 0.1], [-0.05, 0.95]];
    let jpt3 = [[1.1, 0.1, 0.0], [-0.02, 0.98, 0.3], [0.05, -0.1, 1.05]];

    if let Some(TmopMetric::D2(m)) = &m252 {
        let w_before = m.eval_w(&jpt2);
        min_det.set(-0.5);
        let w_after = m.eval_w(&jpt2);
        assert!(
            (w_before - w_after).abs() > 1e-6,
            "metric 252 does not observe the shared min-det update"
        );
    } else {
        panic!("factory returned wrong dimension for 252");
    }
    if let Some(TmopMetric::D3(m)) = &m352 {
        let w_before = m.eval_w(&jpt3);
        min_det.set(0.3);
        let w_after = m.eval_w(&jpt3);
        assert!(
            (w_before - w_after).abs() > 1e-6,
            "metric 352 does not observe the shared min-det update"
        );
    } else {
        panic!("factory returned wrong dimension for 352");
    }

    // Metric 22 (pre-existing) keeps the same behavior.
    let min_det22 = SharedMinDet::new(0.0);
    if let Some(TmopMetric::D2(m)) = metric_from_id_2d(22, &min_det22) {
        let w0 = m.eval_w(&jpt2);
        min_det22.set(-0.3);
        assert!(
            (m.eval_w(&jpt2) - w0).abs() > 1e-6,
            "metric 22 does not observe the shared min-det update"
        );
    } else {
        panic!("factory returned wrong dimension for 22");
    }
}

#[test]
fn d102_factory_new_ids_energy_sanity() {
    // Each round-102 id evaluates to a finite, non-NaN energy at an
    // untangled point and at the identity.
    let min_det = SharedMinDet::new(-0.1);
    let id_jpt2 = [[1.0, 0.0], [0.0, 1.0]];
    let id_jpt3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let sk_jpt2 = [[0.9, 0.2], [-0.1, 1.2]];
    let sk_jpt3 = [[1.2, 0.2, -0.1], [-0.05, 0.9, 0.3], [0.1, -0.2, 1.1]];
    for id in IDS_2D {
        if let Some(TmopMetric::D2(m)) = metric_from_id_2d(*id, &min_det) {
            for (label, jpt) in [("id", id_jpt2), ("sk", sk_jpt2)] {
                let w = m.eval_w(&jpt);
                assert!(w.is_finite(), "2D id {id}: EvalW({label}) = {w}");
            }
        }
    }
    for id in IDS_3D {
        if let Some(TmopMetric::D3(m)) = metric_from_id_3d(*id, &min_det) {
            for (label, jpt) in [("id", id_jpt3), ("sk", sk_jpt3)] {
                let w = m.eval_w(&jpt);
                assert!(w.is_finite(), "3D id {id}: EvalW({label}) = {w}");
            }
        }
    }
}
