//! TargetConstructor for TMOP (Target-Matrix Optimization Paradigm).
//!
//! Ported from MFEM's `fem/tmop.hpp` (class TargetConstructor).
//!
//! TargetConstructor builds the target Jacobian matrix Jtr (reference → target)
//! for each quadrature point in each element. The target matrix defines the
//! "ideal" shape/size that the mesh optimization should achieve.

use crate::tmop::invariants::{InvariantsEvaluator2D, InvariantsEvaluator3D};

/// Target-matrix construction algorithms (matches MFEM's TargetType).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetType {
    /// Ideal shape, unit size; nodes are not used.
    IdealShapeUnitSize,
    /// Ideal shape, equal size/volume; nodes define total target volume.
    IdealShapeEqualSize,
    /// Ideal shape, given size/volume; nodes define target volume at each quad point.
    IdealShapeGivenSize,
    /// Given shape, given size/volume; nodes define exact target Jacobian.
    GivenShapeAndSize,
    /// Full target tensor specified at every quadrature point.
    GivenFull,
}

/// TargetConstructor builds target Jacobian matrices for TMOP.
///
/// This is the serial version. For parallel use, a ParTargetConstructor
/// would be needed (MFEM has ParTargetConstructor for MPI).
#[derive(Debug, Clone)]
pub struct TargetConstructor {
    target_type: TargetType,
    /// Volume scale factor (used by IdealShapeEqualSize).
    volume_scale: f64,
    /// Average volume (computed from mesh).
    avg_volume: f64,
    /// Whether this has been initialized with mesh volume info.
    initialized: bool,
}

impl TargetConstructor {
    /// Create a new TargetConstructor with the given target type.
    pub fn new(target_type: TargetType) -> Self {
        Self {
            target_type,
            volume_scale: 1.0,
            avg_volume: 0.0,
            initialized: false,
        }
    }

    /// Get the target type.
    pub fn target_type(&self) -> TargetType {
        self.target_type
    }

    /// Set the volume scale factor (used by IdealShapeEqualSize).
    pub fn set_volume_scale(&mut self, scale: f64) {
        self.volume_scale = scale;
    }

    /// Initialize with average volume (must be called before ComputeElementTargets
    /// for target types that use volume info).
    pub fn set_avg_volume(&mut self, avg_volume: f64) {
        self.avg_volume = avg_volume;
        self.initialized = true;
    }

    /// Compute the target Jacobian matrix for a 2D element at a single point.
    ///
    /// `ideal_jac` is the ideal-shape Jacobian for the element geometry
    /// (from Geometries.GetGeomToPerfGeomJac).
    ///
    /// Returns the target Jacobian Jtr (2x2 matrix in column-major [f64; 4]).
    pub fn compute_element_target_2d(
        &self,
        ideal_jac: &[f64; 4],
        node_pos: Option<&[f64]>, // [dof*2] column-major node positions
        det_jpr: Option<f64>,      // det of the physical Jacobian (for GivenSize)
    ) -> [f64; 4] {
        match self.target_type {
            TargetType::IdealShapeUnitSize => {
                // Jtr = Wideal (ideal shape, unit size)
                *ideal_jac
            }
            TargetType::IdealShapeEqualSize => {
                // Jtr = c * Wideal, where c = (volume_scale * avg_volume / det(Wideal))^(1/dim)
                assert!(self.initialized, "Must call set_avg_volume first");
                let det_w = det_2x2(ideal_jac);
                let scale = (self.volume_scale * self.avg_volume / det_w).powf(1.0 / 2.0);
                let mut result = [0.0; 4];
                for i in 0..4 {
                    result[i] = scale * ideal_jac[i];
                }
                result
            }
            TargetType::IdealShapeGivenSize => {
                // Jtr = (det(Jpr) / det(Wideal))^(1/dim) * Wideal
                let det_w = det_2x2(ideal_jac);
                let det_j = det_jpr.expect("det_jpr required for IdealShapeGivenSize");
                let scale = (det_j / det_w).powf(1.0 / 2.0);
                let mut result = [0.0; 4];
                for i in 0..4 {
                    result[i] = scale * ideal_jac[i];
                }
                result
            }
            TargetType::GivenShapeAndSize => {
                // Jtr = Jpr (from node positions)
                // node_pos is [dof*2] column-major
                let pos = node_pos.expect("node_pos required for GivenShapeAndSize");
                // For a single element, Jtr = pos^T * dshape (but here we just return pos as-is)
                // This is a simplified version - full version needs element shape functions
                let mut result = [0.0; 4];
                result.copy_from_slice(&pos[..4]);
                result
            }
            TargetType::GivenFull => {
                // Jtr is provided externally (not implemented here)
                panic!("GivenFull requires external matrix coefficient");
            }
        }
    }

    /// Compute the target Jacobian matrix for a 3D element at a single point.
    pub fn compute_element_target_3d(
        &self,
        ideal_jac: &[f64; 9],
        node_pos: Option<&[f64]>,
        det_jpr: Option<f64>,
    ) -> [f64; 9] {
        match self.target_type {
            TargetType::IdealShapeUnitSize => {
                *ideal_jac
            }
            TargetType::IdealShapeEqualSize => {
                assert!(self.initialized, "Must call set_avg_volume first");
                let det_w = det_3x3(ideal_jac);
                let scale = (self.volume_scale * self.avg_volume / det_w).powf(1.0 / 3.0);
                let mut result = [0.0; 9];
                for i in 0..9 {
                    result[i] = scale * ideal_jac[i];
                }
                result
            }
            TargetType::IdealShapeGivenSize => {
                let det_w = det_3x3(ideal_jac);
                let det_j = det_jpr.expect("det_jpr required for IdealShapeGivenSize");
                let scale = (det_j / det_w).powf(1.0 / 3.0);
                let mut result = [0.0; 9];
                for i in 0..9 {
                    result[i] = scale * ideal_jac[i];
                }
                result
            }
            TargetType::GivenShapeAndSize => {
                let pos = node_pos.expect("node_pos required for GivenShapeAndSize");
                let mut result = [0.0; 9];
                result.copy_from_slice(&pos[..9]);
                result
            }
            TargetType::GivenFull => {
                panic!("GivenFull requires external matrix coefficient");
            }
        }
    }

    /// Check if this target type uses physical coordinates.
    pub fn uses_physical_coordinates(&self) -> bool {
        matches!(
            self.target_type,
            TargetType::IdealShapeEqualSize
                | TargetType::IdealShapeGivenSize
                | TargetType::GivenShapeAndSize
                | TargetType::GivenFull
        )
    }

    /// Check if this target type contains volume info.
    pub fn contains_volume_info(&self) -> bool {
        matches!(
            self.target_type,
            TargetType::IdealShapeEqualSize
                | TargetType::IdealShapeGivenSize
                | TargetType::GivenShapeAndSize
        )
    }
}

/// Compute determinant of a 2x2 matrix (column-major).
fn det_2x2(m: &[f64; 4]) -> f64 {
    m[0] * m[3] - m[1] * m[2]
}

/// Compute determinant of a 3x3 matrix (column-major).
fn det_3x3(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Get the ideal-shape Jacobian for a geometry type (simplified).
/// In MFEM this comes from Geometries.GetGeomToPerfGeomJac.
pub fn ideal_shape_jac_2d(geom_type: &str) -> [f64; 4] {
    match geom_type {
        "TRIANGLE" => {
            // Ideal triangle: equilateral with unit area
            // Wideal = [1, 0.5; 0, sqrt(3)/2] (column-major)
            [1.0, 0.0, 0.5, 3.0_f64.sqrt() / 2.0]
        }
        "QUADRILATERAL" => {
            // Ideal quad: unit square
            [1.0, 0.0, 0.0, 1.0]
        }
        _ => [1.0, 0.0, 0.0, 1.0],
    }
}

/// Get the ideal-shape Jacobian for a 3D geometry type.
pub fn ideal_shape_jac_3d(geom_type: &str) -> [f64; 9] {
    match geom_type {
        "TETRAHEDRON" => {
            // Ideal tet: regular tetrahedron with unit volume
            // Wideal = [1, 0.5, 0.5; 0, sqrt(3)/2, sqrt(3)/6; 0, 0, sqrt(6)/3]
            let s3 = 3.0_f64.sqrt();
            let s6 = 6.0_f64.sqrt();
            [
                1.0, 0.0, 0.0,
                0.5, s3 / 2.0, 0.0,
                0.5, s3 / 6.0, s6 / 3.0,
            ]
        }
        "HEXAHEDRON" => {
            // Ideal hex: unit cube
            [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ]
        }
        _ => [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ideal_shape_unit_size_2d() {
        let tc = TargetConstructor::new(TargetType::IdealShapeUnitSize);
        let ideal = ideal_shape_jac_2d("TRIANGLE");
        let jtr = tc.compute_element_target_2d(&ideal, None, None);
        assert_eq!(jtr, ideal);
    }

    #[test]
    fn test_ideal_shape_equal_size_2d() {
        let mut tc = TargetConstructor::new(TargetType::IdealShapeEqualSize);
        tc.set_avg_volume(2.0); // avg volume = 2.0
        let ideal = ideal_shape_jac_2d("TRIANGLE");
        let jtr = tc.compute_element_target_2d(&ideal, None, None);
        // det(Wideal) = sqrt(3)/2 ≈ 0.866
        // scale = (1.0 * 2.0 / 0.866)^(1/2) ≈ 1.52
        let det_w = det_2x2(&ideal);
        let expected_scale = (2.0 / det_w).powf(0.5);
        let mut expected = [0.0; 4];
        for i in 0..4 {
            expected[i] = expected_scale * ideal[i];
        }
        for i in 0..4 {
            assert!((jtr[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ideal_shape_unit_size_3d() {
        let tc = TargetConstructor::new(TargetType::IdealShapeUnitSize);
        let ideal = ideal_shape_jac_3d("TETRAHEDRON");
        let jtr = tc.compute_element_target_3d(&ideal, None, None);
        assert_eq!(jtr, ideal);
    }

    #[test]
    fn test_ideal_shape_equal_size_3d() {
        let mut tc = TargetConstructor::new(TargetType::IdealShapeEqualSize);
        tc.set_avg_volume(1.0);
        let ideal = ideal_shape_jac_3d("TETRAHEDRON");
        let jtr = tc.compute_element_target_3d(&ideal, None, None);
        let det_w = det_3x3(&ideal);
        let expected_scale = (1.0 / det_w).powf(1.0 / 3.0);
        let mut expected = [0.0; 9];
        for i in 0..9 {
            expected[i] = expected_scale * ideal[i];
        }
        for i in 0..9 {
            assert!((jtr[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_uses_physical_coordinates() {
        assert!(!TargetConstructor::new(TargetType::IdealShapeUnitSize).uses_physical_coordinates());
        assert!(TargetConstructor::new(TargetType::IdealShapeEqualSize).uses_physical_coordinates());
        assert!(TargetConstructor::new(TargetType::IdealShapeGivenSize).uses_physical_coordinates());
        assert!(TargetConstructor::new(TargetType::GivenShapeAndSize).uses_physical_coordinates());
    }
}
