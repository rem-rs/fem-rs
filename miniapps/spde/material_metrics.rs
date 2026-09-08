//! MFEM `miniapps/spde/material_metrics.{hpp,cpp}` port.
//!
//! Distance metrics defining the topological support: an octet-truss lattice
//! or randomly placed ellipsoidal particles inside the unit cube.

/// Distance between a point and a segment (Edge::GetDistanceTo).
///
/// Implements the formula used in [1, Example 5] (SPDE miniapp README).
fn edge_distance_to(start: &[f64; 3], end: &[f64; 3], x: &[f64; 3]) -> f64 {
    let a = dist(start, x);
    let b = dist(end, x);
    let c = dist(start, end);
    let s1 = (a * a + b * b) / 2.0;
    let s2 = c * c / 4.0;
    let s3 = ((a * a - b * b) / (2.0 * c)).powi(2);
    (s1 - s2 - s3).abs().sqrt()
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Compute the metric rho describing the material topology.
pub trait MaterialTopology {
    fn compute_metric(&self, x: &[f64; 3]) -> f64;
}

/// Randomly placed ellipsoidal particles (ParticleTopology).
pub struct ParticleTopology {
    /// A_k · x_k (scaled positions).
    particle_positions: Vec<[f64; 3]>,
    /// Random rotations of the shape (A_k = R·diag(shape)·Rᵀ).
    particle_orientations: Vec<[[f64; 3]; 3]>,
    number_of_particles: usize,
}

impl ParticleTopology {
    /// The length of `random_positions` and `random_rotations` must be 3× and
    /// 9× the number of particles, respectively.
    pub fn new(
        length_x: f64,
        length_y: f64,
        length_z: f64,
        random_positions: &[f64],
        random_rotations: &[f64],
    ) -> Self {
        let number_of_particles = random_positions.len() / 3;
        let shape = [length_x, length_y, length_z];
        let mut particle_positions = Vec::with_capacity(number_of_particles);
        let mut particle_orientations = Vec::with_capacity(number_of_particles);
        for i in 0..number_of_particles {
            let idx_pos = i * 3;
            let particle_position = [
                random_positions[idx_pos],
                random_positions[idx_pos + 1],
                random_positions[idx_pos + 2],
            ];

            let idx_rot = i * 9;
            // Row-major rotation matrix.
            let r = [
                [random_rotations[idx_rot], random_rotations[idx_rot + 1], random_rotations[idx_rot + 2]],
                [random_rotations[idx_rot + 3], random_rotations[idx_rot + 4], random_rotations[idx_rot + 5]],
                [random_rotations[idx_rot + 6], random_rotations[idx_rot + 7], random_rotations[idx_rot + 8]],
            ];

            // res = R · diag(shape) · Rᵀ  (MFEM MultADBt with D = diag(shape)).
            let mut res = [[0.0f64; 3]; 3];
            for (a, row) in res.iter_mut().enumerate() {
                for (b, entry) in row.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for kk in 0..3 {
                        sum += r[a][kk] * shape[kk] * r[b][kk];
                    }
                    *entry = sum;
                }
            }
            particle_orientations.push(res);

            // scaled_position = res · particle_position.
            let mut scaled = [0.0f64; 3];
            for (a, entry) in scaled.iter_mut().enumerate() {
                *entry = res[a][0] * particle_position[0]
                    + res[a][1] * particle_position[1]
                    + res[a][2] * particle_position[2];
            }
            particle_positions.push(scaled);
        }
        Self { particle_positions, particle_orientations, number_of_particles }
    }
}

impl MaterialTopology for ParticleTopology {
    fn compute_metric(&self, x: &[f64; 3]) -> f64 {
        let mut min_dist = f64::INFINITY;
        for i in 0..self.number_of_particles {
            let o = &self.particle_orientations[i];
            let y = [
                o[0][0] * x[0] + o[0][1] * x[1] + o[0][2] * x[2],
                o[1][0] * x[0] + o[1][1] * x[1] + o[1][2] * x[2],
                o[2][0] * x[0] + o[2][1] * x[1] + o[2][2] * x[2],
            ];
            let d = dist(&self.particle_positions[i], &y);
            if d < min_dist {
                min_dist = d;
            }
        }
        min_dist
    }
}

/// Octet-truss lattice (OctetTrussTopology).
pub struct OctetTrussTopology {
    /// The edges of the truss (pairs of points).
    edges: Vec<([f64; 3], [f64; 3])>,
}

impl OctetTrussTopology {
    pub fn new() -> Self {
        // Outer structure.
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [0.0, 1.0, 1.0];
        let p3 = [1.0, 0.0, 1.0];
        let p4 = [1.0, 1.0, 0.0];

        // Inner structure.
        let p5 = [0.0, 0.5, 0.5]; // left
        let p6 = [1.0, 0.5, 0.5]; // right
        let p7 = [0.5, 0.0, 0.5]; // bottom
        let p8 = [0.5, 1.0, 0.5]; // top
        let p9 = [0.5, 0.5, 0.0]; // front
        let p10 = [0.5, 0.5, 1.0]; // back

        let corners = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10];
        let mut edges = Vec::new();

        // Outer edges: all pairs of the 4 corner points.
        for i in 0..4 {
            for j in (i + 1)..4 {
                edges.push((corners[i], corners[j]));
            }
        }

        // Inner edges: p5/p6 to p7..p10, plus the p7..p10 cycle.
        let inner = [
            (4, 6), (4, 7), (4, 8), (4, 9), // p5→p7..p10
            (5, 6), (5, 7), (5, 8), (5, 9), // p6→p7..p10
            (6, 8), (6, 9), (7, 8), (7, 9), // p7..p10 cycle
        ];
        for (a, b) in inner {
            edges.push((corners[a], corners[b]));
        }
        Self { edges }
    }
}

impl Default for OctetTrussTopology {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialTopology for OctetTrussTopology {
    fn compute_metric(&self, x: &[f64; 3]) -> f64 {
        // Edges of the outer structure connecting periodic points.
        const PERIODIC_EDGES: usize = 6;

        // 1. x and its ghost points mimicking the periodicity on [0,1]³.
        let mut periodic_points: Vec<[f64; 3]> = vec![*x];
        let dirs: [[f64; 3]; 6] = [
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        ];
        for d in &dirs {
            periodic_points.push([x[0] + d[0], x[1] + d[1], x[2] + d[2]]);
        }

        // 2. Distance from each periodic point to the outer edges.
        let mut min_dist = f64::INFINITY;
        for point in &periodic_points {
            for edge in self.edges.iter().take(PERIODIC_EDGES) {
                let d = edge_distance_to(&edge.0, &edge.1, point);
                if d < min_dist {
                    min_dist = d;
                }
            }
        }

        // 3. Distance between x and the remaining inner edges.
        for edge in self.edges.iter().skip(PERIODIC_EDGES) {
            let d = edge_distance_to(&edge.0, &edge.1, x);
            if d < min_dist {
                min_dist = d;
            }
        }
        min_dist
    }
}
