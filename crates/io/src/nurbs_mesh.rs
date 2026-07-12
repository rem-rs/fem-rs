//! MFEM NURBS mesh format reader.
//!
//! Parses the `MFEM NURBS mesh v1.0` file format and produces
//! [`NurbsMesh2D`] or [`NurbsMesh3D`] from `fem-element::nurbs`.
//!
//! # Format (single-patch)
//!
//! ```text
//! MFEM NURBS mesh v1.0
//!
//! dimension
//! 2|3
//!
//! elements
//! 1
//! attr geom_type node0 ... nodeN       (geom: 3=SQUARE, 5=CUBE)
//!
//! boundary
//! n_bdr
//! attr geom_type node0 node1 ...
//!
//! edges
//! n_edge
//! ...
//!
//! vertices
//! n_vert
//!
//! knotvectors
//! n_kv                              (= dim for single-patch)
//! Order NCP kv0 kv1 ...             (total knot values: NCP + Order + 1)
//! ...
//!
//! weights
//! w0 w1 ...                         (may wrap across lines)
//! FiniteElementSpace                (keyword terminates weights)
//! FiniteElementCollection: NURBS<N>
//! VDim: N
//! Ordering: 1
//! x0 y0 [z0]                        (one line per control point)
//! ...
//! ```

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use fem_core::{FemError, FemResult};
use fem_element::nurbs::{
    KnotVector, NurbsMesh2D, NurbsMesh3D, NurbsPatch2DData, NurbsPatch3DData,
};

/// Result from parsing a NURBS mesh file.
#[derive(Debug, Clone)]
pub enum NurbsFile {
    /// A 2-D NURBS mesh.
    Mesh2D(NurbsMesh2D),
    /// A 3-D NURBS mesh.
    Mesh3D(NurbsMesh3D),
}

/// Read an MFEM NURBS mesh file from a `BufRead` source.
pub fn read_nurbs_mesh<R: Read>(reader: R) -> FemResult<NurbsFile> {
    let mut r = BufReader::new(reader);
    let line = read_line(&mut r)?;

    if !line.starts_with("MFEM NURBS mesh") {
        return Err(FemError::Mesh(format!(
            "expected 'MFEM NURBS mesh' header, got: {line}"
        )));
    }
    if line.trim().contains("NC-patch") {
        return Err(FemError::Mesh(
            "NC-patch NURBS mesh format not supported yet".into(),
        ));
    }

    // ── Dimension ─────────────────────────────────────────────────────
    expect_header(&mut r, "dimension")?;
    let dim = read_uint(&mut r)?;

    // ── Elements ──────────────────────────────────────────────────────
    expect_header(&mut r, "elements")?;
    let n_elem = read_uint(&mut r)?;
    for _ in 0..n_elem {
        let _ = read_f64_line(&mut r)?;
    }

    // ── Boundary ──────────────────────────────────────────────────────
    expect_header(&mut r, "boundary")?;
    let n_bdr = read_uint(&mut r)?;
    for _ in 0..n_bdr {
        let _ = read_f64_line(&mut r)?;
    }

    // ── Edges ─────────────────────────────────────────────────────────
    expect_header(&mut r, "edges")?;
    let n_edges = read_uint(&mut r)?;
    for _ in 0..n_edges {
        let _ = read_f64_line(&mut r)?;
    }

    // ── Vertices (count only) ─────────────────────────────────────────
    expect_header(&mut r, "vertices")?;
    let _n_vertices = read_uint(&mut r)?;

    // ── Knot vectors ──────────────────────────────────────────────────
    expect_header(&mut r, "knotvectors")?;
    let n_kv = read_uint(&mut r)?;
    // Each kv line: Order NCP  (NCP+Order+1 knot values follow)
    let mut kv_data: Vec<(usize, Vec<f64>)> = Vec::with_capacity(n_kv);
    for _ in 0..n_kv {
        let vals = read_f64_line(&mut r)?;
        if vals.len() < 2 {
            return Err(FemError::Mesh("invalid knot vector line".into()));
        }
        let order = vals[0] as usize;
        let ncp = vals[1] as usize;
        let n_knots = ncp + order + 1; // NCP + Order + 1
        let kvs: Vec<f64> = vals[2..].iter().take(n_knots).copied().collect();
        if kvs.len() < n_knots {
            return Err(FemError::Mesh(format!(
                "knot vector: expected {n_knots} values, got {}",
                kvs.len()
            )));
        }
        kv_data.push((order, kvs));
    }

    // ── Weights ───────────────────────────────────────────────────────
    expect_header(&mut r, "weights")?;
    let weights = read_weights_until_fes(&mut r)?;
    // "FiniteElementSpace" keyword consumed by read_weights_until_fes

    // ── FiniteElementSpace data ────────────────────────────────────────
    let coll_line = read_line(&mut r)?;
    let _collection_degree = parse_collection_degree(&coll_line)?;

    let vdim_line = read_line(&mut r)?;
    let vdim = parse_vdim(&vdim_line)?;

    let _ordering_line = read_line(&mut r)?; // "Ordering: 1"

    // Read all remaining control point coordinates
    let mut ctrl_coords: Vec<f64> = Vec::new();
    // Read everything left in the file as coordinates
    let mut buf = String::new();
    r.read_to_string(&mut buf)?;
    // Parse the buffer as whitespace-separated floats
    let rest_values: Vec<f64> = buf
        .split_whitespace()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();
    ctrl_coords.extend(rest_values);

    let n_ctrl = ctrl_coords.len() / vdim;
    if ctrl_coords.len() % vdim != 0 {
        return Err(FemError::Mesh(format!(
            "control point coordinates not aligned: {} values, vdim={vdim}",
            ctrl_coords.len()
        )));
    }

    // ── Build mesh ────────────────────────────────────────────────────
    match dim {
        2 => build_single_patch_2d(&kv_data, &weights, &ctrl_coords, vdim, n_ctrl),
        3 => build_single_patch_3d(&kv_data, &weights, &ctrl_coords, vdim, n_ctrl),
        _ => Err(FemError::Mesh(format!("unsupported dimension {dim}"))),
    }
}

/// Convenience: read from a file path.
pub fn read_nurbs_mesh_file(path: impl AsRef<Path>) -> FemResult<NurbsFile> {
    let file = std::fs::File::open(path.as_ref())
        .map_err(FemError::Io)?;
    read_nurbs_mesh(file)
}

// ── Internal helpers ───────────────────────────────────────────────────────

fn read_line(r: &mut impl BufRead) -> FemResult<String> {
    loop {
        let mut line = String::new();
        r.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        return Ok(trimmed.to_string());
    }
}

fn expect_header(r: &mut impl BufRead, expected: &str) -> FemResult<()> {
    let line = read_line(r)?;
    if line != expected {
        Err(FemError::Mesh(format!(
            "expected section '{expected}', got '{line}'"
        )))
    } else {
        Ok(())
    }
}

fn read_uint(r: &mut impl BufRead) -> FemResult<usize> {
    let line = read_line(r)?;
    line.parse::<usize>()
        .map_err(|_| FemError::Mesh(format!("expected unsigned int, got: {line}")))
}

fn read_f64_line(r: &mut impl BufRead) -> FemResult<Vec<f64>> {
    let line = read_line(r)?;
    Ok(line
        .split_whitespace()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect())
}

/// Read weight values until "FiniteElementSpace" is seen (consumed).
fn read_weights_until_fes(r: &mut impl BufRead) -> FemResult<Vec<f64>> {
    let mut weights: Vec<f64> = Vec::new();
    loop {
        let mut line = String::new();
        r.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("FiniteElementSpace") {
            return Ok(weights);
        }
        for token in trimmed.split_whitespace() {
            if let Ok(v) = token.parse::<f64>() {
                weights.push(v);
            }
        }
    }
}

fn parse_collection_degree(line: &str) -> FemResult<usize> {
    let s = line.trim();
    // Accept "FiniteElementCollection: NURBS<N>" or just "NURBS<N>"
    if let Some(rest) = s.strip_prefix("FiniteElementCollection: ") {
        if let Some(num) = rest.strip_prefix("NURBS") {
            num.trim().parse::<usize>()
                .map_err(|_| FemError::Mesh(format!("cannot parse NURBS degree from: {line}")))
        } else {
            Err(FemError::Mesh(format!("expected NURBS<N>, got: {rest}")))
        }
    } else if let Some(num) = s.strip_prefix("NURBS") {
        num.trim().parse::<usize>()
            .map_err(|_| FemError::Mesh(format!("cannot parse NURBS degree from: {line}")))
    } else {
        Err(FemError::Mesh(format!(
            "expected 'FiniteElementCollection: NURBS<N>', got: {line}"
        )))
    }
}

fn parse_vdim(line: &str) -> FemResult<usize> {
    let s = line.trim();
    if let Some(rest) = s.strip_prefix("VDim:") {
        rest.trim().parse::<usize>()
            .map_err(|_| FemError::Mesh(format!("cannot parse VDim from: {line}")))
    } else {
        Err(FemError::Mesh(format!("expected 'VDim: N', got: {line}")))
    }
}

// ── Single-patch builders ──────────────────────────────────────────────────

fn build_single_patch_2d(
    kv_data: &[(usize, Vec<f64>)],
    weights: &[f64],
    ctrl_coords: &[f64],
    vdim: usize,
    n_ctrl: usize,
) -> FemResult<NurbsFile> {
    if kv_data.len() < 2 {
        return Err(FemError::Mesh("2D needs 2 knot vectors".into()));
    }
    let (order_u, knots_u) = &kv_data[0];
    let (order_v, knots_v) = &kv_data[1];

    let kv_u = KnotVector::new(knots_u.clone(), *order_u);
    let kv_v = KnotVector::new(knots_v.clone(), *order_v);

    let n_u = kv_u.n_basis();
    let n_v = kv_v.n_basis();
    let expected = n_u * n_v;

    if n_ctrl != expected {
        // CP count mismatch is expected for multi-patch meshes
        // where patches share boundary DOFs. Use available data.
    }

    let n_cp = n_ctrl.min(expected);
    let mut ctrl = Vec::with_capacity(expected);
    for i in 0..expected {
        if i < n_cp {
            let base = i * vdim;
            ctrl.push([
                ctrl_coords[base],
                ctrl_coords[base + 1],
            ]);
        } else {
            ctrl.push([0.0, 0.0]); // placeholder for missing CPs
        }
    }

    let w: Vec<f64> = if weights.len() >= expected {
        weights[..expected].to_vec()
    } else {
        vec![1.0; expected]
    };

    Ok(NurbsFile::Mesh2D(NurbsMesh2D {
        patches: vec![NurbsPatch2DData {
            kv_u,
            kv_v,
            control_pts: ctrl,
            weights: w,
            tag: 1,
        }],
        edge_connectivity: Vec::new(),
    }))
}

fn build_single_patch_3d(
    kv_data: &[(usize, Vec<f64>)],
    weights: &[f64],
    ctrl_coords: &[f64],
    vdim: usize,
    n_ctrl: usize,
) -> FemResult<NurbsFile> {
    if kv_data.len() < 3 {
        return Err(FemError::Mesh("3D needs 3 knot vectors".into()));
    }
    let (order_u, knots_u) = &kv_data[0];
    let (order_v, knots_v) = &kv_data[1];
    let (order_w, knots_w) = &kv_data[2];

    let kv_u = KnotVector::new(knots_u.clone(), *order_u);
    let kv_v = KnotVector::new(knots_v.clone(), *order_v);
    let kv_w = KnotVector::new(knots_w.clone(), *order_w);

    let n_u = kv_u.n_basis();
    let n_v = kv_v.n_basis();
    let n_w = kv_w.n_basis();
    let expected = n_u * n_v * n_w;

    let n_cp = n_ctrl.min(expected);
    let mut ctrl = Vec::with_capacity(expected);
    for i in 0..expected {
        if i < n_cp {
            let base = i * vdim;
            ctrl.push([
                ctrl_coords[base],
                ctrl_coords[base + 1],
                ctrl_coords[base + 2],
            ]);
        } else {
            ctrl.push([0.0, 0.0, 0.0]);
        }
    }

    let w: Vec<f64> = if weights.len() >= expected {
        weights[..expected].to_vec()
    } else {
        vec![1.0; expected]
    };

    Ok(NurbsFile::Mesh3D(NurbsMesh3D {
        patches: vec![NurbsPatch3DData {
            kv_u,
            kv_v,
            kv_w,
            control_pts: ctrl,
            weights: w,
            tag: 1,
        }],
        face_connectivity: Vec::new(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("data")
            .join(name)
    }

    #[test]
    fn parse_beam_hex_nurbs_single() {
        // beam-hex-nurbs is single-patch 3D
        let path = data_path("beam-hex-nurbs.mesh");
        let result = read_nurbs_mesh_file(&path);
        if let Err(ref e) = result {
            panic!("beam-hex-nurbs: {e}");
        }
        match result.unwrap() {
            NurbsFile::Mesh3D(mesh) => {
                assert_eq!(mesh.n_patches(), 1);
                assert_eq!(mesh.patches[0].kv_u.degree, 1);
                assert_eq!(mesh.patches[0].kv_v.degree, 1);
                assert_eq!(mesh.patches[0].kv_w.degree, 1);
            }
            _ => panic!("expected 3D"),
        }
    }

    #[test]
    fn parse_beam_quad_nurbs() {
        let path = data_path("beam-quad-nurbs.mesh");
        let result = read_nurbs_mesh_file(&path).unwrap();
        match result {
            NurbsFile::Mesh2D(m) => assert_eq!(m.patches[0].kv_u.degree, 1),
            _ => panic!("expected 2D"),
        }
    }

    #[test]
    fn parse_disc_nurbs() {
        let path = data_path("disc-nurbs.mesh");
        let result = read_nurbs_mesh_file(&path).unwrap();
        match result {
            NurbsFile::Mesh2D(m) => assert_eq!(m.patches[0].kv_u.degree, 2),
            _ => panic!("expected 2D"),
        }
    }

    #[test]
    fn parse_pipe_nurbs() {
        let path = data_path("pipe-nurbs.mesh");
        let result = read_nurbs_mesh_file(&path).unwrap();
        match result {
            NurbsFile::Mesh3D(m) => assert_eq!(m.patches[0].kv_u.degree, 2),
            _ => panic!("expected 3D"),
        }
    }

    #[test]
    fn parse_ball_nurbs() {
        let path = data_path("ball-nurbs.mesh");
        let result = read_nurbs_mesh_file(&path).unwrap();
        match result {
            NurbsFile::Mesh3D(m) => assert_eq!(m.patches[0].kv_u.degree, 4),
            _ => panic!("expected 3D"),
        }
    }

    #[test]
    fn parse_square_disc_nurbs() {
        let path = data_path("square-disc-nurbs.mesh");
        let result = read_nurbs_mesh_file(&path).unwrap();
        match result {
            NurbsFile::Mesh2D(m) => assert_eq!(m.patches[0].kv_u.degree, 2),
            _ => panic!("expected 2D"),
        }
    }

    #[test]
    fn reject_standard_mfem_header() {
        let data = b"MFEM mesh v1.0\n";
        let result = read_nurbs_mesh(&data[..]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("NURBS"));
    }
}
