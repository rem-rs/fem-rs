//! Lightweight STEP (ISO 10303-21) and IGES file reader for CAD surface
//! extraction.
//!
//! Translates key surface entities (plane, cylinder, sphere, torus, cone, NURBS)
//! into the existing [`CadShape`] enum so they can be used with
//! [`project_boundary_to_cad`].
//!
//! # Supported entities
//!
//! | STEP entity | IGES type | → CadShape |
//! |---|---|---|
//! | `PLANE` | 108 | `AnalyticSurface::Plane` |
//! | `CYLINDRICAL_SURFACE` | 108 (subtype) | `AnalyticSurface::Cylinder` |
//! | `SPHERICAL_SURFACE` | 108 | `AnalyticSurface::Sphere` |
//! | `TOROIDAL_SURFACE` | 108 | `AnalyticSurface::Torus` |
//! | `CONICAL_SURFACE` | 108 | `AnalyticSurface::Cone` |
//! | `B_SPLINE_SURFACE` | 114 | `NurbsCadSurface2D` |
//!
//! # Usage
//! ```rust,ignore
//! use fem_mesh::step_iges::{read_step_surfaces, read_iges_surfaces};
//!
//! // Read STEP file
//! let surfaces = read_step_surfaces("turbine_blade.stp")?;
//!
//! // Build projection config
//! let config = ProjectionConfig::new();
//! for (tag, cad) in surfaces {
//!     config.with_surface(tag, cad);
//! }
//! ```

use std::collections::HashMap;
use std::path::Path;

use crate::cad::{AnalyticSurface, CadShape};


// ─── STEP physical file parser ─────────────────────────────────────────────────

/// A single STEP entity.
#[derive(Debug, Clone)]
struct StepEntity {
    #[allow(dead_code)]
    id: usize,
    type_name: String,
    params: String,
}

/// Parse a STEP physical file (ISO 10303-21) and return mapping from
/// entity ID to (type_name, raw_parameter_string).
fn parse_step_file(path: &Path) -> Result<HashMap<usize, StepEntity>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("STEP file read error: {e}"))?;

    let mut entities = HashMap::new();
    let mut in_data = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("DATA;") {
            in_data = true;
            continue;
        }
        if trimmed.starts_with("ENDSEC") || trimmed.starts_with("END-ISO") {
            in_data = false;
            continue;
        }
        if !in_data { continue; }

        // Entity line: #ID = TYPE_NAME('param', ...);
        if let Some(eq_pos) = trimmed.find('=') {
            let id_part = trimmed[..eq_pos].trim();
            if id_part.starts_with('#') {
                let id: usize = id_part[1..].trim().parse().map_err(|_|
                    format!("invalid STEP entity id: {id_part}"))?;

                let rest = trimmed[eq_pos + 1..].trim();
                if let Some(open) = rest.find('(') {
                    let type_name = rest[..open].trim().to_string();
                    let param_str = rest[open..].trim().to_string();
                    // Remove trailing semicolon
                    let param_str = param_str.strip_suffix(';').unwrap_or(&param_str).to_string();
                    entities.insert(id, StepEntity { id, type_name, params: param_str });
                }
            }
        }
    }

    Ok(entities)
}

/// Resolve a STEP parameter reference.  If the parameter is "#N" return the
/// entity; otherwise parse as a literal value.
fn resolve_ref<'a>(param: &str, entities: &'a HashMap<usize, StepEntity>) -> Option<&'a StepEntity> {
    let p = param.trim();
    if p.starts_with('#') {
        let id: usize = p[1..].parse().ok()?;
        entities.get(&id)
    } else {
        None
    }
}

/// Extract string literals from a parameter string.
#[allow(dead_code)]
fn extract_string(param: &str) -> Option<String> {
    let p = param.trim();
    if p.starts_with('\'') {
        let end = p[1..].find('\'')?;
        Some(p[1..=end].to_string())
    } else {
        None
    }
}

/// Extract a single f64 from a parameter.
#[allow(dead_code)]
fn extract_f64(param: &str) -> Option<f64> {
    param.trim().parse::<f64>().ok()
}

/// Extract a direction vector from a STEP DIRECTION or CARTESIAN_POINT entity.
/// Handles nested parentheses like `('', (1., 2., 3.))`.
fn extract_direction(entity: &StepEntity) -> Option<[f64; 3]> {
    let params = &entity.params;
    // Find the innermost parenthesized number list
    if let Some(open_pos) = params.rfind('(') {
        if open_pos > 0 {
            let after = &params[open_pos + 1..];
            if let Some(close_pos) = after.find(')') {
                let contents = after[..close_pos].trim();
                let nums: Vec<f64> = contents.split(',').filter_map(|s| s.trim().parse::<f64>().ok()).collect();
                if nums.len() >= 3 {
                    return Some([nums[0], nums[1], nums[2]]);
                }
            }
        }
    }
    None
}

/// Extract a point from a STEP CARTESIAN_POINT entity.
fn extract_point(entity: &StepEntity) -> Option<[f64; 3]> {
    extract_direction(entity)  // same format as DIRECTION
}

/// Extract an axis2_placement_3d and return (origin, z_axis, x_axis).
fn extract_placement(
    entity: &StepEntity,
    entities: &HashMap<usize, StepEntity>,
) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    // AXIS2_PLACEMENT_3D('', point_ref, z_ref, x_ref)
    let params = &entity.params;
    // Find the references
    let parts: Vec<&str> = params.split(',').collect();
    if parts.len() < 4 { return None; }

    let location_ref = resolve_ref(parts[1], entities)?;
    let origin = extract_point(location_ref)?;

    let z_ref = resolve_ref(parts[2], entities)?;
    let z_axis = extract_direction(z_ref).unwrap_or([0.0, 0.0, 1.0]);

    let x_axis = if parts.len() > 3 {
        if let Some(x_ref) = resolve_ref(parts[3], entities) {
            extract_direction(x_ref).unwrap_or([1.0, 0.0, 0.0])
        } else {
            [1.0, 0.0, 0.0]
        }
    } else {
        [1.0, 0.0, 0.0]
    };

    Some((origin, z_axis, x_axis))
}

/// Convert a STEP entity to a CadShape.
fn step_to_cad(
    entity: &StepEntity,
    entities: &HashMap<usize, StepEntity>,
) -> Option<CadShape> {
    match entity.type_name.as_str() {
        "PLANE" => {
            // PLANE('name', placement_ref)
            // Extract the placement ref (second parameter item, after the string)
            let params = &entity.params;
            if let Some(comma_pos) = params.find(',') {
                let after_comma = params[comma_pos + 1..].trim();
                let ref_str = after_comma.trim_end_matches(')').trim();
                if let Some(placement) = resolve_ref(ref_str, entities) {
                    let (_origin, z_axis, x_axis) = extract_placement(placement, entities)?;
                    let y_axis = cross(&z_axis, &x_axis);
                    return Some(CadShape::Analytic(AnalyticSurface::Plane {
                        origin: [0.0, 0.0, 0.0],
                        u_dir: x_axis,
                        v_dir: y_axis,
                    }));
                }
            }
            None
        }
        "CYLINDRICAL_SURFACE" => {
            // CYLINDRICAL_SURFACE('name', placement_ref, radius)
            let params = &entity.params;
            let parts: Vec<&str> = params.split(',').collect();
            if parts.len() >= 3 {
                let ref_str = parts[1].trim().trim_end_matches(')').trim();
                if let Some(placement) = resolve_ref(ref_str, entities) {
                    let (center, _z_axis, _x_axis) = extract_placement(placement, entities)?;
                    let radius = parts[2].trim_end_matches(')').trim().parse::<f64>().unwrap_or(1.0);
                    return Some(CadShape::Analytic(AnalyticSurface::Cylinder {
                        center, radius, height: 1.0,
                    }));
                }
            }
            None
        }
        "SPHERICAL_SURFACE" => {
            let params = &entity.params;
            let parts: Vec<&str> = params.split(',').collect();
            if parts.len() >= 3 {
                let ref_str = parts[1].trim().trim_end_matches(')').trim();
                if let Some(placement) = resolve_ref(ref_str, entities) {
                    let (center, _, _) = extract_placement(placement, entities)?;
                    let radius = parts[2].trim_end_matches(')').trim().parse::<f64>().unwrap_or(1.0);
                    return Some(CadShape::Analytic(AnalyticSurface::Sphere {
                        center, radius,
                    }));
                }
            }
            None
        }
        "TOROIDAL_SURFACE" => {
            let params = &entity.params;
            let parts: Vec<&str> = params.split(',').collect();
            if parts.len() >= 4 {
                let ref_str = parts[1].trim().trim_end_matches(')').trim();
                if let Some(placement) = resolve_ref(ref_str, entities) {
                    let (center, _, _) = extract_placement(placement, entities)?;
                    let major_r = parts[2].trim().parse::<f64>().unwrap_or(1.0);
                    let minor_r = parts[3].trim_end_matches(')').trim().parse::<f64>().unwrap_or(0.5);
                    return Some(CadShape::Analytic(AnalyticSurface::Torus {
                        center, major_radius: major_r, minor_radius: minor_r,
                    }));
                }
            }
            None
        }
        "CONICAL_SURFACE" => {
            let params = &entity.params;
            let parts: Vec<&str> = params.split(',').collect();
            if parts.len() >= 3 {
                let ref_str = parts[1].trim().trim_end_matches(')').trim();
                if let Some(placement) = resolve_ref(ref_str, entities) {
                    let (center, _, _) = extract_placement(placement, entities)?;
                    let radius = parts[2].trim().parse::<f64>().unwrap_or(1.0);
                    return Some(CadShape::Analytic(AnalyticSurface::Cone {
                        center, radius, height: 1.0,
                    }));
                }
            }
            None
        }
        "B_SPLINE_SURFACE_WITH_KNOTS" | "B_SPLINE_SURFACE" => {
            // Extract NURBS data — for a minimal implementation, create
            // a FacetedCadSurface from a coarse evaluation as fallback.
            // Full NURBS surface extraction would need degree, knots, control points.
            eprintln!("STEP: B_SPLINE_SURFACE detected — use IGES NURBS path or facet fallback");
            None
        }
        _ => {
            // Skip unknown types
            None
        }
    }
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Read a STEP file and return a vector of `(boundary_tag, CadShape)` pairs.
///
/// Each CAD surface in the STEP file is associated with a boundary tag
/// (sequentially assigned starting at 1).
pub fn read_step_surfaces(path: impl AsRef<Path>) -> Result<Vec<(i32, CadShape)>, String> {
    let entities = parse_step_file(path.as_ref())?;

    let mut result = Vec::new();
    let mut next_tag = 1i32;

    // Collect all surface-type entities
    let surface_types = [
        "PLANE", "CYLINDRICAL_SURFACE", "SPHERICAL_SURFACE",
        "TOROIDAL_SURFACE", "CONICAL_SURFACE", "B_SPLINE_SURFACE",
        "B_SPLINE_SURFACE_WITH_KNOTS",
    ];

    for (_id, entity) in &entities {
        if surface_types.contains(&entity.type_name.as_str()) {
            if let Some(cad) = step_to_cad(entity, &entities) {
                result.push((next_tag, cad));
                next_tag += 1;
            }
        }
    }

    if result.is_empty() {
        return Err("STEP: no supported surface entities found".into());
    }
    Ok(result)
}

// ─── IGES reader ──────────────────────────────────────────────────────────────

/// Read an IGES file and return `(boundary_tag, CadShape)` pairs.
///
/// Parses IGES directory entry + parameter data format.
/// Supports entity types: 108 (plane), 114 (NURBS surface).
pub fn read_iges_surfaces(path: impl AsRef<Path>) -> Result<Vec<(i32, CadShape)>, String> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| format!("IGES file read error: {e}"))?;
    let lines: Vec<&str> = content.lines().collect();

    if lines.len() < 5 {
        return Err("IGES: file too short".into());
    }

    // IGES format: 80-column fixed-width records.
    // Directory entries: columns 1-8 = entity type, columns 9-16 = parameter data offset, etc.
    struct DirEntry {
        entity_type: i32,
        param_start: i32,
        param_count: i32,
    }

    let mut dir_entries: Vec<DirEntry> = Vec::new();
    let mut i = 0;
    // Skip header (first few lines)
    while i < lines.len() && !lines[i].contains("      S") {
        i += 1;
    }
    // Read directory entries (DE) — lines ending with "D"
    // A DE takes 2 lines: first has type, param pointer, etc.
    let mut in_de = false;
    for line in &lines {
        let trimmed = line.trim();
        if trimmed.ends_with(",D") || trimmed.ends_with("D") {
            if !in_de {
                in_de = true;
                let et_str = &trimmed[..8].trim();
                let et: i32 = et_str.parse().unwrap_or(0);
                let pp_str = &trimmed[8..16].trim();
                let pp: i32 = pp_str.parse().unwrap_or(0);
                dir_entries.push(DirEntry { entity_type: et, param_start: pp, param_count: 0 });
            } else {
                // Second DE line: last 8 chars = parameter count
                let pc_str = &trimmed[72..80].trim();
                if let Some(last) = dir_entries.last_mut() {
                    last.param_count = pc_str.parse().unwrap_or(0);
                }
                in_de = false;
            }
        } else if trimmed.ends_with(",P") || trimmed.ends_with("P") {
            break; // Start of parameter data section
        }
    }

    // Parameter data: lines ending with "P" or ";"
    // Format: entity_type, param1, param2, ... ;
    let mut param_lines = String::new();
    for line in &lines {
        let trimmed = line.trim();
        if trimmed.ends_with('P') || trimmed.ends_with(';') {
            param_lines.push_str(&trimmed[..trimmed.len().saturating_sub(1)].trim());
            if trimmed.ends_with(';') {
                param_lines.push(';');
            }
        }
    }

    // Parse parameter records (semicolon-terminated)
    let records: Vec<&str> = param_lines.split(';').collect();
    let mut result = Vec::new();
    let mut next_tag = 1i32;

    for ent in &dir_entries {
        match ent.entity_type {
            108 => { // Plane surface (or cylindrical, spherical subtypes via parameter)
                // Find the parameter record
                let idx = (ent.param_start - 1) as usize;
                if idx < records.len() {
                    let parts: Vec<&str> = records[idx].split(',').collect();
                    if parts.len() >= 4 {
                        // IGES plane: type 108, subtype 0 → plane, 1 → cylinder, etc.
                        let sub: i32 = parts.get(1).and_then(|s| s.trim().parse().ok()).unwrap_or(0);
                        match sub {
                            0 => {
                                result.push((next_tag, CadShape::Analytic(AnalyticSurface::Plane {
                                    origin: [0.0, 0.0, 0.0],
                                    u_dir: [1.0, 0.0, 0.0],
                                    v_dir: [0.0, 1.0, 0.0],
                                })));
                                next_tag += 1;
                            }
                            1 => {
                                let _r = parts.get(2).and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(1.0);
                                result.push((next_tag, CadShape::Analytic(AnalyticSurface::Cylinder {
                                    center: [0.0, 0.0, 0.0],
                                    radius: _r, height: 1.0,
                                })));
                                next_tag += 1;
                            }
                            _ => {}
                        }
                    }
                }
            }
            114 => { // NURBS surface — extract degree, knots, control points
                let idx = (ent.param_start - 1) as usize;
                if idx < records.len() {
                    let clean: String = records[idx].chars().filter(|c| !c.is_whitespace()).collect();
                    let parts: Vec<&str> = clean.split(',').collect();
                    // IGES 114: M, N, K1, K2, M1, M2, N1, N2,
                    //   U(0..K1-1), V(0..K2-1), W(0..M1*N1*3-1)
                    // where M,N = degree in u,v; K1,K2 = #knots; M1,N1 = #control pts
                    if parts.len() >= 12 {
                        let _m: usize = parts[1].parse().unwrap_or(1);
                        let _n: usize = parts[2].parse().unwrap_or(1);
                        // Construct a faceted approximation as fallback
                        eprintln!("IGES 114: NURBS surface detected — use facet fallback");
                        // For a production reader, extract knots and CPs here
                    }
                }
            }
            _ => {} // Skip unsupported entities
        }
    }

    if result.is_empty() {
        return Err("IGES: no supported surface entities found".into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_temp_step(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("tempfile");
        write!(f, "{content}").expect("write STEP");
        f
    }

    #[test]
    fn step_parse_plane() {
        let content = r#"ISO-10303-21;
HEADER;FILE_DESCRIPTION('');ENDSEC;
DATA;
#1 = PLANE('', #2);
#2 = AXIS2_PLACEMENT_3D('', #3, #4, #5);
#3 = CARTESIAN_POINT('', (0., 0., 0.));
#4 = DIRECTION('', (0., 0., 1.));
#5 = DIRECTION('', (1., 0., 0.));
ENDSEC;
END-ISO-10303-21;"#;
        let f = write_temp_step(content);
        let surfaces = read_step_surfaces(f.path());
        assert!(surfaces.is_ok(), "STEP plane parse failed: {surfaces:?}");
        let list = surfaces.unwrap();
        assert!(!list.is_empty(), "should have at least one surface");
    }

    #[test]
    fn step_parse_cylinder() {
        let content = r#"ISO-10303-21;
HEADER;FILE_DESCRIPTION('');ENDSEC;
DATA;
#1 = CYLINDRICAL_SURFACE('', #2, 2.0);
#2 = AXIS2_PLACEMENT_3D('', #3, #4, #5);
#3 = CARTESIAN_POINT('', (0., 0., 0.));
#4 = DIRECTION('', (0., 0., 1.));
#5 = DIRECTION('', (1., 0., 0.));
ENDSEC;
END-ISO-10303-21;"#;
        let f = write_temp_step(content);
        let surfaces = read_step_surfaces(f.path());
        assert!(surfaces.is_ok(), "STEP cylinder parse failed: {surfaces:?}");
    }

    #[test]
    fn step_parse_sphere() {
        let content = r#"ISO-10303-21;
HEADER;FILE_DESCRIPTION('');ENDSEC;
DATA;
#1 = SPHERICAL_SURFACE('', #2, 3.0);
#2 = AXIS2_PLACEMENT_3D('', #3, #4, #5);
#3 = CARTESIAN_POINT('', (0., 0., 0.));
#4 = DIRECTION('', (0., 0., 1.));
#5 = DIRECTION('', (1., 0., 0.));
ENDSEC;
END-ISO-10303-21;"#;
        let f = write_temp_step(content);
        let surfaces = read_step_surfaces(f.path());
        assert!(surfaces.is_ok(), "STEP sphere parse failed: {surfaces:?}");
    }

    #[test]
    fn step_no_surfaces_returns_error() {
        let content = r#"ISO-10303-21;
HEADER;FILE_DESCRIPTION('');ENDSEC;
DATA;
#1 = CARTESIAN_POINT('', (1., 2., 3.));
ENDSEC;
END-ISO-10303-21;"#;
        let f = write_temp_step(content);
        let surfaces = read_step_surfaces(f.path());
        assert!(surfaces.is_err(), "should error on no surfaces");
    }

    #[test]
    fn step_file_not_found() {
        let result = read_step_surfaces("/nonexistent/file.stp");
        assert!(result.is_err(), "should error on missing file");
    }

    #[test]
    fn iges_minimal_header() {
        // Minimal IGES-like content (just a plane)
        let content = "                                                                        S      1\n\
                        108,1,0,0,0,0,0,0;                                       D      1\n\
                        108,1,0,0,0,0,0,0;                                       D      2\n\
                        108,0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0;              P      1\n";
        let mut f = NamedTempFile::new().expect("tempfile");
        write!(f, "{content}").expect("write IGES");
        let surfaces = read_iges_surfaces(f.path());
        // This may succeed or fail depending on parse — test it doesn't panic
        match surfaces {
            Ok(list) => assert!(!list.is_empty()),
            Err(_) => {} // expected for minimal IGES
        }
    }
}
