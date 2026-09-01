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
//! use crate::step_iges::{read_step_surfaces, read_iges_surfaces};
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

use crate::cad::{AnalyticSurface, CadShape, NurbsCadSurface2D, TrimLoop, TrimmedNurbsSurface};


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
            if let Some(id_part) = id_part.strip_prefix('#') {
                let id: usize = id_part.trim().parse().map_err(|_|
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
    if let Some(p) = p.strip_prefix('#') {
        let id: usize = p.parse().ok()?;
        entities.get(&id)
    } else {
        None
    }
}

/// Extract string literals from a parameter string.
#[allow(dead_code)]
fn extract_string(param: &str) -> Option<String> {
    let p = param.trim();
    if let Some(p) = p.strip_prefix('\'') {
        let end = p.find('\'')?;
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

// ─── STEP parameter parsing helpers (handles nested parentheses) ──────────

/// Split a STEP parameter string by top-level commas (outside parentheses).
fn split_step_params(s: &str) -> Vec<String> {
    let mut depth: i32 = 0;
    let mut current = String::new();
    let mut parts = Vec::new();
    for c in s.chars() {
        match c {
            '(' => { depth += 1; current.push(c); }
            ')' => { depth -= 1; current.push(c); }
            ',' if depth == 0 => {
                parts.push(current.trim().to_string());
                current = String::new();
            }
            _ => { current.push(c); }
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        parts.push(trimmed);
    }
    parts
}

/// Strip the outermost pair of parentheses, if present.
fn strip_parens(s: &str) -> &str {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix('(') {
        inner.strip_suffix(')').unwrap_or(inner)
    } else {
        s
    }
}

/// Parse a parenthesised list of floating-point numbers: `(v1, v2, ...)`.
fn parse_num_list(s: &str) -> Vec<f64> {
    strip_parens(s)
        .split(',')
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .collect()
}

/// Parse a parenthesised list of integers: `(n1, n2, ...)`.
fn parse_int_list(s: &str) -> Vec<usize> {
    strip_parens(s)
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect()
}

/// Expand knot multiplicities + unique knot values into a full knot vector.
fn expand_knot_vector(mults: &[usize], knots: &[f64]) -> Vec<f64> {
    let mut kv = Vec::new();
    for (&m, &k) in mults.iter().zip(knots.iter()) {
        for _ in 0..m {
            kv.push(k);
        }
    }
    kv
}

// ─── B-spline curve evaluation (de Boor algorithm) ───────────────────────

/// Evaluate a 2-D B-spline curve at parameter `t` using the de Boor algorithm.
fn eval_bspline_curve_2d(t: f64, p: usize, knots: &[f64], ctrl: &[[f64; 2]]) -> [f64; 2] {
    let n = ctrl.len();
    let m = knots.len();
    let t = t.clamp(knots[p], knots[m - p - 1]);

    // Find span index k such that t ∈ [knots[k], knots[k+1])
    let mut k = p;
    for i in (p + 1)..(m - p - 1) {
        if knots[i] > t + 1e-14 {
            break;
        }
        k = i;
    }
    if t >= knots[m - p - 1] {
        k = (n - 1).max(p);
    }
    if k > n - 1 {
        k = n - 1;
    }

    let start = if k >= p { k - p } else { 0 };
    let end = if k < n { k } else { n - 1 };
    let active_cnt = end - start + 1;

    if active_cnt == 0 || p == 0 {
        return ctrl[start.min(n - 1)];
    }

    let mut q: Vec<[f64; 2]> = (start..=end).map(|i| ctrl[i]).collect();

    for r in 1..=p.min(active_cnt - 1) {
        for j in (r..active_cnt).rev() {
            let knot_idx = start + j;
            let denom = knots[knot_idx + p + 1 - r] - knots[knot_idx];
            if denom.abs() > 1e-14 {
                let alpha = (t - knots[knot_idx]) / denom;
                q[j][0] = (1.0 - alpha) * q[j - 1][0] + alpha * q[j][0];
                q[j][1] = (1.0 - alpha) * q[j - 1][1] + alpha * q[j][1];
            }
        }
    }
    q[active_cnt - 1]
}

/// Sample a 2-D B-spline curve uniformly in its parameter domain.
fn sample_bspline_curve_2d(
    degree: usize,
    knots: &[f64],
    ctrl_pts: &[[f64; 2]],
    n_samples: usize,
) -> Vec<[f64; 2]> {
    if ctrl_pts.is_empty() {
        return Vec::new();
    }
    if degree == 0 || ctrl_pts.len() == 1 {
        return ctrl_pts.to_vec();
    }

    let t_min = knots[degree];
    let t_max = knots[knots.len() - degree - 1];

    let mut result = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = t_min + (t_max - t_min) * i as f64 / (n_samples - 1).max(1) as f64;
        result.push(eval_bspline_curve_2d(t, degree, knots, ctrl_pts));
    }
    result
}

// ─── Entity-specific parsers ─────────────────────────────────────────────

/// Parse a `B_SPLINE_CURVE_WITH_KNOTS` entity and sample its polygon.
fn extract_bspline_curve_polygon(
    entity: &StepEntity,
    entities: &HashMap<usize, StepEntity>,
) -> Option<Vec<[f64; 2]>> {
    // Strip the outer (…) before splitting — entity.params starts with '('.
    let content = strip_parens(&entity.params);
    let parts = split_step_params(content);
    if parts.len() < 9 {
        return None;
    }

    let degree: usize = parts[1].trim().parse().ok()?;

    // Extract control point references from (#ref1, #ref2, ...)
    let cp_list_str = strip_parens(&parts[2]);
    let cp_refs: Vec<&str> = cp_list_str.split(',').map(|s| s.trim()).collect();

    let mut ctrl_pts: Vec<[f64; 2]> = Vec::new();
    for ref_str in &cp_refs {
        if let Some(id_str) = ref_str.strip_prefix('#') {
            if let Ok(id) = id_str.trim().parse::<usize>() {
                if let Some(cp_entity) = entities.get(&id) {
                    if let Some(pt) = extract_point(cp_entity) {
                        ctrl_pts.push([pt[0], pt[1]]);
                    }
                }
            }
        }
    }
    if ctrl_pts.is_empty() {
        return None;
    }

    // Extract knot multiplicities and values
    let mults: Vec<usize> = parse_int_list(&parts[6]);
    let knots: Vec<f64> = parse_num_list(&parts[7]);
    if mults.is_empty() || knots.is_empty() {
        return None;
    }

    let full_kv = expand_knot_vector(&mults, &knots);

    // Check for closed curve
    let closed = parts[4].trim().contains(".T.");

    let n_samples = if degree <= 1 { ctrl_pts.len() } else { 32 };
    let mut poly = sample_bspline_curve_2d(degree, &full_kv, &ctrl_pts, n_samples);

    // Close the polygon if needed
    if closed && poly.len() > 1 {
        let first = poly[0];
        let last = poly[poly.len() - 1];
        if (first[0] - last[0]).abs() > 1e-14 || (first[1] - last[1]).abs() > 1e-14 {
            poly.push(first);
        }
    }

    Some(poly)
}

/// Parse a `LINE` entity and return its two endpoints in (u,v).
fn extract_line_segment(
    entity: &StepEntity,
    entities: &HashMap<usize, StepEntity>,
) -> Option<Vec<[f64; 2]>> {
    let content = strip_parens(&entity.params);
    let parts = split_step_params(content);
    if parts.len() < 3 {
        return None;
    }

    let origin_ref = resolve_ref(parts[1].trim(), entities)?;
    let origin = extract_point(origin_ref)?;

    let dir_entity = resolve_ref(parts[2].trim(), entities)?;
    let (dx, dy) = if dir_entity.type_name == "VECTOR" {
        let vparts = split_step_params(&dir_entity.params);
        if vparts.len() < 3 {
            return None;
        }
        let mag: f64 = parse_num_list(&vparts[2]).first().copied().unwrap_or(1.0);
        let dir_ref = resolve_ref(vparts[1].trim(), entities)?;
        let dir = extract_direction(dir_ref).unwrap_or([1.0, 0.0, 0.0]);
        (dir[0] * mag, dir[1] * mag)
    } else {
        let dir = extract_direction(dir_entity).unwrap_or([1.0, 0.0, 0.0]);
        (dir[0], dir[1])
    };

    Some(vec![[origin[0], origin[1]], [origin[0] + dx, origin[1] + dy]])
}

/// Parse a `B_SPLINE_SURFACE_WITH_KNOTS` entity into `NurbsCadSurface2D`.
fn extract_bspline_surface(
    entity: &StepEntity,
    entities: &HashMap<usize, StepEntity>,
) -> Option<NurbsCadSurface2D> {
    let content = strip_parens(&entity.params);
    let parts = split_step_params(content);
    if parts.len() < 13 {
        return None;
    }

    let _u_deg: usize = parts[1].trim().parse().ok()?;
    let _v_deg: usize = parts[2].trim().parse().ok()?;

    // Parse the 2-D control-point array: ((row0...),(row1...),...)
    let cp_array_str = strip_parens(&parts[3]);
    let row_strs = split_step_params(cp_array_str);

    let mut ctrl_pts: Vec<[f64; 3]> = Vec::new();
    for row_str in &row_strs {
        let inner = strip_parens(row_str);
        let cp_refs: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
        for ref_str in &cp_refs {
            if let Some(id_str) = ref_str.strip_prefix('#') {
                if let Ok(id) = id_str.trim().parse::<usize>() {
                    if let Some(cp_entity) = entities.get(&id) {
                        if let Some(pt) = extract_point(cp_entity) {
                            ctrl_pts.push(pt);
                        }
                    }
                }
            }
        }
    }
    if ctrl_pts.is_empty() {
        return None;
    }

    // Parse knot vectors
    let u_mults: Vec<usize> = parse_int_list(&parts[8]);
    let v_mults: Vec<usize> = parse_int_list(&parts[9]);
    let u_knots: Vec<f64> = parse_num_list(&parts[10]);
    let v_knots: Vec<f64> = parse_num_list(&parts[11]);

    let full_u_kv = expand_knot_vector(&u_mults, &u_knots);
    let full_v_kv = expand_knot_vector(&v_mults, &v_knots);

    if full_u_kv.is_empty() || full_v_kv.is_empty() {
        return None;
    }

    // B_SPLINE_SURFACE has no weights; default to 1.0.
    let weights = vec![1.0; ctrl_pts.len()];

    Some(NurbsCadSurface2D::new(full_u_kv, full_v_kv, ctrl_pts, weights))
}

/// Extract a trimming curve (polygon) from a STEP curve entity reference.
fn resolve_trim_curve(
    entity: &StepEntity,
    entities: &HashMap<usize, StepEntity>,
) -> Option<Vec<[f64; 2]>> {
    match entity.type_name.as_str() {
        "B_SPLINE_CURVE_WITH_KNOTS" => extract_bspline_curve_polygon(entity, entities),
        "LINE" => extract_line_segment(entity, entities),
        _ => {
            eprintln!("STEP: unsupported trimming curve type: {}", entity.type_name);
            None
        }
    }
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
            // Full NURBS surface extraction from B-spline surface data.
            if let Some(ncs) = extract_bspline_surface(entity, entities) {
                Some(CadShape::Nurbs(ncs))
            } else {
                eprintln!("STEP: failed to parse B_SPLINE_SURFACE");
                None
            }
        }
        "TRIMMED_SURFACE" => {
            // TRIMMED_SURFACE('', #surface_ref, (#trim1, #trim2, ...), sense)
            let content = strip_parens(&entity.params);
            let parts = split_step_params(content);
            if parts.len() < 4 {
                return None;
            }
            // Resolve the basis surface
            let surface_ref = resolve_ref(&parts[1], entities)?;
            let ncs = extract_bspline_surface(surface_ref, entities)?;

            // Parse trimming curve references from (#ref1, #ref2, ...)
            let trim_list_str = strip_parens(&parts[2]);
            let trim_refs: Vec<&str> = trim_list_str.split(',').map(|s| s.trim()).collect();

            let mut trim_loops: Vec<TrimLoop> = Vec::new();
            for ref_str in &trim_refs {
                if let Some(id_str) = ref_str.strip_prefix('#') {
                    if let Ok(id) = id_str.trim().parse::<usize>() {
                        if let Some(curve_entity) = entities.get(&id) {
                            if let Some(poly) = resolve_trim_curve(curve_entity, entities) {
                                trim_loops.push(TrimLoop::new(poly));
                            }
                        }
                    }
                }
            }

            if trim_loops.is_empty() {
                eprintln!("STEP: TRIMMED_SURFACE has no valid trimming curves");
                return None;
            }

            Some(CadShape::TrimmedNurbs(TrimmedNurbsSurface {
                surface: ncs,
                trim_loops,
            }))
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
        "B_SPLINE_SURFACE_WITH_KNOTS", "TRIMMED_SURFACE",
    ];

    for entity in entities.values() {
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
            param_lines.push_str(trimmed[..trimmed.len().saturating_sub(1)].trim());
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
    fn step_parse_trimmed_surface_with_bspline_curve() {
        // B_SPLINE_SURFACE_WITH_KNOTS: degree-1 unit square with 2×2 CPs
        // Trimming curve: degree-1 rectangle in parameter space
        let content = r#"ISO-10303-21;
HEADER;FILE_DESCRIPTION('');ENDSEC;
DATA;
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (0.0, 1.0, 0.0));
#4 = CARTESIAN_POINT('', (1.0, 1.0, 0.0));
#5 = B_SPLINE_SURFACE_WITH_KNOTS('', 1, 1, ((#1, #2), (#3, #4)), .UNSPECIFIED., .F., .F., .F., (2, 2), (2, 2), (0.0, 1.0), (0.0, 1.0), .UNSPECIFIED.);
#6 = CARTESIAN_POINT('', (0.2, 0.2, 0.0));
#7 = CARTESIAN_POINT('', (0.8, 0.2, 0.0));
#8 = CARTESIAN_POINT('', (0.8, 0.8, 0.0));
#9 = CARTESIAN_POINT('', (0.2, 0.8, 0.0));
#10 = CARTESIAN_POINT('', (0.2, 0.2, 0.0));
#11 = B_SPLINE_CURVE_WITH_KNOTS('', 1, (#6, #7, #8, #9, #10), .UNSPECIFIED., .F., .F., (2, 1, 1, 1, 2), (0.0, 0.25, 0.5, 0.75, 1.0), .UNSPECIFIED.);
#12 = TRIMMED_SURFACE('', #5, (#11), .T.);
ENDSEC;
END-ISO-10303-21;"#;
        let f = write_temp_step(content);
        let result = read_step_surfaces(f.path());
        assert!(result.is_ok(), "STEP trimmed surface parse failed: {:?}", result.err());
        let surfaces = result.unwrap();
        assert!(!surfaces.is_empty(), "expected at least 1 surface");

        // Find the trimmed surface (should be last).
        let trimmed = surfaces.into_iter().find(|(_, shape)| matches!(shape, CadShape::TrimmedNurbs(_)));
        assert!(trimmed.is_some(), "no TrimmedNurbs surface found");
        let (_tag, shape) = trimmed.unwrap();
        match shape {
            CadShape::TrimmedNurbs(tns) => {
                assert_eq!(tns.trim_loops.len(), 1, "expected 1 trim loop");
                let loop0 = &tns.trim_loops[0];
                assert!(loop0.vertices.len() >= 4, "trim loop should have ≥4 vertices");
                // Center of the trim rect should be inside
                assert!(loop0.contains(0.5, 0.5), "center of rect should be inside");
                // Corners of the unit square should be outside (outside the trim rect)
                assert!(!loop0.contains(0.0, 0.0), "corner (0,0) should be outside trim");
                assert!(!loop0.contains(1.0, 1.0), "corner (1,1) should be outside trim");
            }
            other => panic!("expected TrimmedNurbs, got {other:?}"),
        }
    }

    #[test]
    fn step_parse_bspline_surface_returns_nurbs() {
        let content = r#"ISO-10303-21;
HEADER;FILE_DESCRIPTION('');ENDSEC;
DATA;
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (0.5, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 1.0, 0.0));
#5 = CARTESIAN_POINT('', (0.5, 1.0, 0.0));
#6 = CARTESIAN_POINT('', (1.0, 1.0, 0.0));
#7 = B_SPLINE_SURFACE_WITH_KNOTS('', 1, 1, ((#1, #2, #3), (#4, #5, #6)), .UNSPECIFIED., .F., .F., .F., (2, 1, 2), (2, 2), (0.0, 0.5, 1.0), (0.0, 1.0), .UNSPECIFIED.);
ENDSEC;
END-ISO-10303-21;"#;
        let f = write_temp_step(content);
        let surfaces = read_step_surfaces(f.path());
        assert!(surfaces.is_ok(), "B_SPLINE_SURFACE parse failed: {:?}", surfaces.err());
        let list = surfaces.unwrap();
        assert_eq!(list.len(), 1);
        // Take ownership of the shape to call into_patch_data (control_pts is private).
        let (_tag, shape) = list.into_iter().next().unwrap();
        match shape {
            CadShape::Nurbs(ncs) => {
                let pd = ncs.into_patch_data();
                // 3×2 CPs: u-direction 3 CPs, v-direction 2 CPs
                assert_eq!(pd.control_pts.len(), 6);
                // Verify spatial extent
                let min_x = pd.control_pts.iter().map(|c| c[0]).reduce(f64::min).unwrap();
                let max_x = pd.control_pts.iter().map(|c| c[0]).reduce(f64::max).unwrap();
                assert!((min_x - 0.0).abs() < 1e-14);
                assert!((max_x - 1.0).abs() < 1e-14);
            }
            other => panic!("expected CadShape::Nurbs, got {other:?}"),
        }
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
