//! GPU-accelerated IGA Bezier extraction assembly.
//!
//! Dispatches WGSL compute shaders that evaluate NURBS basis functions via
//! Bernstein polynomials on uniform knot vectors (extraction matrix C = I).
//! Each work-group processes one element independently.
//!
//! ## Supported degrees
//! - 2-D: `p,q ≤ 3` (max 16 local basis functions)
//! - 3-D: `p,q,r ≤ 2` (max 27 local basis functions)
//!
//! ## Precision
//! All shaders use `f64` and require `SHADER_F64` support.

use bytemuck::{Pod, Zeroable};

use crate::context::GpuContext;

// ─── GPU-side element input structs ──────────────────────────────────────────

/// GPU-side input for one 2-D IGA Bezier element.
///
/// Every element gets 16 slots (cubic max: 4×4 = 16). Lower-degree elements
/// leave trailing slots zero; unused COO entries are filtered on the host.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuIgaBezier2DElement {
    /// `[n_local][2]` — control point coordinates, padded to 16×2
    pub cpts: [f64; 32],
    /// `[n_local]` — NURBS weights, padded to 16
    pub weights: [f64; 16],
    /// `[n_local]` — global DOF indices, padded to 16
    pub dofs: [u32; 16],
}

/// GPU-side input for one 3-D IGA Bezier element.
///
/// Max 3×3×3 = 27 (degree 2 in each direction).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuIgaBezier3DElement {
    /// `[n_local][3]` — control point coordinates, padded to 27×3
    pub cpts: [f64; 81],
    /// `[n_local]` — NURBS weights, padded to 27
    pub weights: [f64; 27],
    /// `[n_local]` — global DOF indices, padded to 27
    pub dofs: [u32; 27],
    /// Padding to multiple of 16 bytes
    pub _pad: u32,
}

// ─── 2-D diffusion ───────────────────────────────────────────────────────────

/// Assemble 2-D IGA Bezier diffusion stiffness matrix on GPU.
///
/// Each element must pre-pack `control_pts`, `weights`, and `dofs` into
/// [`GpuIgaBezier2DElement`].  Returns f64 COO triplets — one per non-zero
/// entry of the assembled element matrices.
///
/// Requires `gpu.features.native_f64`.
pub fn assemble_iga_bezier_diffusion_2d(
    gpu: &GpuContext,
    elements: &[GpuIgaBezier2DElement],
    p: u32,
    q: u32,
    quad_order: u32,
) -> Vec<(u32, u32, f64)> {
    assert!(gpu.features.native_f64, "SHADER_F64 required");
    let n_elem = elements.len();

    // Pack params: [n_elements, p, q, quad_order] as 4 × u32 = 16 bytes
    let params: [u32; 4] = [n_elem as u32, p, q, quad_order];
    let param_bytes: Vec<u8> = params.iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    super::assembly::run_assembly_shader_f64_with_params(
        gpu,
        bytemuck::cast_slice(elements),
        n_elem,
        256, // worst-case entries: 16 × 16
        include_str!("iga_bezier_2d_diffusion_f64.wgsl"),
        &param_bytes,
    )
}

// ─── 2-D mass ────────────────────────────────────────────────────────────────

/// Assemble 2-D IGA Bezier mass matrix on GPU.
///
/// Requires `gpu.features.native_f64`.
pub fn assemble_iga_bezier_mass_2d(
    gpu: &GpuContext,
    elements: &[GpuIgaBezier2DElement],
    p: u32,
    q: u32,
    quad_order: u32,
) -> Vec<(u32, u32, f64)> {
    assert!(gpu.features.native_f64, "SHADER_F64 required");
    let n_elem = elements.len();

    let params: [u32; 4] = [n_elem as u32, p, q, quad_order];
    let param_bytes: Vec<u8> = params.iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    super::assembly::run_assembly_shader_f64_with_params(
        gpu,
        bytemuck::cast_slice(elements),
        n_elem,
        256,
        include_str!("iga_bezier_2d_mass_f64.wgsl"),
        &param_bytes,
    )
}

// ─── 3-D diffusion ───────────────────────────────────────────────────────────

/// Assemble 3-D IGA Bezier diffusion stiffness matrix on GPU.
///
/// Requires `gpu.features.native_f64`.
pub fn assemble_iga_bezier_diffusion_3d(
    gpu: &GpuContext,
    elements: &[GpuIgaBezier3DElement],
    p: u32,
    q: u32,
    r: u32,
    quad_order: u32,
) -> Vec<(u32, u32, f64)> {
    assert!(gpu.features.native_f64, "SHADER_F64 required");
    let n_elem = elements.len();

    // Params: [n_elements, p, q, r, quad_order, 0, 0, 0] = 8 × u32 = 32 bytes
    let params: [u32; 8] = [n_elem as u32, p, q, r, quad_order, 0, 0, 0];
    let param_bytes: Vec<u8> = params.iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    super::assembly::run_assembly_shader_f64_with_params(
        gpu,
        bytemuck::cast_slice(elements),
        n_elem,
        729, // worst-case entries: 27 × 27
        include_str!("iga_bezier_3d_diffusion_f64.wgsl"),
        &param_bytes,
    )
}

// ─── 3-D mass ────────────────────────────────────────────────────────────────

/// Assemble 3-D IGA Bezier mass matrix on GPU.
///
/// Requires `gpu.features.native_f64`.
pub fn assemble_iga_bezier_mass_3d(
    gpu: &GpuContext,
    elements: &[GpuIgaBezier3DElement],
    p: u32,
    q: u32,
    r: u32,
    quad_order: u32,
) -> Vec<(u32, u32, f64)> {
    assert!(gpu.features.native_f64, "SHADER_F64 required");
    let n_elem = elements.len();

    let params: [u32; 8] = [n_elem as u32, p, q, r, quad_order, 0, 0, 0];
    let param_bytes: Vec<u8> = params.iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    super::assembly::run_assembly_shader_f64_with_params(
        gpu,
        bytemuck::cast_slice(elements),
        n_elem,
        729,
        include_str!("iga_bezier_3d_mass_f64.wgsl"),
        &param_bytes,
    )
}
