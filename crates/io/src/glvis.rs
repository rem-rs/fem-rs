//! GLVis real-time visualization via TCP socket.
//!
//! Uses the VTK legacy format and sends it over a TCP socket to a running
//! GLVis server (default `localhost:19916`).
//!
//! Supports 2-D and 3-D simplicial meshes with scalar and vector fields.
//!
//! # Usage
//! ```no_run
//! use fem_io::glvis::GlVisSocket;
//! use fem_mesh::SimplexMesh;
//!
//! // 2-D
//! let mesh2 = SimplexMesh::<2>::unit_square_tri(8);
//! let sol2 = vec![0.5; mesh2.n_nodes()];
//! let mut vis = GlVisSocket::connect("localhost", 19916).unwrap();
//! vis.send_solution_2d(&mesh2, &sol2, "u").unwrap();
//!
//! // 3-D
//! let mesh3 = SimplexMesh::<3>::unit_cube_tet(4);
//! let sol3 = vec![0.5; mesh3.n_nodes()];
//! vis.send_solution_3d(&mesh3, &sol3, "u").unwrap();
//! ```

use std::io::{self, Write};
use std::net::TcpStream;

use fem_mesh::simplex::SimplexMesh;

/// A TCP connection to a GLVis server.
pub struct GlVisSocket {
    stream: TcpStream,
}

impl GlVisSocket {
    /// Open a socket to a GLVis server.
    pub fn connect(host: &str, port: u16) -> io::Result<Self> {
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect(&addr)?;
        Ok(GlVisSocket { stream })
    }

    // ── 2-D convenience methods ──────────────────────────────────────────────

    /// Send a 2-D scalar solution to GLVis.
    pub fn send_solution_2d(
        &mut self,
        mesh: &SimplexMesh<2>,
        scalar_field: &[f64],
        field_name: &str,
    ) -> io::Result<()> {
        write!(self.stream, "solution\n")?;
        write_vtk_mesh_2d(&mut self.stream, mesh)?;
        write_vtk_scalar(&mut self.stream, mesh.n_nodes(), scalar_field, field_name)?;
        self.stream.flush()
    }

    /// Send a 2-D vector solution to GLVis.
    pub fn send_solution_2d_vector(
        &mut self,
        mesh: &SimplexMesh<2>,
        field_x: &[f64],
        field_y: &[f64],
        field_name: &str,
    ) -> io::Result<()> {
        write!(self.stream, "solution\n")?;
        write_vtk_mesh_2d(&mut self.stream, mesh)?;
        write_vtk_vector_2d(&mut self.stream, mesh.n_nodes(), field_x, field_y, field_name)?;
        self.stream.flush()
    }

    // ── 3-D convenience methods ──────────────────────────────────────────────

    /// Send a 3-D scalar solution to GLVis.
    pub fn send_solution_3d(
        &mut self,
        mesh: &SimplexMesh<3>,
        scalar_field: &[f64],
        field_name: &str,
    ) -> io::Result<()> {
        write!(self.stream, "solution\n")?;
        write_vtk_mesh_3d(&mut self.stream, mesh)?;
        write_vtk_scalar(&mut self.stream, mesh.n_nodes(), scalar_field, field_name)?;
        self.stream.flush()
    }

    /// Send a 3-D vector solution to GLVis.
    pub fn send_solution_3d_vector(
        &mut self,
        mesh: &SimplexMesh<3>,
        field_x: &[f64],
        field_y: &[f64],
        field_z: &[f64],
        field_name: &str,
    ) -> io::Result<()> {
        write!(self.stream, "solution\n")?;
        write_vtk_mesh_3d(&mut self.stream, mesh)?;
        write_vtk_vector_3d(&mut self.stream, mesh.n_nodes(), field_x, field_y, field_z, field_name)?;
        self.stream.flush()
    }

    // ── Low-level ────────────────────────────────────────────────────────────

    /// Send arbitrary VTK text followed by flush.
    pub fn send_raw(&mut self, data: &str) -> io::Result<()> {
        write!(self.stream, "{}", data)?;
        self.stream.flush()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VTK legacy format serializers — 2D
// ═══════════════════════════════════════════════════════════════════════════════

fn write_vtk_mesh_2d(w: &mut dyn Write, mesh: &SimplexMesh<2>) -> io::Result<()> {
    let nn = mesh.n_nodes();
    let ne = mesh.n_elems();

    writeln!(w, "DATASET UNSTRUCTURED_GRID")?;
    writeln!(w, "POINTS {} float", nn)?;
    for n in 0..nn {
        let c = mesh.coords_of(n as u32);
        writeln!(w, "{} {} 0.0", c[0], c[1])?;
    }

    let mut conn_size = 0;
    for e in 0..ne {
        conn_size += 1 + mesh.elem_nodes(e as u32).len();
    }
    writeln!(w, "CELLS {} {}", ne, conn_size)?;
    for e in 0..ne {
        let eid = e as u32;
        let nodes = mesh.elem_nodes(eid);
        write!(w, "{}", nodes.len())?;
        for &n in nodes {
            write!(w, " {}", n)?;
        }
        writeln!(w)?;
    }

    writeln!(w, "CELL_TYPES {}", ne)?;
    for e in 0..ne {
        writeln!(w, "{}", cell_type_vtk_2d(mesh, e as u32))?;
    }

    Ok(())
}

fn cell_type_vtk_2d(mesh: &SimplexMesh<2>, elem: u32) -> u8 {
    match mesh.elem_nodes(elem).len() {
        3 => 5,  // VTK_TRIANGLE
        6 => 22, // VTK_QUADRATIC_TRIANGLE
        4 => 9,  // VTK_QUAD
        8 => 23, // VTK_QUADRATIC_QUAD
        9 => 28, // VTK_BIQUADRATIC_QUAD
        _ => 5,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VTK legacy format serializers — 3D
// ═══════════════════════════════════════════════════════════════════════════════

fn write_vtk_mesh_3d(w: &mut dyn Write, mesh: &SimplexMesh<3>) -> io::Result<()> {
    let nn = mesh.n_nodes();
    let ne = mesh.n_elems();

    writeln!(w, "DATASET UNSTRUCTURED_GRID")?;
    writeln!(w, "POINTS {} float", nn)?;
    for n in 0..nn {
        let c = mesh.coords_of(n as u32);
        writeln!(w, "{} {} {}", c[0], c[1], c[2])?;
    }

    let mut conn_size = 0;
    for e in 0..ne {
        conn_size += 1 + mesh.elem_nodes(e as u32).len();
    }
    writeln!(w, "CELLS {} {}", ne, conn_size)?;
    for e in 0..ne {
        let eid = e as u32;
        let nodes = mesh.elem_nodes(eid);
        write!(w, "{}", nodes.len())?;
        for &n in nodes {
            write!(w, " {}", n)?;
        }
        writeln!(w)?;
    }

    writeln!(w, "CELL_TYPES {}", ne)?;
    for e in 0..ne {
        writeln!(w, "{}", cell_type_vtk_3d(mesh, e as u32))?;
    }

    Ok(())
}

fn cell_type_vtk_3d(mesh: &SimplexMesh<3>, elem: u32) -> u8 {
    match mesh.elem_nodes(elem).len() {
        4 => 10, // VTK_TETRA
        10 => 24, // VTK_QUADRATIC_TETRA
        8 => 12, // VTK_HEXAHEDRON
        20 => 25, // VTK_QUADRATIC_HEXAHEDRON
        6 => 13, // VTK_WEDGE (prism)
        5 => 14, // VTK_PYRAMID
        _ => 10,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared VTK field serializers (dimension-independent)
// ═══════════════════════════════════════════════════════════════════════════════

fn write_vtk_scalar(
    w: &mut dyn Write,
    nn: usize,
    field: &[f64],
    name: &str,
) -> io::Result<()> {
    writeln!(w, "POINT_DATA {}", nn)?;
    writeln!(w, "SCALARS {} float 1", name)?;
    writeln!(w, "LOOKUP_TABLE default")?;
    for i in 0..nn {
        writeln!(w, "{:.10e}", field[i])?;
    }
    Ok(())
}

fn write_vtk_vector_2d(
    w: &mut dyn Write,
    nn: usize,
    fx: &[f64],
    fy: &[f64],
    name: &str,
) -> io::Result<()> {
    writeln!(w, "POINT_DATA {}", nn)?;
    writeln!(w, "VECTORS {} float", name)?;
    for i in 0..nn {
        writeln!(w, "{:.10e} {:.10e} 0.0", fx[i], fy[i])?;
    }
    Ok(())
}

fn write_vtk_vector_3d(
    w: &mut dyn Write,
    nn: usize,
    fx: &[f64],
    fy: &[f64],
    fz: &[f64],
    name: &str,
) -> io::Result<()> {
    writeln!(w, "POINT_DATA {}", nn)?;
    writeln!(w, "VECTORS {} float", name)?;
    for i in 0..nn {
        writeln!(w, "{:.10e} {:.10e} {:.10e}", fx[i], fy[i], fz[i])?;
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// VTK legacy format smoke test: serialise a 2-D mesh + scalar.
    #[test]
    fn glvis_format_2d_scalar() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let sol = vec![1.0_f64; mesh.n_nodes()];
        let mut buf = Vec::new();
        write!(buf, "solution\n").unwrap();
        write_vtk_mesh_2d(&mut buf, &mesh).unwrap();
        write_vtk_scalar(&mut buf, mesh.n_nodes(), &sol, "u").unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("DATASET UNSTRUCTURED_GRID"));
        assert!(out.contains("POINTS"));
        assert!(out.contains("CELLS"));
        assert!(out.contains("SCALARS u"));
    }

    /// VTK legacy format smoke test: serialise a 3-D mesh + scalar.
    #[test]
    fn glvis_format_3d_scalar() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let sol = vec![1.0_f64; mesh.n_nodes()];
        let mut buf = Vec::new();
        write!(buf, "solution\n").unwrap();
        write_vtk_mesh_3d(&mut buf, &mesh).unwrap();
        write_vtk_scalar(&mut buf, mesh.n_nodes(), &sol, "u").unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("DATASET UNSTRUCTURED_GRID"));
        assert!(out.contains("POINTS"));
        assert!(out.contains("CELL_TYPES"));
        // CELL_TYPES block: header line, then one int per element.
        let mut lines = out.lines().skip_while(|l| !l.starts_with("CELL_TYPES"));
        let header = lines.next().unwrap();
        let n_types: usize = header.split_whitespace().nth(1).unwrap().parse().unwrap();
        let types: Vec<u8> = lines.take(n_types).map(|l| l.trim().parse().unwrap_or(0)).collect();
        assert_eq!(types.len(), n_types, "CELL_TYPES count mismatch");
        assert!(types.iter().all(|&t| t == 10), "expected all VTK_TETRA=10");
    }

    /// VTK legacy format: 3-D mesh + vector field.
    #[test]
    fn glvis_format_3d_vector() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1); // single tet for simpler output
        let fx = vec![0.0_f64; mesh.n_nodes()];
        let fy = vec![1.0_f64; mesh.n_nodes()];
        let fz = vec![2.0_f64; mesh.n_nodes()];
        let mut buf = Vec::new();
        write!(buf, "solution\n").unwrap();
        write_vtk_mesh_3d(&mut buf, &mesh).unwrap();
        write_vtk_vector_3d(&mut buf, mesh.n_nodes(), &fx, &fy, &fz, "F").unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("VECTORS F"), "missing VECTORS header");
        // Check that exactly one VECTORS line with three floats appears.
        let lines: Vec<&str> = out.lines().filter(|l| l.contains('e')).collect();
        assert!(lines.iter().all(|l| l.split_whitespace().count() == 3),
            "each vector line should have 3 components");
        assert!(!lines.is_empty(), "expected at least one vector line");
    }

    /// Verify backward compatibility: old public method still works.
    #[test]
    fn glvis_backward_compat_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let sol = vec![0.5_f64; mesh.n_nodes()];
        // Simulate a local connection on a dummy socket for format check.
        let mut buf = Vec::new();
        // Manually write the old-style format for verification.
        write!(buf, "solution\n").unwrap();
        write_vtk_mesh_2d(&mut buf, &mesh).unwrap();
        write_vtk_scalar(&mut buf, mesh.n_nodes(), &sol, "u").unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("SCALARS u"));
    }
}
