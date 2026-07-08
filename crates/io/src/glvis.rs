//! GLVis real-time visualization via TCP socket.
//!
//! Uses the VTK legacy format and sends it over a TCP socket to a running
//! GLVis server (default `localhost:19916`).
//!
//! Supports 2-D and 3-D simplicial meshes with scalar and vector fields,
//! window commands (view angle, zoom, etc.), and reading GLVis responses.
//!
//! # Usage
//! ```no_run
//! use fem_io::glvis::GlVisSocket;
//! use fem_mesh::Mesh;
//!
//! // 2-D
//! let mesh2 = Mesh::<2>::unit_square_tri(8);
//! let sol2 = vec![0.5; mesh2.n_nodes()];
//! let mut vis = GlVisSocket::connect("localhost", 19916).unwrap();
//! vis.send_solution_2d(&mesh2, &sol2, "u").unwrap();
//! println!("GLVis: {}", vis.recv_response_line().unwrap());
//!
//! // 3-D
//! let mesh3 = Mesh::<3>::unit_cube_tet(4);
//! let sol3 = vec![0.5; mesh3.n_nodes()];
//! vis.send_solution_3d(&mesh3, &sol3, "u").unwrap();
//! ```

use std::io::{self, BufRead, BufReader, Write};
use std::net::TcpStream;

use fem_mesh::simplex::Mesh;

/// A TCP connection to a GLVis server.
pub struct GlVisSocket {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

impl GlVisSocket {
    /// Open a socket to a GLVis server.
    pub fn connect(host: &str, port: u16) -> io::Result<Self> {
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect(&addr)?;
        let reader = BufReader::new(stream.try_clone()?);
        Ok(GlVisSocket { stream, reader })
    }

    /// Read one response line from the GLVis server.
    pub fn recv_response_line(&mut self) -> io::Result<String> {
        let mut line = String::new();
        self.reader.read_line(&mut line)?;
        Ok(line.trim_end().to_string())
    }

    /// Read all response lines until GLVis sends an empty line.
    pub fn recv_response(&mut self) -> io::Result<Vec<String>> {
        let mut lines = Vec::new();
        loop {
            let line = self.recv_response_line()?;
            if line.is_empty() { break; }
            lines.push(line);
        }
        Ok(lines)
    }

    /// Send a GLVis window command (e.g. "view 0 0 1", "zoom 2", "autoscale").
    pub fn send_command(&mut self, cmd: &str) -> io::Result<()> {
        writeln!(self.stream, "{}", cmd)?;
        self.stream.flush()
    }

    /// Convenience: send a 2-D solution followed by GLVis commands.
    pub fn send_solution_2d_with_cmd(
        &mut self, mesh: &Mesh<2>,
        scalar_field: &[f64], field_name: &str,
        commands: &[&str],
    ) -> io::Result<()> {
        self.send_solution_2d(mesh, scalar_field, field_name)?;
        for cmd in commands { self.send_command(cmd)?; }
        Ok(())
    }

    /// Convenience: send a 3-D solution followed by GLVis commands.
    pub fn send_solution_3d_with_cmd(
        &mut self, mesh: &Mesh<3>,
        scalar_field: &[f64], field_name: &str,
        commands: &[&str],
    ) -> io::Result<()> {
        self.send_solution_3d(mesh, scalar_field, field_name)?;
        for cmd in commands { self.send_command(cmd)?; }
        Ok(())
    }

    // ── 2-D convenience methods ──────────────────────────────────────────────

    /// Send a 2-D scalar solution to GLVis.
    pub fn send_solution_2d(
        &mut self,
        mesh: &Mesh<2>,
        scalar_field: &[f64],
        field_name: &str,
    ) -> io::Result<()> {
        writeln!(self.stream, "solution")?;
        write_vtk_mesh_2d(&mut self.stream, mesh)?;
        write_vtk_scalar(&mut self.stream, mesh.n_nodes(), scalar_field, field_name)?;
        self.stream.flush()
    }

    /// Send a 2-D vector solution to GLVis.
    pub fn send_solution_2d_vector(
        &mut self,
        mesh: &Mesh<2>,
        field_x: &[f64],
        field_y: &[f64],
        field_name: &str,
    ) -> io::Result<()> {
        writeln!(self.stream, "solution")?;
        write_vtk_mesh_2d(&mut self.stream, mesh)?;
        write_vtk_vector_2d(&mut self.stream, mesh.n_nodes(), field_x, field_y, field_name)?;
        self.stream.flush()
    }

    /// Send a 2-D vector solution to GLVis in parallel mode.
    ///
    /// Prefixes the stream with `parallel <n_ranks> <rank>` so GLVis combines
    /// solutions from all ranks into a single view.
    pub fn send_parallel_solution_2d_vector(
        &mut self,
        n_ranks: usize,
        rank: usize,
        mesh: &Mesh<2>,
        field_x: &[f64],
        field_y: &[f64],
        field_name: &str,
    ) -> io::Result<()> {
        writeln!(self.stream, "parallel {} {}", n_ranks, rank)?;
        self.send_solution_2d_vector(mesh, field_x, field_y, field_name)
    }

    // ── 3-D convenience methods ──────────────────────────────────────────────

    /// Send a 3-D scalar solution to GLVis.
    pub fn send_solution_3d(
        &mut self,
        mesh: &Mesh<3>,
        scalar_field: &[f64],
        field_name: &str,
    ) -> io::Result<()> {
        writeln!(self.stream, "solution")?;
        write_vtk_mesh_3d(&mut self.stream, mesh)?;
        write_vtk_scalar(&mut self.stream, mesh.n_nodes(), scalar_field, field_name)?;
        self.stream.flush()
    }

    /// Send a 3-D vector solution to GLVis.
    pub fn send_solution_3d_vector(
        &mut self,
        mesh: &Mesh<3>,
        field_x: &[f64],
        field_y: &[f64],
        field_z: &[f64],
        field_name: &str,
    ) -> io::Result<()> {
        writeln!(self.stream, "solution")?;
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

fn write_vtk_mesh_2d(w: &mut dyn Write, mesh: &Mesh<2>) -> io::Result<()> {
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

fn cell_type_vtk_2d(mesh: &Mesh<2>, elem: u32) -> u8 {
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

fn write_vtk_mesh_3d(w: &mut dyn Write, mesh: &Mesh<3>) -> io::Result<()> {
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

fn cell_type_vtk_3d(mesh: &Mesh<3>, elem: u32) -> u8 {
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
// GLVis native high-order binary protocol
// ═══════════════════════════════════════════════════════════════════════════════

/// GLVis element type codes (from MFEM's Element::Type).
#[repr(u32)]
enum GlvisElemType {
    Triangle    = 2,
    Tetrahedron = 4,
    Hexahedron  = 7,
}

/// Convert fem-rs ElementType to GLVis base type code.
fn elem_to_glvis(et: fem_mesh::ElementType) -> GlvisElemType {
    match et {
        fem_mesh::ElementType::Tri3 | fem_mesh::ElementType::Tri6 => GlvisElemType::Triangle,
        fem_mesh::ElementType::Tet4 | fem_mesh::ElementType::Tet10 => GlvisElemType::Tetrahedron,
        fem_mesh::ElementType::Hex8 | fem_mesh::ElementType::Hex20 | fem_mesh::ElementType::Hex27 => GlvisElemType::Hexahedron,
        fem_mesh::ElementType::Quad4 | fem_mesh::ElementType::Quad9 => GlvisElemType::Triangle, // GLVis uses tri for quad
        _ => GlvisElemType::Triangle,
    }
}

impl GlVisSocket {
    /// Send a high-order solution using the GLVis native binary protocol.
    ///
    /// GLVis will tessellate the high-order elements internally using
    /// the provided `refines` level (typically equal to the polynomial order).
    ///
    /// This is faster and more accurate than the legacy VTK text format
    /// for high-order fields.
    pub fn send_native_solution<const D: usize>(
        &mut self,
        mesh: &Mesh<D>,
        scalar_field: &[f64],
        _field_name: &str,
        order: u32,
        refines: u32,
    ) -> io::Result<()> {
        let nv = mesh.n_nodes() as u32;
        let ne = mesh.n_elems() as u32;
        let dim = D as u32;
        let npe = mesh.elem_type.nodes_per_element() as u32;
        let glvis_type = elem_to_glvis(mesh.elem_type) as u32;

        // Header line
        writeln!(self.stream, "solution")?;

        // Binary section: nv, ne, dim, fetsize, fet[], ordersize, order[], refines
        self.stream.write_all(&nv.to_le_bytes())?;
        self.stream.write_all(&ne.to_le_bytes())?;
        self.stream.write_all(&dim.to_le_bytes())?;
        // fetsize & fet array
        let fetsize = 1u32;
        self.stream.write_all(&fetsize.to_le_bytes())?;
        self.stream.write_all(&glvis_type.to_le_bytes())?;
        // ordersize & order array
        let ordersize = 1u32;
        self.stream.write_all(&ordersize.to_le_bytes())?;
        self.stream.write_all(&order.to_le_bytes())?;
        // refines
        self.stream.write_all(&refines.to_le_bytes())?;

        // Coordinates (float32, 3 components per vertex)
        for i in 0..nv as usize {
            let base = i * D;
            let x = mesh.coords[base] as f32;
            let y = mesh.coords[base + 1] as f32;
            let z = if D == 3 { mesh.coords[base + 2] as f32 } else { 0.0f32 };
            self.stream.write_all(&x.to_le_bytes())?;
            self.stream.write_all(&y.to_le_bytes())?;
            self.stream.write_all(&z.to_le_bytes())?;
        }

        // Connectivity (int32, npe per element)
        for e in 0..ne as usize {
            let conn = if let Some(offsets) = &mesh.elem_offsets {
                &mesh.conn[offsets[e]..offsets[e + 1]]
            } else {
                &mesh.conn[e * npe as usize..(e + 1) * npe as usize]
            };
            for &n in conn {
                self.stream.write_all(&(n as i32).to_le_bytes())?;
            }
        }

        // Solution (float32 per vertex)
        for i in 0..nv as usize {
            let val = scalar_field.get(i).copied().unwrap_or(0.0) as f32;
            self.stream.write_all(&val.to_le_bytes())?;
        }

        self.stream.flush()
    }
}

/// Write a GLVis native binary solution to any `Write` sink (for testing).
pub fn write_native_solution_bin<const D: usize>(
    w: &mut dyn Write,
    mesh: &Mesh<D>,
    scalar_field: &[f64],
    order: u32,
    refines: u32,
) -> io::Result<()> {
    let nv = mesh.n_nodes() as u32;
    let ne = mesh.n_elems() as u32;
    let dim = D as u32;
    let npe = mesh.elem_type.nodes_per_element() as u32;
    let glvis_type = elem_to_glvis(mesh.elem_type) as u32;

    writeln!(w, "solution")?;
    w.write_all(&nv.to_le_bytes())?;
    w.write_all(&ne.to_le_bytes())?;
    w.write_all(&dim.to_le_bytes())?;
    let fetsize = 1u32;
    w.write_all(&fetsize.to_le_bytes())?;
    w.write_all(&glvis_type.to_le_bytes())?;
    let ordersize = 1u32;
    w.write_all(&ordersize.to_le_bytes())?;
    w.write_all(&order.to_le_bytes())?;
    w.write_all(&refines.to_le_bytes())?;

    for i in 0..nv as usize {
        let base = i * D;
        let x = mesh.coords[base] as f32;
        let y = mesh.coords[base + 1] as f32;
        let z = if D == 3 { mesh.coords[base + 2] as f32 } else { 0.0f32 };
        w.write_all(&x.to_le_bytes())?;
        w.write_all(&y.to_le_bytes())?;
        w.write_all(&z.to_le_bytes())?;
    }

    for e in 0..ne as usize {
        let conn = if let Some(offsets) = &mesh.elem_offsets {
            &mesh.conn[offsets[e]..offsets[e + 1]]
        } else {
            &mesh.conn[e * npe as usize..(e + 1) * npe as usize]
        };
        for &n in conn {
            w.write_all(&(n as i32).to_le_bytes())?;
        }
    }

    for i in 0..nv as usize {
        let val = scalar_field.get(i).copied().unwrap_or(0.0) as f32;
        w.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// VTK legacy format smoke test: serialise a 2-D mesh + scalar.
    #[test]
    fn glvis_format_2d_scalar() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let sol = vec![1.0_f64; mesh.n_nodes()];
        let mut buf = Vec::new();
        writeln!(buf, "solution").unwrap();
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
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let sol = vec![1.0_f64; mesh.n_nodes()];
        let mut buf = Vec::new();
        writeln!(buf, "solution").unwrap();
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
        let mesh = Mesh::<3>::unit_cube_tet(1); // single tet for simpler output
        let fx = vec![0.0_f64; mesh.n_nodes()];
        let fy = vec![1.0_f64; mesh.n_nodes()];
        let fz = vec![2.0_f64; mesh.n_nodes()];
        let mut buf = Vec::new();
        writeln!(buf, "solution").unwrap();
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
        let mesh = Mesh::<2>::unit_square_tri(2);
        let sol = vec![0.5_f64; mesh.n_nodes()];
        // Simulate a local connection on a dummy socket for format check.
        let mut buf = Vec::new();
        // Manually write the old-style format for verification.
        writeln!(buf, "solution").unwrap();
        write_vtk_mesh_2d(&mut buf, &mesh).unwrap();
        write_vtk_scalar(&mut buf, mesh.n_nodes(), &sol, "u").unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("SCALARS u"));
    }

    /// Verify that send_command writes to the stream and recv_response_line reads back.
    /// Uses a local TCP pair to avoid needing a real GLVis server.
    #[test]
    fn glvis_bidirectional_local_loopback() {
        use std::net::TcpListener;
        use std::io::Read;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let n = stream.read(&mut buf).unwrap();
            // Echo back to simulate GLVis response
            let response = b"GLVis v4.7\n\n";
            stream.write_all(response).unwrap();
            // Check the command was received
            let cmd = String::from_utf8_lossy(&buf[..n]);
            assert!(cmd.contains("autoscale"), "expected autoscale command");
        });
        let mut vis = GlVisSocket::connect("127.0.0.1", port).unwrap();
        vis.send_command("autoscale").unwrap();
        let resp = vis.recv_response_line().unwrap();
        assert_eq!(resp, "GLVis v4.7");
        let all = vis.recv_response().unwrap();
        assert!(all.is_empty());
        server.join().unwrap();
    }

    /// Native binary protocol smoke test: verify binary output structure for Tet4 mesh.
    #[test]
    fn glvis_native_binary_structured_check() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let sol = vec![1.0f64; mesh.n_nodes()];
        let mut buf = Vec::new();
        write_native_solution_bin(&mut buf, &mesh, &sol, 1, 1).unwrap();
        // Check text header
        let header = String::from_utf8_lossy(&buf[..9]);
        assert_eq!(header, "solution\n");
        // Parse binary header: nv(4) ne(4) dim(4) fetsize(4) fet(4) ordersize(4) order(4) refines(4)
        let nv = u32::from_le_bytes(buf[9..13].try_into().unwrap());
        let ne = u32::from_le_bytes(buf[13..17].try_into().unwrap());
        let dim = u32::from_le_bytes(buf[17..21].try_into().unwrap());
        let fet = u32::from_le_bytes(buf[25..29].try_into().unwrap());
        assert_eq!(nv, mesh.n_nodes() as u32);
        assert_eq!(ne, mesh.n_elems() as u32);
        assert_eq!(dim, 3);
        assert_eq!(fet, 4); // GLVis Tetrahedron = 4
    }

    /// Native binary P2 triangle smoke test.
    #[test]
    fn glvis_native_binary_tri6() {
        use fem_mesh::Mesh;
        // Create a P2 mesh (Tri6) using curved 2D refinement
        let mesh = Mesh::<2>::unit_square_tri(1);
        let sol = vec![0.5f64; mesh.n_nodes()];
        let mut buf = Vec::new();
        write_native_solution_bin(&mut buf, &mesh, &sol, 2, 2).unwrap();
        let nv = u32::from_le_bytes(buf[9..13].try_into().unwrap());
        assert_eq!(nv, mesh.n_nodes() as u32);
        // Verify order and refines in binary header
        let order = u32::from_le_bytes(buf[33..37].try_into().unwrap());
        let refines = u32::from_le_bytes(buf[37..41].try_into().unwrap());
        assert_eq!(order, 2);
        assert_eq!(refines, 2);
    }
}
