//! GLVis real-time visualization via TCP socket.
//!
//! Uses the VTK legacy format and sends it over a TCP socket to a running
//! GLVis server (default `localhost:19916`).
//!
//! # Usage
//! ```no_run
//! use fem_io::glvis::GlVisSocket;
//! use fem_mesh::SimplexMesh;
//!
//! let mesh = SimplexMesh::<2>::unit_square_tri(8);
//! let sol = vec![0.5; mesh.n_nodes()];
//!
//! let mut vis = GlVisSocket::connect("localhost", 19916).unwrap();
//! vis.send_solution(&mesh, &sol, "u").unwrap();
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

    /// Send a "solution" header followed by mesh and scalar field data,
    /// then flush.
    pub fn send_solution(
        &mut self,
        mesh: &SimplexMesh<2>,
        scalar_field: &[f64],
        field_name: &str,
    ) -> io::Result<()> {
        write!(self.stream, "solution\n")?;
        write_vtk_legacy_mesh_2d(&mut self.stream, mesh)?;
        write_vtk_legacy_scalar(&mut self.stream, mesh, scalar_field, field_name)?;
        self.stream.flush()
    }

    /// Send a "solution" header followed by mesh and vector field data,
    /// then flush.
    pub fn send_solution_vector(
        &mut self,
        mesh: &SimplexMesh<2>,
        field_x: &[f64],
        field_y: &[f64],
        field_name: &str,
    ) -> io::Result<()> {
        write!(self.stream, "solution\n")?;
        write_vtk_legacy_mesh_2d(&mut self.stream, mesh)?;
        write_vtk_legacy_vector(&mut self.stream, mesh, field_x, field_y, field_name)?;
        self.stream.flush()
    }

    /// Send arbitrary VTK text followed by flush.
    pub fn send_raw(&mut self, data: &str) -> io::Result<()> {
        write!(self.stream, "{}", data)?;
        self.stream.flush()
    }
}

// ─── VTK legacy format serializers ─────────────────────────────────────────

fn write_vtk_legacy_mesh_2d(w: &mut dyn Write, mesh: &SimplexMesh<2>) -> io::Result<()> {
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
        let typ = cell_type_vtk(mesh, e as u32);
        writeln!(w, "{}", typ)?;
    }

    Ok(())
}

fn write_vtk_legacy_scalar(
    w: &mut dyn Write,
    mesh: &SimplexMesh<2>,
    field: &[f64],
    name: &str,
) -> io::Result<()> {
    let nn = mesh.n_nodes();
    writeln!(w, "POINT_DATA {}", nn)?;
    writeln!(w, "SCALARS {} float 1", name)?;
    writeln!(w, "LOOKUP_TABLE default")?;
    for i in 0..nn {
        writeln!(w, "{:.10e}", field[i])?;
    }
    Ok(())
}

fn write_vtk_legacy_vector(
    w: &mut dyn Write,
    mesh: &SimplexMesh<2>,
    fx: &[f64],
    fy: &[f64],
    name: &str,
) -> io::Result<()> {
    let nn = mesh.n_nodes();
    writeln!(w, "POINT_DATA {}", nn)?;
    writeln!(w, "VECTORS {} float", name)?;
    for i in 0..nn {
        writeln!(w, "{:.10e} {:.10e} 0.0", fx[i], fy[i])?;
    }
    Ok(())
}

fn cell_type_vtk(mesh: &SimplexMesh<2>, elem: u32) -> u8 {
    let nodes = mesh.elem_nodes(elem);
    match nodes.len() {
        3 => 5,  // VTK_TRIANGLE
        6 => 22, // VTK_QUADRATIC_TRIANGLE
        4 => 9,  // VTK_QUAD
        8 => 23, // VTK_QUADRATIC_QUAD
        9 => 28, // VTK_BIQUADRATIC_QUAD
        _ => 5,
    }
}
