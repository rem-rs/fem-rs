//! GMSH v2.2 ASCII format writer.
//!
//! Writes a [`SimplexMesh`] (plus optional boundary faces) to Gmsh
//! v2.2 ASCII format, viewable in the GMSH GUI.

use std::io::Write;

use fem_core::{FemResult};
use fem_mesh::{element_type::ElementType, topology::MeshTopology, SimplexMesh};

fn elem_type_code(et: ElementType) -> Option<u32> {
    Some(match et {
        ElementType::Line2 => 1,
        ElementType::Tri3 => 2,
        ElementType::Quad4 => 3,
        ElementType::Tet4 => 4,
        ElementType::Hex8 => 5,
        ElementType::Prism6 => 6,
        ElementType::Pyramid5 => 7,
        ElementType::Line3 => 8,
        ElementType::Tri6 => 9,
        ElementType::Quad9 => 10,
        ElementType::Tet10 => 11,
        ElementType::Hex20 => 12,
        ElementType::Point1 => 15,
        ElementType::Quad8 => 16,
        _ => return None,
    })
}

/// Write a `SimplexMesh<3>` as GMSH v2.2 ASCII.
pub fn write_msh<W: Write>(mesh: &SimplexMesh<3>, writer: &mut W) -> FemResult<()> {
    writeln!(writer, "$MeshFormat")?;
    writeln!(writer, "2.2 0 8")?;
    writeln!(writer, "$EndMeshFormat")?;

    let n_nodes = mesh.n_nodes();
    writeln!(writer, "$Nodes")?;
    writeln!(writer, "{n_nodes}")?;
    for i in 0..n_nodes {
        let c = mesh.node_coords(i as u32);
        writeln!(writer, "{} {} {} {}", i + 1, c[0], c[1], c[2])?;
    }
    writeln!(writer, "$EndNodes")?;

    let n_elems = mesh.n_elements();
    let n_faces = mesh.n_faces();
    writeln!(writer, "$Elements")?;
    writeln!(writer, "{}", n_elems + n_faces)?;

    let mut idx = 0u64;
    for e in 0..n_elems {
        idx += 1;
        let et = mesh.element_type_at(e as u32);
        let code = elem_type_code(et).unwrap_or(2);
        let nodes = mesh.element_nodes(e as u32);
        let tag = mesh.element_tag(e as u32);
        write!(writer, "{idx} {code} 1 {tag}")?;
        for &n in nodes { write!(writer, " {}", n + 1)?; }
        writeln!(writer)?;
    }

    for f in 0..n_faces {
        idx += 1;
        let et = if let Some(ref ftypes) = mesh.face_types { ftypes[f] } else { mesh.face_type };
        let code = elem_type_code(et).unwrap_or(2);
        let npe = et.nodes_per_element();
        let tag = mesh.face_tags[f] as i32;
        let off = if let Some(ref o) = mesh.face_offsets { o[f] } else { f * mesh.face_type.nodes_per_element() };
        write!(writer, "{idx} {code} 1 {tag}")?;
        for j in 0..npe { write!(writer, " {}", mesh.face_conn[off + j] + 1)?; }
        writeln!(writer)?;
    }

    writeln!(writer, "$EndElements")?;
    Ok(())
}

/// Write to a GMSH file.
pub fn write_msh_file(mesh: &SimplexMesh<3>, path: impl AsRef<std::path::Path>) -> FemResult<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write_msh(mesh, &mut f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_cube() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let mut buf = Vec::new();
        write_msh(&mesh, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.starts_with("$MeshFormat\n2.2"));
        assert!(out.contains("$Nodes\n"));
        assert!(out.contains("$Elements\n"));
        assert!(out.contains("$EndElements"));
    }
}
