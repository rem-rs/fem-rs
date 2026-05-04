//! # XDMF metadata writer
//!
//! Generates XDMF (eXtensible Data Model and Format) XML metadata files that
//! describe mesh topology, geometry, and field data stored in external HDF5
//! files.  ParaView reads `.xmf` files to visualise FEM results.
//!
//! ## Features
//!
//! * **Serial mode** (`num_ranks == 1`): single `<Grid>` referencing one HDF5.
//! * **Parallel mode** (`num_ranks > 1`): `<Grid GridType="Collection">` with
//!   per-rank `<Grid GridType="Uniform">` sub-grids.
//!
//! ## Usage
//!
//! ```ignore
//! use fem_io::xdmf::{write_xdmf, XdmfField, XdmfCenter};
//! use fem_mesh::ElementType;
//!
//! let fields = vec![
//!     XdmfField {
//!         name: "temperature".into(),
//!         hdf5_path: "solution.h5".into(),
//!         dataset_path: "/fields/temperature/values".into(),
//!         center: XdmfCenter::Node,
//!     },
//! ];
//!
//! write_xdmf(
//!     "solution.xmf",
//!     4,                           // 4 MPI ranks
//!     ElementType::Tet4,
//!     3,                           // 3D
//!     &[125, 125, 125, 125],       // nodes per rank
//!     &[64, 64, 64, 64],           // elems per rank
//!     "solution_rank_{}.h5",       // "{}" replaced by rank
//!     &fields,
//! ).unwrap();
//! ```

use std::io::{self, Write};
use std::path::Path;

use fem_mesh::ElementType;

// ── public types ─────────────────────────────────────────────────────────────

/// Where field data is centered (nodes or cells).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdmfCenter {
    /// Data defined per node.
    Node,
    /// Data defined per element (cell).
    Cell,
}

impl XdmfCenter {
    fn as_xml_attr(&self) -> &'static str {
        match self {
            XdmfCenter::Node => "Node",
            XdmfCenter::Cell => "Cell",
        }
    }
}

/// Metadata for one field to be referenced in the XDMF file.
#[derive(Debug, Clone)]
pub struct XdmfField {
    /// Display name in ParaView.
    pub name: String,
    /// Path to the HDF5 file (e.g. `"solution_rank_0.h5"`).
    pub hdf5_path: String,
    /// Dataset path inside the HDF5 file (e.g. `"/fields/temperature/values"`).
    pub dataset_path: String,
    /// Center type: `Node` or `Cell`.
    pub center: XdmfCenter,
}

// ── ElementType → XDMF topology string ──────────────────────────────────────

/// Map `ElementType` to its XDMF topology name.
fn xdmf_topology_type(elem: ElementType) -> &'static str {
    match elem {
        ElementType::Line2 => "Polyvertex",
        ElementType::Line3 => "Polyvertex",
        ElementType::Tri3 => "Triangle",
        ElementType::Tri6 => "Triangle_6",
        ElementType::Quad4 => "Quadrilateral",
        ElementType::Quad8 => "Quadrilateral_8",
        ElementType::Tet4 => "Tetrahedron",
        ElementType::Tet10 => "Tetrahedron_10",
        ElementType::Hex8 => "Hexahedron",
        ElementType::Hex20 => "Hexahedron_20",
        ElementType::Prism6 => "Wedge",
        ElementType::Pyramid5 => "Pyramid",
        _ => "Mixed",
    }
}

/// Return the XDMF v3 integer topology type code for use in Mixed topology arrays.
///
/// Mixed topology arrays interleave `[type_code, n0, n1, …]` for each element.
/// This function returns the XDMF type code for the most common 3-D volume elements:
///
/// | ElementType  | XDMF code |
/// |--------------|-----------|
/// | Tet4         | 6         |
/// | Pyramid5     | 7         |
/// | Prism6       | 8         |
/// | Hex8         | 9         |
/// | Tri3         | 4         |
/// | Quad4        | 5         |
pub fn xdmf_topology_code(elem: ElementType) -> u32 {
    match elem {
        ElementType::Tri3     => 4,
        ElementType::Quad4    => 5,
        ElementType::Tet4     => 6,
        ElementType::Pyramid5 => 7,
        ElementType::Prism6   => 8,
        ElementType::Hex8     => 9,
        _                     => 1, // Polyvertex fallback
    }
}

/// Number of nodes per element for topology (same as ElementType::nodes_per_element).
fn nodes_per_element(elem: ElementType) -> usize {
    elem.nodes_per_element()
}

/// XDMF geometry type string from spatial dimension.
fn xdmf_geometry_type(dim: usize) -> &'static str {
    match dim {
        2 => "XY",
        3 => "XYZ",
        _ => panic!("xdmf: unsupported spatial dimension {dim} (only 2 or 3)"),
    }
}

// ── XML generation helpers ───────────────────────────────────────────────────

/// Write an XML opening tag with optional attributes.
fn open_tag<W: io::Write>(
    w: &mut W,
    name: &str,
    attrs: &[(&str, &str)],
    indent: usize,
) -> io::Result<()> {
    let prefix = "  ".repeat(indent);
    write!(w, "{prefix}<{name}")?;
    for (k, v) in attrs {
        // Escape quotes in attribute values
        let escaped = v.replace('"', "&quot;");
        write!(w, " {k}=\"{escaped}\"")?;
    }
    writeln!(w, ">")?;
    Ok(())
}

/// Write an XML close tag.
fn close_tag<W: io::Write>(w: &mut W, name: &str, indent: usize) -> io::Result<()> {
    let prefix = "  ".repeat(indent);
    writeln!(w, "{prefix}</{name}>")?;
    Ok(())
}

/// Write `<DataItem>` with inner text referencing HDF5 data.
fn write_data_item_with_text<W: io::Write>(
    w: &mut W,
    dimensions: &str,
    hdf5_path: &str,
    dataset_path: &str,
    indent: usize,
) -> io::Result<()> {
    let prefix = "  ".repeat(indent);
    let content_prefix = "  ".repeat(indent + 1);
    let number_type = "Float";
    let precision = "8";
    let format = "HDF";
    let full_path = format!("{hdf5_path}:{dataset_path}");
    write!(
        w,
        "{prefix}<DataItem DataType=\"{number_type}\" Precision=\"{precision}\" \
         Format=\"{format}\" Dimensions=\"{dimensions}\">\n"
    )?;
    writeln!(w, "{content_prefix}{full_path}")?;
    writeln!(w, "{prefix}</DataItem>")?;
    Ok(())
}

// ── main entry point ─────────────────────────────────────────────────────────

/// Write an XDMF metadata file.
///
/// # Arguments
///
/// * `path` — Output `.xmf` file path.
/// * `num_ranks` — Number of MPI ranks (1 = serial).
/// * `elem_type` — Element type for connectivity topology.
/// * `dim` — Spatial dimension (2 = XY, 3 = XYZ).
/// * `n_nodes_per_rank` — Number of (owned) nodes per rank.
/// * `n_elems_per_rank` — Number of elements per rank.
/// * `hdf5_path_template` — Template with `{}` for rank number, e.g.
///   `"solution_rank_{}.h5"`.  For serial mode the `{}` is replaced with `0`.
/// * `fields` — Field metadata array.
pub fn write_xdmf(
    path: impl AsRef<Path>,
    num_ranks: usize,
    elem_type: ElementType,
    dim: usize,
    n_nodes_per_rank: &[usize],
    n_elems_per_rank: &[usize],
    hdf5_path_template: &str,
    fields: &[XdmfField],
) -> io::Result<()> {
    let mut xml = Vec::<u8>::new();

    // ── XML declaration ──────────────────────────────────────────────────────
    writeln!(xml, r#"<?xml version="1.0" ?>"#)?;
    writeln!(xml, r#"<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>"#)?;
    writeln!(xml, r#"<Xdmf Version="3.0">"#)?;
    writeln!(xml, r#"  <Domain>"#)?;

    let topology_name = xdmf_topology_type(elem_type);
    let npe = nodes_per_element(elem_type);
    let geometry_name = xdmf_geometry_type(dim);

    if num_ranks == 1 {
        // ── Serial mode: single Uniform grid ────────────────────────────────
        let nn = n_nodes_per_rank.first().copied().unwrap_or(0);
        let ne = n_elems_per_rank.first().copied().unwrap_or(0);
        let h5_path = hdf5_path_template.replace("{}", "0");

        write_grid(
            &mut xml,
            "grid",
            topology_name,
            npe,
            ne,
            geometry_name,
            dim,
            nn,
            &h5_path,
            fields,
            2,
        )?;
    } else {
        // ── Parallel mode: Collection of per-rank grids ──────────────────────
        writeln!(
            xml,
            r#"    <Grid GridType="Collection" CollectionType="Spatial">"#
        )?;

        for rank in 0..num_ranks {
            let nn = n_nodes_per_rank.get(rank).copied().unwrap_or(0);
            let ne = n_elems_per_rank.get(rank).copied().unwrap_or(0);
            let h5_path = hdf5_path_template.replace("{}", &rank.to_string());
            let grid_name = format!("rank_{rank}");

            write_grid(
                &mut xml,
                &grid_name,
                topology_name,
                npe,
                ne,
                geometry_name,
                dim,
                nn,
                &h5_path,
                fields,
                3,
            )?;
        }

        writeln!(xml, r#"    </Grid>"#)?;
    }

    writeln!(xml, r#"  </Domain>"#)?;
    writeln!(xml, r#"</Xdmf>"#)?;

    // ── Write to file ────────────────────────────────────────────────────────
    let mut file = std::fs::File::create(path.as_ref())?;
    file.write_all(&xml)?;

    Ok(())
}

// ── per-grid writer ──────────────────────────────────────────────────────────

/// Write a single `<Grid GridType="Uniform">` block.
#[allow(clippy::too_many_arguments)]
fn write_grid<W: io::Write>(
    w: &mut W,
    name: &str,
    topology_name: &str,
    npe: usize,
    ne: usize,
    geometry_name: &str,
    dim: usize,
    nn: usize,
    h5_path: &str,
    fields: &[XdmfField],
    indent: usize,
) -> io::Result<()> {
    let p = indent;

    open_tag(w, "Grid", &[("Name", name), ("GridType", "Uniform")], p)?;

    // ── Topology ─────────────────────────────────────────────────────────────
    let topo_dims = format!("{ne} {npe}");
    open_tag(
        w,
        "Topology",
        &[
            ("TopologyType", topology_name),
            ("NumberOfElements", &ne.to_string()),
        ],
        p + 1,
    )?;
    write_data_item_with_text(w, &topo_dims, h5_path, "/mesh/conn", p + 2)?;
    close_tag(w, "Topology", p + 1)?;

    // ── Geometry ─────────────────────────────────────────────────────────────
    let geom_dims = format!("{nn} {dim}");
    open_tag(w, "Geometry", &[("GeometryType", geometry_name)], p + 1)?;
    write_data_item_with_text(w, &geom_dims, h5_path, "/mesh/coords", p + 2)?;
    close_tag(w, "Geometry", p + 1)?;

    // ── Fields ───────────────────────────────────────────────────────────────
    for field in fields {
        let dim_str = match field.center {
            XdmfCenter::Node => format!("{nn}"),
            XdmfCenter::Cell => format!("{ne}"),
        };

        open_tag(
            w,
            "Attribute",
            &[
                ("Name", &field.name),
                ("AttributeType", "Scalar"),
                ("Center", field.center.as_xml_attr()),
            ],
            p + 1,
        )?;
        write_data_item_with_text(w, &dim_str, h5_path, &field.dataset_path, p + 2)?;
        close_tag(w, "Attribute", p + 1)?;
    }

    close_tag(w, "Grid", p)?;
    Ok(())
}

// ── mixed-topology helper ─────────────────────────────────────────────────────

/// Write an XDMF file for a mixed-element-type mesh stored in one HDF5 file.
///
/// The HDF5 file must contain:
/// - `/mesh/coords`     — float64 array of shape `[n_nodes, dim]`
/// - `/mesh/conn_mixed` — int64 flat array with interleaved XDMF type codes
///   `[code, n0, n1, …, code, n0, n1, …]`  (see [`xdmf_topology_code`])
///
/// Use this function when your mesh contains multiple element types
/// (e.g. Tet4 + Prism6).  For uniform-type meshes, prefer [`write_xdmf`].
///
/// # Arguments
/// * `path`             — Output `.xmf` path.
/// * `n_nodes`          — Total node count.
/// * `n_elems`          — Total element count.
/// * `mixed_conn_len`   — Length of the flat mixed connectivity array
///   (sum over all elements of `1 + nodes_per_elem`).
/// * `dim`              — Spatial dimension (2 or 3).
/// * `hdf5_path`        — Path to the HDF5 file containing mesh + field data.
/// * `fields`           — Field metadata.
pub fn write_xdmf_mixed(
    path: impl AsRef<Path>,
    n_nodes: usize,
    n_elems: usize,
    mixed_conn_len: usize,
    dim: usize,
    hdf5_path: &str,
    fields: &[XdmfField],
) -> io::Result<()> {
    let mut xml = Vec::<u8>::new();
    let geometry_name = xdmf_geometry_type(dim);

    writeln!(xml, r#"<?xml version="1.0" ?>"#)?;
    writeln!(xml, r#"<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>"#)?;
    writeln!(xml, r#"<Xdmf Version="3.0">"#)?;
    writeln!(xml, r#"  <Domain>"#)?;

    open_tag(&mut xml, "Grid", &[("Name", "mixed_mesh"), ("GridType", "Uniform")], 2)?;

    // Mixed topology
    open_tag(
        &mut xml,
        "Topology",
        &[
            ("TopologyType", "Mixed"),
            ("NumberOfElements", &n_elems.to_string()),
        ],
        3,
    )?;
    let conn_dims = format!("{mixed_conn_len}");
    write_data_item_with_text(&mut xml, &conn_dims, hdf5_path, "/mesh/conn_mixed", 4)?;
    close_tag(&mut xml, "Topology", 3)?;

    // Geometry
    let geom_dims = format!("{n_nodes} {dim}");
    open_tag(&mut xml, "Geometry", &[("GeometryType", geometry_name)], 3)?;
    write_data_item_with_text(&mut xml, &geom_dims, hdf5_path, "/mesh/coords", 4)?;
    close_tag(&mut xml, "Geometry", 3)?;

    // Fields
    for field in fields {
        let dim_str = match field.center {
            XdmfCenter::Node => format!("{n_nodes}"),
            XdmfCenter::Cell => format!("{n_elems}"),
        };
        open_tag(
            &mut xml,
            "Attribute",
            &[
                ("Name", &field.name),
                ("AttributeType", "Scalar"),
                ("Center", field.center.as_xml_attr()),
            ],
            3,
        )?;
        write_data_item_with_text(&mut xml, &dim_str, hdf5_path, &field.dataset_path, 4)?;
        close_tag(&mut xml, "Attribute", 3)?;
    }

    close_tag(&mut xml, "Grid", 2)?;
    writeln!(xml, r#"  </Domain>"#)?;
    writeln!(xml, r#"</Xdmf>"#)?;

    let mut file = std::fs::File::create(path.as_ref())?;
    file.write_all(&xml)?;
    Ok(())
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xdmf_topology_code_values() {
        assert_eq!(xdmf_topology_code(ElementType::Tet4),     6);
        assert_eq!(xdmf_topology_code(ElementType::Pyramid5), 7);
        assert_eq!(xdmf_topology_code(ElementType::Prism6),   8);
        assert_eq!(xdmf_topology_code(ElementType::Hex8),     9);
        assert_eq!(xdmf_topology_code(ElementType::Tri3),     4);
        assert_eq!(xdmf_topology_code(ElementType::Quad4),    5);
    }

    #[test]
    fn write_xdmf_serial_tet4_produces_valid_xml() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_xdmf_serial_{}.xmf", std::process::id()));
        write_xdmf(
            &path,
            1,
            ElementType::Tet4,
            3,
            &[8],
            &[4],
            "mesh.h5",
            &[XdmfField {
                name: "u".into(),
                hdf5_path: "mesh.h5".into(),
                dataset_path: "/fields/u".into(),
                center: XdmfCenter::Node,
            }],
        )
        .unwrap();
        let txt = std::fs::read_to_string(&path).unwrap();
        assert!(txt.contains("Tetrahedron"));
        assert!(txt.contains("NumberOfElements=\"4\""));
        assert!(txt.contains("/fields/u"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_xdmf_parallel_creates_collection() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_xdmf_parallel_{}.xmf", std::process::id()));
        write_xdmf(
            &path,
            3,
            ElementType::Hex8,
            3,
            &[8, 8, 8],
            &[1, 1, 1],
            "solution_{}.h5",
            &[],
        )
        .unwrap();
        let txt = std::fs::read_to_string(&path).unwrap();
        assert!(txt.contains("CollectionType=\"Spatial\""));
        assert!(txt.contains("rank_0"));
        assert!(txt.contains("rank_2"));
        assert!(txt.contains("Hexahedron"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_xdmf_mixed_produces_mixed_topology_xml() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_xdmf_mixed_{}.xmf", std::process::id()));
        // Mixed Tet4 (1+4 tokens) + Prism6 (1+6 tokens) = 11 mixed conn entries.
        let mixed_conn_len = (1 + 4) + (1 + 6);
        write_xdmf_mixed(
            &path,
            8,
            2,
            mixed_conn_len,
            3,
            "mixed_mesh.h5",
            &[XdmfField {
                name: "temperature".into(),
                hdf5_path: "mixed_mesh.h5".into(),
                dataset_path: "/fields/T".into(),
                center: XdmfCenter::Cell,
            }],
        )
        .unwrap();
        let txt = std::fs::read_to_string(&path).unwrap();
        assert!(txt.contains("TopologyType=\"Mixed\""));
        assert!(txt.contains("NumberOfElements=\"2\""));
        assert!(txt.contains("conn_mixed"));
        assert!(txt.contains("/fields/T"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_xdmf_prism6_topology_name() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_xdmf_prism6_{}.xmf", std::process::id()));
        write_xdmf(&path, 1, ElementType::Prism6, 3, &[6], &[1], "mesh.h5", &[]).unwrap();
        let txt = std::fs::read_to_string(&path).unwrap();
        assert!(txt.contains("Wedge"), "Prism6 should use XDMF topology 'Wedge'");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_xdmf_pyramid5_topology_name() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_xdmf_pyramid5_{}.xmf", std::process::id()));
        write_xdmf(&path, 1, ElementType::Pyramid5, 3, &[5], &[1], "mesh.h5", &[]).unwrap();
        let txt = std::fs::read_to_string(&path).unwrap();
        assert!(txt.contains("Pyramid"), "Pyramid5 should use XDMF topology 'Pyramid'");
        let _ = std::fs::remove_file(path);
    }
}
