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
