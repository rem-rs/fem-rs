//! MFEM `miniapps/spde/visualizer.{hpp,cpp}` port.
//!
//! GLVis socket export is not supported by the Rust port; ParaView export
//! writes a VTK UnstructuredGrid file (ASCII) under `ParaView/` instead of a
//! ParaView DataCollection (BINARY, high-order).

use std::fs;
use std::path::Path;

use fem_io::vtk::{DataArray, VtkWriter};
use fem_mesh::Mesh;

/// Fields exported by the miniapp (random_field plus, in 3D, the topological
/// support, the imperfect topology and the level set).
pub struct Visualizer<'a, const D: usize> {
    mesh: &'a Mesh<D>,
    order: u8,
    fields: Vec<(&'static str, &'a [f64])>,
}

impl<'a, const D: usize> Visualizer<'a, D> {
    pub fn new(
        mesh: &'a Mesh<D>,
        order: u8,
        random_field: &'a [f64],
        topological_support: &'a [f64],
        imperfect_topology: &'a [f64],
        level_set: &'a [f64],
        is_3d: bool,
    ) -> Self {
        let mut fields = vec![("random_field", random_field)];
        if is_3d {
            fields.push(("topological_support", topological_support));
            fields.push(("imperfect_topology", imperfect_topology));
            fields.push(("level_set", level_set));
        }
        Self { mesh, order, fields }
    }

    /// Export the fields to `ParaView/SurrogateMaterial.vtu`.
    pub fn export_to_para_view(&self) -> std::io::Result<()> {
        fs::create_dir_all("ParaView")?;
        let path = Path::new("ParaView").join("SurrogateMaterial.vtu");

        if self.order == 1 {
            let mut w = VtkWriter::new(self.mesh);
            for (name, values) in &self.fields {
                w.add_point_data(DataArray::scalars(*name, values.to_vec()));
            }
            w.write_file(&path)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        } else {
            // Higher-order fields are tessellated one file per field.
            for (name, values) in &self.fields {
                let field_path = Path::new("ParaView")
                    .join(format!("SurrogateMaterial_{name}.vtu"));
                fem_io::vtk::write_vtu_higher_order(&field_path, self.mesh, self.order, name, values)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            }
        }
        println!("ParaView export: {}", path.display());
        Ok(())
    }

    /// GLVis is not supported: the C++ `socketstream` has no Rust counterpart.
    pub fn send_to_gl_vis(&self) {
        eprintln!(
            "generate_random_field (Rust port): GLVis export is not supported; run with -no-vis."
        );
    }
}
