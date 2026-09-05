//! MFEM `DataCollection` (VisIt format) writer.
//!
//! Mirrors the layout MFEM 4.10 produces for `VisItDataCollection::Save()`
//! (see `miniapps/electromagnetics/*` `WriteVisItFields` and the sample
//! `<prefix>_<cycle>.mfem_root` files):
//!
//! ```text
//! <prefix>_<cycle>            directory (per-rank files when n_ranks > 1)
//!   mesh.<rank %06d>          MFEM mesh text (write_mfem format)
//!   <field>.<rank %06d>       MFEM grid-function text
//! <prefix>_<cycle>.mfem_root  JSON root ("dsets.main": cycle/domains/fields…)
//! ```
//!
//! The root is hand-serialized JSON (no serde dependency): the structure is
//! fixed by MFEM and the only escaping needed is `"`/`\` in names.

use std::fs;
use std::io::Write;
use std::path::Path;

/// A grid-function field to store in the collection.
pub struct DcField {
    pub name: String,
    /// MFEM finite-element-collection name, e.g. `"H1_2D_P4"`.
    pub basis: String,
    /// Polynomial order (also used for the `lod`/`order` tags).
    pub order: u32,
    /// Number of vector components (VDim).
    pub vdim: u32,
    /// DOF values (one per line in the written gf file).
    pub values: Vec<f64>,
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Serialize the root file `{"dsets":{"main":{…}}}`.
fn root_json(prefix: &str, cycle: usize, domains: usize, topo_dim: u32,
             spatial_dim: u32, fields: &[DcField]) -> String {
    let dir = format!("{prefix}_{cycle:06}");
    let mut s = String::new();
    s.push_str("{\n  \"dsets\": {\n    \"main\": {\n");
    s.push_str(&format!("      \"cycle\": {cycle},\n"));
    s.push_str(&format!("      \"domains\": {domains},\n"));
    s.push_str("      \"fields\": {\n");
    for (i, f) in fields.iter().enumerate() {
        s.push_str(&format!(
            "        {}: {{\n          \"path\": {}",
            json_escape(&f.name),
            json_escape(&format!("{dir}/{}.%06d", f.name))
        ));
        s.push_str(&format!(
            ",\n          \"tags\": {{\n            \"assoc\": \"nodes\",\n            \"basis\": {},\n            \"comps\": \"{}\",\n            \"lod\": \"{}\",\n            \"order\": \"{}\"\n          }}\n        }}",
            json_escape(&f.basis),
            f.vdim,
            f.order,
            f.order
        ));
        if i + 1 < fields.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("      },\n");
    s.push_str(&format!(
        "      \"mesh\": {{\n        \"format\": \"0\",\n        \"path\": {},\n        \"tags\": {{\n          \"max_lods\": \"32\",\n          \"spatial_dim\": \"{}\",\n          \"topo_dim\": \"{}\"\n        }}\n      }},\n",
        json_escape(&format!("{dir}/mesh.%06d")),
        spatial_dim,
        topo_dim
    ));
    s.push_str(&format!("      \"time\": {cycle},\n      \"time_step\": 0\n"));
    s.push_str("    }\n  }\n}\n");
    s
}

/// Write one grid-function slice file (`FiniteElementSpace` header + values).
fn gf_text(basis: &str, vdim: u32, values: &[f64]) -> String {
    let mut s = String::new();
    s.push_str("FiniteElementSpace\n");
    s.push_str(&format!("FiniteElementCollection: {basis}\n"));
    s.push_str(&format!("VDim: {vdim}\n"));
    s.push_str("Ordering: 0\n\n");
    for v in values {
        s.push_str(&format!("{v:.14e}\n"));
    }
    s
}

/// Save a VisIt-style DataCollection under `out_dir` (default ".").
///
/// * `prefix` — collection root name (e.g. `"Volta-AMR-Parallel"`).
/// * `cycle` — time/cycle index (forms the directory/root suffix).
/// * `rank`/`n_ranks` — MPI-style slice index (single process: 0/1).
/// * `mesh_txt` — the mesh serialized in MFEM text format (`write_mfem`).
pub fn save_visit_collection(
    out_dir: &str,
    prefix: &str,
    cycle: usize,
    rank: u32,
    n_ranks: u32,
    topo_dim: u32,
    spatial_dim: u32,
    mesh_txt: &str,
    fields: &[DcField],
) -> std::io::Result<()> {
    let dir_name = format!("{prefix}_{cycle:06}");
    let dir = Path::new(out_dir).join(&dir_name);
    fs::create_dir_all(&dir)?;

    // Mesh slice.
    let mesh_file = dir.join(format!("mesh.{rank:06}"));
    fs::write(&mesh_file, mesh_txt)?;

    // Field slices.
    for f in fields {
        let fname = dir.join(format!("{}.{rank:06}", f.name));
        fs::write(&fname, gf_text(&f.basis, f.vdim, &f.values))?;
    }

    // Root file (one per rank in MFEM writes a root only from rank 0; the
    // JSON carries the domain count for the reader).
    if rank == 0 {
        let root_name = format!("{prefix}_{cycle:06}.mfem_root");
        let root_path = Path::new(out_dir).join(&root_name);
        let root = root_json(prefix, cycle, n_ranks as usize, topo_dim,
                             spatial_dim, fields);
        let mut f = fs::File::create(&root_path)?;
        f.write_all(root.as_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_matches_mfem_layout() {
        let dir = std::env::temp_dir().join("femio_dc_test");
        let _ = fs::remove_dir_all(&dir);
        let mesh_txt = "MFEM mesh v1.0\n\ndimension\n2\n";
        let fields = vec![DcField {
            name: "Rho Source".to_string(),
            basis: "H1_2D_P4".to_string(),
            order: 4,
            vdim: 1,
            values: vec![0.25, -1.5e-11],
        }];
        save_visit_collection(dir.to_str().unwrap(), "Example23", 0, 0, 1,
                              2, 2, mesh_txt, &fields)
            .expect("save failed");

        // Files mirror the C++ sample: dir with mesh.000000/field.000000 +
        // root file "<prefix>_000000.mfem_root".
        let dir0 = dir.join("Example23_000000");
        assert!(dir0.join("mesh.000000").exists());
        assert!(dir0.join("Rho Source.000000").exists());
        let root = fs::read_to_string(dir.join("Example23_000000.mfem_root"))
            .expect("root file");
        assert!(root.contains("\"cycle\": 0"));
        assert!(root.contains("\"domains\": 1"));
        assert!(root.contains("\"basis\": \"H1_2D_P4\""));
        assert!(root.contains("Example23_000000/mesh.%06d"));
        // Field name with a space must be JSON-escaped/quoted.
        assert!(root.contains("\"Rho Source\""));
        // gf slice header.
        let gf = fs::read_to_string(dir0.join("Rho Source.000000")).unwrap();
        assert!(gf.starts_with("FiniteElementSpace\nFiniteElementCollection: H1_2D_P4\n"));
        let _ = fs::remove_dir_all(&dir);
    }
}
