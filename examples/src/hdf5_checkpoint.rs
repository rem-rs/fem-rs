use fem_io_hdf5_parallel::RankFieldF64;

pub fn scalar_rank_field_f64(name: &str, value: f64) -> RankFieldF64 {
    RankFieldF64 {
        name: name.into(),
        global_offset: 0,
        global_len: 1,
        values: vec![value],
    }
}

pub fn vector_rank_field_f64(name: &str, values: Vec<f64>) -> RankFieldF64 {
    RankFieldF64 {
        name: name.into(),
        global_offset: 0,
        global_len: values.len() as u64,
        values,
    }
}

#[cfg(feature = "io_hdf5")]
use std::path::Path;

#[cfg(feature = "io_hdf5")]
use fem_io_hdf5_parallel::{
    materialize_global_field_f64,
    write_xdmf_polyvertex_scalar_timeseries_sidecar,
};

#[cfg(feature = "io_hdf5")]
pub fn checkpoint_sidecar_path(h5_path: &str, field: &str) -> Result<String, String> {
    let stem = Path::new(h5_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "invalid checkpoint file name".to_string())?;
    let parent = Path::new(h5_path).parent().unwrap_or_else(|| Path::new(""));
    Ok(parent
        .join(format!("{stem}_{field}.xdmf"))
        .to_string_lossy()
        .to_string())
}

#[cfg(feature = "io_hdf5")]
pub fn write_scalar_checkpoint_xdmf_sidecars(
    h5_path: &str,
    step: u64,
    time: f64,
    fields: &[&str],
) -> Result<(), String> {
    for &field in fields {
        let global_len = materialize_global_field_f64(h5_path, 1, step, field)
            .map_err(|e| e.to_string())?;
        let xdmf_path = checkpoint_sidecar_path(h5_path, field)?;
        write_xdmf_polyvertex_scalar_timeseries_sidecar(
            &xdmf_path,
            h5_path,
            field,
            global_len,
            &[(step, time)],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}