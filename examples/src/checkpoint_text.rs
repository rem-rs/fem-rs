use std::{fs, io, path::Path};

pub fn ensure_parent_dir(path: &str) -> io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

pub fn format_vec_f64(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>()
        .join(",")
}

pub fn parse_vec_f64(value: &str) -> Result<Vec<f64>, String> {
    if value.trim().is_empty() {
        Ok(Vec::new())
    } else {
        value
            .split(',')
            .map(|entry| entry.parse::<f64>().map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()
    }
}