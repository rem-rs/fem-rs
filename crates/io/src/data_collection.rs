//! MFEM `DataCollection` (VisIt format) writer/reader.
//!
//! Mirrors the layout MFEM 4.10 produces for `VisItDataCollection::Save()`
//! (`fem/datacollection.cpp`: `DataCollection::Save`/`SaveMesh`/`SaveOneField`
//! + `VisItDataCollection::SaveRootFile`/`GetVisItRootString`):
//!
//! ```text
//! <prefix_path><name>_<cycle %06d>            directory (one file per rank)
//!   mesh.<rank %06d>                          MFEM mesh text (`Mesh::Print`)
//!   <field>.<rank %06d>                       MFEM grid-function text
//! <prefix_path><name>_<cycle %06d>.mfem_root  JSON root, rank 0 only
//! ```
//!
//! Formatting notes (all verified against 4.10 output, see the tests):
//!
//! * The root file is `picojson::value::serialize(true)`: 2-space indent, keys
//!   in `std::map` (lexicographic) order, one trailing `\n` after the outer `}`.
//! * Tag values are JSON *strings* (`"comps": "2"`), `cycle`/`domains`/`time`/
//!   `time_step` are numbers written with picojson's rule: `%.f` when integral
//!   and `|v| < 2^53`, otherwise `%.17g`.
//! * Grid-function DOFs use the stream's precision 6 (`precision_default`),
//!   i.e. C's `%.6g`; subnormal values are flushed to zero
//!   (`ZeroSubnormal`). One DOF per line for `Ordering: 0` (byNODES), `VDim`
//!   values per line for `Ordering: 1` (byVDIM) — `Vector::Print`.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Digits of the cycle/rank suffixes (`pad_digits_default`).
const PAD_DIGITS: usize = 6;
/// `DataCollection::precision_default`, the ostream precision of every write.
const PRECISION: usize = 6;
/// `VisItDataCollection::visit_max_levels_of_detail`.
const MAX_LODS: u32 = 32;
/// Largest integral double representable exactly (`2^53`), picojson's cutoff
/// between the `%.f` and `%.17g` number formats.
const PICOJSON_INT_LIMIT: f64 = 9007199254740992.0;

/// Mesh container format (`DataCollection::SetFormat`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DcFormat {
    /// `SERIAL_FORMAT`: `data` file is `mesh`, `"format": "0"`.
    #[default]
    Serial,
    /// `PARALLEL_FORMAT`: mesh file is `pmesh`, `"format": "1"`.
    Parallel,
}

impl DcFormat {
    fn code(self) -> u32 {
        match self {
            DcFormat::Serial => 0,
            DcFormat::Parallel => 1,
        }
    }

    /// `DataCollection::GetMeshShortFileName`.
    fn mesh_stem(self) -> &'static str {
        match self {
            DcFormat::Serial => "mesh",
            DcFormat::Parallel => "pmesh",
        }
    }
}

/// One registered grid function (`VisItDataCollection::RegisterField`).
#[derive(Clone, Debug)]
pub struct DcField {
    pub name: String,
    /// MFEM finite-element-collection name, e.g. `"H1_2D_P4"`.
    pub basis: String,
    /// `FEColl()->GetOrder()` — the `"order"` root tag.
    pub order: u32,
    /// `std::max(1, max element order)` — the `"lod"` root tag. Starts at 1 in
    /// C++, so a P0 field has `lod == 1` while `order == 0`.
    pub lod: u32,
    /// Number of vector components (VDim) — the `"comps"` root tag.
    pub vdim: u32,
    /// FES ordering: `false` = `Ordering: 0` (byNODES, one DOF per line),
    /// `true` = `Ordering: 1` (byVDIM, `vdim` DOFs per line).
    pub by_vdim: bool,
    /// DOF values in the FES's own ordering.
    pub values: Vec<f64>,
}

impl DcField {
    /// Register a nodal (`assoc: "nodes"`) grid function: the common case,
    /// `LOD` derived as C++ does (`max(1, order)`).
    pub fn nodes(
        name: impl Into<String>,
        basis: impl Into<String>,
        order: u32,
        vdim: u32,
        values: Vec<f64>,
    ) -> Self {
        DcField {
            name: name.into(),
            basis: basis.into(),
            order,
            lod: order.max(1),
            vdim,
            by_vdim: false,
            values,
        }
    }

    /// Override the `"lod"` tag (variable-order spaces, explicit LODs).
    pub fn with_lod(mut self, lod: u32) -> Self {
        self.lod = lod;
        self
    }

    /// Write the DOFs with `Ordering: 1` (byVDIM) rows.
    pub fn by_vdim(mut self) -> Self {
        self.by_vdim = true;
        self
    }
}

/// A VisIt-style `DataCollection` step, mirroring the C++ object's state.
///
/// ```no_run
/// use fem_io::data_collection::{DcField, VisItCollection};
/// let mut dc = VisItCollection::new("Maxwell-Parallel");
/// dc.set_cycle(3).set_time(0.5);
/// dc.register_field(DcField::nodes("E", "H1_2D_P1", 1, 3, vec![0.0; 12]));
/// dc.save(0, "MFEM mesh v1.0\n\ndimension\n2\n").unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct VisItCollection {
    /// Collection name (`DataCollection::name`).
    pub name: String,
    /// Directory prefix (`DataCollection::SetPrefixPath`), `/`-terminated.
    pub prefix_path: String,
    /// Step index; forms the `_%06d` suffix of the directory and root file.
    pub cycle: usize,
    /// Simulation time — the `"time"` root number.
    pub time: f64,
    /// Time-step size — the `"time_step"` root number.
    pub time_step: f64,
    /// Number of pieces; the `"domains"` root number and `<rank>` in paths.
    pub n_ranks: u32,
    /// Mesh container format.
    pub format: DcFormat,
    /// `"max_lods"` mesh tag (`visit_max_levels_of_detail`, default 32).
    pub max_lods: u32,
    /// `spatial_dim` mesh tag (from the mesh that will be written).
    pub spatial_dim: u32,
    /// `topo_dim` mesh tag.
    pub topo_dim: u32,
    /// Registered fields, written in lexicographic name order (as `std::map`).
    pub fields: Vec<DcField>,
}

impl VisItCollection {
    pub fn new(name: impl Into<String>) -> Self {
        VisItCollection {
            name: name.into(),
            prefix_path: String::new(),
            cycle: 0,
            time: 0.0,
            time_step: 0.0,
            n_ranks: 1,
            format: DcFormat::Serial,
            max_lods: MAX_LODS,
            spatial_dim: 0,
            topo_dim: 0,
            fields: Vec::new(),
        }
    }

    /// `DataCollection::SetPrefixPath` — a trailing `/` is appended when
    /// missing (an empty prefix clears it).
    pub fn set_prefix_path(&mut self, prefix: &str) -> &mut Self {
        if prefix.is_empty() {
            self.prefix_path.clear();
        } else {
            self.prefix_path = if prefix.ends_with('/') {
                prefix.to_string()
            } else {
                format!("{prefix}/")
            };
        }
        self
    }

    pub fn set_cycle(&mut self, cycle: usize) -> &mut Self {
        self.cycle = cycle;
        self
    }

    pub fn set_time(&mut self, time: f64) -> &mut Self {
        self.time = time;
        self
    }

    pub fn set_time_step(&mut self, time_step: f64) -> &mut Self {
        self.time_step = time_step;
        self
    }

    pub fn set_format(&mut self, format: DcFormat) -> &mut Self {
        self.format = format;
        self
    }

    pub fn set_max_lods(&mut self, max_lods: u32) -> &mut Self {
        self.max_lods = max_lods;
        self
    }

    /// `VisItDataCollection::RegisterField` — replaces a same-named field.
    pub fn register_field(&mut self, field: DcField) -> &mut Self {
        self.fields.retain(|f| f.name != field.name);
        self.fields.push(field);
        self
    }

    /// Collection directory name, e.g. `"Maxwell-Parallel_000003"`.
    pub fn dir_name(&self) -> String {
        format!("{}_{:0width$}", self.name, self.cycle, width = PAD_DIGITS)
    }

    /// Mesh slice file name inside the collection dir (with the rank suffix).
    pub fn mesh_file_name(&self, rank: u32) -> String {
        format!("{}.{:0width$}", self.format.mesh_stem(), rank, width = PAD_DIGITS)
    }

    /// `DataCollection::GetFieldFileName`.
    pub fn field_file_name(&self, field: &str, rank: u32) -> String {
        format!("{field}.{:0width$}", rank, width = PAD_DIGITS)
    }

    /// `VisItDataCollection::Save()` + `SaveRootFile()`: create the
    /// `<prefix_path><name>_<cycle>` directory, write the mesh slice and one
    /// file per registered field (all with the rank suffix), and write the
    /// `.mfem_root` JSON on rank 0 only.
    pub fn save(&self, rank: u32, mesh_txt: &str) -> std::io::Result<()> {
        let base = PathBuf::from(&self.prefix_path);
        let dir = base.join(self.dir_name());
        fs::create_dir_all(&dir)?;

        fs::write(dir.join(self.mesh_file_name(rank)), mesh_txt)?;
        for f in &self.fields {
            fs::write(
                dir.join(self.field_file_name(&f.name, rank)),
                gf_text(f),
            )?;
        }

        if rank == 0 {
            let root = base.join(format!("{}.mfem_root", self.dir_name()));
            let mut file = fs::File::create(root)?;
            file.write_all(self.root_json().as_bytes())?;
        }
        Ok(())
    }

    /// `VisItDataCollection::GetVisItRootString` — the byte-exact root file
    /// body (including the single trailing newline).
    pub fn root_json(&self) -> String {
        let dir = self.dir_name();
        let fields: Vec<&DcField> = {
            let mut v: Vec<&DcField> = self.fields.iter().collect();
            v.sort_by(|a, b| a.name.cmp(&b.name));
            v
        };

        let mut s = String::new();
        s.push_str("{\n  \"dsets\": {\n    \"main\": {\n");
        s.push_str(&format!("      \"cycle\": {},\n", number(self.cycle as f64)));
        s.push_str(&format!("      \"domains\": {},\n", number(self.n_ranks as f64)));
        if !fields.is_empty() {
            s.push_str("      \"fields\": {\n");
            for (i, f) in fields.iter().enumerate() {
                s.push_str(&format!("        {}: {{\n", json_escape(&f.name)));
                s.push_str(&format!(
                    "          \"path\": {},\n",
                    json_escape(&format!("{dir}/{}.%06d", f.name))
                ));
                s.push_str("          \"tags\": {\n");
                s.push_str("            \"assoc\": \"nodes\",\n");
                s.push_str(&format!(
                    "            \"basis\": {},\n",
                    json_escape(&f.basis)
                ));
                s.push_str(&format!("            \"comps\": \"{}\",\n", f.vdim));
                s.push_str(&format!("            \"lod\": \"{}\",\n", f.lod));
                s.push_str(&format!("            \"order\": \"{}\"\n", f.order));
                s.push_str("          }\n        }");
                s.push_str(if i + 1 < fields.len() { ",\n" } else { "\n" });
            }
            s.push_str("      },\n");
        }
        s.push_str("      \"mesh\": {\n");
        s.push_str(&format!("        \"format\": \"{}\",\n", self.format.code()));
        s.push_str(&format!(
            "        \"path\": {},\n",
            json_escape(&format!("{dir}/{}.%06d", self.format.mesh_stem()))
        ));
        s.push_str("        \"tags\": {\n");
        s.push_str(&format!("          \"max_lods\": \"{}\",\n", self.max_lods));
        s.push_str(&format!("          \"spatial_dim\": \"{}\",\n", self.spatial_dim));
        s.push_str(&format!("          \"topo_dim\": \"{}\"\n", self.topo_dim));
        s.push_str("        }\n      },\n");
        s.push_str(&format!("      \"time\": {},\n", number(self.time)));
        s.push_str(&format!("      \"time_step\": {}\n", number(self.time_step)));
        s.push_str("    }\n  }\n}\n");
        s
    }
}

/// JSON string literal, matching `picojson::serialize_str`.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// picojson's `value::to_str` for a number: `%.f` for exactly-representable
/// small integers, `%.17g` otherwise.
fn number(v: f64) -> String {
    if v.abs() < PICOJSON_INT_LIMIT && v.fract() == 0.0 {
        format!("{v:.0}")
    } else {
        format_g(v, 17)
    }
}

/// C's `printf("%.{prec}g", v)` (glibc: `%.*g` with trailing zeros stripped,
/// scientific when the decimal exponent is `< -4` or `>= prec`).
fn format_g(v: f64, prec: usize) -> String {
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    // `%.{prec-1}e` gives the correctly-rounded `prec` significant digits and
    // the exponent that `%g` uses for its fixed/scientific decision.
    let sci = format!("{:.*e}", prec - 1, v);
    let (mant, exp) = sci.split_once('e').expect("rust {:e} always has 'e'");
    let exp: i32 = exp.parse().expect("exponent is an integer");
    let neg = mant.starts_with('-');
    let digits: Vec<u8> = mant
        .bytes()
        .filter(|b| b.is_ascii_digit())
        .collect();

    let mut out = String::new();
    if neg {
        out.push('-');
    }
    let max_exp = prec as i32;
    if exp < -4 || exp >= max_exp {
        // Scientific: d[.ddd]e±XX, fraction trailing zeros stripped, the
        // exponent always signed with at least two digits.
        let mut frac = String::from_utf8(digits[1..].to_vec()).unwrap();
        while frac.ends_with('0') {
            frac.pop();
        }
        out.push(digits[0] as char);
        if !frac.is_empty() {
            out.push('.');
            out.push_str(&frac);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        let a = exp.unsigned_abs();
        if a < 10 {
            out.push('0');
        }
        out.push_str(&a.to_string());
        out
    } else {
        // Fixed: place the decimal point and strip trailing fraction zeros.
        if exp >= 0 {
            let ip = exp as usize + 1;
            if ip >= digits.len() {
                out.push_str(&String::from_utf8(digits.clone()).unwrap());
                for _ in digits.len()..ip {
                    out.push('0');
                }
            } else {
                out.push_str(&String::from_utf8(digits[..ip].to_vec()).unwrap());
                out.push('.');
                out.push_str(&String::from_utf8(digits[ip..].to_vec()).unwrap());
            }
        } else {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(&String::from_utf8(digits).unwrap());
        }
        if out.contains('.') {
            while out.ends_with('0') {
                out.pop();
            }
            if out.ends_with('.') {
                out.pop();
            }
        }
        out
    }
}

/// One scalar as written by `Vector::Print` / `operator<<` at precision 6,
/// with subnormal flushing (`ZeroSubnormal`).
fn gf_value(v: f64) -> String {
    if v.is_subnormal() {
        "0".to_string()
    } else {
        format_g(v, PRECISION)
    }
}

/// `FiniteElementSpace::Save` + `GridFunction::Save` header.
fn gf_header(f: &DcField) -> String {
    format!(
        "FiniteElementSpace\nFiniteElementCollection: {}\nVDim: {}\nOrdering: {}\n\n",
        f.basis,
        f.vdim,
        if f.by_vdim { 1 } else { 0 }
    )
}

/// Write one grid-function slice file (`FiniteElementSpace` header + values).
fn gf_text(f: &DcField) -> String {
    let mut s = gf_header(f);
    // Vector::Print(os, width) with width = 1 for byNODES, VDim for byVDIM; no
    // output at all for an empty vector.
    let width = if f.by_vdim { f.vdim.max(1) as usize } else { 1 };
    for (i, v) in f.values.iter().enumerate() {
        s.push_str(&gf_value(*v));
        if i + 1 == f.values.len() {
            break;
        }
        s.push(if (i + 1) % width == 0 { '\n' } else { ' ' });
    }
    if !f.values.is_empty() {
        s.push('\n');
    }
    s
}

/// Parse a `.mfem_root` JSON file and extract field metadata.
pub fn read_visit_root(root_path: &Path) -> std::io::Result<(usize, usize, Vec<DcField>)> {
    let content = fs::read_to_string(root_path)?;
    let content = content.trim();

    // Extract cycle
    let cycle = extract_json_number(content, "\"cycle\":")
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing cycle"))?;

    // Extract domains
    let domains = extract_json_number(content, "\"domains\":")
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing domains"))?;

    // Extract fields - find "fields": { ... } and parse each
    // "name": { "path": ..., "tags": { "basis": ..., "comps": ..., ... } }
    // entry with brace matching (handles both the C++ VisIt layout, where
    // fields nest inside dsets.main, and the writer's own layout).
    let mut fields = Vec::new();
    if let Some(fields_start) = content.find("\"fields\":") {
        let after_key = &content[fields_start..];
        if let Some(rel) = after_key.find('{') {
            let obj = match match_braces(&after_key[rel..]) {
                Some(span) => &after_key[rel + 1..rel + span - 1],
                None => "",
            };
            // Iterate top-level entries "name": { ... }.
            let bytes = obj.as_bytes();
            let mut i = 0usize;
            while i < bytes.len() {
                if bytes[i] != b'"' {
                    i += 1;
                    continue;
                }
                // key string
                let Some(key_end) = obj[i + 1..].find('"') else { break };
                let name = &obj[i + 1..i + 1 + key_end];
                let mut j = i + 1 + key_end + 1;
                // skip to value
                while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b':') {
                    j += 1;
                }
                if j >= bytes.len() || bytes[j] != b'{' {
                    i = j;
                    continue;
                }
                // brace-match the value object
                let Some(span) = match_braces(&obj[j..]) else { break };
                let body = &obj[j..j + span];
                let basis = extract_json_str(body, "\"basis\"");
                let vdim = extract_json_usize(body, "\"comps\"").unwrap_or(1) as u32;
                let order = extract_json_usize(body, "\"order\"").unwrap_or(1) as u32;
                let lod = extract_json_usize(body, "\"lod\"").unwrap_or(order.max(1) as usize) as u32;
                if let Some(basis) = basis {
                    let mut f = DcField::nodes(name.to_string(), basis.to_string(), order, vdim, Vec::new());
                    f.lod = lod;
                    fields.push(f);
                }
                i = j + span;
            }
        }
    }

    Ok((cycle, domains, fields))
}

/// Match a `{...}` block starting at `s[0] == '{'`; returns the span length
/// including both braces.
fn match_braces(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract a quoted string value following `"key":`.
fn extract_json_str<'a>(content: &'a str, key: &str) -> Option<&'a str> {
    let pos = content.find(key)? + key.len();
    let rest = &content[pos..];
    let rest = rest.trim_start();
    let rest = rest.strip_prefix(':').unwrap_or(rest).trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(&rest[..end])
}

/// Extract a numeric value following `"key":` (bare or quoted).
fn extract_json_usize(content: &str, key: &str) -> Option<usize> {
    if let Some(v) = extract_json_number(content, key) {
        return Some(v);
    }
    extract_json_str(content, key)?.trim().parse().ok()
}

fn extract_json_number(content: &str, key: &str) -> Option<usize> {
    let pos = content.find(key)? + key.len();
    let rest = &content[pos..];
    let rest = rest.trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Read a mesh slice file (MFEM text format).
pub fn read_mesh_slice(path: &Path) -> std::io::Result<String> {
    fs::read_to_string(path)
}

/// Read a grid-function slice file.
pub fn read_gf_slice(path: &Path) -> std::io::Result<(String, u32, Vec<f64>)> {
    let content = fs::read_to_string(path)?;
    let mut lines = content.lines();
    // Skip header lines
    let mut basis = String::new();
    let mut vdim = 1u32;
    for line in &mut lines {
        if line.starts_with("FiniteElementCollection:") {
            basis = line["FiniteElementCollection:".len()..].trim().to_string();
        } else if line.starts_with("VDim:") {
            vdim = line["VDim:".len()..].trim().parse().unwrap_or(1);
        } else if line.is_empty() {
            break;
        }
    }
    // Read values
    let mut values = Vec::new();
    for line in lines {
        for tok in line.split_whitespace() {
            if let Ok(v) = tok.parse::<f64>() {
                values.push(v);
            }
        }
    }
    Ok((basis, vdim, values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_g_matches_c_printf() {
        // Reference strings from glibc printf (see tmp/fmt_probe.c).
        assert_eq!(format_g(0.0, 17), "0");
        assert_eq!(format_g(0.125, 17), "0.125");
        assert_eq!(format_g(2.5, 17), "2.5");
        assert_eq!(format_g(1.0 / 3.0, 17), "0.33333333333333331");
        assert_eq!(format_g(-1.0 / 3.0, 17), "-0.33333333333333331");
        assert_eq!(format_g(0.1, 17), "0.10000000000000001");
        assert_eq!(format_g(1e-9, 17), "1.0000000000000001e-09");
        assert_eq!(format_g(1e20, 17), "1e+20");
        assert_eq!(format_g(1e17, 17), "1e+17");
        assert_eq!(format_g(1e18, 17), "1e+18");
        assert_eq!(format_g(5e-324, 17), "4.9406564584124654e-324");
        assert_eq!(format_g(1e-300, 17), "1e-300");
        assert_eq!(format_g(0.0001, 17), "0.0001");
        // precision 6 (ostream default) forms.
        assert_eq!(format_g(0.0, 6), "0");
        assert_eq!(format_g(-0.0, 6), "-0");
        assert_eq!(format_g(1.0 / 3.0, 6), "0.333333");
        assert_eq!(format_g(-1.0 / 3.0, 6), "-0.333333");
        assert_eq!(format_g(-1.0, 6), "-1");
        assert_eq!(format_g(100.14285714285714, 6), "100.143");
        assert_eq!(format_g(1e-9, 6), "1e-09");
        assert_eq!(format_g(1234567.0, 6), "1.23457e+06");
        assert_eq!(format_g(123456789.0, 6), "1.23457e+08");
        assert_eq!(format_g(0.000123456789, 6), "0.000123457");
        assert_eq!(format_g(999999.5, 6), "1e+06");
        assert_eq!(format_g(3.14159265358979, 6), "3.14159");
        assert_eq!(format_g(1.0 / 7.0, 6), "0.142857");
    }

    #[test]
    fn number_matches_picojson() {
        assert_eq!(number(0.0), "0");
        assert_eq!(number(1.0), "1");
        assert_eq!(number(32.0), "32");
        assert_eq!(number(-2.0), "-2");
        assert_eq!(number(0.125), "0.125");
        assert_eq!(number(2.5), "2.5");
        assert_eq!(number(100.14285714285714), "100.14285714285714");
        assert_eq!(number(1e20), "1e+20");
    }

    /// The C++ probe `tmp/visit_probe.cpp` output (serial, 2x2 quad mesh,
    /// fields E/Phi/"Rho Source"/V, cycles 0 and 7) reproduced byte for byte.
    fn probe_mesh_txt() -> &'static str {
        "MFEM mesh v1.0\n\n#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n\
         # POINT       = 0\n# SEGMENT     = 1\n# TRIANGLE    = 2\n# SQUARE      = 3\n\
         # TETRAHEDRON = 4\n# CUBE        = 5\n# PRISM       = 6\n# PYRAMID     = 7\n#\n\
         \ndimension\n2\n\nelements\n4\n1 3 0 1 4 3\n1 3 3 4 7 6\n1 3 4 5 8 7\n1 3 1 2 5 4\n\
         \nboundary\n8\n1 1 0 1\n1 1 1 2\n3 1 7 6\n3 1 8 7\n4 1 3 0\n4 1 6 3\n2 1 2 5\n2 1 5 8\n\
         \nvertices\n9\n2\n0 0\n0.5 0\n1 0\n0 0.5\n0.5 0.5\n1 0.5\n0 1\n0.5 1\n1 1\n"
    }

    fn probe_dc(cycle: usize, time: f64, time_step: f64) -> VisItCollection {
        let mut dc = VisItCollection::new("Probe");
        dc.set_cycle(cycle).set_time(time).set_time_step(time_step);
        // Phi: H1_2D_P1 scalar; values 0.1*(i+1) - 1/3.
        let phi: Vec<f64> = (0..9).map(|i| 0.1 * (i as f64 + 1.0) - 1.0 / 3.0).collect();
        dc.register_field(DcField::nodes("Phi", "H1_2D_P1", 1, 1, phi));
        // E: H1_2D_P1 vdim 2; values -0.25*(i+1) + 1e-9.
        let e: Vec<f64> = (0..18).map(|i| -0.25 * (i as f64 + 1.0) + 1e-9).collect();
        dc.register_field(DcField::nodes("E", "H1_2D_P1", 1, 2, e));
        // "Rho Source": L2_2D_P0 — order 0 but lod 1 (C++ clamps LOD at 1).
        let p0: Vec<f64> = (0..4).map(|i| i as f64 + 0.5).collect();
        dc.register_field(DcField::nodes("Rho Source", "L2_2D_P0", 0, 1, p0));
        // V: H1_2D_P1 vdim 2 written byVDIM (rows of 2).
        let v: Vec<f64> = (0..18).map(|i| 100.0 + i as f64 + 1.0 / 7.0).collect();
        dc.register_field(DcField::nodes("V", "H1_2D_P1", 1, 2, v).by_vdim());
        dc.spatial_dim = 2;
        dc.topo_dim = 2;
        dc
    }

    #[test]
    fn root_json_matches_cpp() {
        let dc = probe_dc(0, 0.0, 0.0);
        // Copied verbatim from the C++ probe's Probe_000000.mfem_root.
        #[rustfmt::skip]
        let expected = [
            "{",
            "  \"dsets\": {",
            "    \"main\": {",
            "      \"cycle\": 0,",
            "      \"domains\": 1,",
            "      \"fields\": {",
            "        \"E\": {",
            "          \"path\": \"Probe_000000/E.%06d\",",
            "          \"tags\": {",
            "            \"assoc\": \"nodes\",",
            "            \"basis\": \"H1_2D_P1\",",
            "            \"comps\": \"2\",",
            "            \"lod\": \"1\",",
            "            \"order\": \"1\"",
            "          }",
            "        },",
            "        \"Phi\": {",
            "          \"path\": \"Probe_000000/Phi.%06d\",",
            "          \"tags\": {",
            "            \"assoc\": \"nodes\",",
            "            \"basis\": \"H1_2D_P1\",",
            "            \"comps\": \"1\",",
            "            \"lod\": \"1\",",
            "            \"order\": \"1\"",
            "          }",
            "        },",
            "        \"Rho Source\": {",
            "          \"path\": \"Probe_000000/Rho Source.%06d\",",
            "          \"tags\": {",
            "            \"assoc\": \"nodes\",",
            "            \"basis\": \"L2_2D_P0\",",
            "            \"comps\": \"1\",",
            "            \"lod\": \"1\",",
            "            \"order\": \"0\"",
            "          }",
            "        },",
            "        \"V\": {",
            "          \"path\": \"Probe_000000/V.%06d\",",
            "          \"tags\": {",
            "            \"assoc\": \"nodes\",",
            "            \"basis\": \"H1_2D_P1\",",
            "            \"comps\": \"2\",",
            "            \"lod\": \"1\",",
            "            \"order\": \"1\"",
            "          }",
            "        }",
            "      },",
            "      \"mesh\": {",
            "        \"format\": \"0\",",
            "        \"path\": \"Probe_000000/mesh.%06d\",",
            "        \"tags\": {",
            "          \"max_lods\": \"32\",",
            "          \"spatial_dim\": \"2\",",
            "          \"topo_dim\": \"2\"",
            "        }",
            "      },",
            "      \"time\": 0,",
            "      \"time_step\": 0",
            "    }",
            "  }",
            "}",
            "",
        ].join("\n");
        assert_eq!(dc.root_json(), expected);
    }

    #[test]
    fn root_json_nonzero_cycle_and_time() {
        let dc = probe_dc(7, 0.125, 2.5);
        let root = dc.root_json();
        assert!(root.contains("\"cycle\": 7"));
        assert!(root.contains("\"time\": 0.125"));
        assert!(root.contains("\"time_step\": 2.5"));
        assert!(root.contains("Probe_000007/E.%06d"));
    }

    #[test]
    fn gf_text_matches_cpp() {
        let dc = probe_dc(0, 0.0, 0.0);
        let phi = dc.fields.iter().find(|f| f.name == "Phi").unwrap();
        assert_eq!(
            gf_text(phi),
            "FiniteElementSpace\nFiniteElementCollection: H1_2D_P1\nVDim: 1\nOrdering: 0\n\n\
             -0.233333\n-0.133333\n-0.0333333\n0.0666667\n0.166667\n\
             0.266667\n0.366667\n0.466667\n0.566667\n"
        );
        // byNODES vdim-2 field: still one DOF per line.
        let e = dc.fields.iter().find(|f| f.name == "E").unwrap();
        assert_eq!(
            gf_text(e),
            "FiniteElementSpace\nFiniteElementCollection: H1_2D_P1\nVDim: 2\nOrdering: 0\n\n\
             -0.25\n-0.5\n-0.75\n-1\n-1.25\n-1.5\n-1.75\n-2\n-2.25\n-2.5\n-2.75\n-3\n\
             -3.25\n-3.5\n-3.75\n-4\n-4.25\n-4.5\n"
        );
        // P0 field: lod clamped at 1, order 0.
        let p0 = dc.fields.iter().find(|f| f.name == "Rho Source").unwrap();
        assert_eq!(
            gf_text(p0),
            "FiniteElementSpace\nFiniteElementCollection: L2_2D_P0\nVDim: 1\nOrdering: 0\n\n\
             0.5\n1.5\n2.5\n3.5\n"
        );
        // byVDIM: VDim values per line, space separated.
        let v = dc.fields.iter().find(|f| f.name == "V").unwrap();
        assert_eq!(
            gf_text(v),
            "FiniteElementSpace\nFiniteElementCollection: H1_2D_P1\nVDim: 2\nOrdering: 1\n\n\
             100.143 101.143\n102.143 103.143\n104.143 105.143\n106.143 107.143\n\
             108.143 109.143\n110.143 111.143\n112.143 113.143\n114.143 115.143\n\
             116.143 117.143\n"
        );
    }

    #[test]
    fn save_writes_cpp_file_set() {
        let dir = std::env::temp_dir().join("femio_dc_probe");
        let _ = fs::remove_dir_all(&dir);
        let dc = probe_dc(0, 0.0, 0.0);
        let dc = {
            let mut d = dc;
            d.set_prefix_path(dir.to_str().unwrap());
            d
        };
        dc.save(0, probe_mesh_txt()).expect("save failed");

        let step = dir.join("Probe_000000");
        assert!(step.join("mesh.000000").exists());
        for f in ["E", "Phi", "Rho Source", "V"] {
            assert!(step.join(format!("{f}.000000")).exists(), "missing {f}");
        }
        assert!(dir.join("Probe_000000.mfem_root").exists());
        assert!(read_gf_slice(&step.join("V.000000")).is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_root_round_trip() {
        let dir = std::env::temp_dir().join("femio_dc_roundtrip");
        let _ = fs::remove_dir_all(&dir);
        let mut dc = probe_dc(7, 0.125, 2.5);
        dc.set_prefix_path(dir.to_str().unwrap());
        dc.save(0, probe_mesh_txt()).expect("save failed");

        let (cycle, domains, fields) =
            read_visit_root(&dir.join("Probe_000007.mfem_root")).expect("read failed");
        assert_eq!(cycle, 7);
        assert_eq!(domains, 1);
        assert_eq!(fields.len(), 4);
        let p0 = fields.iter().find(|f| f.name == "Rho Source").unwrap();
        assert_eq!(p0.basis, "L2_2D_P0");
        assert_eq!(p0.order, 0);
        assert_eq!(p0.lod, 1);
        assert_eq!(p0.vdim, 1);
        let (basis, vdim, values) = read_gf_slice(&dir.join("Probe_000007/V.000000")).unwrap();
        assert_eq!(basis, "H1_2D_P1");
        assert_eq!(vdim, 2);
        assert_eq!(values.len(), 18);
        // DOFs are written at ostream precision 6, so 100 + 1/7 round-trips as
        // "100.143".
        assert_eq!(values[0], 100.143);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parallel_format_uses_pmesh() {
        let mut dc = VisItCollection::new("P");
        dc.set_format(DcFormat::Parallel);
        dc.n_ranks = 4;
        dc.spatial_dim = 3;
        dc.topo_dim = 3;
        dc.set_cycle(3);
        let root = dc.root_json();
        assert!(root.contains("\"format\": \"1\""));
        assert!(root.contains("P_000003/pmesh.%06d"));
        assert!(root.contains("\"domains\": 4"));
        assert_eq!(dc.mesh_file_name(2), "pmesh.000002");
        assert_eq!(dc.field_file_name("Phi", 3), "Phi.000003");
    }

    #[test]
    fn empty_collection_omits_fields() {
        let mut dc = VisItCollection::new("Empty");
        dc.spatial_dim = 1;
        dc.topo_dim = 1;
        let root = dc.root_json();
        assert!(!root.contains("fields"));
        assert!(root.contains("\"mesh\": {"));
    }

    #[test]
    fn subnormal_values_flush_to_zero() {
        let f = DcField::nodes("f", "H1_2D_P1", 1, 1, vec![5e-324, 1.0]);
        assert_eq!(
            gf_text(&f),
            "FiniteElementSpace\nFiniteElementCollection: H1_2D_P1\nVDim: 1\nOrdering: 0\n\n0\n1\n"
        );
    }
}
