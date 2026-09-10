//! Minimal NetCDF classic-format (CDF-1 / CDF-2) reader.
//!
//! This is a self-contained parser for the NetCDF "classic" binary format as
//! spoken by Cubit/Genesis Exodus II files (`*.exo`, `*.gen`), replacing the
//! netcdf-c dependency that MFEM's `Mesh::ReadCubit` relies on (MFEM links
//! netcdf-c; here the format is parsed directly).
//!
//! Supported:
//! - CDF-1 (32-bit offsets) and CDF-2 (64-bit offsets) magic variants
//! - dimensions (including the unlimited/record dimension)
//! - global/variable attributes (parsed and skipped)
//! - variables of type NC_BYTE / NC_CHAR / NC_SHORT / NC_INT / NC_FLOAT /
//!   NC_DOUBLE, including record (unlimited-first-dimension) variables
//!
//! Not supported (not produced by Cubit Genesis exports): CDF-5 (64-bit data),
//! HDF5-based NetCDF-4 files.

use fem_core::{FemError, FemResult};

// NetCDF type codes.
const NC_BYTE: u32 = 1;
const NC_CHAR: u32 = 2;
const NC_SHORT: u32 = 3;
const NC_INT: u32 = 4;
const NC_FLOAT: u32 = 5;
const NC_DOUBLE: u32 = 6;

// List tags.
const NC_DIM: u32 = 0x0A;
const NC_VAR: u32 = 0x0B;
const NC_ATT: u32 = 0x0C;

fn nc_err(msg: impl Into<String>) -> FemError {
    FemError::Mesh(format!("netcdf: {}", msg.into()))
}

fn type_size(t: u32) -> FemResult<usize> {
    Ok(match t {
        NC_BYTE | NC_CHAR => 1,
        NC_SHORT => 2,
        NC_INT | NC_FLOAT => 4,
        NC_DOUBLE => 8,
        other => return Err(nc_err(format!("unsupported type code {other}"))),
    })
}

#[derive(Debug, Clone)]
struct DimInfo {
    name: String,
    /// `0` for the unlimited (record) dimension; actual length = `numrecs`.
    size: u32,
}

#[derive(Debug, Clone)]
struct VarInfo {
    name: String,
    dimids: Vec<u32>,
    nc_type: u32,
    /// Absolute byte offset of the variable's data.
    begin: u64,
}

/// A parsed NetCDF classic file (header + backing bytes).
pub struct NetCdfFile {
    data: Vec<u8>,
    /// Number of records (length of the unlimited dimension).
    numrecs: u32,
    dims: Vec<DimInfo>,
    vars: Vec<VarInfo>,
}

impl NetCdfFile {
    /// Parse a NetCDF classic file from bytes.
    pub fn from_bytes(data: Vec<u8>) -> FemResult<Self> {
        let mut p = Parser { data: &data, pos: 0 };

        // Magic: 'C' 'D' 'F' version.
        if p.remaining() < 4
            || p.data[0] != b'C'
            || p.data[1] != b'D'
            || p.data[2] != b'F'
        {
            return Err(nc_err("not a NetCDF classic file (bad magic)"));
        }
        let version = p.data[3];
        if version != 1 && version != 2 {
            return Err(nc_err(format!(
                "unsupported NetCDF variant (version {version}; only classic CDF-1/CDF-2 supported)"
            )));
        }
        let two_byte_offsets = version == 2;
        p.pos = 4;

        let numrecs = p.u32()?;

        // dim_list
        let mut dims = Vec::new();
        let tag = p.u32()?;
        if tag == NC_DIM {
            let n = p.u32()? as usize;
            dims.reserve(n);
            for _ in 0..n {
                let name = p.name()?;
                let size = p.u32()?;
                dims.push(DimInfo { name, size });
            }
        } else if tag != 0 {
            return Err(nc_err("bad dim_list tag"));
        }

        // gatt_list (parsed and skipped)
        p.skip_attr_list(two_byte_offsets)?;

        // var_list
        let mut vars = Vec::new();
        let tag = p.u32()?;
        if tag == NC_VAR {
            let n = p.u32()? as usize;
            vars.reserve(n);
            for _ in 0..n {
                let name = p.name()?;
                let ndims = p.u32()? as usize;
                let mut dimids = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    dimids.push(p.u32()?);
                }
                p.skip_attr_list(two_byte_offsets)?;
                let nc_type = p.u32()?;
                let _vsize = p.u32()?;
                let begin = if two_byte_offsets { p.u64()? } else { p.u32()? as u64 };
                vars.push(VarInfo {
                    name,
                    dimids,
                    nc_type,
                    begin,
                });
            }
        } else if tag != 0 {
            return Err(nc_err("bad var_list tag"));
        }

        Ok(NetCdfFile {
            data,
            numrecs,
            dims,
            vars,
        })
    }

    /// Open a NetCDF file from disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> FemResult<Self> {
        let data = std::fs::read(path)?;
        NetCdfFile::from_bytes(data)
    }

    fn find_var(&self, name: &str) -> FemResult<&VarInfo> {
        self.vars
            .iter()
            .find(|v| v.name == name)
            .ok_or_else(|| nc_err(format!("no such variable '{name}'")))
    }

    /// Returns true if a dimension with that name exists.
    pub fn has_dimension(&self, name: &str) -> bool {
        self.dims.iter().any(|d| d.name == name)
    }

    /// Length of a dimension (the unlimited dimension resolves to `numrecs`).
    pub fn dimension(&self, name: &str) -> FemResult<u64> {
        self.dims
            .iter()
            .find(|d| d.name == name)
            .map(|d| if d.size == 0 { self.numrecs as u64 } else { d.size as u64 })
            .ok_or_else(|| nc_err(format!("no such dimension '{name}'")))
    }

    /// Returns true if a variable with that name exists.
    pub fn has_variable(&self, name: &str) -> bool {
        self.vars.iter().any(|v| v.name == name)
    }

    /// Read an entire variable as i32 (accepts NC_BYTE/NC_SHORT/NC_INT).
    pub fn read_var_i32(&self, name: &str) -> FemResult<Vec<i32>> {
        let v = self.find_var(name)?;
        match v.nc_type {
            NC_BYTE | NC_SHORT | NC_INT => {}
            other => {
                return Err(nc_err(format!(
                    "variable '{name}' has type code {other}, expected integer"
                )))
            }
        }
        let raw = self.read_raw(v)?;
        Ok(match v.nc_type {
            NC_BYTE => raw.into_iter().map(|b| b as i8 as i32).collect(),
            NC_SHORT => raw
                .chunks_exact(2)
                .map(|c| i16::from_be_bytes([c[0], c[1]]) as i32)
                .collect(),
            _ => raw
                .chunks_exact(4)
                .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        })
    }

    /// Read an entire variable as f64 (accepts NC_FLOAT/NC_DOUBLE).
    pub fn read_var_f64(&self, name: &str) -> FemResult<Vec<f64>> {
        let v = self.find_var(name)?;
        match v.nc_type {
            NC_FLOAT | NC_DOUBLE => {}
            other => {
                return Err(nc_err(format!(
                    "variable '{name}' has type code {other}, expected float"
                )))
            }
        }
        let raw = self.read_raw(v)?;
        Ok(match v.nc_type {
            NC_FLOAT => raw
                .chunks_exact(4)
                .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect(),
            _ => raw
                .chunks_exact(8)
                .map(|c| {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(c);
                    f64::from_be_bytes(b)
                })
                .collect(),
        })
    }

    /// Read an NC_CHAR variable as a matrix of strings.
    ///
    /// Returns one string per row of the first dimension (rows of the second
    /// dimension's length, NUL/space trimmed), e.g. `eb_names` / `ss_names`.
    pub fn read_char_rows(&self, name: &str) -> FemResult<Vec<String>> {
        let v = self.find_var(name)?;
        if v.nc_type != NC_CHAR {
            return Err(nc_err(format!("variable '{name}' is not NC_CHAR")));
        }
        let raw = self.read_raw(v)?;
        let cols: usize = v
            .dimids
            .get(1)
            .map(|&d| dim_len(&self.dims, d, self.numrecs) as usize)
            .unwrap_or(1);
        Ok(raw
            .chunks(cols)
            .map(|row| {
                let s = String::from_utf8_lossy(row);
                s.trim_end_matches(['\0', ' ']).to_string()
            })
            .collect())
    }

    /// Read the raw (big-endian) bytes of a variable, handling record layout.
    fn read_raw(&self, v: &VarInfo) -> FemResult<Vec<u8>> {
        let el = type_size(v.nc_type)?;
        let dims: Vec<u64> = v
            .dimids
            .iter()
            .map(|&d| dim_len(&self.dims, d, self.numrecs))
            .collect();
        let is_record = !dims.is_empty() && self.dims[v.dimids[0] as usize].size == 0;

        let mut out = Vec::new();
        if !is_record {
            let n: u64 = dims.iter().product::<u64>().max(1);
            let start = v.begin as usize;
            let end = start + n as usize * el;
            if end > self.data.len() {
                return Err(nc_err("variable data out of file bounds"));
            }
            out.extend_from_slice(&self.data[start..end]);
        } else {
            let per_rec: usize = if dims.len() > 1 {
                dims[1..].iter().product::<u64>() as usize
            } else {
                1
            };
            let recsize = compute_recsize(&self.dims, &self.vars);
            for r in 0..self.numrecs as usize {
                let start = v.begin as usize + r * recsize;
                let end = start + per_rec * el;
                if end > self.data.len() {
                    return Err(nc_err("record data out of file bounds"));
                }
                out.extend_from_slice(&self.data[start..end]);
            }
        }
        Ok(out)
    }
}

/// Sum of the padded record-slice sizes over all record variables.
fn compute_recsize(dims: &[DimInfo], vars: &[VarInfo]) -> usize {
    let mut recsize = 0;
    for v in vars {
        let is_record = !v.dimids.is_empty() && v.dimids[0] == 0;
        if !is_record {
            continue;
        }
        let el = type_size(v.nc_type).unwrap_or(4);
        let prod: usize = v.dimids[1..]
            .iter()
            .map(|&d| dim_len(dims, d, 0) as usize)
            .product();
        recsize += pad4(el * prod.max(1));
    }
    recsize
}

fn dim_len(dims: &[DimInfo], dimid: u32, numrecs: u32) -> u64 {
    match dims.get(dimid as usize) {
        Some(d) => {
            if d.size == 0 {
                numrecs as u64
            } else {
                d.size as u64
            }
        }
        None => 0,
    }
}

fn pad4(n: usize) -> usize {
    (n + 3) & !3
}

struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn u32(&mut self) -> FemResult<u32> {
        if self.remaining() < 4 {
            return Err(nc_err("truncated header"));
        }
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> FemResult<u64> {
        if self.remaining() < 8 {
            return Err(nc_err("truncated header"));
        }
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.data[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(u64::from_be_bytes(b))
    }

    fn name(&mut self) -> FemResult<String> {
        let n = self.u32()? as usize;
        if self.remaining() < n {
            return Err(nc_err("truncated name"));
        }
        let s = String::from_utf8_lossy(&self.data[self.pos..self.pos + n]).into_owned();
        self.pos += pad4(n);
        Ok(s)
    }

    /// Parse an attribute list (values are skipped, not stored).
    fn skip_attr_list(&mut self, _two_byte_offsets: bool) -> FemResult<()> {
        let tag = self.u32()?;
        if tag == 0 {
            return Ok(());
        }
        if tag != NC_ATT {
            return Err(nc_err("bad att_list tag"));
        }
        let n = self.u32()? as usize;
        for _ in 0..n {
            let _name = self.name()?;
            let nc_type = self.u32()?;
            let nelems = self.u32()? as usize;
            let sz = type_size(nc_type)? * nelems;
            self.pos += pad4(sz);
            if self.pos > self.data.len() {
                return Err(nc_err("truncated attribute values"));
            }
        }
        Ok(())
    }
}

// ─── Minimal NetCDF-3 writer (CDF-1, fixed dimensions) ──────────────────────

/// A typed NetCDF data payload.
#[derive(Debug, Clone)]
pub enum NcValue {
    /// NC_INT values.
    Int(Vec<i32>),
    /// NC_DOUBLE values.
    Double(Vec<f64>),
    /// NC_CHAR bytes (row-major for multi-dimensional character variables).
    Char(Vec<u8>),
}

impl NcValue {
    fn nc_type(&self) -> u32 {
        match self {
            NcValue::Int(_) => NC_INT,
            NcValue::Double(_) => NC_DOUBLE,
            NcValue::Char(_) => NC_CHAR,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            NcValue::Int(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
            NcValue::Double(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
            NcValue::Char(v) => v.clone(),
        }
    }
}

struct NcOutVar {
    name: String,
    dimids: Vec<u32>,
    values: NcValue,
    attrs: Vec<(String, NcValue)>,
}

/// Builder for a NetCDF classic (CDF-1) file with fixed (non-record)
/// dimensions — sufficient for Exodus II/Genesis mesh files.
pub struct NcOutput {
    dims: Vec<(String, u32)>,
    vars: Vec<NcOutVar>,
    globals: Vec<(String, NcValue)>,
}

impl Default for NcOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl NcOutput {
    /// Create an empty file builder.
    pub fn new() -> Self {
        NcOutput {
            dims: Vec::new(),
            vars: Vec::new(),
            globals: Vec::new(),
        }
    }

    /// Define a dimension; returns its dim id.
    pub fn add_dim(&mut self, name: &str, size: u32) -> u32 {
        self.dims.push((name.to_string(), size));
        (self.dims.len() - 1) as u32
    }

    /// Define a variable and write its data; returns the var index (for
    /// [`NcOutput::add_var_attr`]).
    pub fn add_var(&mut self, name: &str, dimids: &[u32], values: NcValue) -> usize {
        self.vars.push(NcOutVar {
            name: name.to_string(),
            dimids: dimids.to_vec(),
            values,
            attrs: Vec::new(),
        });
        self.vars.len() - 1
    }

    /// Convenience: define an NC_INT variable.
    pub fn add_var_i32(&mut self, name: &str, dimids: &[u32], data: &[i32]) -> usize {
        self.add_var(name, dimids, NcValue::Int(data.to_vec()))
    }

    /// Convenience: define an NC_DOUBLE variable.
    pub fn add_var_f64(&mut self, name: &str, dimids: &[u32], data: &[f64]) -> usize {
        self.add_var(name, dimids, NcValue::Double(data.to_vec()))
    }

    /// Attach an attribute to a previously defined variable.
    pub fn add_var_attr(&mut self, var_index: usize, name: &str, value: NcValue) {
        self.vars[var_index].attrs.push((name.to_string(), value));
    }

    /// Add a global attribute.
    pub fn add_global_attr(&mut self, name: &str, value: NcValue) {
        self.globals.push((name.to_string(), value));
    }

    /// Serialize the file (CDF-1, big-endian, 32-bit offsets).
    pub fn finish(self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"CDF\x01");
        out.extend_from_slice(&0u32.to_be_bytes()); // numrecs (no record dim)

        // dim_list
        if self.dims.is_empty() {
            out.extend_from_slice(&0u32.to_be_bytes());
        } else {
            out.extend_from_slice(&NC_DIM.to_be_bytes());
            out.extend_from_slice(&(self.dims.len() as u32).to_be_bytes());
            for (name, size) in &self.dims {
                write_nc_name(&mut out, name);
                out.extend_from_slice(&size.to_be_bytes());
            }
        }

        // gatt_list
        write_attr_list(&mut out, &self.globals);

        // var_list: build entries first (with a begin placeholder), then patch.
        let mut var_entries: Vec<Vec<u8>> = Vec::new();
        for v in &self.vars {
            let mut e = Vec::new();
            write_nc_name(&mut e, &v.name);
            e.extend_from_slice(&(v.dimids.len() as u32).to_be_bytes());
            for &d in &v.dimids {
                e.extend_from_slice(&d.to_be_bytes());
            }
            write_attr_list(&mut e, &v.attrs);
            e.extend_from_slice(&v.values.nc_type().to_be_bytes());
            let raw = v.values.bytes().len();
            e.extend_from_slice(&(raw as u32).to_be_bytes()); // vsize (unpadded)
            e.extend_from_slice(&0u32.to_be_bytes()); // begin placeholder
            var_entries.push(e);
        }

        if var_entries.is_empty() {
            out.extend_from_slice(&0u32.to_be_bytes());
        } else {
            out.extend_from_slice(&NC_VAR.to_be_bytes());
            out.extend_from_slice(&(var_entries.len() as u32).to_be_bytes());
            for e in &var_entries {
                out.extend_from_slice(e);
            }
        }

        // Patch the begin offsets, then append the (padded) data.
        let mut begin = out.len() as u32;
        let mut entry_pos = out.len() - var_entries.iter().map(|e| e.len()).sum::<usize>();
        for (e, v) in var_entries.iter().zip(self.vars.iter()) {
            let at = entry_pos + e.len() - 4;
            out[at..at + 4].copy_from_slice(&begin.to_be_bytes());
            entry_pos += e.len();
            begin += pad4(v.values.bytes().len()) as u32;
        }
        for v in &self.vars {
            out.extend_from_slice(&v.values.bytes());
            while out.len() % 4 != 0 {
                out.push(0);
            }
        }
        out
    }
}

fn write_attr_list(out: &mut Vec<u8>, attrs: &[(String, NcValue)]) {
    if attrs.is_empty() {
        out.extend_from_slice(&0u32.to_be_bytes());
        return;
    }
    out.extend_from_slice(&NC_ATT.to_be_bytes());
    out.extend_from_slice(&(attrs.len() as u32).to_be_bytes());
    for (name, value) in attrs {
        write_nc_name(out, name);
        out.extend_from_slice(&value.nc_type().to_be_bytes());
        let nelems = match value {
            NcValue::Int(v) => v.len(),
            NcValue::Double(v) => v.len(),
            NcValue::Char(v) => v.len(),
        };
        out.extend_from_slice(&(nelems as u32).to_be_bytes());
        out.extend_from_slice(&value.bytes());
        while out.len() % 4 != 0 {
            out.push(0);
        }
    }
}

fn write_nc_name(out: &mut Vec<u8>, name: &str) {
    out.extend_from_slice(&(name.len() as u32).to_be_bytes());
    out.extend_from_slice(name.as_bytes());
    while out.len() % 4 != 0 {
        out.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_fixed_dims() {
        let mut w = NcOutput::new();
        let d_n = w.add_dim("num_nodes", 4);
        let d_s = w.add_dim("len_string", 8);
        let d_b = w.add_dim("num_el_blk", 2);
        w.add_var_f64("coordx", &[d_n], &[1.0, 2.0, 3.0, 4.5]);
        let blk = w.add_var("eb_prop1", &[d_b], NcValue::Int(vec![1, 2]));
        w.add_global_attr("title", NcValue::Char(b"roundtrip".to_vec()));
        w.add_var_attr(blk, "name", NcValue::Char(b"ab".to_vec()));
        w.add_var(
            "eb_names",
            &[d_b, d_s],
            NcValue::Char(b"block_a blk2    ".to_vec()),
        );
        let bytes = w.finish();

        let f = NetCdfFile::from_bytes(bytes).expect("parse failed");
        assert!(f.has_dimension("num_nodes"));
        assert_eq!(f.dimension("num_nodes").unwrap(), 4);
        assert!(!f.has_dimension("missing"));
        assert!(f.has_variable("coordx"));
        assert!(!f.has_variable("missing"));

        let x = f.read_var_f64("coordx").unwrap();
        assert_eq!(x, vec![1.0, 2.0, 3.0, 4.5]);

        let ids = f.read_var_i32("eb_prop1").unwrap();
        assert_eq!(ids, vec![1, 2]);

        let names = f.read_char_rows("eb_names").unwrap();
        assert_eq!(names, vec!["block_a".to_string(), "blk2".to_string()]);
    }
}
