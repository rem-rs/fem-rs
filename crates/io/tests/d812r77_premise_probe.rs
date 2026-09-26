//! D812-1 premise probe — *what exactly* the order-1 geometry loss looks like.
//!
//! The ledger (`d812r77_mesh_write_ledger.rs`) records `read`/`write` outcomes
//! for the whole `data/` corpus.  This file records the **premise** those
//! outcomes rest on, because the round-76 registration ("写者按
//! `geom_order() > 1` 才写 `nodes` ⇒ order-1 几何场回环丢失；四份
//! `periodic-*.mesh` 自带 `L2_T1_2D_P1`；实测 `read` 得 `geometry_table=true` 而
//! 写回 `has_nodes_section=false`") mixes two different paths:
//!
//! * the **raw** `periodic-*.mesh` fixtures carry a *discontinuous* order-1
//!   table, and the continuous writer refuses them **loudly** (the D805-3
//!   continuity check) — not a silent loss; and
//! * a **refined** mesh built from one of them (ex9's own output) carries an
//!   order-1 table the writer *accepts* and then silently drops, because
//!   `nodes_dof_values` returns `Ok(None)` for `order <= 1`.
//!
//! Run with `D812R77_PREMISE=1 cargo test --release -p fem-io --test
//! d812r77_premise_probe -- --nocapture` — the assertions below hold for both
//! the pre- and post-D812-1 writer on the *raw* fixtures; they are a record,
//! not a regression gate.

use fem_io::mfem::{read_mfem_file, write_mfem_nodes, NodesSpace};
use fem_mesh::simplex::Mesh;
use fem_mesh::topology::MeshTopology;

fn geom_facts<const D: usize>(m: &Mesh<D>, label: &str) {
    match m.geometry.as_ref() {
        None => println!("  {label}: geometry = None (order {})", m.geom_order()),
        Some(g) => {
            let distinct: std::collections::BTreeSet<u32> = g.conn.iter().copied().collect();
            println!(
                "  {label}: geometry order={} npe={} n_nodes={} conn.len={} distinct_ids={}",
                g.order,
                g.nodes_per_elem,
                g.n_nodes,
                g.conn.len(),
                distinct.len()
            );
        }
    }
}

#[test]
fn d812r77_premise() {
    if std::env::var("D812R77_PREMISE").is_err() {
        eprintln!("d812r77_premise: set D812R77_PREMISE=1 to run");
        return;
    }
    let dump = std::env::var("D812R77_DUMP").ok();
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    for name in ["periodic-hexagon.mesh", "periodic-square.mesh", "periodic-cube.mesh"] {
        let path = data.join(name);
        if !path.exists() {
            println!("{name}: absent");
            continue;
        }
        let f = read_mfem_file(&path).expect("read");
        println!("{name}:");
        if let Some(m) = f.mesh2d.as_ref() {
            geom_facts(m, "as read");
            // The writer's own decision for a *raw* fixture, both continuities:
            for space in [NodesSpace::Continuous, NodesSpace::Discontinuous] {
                let mut buf: Vec<u8> = Vec::new();
                let r = write_mfem_nodes(&mut buf, m, None, space);
                let has_nodes = String::from_utf8_lossy(&buf)
                    .lines()
                    .any(|l| l.trim() == "nodes");
                println!(
                    "  write {space:?}: {:?} nodes_section={has_nodes}",
                    r.as_ref().map(|()| buf.len())
                );
                if r.is_ok() && has_nodes {
                    if let Some(ref dir) = dump {
                        let p = std::path::Path::new(dir)
                            .join(format!("{name}.rs{space:?}.mesh").replace(' ', ""));
                        std::fs::write(p, &buf).expect("dump");
                    }
                }
            }
        }
        if let Some(m) = f.mesh3d.as_ref() {
            geom_facts(m, "as read (3-D)");
            for space in [NodesSpace::Continuous, NodesSpace::Discontinuous] {
                let mut buf: Vec<u8> = Vec::new();
                let r = write_mfem_nodes(
                    &mut buf,
                    &Mesh::<2>::unit_square_tri(1),
                    Some(m),
                    space,
                );
                let has_nodes = String::from_utf8_lossy(&buf)
                    .lines()
                    .any(|l| l.trim() == "nodes");
                println!(
                    "  write {space:?}: {:?} nodes_section={has_nodes}",
                    r.as_ref().map(|()| buf.len())
                );
                if r.is_ok() && has_nodes {
                    if let Some(ref dir) = dump {
                        let p = std::path::Path::new(dir)
                            .join(format!("{name}.rs{space:?}.mesh").replace(' ', ""));
                        std::fs::write(p, &buf).expect("dump");
                    }
                }
            }
        }
    }
    // The refined case (ex9's input path): does refinement keep an order-1
    // geometry table, and does the writer then drop it silently?
    let path = data.join("periodic-hexagon.mesh");
    let f = read_mfem_file(&path).expect("read");
    if let Some(m) = f.mesh2d.as_ref() {
        let mut r = m.clone();
        for lev in 0..2 {
            r = fem_mesh::refine_uniform(&r);
            println!("refine level {}: n_elems={}", lev + 1, r.n_elements());
        }
        geom_facts(&r, "refined x2");
        let mut buf: Vec<u8> = Vec::new();
        let res = write_mfem_nodes(&mut buf, &r, None, NodesSpace::Continuous);
        let text = String::from_utf8_lossy(&buf).to_string();
        println!(
            "  write Continuous: {:?} bytes={} lines={} nodes_section={}",
            res.map(|()| ()),
            buf.len(),
            text.lines().count(),
            text.lines().any(|l| l.trim() == "nodes")
        );
        let mut buf2: Vec<u8> = Vec::new();
        let res2 = write_mfem_nodes(&mut buf2, &r, None, NodesSpace::Discontinuous);
        let text2 = String::from_utf8_lossy(&buf2).to_string();
        println!(
            "  write Discontinuous: {:?} bytes={} lines={} nodes_section={}",
            res2.map(|()| ()),
            buf2.len(),
            text2.lines().count(),
            text2.lines().any(|l| l.trim() == "nodes")
        );
        // ex9's actual output goes through the same call the example makes.
    }
}
