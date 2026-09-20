//! D496 — `Mesh::PrintCharacteristics` / `Mesh::PrintInfo` (generic meshes)
//! against MFEM 4.10 ground truth.
//!
//! The expected blocks were produced by the C++ probe `tmp/d496/
//! run_probe.sh` (program: `Mesh mesh(file,1,1); mesh.UniformRefinement();
//! mesh.PrintInfo();`) against `$HOME/mfem410_ser` (MFEM 4.10, serial, no
//! LAPACK) on the repo's own `data/` meshes, and are embedded verbatim
//! below.  The Rust side reads the same mesh with `fem_io::mfem`,
//! optionally refines it once (`Mesh::UniformRefinement` ports), builds the
//! edge table and calls [`fem_mesh::mesh_characteristics`]'s
//! `print_characteristics`.

use fem_io::mfem::read_mfem_file;
use fem_mesh::amr::{refine_uniform, refine_uniform_3d};
use fem_mesh::topology::MeshTopology as _; // n_nodes counter deriving

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data");

fn expected(blocks: &[&str]) -> String {
    let mut s = String::new();
    for b in blocks {
        s.push_str(b);
    }
    s
}

/// star.mesh (quad star, straight-sided) — levels 0 and 1.
#[test]
fn star_mesh_r0_r1() {
    let file = format!("{DATA}/star.mesh");
    let mesh = read_mfem_file(&file).expect("read").mesh2d.expect("2-D mesh");
    let mut m = mesh.clone();
    m.build_edge_connectivity();
    assert_eq!(m.print_characteristics(), expected(&[
        "Mesh Characteristics:\n",
        "Dimension          : 2\n",
        "Space dimension    : 2\n",
        "Number of vertices : 31\n",
        "Number of edges    : 50\n",
        "Number of elements : 20  --  20 Square(s)\n",
        "Number of bdr elem : 20\n",
        "Euler Number       : 1\n",
        "h_min              : 0.487609\n",
        "h_max              : 0.487611\n",
        "kappa_min          : 1.37637\n",
        "kappa_max          : 1.37639\n",
        "\n",
    ]));

    let mut m1 = refine_uniform(&mesh);
    m1.build_edge_connectivity();
    assert_eq!(m1.n_nodes(), 101);
    assert_eq!(m1.print_characteristics(), expected(&[
        "Mesh Characteristics:\n",
        "Dimension          : 2\n",
        "Space dimension    : 2\n",
        "Number of vertices : 101\n",
        "Number of edges    : 180\n",
        "Number of elements : 80  --  80 Square(s)\n",
        "Number of bdr elem : 40\n",
        "Euler Number       : 1\n",
        "h_min              : 0.243804\n",
        "h_max              : 0.243806\n",
        "kappa_min          : 1.37637\n",
        "kappa_max          : 1.37639\n",
        "\n",
    ]));
}

/// square-disc.mesh (triangles) — levels 0 and 1.
#[test]
fn square_disc_mesh_r0_r1() {
    let file = format!("{DATA}/square-disc.mesh");
    let mesh = read_mfem_file(&file).expect("read").mesh2d.expect("2-D mesh");
    let mut m = mesh.clone();
    m.build_edge_connectivity();
    assert_eq!(m.print_characteristics(), expected(&[
        "Mesh Characteristics:\n",
        "Dimension          : 2\n",
        "Space dimension    : 2\n",
        "Number of vertices : 101\n",
        "Number of edges    : 255\n",
        "Number of elements : 154  --  154 Triangle(s)\n",
        "Number of bdr elem : 48\n",
        "Euler Number       : 0\n",
        "h_min              : 0.0548321\n",
        "h_max              : 0.194754\n",
        "kappa_min          : 1.04322\n",
        "kappa_max          : 2.72622\n",
        "\n",
    ]));

    let mut m1 = refine_uniform(&mesh);
    m1.build_edge_connectivity();
    assert_eq!(m1.print_characteristics(), expected(&[
        "Mesh Characteristics:\n",
        "Dimension          : 2\n",
        "Space dimension    : 2\n",
        "Number of vertices : 356\n",
        "Number of edges    : 972\n",
        "Number of elements : 616  --  616 Triangle(s)\n",
        "Number of bdr elem : 96\n",
        "Euler Number       : 0\n",
        "h_min              : 0.027416\n",
        "h_max              : 0.0973772\n",
        "kappa_min          : 1.04322\n",
        "kappa_max          : 2.72622\n",
        "\n",
    ]));
}

/// star-mixed.mesh (triangles + squares) — the `PrintElementsByGeometry`
/// "N Triangle(s) + M Square(s)" summary; levels 0 and 1.
#[test]
fn star_mixed_mesh_r0_r1() {
    let file = format!("{DATA}/star-mixed.mesh");
    let mesh = read_mfem_file(&file).expect("read").mesh2d.expect("2-D mesh");
    let mut m = mesh.clone();
    m.build_edge_connectivity();
    assert_eq!(m.print_characteristics(), expected(&[
        "Mesh Characteristics:\n",
        "Dimension          : 2\n",
        "Space dimension    : 2\n",
        "Number of vertices : 31\n",
        "Number of edges    : 60\n",
        "Number of elements : 30  --  20 Triangle(s) + 10 Square(s)\n",
        "Number of bdr elem : 20\n",
        "Euler Number       : 1\n",
        "h_min              : 0.48761\n",
        "h_max              : 0.523973\n",
        "kappa_min          : 1.2584\n",
        "kappa_max          : 1.37639\n",
        "\n",
    ]));

    let mut m1 = refine_uniform(&mesh);
    m1.build_edge_connectivity();
    assert_eq!(m1.print_characteristics(), expected(&[
        "Mesh Characteristics:\n",
        "Dimension          : 2\n",
        "Space dimension    : 2\n",
        "Number of vertices : 101\n",
        "Number of edges    : 220\n",
        "Number of elements : 120  --  80 Triangle(s) + 40 Square(s)\n",
        "Number of bdr elem : 40\n",
        "Euler Number       : 1\n",
        "h_min              : 0.243804\n",
        "h_max              : 0.261986\n",
        "kappa_min          : 1.2584\n",
        "kappa_max          : 1.37639\n",
        "\n",
    ]));
}

/// beam-hex.mesh (cubes) — levels 0 and 1: the 3-D block prints interior
/// faces too (`Number of faces`) and `EulerNumber = V - E + F - NE`.
#[test]
fn beam_hex_mesh_r0_r1() {
    let file = format!("{DATA}/beam-hex.mesh");
    let mesh = read_mfem_file(&file).expect("read").mesh3d.expect("3-D mesh");
    let mut m = mesh.clone();
    m.build_edge_connectivity();
    assert_eq!(m.print_characteristics(), expected(&[
        "Mesh Characteristics:\n",
        "Dimension          : 3\n",
        "Space dimension    : 3\n",
        "Number of vertices : 36\n",
        "Number of edges    : 68\n",
        "Number of faces    : 41  --  41 Square(s)\n",
        "Number of elements : 8  --  8 Cube(s)\n",
        "Number of bdr elem : 34  --  34 Square(s)\n",
        "Euler Number       : 1\n",
        "h_min              : 1\n",
        "h_max              : 1\n",
        "kappa_min          : 1\n",
        "kappa_max          : 1\n",
        "\n",
    ]));

    let mut m1 = refine_uniform_3d(&mesh);
    m1.build_edge_connectivity();
    assert_eq!(m1.print_characteristics(), expected(&[
        "Mesh Characteristics:\n",
        "Dimension          : 3\n",
        "Space dimension    : 3\n",
        "Number of vertices : 153\n",
        "Number of edges    : 348\n",
        "Number of faces    : 260  --  260 Square(s)\n",
        "Number of elements : 64  --  64 Cube(s)\n",
        "Number of bdr elem : 136  --  136 Square(s)\n",
        "Euler Number       : 1\n",
        "h_min              : 0.5\n",
        "h_max              : 0.5\n",
        "kappa_min          : 1\n",
        "kappa_max          : 1\n",
        "\n",
    ]));
}

/// beam-tet.mesh (tets) — levels 0 and 1.
#[test]
fn beam_tet_mesh_r0_r1() {
    let file = format!("{DATA}/beam-tet.mesh");
    let mesh = read_mfem_file(&file).expect("read").mesh3d.expect("3-D mesh");
    let mut m = mesh.clone();
    m.build_edge_connectivity();
    assert_eq!(m.print_characteristics(), expected(&[
        "Mesh Characteristics:\n",
        "Dimension          : 3\n",
        "Space dimension    : 3\n",
        "Number of vertices : 36\n",
        "Number of edges    : 117\n",
        "Number of faces    : 130  --  130 Triangle(s)\n",
        "Number of elements : 48  --  48 Tetrahedron(s)\n",
        "Number of bdr elem : 68  --  68 Triangle(s)\n",
        "Euler Number       : 1\n",
        "h_min              : 1.12246\n",
        "h_max              : 1.12246\n",
        "kappa_min          : 2.41421\n",
        "kappa_max          : 2.41421\n",
        "\n",
    ]));

    let mut m1 = refine_uniform_3d(&mesh);
    m1.build_edge_connectivity();
    assert_eq!(m1.print_characteristics(), expected(&[
        "Mesh Characteristics:\n",
        "Dimension          : 3\n",
        "Space dimension    : 3\n",
        "Number of vertices : 153\n",
        "Number of edges    : 672\n",
        "Number of faces    : 904  --  904 Triangle(s)\n",
        "Number of elements : 384  --  384 Tetrahedron(s)\n",
        "Number of bdr elem : 272  --  272 Triangle(s)\n",
        "Euler Number       : 1\n",
        "h_min              : 0.561231\n",
        "h_max              : 0.561231\n",
        "kappa_min          : 2.41421\n",
        "kappa_max          : 2.41421\n",
        "\n",
    ]));
}

/// fichera.mesh (cubes with a re-entrant corner) — levels 0 and 1.
#[test]
fn fichera_mesh_r0_r1() {
    let file = format!("{DATA}/fichera.mesh");
    let mesh = read_mfem_file(&file).expect("read").mesh3d.expect("3-D mesh");
    let mut m = mesh.clone();
    m.build_edge_connectivity();
    assert_eq!(m.print_characteristics(), expected(&[
        "Mesh Characteristics:\n",
        "Dimension          : 3\n",
        "Space dimension    : 3\n",
        "Number of vertices : 26\n",
        "Number of edges    : 51\n",
        "Number of faces    : 33  --  33 Square(s)\n",
        "Number of elements : 7  --  7 Cube(s)\n",
        "Number of bdr elem : 24  --  24 Square(s)\n",
        "Euler Number       : 1\n",
        "h_min              : 1\n",
        "h_max              : 1\n",
        "kappa_min          : 1\n",
        "kappa_max          : 1\n",
        "\n",
    ]));

    let mut m1 = refine_uniform_3d(&mesh);
    m1.build_edge_connectivity();
    assert_eq!(m1.print_characteristics(), expected(&[
        "Mesh Characteristics:\n",
        "Dimension          : 3\n",
        "Space dimension    : 3\n",
        "Number of vertices : 117\n",
        "Number of edges    : 276\n",
        "Number of faces    : 216  --  216 Square(s)\n",
        "Number of elements : 56  --  56 Cube(s)\n",
        "Number of bdr elem : 96  --  96 Square(s)\n",
        "Euler Number       : 1\n",
        "h_min              : 0.5\n",
        "h_max              : 0.5\n",
        "kappa_min          : 1\n",
        "kappa_max          : 1\n",
        "\n",
    ]));
}
