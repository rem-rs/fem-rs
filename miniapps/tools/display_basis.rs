//! # Display Basis Miniapp (port of MFEM `miniapps/tools/display-basis.cpp`)
//!
//! Visualizes finite element basis functions on a single reference mesh
//! element in 2D and 3D. This port keeps the computational core (reference
//! element mesh + FE space construction + element VDofs layout) and prints
//! the basis-function (DoF) inventory; the interactive GLVis window loop and
//! the element deformations are not ported (no GLVis in this port).
//!
//! Supported basis types in this port: H1 (b=0), Nedelec (b=1),
//! Raviart-Thomas (b=2), L2 Gauss-Legendre (b=3), Fixed order 1 (b=4, ==
//! H1 order 1).  L2 is supported at any order on Quad/Hex (MFEM
//! `L2_FECollection` Gauss-Legendre tensor nodes, lexicographic `L2_DOF_MAP`
//! order — vsize and element VDofs match the C++ miniapp exactly) and on
//! Tri/Tet up to order 3.  Positive / Serendipity / Crouzeix-Raviart /
//! Gauss discontinuous collections are not implemented in fem-rs and are
//! rejected like the C++ FEC == NULL path.
//!
//! Sample runs:
//!   cargo run --release --example tools_display_basis -- -e 2 -b 3 -o 3
//!   cargo run --release --example tools_display_basis -- -e 5 -b 1 -o 1
//!   cargo run --release --example tools_display_basis -- -e 5 -b 3 -o 4
//!   cargo run --release --example tools_display_basis -- -e 3 -b 2 -o 2

use fem_mesh::{Mesh, element_type::ElementType};
use fem_space::{FESpace, HCurlSpace, HDivSpace, H1Space, L2Basis, L2Space};

fn elem_type_str(e: &str) -> &'static str {
    match e {
        "2" => "TRIANGLE",
        "3" => "QUADRILATERAL",
        "4" => "TETRAHEDRON",
        "5" => "HEXAHEDRON",
        _ => "INVALID",
    }
}

fn basis_type_str(b: &str) -> &'static str {
    match b {
        "0" | "h" => "Continuous (H1)",
        "1" | "n" => "Nedelec",
        "2" | "r" => "Raviart-Thomas",
        "3" | "l" => "Discontinuous (L2)",
        "4" | "f" => "Fixed Order Continuous",
        _ => "INVALID",
    }
}

fn map_type_str(b: &str) -> &'static str {
    match b {
        "1" | "n" => "H_CURL",
        "2" | "r" => "H_DIV",
        _ => "VALUE",
    }
}

/// Build the single reference element mesh (miniapps/common ElementMeshStream).
fn reference_mesh(e: &str) -> (Mesh<2>, Option<Mesh<3>>) {
    match e {
        // TRIANGLE
        "2" => (
            Mesh::uniform(
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                vec![0, 1, 2],
                vec![1],
                ElementType::Tri3,
                vec![0, 1, 1, 2, 2, 0],
                vec![1, 1, 1],
                ElementType::Line2,
            ),
            None,
        ),
        // QUADRILATERAL
        "3" => (
            Mesh::uniform(
                vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                vec![0, 1, 2, 3],
                vec![1],
                ElementType::Quad4,
                vec![0, 1, 1, 2, 2, 3, 3, 0],
                vec![1, 1, 1, 1],
                ElementType::Line2,
            ),
            None,
        ),
        // TETRAHEDRON
        "4" => (
            Mesh::<2>::uniform(vec![], vec![], vec![], ElementType::Tri3, vec![], vec![], ElementType::Line2), // placeholder, unused
            Some(Mesh::uniform(
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                vec![0, 1, 2, 3],
                vec![1],
                ElementType::Tet4,
                vec![0, 2, 1, 1, 2, 3, 2, 0, 3, 0, 1, 3],
                vec![1, 1, 1, 1],
                ElementType::Tri3,
            )),
        ),
        // HEXAHEDRON
        "5" => (
            Mesh::<2>::uniform(vec![], vec![], vec![], ElementType::Tri3, vec![], vec![], ElementType::Line2), // placeholder, unused
            Some(Mesh::uniform(
                vec![
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                    1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
                ],
                vec![0, 1, 2, 3, 4, 5, 6, 7],
                vec![1],
                ElementType::Hex8,
                vec![
                    0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7,
                ],
                vec![1; 6],
                ElementType::Quad4,
            )),
        ),
        _ => panic!("Unsupported element type for this port (1D segment not supported)"),
    }
}

fn main() {
    // Parse command-line options.
    let mut e_int = -1i32;
    let mut b_int = -1i32;
    let mut b_order = 2usize;
    let mut only_some = -1i64;

    let args: Vec<String> = std::env::args().collect();
    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-e" | "--elem-type" => e_int = it.next().unwrap().parse().unwrap(),
            "-b" | "--basis-type" => b_int = it.next().unwrap().parse().unwrap(),
            "-o" | "--order" => b_order = it.next().unwrap().parse().unwrap(),
            "-only" | "--onlySome" => only_some = it.next().unwrap().parse().unwrap(),
            "-nx" | "--num-win-x" | "-ny" | "--num-win-y" | "-w" | "--width" | "-h"
            | "--height" | "-p" | "--send-port" => {
                let _ = it.next(); // GLVis layout options: accepted, ignored
            }
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            other => panic!("Unknown option: {other}"),
        }
    }

    // C++: default TRIANGLE / 'h'; only override when in range.
    let e_str = if (1..7).contains(&e_int) { e_int.to_string() } else { "2".to_string() };
    if e_str == "1" {
        panic!("1D segment is not supported in this port");
    }
    let b_str = match b_int {
        0 => "h",
        1 => "n",
        2 => "r",
        3 => "l",
        4 => "f",
        5 | 6 | 7 => {
            // Positive/Gauss-discont/Crouzeix-Raviart collections not in fem-rs
            println!("Invalid combination of basis info (try again)");
            std::process::exit(0);
        }
        _ => "h",
    };
    // 'f' only matches H1 at order 1 (LinearFECollection).
    if b_str == "f" && b_order != 1 {
        println!("Invalid combination of basis info (try again)");
        std::process::exit(0);
    }

    // Print the user input block (C++ main loop header, print_char = true).
    println!();
    println!("Element Type:          {}", elem_type_str(&e_str));
    println!("Basis Type:            {}", basis_type_str(b_str));
    println!("Basis function order:  {b_order}");
    println!("Map Type:              {}", map_type_str(b_str));

    let (_mesh2, mesh3) = reference_mesh(&e_str);
    let is3d = mesh3.is_some();

    // Build the FE space and report the DoF inventory.
    let invalid = |msg: &str| -> ! {
        println!("Invalid combination of basis info ({msg})");
        std::process::exit(0);
    };

    let (ndof, vdofs): (usize, Vec<i64>) = if is3d {
        let mesh = mesh3.unwrap();
        match b_str {
            "h" | "f" => {
                let space = H1Space::new(mesh, b_order as u8);
                let vdofs = space.element_dofs(0).iter().map(|&d| d as i64).collect();
                (space.n_dofs(), vdofs)
            }
            "n" => {
                let space = HCurlSpace::new(mesh, b_order as u8);
                let vdofs = space.element_dofs(0).iter().map(|&d| d as i64).collect();
                (space.n_dofs(), vdofs)
            }
            "r" => {
                if b_order < 1 {
                    invalid("RT order");
                }
                let space = HDivSpace::new(mesh, (b_order - 1) as u8);
                let vdofs = space.element_dofs(0).iter().map(|&d| d as i64).collect();
                (space.n_dofs(), vdofs)
            }
            "l" => {
                let space = L2Space::new_with_basis(mesh, b_order as u8, L2Basis::GaussLegendre);
                let vdofs = space.element_dofs(0).iter().map(|&d| d as i64).collect();
                (space.n_dofs(), vdofs)
            }
            _ => invalid("unsupported"),
        }
    } else {
        let mesh = _mesh2;
        match b_str {
            "h" | "f" => {
                let space = H1Space::new(mesh, b_order as u8);
                let vdofs = space.element_dofs(0).iter().map(|&d| d as i64).collect();
                (space.n_dofs(), vdofs)
            }
            "n" => {
                let space = HCurlSpace::new(mesh, b_order as u8);
                let vdofs = space.element_dofs(0).iter().map(|&d| d as i64).collect();
                (space.n_dofs(), vdofs)
            }
            "r" => {
                if b_order < 1 {
                    invalid("RT order");
                }
                let space = HDivSpace::new(mesh, (b_order - 1) as u8);
                let vdofs = space.element_dofs(0).iter().map(|&d| d as i64).collect();
                (space.n_dofs(), vdofs)
            }
            "l" => {
                let space = L2Space::new_with_basis(mesh, b_order as u8, L2Basis::GaussLegendre);
                let vdofs = space.element_dofs(0).iter().map(|&d| d as i64).collect();
                (space.n_dofs(), vdofs)
            }
            _ => invalid("unsupported"),
        }
    };

    println!("Number of basis functions: {ndof}");
    println!("Element VDofs: {}", vdofs.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" "));

    // Window inventory (C++: one GLVis window per DoF).
    let mut only_some = only_some;
    let mut stop_at = ndof;
    if ndof > 25 && only_some == -1 {
        println!();
        println!(
            "There are more than 25 windows to open.\nOnly showing Dofs 1-10 to avoid crashing.\nUse the option -only N to show Dofs N to N+9 instead."
        );
        only_some = 1;
    }
    let mut i = 0usize;
    while i < stop_at {
        if i == 0 && only_some > 0 && (only_some as usize) < ndof {
            i = (only_some - 1) as usize;
            stop_at = ndof.min(only_some as usize + 9);
        }
        println!("DoF {}", i + 1);
        i += 1;
    }
}
