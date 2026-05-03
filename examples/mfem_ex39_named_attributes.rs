//! mfem_ex39_named_attributes - baseline named-attribute workflow demo.
//!
//! Demonstrates:
//! 1) GMSH PhysicalNames -> NamedAttributeRegistry
//! 2) named set queries on mesh
//! 3) named-set driven submesh extraction
//! 4) multi-set boundary aggregation (--merge-boundary mode)

use fem_io::read_msh;
use fem_mesh::{extract_submesh_by_name, NamedAttributeRegistry, SimplexMesh};
use std::collections::HashSet;

const DEMO_MSH_TEXT: &str = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
2 1 "fluid"
1 1 "inlet"
1 3 "outlet"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
6
1 1 2 1 1 1 2
2 1 2 2 2 2 3
3 1 2 3 3 3 4
4 1 2 4 4 4 1
5 2 2 1 1 1 2 3
6 2 2 1 1 1 3 4
$EndElements
"#;

fn load_demo_mesh() -> (SimplexMesh<2>, NamedAttributeRegistry) {
    let msh = read_msh(DEMO_MSH_TEXT.as_bytes()).expect("failed to parse in-memory gmsh");
    let registry = msh.named_attribute_registry();
    let mesh: SimplexMesh<2> = msh.into_2d().expect("expected 2D mesh");
    (mesh, registry)
}

fn main() {
    let args = parse_args();
    println!("=== mfem_ex39_named_attributes: baseline named set workflow ===");
    if args.merge_boundary {
        println!("  Mode: merge-boundary (inlet + outlet aggregation)");
    }
    if args.intersection_region {
        println!("  Mode: intersection-region (inlet intersect outlet)");
    }
    if args.difference_region {
        println!("  Mode: difference-region (inlet \\ outlet)");
    }

    let (mesh, registry) = load_demo_mesh();

    let fluid_elems = mesh
        .element_ids_for_named_set(&registry, "fluid")
        .expect("missing named set: fluid");
    let inlet_faces = mesh
        .face_ids_for_named_set(&registry, "inlet")
        .expect("missing named set: inlet");
    let outlet_faces = mesh
        .face_ids_for_named_set(&registry, "outlet")
        .expect("missing named set: outlet");

    let fluid_sub = extract_submesh_by_name(&mesh, &registry, "fluid")
        .expect("submesh extraction by named set failed");

    println!(
        "  mesh: n_nodes={}, n_elems={}, n_faces={}",
        mesh.n_nodes(),
        mesh.n_elems(),
        mesh.n_faces()
    );
    println!(
        "  named sets: fluid elems={}, inlet faces={}, outlet faces={}, fluid submesh elems={}",
        fluid_elems.len(),
        inlet_faces.len(),
        outlet_faces.len(),
        fluid_sub.mesh.n_elems()
    );

    if args.merge_boundary {
        let mut merged_boundary: HashSet<u32> = inlet_faces.iter().copied().collect();
        merged_boundary.extend(outlet_faces.iter().copied());
        println!(
            "  merged boundary (inlet union outlet): {} faces",
            merged_boundary.len()
        );
        assert_eq!(
            merged_boundary.len(),
            inlet_faces.len() + outlet_faces.len()
        );
    }

    if args.intersection_region {
        let inlet_set: HashSet<u32> = inlet_faces.iter().copied().collect();
        let outlet_set: HashSet<u32> = outlet_faces.iter().copied().collect();
        let intersection: HashSet<u32> = inlet_set
            .intersection(&outlet_set)
            .copied()
            .collect();
        println!(
            "  intersection (inlet intersect outlet): {} faces",
            intersection.len()
        );
    }

    if args.difference_region {
        let inlet_set: HashSet<u32> = inlet_faces.iter().copied().collect();
        let outlet_set: HashSet<u32> = outlet_faces.iter().copied().collect();
        let difference: HashSet<u32> = inlet_set
            .difference(&outlet_set)
            .copied()
            .collect();
        println!(
            "  difference (inlet \\ outlet): {} faces",
            difference.len()
        );
    }

    assert_eq!(fluid_elems.len(), mesh.n_elems());
    assert!(!inlet_faces.is_empty());
    assert!(!outlet_faces.is_empty());
    assert_eq!(fluid_sub.mesh.n_elems(), mesh.n_elems());

    println!("  PASS");
}

struct Args {
    merge_boundary: bool,
    intersection_region: bool,
    difference_region: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        merge_boundary: false,
        intersection_region: false,
        difference_region: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--merge-boundary" => { args.merge_boundary = true; }
            "--intersection-region" => { args.intersection_region = true; }
            "--difference-region" => { args.difference_region = true; }
            _ => {}
        }
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::topology::MeshTopology;

    fn load_named_sets() -> (SimplexMesh<2>, NamedAttributeRegistry, Vec<u32>, Vec<u32>) {
        let (mesh, registry) = load_demo_mesh();
        let inlet = mesh
            .face_ids_for_named_set(&registry, "inlet")
            .expect("missing inlet");
        let outlet = mesh
            .face_ids_for_named_set(&registry, "outlet")
            .expect("missing outlet");
        (mesh, registry, inlet, outlet)
    }

    #[test]
    fn named_attributes_merge_boundary_mode() {
        let (_mesh, _registry, inlet, outlet) = load_named_sets();

        let mut merged: std::collections::HashSet<u32> = inlet.iter().copied().collect();
        merged.extend(outlet.iter().copied());

        assert!(!inlet.is_empty());
        assert!(!outlet.is_empty());
        assert_eq!(merged.len(), inlet.len() + outlet.len());
    }

    #[test]
    fn named_attributes_intersection_mode() {
        let (_mesh, _registry, inlet, outlet) = load_named_sets();

        let inlet_set: std::collections::HashSet<u32> = inlet.iter().copied().collect();
        let outlet_set: std::collections::HashSet<u32> = outlet.iter().copied().collect();
        let intersection: std::collections::HashSet<u32> = inlet_set
            .intersection(&outlet_set)
            .copied()
            .collect();

        // For this mesh, inlet and outlet don't share faces, so intersection is empty
        assert_eq!(intersection.len(), 0);
    }

    #[test]
    fn named_attributes_difference_mode() {
        let (_mesh, _registry, inlet, outlet) = load_named_sets();

        let inlet_set: std::collections::HashSet<u32> = inlet.iter().copied().collect();
        let outlet_set: std::collections::HashSet<u32> = outlet.iter().copied().collect();
        let difference: std::collections::HashSet<u32> = inlet_set
            .difference(&outlet_set)
            .copied()
            .collect();

        // For this mesh, inlet \ outlet = inlet (since they don't intersect)
        assert_eq!(difference.len(), inlet.len());
    }

    #[test]
    fn named_attributes_boundary_sets_match_expected_geometry() {
        let (mesh, _registry, inlet, outlet) = load_named_sets();

        for &face in &inlet {
            for &node in mesh.bface_nodes(face) {
                let coords = mesh.node_coords(node);
                assert!(coords[1].abs() < 1e-12, "expected inlet nodes on y=0, got y={}", coords[1]);
            }
        }

        for &face in &outlet {
            for &node in mesh.bface_nodes(face) {
                let coords = mesh.node_coords(node);
                assert!((coords[1] - 1.0).abs() < 1e-12, "expected outlet nodes on y=1, got y={}", coords[1]);
            }
        }
    }

    #[test]
    fn named_attributes_fluid_submesh_roundtrips_parent_nodal_field() {
        let (mesh, registry) = load_demo_mesh();
        let fluid_sub = extract_submesh_by_name(&mesh, &registry, "fluid")
            .expect("submesh extraction by named set failed");

        let parent_values: Vec<f64> = (0..mesh.n_nodes())
            .map(|idx| {
                let coords = mesh.node_coords(idx as u32);
                coords[0] + 2.0 * coords[1]
            })
            .collect();
        let sub_values = fluid_sub.transfer_from_parent(&parent_values);
        let roundtrip = fluid_sub.transfer_to_parent(&sub_values, mesh.n_nodes());

        assert_eq!(fluid_sub.mesh.n_elems(), mesh.n_elems());
        assert_eq!(fluid_sub.parent_elem_ids.len(), mesh.n_elems());
        assert_eq!(fluid_sub.parent_node_of_sub.len(), mesh.n_nodes());

        for &parent_node in &fluid_sub.parent_node_of_sub {
            let idx = parent_node as usize;
            assert!(
                (roundtrip[idx] - parent_values[idx]).abs() < 1e-12,
                "roundtrip mismatch at parent node {}: got {} expected {}",
                idx,
                roundtrip[idx],
                parent_values[idx]
            );
        }
    }

    #[test]
    fn named_attributes_missing_sets_fail_cleanly() {
        let (mesh, registry) = load_demo_mesh();

        let element_err = mesh
            .element_ids_for_named_set(&registry, "missing")
            .expect_err("expected missing element set error");
        let face_err = mesh
            .face_ids_for_named_set(&registry, "missing")
            .expect_err("expected missing face set error");
        let submesh_err = extract_submesh_by_name(&mesh, &registry, "missing")
            .expect_err("expected missing submesh set error");

        assert!(format!("{element_err}").contains("named attribute set not found"));
        assert!(format!("{face_err}").contains("named attribute set not found"));
        assert!(format!("{submesh_err}").contains("named attribute set not found"));
    }

    /// The registry parsed from the demo mesh contains all three expected named sets.
    #[test]
    fn named_attributes_registry_contains_expected_names() {
        let (_, registry) = load_demo_mesh();
        let names = registry.names();
        assert!(names.contains(&"fluid"),  "expected 'fluid' in registry: {:?}",  names);
        assert!(names.contains(&"inlet"),  "expected 'inlet' in registry: {:?}",  names);
        assert!(names.contains(&"outlet"), "expected 'outlet' in registry: {:?}", names);
    }

    /// The 'fluid' named set covers all elements in the demo mesh.
    #[test]
    fn named_attributes_fluid_elements_cover_full_mesh() {
        let (mesh, registry) = load_demo_mesh();
        let fluid_elems = mesh
            .element_ids_for_named_set(&registry, "fluid")
            .expect("missing fluid set");
        assert_eq!(fluid_elems.len(), mesh.n_elems(),
            "expected fluid elements to cover full mesh: got {} of {}",
            fluid_elems.len(), mesh.n_elems());
    }
}

