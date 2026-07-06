//! mfem_ex39_named_attributes - named-configuration workflow demo.
//!
//! Demonstrates:
//! 1) Gmsh PhysicalNames -> NamedAttributeRegistry
//! 2) named set queries on mesh
//! 3) named-set driven submesh extraction
//! 4) multi-set boundary aggregation (--merge-boundary mode)
//! 5) named-set driven scalar solve on an imported-style Gmsh mesh (--solve-poisson)
//! 6) Abaqus named node/element set import to solve + VTK workflow (--abaqus-demo/--abaqus)
//! 7) Netgen boundary-tag import to VTK workflow (--netgen-demo/--netgen)
//! 8) unified imported-workflow entry surface via --import-format/--input

use fem_assembly::{
    Assembler,
    coefficient::PWConstCoeff,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_core::FemResult;
use fem_io::{
    abaqus::{AbaqusInpData, read_abaqus_inp_full, read_abaqus_inp_full_file},
    netgen::{read_netgen_vol, read_netgen_vol_file},
    read_msh, read_msh_file,
    FIELD_BOUNDARY_MASK,
    vtk_abaqus_solution_fields,
    vtk_imported_mask_fields,
    vtk_named_attribute_solution_fields,
    vtk_named_boundary_fields,
    vtk::{DataArray, VtkWriter},
};
use fem_mesh::{extract_submesh_by_name, topology::MeshTopology, NamedAttributeRegistry, SimplexMesh};
use fem_solver::{SolverConfig, solve_gmres};
use fem_space::{H1Space, constraints::apply_dirichlet, fe_space::FESpace};
use std::collections::{BTreeMap, HashSet};

#[cfg(test)]
use fem_io::{
    FIELD_DRIVE_MASK,
    FIELD_FIXED_MASK,
    FIELD_FLUID_ID,
    FIELD_INLET_MASK,
    FIELD_KAPPA,
    FIELD_LINER_ID,
    FIELD_MATERIAL_ID,
    FIELD_MERGED_BOUNDARY_MASK,
    FIELD_OUTLET_MASK,
    FIELD_SOLUTION,
    FIELD_SOURCE_STRENGTH,
};

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

const SOLVER_MSH_TEXT: &str = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
4
2 1 "fluid"
2 2 "liner"
1 4 "inlet"
1 2 "outlet"
$EndPhysicalNames
$Nodes
9
1 0 0 0
2 0.5 0 0
3 1 0 0
4 0 0.5 0
5 0.5 0.5 0
6 1 0.5 0
7 0 1 0
8 0.5 1 0
9 1 1 0
$EndNodes
$Elements
16
1 1 2 1 1 1 2
2 1 2 1 1 2 3
3 1 2 2 2 3 6
4 1 2 2 2 6 9
5 1 2 3 3 9 8
6 1 2 3 3 8 7
7 1 2 4 4 7 4
8 1 2 4 4 4 1
9 2 2 1 1 1 2 5
10 2 2 1 1 1 5 4
11 2 2 2 2 2 3 6
12 2 2 2 2 2 6 5
13 2 2 1 1 4 5 8
14 2 2 1 1 4 8 7
15 2 2 2 2 5 6 9
16 2 2 2 2 5 9 8
$EndElements
"#;

const ABAQUS_DEMO_INP: &str = r#"*Heading
** Minimal named-set diffusion workflow fixture
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
*Element, type=C3D4, elset=MAT_A
1, 1, 2, 3, 4
*Nset, nset=FIXED
1, 2
*Nset, nset=DRIVE
4
"#;

const NETGEN_DEMO_VOL: &str = r#"
dimension
3

points
4
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

volumeelements
1
1 4 1 2 3 4

surfaceelements
4
3 3 1 2 3
5 3 1 2 4
3 3 1 3 4
3 3 2 3 4
"#;

#[derive(Debug, Clone)]
struct DiffusionSolveResult<const D: usize> {
    mesh: SimplexMesh<D>,
    solution: Vec<f64>,
    iterations: usize,
    final_residual: f64,
    converged: bool,
}

fn load_demo_mesh() -> (SimplexMesh<2>, NamedAttributeRegistry) {
    let msh = read_msh(DEMO_MSH_TEXT.as_bytes()).expect("failed to parse in-memory gmsh");
    let registry = msh.named_attribute_registry();
    let mesh: SimplexMesh<2> = msh.into_2d().expect("expected 2D mesh");
    (mesh, registry)
}

fn load_solver_demo_mesh() -> (SimplexMesh<2>, NamedAttributeRegistry) {
    let msh = read_msh(SOLVER_MSH_TEXT.as_bytes()).expect("failed to parse solve demo gmsh");
    let registry = msh.named_attribute_registry();
    let mesh: SimplexMesh<2> = msh.into_2d().expect("expected 2D solve demo mesh");
    (mesh, registry)
}

fn load_gmsh_file_mesh(path: impl AsRef<std::path::Path>) -> (SimplexMesh<2>, NamedAttributeRegistry) {
    let msh = read_msh_file(path).expect("failed to read Gmsh mesh file");
    let registry = msh.named_attribute_registry();
    let mesh: SimplexMesh<2> = msh.into_2d().expect("expected 2D mesh");
    (mesh, registry)
}

fn load_abaqus_demo_data() -> AbaqusInpData {
    read_abaqus_inp_full(ABAQUS_DEMO_INP.as_bytes()).expect("failed to parse demo Abaqus input")
}

fn load_netgen_demo_mesh() -> SimplexMesh<3> {
    read_netgen_vol(NETGEN_DEMO_VOL.as_bytes()).expect("failed to parse demo Netgen volume mesh")
}

#[cfg(test)]
fn unique_temp_vtk_path(stem: &str) -> std::path::PathBuf {
    let mut out = std::env::temp_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    out.push(format!("{}_{}_{}.vtu", stem, std::process::id(), stamp));
    out
}

#[cfg(test)]
fn example_mesh_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("meshes").join(name)
}

fn point_mask_from_node_ids<const D: usize>(mesh: &SimplexMesh<D>, node_ids: &[u32]) -> Vec<f64> {
    let mut mask = vec![0.0; mesh.n_nodes()];
    for &node in node_ids {
        mask[node as usize] = 1.0;
    }
    mask
}

fn point_mask_for_boundary_tag<const D: usize>(mesh: &SimplexMesh<D>, tag: i32) -> Vec<f64> {
    let mut mask = vec![0.0; mesh.n_nodes()];
    for face in 0..mesh.n_boundary_faces() as u32 {
        if mesh.face_tag(face) == tag {
            for &node in mesh.face_nodes(face) {
                mask[node as usize] = 1.0;
            }
        }
    }
    mask
}

fn material_id_field<const D: usize>(mesh: &SimplexMesh<D>) -> Vec<f64> {
    mesh.elem_tags.iter().map(|&tag| tag as f64).collect()
}

fn unique_nodes_for_named_boundary_set(
    mesh: &SimplexMesh<2>,
    registry: &NamedAttributeRegistry,
    name: &str,
) -> Vec<u32> {
    let faces = mesh
        .face_ids_for_named_set(registry, name)
        .expect("missing named boundary set");
    let mut nodes = std::collections::BTreeSet::new();
    for face in faces {
        for &node in mesh.bface_nodes(face) {
            nodes.insert(node);
        }
    }
    nodes.into_iter().collect()
}

fn nodal_mask_for_boundary_set(
    mesh: &SimplexMesh<2>,
    registry: &NamedAttributeRegistry,
    name: &str,
) -> Vec<f64> {
    point_mask_from_node_ids(mesh, &unique_nodes_for_named_boundary_set(mesh, registry, name))
}

fn cell_mask_for_named_region(
    mesh: &SimplexMesh<2>,
    registry: &NamedAttributeRegistry,
    name: &str,
) -> Vec<f64> {
    let elems = mesh
        .element_ids_for_named_set(registry, name)
        .expect("missing named region set");
    let mut mask = vec![0.0; mesh.n_elems()];
    for elem in elems {
        mask[elem as usize] = 1.0;
    }
    mask
}

fn maybe_cell_mask_for_named_region(
    mesh: &SimplexMesh<2>,
    registry: &NamedAttributeRegistry,
    name: &str,
) -> Option<Vec<f64>> {
    if registry.names().contains(&name) {
        Some(cell_mask_for_named_region(mesh, registry, name))
    } else {
        None
    }
}

fn merged_boundary_mask(mesh: &SimplexMesh<2>, registry: &NamedAttributeRegistry) -> Vec<f64> {
    nodal_mask_for_boundary_set(mesh, registry, "inlet")
        .into_iter()
        .zip(nodal_mask_for_boundary_set(mesh, registry, "outlet"))
        .map(|(inlet, outlet)| if inlet > 0.0 || outlet > 0.0 { 1.0 } else { 0.0 })
        .collect()
}

fn write_vtk_with_fields<const D: usize>(
    mesh: &SimplexMesh<D>,
    path: impl AsRef<std::path::Path>,
    point_data: Vec<DataArray>,
    cell_data: Vec<DataArray>,
) -> FemResult<()> {
    let mut writer = VtkWriter::new(mesh);
    for arr in point_data {
        writer.add_point_data(arr);
    }
    for arr in cell_data {
        writer.add_cell_data(arr);
    }
    writer.write_file(path)
}

fn cell_values_from_tags<const D: usize>(
    mesh: &SimplexMesh<D>,
    values_by_tag: &[(i32, f64)],
    default: f64,
) -> Vec<f64> {
    let by_tag: BTreeMap<i32, f64> = values_by_tag.iter().copied().collect();
    mesh.elem_tags
        .iter()
        .map(|tag| by_tag.get(tag).copied().unwrap_or(default))
        .collect()
}

fn cell_values_from_named_regions(
    mesh: &SimplexMesh<2>,
    registry: &NamedAttributeRegistry,
    values_by_name: &[(&str, f64)],
) -> Vec<f64> {
    let mut out = vec![0.0; mesh.n_elems()];
    for &(name, value) in values_by_name {
        let elems = mesh
            .element_ids_for_named_set(registry, name)
            .unwrap_or_else(|_| panic!("missing named region set: {name}"));
        for elem in elems {
            out[elem as usize] = value;
        }
    }
    out
}

fn assemble_named_region_source_p1(
    mesh: &SimplexMesh<2>,
    registry: &NamedAttributeRegistry,
    region_sources: &[(&str, f64)],
) -> Vec<f64> {
    let mut rhs = vec![0.0; mesh.n_nodes()];
    for &(name, value) in region_sources {
        if value.abs() < 1.0e-14 {
            continue;
        }
        let sub = extract_submesh_by_name(mesh, registry, name)
            .unwrap_or_else(|_| panic!("submesh extraction by named set failed: {name}"));
        let sub_space = H1Space::new(sub.mesh.clone(), 1);
        let source = DomainSourceIntegrator::new(move |_x: &[f64]| value);
        let sub_rhs = Assembler::assemble_linear(&sub_space, &[&source], 3);
        let lifted = sub.transfer_to_parent(&sub_rhs, mesh.n_nodes());
        for (dst, src) in rhs.iter_mut().zip(lifted) {
            *dst += src;
        }
    }
    rhs
}

fn solve_diffusion_p1<const D: usize, C: fem_assembly::postproc::coefficient::ScalarCoeff>(
    mesh: SimplexMesh<D>,
    dirichlet: &[(u32, f64)],
    kappa: C,
    mut rhs: Vec<f64>,
) -> DiffusionSolveResult<D> {
    let space = H1Space::new(mesh.clone(), 1);
    let diffusion = DiffusionIntegrator { kappa };
    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 3);
    assert_eq!(rhs.len(), space.n_dofs(), "rhs size must match H1 DOF count");

    let mut imposed = BTreeMap::new();
    for &(dof, value) in dirichlet {
        imposed.insert(dof, value);
    }
    let dofs: Vec<u32> = imposed.keys().copied().collect();
    let vals: Vec<f64> = imposed.values().copied().collect();
    apply_dirichlet(&mut mat, &mut rhs, &dofs, &vals);

    let mut solution = vec![0.0; space.n_dofs()];
    let cfg = SolverConfig {
        rtol: 1.0e-10,
        atol: 0.0,
        max_iter: 2_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_gmres(&mat, &rhs, &mut solution, 30, &cfg).expect("diffusion solve failed");

    DiffusionSolveResult {
        mesh,
        solution,
        iterations: res.iterations,
        final_residual: res.final_residual,
        converged: res.converged,
    }
}

fn write_named_attribute_vtk(
    mesh: &SimplexMesh<2>,
    registry: &NamedAttributeRegistry,
    path: impl AsRef<std::path::Path>,
    include_merged_boundary: bool,
) -> FemResult<()> {
    let merged = include_merged_boundary.then(|| merged_boundary_mask(mesh, registry));
    let (point_data, cell_data) = vtk_named_boundary_fields(
        nodal_mask_for_boundary_set(mesh, registry, "inlet"),
        nodal_mask_for_boundary_set(mesh, registry, "outlet"),
        merged,
        cell_mask_for_named_region(mesh, registry, "fluid"),
    );
    write_vtk_with_fields(
        mesh,
        path,
        point_data,
        cell_data,
    )
}

fn write_named_attribute_solution_vtk(
    mesh: &SimplexMesh<2>,
    registry: &NamedAttributeRegistry,
    solution: Vec<f64>,
    material_kappa: &[(i32, f64)],
    region_sources: &[(&str, f64)],
    path: impl AsRef<std::path::Path>,
    include_merged_boundary: bool,
) -> FemResult<()> {
    let merged = include_merged_boundary.then(|| merged_boundary_mask(mesh, registry));
    let (point_data, cell_data) = vtk_named_attribute_solution_fields(
        solution,
        nodal_mask_for_boundary_set(mesh, registry, "inlet"),
        nodal_mask_for_boundary_set(mesh, registry, "outlet"),
        merged,
        material_id_field(mesh),
        cell_values_from_tags(mesh, material_kappa, 1.0),
        cell_values_from_named_regions(mesh, registry, region_sources),
        cell_mask_for_named_region(mesh, registry, "fluid"),
        maybe_cell_mask_for_named_region(mesh, registry, "liner"),
    );
    write_vtk_with_fields(mesh, path, point_data, cell_data)
}

fn write_abaqus_solution_vtk(
    data: &AbaqusInpData,
    fixed_set: &str,
    drive_set: &str,
    solution: Vec<f64>,
    path: impl AsRef<std::path::Path>,
) -> FemResult<()> {
    let fixed_nodes = data.node_sets.get(fixed_set).expect("missing fixed node set");
    let drive_nodes = data.node_sets.get(drive_set).expect("missing drive node set");
    let (point_data, cell_data) = vtk_abaqus_solution_fields(
        solution,
        point_mask_from_node_ids(&data.mesh, fixed_nodes),
        point_mask_from_node_ids(&data.mesh, drive_nodes),
        material_id_field(&data.mesh),
    );
    write_vtk_with_fields(
        &data.mesh,
        path,
        point_data,
        cell_data,
    )
}

fn write_netgen_boundary_vtk(
    mesh: &SimplexMesh<3>,
    boundary_tag: i32,
    path: impl AsRef<std::path::Path>,
) -> FemResult<()> {
    let (point_data, cell_data) = vtk_imported_mask_fields(
        FIELD_BOUNDARY_MASK,
        point_mask_for_boundary_tag(mesh, boundary_tag),
        material_id_field(mesh),
    );
    write_vtk_with_fields(
        mesh,
        path,
        point_data,
        cell_data,
    )
}

fn print_workflow_banner(title: &str) {
    println!("=== mfem_ex39_named_attributes: {title} ===");
}

fn print_solver_summary<const D: usize>(result: &DiffusionSolveResult<D>) {
    println!(
        "  solve: {} iterations, residual = {:.3e}, converged = {}",
        result.iterations,
        result.final_residual,
        result.converged
    );
}

fn print_vtk_status(path: Option<&str>, hint: &str) {
    if let Some(path) = path {
        println!("  VTK written to: {path}");
    } else {
        println!("  (Pass --vtk output.vtu to {hint})");
    }
}

#[cfg(test)]
fn assert_vtk_has_field(vtk: &str, field_name: &str) {
    assert!(
        vtk.contains(&format!(r#"Name="{}""#, field_name)),
        "expected VTK field {field_name} in output"
    );
}

fn run_named_attribute_workflow(args: &Args) {
    print_workflow_banner("named set workflow");
    if args.merge_boundary {
        println!("  Mode: merge-boundary (inlet + outlet aggregation)");
    }
    if args.intersection_region {
        println!("  Mode: intersection-region (inlet intersect outlet)");
    }
    if args.difference_region {
        println!("  Mode: difference-region (inlet \\ outlet)");
    }
    if args.solve_poisson {
        println!("  Mode: solve-poisson (u=1 on inlet, u=0 on outlet, named materials + region source)");
    }

    let (mesh, registry) = if let Some(ref path) = args.gmsh_input() {
        load_gmsh_file_mesh(path)
    } else if args.solve_poisson {
        load_solver_demo_mesh()
    } else {
        load_demo_mesh()
    };

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
    let liner_elems = maybe_cell_mask_for_named_region(&mesh, &registry, "liner")
        .map(|mask| mask.into_iter().filter(|&value| value > 0.5).count())
        .unwrap_or(0);

    println!(
        "  mesh: n_nodes={}, n_elems={}, n_faces={}",
        mesh.n_nodes(),
        mesh.n_elems(),
        mesh.n_faces()
    );
    println!(
        "  named sets: fluid elems={}, liner elems={}, inlet faces={}, outlet faces={}, fluid submesh elems={}",
        fluid_elems.len(),
        liner_elems,
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
        assert_eq!(merged_boundary.len(), inlet_faces.len() + outlet_faces.len());
    }

    if args.intersection_region {
        let inlet_set: HashSet<u32> = inlet_faces.iter().copied().collect();
        let outlet_set: HashSet<u32> = outlet_faces.iter().copied().collect();
        let intersection: HashSet<u32> = inlet_set.intersection(&outlet_set).copied().collect();
        println!(
            "  intersection (inlet intersect outlet): {} faces",
            intersection.len()
        );
    }

    if args.difference_region {
        let inlet_set: HashSet<u32> = inlet_faces.iter().copied().collect();
        let outlet_set: HashSet<u32> = outlet_faces.iter().copied().collect();
        let difference: HashSet<u32> = inlet_set.difference(&outlet_set).copied().collect();
        println!(
            "  difference (inlet \\ outlet): {} faces",
            difference.len()
        );
    }

    assert!(!inlet_faces.is_empty());
    assert!(!outlet_faces.is_empty());
    if args.solve_poisson {
        assert_eq!(fluid_elems.len() + liner_elems, mesh.n_elems());
        assert_eq!(fluid_sub.mesh.n_elems(), fluid_elems.len());
    } else {
        assert_eq!(fluid_elems.len(), mesh.n_elems());
        assert_eq!(fluid_sub.mesh.n_elems(), mesh.n_elems());
    }

    if args.solve_poisson {
        let material_kappa = [(1, 10.0), (2, 1.0)];
        let region_sources = [("fluid", 2.0), ("liner", 0.0)];
        let inlet_nodes = unique_nodes_for_named_boundary_set(&mesh, &registry, "inlet");
        let outlet_nodes = unique_nodes_for_named_boundary_set(&mesh, &registry, "outlet");
        let mut dirichlet: Vec<(u32, f64)> = inlet_nodes.iter().map(|&node| (node, 1.0)).collect();
        dirichlet.extend(outlet_nodes.iter().map(|&node| (node, 0.0)));
        let rhs = assemble_named_region_source_p1(&mesh, &registry, &region_sources);
        let result = solve_diffusion_p1(
            mesh,
            &dirichlet,
            PWConstCoeff::new(material_kappa).with_default(1.0),
            rhs,
        );
        print_solver_summary(&result);
        println!("  materials: fluid kappa=10.0, liner kappa=1.0");
        println!("  sources: fluid q=2.0, liner q=0.0");

        if let Some(ref path) = args.vtk {
            write_named_attribute_solution_vtk(
                &result.mesh,
                &registry,
                result.solution.clone(),
                &material_kappa,
                &region_sources,
                path,
                args.merge_boundary,
            )
            .expect("named-attribute solve VTK export failed");
        }
        print_vtk_status(args.vtk.as_deref(), "write solution + named-set masks");
    } else if let Some(ref path) = args.vtk {
        write_named_attribute_vtk(&mesh, &registry, path, args.merge_boundary)
            .expect("named-attribute VTK export failed");
        print_vtk_status(args.vtk.as_deref(), "write named-attribute masks");
    } else {
        print_vtk_status(None, "write named-attribute masks");
    }

    println!("  PASS");
}

fn run_abaqus_workflow(args: &Args) {
    let data = if let Some(ref path) = args.abaqus_input() {
        read_abaqus_inp_full_file(path).expect("failed to read Abaqus input file")
    } else {
        load_abaqus_demo_data()
    };

    let fixed_nodes = data.node_sets.get(&args.fixed_set).expect("missing fixed node set");
    let drive_nodes = data.node_sets.get(&args.drive_set).expect("missing drive node set");
    let mat_a = data.elem_sets.get("MAT_A");

    print_workflow_banner("Abaqus imported workflow");
    println!(
        "  mesh: n_nodes={}, n_elems={}, fixed nodes={}, drive nodes={}",
        data.mesh.n_nodes(),
        data.mesh.n_elems(),
        fixed_nodes.len(),
        drive_nodes.len()
    );
    if let Some(mat_a) = mat_a {
        println!("  MAT_A elements: {}", mat_a.len());
    }

    let mut dirichlet: Vec<(u32, f64)> = fixed_nodes.iter().map(|&node| (node, 0.0)).collect();
    dirichlet.extend(drive_nodes.iter().map(|&node| (node, args.drive_value)));
    let result = solve_diffusion_p1(data.mesh.clone(), &dirichlet, 1.0, vec![0.0; data.mesh.n_nodes()]);
    print_solver_summary(&result);

    if let Some(ref path) = args.vtk {
        write_abaqus_solution_vtk(&data, &args.fixed_set, &args.drive_set, result.solution, path)
            .expect("Abaqus workflow VTK export failed");
    }
    print_vtk_status(args.vtk.as_deref(), "write imported-set solution fields");

    println!("  PASS");
}

fn run_netgen_workflow(args: &Args) {
    let mesh = if let Some(ref path) = args.netgen_input() {
        read_netgen_vol_file(path).expect("failed to read Netgen volume mesh")
    } else {
        load_netgen_demo_mesh()
    };

    let mask = point_mask_for_boundary_tag(&mesh, args.boundary_tag);
    let covered_nodes = mask.iter().filter(|&&value| value > 0.0).count();

    print_workflow_banner("Netgen imported workflow");
    println!(
        "  mesh: n_nodes={}, n_elems={}, n_faces={}, selected boundary tag={}, covered nodes={}",
        mesh.n_nodes(),
        mesh.n_elems(),
        mesh.n_faces(),
        args.boundary_tag,
        covered_nodes
    );

    if let Some(ref path) = args.vtk {
        write_netgen_boundary_vtk(&mesh, args.boundary_tag, path)
            .expect("Netgen workflow VTK export failed");
    }
    print_vtk_status(args.vtk.as_deref(), "write boundary-tag masks");

    println!("  PASS");
}

fn main() {
    let args = parse_args();
    match args.selected_import_format() {
        ImportFormat::Gmsh => run_named_attribute_workflow(&args),
        ImportFormat::Abaqus => run_abaqus_workflow(&args),
        ImportFormat::Netgen => run_netgen_workflow(&args),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImportFormat {
    Gmsh,
    Abaqus,
    Netgen,
}

struct Args {
    merge_boundary: bool,
    intersection_region: bool,
    difference_region: bool,
    solve_poisson: bool,
    import_format: Option<ImportFormat>,
    input: Option<String>,
    abaqus_demo: bool,
    abaqus: Option<String>,
    fixed_set: String,
    drive_set: String,
    drive_value: f64,
    netgen_demo: bool,
    netgen: Option<String>,
    boundary_tag: i32,
    vtk: Option<String>,
}

impl Args {
    fn selected_import_format(&self) -> ImportFormat {
        self.import_format
            .or_else(|| {
                if self.abaqus_demo || self.abaqus.is_some() {
                    Some(ImportFormat::Abaqus)
                } else if self.netgen_demo || self.netgen.is_some() {
                    Some(ImportFormat::Netgen)
                } else {
                    None
                }
            })
            .unwrap_or(ImportFormat::Gmsh)
    }

    fn gmsh_input(&self) -> Option<&str> {
        match self.selected_import_format() {
            ImportFormat::Gmsh => self.input.as_deref(),
            _ => None,
        }
    }

    fn abaqus_input(&self) -> Option<&str> {
        if self.selected_import_format() == ImportFormat::Abaqus {
            self.input.as_deref().or(self.abaqus.as_deref())
        } else {
            self.abaqus.as_deref()
        }
    }

    fn netgen_input(&self) -> Option<&str> {
        if self.selected_import_format() == ImportFormat::Netgen {
            self.input.as_deref().or(self.netgen.as_deref())
        } else {
            self.netgen.as_deref()
        }
    }
}

fn parse_args() -> Args {
    parse_args_from(std::env::args().skip(1))
}

fn parse_args_from<I>(iter: I) -> Args
where
    I: IntoIterator<Item = String>,
{
    let mut args = Args {
        merge_boundary: false,
        intersection_region: false,
        difference_region: false,
        solve_poisson: false,
        import_format: None,
        input: None,
        abaqus_demo: false,
        abaqus: None,
        fixed_set: "FIXED".to_string(),
        drive_set: "DRIVE".to_string(),
        drive_value: 1.0,
        netgen_demo: false,
        netgen: None,
        boundary_tag: 5,
        vtk: None,
    };
    let mut it = iter.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--merge-boundary" => { args.merge_boundary = true; }
            "--intersection-region" => { args.intersection_region = true; }
            "--difference-region" => { args.difference_region = true; }
            "--solve-poisson" => { args.solve_poisson = true; }
            "--import-format" => {
                let format = it.next().expect("expected format after --import-format");
                args.import_format = Some(match format.as_str() {
                    "gmsh" => ImportFormat::Gmsh,
                    "abaqus" => ImportFormat::Abaqus,
                    "netgen" => ImportFormat::Netgen,
                    _ => panic!("invalid --import-format: {format}"),
                });
            }
            "--input" => {
                args.input = Some(it.next().expect("expected input path after --input"));
            }
            "--abaqus-demo" => { args.abaqus_demo = true; }
            "--abaqus" => {
                args.abaqus = Some(it.next().expect("expected input path after --abaqus"));
            }
            "--fixed-set" => {
                args.fixed_set = it.next().expect("expected set name after --fixed-set");
            }
            "--drive-set" => {
                args.drive_set = it.next().expect("expected set name after --drive-set");
            }
            "--drive-value" => {
                args.drive_value = it
                    .next()
                    .expect("expected scalar after --drive-value")
                    .parse()
                    .expect("invalid --drive-value");
            }
            "--netgen-demo" => { args.netgen_demo = true; }
            "--netgen" => {
                args.netgen = Some(it.next().expect("expected input path after --netgen"));
            }
            "--boundary-tag" => {
                args.boundary_tag = it
                    .next()
                    .expect("expected integer after --boundary-tag")
                    .parse()
                    .expect("invalid --boundary-tag");
            }
            "--vtk" => {
                args.vtk = Some(it.next().expect("expected output path after --vtk"));
            }
            _ => {}
        }
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn named_attributes_vtk_export_writes_boundary_and_material_fields() {
        let (mesh, registry) = load_demo_mesh();

        let inlet_mask = nodal_mask_for_boundary_set(&mesh, &registry, "inlet");
        let outlet_mask = nodal_mask_for_boundary_set(&mesh, &registry, "outlet");
        let fluid_id = cell_mask_for_named_region(&mesh, &registry, "fluid");
        assert_eq!(inlet_mask, vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(outlet_mask, vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!(fluid_id, vec![1.0, 1.0]);

        let out = unique_temp_vtk_path("mfem_ex39_named_attributes");

        write_named_attribute_vtk(&mesh, &registry, &out, true)
            .expect("write named-attribute VTK workflow output");

        let vtk = std::fs::read_to_string(&out).expect("read named-attribute VTK output");
        let _ = std::fs::remove_file(&out);

        assert_vtk_has_field(&vtk, FIELD_INLET_MASK);
        assert_vtk_has_field(&vtk, FIELD_OUTLET_MASK);
        assert_vtk_has_field(&vtk, FIELD_MERGED_BOUNDARY_MASK);
        assert_vtk_has_field(&vtk, FIELD_FLUID_ID);
    }

    #[test]
    fn named_attributes_poisson_solve_respects_named_boundary_values() {
        let (mesh, registry) = load_solver_demo_mesh();
        let fluid_elems = mesh.element_ids_for_named_set(&registry, "fluid").unwrap();
        let liner_elems = mesh.element_ids_for_named_set(&registry, "liner").unwrap();
        let inlet_nodes = unique_nodes_for_named_boundary_set(&mesh, &registry, "inlet");
        let outlet_nodes = unique_nodes_for_named_boundary_set(&mesh, &registry, "outlet");
        let mut dirichlet: Vec<(u32, f64)> = inlet_nodes.iter().map(|&node| (node, 1.0)).collect();
        dirichlet.extend(outlet_nodes.iter().map(|&node| (node, 0.0)));

        let no_source = solve_diffusion_p1(
            mesh.clone(),
            &dirichlet,
            PWConstCoeff::new([(1, 10.0), (2, 1.0)]).with_default(1.0),
            vec![0.0; mesh.n_nodes()],
        );

        let result = solve_diffusion_p1(
            mesh.clone(),
            &dirichlet,
            PWConstCoeff::new([(1, 10.0), (2, 1.0)]).with_default(1.0),
            assemble_named_region_source_p1(&mesh, &registry, &[("fluid", 2.0), ("liner", 0.0)]),
        );
        assert!(result.converged, "named-set Poisson solve should converge");
        assert!(result.final_residual < 1.0e-8, "residual = {}", result.final_residual);
        assert_eq!(fluid_elems.len(), 4, "solve fixture should expose four fluid elements");
        assert_eq!(liner_elems.len(), 4, "solve fixture should expose four liner elements");

        for &node in &inlet_nodes {
            assert!((result.solution[node as usize] - 1.0).abs() < 1.0e-12);
        }
        for &node in &outlet_nodes {
            assert!(result.solution[node as usize].abs() < 1.0e-12);
        }

        let free_values: Vec<f64> = result.solution
            .iter()
            .enumerate()
            .filter(|(node, _)| !inlet_nodes.contains(&(*node as u32)) && !outlet_nodes.contains(&(*node as u32)))
            .map(|(_, &value)| value)
            .collect();
        assert!(!free_values.is_empty(), "solver mesh should expose unconstrained nodes");
        assert!(free_values.iter().any(|&value| value > 0.0 && value < 1.0));
        assert!(result.solution[4] > no_source.solution[4], "fluid-only source should raise the interface node value: with source={} without source={}", result.solution[4], no_source.solution[4]);
        let out = unique_temp_vtk_path("mfem_ex39_named_solve");
        write_named_attribute_solution_vtk(
            &result.mesh,
            &registry,
            result.solution,
            &[(1, 10.0), (2, 1.0)],
            &[("fluid", 2.0), ("liner", 0.0)],
            &out,
            true,
        )
        .expect("write named material solve VTK");
        let vtk = std::fs::read_to_string(&out).expect("read named material solve VTK");
        let _ = std::fs::remove_file(&out);

        assert_vtk_has_field(&vtk, FIELD_SOLUTION);
        assert_vtk_has_field(&vtk, FIELD_MATERIAL_ID);
        assert_vtk_has_field(&vtk, FIELD_KAPPA);
        assert_vtk_has_field(&vtk, FIELD_SOURCE_STRENGTH);
        assert_vtk_has_field(&vtk, FIELD_FLUID_ID);
        assert_vtk_has_field(&vtk, FIELD_LINER_ID);
    }

    #[test]
    fn parse_args_supports_unified_import_surface_for_abaqus() {
        let args = parse_args_from([
            "--import-format".to_string(),
            "abaqus".to_string(),
            "--input".to_string(),
            "fixture.inp".to_string(),
            "--drive-set".to_string(),
            "LOAD".to_string(),
        ]);

        assert_eq!(args.selected_import_format(), ImportFormat::Abaqus);
        assert_eq!(args.abaqus_input(), Some("fixture.inp"));
        assert_eq!(args.drive_set, "LOAD");
    }

    #[test]
    fn parse_args_supports_unified_import_surface_for_netgen() {
        let args = parse_args_from([
            "--import-format".to_string(),
            "netgen".to_string(),
            "--input".to_string(),
            "fixture.vol".to_string(),
            "--boundary-tag".to_string(),
            "7".to_string(),
        ]);

        assert_eq!(args.selected_import_format(), ImportFormat::Netgen);
        assert_eq!(args.netgen_input(), Some("fixture.vol"));
        assert_eq!(args.boundary_tag, 7);
    }

    #[test]
    fn parse_args_supports_unified_import_surface_for_gmsh() {
        let args = parse_args_from([
            "--import-format".to_string(),
            "gmsh".to_string(),
            "--input".to_string(),
            "fixture.msh".to_string(),
            "--solve-poisson".to_string(),
        ]);

        assert_eq!(args.selected_import_format(), ImportFormat::Gmsh);
        assert_eq!(args.gmsh_input(), Some("fixture.msh"));
        assert!(args.solve_poisson);
    }

    #[test]
    fn abaqus_named_sets_drive_example_solution_and_vtk_fields() {
        let data = load_abaqus_demo_data();
        let fixed_nodes = data.node_sets.get("FIXED").unwrap().clone();
        let drive_nodes = data.node_sets.get("DRIVE").unwrap().clone();

        let mut dirichlet: Vec<(u32, f64)> = fixed_nodes.iter().map(|&node| (node, 0.0)).collect();
        dirichlet.extend(drive_nodes.iter().map(|&node| (node, 1.0)));
        let result = solve_diffusion_p1(data.mesh.clone(), &dirichlet, 1.0, vec![0.0; data.mesh.n_nodes()]);
        assert!(result.converged, "Abaqus imported solve should converge");

        for &node in &fixed_nodes {
            assert!(result.solution[node as usize].abs() < 1.0e-12);
        }
        for &node in &drive_nodes {
            assert!((result.solution[node as usize] - 1.0).abs() < 1.0e-12);
        }

        let out = unique_temp_vtk_path("mfem_ex39_abaqus");
        write_abaqus_solution_vtk(&data, "FIXED", "DRIVE", result.solution, &out)
            .expect("write Abaqus example workflow VTK");
        let vtk = std::fs::read_to_string(&out).expect("read Abaqus VTK output");
        let _ = std::fs::remove_file(&out);

        assert_vtk_has_field(&vtk, FIELD_SOLUTION);
        assert_vtk_has_field(&vtk, FIELD_FIXED_MASK);
        assert_vtk_has_field(&vtk, FIELD_DRIVE_MASK);
        assert_vtk_has_field(&vtk, FIELD_MATERIAL_ID);
    }

    #[test]
    fn netgen_boundary_tag_example_exports_mask_field() {
        let mesh = load_netgen_demo_mesh();
        let mask = point_mask_for_boundary_tag(&mesh, 5);
        assert_eq!(mask, vec![1.0, 1.0, 0.0, 1.0]);

        let out = unique_temp_vtk_path("mfem_ex39_netgen");
        write_netgen_boundary_vtk(&mesh, 5, &out).expect("write Netgen example workflow VTK");
        let vtk = std::fs::read_to_string(&out).expect("read Netgen VTK output");
        let _ = std::fs::remove_file(&out);

        assert_vtk_has_field(&vtk, FIELD_BOUNDARY_MASK);
        assert_vtk_has_field(&vtk, FIELD_MATERIAL_ID);
    }

    #[test]
    fn abaqus_file_fixture_workflow_exports_fields() {
        let data = read_abaqus_inp_full_file(example_mesh_path("named_sets_tet.inp"))
            .expect("read Abaqus example mesh fixture");
        let fixed_nodes = data.node_sets.get("FIXED").unwrap().clone();
        let drive_nodes = data.node_sets.get("DRIVE").unwrap().clone();

        let mut dirichlet: Vec<(u32, f64)> = fixed_nodes.iter().map(|&node| (node, 0.0)).collect();
        dirichlet.extend(drive_nodes.iter().map(|&node| (node, 1.0)));
        let result = solve_diffusion_p1(data.mesh.clone(), &dirichlet, 1.0, vec![0.0; data.mesh.n_nodes()]);
        assert!(result.converged, "Abaqus file workflow should converge");

        let out = unique_temp_vtk_path("mfem_ex39_abaqus_file");
        write_abaqus_solution_vtk(&data, "FIXED", "DRIVE", result.solution, &out)
            .expect("write Abaqus file workflow VTK");
        let vtk = std::fs::read_to_string(&out).expect("read Abaqus file workflow VTK");
        let _ = std::fs::remove_file(&out);

        assert_vtk_has_field(&vtk, FIELD_SOLUTION);
        assert_vtk_has_field(&vtk, FIELD_FIXED_MASK);
        assert_vtk_has_field(&vtk, FIELD_DRIVE_MASK);
        assert_vtk_has_field(&vtk, FIELD_MATERIAL_ID);
    }

    #[test]
    fn netgen_file_fixture_workflow_exports_fields() {
        let mesh = read_netgen_vol_file(example_mesh_path("surface_tags_tet.vol"))
            .expect("read Netgen example mesh fixture");
        let out = unique_temp_vtk_path("mfem_ex39_netgen_file");
        write_netgen_boundary_vtk(&mesh, 5, &out).expect("write Netgen file workflow VTK");
        let vtk = std::fs::read_to_string(&out).expect("read Netgen file workflow VTK");
        let _ = std::fs::remove_file(&out);

        assert_vtk_has_field(&vtk, FIELD_BOUNDARY_MASK);
        assert_vtk_has_field(&vtk, FIELD_MATERIAL_ID);
    }
}

