//! D819 (round 83, lane B) — Quad8/Quad9 mixed-assembly dispatch adjudication.
//!
//! **Registered premise refuted (discipline ⑥).**  D809-2 assumed MFEM's
//! `MakeCartesian2D(..., QUADRILATERAL)` + midpoint refinement could produce a
//! Quad8 real mesh.  Probes against MFEM 4.10 (`$HOME/mfem410_ser`, sources in
//! `tmp/d83b/p1_refine.cpp` / `p2_gmsh_quad8.cpp` / `p3_fixture_oracle.cpp`):
//!
//! * `MakeCartesian2D` + `UniformRefinement` + `GeneralRefinement` keep
//!   4-vertex `Geometry::SQUARE` cells forever (MFEM has no 8/9-node quad
//!   geometry element at all — `Mesh::GetElementType` maps format codes 0-7
//!   only, `mesh.cpp:4982`), and `SetCurvature(2)` carries high order in a
//!   9-dofs/element **Nodes grid function**, not in the element rows;
//! * the own-format mesh file has no Quad8/Quad9 codes;
//! * the Gmsh reader (`mesh/gmsh.cpp`, `GmshReader::types`) accepts only the
//!   *complete* quadrilaterals 3/10/36-38/47-51 — the 8-node serendipity quad
//!   (Gmsh type 16) **aborts** (`MFEM abort: Unknown Gmsh element type.`,
//!   `mesh/gmsh.cpp:677`), while the 9-node quad (type 10) is read as a
//!   SQUARE mesh + order-2 Nodes.
//!
//! So the "real mesh fixture" this lane certifies is the **Gmsh type-10
//! (Quad9) curved quad mesh** — written by probe 3, re-parsed and certified by
//! MFEM's own reader, then read by fem-rs (`data/d819_quad9_curved.msh`).
//!
//! **Dispatch verdict (the D809-2 review question).**  A Quad8/Quad9 mesh cell
//! is a *geometry row label* (fem-rs D243: serendipity / tensor-Q2
//! isoparametric geometry map), never a field family — the D581 pattern (one
//! CUBE geometry → every hex cell label shares the tensor H¹ family) applied
//! to quads: MFEM has one SQUARE geometry and `H1_FECollection` builds the
//! tensor `H1_QuadrilateralElement(p)` on it.  Serendipity is a *collection*
//! choice (`H1Ser_FECollection`, 8 dofs at p = 2 vs the tensor 9 — probe 1
//! shows both on the same mesh), and no fem-rs space numbers dofs
//! serendipitously: `DofManager` routes Quad8/Quad9 cells with the tensor
//! family (`mesh::simplex::h1_family_dofs` → `QuadQk`).  The mixed module's
//! old `QuadSerendipityPk` arms therefore disagreed with the space's own
//! element-dof counts (9 rows over an 8-basis element matrix on a Quad9 cell —
//! an out-of-bounds/garbage assembly) and with every family source in the
//! tree.  This file pins the corrected arms (`QuadQk`) and the adjudication
//! against MFEM on the real fixture: per-element mass/diffusion matrices on
//! the curved Quad9 cells agree entrywise to ≤ 1e-12.
//!
//! ```text
//! cargo test -p fem-assembly --test d819_quad8_quad9_mixed_dispatch -- --nocapture
//! ```

use fem_assembly::mixed::{ref_elem_vec, ref_elem_vol_with_pyramid_basis};
use fem_element::lagrange::factory::QuadQk;
use fem_element::lagrange::PyramidBasisType;
use fem_element::ReferenceElement;
use fem_io::gmsh::read_msh;
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::h1_family_dofs;
use fem_mesh::topology::MeshTopology;
use fem_mesh::transformation::geometry_jacobian;
use fem_space::fe_space::SpaceType;

/// The MFEM-certified real mesh fixture (probe 3 wrote it, MFEM 4.10's own
/// Gmsh reader re-parsed it as SQUARE + order-2 Nodes, then the oracle element
/// matrices were dumped from that parse).
const FIXTURE_Q9: &str = include_str!("../../../data/d819_quad9_curved.msh");

/// A one-element 8-node serendipity quad (Gmsh type 16), hand-written: MFEM
/// 4.10 rejects this code outright (probe 2), so this fixture lives on the
/// fem-rs side only — it pins that the io reader's `Quad8` label (D297) and
/// the mixed module's tensor dispatch agree.
const FIXTURE_Q8: &str = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
8
1 0.0000000000000000e+00 0.0000000000000000e+00 0
2 1.0000000000000000e+00 0.0000000000000000e+00 0
3 1.0700000000000001e+00 1.0500000000000000e+00 0
4 -5.0000000000000000e-02 9.4000000000000000e-01 0
5 5.0000000000000000e-01 3.0000000000000000e-03 0
6 1.0300000000000000e+00 5.1000000000000000e-01 0
7 5.2000000000000000e-01 1.0100000000000000e+00 0
8 -2.0000000000000000e-02 4.7000000000000000e-01 0
$EndNodes
$Elements
1
1 16 1 1 1 2 3 4 5 6 7 8
$EndElements
"#;

/// MFEM 4.10 oracle, dumped by `tmp/d83b/p3_fixture_oracle.cpp` (WSL,
/// `$HOME/mfem410_ser`): `H1_FECollection(2,2)` element mass and diffusion
/// matrices on the two curved SQUARE cells of `data/d819_quad9_curved.msh`,
/// with a converged order-30 square rule (16x16 Gauss points; the diffusion
/// integrand is rational on a Q2 geometry).  `node_ref` rows are the element
/// reference DOF coords in slot order.
const ORACLE: &str = r#"
node_ref 0 0 0
node_ref 1 1 0
node_ref 2 1 1
node_ref 3 0 1
node_ref 4 0.5 0
node_ref 5 1 0.5
node_ref 6 0.5 1
node_ref 7 0 0.5
node_ref 8 0.5 0.5
mass 0 0 0 0.020753520601070521
mass 0 0 1 -0.0046906332642251431
mass 0 0 2 0.0011352707844453859
mass 0 0 3 -0.0048201177064729345
mass 0 0 4 0.010750656449990013
mass 0 0 5 -0.0023725163479138738
mass 0 0 6 -0.0024587189736814414
mass 0 0 7 0.010657790764853613
mass 0 0 8 0.0055503277761189242
mass 0 1 0 -0.0046906332642251431
mass 0 1 1 0.016660632307914619
mass 0 1 2 -0.0042581317614996082
mass 0 1 3 0.0011352707844453861
mass 0 1 4 0.0080118766069105871
mass 0 1 5 0.0082499447044729907
mass 0 1 6 -0.002082364164100103
mass 0 1 7 -0.0023725163479138738
mass 0 1 8 0.003939737615536577
mass 0 2 0 0.0011352707844453859
mass 0 2 1 -0.0042581317614996082
mass 0 2 2 0.017475437348658727
mass 0 2 3 -0.0043880329407513484
mass 0 2 4 -0.0020823641641001035
mass 0 2 5 0.0087825823415254509
mass 0 2 6 0.0086946113271852699
mass 0 2 7 -0.0021685667898676698
mass 0 2 8 0.0043897190408638343
mass 0 3 0 -0.0048201177064729345
mass 0 3 1 0.0011352707844453861
mass 0 3 2 -0.0043880329407513484
mass 0 3 3 0.017713359465234356
mass 0 3 4 -0.0024587189736814414
mass 0 3 5 -0.0021685667898676693
mass 0 3 6 0.0088575204358201255
mass 0 3 7 0.008622680061038137
mass 0 3 8 0.0042845481186068474
mass 0 4 0 0.010750656449990013
mass 0 4 1 0.0080118766069105871
mass 0 4 2 -0.0020823641641001035
mass 0 4 3 -0.0024587189736814414
mass 0 4 4 0.075345900773778549
mass 0 4 5 0.003939737615536577
mass 0 4 6 -0.018174777371367613
mass 0 4 7 0.0055503277761189225
mass 0 4 8 0.038153315737247041
mass 0 5 0 -0.0023725163479138738
mass 0 5 1 0.0082499447044729907
mass 0 5 2 0.0087825823415254509
mass 0 5 3 -0.0021685667898676693
mass 0 5 4 0.003939737615536577
mass 0 5 5 0.067940733345122933
mass 0 5 6 0.0043897190408638343
mass 0 5 7 -0.018173666072690397
mass 0 5 8 0.033190960968814846
mass 0 6 0 -0.0024587189736814414
mass 0 6 1 -0.002082364164100103
mass 0 6 2 0.0086946113271852699
mass 0 6 3 0.0088575204358201255
mass 0 6 4 -0.018174777371367613
mass 0 6 5 0.0043897190408638343
mass 0 6 6 0.069983104951002004
mass 0 6 7 0.0042845481186068483
mass 0 6 8 0.034545793748223412
mass 0 7 0 0.010657790764853613
mass 0 7 1 -0.0023725163479138738
mass 0 7 2 -0.0021685667898676698
mass 0 7 3 0.008622680061038137
mass 0 7 4 0.0055503277761189225
mass 0 7 5 -0.018173666072690397
mass 0 7 6 0.0042845481186068483
mass 0 7 7 0.077372714198176429
mass 0 7 8 0.039503703321946755
mass 0 8 0 0.0055503277761189242
mass 0 8 1 0.003939737615536577
mass 0 8 2 0.0043897190408638343
mass 0 8 3 0.0042845481186068474
mass 0 8 4 0.038153315737247041
mass 0 8 5 0.033190960968814846
mass 0 8 6 0.034545793748223412
mass 0 8 7 0.039503703321946755
mass 0 8 8 0.29098100659830956
stiff 0 0 0 0.64467366773114254
stiff 0 0 1 -0.034866182717235635
stiff 0 0 2 -0.027580420104371992
stiff 0 0 3 -0.030038831803734723
stiff 0 0 4 -0.18932473474520861
stiff 0 0 5 0.12805333194926874
stiff 0 0 6 0.12599152858037965
stiff 0 0 7 -0.20669281452138893
stiff 0 0 8 -0.41021554436885066
stiff 0 1 0 -0.034866182717235635
stiff 0 1 1 0.58100073633364147
stiff 0 1 2 -0.034426587017857037
stiff 0 1 3 -0.016897794086821512
stiff 0 1 4 -0.21107573014019079
stiff 0 1 5 -0.1991176351477589
stiff 0 1 6 0.085426136099861993
stiff 0 1 7 0.093632089758072287
stiff 0 1 8 -0.26367503308171203
stiff 0 2 0 -0.027580420104371992
stiff 0 2 1 -0.034426587017857037
stiff 0 2 2 0.71476630796309337
stiff 0 2 3 -0.034338550872522279
stiff 0 2 4 0.14005721454318981
stiff 0 2 5 -0.1878654296896233
stiff 0 2 6 -0.21300956645285193
stiff 0 2 7 0.1421370798304929
stiff 0 2 8 -0.49974004819954937
stiff 0 3 0 -0.030038831803734723
stiff 0 3 1 -0.016897794086821512
stiff 0 3 2 -0.034338550872522279
stiff 0 3 3 0.58369349608970056
stiff 0 3 4 0.092445248058388038
stiff 0 3 5 0.085650375497859726
stiff 0 3 6 -0.17458081704584735
stiff 0 3 7 -0.2359441190555987
stiff 0 3 8 -0.26998900678142401
stiff 0 4 0 -0.18932473474520861
stiff 0 4 1 -0.21107573014019079
stiff 0 4 2 0.14005721454318981
stiff 0 4 3 0.092445248058388038
stiff 0 4 4 1.9758135226496494
stiff 0 4 5 -0.44374044882097624
stiff 0 4 6 0.0018143921236160568
stiff 0 4 7 -0.3108802675440433
stiff 0 4 8 -1.0551091961244266
stiff 0 5 0 0.12805333194926874
stiff 0 5 1 -0.1991176351477589
stiff 0 5 2 -0.1878654296896233
stiff 0 5 3 0.085650375497859726
stiff 0 5 4 -0.44374044882097624
stiff 0 5 5 1.9498870518090803
stiff 0 5 6 -0.21253080536408003
stiff 0 5 7 -0.0044990448421990074
stiff 0 5 8 -1.1158373953915723
stiff 0 6 0 0.12599152858037965
stiff 0 6 1 0.085426136099861993
stiff 0 6 2 -0.21300956645285193
stiff 0 6 3 -0.17458081704584735
stiff 0 6 4 0.0018143921236160568
stiff 0 6 5 -0.21253080536408003
stiff 0 6 6 1.9593989413397535
stiff 0 6 7 -0.4511494360503206
stiff 0 6 8 -1.1213603732305115
stiff 0 7 0 -0.20669281452138893
stiff 0 7 1 0.093632089758072287
stiff 0 7 2 0.1421370798304929
stiff 0 7 3 -0.2359441190555987
stiff 0 7 4 -0.3108802675440433
stiff 0 7 5 -0.0044990448421990074
stiff 0 7 6 -0.4511494360503206
stiff 0 7 7 1.9655594824427953
stiff 0 7 8 -0.99216297001781162
stiff 0 8 0 -0.41021554436885066
stiff 0 8 1 -0.26367503308171203
stiff 0 8 2 -0.49974004819954937
stiff 0 8 3 -0.26998900678142401
stiff 0 8 4 -1.0551091961244266
stiff 0 8 5 -1.1158373953915723
stiff 0 8 6 -1.1213603732305115
stiff 0 8 7 -0.99216297001781162
stiff 0 8 8 5.728089567195858
mass 1 0 0 0.015213959666290713
mass 1 0 1 -0.0037457920699832969
mass 1 0 2 0.0010117913333636066
mass 1 0 3 -0.0040704970302839916
mass 1 0 4 0.00755629859102589
mass 1 0 5 -0.0018112712013745834
mass 1 0 6 -0.0020362822590705297
mass 1 0 7 0.007386908434707508
mass 1 0 8 0.0036591730765332831
mass 1 1 0 -0.0037457920699832969
mass 1 1 1 0.015016937448363521
mass 1 1 2 -0.0040319185635048025
mass 1 1 3 0.0010117913333636066
mass 1 1 4 0.0074268696889073045
mass 1 1 5 0.0072755019182759286
mass 1 1 6 -0.0020108830743838964
mass 1 1 7 -0.0018112712013745834
mass 1 1 8 0.0035859117289650474
mass 1 2 0 0.0010117913333636066
mass 1 2 1 -0.0040319185635048025
mass 1 2 2 0.017395606716493959
mass 1 2 3 -0.0043859366629566054
mass 1 2 4 -0.0020108830743838964
mass 1 2 5 0.0088521723357432806
mass 1 2 6 0.0087406433719628349
mass 1 2 7 -0.002235894132079844
mass 1 2 8 0.0044576205685705316
mass 1 3 0 -0.0040704970302839916
mass 1 3 1 0.0010117913333636066
mass 1 3 2 -0.0043859366629566054
mass 1 3 3 0.017490510096352163
mass 1 3 4 -0.0020362822590705293
mass 1 3 5 -0.0022358941320798444
mass 1 3 6 0.0088031032798635989
mass 1 3 7 0.0088950796864284629
mass 1 3 8 0.0044859559597488462
mass 1 4 0 0.00755629859102589
mass 1 4 1 0.0074268696889073045
mass 1 4 2 -0.0020108830743838964
mass 1 4 3 -0.0020362822590705293
mass 1 4 4 0.059227178306965091
mass 1 4 5 0.0035859117289650491
mass 1 4 6 -0.016167101528804517
mass 1 4 7 0.0036591730765332849
mass 1 4 8 0.02852103057669517
mass 1 5 0 -0.0018112712013745834
mass 1 5 1 0.0072755019182759286
mass 1 5 2 0.0088521723357432806
mass 1 5 3 -0.0022358941320798444
mass 1 5 4 0.0035859117289650491
mass 1 5 5 0.064091508597892713
mass 1 5 6 0.0044576205685705325
mass 1 5 7 -0.016088933157734891
mass 1 5 8 0.031959599948011744
mass 1 6 0 -0.0020362822590705297
mass 1 6 1 -0.0020108830743838964
mass 1 6 2 0.0087406433719628349
mass 1 6 3 0.0088031032798635989
mass 1 6 4 -0.016167101528804517
mass 1 6 5 0.0044576205685705325
mass 1 6 6 0.070711990582790235
mass 1 6 7 0.0044859559597488462
mass 1 6 8 0.036147375538522861
mass 1 7 0 0.007386908434707508
mass 1 7 1 -0.0018112712013745834
mass 1 7 2 -0.002235894132079844
mass 1 7 3 0.0088950796864284629
mass 1 7 4 0.0036591730765332849
mass 1 7 5 -0.016088933157734891
mass 1 7 6 0.0044859559597488462
mass 1 7 7 0.06475330309688794
mass 1 7 8 0.03239613268292784
mass 1 8 0 0.0036591730765332831
mass 1 8 1 0.0035859117289650474
mass 1 8 2 0.0044576205685705316
mass 1 8 3 0.0044859559597488462
mass 1 8 4 0.02852103057669517
mass 1 8 5 0.031959599948011744
mass 1 8 6 0.036147375538522861
mass 1 8 7 0.03239613268292784
mass 1 8 8 0.25706734003602022
stiff 1 0 0 0.6511877917667197
stiff 1 0 1 -0.023163827089107535
stiff 1 0 2 -0.022961035584968474
stiff 1 0 3 -0.041642528386565417
stiff 1 0 4 -0.22753701609717986
stiff 1 0 5 0.11011647937761736
stiff 1 0 6 0.13031601778308496
stiff 1 0 7 -0.17120275610108832
stiff 1 0 8 -0.40511312566851215
stiff 1 1 0 -0.023163827089107535
stiff 1 1 1 0.65017939540476155
stiff 1 1 2 -0.042418092317531256
stiff 1 1 3 -0.022646640430059154
stiff 1 1 4 -0.23436901534343962
stiff 1 1 5 -0.16478744421634289
stiff 1 1 6 0.12925247447183216
stiff 1 1 7 0.10960026260254431
stiff 1 1 8 -0.40164711308265766
stiff 1 2 0 -0.022961035584968474
stiff 1 2 1 -0.042418092317531256
stiff 1 2 2 0.57506988441568574
stiff 1 2 3 -0.02988635465940221
stiff 1 2 4 0.104705565044534
stiff 1 2 5 -0.16477337417289309
stiff 1 2 6 -0.23083163494725498
stiff 1 2 7 0.10775584261770096
stiff 1 2 8 -0.29666080039587056
stiff 1 3 0 -0.041642528386565417
stiff 1 3 1 -0.022646640430059154
stiff 1 3 2 -0.02988635465940221
stiff 1 3 3 0.5711358172510449
stiff 1 3 4 0.10290303728798686
stiff 1 3 5 0.10583619528117609
stiff 1 3 6 -0.22926943524281698
stiff 1 3 7 -0.16695850347590818
stiff 1 3 8 -0.28947158762545572
stiff 1 4 0 -0.22753701609717986
stiff 1 4 1 -0.23436901534343962
stiff 1 4 2 0.104705565044534
stiff 1 4 3 0.10290303728798686
stiff 1 4 4 1.9019598511811864
stiff 1 4 5 -0.31332775540209462
stiff 1 4 6 -0.018131650834062509
stiff 1 4 7 -0.31150615519904007
stiff 1 4 8 -1.0046968606378919
stiff 1 5 0 0.11011647937761736
stiff 1 5 1 -0.16478744421634289
stiff 1 5 2 -0.16477337417289309
stiff 1 5 3 0.10583619528117609
stiff 1 5 4 -0.31332775540209462
stiff 1 5 5 2.0300579327066743
stiff 1 5 6 -0.42223758288830526
stiff 1 5 7 0.027747125975195974
stiff 1 5 8 -1.2086315766610289
stiff 1 6 0 0.13031601778308496
stiff 1 6 1 0.12925247447183216
stiff 1 6 2 -0.23083163494725498
stiff 1 6 3 -0.22926943524281698
stiff 1 6 4 -0.018131650834062509
stiff 1 6 5 -0.42223758288830526
stiff 1 6 6 2.0019377274472077
stiff 1 6 7 -0.43225656136352886
stiff 1 6 8 -0.92877935442615622
stiff 1 7 0 -0.17120275610108832
stiff 1 7 1 0.10960026260254431
stiff 1 7 2 0.10775584261770096
stiff 1 7 3 -0.16695850347590818
stiff 1 7 4 -0.31150615519904007
stiff 1 7 5 0.027747125975195974
stiff 1 7 6 -0.43225656136352886
stiff 1 7 7 2.0300052128076875
stiff 1 7 8 -1.1931844678635639
stiff 1 8 0 -0.40511312566851215
stiff 1 8 1 -0.40164711308265766
stiff 1 8 2 -0.29666080039587056
stiff 1 8 3 -0.28947158762545572
stiff 1 8 4 -1.0046968606378919
stiff 1 8 5 -1.2086315766610289
stiff 1 8 6 -0.92877935442615622
stiff 1 8 7 -1.1931844678635639
stiff 1 8 8 5.7281848863611371
"#;

/// Parse the `node_ref` / `mass` / `stiff` lines of the oracle dump.
fn parse_oracle() -> (
    Vec<[f64; 2]>,
    std::collections::HashMap<(usize, usize, usize), f64>,
    std::collections::HashMap<(usize, usize, usize), f64>,
) {
    let mut nodes = Vec::new();
    let mut mass = std::collections::HashMap::new();
    let mut stiff = std::collections::HashMap::new();
    for line in ORACLE.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("node_ref") => nodes.push([f[2].parse().unwrap(), f[3].parse().unwrap()]),
            Some("mass") => {
                mass.insert(
                    (
                        f[1].parse().unwrap(),
                        f[2].parse().unwrap(),
                        f[3].parse().unwrap(),
                    ),
                    f[4].parse().unwrap(),
                );
            }
            Some("stiff") => {
                stiff.insert(
                    (
                        f[1].parse().unwrap(),
                        f[2].parse().unwrap(),
                        f[3].parse().unwrap(),
                    ),
                    f[4].parse().unwrap(),
                );
            }
            _ => {}
        }
    }
    assert_eq!(
        nodes.len(),
        9,
        "9 reference dofs of H1_QuadrilateralElement(2)"
    );
    assert_eq!(mass.len(), 162, "2 cells x 9x9 mass entries");
    assert_eq!(stiff.len(), 162, "2 cells x 9x9 diffusion entries");
    (nodes, mass, stiff)
}

fn quad(t: ElementType, p: u8) -> Box<dyn ReferenceElement> {
    ref_elem_vol_with_pyramid_basis(t, p, PyramidBasisType::default()).unwrap()
}

/// D819: the scalar H¹ arms of the mixed table route the quadratic quad cell
/// labels through the same tensor `QuadQk` family as `Quad4` — the family the
/// spaces themselves number these cells with (`h1_family_dofs`).  Pre-D819 the
/// arms returned `QuadSerendipityPk` (8 dofs at p = 2 against a space with 9
/// element dofs on a Quad9 cell).
#[test]
fn d819_quad8_quad9_scalar_h1_dispatch_is_tensor_qk() {
    // Order 0 is the family-agnostic P0 arm.
    assert_eq!(quad(ElementType::Quad9, 0).n_dofs(), 1);
    assert_eq!(quad(ElementType::Quad8, 0).n_dofs(), 1);

    for p in 1..=6u8 {
        let q4 = quad(ElementType::Quad4, p);
        let q8 = quad(ElementType::Quad8, p);
        let q9 = quad(ElementType::Quad9, p);
        let tensor = QuadQk::new(p as usize);
        assert_eq!(q8.n_dofs(), tensor.n_dofs(), "Quad8 p={p} dof count");
        assert_eq!(q9.n_dofs(), tensor.n_dofs(), "Quad9 p={p} dof count");
        // Same family the mesh-side H¹ dof counting uses for these cells.
        assert_eq!(q8.n_dofs(), h1_family_dofs(ElementType::Quad8, p));
        assert_eq!(q9.n_dofs(), h1_family_dofs(ElementType::Quad9, p));
        // Same lattice as the Quad4 arm — one SQUARE geometry, one family.
        assert_eq!(q8.dof_coords(), q4.dof_coords(), "Quad8 p={p} lattice");
        assert_eq!(q9.dof_coords(), q4.dof_coords(), "Quad9 p={p} lattice");
    }

    // The red state this fix removes, stated positively: at order 2 a Quad9
    // cell must expose the 9-dof tensor element (the serendipity element has
    // 8 dofs and cannot index a space with 9 element dofs).
    assert_eq!(quad(ElementType::Quad9, 2).n_dofs(), 9);
    assert_eq!(quad(ElementType::Quad8, 2).n_dofs(), 9);
}

/// D765's HCurl quad pairing is untouched: the `Quad8 | Quad9` vector arms
/// keep selecting the same ND family as `Quad4` (the family the HCurl space's
/// slot/sign tables describe — `hcurl.rs` handles `Quad4 | Quad8` cells).
#[test]
fn d819_quad8_quad9_hcurl_dispatch_still_paired_with_quad4() {
    for p in 1..=4u8 {
        let a = ref_elem_vec(ElementType::Quad4, p, SpaceType::HCurl).unwrap();
        let b8 = ref_elem_vec(ElementType::Quad8, p, SpaceType::HCurl).unwrap();
        let b9 = ref_elem_vec(ElementType::Quad9, p, SpaceType::HCurl).unwrap();
        assert_eq!(a.n_dofs(), b8.n_dofs(), "HCurl Quad8 p={p}");
        assert_eq!(a.n_dofs(), b9.n_dofs(), "HCurl Quad9 p={p}");
    }
}

/// The Gmsh type-16 fixture reads into `Quad8` mesh cells in fem-rs (D297's
/// io mapping — MFEM aborts on the same file), and the mixed table still
/// hands out the tensor element for such cells.
#[test]
fn d819_gmsh_quad8_fixture_reads_and_dispatches_tensor() {
    let msh = read_msh(FIXTURE_Q8.as_bytes()).expect("parse Quad8 fixture");
    let mesh = msh.into_2d().expect("2-D mesh");
    assert_eq!(mesh.n_elements(), 1);
    assert_eq!(mesh.element_type(0), ElementType::Quad8);
    assert_eq!(mesh.element_nodes(0).len(), 8);
    // Tensor dispatch on the Quad8 label (this is the D819 verdict arm).
    assert_eq!(quad(ElementType::Quad8, 2).n_dofs(), 9);
}

/// The MFEM-certified Quad9 real-mesh fixture: fem-rs reads the same bytes as
/// two `Quad9` cells with order-2 geometry, and the tensor `QuadQk(2)` element
/// the (fixed) mixed dispatch selects reproduces MFEM's
/// `H1_FECollection(2,2)` element mass and diffusion matrices on both curved
/// cells entrywise (≤ 1e-12).  This pins the whole adjudicated stack on real
/// geometry: family (tensor, not serendipity), lattice (MFEM slot order) and
/// the Quad9 isoparametric cell map.
#[test]
fn d819_quad9_curved_fixture_matches_mfem_element_matrices() {
    let msh = read_msh(FIXTURE_Q9.as_bytes()).expect("parse Quad9 fixture");
    let mesh = msh.into_2d().expect("2-D mesh");
    assert_eq!(mesh.n_elements(), 2, "two warped quad9 cells");
    for e in mesh.elem_iter() {
        assert_eq!(mesh.element_type(e), ElementType::Quad9);
    }
    assert_eq!(mesh.geom_order(), 2, "order-2 geometry attached (D341)");

    // The element the corrected mixed dispatch selects for (Quad9, p=2).
    let re = quad(ElementType::Quad9, 2);
    assert_eq!(re.n_dofs(), 9);

    // MFEM slot alignment: fem-rs `QuadQk(2)` dofs are MFEM's
    // `H1_QuadrilateralElement(2)` nodes in slot order.
    let (node_ref, mass_want, stiff_want) = parse_oracle();
    for (k, c) in re.dof_coords().iter().enumerate() {
        assert!(
            (c[0] - node_ref[k][0]).abs() < 1e-15 && (c[1] - node_ref[k][1]).abs() < 1e-15,
            "dof {k}: {:?} vs MFEM {:?}",
            c,
            node_ref[k]
        );
    }

    // Deeply converged quadrature on [0,1]^2: order 30 → 16x16 Gauss points.
    // The mass integrand is polynomial (exact much earlier), but the
    // diffusion integrand on a Q2 geometry is *rational* (adj(J)^T adj(J) /
    // det J), so both sides of the comparison integrate with a rule far past
    // convergence and only rounding separates them.
    let q = re.quadrature(30);
    assert_eq!(q.points.len(), 256);

    let mut phi = vec![0.0_f64; 9];
    let mut grad = vec![0.0_f64; 18];
    for e in mesh.elem_iter() {
        let mut mass = vec![0.0_f64; 81];
        let mut stiff = vec![0.0_f64; 81];
        for (qi, xi) in q.points.iter().enumerate() {
            // The cell's own Quad9 tensor-Q2 isoparametric geometry (D243) —
            // the same map MFEM's curved SQUARE evaluates through its Nodes.
            let (det, inv_t) = geometry_jacobian(&mesh, e, xi, 2);
            let w = q.weights[qi] * det.abs();
            re.eval_basis(xi, &mut phi);
            re.eval_grad_basis(xi, &mut grad);
            let mut gp = [[0.0_f64; 2]; 9];
            for i in 0..9 {
                gp[i] = [
                    inv_t[(0, 0)] * grad[2 * i] + inv_t[(0, 1)] * grad[2 * i + 1],
                    inv_t[(1, 0)] * grad[2 * i] + inv_t[(1, 1)] * grad[2 * i + 1],
                ];
            }
            for i in 0..9 {
                for j in 0..9 {
                    mass[i * 9 + j] += w * phi[i] * phi[j];
                    stiff[i * 9 + j] += w * (gp[i][0] * gp[j][0] + gp[i][1] * gp[j][1]);
                }
            }
        }
        for i in 0..9 {
            for j in 0..9 {
                let mw = mass_want[&(e as usize, i, j)];
                let sw = stiff_want[&(e as usize, i, j)];
                assert!(
                    (mass[i * 9 + j] - mw).abs() <= 1e-12 * mw.abs().max(1.0),
                    "cell {e} mass ({i},{j}): {} vs MFEM {mw}",
                    mass[i * 9 + j]
                );
                assert!(
                    (stiff[i * 9 + j] - sw).abs() <= 1e-12 * sw.abs().max(1.0),
                    "cell {e} stiff ({i},{j}): {} vs MFEM {sw}",
                    stiff[i * 9 + j]
                );
            }
        }
    }
}
