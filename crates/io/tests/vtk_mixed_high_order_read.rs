//! 混合高阶 VTU 读取回归：Quad4+Quad8+Tri6（2D）与 Tet10+Hex20（3D）。
//!
//! 修复背景（round-202，mulphy 显示端高阶降阶实测触发）：混合网格此前
//! 只填 `elem_types` 不填 `elem_offsets`——`n_elems()`/`elem_nodes()` 在
//! offsets 为 None 时按 elem_type（uniform 回落 Tet4）的统一 npe 整除推
//! 进，节点数不一的混合表算错单元数并越界。且 VTK offsets 是端点式
//! （无前导 0），Mesh 约定 `offsets[e]..offsets[e+1]`，须补前导 0。

use fem_io::vtk_reader::read_vtu_str;
use fem_mesh::ElementType;

const MIXED_2D: &str = r#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="18" NumberOfCells="3">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 0  1 0 0  0 1 0  1 1 0
          1 0 0  2 0 0  2 1 0  1 1 0  1.5 0 0  2 0.5 0  1.5 1 0  1 0.5 0
          2 0 0  3 0 0  2 1 0  2.5 0 0  2.5 0.5 0  2 0.5 0
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          0 1 3 2
          4 5 6 7 8 9 10 11
          12 13 14 15 16 17
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          4 12 18
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          9 9 5
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"#;

/// 混合 2D 表：3 单元逐一对型、角点切片（VTK 序角点在前）、行对齐不错位。
#[test]
fn mixed_high_order_2d_rows_align() {
    let data = read_vtu_str(MIXED_2D).expect("read");
    let mesh = &data.mesh;
    assert_eq!(mesh.n_elems(), 3, "端点式 offsets 补 0 后 n+1");
    assert_eq!(mesh.element_type_at(0), ElementType::Quad4);
    assert_eq!(mesh.element_type_at(1), ElementType::Quad8);
    assert_eq!(mesh.element_type_at(2), ElementType::Tri6);
    assert_eq!(mesh.elem_nodes(0), &[0, 1, 3, 2]);
    assert_eq!(mesh.elem_nodes(1), &[4, 5, 6, 7, 8, 9, 10, 11]);
    assert_eq!(mesh.elem_nodes(2), &[12, 13, 14, 15, 16, 17]);
}

/// 混合 3D 表（Tet10 + Hex20）：单元数/类型/角点前 4、前 8 槽位。
#[test]
fn mixed_high_order_3d_rows_align() {
    let xml = r#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="30" NumberOfCells="2">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 0  1 0 0  0 1 0  0 0 1
          0.5 0 0  0.5 0.5 0  0 0.5 0  0 0 0.5  0.5 0 0.5  0 0.5 0.5
          1 0 0  2 0 0  2 1 0  1 1 0  1 0 1  2 0 1  2 1 1  1 1 1
          1.5 0 0  2 0.5 0  1.5 1 0  1 0.5 0
          1 0 0.5  2 0 0.5  2 1 0.5  1 1 0.5
          1.5 0 1  2 0.5 1  1.5 1 1  1 1 1
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          0 1 2 3 4 5 6 7 8 9
          10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          10 30
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          10 25
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"#;
    let data = read_vtu_str(xml).expect("read");
    let mesh = &data.mesh;
    assert_eq!(mesh.n_elems(), 2);
    assert_eq!(mesh.element_type_at(0), ElementType::Tet10);
    assert_eq!(mesh.element_type_at(1), ElementType::Hex20);
    // Tet10 前 4 槽 = 角点；Hex20 经 VTK→canonical 重排后角点仍在前 8 槽
    // 原位（D262 重排表只动 8..20 槽）。
    assert_eq!(mesh.elem_nodes(0)[..4], [0, 1, 2, 3]);
    assert_eq!(mesh.elem_nodes(1)[..8], [10, 11, 12, 13, 14, 15, 16, 17]);
}

/// uniform 网格行为不变：types/elem_offsets 保持 None（整除路径）。
#[test]
fn uniform_mesh_stays_on_fast_path() {
    let xml = r#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 0  1 0 0  0 1 0  0 0 1
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          0 1 2 3
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          4
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          10
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"#;
    let data = read_vtu_str(xml).expect("read");
    let mesh = &data.mesh;
    assert_eq!(mesh.n_elems(), 1);
    assert_eq!(mesh.element_type_at(0), ElementType::Tet4);
    assert_eq!(mesh.elem_nodes(0), &[0, 1, 2, 3]);
}
