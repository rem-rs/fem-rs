//! Vector-DOF ordering — mirrors MFEM `linalg/ordering.hpp` semantics.
//!
//! ⚠️ **MFEM 的命名与直觉相反**（fem-rs 与 MFEM 保持一致，见下）：
//!
//! | 变体 | MFEM `Ordering::Map` 公式 | 全局布局效果 |
//! |---|---|---|
//! | [`Ordering::ByNodes`]（= MFEM `byNODES`，值 0，**MFEM 默认**） | `vdof = dof + ndofs*vd` | **块布局**：先全部分量 0 的 DOF，再分量 1 …（x 块 + y 块） |
//! | [`Ordering::ByVdim`]（= MFEM `byVDIM`，值 1） | `vdof = vd + vdim*dof` | **按节点交错**（node-major） |
//!
//! MFEM 按"内层循环"命名：`byNODES` 中 nodes 是内层循环（`dof+1` 相邻），
//! 结果却是分量分块；`byVDIM` 中 vdim 是内层循环（`vd+1` 相邻），结果反而
//! 是按节点交错。**不要凭名字猜语义。**
//!
//! fem-rs 的向量 FE 空间（如 [`crate::VectorH1Space`]）使用
//! [`Ordering::ByNodes`]（= MFEM 默认 `byNODES`），全局 DOF 编号
//! `c*n_scalar + s` 即公式 `dof + ndofs*vd`；写 `.gf` 文件时头部
//! `Ordering: 0` 与之对应。

/// Vector-DOF ordering (values match MFEM: `byNODES = 0`, `byVDIM = 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    /// MFEM `Ordering::byNODES` — **block layout**:
    /// `vdof = dof + ndofs*vd`.  The fem-rs default (matches `VectorH1Space`).
    ByNodes = 0,
    /// MFEM `Ordering::byVDIM` — node-major interleaved:
    /// `vdof = vd + vdim*dof`.
    ByVdim = 1,
}

impl Ordering {
    /// MFEM `Ordering::Map`: global vdof for scalar DOF `dof`, component `vd`,
    /// in a space with `ndofs` scalar DOFs and `vdim` components.
    #[inline]
    pub fn map(self, ndofs: usize, vdim: usize, dof: usize, vd: usize) -> usize {
        match self {
            Ordering::ByNodes => dof + ndofs * vd,
            Ordering::ByVdim => vd + vdim * dof,
        }
    }

    /// MFEM `Ordering::DofsToVDofs`: expand a scalar-DOF list into the
    /// per-element vdof list for this ordering.
    ///
    /// - `ByNodes` → interleaved (node-major): `[s0_x, s0_y, s1_x, s1_y, …]`
    ///   — exactly the `VectorH1Space` element DOF table.
    /// - `ByVdim` → component-major: `[s0_x, s1_x, …, s0_y, s1_y, …]`.
    pub fn dofs_to_vdofs(
        self,
        ndofs: usize,
        vdim: usize,
        dofs: &[usize],
        out: &mut Vec<usize>,
    ) {
        out.clear();
        match self {
            Ordering::ByNodes => {
                for &d in dofs {
                    for vd in 0..vdim {
                        out.push(d + ndofs * vd);
                    }
                }
            }
            Ordering::ByVdim => {
                for vd in 0..vdim {
                    for &d in dofs {
                        out.push(vd + vdim * d);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_matches_mfem_formulas() {
        // 2 scalar DOFs, vdim = 2 (the ex28 situation).
        let ndofs = 2;
        let vdim = 2;
        // byNODES: block layout (x-block then y-block).
        assert_eq!(Ordering::ByNodes.map(ndofs, vdim, 0, 0), 0);
        assert_eq!(Ordering::ByNodes.map(ndofs, vdim, 1, 0), 1);
        assert_eq!(Ordering::ByNodes.map(ndofs, vdim, 0, 1), 2);
        assert_eq!(Ordering::ByNodes.map(ndofs, vdim, 1, 1), 3);
        // byVDIM: node-major interleaved.
        assert_eq!(Ordering::ByVdim.map(ndofs, vdim, 0, 0), 0);
        assert_eq!(Ordering::ByVdim.map(ndofs, vdim, 0, 1), 1);
        assert_eq!(Ordering::ByVdim.map(ndofs, vdim, 1, 0), 2);
        assert_eq!(Ordering::ByVdim.map(ndofs, vdim, 1, 1), 3);
    }

    #[test]
    fn dofs_to_vdofs_matches_vector_h1_element_table() {
        let ndofs = 100;
        let vdim = 2;
        let dofs = [7usize, 42];
        let mut out = Vec::new();
        Ordering::ByNodes.dofs_to_vdofs(ndofs, vdim, &dofs, &mut out);
        // Interleaved: (7x, 7y, 42x, 42y) with the block-layout mapping.
        assert_eq!(out, vec![7, 7 + 100, 42, 42 + 100]);
        Ordering::ByVdim.dofs_to_vdofs(ndofs, vdim, &dofs, &mut out);
        // byVDIM element table is component-major: [vd0's dofs, vd1's dofs],
        // each vdof = vd + vdim*dof.
        assert_eq!(out, vec![2 * 7, 2 * 42, 2 * 7 + 1, 2 * 42 + 1]);
    }

    #[test]
    fn values_match_mfem_header() {
        // .gf header writes `Ordering: <value>`; byNODES must be 0.
        assert_eq!(Ordering::ByNodes as i32, 0);
        assert_eq!(Ordering::ByVdim as i32, 1);
    }
}
