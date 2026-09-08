//! Element / DOF marking for the shifted boundary and interface methods —
//! 1:1 serial port of MFEM `miniapps/shifted/marking.{hpp,cpp}`.
//!
//! The C++ is parallel: `MarkElements` additionally checks face-neighbour
//! elements, `ListEssentialTDofs` synchronizes across ranks, and boundary
//! attributes are permuted when SBM faces sit on the true boundary.  In serial
//! all of those reduce to the local branch only.

use std::collections::HashSet;

use fem_mesh::topology::MeshTopology;
use fem_space::dof_manager::{DofManager, EdgeKey, FaceKey, QuadFaceKey};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

use crate::postproc::grid_function::GridFunction;

/// Element type related to shifted boundaries (MFEM `SBElementType`).
///
/// For more than one level set, the marker is set to `CUT + level_set_index`
/// to discern between the different level sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SBElementType {
    Inside = 0,
    Outside = 1,
    Cut = 2,
}

use SBElementType::{Cut, Inside, Outside};

/// Marking of elements, faces and DOFs for the shifted boundary method
/// (MFEM `ShiftedFaceMarker`).  A point is considered *inside* when the level
/// set function is positive.
pub struct ShiftedFaceMarker<'a, M: MeshTopology + Clone + 'static> {
    /// Mesh whose elements have to be marked.
    mesh: &'a M,
    /// FESpace associated with the solution.
    pfes: &'a H1Space<M>,
    /// Indicates whether cut cells will be included in assembly.
    include_cut_cell: bool,
    /// Indicates whether all elements have been marked at least once.
    initial_marking_done: bool,
    level_set_index: i32,
}

impl<'a, M: MeshTopology + Clone + 'static> ShiftedFaceMarker<'a, M> {
    pub fn new(mesh: &'a M, pfes: &'a H1Space<M>, include_cut_cell: bool) -> Self {
        ShiftedFaceMarker {
            mesh,
            pfes,
            include_cut_cell,
            initial_marking_done: false,
            level_set_index: 0,
        }
    }

    fn outside_of_domain(&self, value: f64) -> bool {
        // Tolerance relevant for points exactly on the zero level set.
        const EPS: f64 = 1e-10;
        if self.include_cut_cell {
            // Points on the zero LS are considered outside the domain.
            value - EPS < 0.0
        } else {
            // Points on the zero LS are considered inside the domain.
            value + EPS < 0.0
        }
    }

    /// Mark all the elements in the mesh using the [`SBElementType`]
    /// (MFEM `MarkElements`; the face-neighbour branch is a serial no-op).
    pub fn mark_elements(
        &mut self,
        ls: &GridFunction<'_, H1Space<M>>,
        elem_marker: &mut Vec<i32>,
    ) {
        let ne = self.mesh.n_elements();
        elem_marker.clear();
        elem_marker.resize(ne, Inside as i32);
        if self.initial_marking_done {
            self.level_set_index += 1;
        }

        for e in self.mesh.elem_iter() {
            // Evaluate the level set at the FE nodes of this element
            // (MFEM: `pfes_sltn->GetFE(i)->GetNodes()` + `ls_func.GetValues`).
            let pts = super::elem_dof_points(self.pfes, e);
            let nd = pts.len();
            let mut count = 0_usize;
            for (xi, _x) in &pts {
                let v = ls.evaluate_at_element(e, xi);
                if self.outside_of_domain(v) {
                    count += 1;
                }
            }
            let m = &mut elem_marker[e as usize];
            if count == nd {
                // completely outside
                *m = Outside as i32;
            } else if count > 0 {
                // partially outside
                assert!(
                    *m <= Outside as i32,
                    "One element cut by multiple level-sets."
                );
                *m = Cut as i32 + self.level_set_index;
            }
        }
        self.initial_marking_done = true;
    }

    /// List DOFs associated with the surrogate boundary
    /// (MFEM `ListShiftedFaceDofs` with the default `func_dof_marking = true`,
    /// i.e. the `ListShiftedFaceDofs2` implementation).
    ///
    /// If `include_cut_cell = false`, the surrogate boundary consists of the
    /// faces between elements cut by the true boundary and *inside* elements;
    /// if `include_cut_cell = true`, of the faces between *outside* elements
    /// and cut elements.
    pub fn list_shifted_face_dofs(
        &self,
        elem_marker: &[i32],
        sface_dof_list: &mut Vec<usize>,
    ) {
        sface_dof_list.clear();

        // Cell-centred marker: 0 inside, 1 outside (or cut, when cut cells are
        // excluded from assembly).
        let mat: Vec<f64> = elem_marker
            .iter()
            .map(|&m| {
                if m == Outside as i32 || (m >= Cut as i32 && !self.include_cut_cell) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        // ProjectDiscCoefficient(mat, ARITHMETIC): DOFs shared by inside and
        // outside/cut elements receive a value strictly between 0 and 1.
        let marker = super::project_disc_average(self.pfes, |e, _xi, _x| mat[e as usize]);
        for (j, &v) in marker.iter().enumerate() {
            if v > 0.1 && v < 0.9 {
                sface_dof_list.push(j);
            }
        }

        // Add boundary faces that we want to model as SBM faces.
        if self.include_cut_cell {
            for f in 0..self.mesh.n_boundary_faces() as u32 {
                let (e1, _e2) = self.mesh.face_elements(f);
                if elem_marker[e1 as usize] >= Cut as i32 {
                    sface_dof_list
                        .extend(face_dofs(self.mesh, self.pfes.dof_manager(), f).into_iter().map(|d| d as usize));
                }
            }
        }
    }

    /// List the DOFs that will be inactive for the computation on the surrogate
    /// domain (MFEM `ListEssentialTDofs`): DOFs of elements located outside the
    /// true domain (and, when `include_cut_cell = false`, of cut elements),
    /// plus the true-boundary DOFs, minus the DOFs on the surrogate boundary.
    ///
    /// `ess_shift_bdr` receives one flag per distinct boundary tag
    /// (ascending), 1 where the tag contains SBM faces at the true boundary
    /// (C++: the appended `bdr_attributes.Max()+1` attribute).
    pub fn list_essential_tdofs(
        &self,
        elem_marker: &[i32],
        sface_dof_list: &[usize],
        ess_tdof_list: &mut Vec<usize>,
        ess_shift_bdr: &mut Vec<i32>,
    ) {
        let dm = self.pfes.dof_manager();
        let tags = super::boundary_tags(self.mesh);
        ess_shift_bdr.clear();
        ess_shift_bdr.resize(tags.len(), 0);

        let mut ess_vdofs = vec![false; self.pfes.n_dofs()];
        // True-boundary DOFs; SBM faces at the true boundary are excluded from
        // the essential set and flagged in `ess_shift_bdr`.
        let mut ess_bdr_dofs: HashSet<usize> = HashSet::new();
        for f in 0..self.mesh.n_boundary_faces() as u32 {
            let tag = self.mesh.face_tag(f);
            let ti = tags.binary_search(&tag).expect("tag from the mesh");
            let (e1, _e2) = self.mesh.face_elements(f);
            let fdofs = face_dofs(self.mesh, dm, f);
            let sbm = self.include_cut_cell && elem_marker[e1 as usize] >= Cut as i32;
            if sbm {
                ess_shift_bdr[ti] = 1;
            } else {
                for d in fdofs {
                    ess_bdr_dofs.insert(d as usize);
                }
            }
        }

        // DOFs of elements outside the domain (and intersected by the boundary,
        // when cut cells are not included).
        for e in self.mesh.elem_iter() {
            let m = elem_marker[e as usize];
            let inactive = if self.include_cut_cell {
                m == Outside as i32
            } else {
                m == Outside as i32 || m >= Cut as i32
            };
            if inactive {
                for &d in self.pfes.element_dofs(e) {
                    ess_vdofs[d as usize] = true;
                }
            }
        }

        // Mark the essential true-boundary DOFs.
        for &d in &ess_bdr_dofs {
            ess_vdofs[d] = true;
        }

        // Unmark DOFs that are on SBM faces (but not on Dirichlet boundaries).
        for &d in sface_dof_list {
            if !ess_bdr_dofs.contains(&d) {
                ess_vdofs[d] = false;
            }
        }

        ess_tdof_list.clear();
        for (i, &m) in ess_vdofs.iter().enumerate() {
            if m {
                ess_tdof_list.push(i);
            }
        }
    }
}

/// DOFs of an H1 space located on mesh face `f` (vertex DOFs, edge DOFs and,
/// in 3D, face-interior DOFs — mirroring `fem_space::dirichlet::boundary_dofs`).
fn face_dofs<M: MeshTopology>(mesh: &M, dm: &DofManager, f: u32) -> Vec<u32> {
    let nodes = mesh.face_nodes(f);
    let mut out: Vec<u32> = Vec::new();

    // Vertex DOFs (NC-safe: translate through phys_to_vertex_dof).
    for &node in nodes {
        let d = dm
            .phys_to_vertex_dof
            .get(&node)
            .copied()
            .unwrap_or(node as u32);
        out.push(d);
    }

    // Edge DOFs for each boundary edge of the face (wrap-around).
    for i in 0..nodes.len() {
        let key = EdgeKey::new(nodes[i], nodes[(i + 1) % nodes.len()]);
        if let Some(dofs) = dm.edge_pk_map.get(&key) {
            out.extend_from_slice(dofs);
        } else if let Some(dofs) = dm.edge_dof2_map.get(&key) {
            out.extend_from_slice(dofs);
        } else if let Some(&d) = dm.edge_dof_map.get(&key) {
            out.push(d);
        }
    }

    // Face-interior DOFs in 3D.
    if mesh.dim() == 3 {
        match nodes.len() {
            3 => {
                let key = FaceKey::new(nodes[0], nodes[1], nodes[2]);
                if let Some(dofs) = dm.face_pk_map.get(&key) {
                    out.extend_from_slice(dofs);
                }
            }
            4 => {
                let key = QuadFaceKey::new(nodes[0], nodes[1], nodes[2], nodes[3]);
                if let Some(dofs) = dm.quad_face_pk_map.get(&key) {
                    out.extend_from_slice(dofs);
                }
            }
            _ => {}
        }
    }
    out
}
