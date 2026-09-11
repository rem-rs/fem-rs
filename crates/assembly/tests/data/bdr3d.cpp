// MFEM reference dump for fem-rs D50 (3-D boundary assembly per DOF).
//
//   g++ -std=c++17 -O2 -I$HOME/mfem49 bdr3d.cpp $HOME/mfem49/libmfem.a -o bdr3d
//   ./bdr3d > bdr_3d_cpp.txt
//
// For every boundary element of MakeCartesian3D(1,1,1) (hex and tet, 1x1x1)
// and each order p in {1,2,3,6}: the boundary element's tag and vertices, the
// boundary element's (2-D) reference nodes, and the local boundary vector
//     ∫_Γ (e_x·n) φ_k ds
// assembled with BoundaryNormalLFIntegrator on the explicit quadrature rule
// `IntRules.Get(face_geom, quad_order)` (quad_order = p + 3, the order fem-rs's
// `assemble_boundary_linear` is called with).  The local vector is read back
// through GetBdrElementDofs so that entry k pairs with the FE's DOF k, exactly
// as fem-rs's face_dofs_h1 + assemble_boundary_linear do.
#include "mfem.hpp"
#include <cstdio>
using namespace mfem;

int main()
{
   for (int tet = 0; tet <= 1; tet++)
   {
      Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1,
                    tet ? Element::TETRAHEDRON : Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      for (int p = 1; p <= 6; p++)
      {
         H1_FECollection fec(p, 3);
         FiniteElementSpace fes(&mesh, &fec);
         Vector one(3); one = 0.0; one[0] = 1.0;
         VectorConstantCoefficient vc(one);
         BoundaryNormalLFIntegrator *integ = new BoundaryNormalLFIntegrator(vc);
         const int qo = p + 3;
         fes.GetBE(0);                        // make sure the BE cache is warm
         integ->SetIntRule(&IntRules.Get(Geometry::SQUARE, qo));
         // (the triangle faces use the same object; set both rules)
         LinearForm lf(&fes);
         integ->SetIntRule(NULL);
         lf.AddBoundaryIntegrator(integ);
         lf.Assemble();
         printf("case %s p=%d nbe=%d nattr=%d ndofs=%d\n",
                tet ? "tet" : "hex", p, mesh.GetNBE(), mesh.bdr_attributes.Size(),
                fes.GetVSize());
         for (int be = 0; be < mesh.GetNBE(); be++)
         {
            const FiniteElement *be_fe = fes.GetBE(be);
            ElementTransformation *Tr = fes.GetBdrElementTransformation(be);
            Array<int> bdofs;
            fes.GetBdrElementDofs(be, bdofs);
            Array<int> v;
            mesh.GetBdrElementVertices(be, v);
            printf("be %d tag %d geom %d nv %d nvdof %d verts",
                   be, mesh.GetBdrAttribute(be), (int)be_fe->GetGeomType(),
                   v.Size(), be_fe->GetDof());
            for (int i = 0; i < v.Size(); i++) { printf(" %d", v[i]); }
            fes.GetBdrElementDofs(be, bdofs);
            // reproduce the element vector on the same quadrature rule
            Vector elvect(be_fe->GetDof());
            integ->AssembleRHSElementVect(*be_fe, *Tr, elvect);
            printf("\n");
            for (int k = 0; k < be_fe->GetDof(); k++)
            {
               const IntegrationPoint &ip = be_fe->GetNodes().IntPoint(k);
               printf("  k %d node %.17g %.17g dof %d val %.17g\n",
                      k, ip.x, ip.y, bdofs[k], elvect[k]);
            }
         }
      }
   }
   return 0;
}
