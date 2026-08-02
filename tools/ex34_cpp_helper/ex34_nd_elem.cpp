// Dump curl-curl element matrix for element 0 (and 16) of the ND space.
#include "mfem.hpp"
#include <cstdio>
using namespace mfem;
int main() {
   Mesh mesh("/mnt/c/Users/lilu/works/fem-pro/fem-rs/data/fichera-mixed.mesh", 1, 1);
   Array<int> submesh_elems; submesh_elems.Append(0); submesh_elems.Append(2);
   submesh_elems.Append(3); submesh_elems.Append(4); submesh_elems.Append(9);
   int max_attr = 0;
   for (int i = 0; i < mesh.GetNE(); i++) max_attr = std::max(max_attr, mesh.GetElement(i)->GetAttribute());
   int submesh_attr = max_attr + 1;
   for (int i = 0; i < submesh_elems.Size(); i++)
      mesh.GetElement(submesh_elems[i])->SetAttribute(submesh_attr);
   mesh.UniformRefinement();
   ND_FECollection fec_nd(1, 3);
   FiniteElementSpace fes_nd(&mesh, &fec_nd);
   ConstantCoefficient muinv(1.0), delta(1e-6);
   CurlCurlIntegrator cc(muinv);
   VectorFEMassIntegrator mass(delta);
   DenseMatrix elmat, elmat2;
   Array<int> vdofs;
   ElementTransformation *tr;
   for (int e = 0; e < mesh.GetNE(); e += 1) {
      if (e != 0 && e != 16 && e != 50) continue;
      const FiniteElement &fe = *fes_nd.GetFE(e);
      tr = fes_nd.GetElementTransformation(e);
      fes_nd.GetElementVDofs(e, vdofs);
      elmat.SetSize(fe.GetDof(), fe.GetDof());
      elmat = 0.0;
      cc.AssembleElementMatrix(fe, *tr, elmat);
      elmat2.SetSize(fe.GetDof(), fe.GetDof());
      elmat2 = 0.0;
      mass.AssembleElementMatrix(fe, *tr, elmat2);
      elmat += elmat2;
      printf("elem %d dofs:", e);
      for (int k = 0; k < vdofs.Size(); k++) printf(" %d", vdofs[k]);
      printf("\n");
      for (int i = 0; i < fe.GetDof(); i++) {
         for (int j = 0; j < fe.GetDof(); j++)
            printf("  %.17g", (double)elmat(i, j));
         printf("\n");
      }
      printf("---\n");
   }
   return 0;
}
