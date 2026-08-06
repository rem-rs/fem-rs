#include "mfem.hpp"
#include <iostream>
using namespace mfem;
static real_t a_ = 0.2;
void quad_trans(real_t u, real_t v, real_t &x, real_t &y)
{
   real_t a = a_;
   real_t d = 4.0 * a * (M_SQRT2 - 2.0 * a) * (1.0 - 2.0 * v);
   real_t v0 = (1.0 + M_SQRT2) * (M_SQRT2 * a - 2.0 * v) *
               ((4.0 - 3 * M_SQRT2) * a + (8.0 * (M_SQRT2 - 1.0) * a - 2.0) * v) / d;
   real_t r = 2.0 * ((M_SQRT2 - 1.0) * a * a * (1.0 - 4.0 *v) +
                     2.0 * (1.0 + M_SQRT2 * (1.0 + 2.0 * (2.0 * a - M_SQRT2 - 1.0) * a)) * v * v) / d;
   real_t t = asin(v / r) * u / v;
   x = r * sin(t);
   y = r * cos(t) - v0;
}
void trans(const Vector &u, Vector &x)
{
   real_t tol = 1e-4;
   if (u[1] > 0.5 - tol || u[1] < -0.5 + tol) { x = u; return; }
   if (u[0] > 1.0 - tol || u[0] < -1.0 + tol || fabs(u[0]) < tol) { x = u; return; }
   if (u[0] > 0.0)
   {
      if (u[1] > fabs(u[0] - 0.5)) { quad_trans(u[0] - 0.5, u[1], x[0], x[1]); x[0] += 0.5; return; }
      if (u[1] < -fabs(u[0] - 0.5)) { quad_trans(u[0] - 0.5, -u[1], x[0], x[1]); x[0] += 0.5; x[1] *= -1.0; return; }
      if (u[0] - 0.5 > fabs(u[1])) { quad_trans(u[1], u[0] - 0.5, x[1], x[0]); x[0] += 0.5; return; }
      if (u[0] - 0.5 < -fabs(u[1])) { quad_trans(u[1], 0.5 - u[0], x[1], x[0]); x[0] *= -1.0; x[0] += 0.5; return; }
   }
   else
   {
      if (u[1] > fabs(u[0] + 0.5)) { quad_trans(u[0] + 0.5, u[1], x[0], x[1]); x[0] -= 0.5; return; }
      if (u[1] < -fabs(u[0] + 0.5)) { quad_trans(u[0] + 0.5, -u[1], x[0], x[1]); x[0] -= 0.5; x[1] *= -1.0; return; }
      if (u[0] + 0.5 > fabs(u[1])) { quad_trans(u[1], u[0] + 0.5, x[1], x[0]); x[0] -= 0.5; return; }
      if (u[0] + 0.5 < -fabs(u[1])) { quad_trans(u[1], -0.5 - u[0], x[1], x[0]); x[0] *= -1.0; x[0] -= 0.5; return; }
   }
   x = u;
}
Mesh *Build(int ref, bool transform)
{
   Mesh * mesh = new Mesh(2, 29, 16, 24, 2);
   int vi[4];
   for (int i=0; i<2; i++)
   {
      int o = 13 * i;
      vi[0] = o + 0; vi[1] = o + 3; vi[2] = o + 4; vi[3] = o + 1; mesh->AddQuad(vi);
      vi[0] = o + 1; vi[1] = o + 4; vi[2] = o + 5; vi[3] = o + 2; mesh->AddQuad(vi);
      vi[0] = o + 5; vi[1] = o + 8; vi[2] = o + 9; vi[3] = o + 2; mesh->AddQuad(vi);
      vi[0] = o + 8; vi[1] = o + 12; vi[2] = o + 15; vi[3] = o + 9; mesh->AddQuad(vi);
      vi[0] = o + 11; vi[1] = o + 14; vi[2] = o + 15; vi[3] = o + 12; mesh->AddQuad(vi);
      vi[0] = o + 10; vi[1] = o + 13; vi[2] = o + 14; vi[3] = o + 11; mesh->AddQuad(vi);
      vi[0] = o + 6; vi[1] = o + 13; vi[2] = o + 10; vi[3] = o + 7; mesh->AddQuad(vi);
      vi[0] = o + 0; vi[1] = o + 6; vi[2] = o + 7; vi[3] = o + 3; mesh->AddQuad(vi);
   }
   vi[0] =  0; vi[1] =  6; mesh->AddBdrSegment(vi, 1);
   vi[0] =  6; vi[1] = 13; mesh->AddBdrSegment(vi, 1);
   vi[0] = 13; vi[1] = 19; mesh->AddBdrSegment(vi, 1);
   vi[0] = 19; vi[1] = 26; mesh->AddBdrSegment(vi, 1);
   vi[0] = 28; vi[1] = 22; mesh->AddBdrSegment(vi, 2);
   vi[0] = 22; vi[1] = 15; mesh->AddBdrSegment(vi, 2);
   vi[0] = 15; vi[1] =  9; mesh->AddBdrSegment(vi, 2);
   vi[0] =  9; vi[1] =  2; mesh->AddBdrSegment(vi, 2);
   for (int i=0; i<2; i++)
   {
      int o = 13 * i;
      vi[0] = o +  7; vi[1] = o +  3; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o + 10; vi[1] = o +  7; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o + 11; vi[1] = o + 10; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o + 12; vi[1] = o + 11; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  8; vi[1] = o + 12; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  5; vi[1] = o +  8; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  4; vi[1] = o +  5; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  3; vi[1] = o +  4; mesh->AddBdrSegment(vi, 3 + i);
   }
   real_t d[2];
   real_t a = a_ / M_SQRT2;
   d[0] = -1.0; d[1] = -0.5; mesh->AddVertex(d);
   d[0] = -1.0; d[1] =  0.0; mesh->AddVertex(d);
   d[0] = -1.0; d[1] =  0.5; mesh->AddVertex(d);
   d[0] = -0.5 - a; d[1] =   -a; mesh->AddVertex(d);
   d[0] = -0.5 - a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] = -0.5 - a; d[1] =    a; mesh->AddVertex(d);
   d[0] = -0.5; d[1] = -0.5; mesh->AddVertex(d);
   d[0] = -0.5; d[1] =   -a; mesh->AddVertex(d);
   d[0] = -0.5; d[1] =    a; mesh->AddVertex(d);
   d[0] = -0.5; d[1] =  0.5; mesh->AddVertex(d);
   d[0] = -0.5 + a; d[1] =   -a; mesh->AddVertex(d);
   d[0] = -0.5 + a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] = -0.5 + a; d[1] =    a; mesh->AddVertex(d);
   d[0] =  0.0; d[1] = -0.5; mesh->AddVertex(d);
   d[0] =  0.0; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  0.0; d[1] =  0.5; mesh->AddVertex(d);
   d[0] =  0.5 - a; d[1] =   -a; mesh->AddVertex(d);
   d[0] =  0.5 - a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  0.5 - a; d[1] =    a; mesh->AddVertex(d);
   d[0] =  0.5; d[1] = -0.5; mesh->AddVertex(d);
   d[0] =  0.5; d[1] =   -a; mesh->AddVertex(d);
   d[0] =  0.5; d[1] =    a; mesh->AddVertex(d);
   d[0] =  0.5; d[1] =  0.5; mesh->AddVertex(d);
   d[0] =  0.5 + a; d[1] =   -a; mesh->AddVertex(d);
   d[0] =  0.5 + a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  0.5 + a; d[1] =    a; mesh->AddVertex(d);
   d[0] =  1.0; d[1] = -0.5; mesh->AddVertex(d);
   d[0] =  1.0; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  1.0; d[1] =  0.5; mesh->AddVertex(d);
   mesh->FinalizeTopology();
   mesh->SetCurvature(1, true);
   {
      Array<int> v2v(mesh->GetNV());
      for (int i = 0; i < v2v.Size() - 3; i++) { v2v[i] = i; }
      v2v[v2v.Size() - 3] = 0;
      v2v[v2v.Size() - 2] = 1;
      v2v[v2v.Size() - 1] = 2;
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         Element *el = mesh->GetElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++) { v[j] = v2v[v[j]]; }
      }
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         Element *el = mesh->GetBdrElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++) { v[j] = v2v[v[j]]; }
      }
      mesh->RemoveUnusedVertices();
      mesh->RemoveInternalBoundaries();
   }
   mesh->SetCurvature(3, true);
   for (int l = 0; l < ref; l++) { mesh->UniformRefinement(); }
   if (transform) { mesh->Transform(trans); }
   return mesh;
}
int main(int argc, char *argv[])
{
   std::cout.precision(10);
   // stage A: after stitch (before curvature)
   {
      Mesh *m = Build(0, false);
      std::cout << "STITCH nv=" << m->GetNV() << " ne=" << m->GetNE() << "\n";
      for (int i = 0; i < m->GetNV(); i++)
      {
         const double *v = m->GetVertex(i);
         if (v[0] < -0.99 || v[0] > 0.99 || v[0] == 0.0)
            std::cout << "  v" << i << " = (" << v[0] << "," << v[1] << ")\n";
      }
      delete m;
   }
   // stage B: after ref x2 + transform (full pipeline)
   {
      Mesh *m = Build(2, true);
      std::cout << "FULL nv=" << m->GetNV() << " ne=" << m->GetNE() << "\n";
      int n0 = 0, np = 0, nm = 0;
      for (int i = 0; i < m->GetNV(); i++)
      {
         const double *v = m->GetVertex(i);
         if (v[0] == 0.0) n0++;
         if (v[0] > 0.99) np++;
         if (v[0] < -0.99) nm++;
      }
      std::cout << "  verts at x=0: " << n0 << ", x>0.99: " << np << ", x<-0.99: " << nm << "\n";
      for (int i = 0; i < 5; i++)
      {
         const double *v = m->GetVertex(i);
         std::cout << "  v" << i << " = (" << v[0] << "," << v[1] << ")\n";
      }
      delete m;
   }
   return 0;
}
