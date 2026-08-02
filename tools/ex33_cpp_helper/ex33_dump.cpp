// ex33 AAA dump helper — prints poles/zeros/coeffs at full precision.
// Build (WSL): g++ -std=c++17 -O3 -I~/works/mfem -I~/works/mfem/examples \
//              ex33_dump.cpp -L~/works/mfem -lmfem -llapack -lblas -o ex33_dump
#include "mfem.hpp"
#include "ex33.hpp"
#include <cstdio>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   double alpha = 0.33;
   if (argc > 1) { alpha = atof(argv[1]); }

   // Re-implement ComputePartialFractionApproximation with extra prints
   // to expose zeros before/after DeleteFirst.
   MFEM_VERIFY(alpha < 1. && alpha > 0., "alpha in (0,1)");
   const real_t lmax = 1000.;
   const real_t tol = 1e-10;
   const int npoints = 1000;
   const int max_order = 100;

   Vector x(npoints), val(npoints);
   real_t dx = lmax / (real_t)(npoints-1);
   for (int i = 0; i < npoints; i++)
   {
      x(i) = dx * (real_t)i;
      val(i) = pow(x(i), 1.-alpha);
   }

   Array<real_t> z, f;
   Vector w;
   RationalApproximation_AAA(val, x, z, f, w, tol, max_order);

   printf("AAA: z.Size=%d w.Size=%d\n", z.Size(), w.Size());
   for (int i = 0; i < z.Size(); i++)
   {
      printf("  z[%d]=%.17g f[%d]=%.17g w[%d]=%.17g\n",
             i, (double)z[i], i, (double)f[i], i, (double)w[i]);
   }

   Vector vecz, vecf;
   vecz.SetDataAndSize(z.GetData(), z.Size());
   vecf.SetDataAndSize(f.GetData(), f.Size());

   real_t scale;
   Array<real_t> poles, zeros;
   ComputePolesAndZeros(vecz, vecf, w, poles, zeros, scale);
   printf("poles.Size=%d zeros.Size=%d scale=%.17g\n",
          poles.Size(), zeros.Size(), (double)scale);
   for (int i = 0; i < poles.Size(); i++)
   {
      printf("  pole[%d]=%.17g\n", i, (double)poles[i]);
   }
   for (int i = 0; i < zeros.Size(); i++)
   {
      printf("  zero[%d]=%.17g\n", i, (double)zeros[i]);
   }

   zeros.DeleteFirst(0.0);
   printf("after DeleteFirst(0.0): zeros.Size=%d\n", zeros.Size());

   Array<real_t> coeffs;
   PartialFractionExpansion(scale, poles, zeros, coeffs);
   printf("coeffs.Size=%d\n", coeffs.Size());
   for (int i = 0; i < coeffs.Size(); i++)
   {
      printf("  c[%d]=%.17g  d[%d]=%.17g\n",
             i, (double)coeffs[i], i, (double)(-poles[i]));
   }
   return 0;
}
