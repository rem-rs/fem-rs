#!/usr/bin/env python3
"""MFEM ex31 reference — anisotropic Maxwell (H(curl) manufactured solution).

Verifies mesh topology and DOF counts match fem-rs for the anisotropic
curl-curl + mass problem on [0,1]² with PEC BC.

Manufactured: E = (sin(πy), sin(πx)), Σ = diag(4.0, 1.5)

Note: MFEM Python's ComputeLpError is broken for H(curl) spaces (SWIG
limitation). Use MFEM C++ ex31 for L² error comparison.

Run:
  uv run python tests/mfem_references/run_ex31.py
"""
import json, sys
from math import pi, sin
from mfem.ser import *
import numpy as np


mesh = Mesh.MakeCartesian2D(8, 8, Element.TRIANGLE, True, 1.0, 1.0)
fec = ND_FECollection(1, 2)
fespace = FiniteElementSpace(mesh, fec)
ndofs = fespace.GetNDofs()

sys.stderr.write("Mesh: %d nodes, %d elements\n" % (mesh.GetNV(), mesh.GetNE()))
sys.stderr.write("Space: %d DOFs (fem-rs baseline: 208)\n" % ndofs)

# Assemble anisotropic curl-curl + mass
sigma_x, sigma_y = 4.0, 1.5
a = BilinearForm(fespace)
a.AddDomainIntegrator(CurlCurlIntegrator(ConstantCoefficient(1.0)))
# VectorFEMassIntegrator with anisotropic coefficient not available in Python
a.AddDomainIntegrator(VectorFEMassIntegrator(VectorConstantCoefficient([sigma_x, sigma_y])))
a.Assemble()

A_sp = a.SpMat()
nnz = A_sp.NumberOfNonZeroElements()

results = {
    "summary": "ex31: mesh/space verification via MFEM Python",
    "verified_metrics": {
        "n_dofs": {"mfem": ndofs, "fem_rs": 208, "match": ndofs == 208},
        "n_nodes": mesh.GetNV(),
        "n_elements": mesh.GetNE(),
        "matrix_nnz": nnz,
    },
    "note": "Full anisotropic solve + L² error requires MFEM C++ ex31. "
            "Python SWIG bindings lack anisotropic VectorFEMassIntegrator support.",
}
print(json.dumps(results, indent=2))
