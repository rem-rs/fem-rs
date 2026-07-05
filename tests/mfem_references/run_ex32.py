#!/usr/bin/env python3
"""MFEM ex32 reference — impedance Maxwell (H(curl) with Robin BC).

Verifies mesh topology and DOF counts for the impedance Maxwell problem
on [0,1]² with tangential Robin BC: n×(curl E) + γ(n×E) = g.

Manufactured: E = (0, cos(πx)sin(πy))

Note: MFEM Python SWIG bindings lack tangential boundary integrators.
Use MFEM C++ ex32 for full numerical comparison.

Run:
  uv run python tests/mfem_references/run_ex32.py
"""
import json, sys
from math import pi, sin, cos
from mfem.ser import *
import numpy as np


mesh = Mesh.MakeCartesian2D(8, 8, Element.TRIANGLE, True, 1.0, 1.0)
fec = ND_FECollection(1, 2)
fespace = FiniteElementSpace(mesh, fec)
ndofs = fespace.GetNDofs()

sys.stderr.write("Mesh: %d nodes, %d elements\n" % (mesh.GetNV(), mesh.GetNE()))
sys.stderr.write("Space: %d DOFs (fem-rs baseline: 208)\n" % ndofs)

# Assemble volume terms only (curl-curl + mass)
a = BilinearForm(fespace)
a.AddDomainIntegrator(CurlCurlIntegrator(ConstantCoefficient(1.0)))
a.AddDomainIntegrator(VectorFEMassIntegrator(ConstantCoefficient(1.0)))
a.Assemble()

results = {
    "summary": "ex32: mesh/space verification via MFEM Python",
    "verified_metrics": {
        "n_dofs": {"mfem": ndofs, "fem_rs": 208, "match": ndofs == 208},
        "n_nodes": mesh.GetNV(),
        "n_elements": mesh.GetNE(),
    },
    "note": "Tangential Robin BC requires MFEM C++ ex32 for full assembly. "
            "Python bindings lack VectorFEBoundaryTangentLFIntegrator.",
}
print(json.dumps(results, indent=2))
