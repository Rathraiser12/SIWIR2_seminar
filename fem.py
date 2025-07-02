#!/usr/bin/env python3
# helmholtz_disk_eigs.py   –  FEniCS‑x ≥0.8  –  PETSc / SLEPc ≥3.19
#
# geometry & weak form exactly as in Assignment‑2.pdf
#
# optional PETSc/SLEPc flags (after “--”) e.g.:
#   mpirun -n 4 python -m petsc4py helmholtz_disk_eigs.py -- \
#          -eps_type krylovschur -st_type sinvert -eps_target 90 -eps_monitor_conv
#
# ----------------------------------------------------------------------

import sys, petsc4py
petsc4py.init(sys.argv)                # let PETSc parse -eps_* CLI flags first

from mpi4py import MPI
rank = MPI.COMM_WORLD.rank

import numpy as np, ufl
from petsc4py import PETSc
from slepc4py import SLEPc

# ----------------------------------------------------------------------
# 1.  Gmsh: unit‑disk mesh (radius 1) ----------------------------------
# ----------------------------------------------------------------------
import gmsh, dolfinx
from dolfinx import mesh as dmesh
from dolfinx.io import gmshio
from dolfinx.fem import (
    functionspace, locate_dofs_topological, dirichletbc, Constant, form,
)
from dolfinx.fem.petsc import assemble_matrix

hmax = 0.05               # element size   (≈ #dofs ~ O(hmax⁻²))
delta = 0.01              # coefficient parameter from Eq.(5)
def vec_numpy(v: PETSc.Vec) -> np.ndarray:
    """Return *read‑write* NumPy view of a PETSc Vec — works on every version."""
    for attr in ("array_real", "array_r", "array"):     # newest → oldest
        if hasattr(v, attr):
            return getattr(v, attr)
    return v.getArray()  
def fun_vec(u_fun: dolfinx.fem.Function)-> PETSc.Vec:
    """Return the PETSc Vec that stores a dolfinx Function’s coefficients."""
    return u_fun.vector if hasattr(u_fun, "vector") else u_fun.x
if rank == 0:
    gmsh.initialize([])
    gmsh.model.add("unit_disk")

    disk = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)   # centre (0,0), r=1
    gmsh.model.occ.synchronize()

    # physical groups (2 = surface, 1 = boundary curves)
    gmsh.model.addPhysicalGroup(2, [disk], tag=2)
    curves = [e[1] for e in gmsh.model.getEntities(1)]
    gmsh.model.addPhysicalGroup(1, curves, tag=1)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
    gmsh.model.mesh.generate(2)

domain, cell_tags, facet_tags = gmshio.model_to_mesh(
    gmsh.model if rank == 0 else None,
    MPI.COMM_WORLD, 0, gdim=2)

if rank == 0:
    gmsh.finalize()

# ----------------------------------------------------------------------
# 2.  FE space (continuous P1) -----------------------------------------
# ----------------------------------------------------------------------
V = functionspace(domain, ("Lagrange", 1))

# ----------------------------------------------------------------------
# 3.  Dirichlet wall BC on Γ (boundary tag = 1) ------------------------
# ----------------------------------------------------------------------
wall_facets = facet_tags.indices[np.where(facet_tags.values == 1)]
wall_dofs   = locate_dofs_topological(V, 1, wall_facets)
bc = dirichletbc(Constant(domain, PETSc.ScalarType(0.0)), wall_dofs, V)

# ----------------------------------------------------------------------
# 4.  k²(x,y) profile  (Eq. 5) -----------------------------------------
# ----------------------------------------------------------------------
x  = ufl.SpatialCoordinate(domain)
r2 = x[0]**2 + x[1]**2
k2 = (100.0 + delta) * ufl.exp(-50.0 * r2) - 100.0

# ----------------------------------------------------------------------
# 5. forms ----------------------------------------------------
# ----------------------------------------------------------------------
u, v  = ufl.TrialFunction(V), ufl.TestFunction(V)
a_form = form( ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx - k2*u*v*ufl.dx )
m_form = form( u*v*ufl.dx )

# ----------------------------------------------------------------------
# 6.  Assemble global PETSc matrices -----------------------------------
# ----------------------------------------------------------------------
A = assemble_matrix(a_form, bcs=[bc]);  A.assemble()
M = assemble_matrix(m_form, bcs=[bc]);  M.assemble()

# ----------------------------------------------------------------------
# 7.  SLEPc eigen‑solver -----------------------------------------------
# ----------------------------------------------------------------------
eps = SLEPc.EPS().create(domain.comm)
eps.setOperators(A, M)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

nev = 8                       # ask for 8 modes (change as you like)
eps.setDimensions(nev=nev, ncv=PETSc.DECIDE)

# default is SMALLEST_REAL; user can override from CLI
eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)

# honour any -eps_* flags passed on command line
eps.setFromOptions()
eps.solve()


# ----------------------------------------------------------------------
# 8.  Report ------------------------------------------------------------
# ----------------------------------------------------------------------
nconv = eps.getConverged()
if rank == 0:
    if nconv == 0:
        print("No eigen‑pairs converged!")
    else:
        print(f"\n{nconv} eigen‑pairs converged (requested {nev}).")
        eigs = [eps.getEigenvalue(i).real for i in range(nconv)]
        for i, lam in enumerate(sorted(eigs), 1):
            print(f"  {i:2d}: λ = {lam:.10f}")
from dolfinx.io import XDMFFile

def coeff_vec(u):
    """Return the coefficient Vec of a Function (old `.x` vs new `.vector`)."""
    return u.vector if hasattr(u, "vector") else u.x

def sync_ghosts(vec):
    """Synchronise ghost entries (old `.scatter_forward` vs new `.ghostUpdate`)."""
    try:    # ≥ 0.7
        vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)
    except AttributeError:   # 0.5 – 0.6
        vec.scatter_forward()

# ------------------------------------------------------------------#
# export loop  (put right after eps.solve())                         #
# ------------------------------------------------------------------#
import numpy as np

# resolve version differences once
def coeff_vec(u):
    return u.vector if hasattr(u, "vector") else u.x

for i in range(nconv):
    Vr, _ = A.getVecs()                  # SLEPc returns owned dofs only
    eps.getEigenvector(i, Vr, None)

    u_fun = dolfinx.fem.Function(V)
    coeff = coeff_vec(u_fun)

    # copy the owned part  (no ghosts in Vr)
    n_owned = Vr.getLocalSize()
    coeff.array[:n_owned] = Vr.getArray()

    # make ghosts consistent for rank-0 gather (API-agnostic)
    try:
        coeff.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
    except AttributeError:
        coeff.scatter_forward()

    # Gather coordinates and values on rank-0
    coords2d = domain.geometry.x[:, :2]  
    xyz_loc    = np.c_[coords2d, coeff.array]  # (N_loc, 3)

    xyz_all = domain.comm.gather(xyz_loc, root=0)

    if domain.comm.rank == 0:
        xyz = np.vstack(xyz_all)
        np.savetxt(f"mode{i}.dat", xyz, fmt="%.15e")
        print(f"  ↳ wrote mode{i}.dat  ({xyz.shape[0]} points)")
