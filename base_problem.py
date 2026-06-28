"""
base_problem.py
Common functions used by both forward and inverse solvers
"""

import numpy as np
from petsc4py import PETSc
import ufl
from dolfinx.fem import assemble_scalar, form


class BaseProblem:
    """
    Base class with shared functions.
    """
    
    def __init__(self):
        print("[BaseProblem] Initialized base utilities")
    
    def extract_submatrix(self, A, V, mpc=None, bcs=None):
        """
        Extract submatrix for unconstrained DOFs.
        """
        print("  [extract_submatrix] Removing constrained DOFs...")
        
        num_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        num_ghosts = V.dofmap.index_map.num_ghosts * V.dofmap.index_map_bs
        total_dofs = num_dofs_local + num_ghosts
        
        print(f"    Total DOFs (local+ghost): {total_dofs}")
        print(f"    Local DOFs: {num_dofs_local}")
        
        if mpc is not None:
            slave_local = np.array(mpc.is_slave, dtype=np.int32)
            num_slave = np.sum(slave_local[:num_dofs_local])
            print(f"    MPC slave DOFs: {num_slave}")
            mask = np.ones(total_dofs, dtype=np.int32)
            mask[slave_local.astype(bool)] = 0
            idx_set = np.flatnonzero(mask).astype(np.int32)
            idx_set = idx_set[idx_set < num_dofs_local]
        elif bcs is not None:
            not_dirichlet = np.full(total_dofs, 1, dtype=np.int32)
            for bc in bcs:
                not_dirichlet[bc.dof_indices()[0]] = 0
            idx_set = np.flatnonzero(not_dirichlet).astype(np.int32)
            idx_set = idx_set[idx_set < num_dofs_local]
        else:
            raise ValueError("Either bcs or mpc must be provided")
        
        num_free = len(idx_set)
        print(f"    Free DOFs after removal: {num_free}")
        
        isx = PETSc.IS(A.getComm()).createGeneral(idx_set)
        lgm_row = A.getLGMap()[0].applyIS(isx)
        lgm_col = A.getLGMap()[1].applyIS(isx)
        
        A_sub = A.createSubMatrix(lgm_row, lgm_col)
        A_sub.assemble()
        
        print(f"     Submatrix size: {A_sub.getSize()[0]} x {A_sub.getSize()[1]}")
        
        return A_sub, idx_set
    
    def compute_polarization_components(self, u_expr, dx):
        """
        Compute polarization values (pol_x, pol_y, pol_z).
        """
        u_mag_squared = ufl.real(ufl.inner(u_expr, u_expr))
        u_x_squared = ufl.real(u_expr[0] * ufl.conj(u_expr[0]))
        u_y_squared = ufl.real(u_expr[1] * ufl.conj(u_expr[1]))
        u_z_squared = ufl.real(u_expr[2] * ufl.conj(u_expr[2]))
        
        total = np.real(assemble_scalar(form(u_mag_squared * dx)))
        x_energy = np.real(assemble_scalar(form(u_x_squared * dx)))
        y_energy = np.real(assemble_scalar(form(u_y_squared * dx)))
        z_energy = np.real(assemble_scalar(form(u_z_squared * dx)))
        
        eps = 1e-15
        if total > eps:
            return x_energy / total, y_energy / total, z_energy / total
        return 0.0, 0.0, 0.0
    
    def compute_curl_components(self, u_expr, dx):
        """
        Compute curl values (curl_x, curl_y, curl_z).
        """
        curl_u = ufl.curl(u_expr)
        
        curl_mag_squared = ufl.real(ufl.inner(curl_u, curl_u))
        curl_x_squared = ufl.real(curl_u[0] * ufl.conj(curl_u[0]))
        curl_y_squared = ufl.real(curl_u[1] * ufl.conj(curl_u[1]))
        curl_z_squared = ufl.real(curl_u[2] * ufl.conj(curl_u[2]))
        
        total_curl = np.real(assemble_scalar(form(curl_mag_squared * dx)))
        curl_x_energy = np.real(assemble_scalar(form(curl_x_squared * dx)))
        curl_y_energy = np.real(assemble_scalar(form(curl_y_squared * dx)))
        curl_z_energy = np.real(assemble_scalar(form(curl_z_squared * dx)))
        
        eps = 1e-15
        if total_curl > eps:
            return curl_x_energy / total_curl, curl_y_energy / total_curl, curl_z_energy / total_curl
        return 0.0, 0.0, 0.0