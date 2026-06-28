"""
periodic_bc.py
Creates periodic boundary conditions with Bloch phase
"""

import numpy as np
import dolfinx_mpc


class PeriodicBCManager:
    """
    Manages Bloch-Floquet periodic boundary conditions.
    """
    
    def __init__(self, mesh, periodic_direction='z'):
        self.mesh = mesh
        self.direction = periodic_direction
        self.dir_map = {'x': 0, 'y': 1, 'z': 2}
        self.idx = self.dir_map[periodic_direction]
        
        coords = mesh.geometry.x
        self.L = coords[:, self.idx].max() - coords[:, self.idx].min()
        self.basis = np.zeros(3)
        self.basis[self.idx] = self.L
        
        print(f"[PeriodicBCManager] Initialized")
        print(f"  Direction: {periodic_direction}")
        print(f"  Periodic length L = {self.L:.6f} m")
    
    def indicator(self, x, tol=1e-8):
        """Points on boundary 1 (locate max side)"""
        x_max = self.mesh.geometry.x[:, self.idx].max()
        return np.isclose(x[self.idx], x_max, atol=tol)
    
    def relation(self, x, tol=1e-8):
        """Map boundary 1 to boundary 2"""
        out = x.copy()
        out[self.idx] = x[self.idx] - self.basis[self.idx]
        return out
    
    def create_mpc(self, V, phase=1.0):
        """
        Create MultiPointConstraint with given phase.
        """
        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_periodic_constraint_geometrical(
            V,
            indicator=self.indicator,
            relation=self.relation,
            bcs=[],
            scale=phase
        )
        mpc.finalize()
        return mpc