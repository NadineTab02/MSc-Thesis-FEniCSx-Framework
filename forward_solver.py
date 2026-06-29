"""
forward_solver.py
Solves ω(k) dispersion problem using SLEPc EPS
"""

import numpy as np
import time
from slepc4py import SLEPc
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.fem import form, Function
from dolfinx import fem
import dolfinx_mpc
import ufl

from .base_problem import BaseProblem
from .mesh_generator import MeshGenerator
from .material_assigner import MaterialAssigner
from .periodic_bc import PeriodicBCManager


class ForwardSolver(BaseProblem):
    """Solver for forward problem"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.results = None
        print("\n[ForwardSolver] Initialized")
        print(f"  Step file: {config['step_filename']}")
        print(f"  Mesh name: {config['mesh_name']}")
        print(f"  Mesh size lc: {config['lc']}")
        print(f"  Periodic direction: {config['periodic_direction']}")
        print(f"  Number of k values: {len(config['k_values'])}")
        print(f"  NEV (eigenvalues per k): {config.get('nev', 20)}")
    
    def solve(self):
        """Run the forward solver for all k values"""
        
        print("\n" + "="*70)
        print("FORWARD SOLVER STARTED (ω(k) problem)")
        print("="*70)
        
        # ------------- STEP 1: CREATE MESH --------------
        print("\n[Step 1/6] Creating mesh...")
        generator = MeshGenerator()
        
        mesh, cell_tags, facet_tags = generator.from_step_file(
            self.config['step_filename'],
            self.config['mesh_name'],
            self.config['lc'],
            periodic_direction=self.config['periodic_direction']
        )
        dim = mesh.geometry.dim
        
        # Get geometry properties
        L_f, A, I, radius = generator.get_geometry_properties(
            mesh, facet_tags, 
            periodic_direction=self.config['periodic_direction']
        )
        print(f"  Periodic length L_f = {L_f:.6f} m")
        # ------------ STEP 2: CREATE FUNCTION SPACE ------------
        print("\n[Step 2/6] Creating function space...")
        from basix.ufl import element
        Ve = element("Lagrange", mesh.basix_cell(), 2, shape=(dim,))
        V = fem.functionspace(mesh, Ve)
        print(f"  Function space: Lagrange degree 2, dimension {dim}")
        print(f"  DOFs: {V.dofmap.index_map.size_local * V.dofmap.index_map_bs}")
        
        u_trial = ufl.TrialFunction(V)
        v_test = ufl.TestFunction(V)
        dx = ufl.Measure("dx", domain=mesh)
        
        # ------------ STEP 3: ASSIGN MATERIALS ------------
        print("\n[Step 3/6] Assigning materials...")
        assigner = MaterialAssigner(mesh, cell_tags)
        E, rho, nu, mu, lambda_ = assigner.assign(self.config['material_properties'])
        
        # ------------ STEP 4: DEFINE WEAK FORMS ------------
        print("\n[Step 4/6] Defining weak forms...")
        
        def epsilon(u):
            return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        
        def sigma(u):
            return lambda_ * ufl.nabla_div(u) * ufl.Identity(dim) + 2 * mu * epsilon(u)
        
        a = ufl.inner(sigma(u_trial), epsilon(v_test)) * dx
        m = rho * ufl.inner(u_trial, v_test) * dx
        print("  ✓ Stiffness form (a) and mass form (m) defined")
        
        # ------------ STEP 5: PERIODIC BC MANAGER ------------
        print("\n[Step 5/6] Setting up periodic boundary conditions...")
        pbc = PeriodicBCManager(mesh, self.config['periodic_direction'])
        
        # ------------ STEP 6: LOOP OVER k VALUES ------------
        k_values = self.config['k_values']
        if self.config.get('k_values_in_pi_over_a', False):
            k_values = k_values * np.pi / L_f
            print(f"\n  Converted k values from π/a to 1/m using L_f = {L_f:.6f} m")
        
        num_eigval = self.config.get('nev', 20)
        
        print(f"\n[Step 6/6] Solving for {len(k_values)} k values...")
        print(f"  Requesting {num_eigval} eigenvalues per k")
        print("-"*50)
        
        # Storage for results
        frequencies = []
        all_polarizations_x = []
        all_polarizations_y = []
        all_polarizations_z = []
        all_curls_x = []
        all_curls_y = []
        all_curls_z = []
        all_mode_shapes = []
        
        for idx, k in enumerate(k_values):
            k_start = time.time() 
            print(f"\n  k = {k:.6f} 1/m  ({idx+1}/{len(k_values)})")
            print(f"    Phase factor = exp(-i*k*L) = {np.exp(-1j * k * L_f):.4f}")
            
            phase = np.exp(-1j * k * L_f)
            mpc = pbc.create_mpc(V, phase=phase)
            
            print("    Assembling matrices...")
            K = dolfinx_mpc.assemble_matrix(form(a), mpc)
            M = dolfinx_mpc.assemble_matrix(form(m), mpc)
            K.assemble()
            M.assemble()
            print(f"      K size: {K.getSize()[0]} x {K.getSize()[1]}")
            print(f"      M size: {M.getSize()[0]} x {M.getSize()[1]}")
            
            print("    Extracting submatrices...")
            K_sub, idx_set = self.extract_submatrix(K, V, mpc=mpc)
            M_sub, _ = self.extract_submatrix(M, V, mpc=mpc)
            
            print("    Solving eigenvalue problem (EPS)...")
            eps = SLEPc.EPS().create(mesh.comm)
            eps.setOperators(K_sub, M_sub)
            eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
            eps.setTolerances(tol=1e-9)
            eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
            eps.setDimensions(nev=num_eigval)
            
            st = eps.getST()
            st.setType(SLEPc.ST.Type.SINVERT)
            
            eps.solve()
            nconv = eps.getConverged()
            print(f"    Converged: {nconv} eigenvalues")
            
            freqs_at_k = []
            pol_x_list, pol_y_list, pol_z_list = [], [], []
            curl_x_list, curl_y_list, curl_z_list = [], [], []
            modes_at_k = []
            
            for i in range(nconv):
                val = eps.getEigenvalue(i)
                if val.real <= 0:
                    continue
                
                omega = np.sqrt(val.real)
                f = omega / (2 * np.pi)
                freqs_at_k.append(f)
                
                vr = K_sub.createVecRight()
                eps.getEigenvector(i, vr)
                
                eh = Function(V)
                full_vec = eh.x.petsc_vec
                full_vec.set(0.0)
                full_vec.setValues(idx_set, vr.getArray())
                full_vec.assemble()
                
                mpc.backsubstitution(eh)  

                # Correct format to plot mode shapes later
                u_complex = eh.x.array.copy()
                u_complex_reshaped = u_complex.reshape(-1, dim)

                # Real and imaginary parts
                u_real = np.real(u_complex_reshaped)
                u_imag = np.imag(u_complex_reshaped)
                u_magnitude = np.abs(u_complex_reshaped)

                modes_at_k.append({
                    'mode_index': i,
                    'k_real': k,
                    'frequency': f,
                    'displacement_real': u_real,
                    'displacement_imag': u_imag,
                    'displacement_magnitude': u_magnitude,
                    'displacement_complex': u_complex_reshaped,
                })

                # Calculate polarization and curl for each mode
                u_expr = ufl.as_vector([eh[j] for j in range(dim)])
                pol_x, pol_y, pol_z = self.compute_polarization_components(u_expr, dx)
                curl_x, curl_y, curl_z = self.compute_curl_components(u_expr, dx)
                
                pol_x_list.append(pol_x)
                pol_y_list.append(pol_y)
                pol_z_list.append(pol_z)
                curl_x_list.append(curl_x)
                curl_y_list.append(curl_y)
                curl_z_list.append(curl_z)
                
                print(f"      Mode {i}: f = {f:.2f} Hz")
            
            frequencies.append(freqs_at_k)
            all_polarizations_x.append(pol_x_list)
            all_polarizations_y.append(pol_y_list)
            all_polarizations_z.append(pol_z_list)
            all_curls_x.append(curl_x_list)
            all_curls_y.append(curl_y_list)
            all_curls_z.append(curl_z_list)
            all_mode_shapes.append(modes_at_k)
            
            print(f"    Found {len(freqs_at_k)} modes for this k")
            k_elapsed = time.time() - k_start  
            print(f"    Time for this k: {k_elapsed:.2f} seconds")  
        
        # For plotting: scale k to π/a units
        k_scaled_for_plotting = k_values * L_f / np.pi
        
        self.results = {
            'k_values': k_values,
            'k_scaled': k_scaled_for_plotting,
            'frequencies': frequencies,
            'polarizations': [all_polarizations_x, all_polarizations_y, all_polarizations_z],
            'curls': [all_curls_x, all_curls_y, all_curls_z],
            'mode_shapes': all_mode_shapes,
            'L_periodic': L_f,
            'periodic_direction': self.config['periodic_direction']
        }
        
        print("\n" + "-"*70)
        print("FORWARD SOLVER COMPLETED")
        print("-"*70)
        total_modes = sum(len(f) for f in frequencies)
        print(f"  Total modes found: {total_modes}")
        print(f"  Results stored with keys: {list(self.results.keys())}")
        print("-"*70 + "\n")
        
        return self.results
