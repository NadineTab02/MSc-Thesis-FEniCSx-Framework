"""
inverse_solver.py
Solves k(ω) quadratic eigenvalue problem using SLEPc PEP
"""

import numpy as np
from slepc4py import SLEPc
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.fem import form, Function
from dolfinx import fem
import dolfinx_mpc
import ufl
import time

from .base_problem import BaseProblem
from .mesh_generator import MeshGenerator
from .material_assigner import MaterialAssigner
from .periodic_bc import PeriodicBCManager


class InverseSolver(BaseProblem):
    """Solver for inverse problem: find complex k(omega)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.results = None
        print("\n[InverseSolver] Initialized")
        print(f"  Step file: {config['step_filename']}")
        print(f"  Mesh name: {config['mesh_name']}")
        print(f"  Mesh size lc: {config['lc']}")
        print(f"  Periodic direction: {config['periodic_direction']}")
        print(f"  Frequency range: {config['frequencies'][0]:.1f} - {config['frequencies'][-1]:.1f} Hz")
        print(f"  Number of frequencies: {len(config['frequencies'])}")
        print(f"  NEV (eigenvalues per frequency): {config.get('nev', 30)}")
    
    def solve(self):
        """Run the inverse solver for all frequencies"""
        
        print("\n" + "-"*70)
        print("INVERSE SOLVER STARTED (k(ω) problem)")
        print("-"*70)
        
        # ------------ STEP 1: CREATE MESH ------------
        print("\n[Step 1/6] Creating mesh...")
        generator = MeshGenerator()
        
        mesh, cell_tags, facet_tags = generator.from_step_file(
            self.config['step_filename'],
            self.config['mesh_name'],
            self.config['lc'],
            periodic_direction=self.config['periodic_direction']
        )
        dim = mesh.geometry.dim
        
        # Get periodic length from mesh generator 
        L_f = generator.L_periodic
        print(f"  Periodic length L_f = {L_f:.6f} m")
        
        # ------------ STEP 2: CREATE FUNCTION SPACE ------------
        print("\n[Step 2/6] Creating function space...")
        from basix.ufl import element
        Ve = element("Lagrange", mesh.basix_cell(), 2, shape=(dim,))
        V = fem.functionspace(mesh, Ve)
        print(f"  Function space: Lagrange degree 2, dimension {dim}")
        
        u_trial = ufl.TrialFunction(V)
        v_test = ufl.TestFunction(V)
        dx = ufl.Measure("dx", domain=mesh)
        
        # ------------ STEP 3: ASSIGN MATERIALS ------------
        print("\n[Step 3/6] Assigning materials...")
        assigner = MaterialAssigner(mesh, cell_tags)
        E, rho, nu, mu, lambda_ = assigner.assign(self.config['material_properties'])
        
        # ------------ STEP 4: DEFINE WEAK FORMS ------------
        print("\n[Step 4/6] Defining weak forms for quadratic eigenvalue problem...")
        
        direction = self.config['periodic_direction']
        direction_map = {'x': 0, 'y': 1, 'z': 2}
        d_idx = direction_map[direction]
        e_d = ufl.as_vector([float(i == d_idx) for i in range(dim)])
        
        def epsilon(u):
            return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        
        def kappa(u):
            i, j = ufl.indices(2)
            Aij = u[i] * e_d[j] + e_d[i] * u[j]
            A = ufl.as_tensor(Aij, (i, j))
            return 0.5 * A
        
        # Fourth-order elasticity tensor
        Id = ufl.Identity(dim)
        indices = ufl.indices(4)
        
        def delta_product(i, j, k, l):
            return ufl.as_tensor(Id[i, j] * Id[k, l], indices)
        
        i, j, k, l = indices
        C = lambda_ * delta_product(i, j, k, l) + mu * (delta_product(i, k, j, l) + delta_product(i, l, k, j))
        
        # Four matrices 
        a_K = C[i, j, k, l] * ufl.conj(epsilon(v_test)[i, j]) * epsilon(u_trial)[k, l] * dx
        a_linear = (-C[i, j, k, l] * ufl.conj(kappa(v_test)[i, j]) * epsilon(u_trial)[k, l] + 
                     C[i, j, k, l] * ufl.conj(epsilon(v_test)[i, j]) * kappa(u_trial)[k, l]) * dx
        a_quadratic = C[i, j, k, l] * ufl.conj(kappa(v_test)[i, j]) * kappa(u_trial)[k, l] * dx
        m_form = rho * ufl.conj(v_test[i]) * u_trial[i] * dx
        
        print("  ✓ Forms defined: K0 (a_K), K1 (a_linear), K2 (a_quadratic), M (m_form)")
        
        # ------------ STEP 5: ASSEMBLE BASE MATRICES (phase=1) ------------
        print("\n[Step 5/6] Assembling base matrices with phase=1...")
        pbc = PeriodicBCManager(mesh, direction)
        mpc_base = pbc.create_mpc(V, phase=1.0)
        
        print("  Assembling K0...")
        K0 = dolfinx_mpc.assemble_matrix(form(a_K), mpc_base)
        K0.assemble()
        
        print("  Assembling K1...")
        K1 = dolfinx_mpc.assemble_matrix(form(a_linear), mpc_base)
        K1.assemble()
        
        print("  Assembling K2...")
        K2 = dolfinx_mpc.assemble_matrix(form(a_quadratic), mpc_base)
        K2.assemble()
        
        print("  Assembling M...")
        M = dolfinx_mpc.assemble_matrix(form(m_form), mpc_base)
        M.assemble()
        
        print("  Extracting submatrices...")
        K0_sub, idx_set = self.extract_submatrix(K0, V, mpc=mpc_base)
        K1_sub, _ = self.extract_submatrix(K1, V, mpc=mpc_base)
        K2_sub, _ = self.extract_submatrix(K2, V, mpc=mpc_base)
        M_sub, _ = self.extract_submatrix(M, V, mpc=mpc_base)
        
        # Prepare K2 for PEP 
        K2_pep = K2_sub.copy()
        K2_pep.scale(-1.0)
        print("  ✓ Base matrices assembled and reduced")
        
        # ------------ STEP 6: LOOP OVER FREQUENCIES ------------
        tol = 1e-9
        frequencies = self.config['frequencies']
        num_k = self.config.get('nev', 30)
        
        print(f"\n[Step 6/6] Solving for {len(frequencies)} frequencies...")
        print(f"  Requesting {num_k} eigenvalues per frequency")
        print("-"*50)
        
        # Storage for results
        dispersion_k_real = []
        dispersion_k_imag = []
        all_polarizations_x = []
        all_polarizations_y = []
        all_polarizations_z = []
        all_curls_x = []
        all_curls_y = []
        all_curls_z = []
        all_mode_shapes = []
        
        for idx, f in enumerate(frequencies):
            f_start = time.time()
            print(f"\n  f = {f:.1f} Hz  ({idx+1}/{len(frequencies)})")
            
            omega = 2 * np.pi * f
            omega_sq = omega**2
            
            # Build A0 = K0 - ω²M
            A0 = K0_sub.copy()
            A0.axpy(-omega_sq, M_sub)
            A0.assemble()
            
            print("    Setting up PEP solver...")
            pep = SLEPc.PEP().create(mesh.comm)
            pep.setOperators([A0, K1_sub, K2_pep])
            pep.setProblemType(SLEPc.PEP.ProblemType.GYROSCOPIC)
            pep.setDimensions(nev=num_k)
            

            pep.setExtract(SLEPc.PEP.Extract.RESIDUAL)  
            target = 1j * 1.0 
            pep.setTarget(target)
            pep.setWhichEigenpairs(SLEPc.PEP.Which.TARGET_MAGNITUDE)
            
            st = pep.getST()
            st.setType(SLEPc.ST.Type.SINVERT)
            ksp = st.getKSP()
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("lu")
            pc.setFactorSolverType("mumps")
            
            print("    Solving quadratic eigenvalue problem (PEP)...")
            pep.solve()
            nconv = pep.getConverged()
            print(f"    Converged: {nconv} eigenvalues")
            
            k_real_list = []
            k_imag_list = []
            pol_x_list, pol_y_list, pol_z_list = [], [], []
            curl_x_list, curl_y_list, curl_z_list = [], [], []
            modes_at_f = []
            
            for i in range(nconv):
                vr = A0.createVecRight()
                Lambda = pep.getEigenpair(i, vr)             
            
                k_real = Lambda.imag
                k_imag = -Lambda.real
                k_complex = -Lambda * 1j
                k_real_list.append(k_real)
                k_imag_list.append(k_imag)
                
                # Reconstruct full eigenvector
                eh = Function(V)
                full_vec = eh.x.petsc_vec
                full_vec.set(0.0)
                full_vec.setValues(idx_set, vr.getArray())
                full_vec.assemble()
                
                # Extract mode shapes
                mpc_base.backsubstitution(eh)   

                u_complex = eh.x.array.copy()
                u_complex_reshaped = u_complex.reshape(-1, dim)


                # Section below is an attempt to extract mode shapes in the correct way; should be modifed and tested

                coords = mesh.geometry.x
                z_coords = coords[:, 2]  # Periodic direction
               
                # Apply Bloch phase to each node
                u_full_bloch = u_complex_reshaped * np.exp(1j * k_complex * z_coords[:, None])
        
                u_real = np.real(u_full_bloch)      
                u_imag = np.imag(u_full_bloch)      
                u_magnitude = np.abs(u_full_bloch)  
                
        
                modes_at_f.append({
                    'mode_index': i,
                    'k_real': k_real,
                    'k_imag': k_imag,
                    'displacement_real': u_real,
                    'displacement_imag': u_imag,
                    'displacement_magnitude': u_magnitude,
                    'displacement_complex': u_complex_reshaped,
                })
                
                u_expr = ufl.as_vector([eh[j] for j in range(dim)])
                pol_x, pol_y, pol_z = self.compute_polarization_components(u_expr, dx)
                curl_x, curl_y, curl_z = self.compute_curl_components(u_expr, dx)
                
                pol_x_list.append(pol_x)
                pol_y_list.append(pol_y)
                pol_z_list.append(pol_z)
                curl_x_list.append(curl_x)
                curl_y_list.append(curl_y)
                curl_z_list.append(curl_z)
                
                print(f"      Mode {i}: k = {k_real:.6f} + {k_imag:.6f}i")
            
            dispersion_k_real.append(k_real_list)
            dispersion_k_imag.append(k_imag_list)
            all_polarizations_x.append(pol_x_list)
            all_polarizations_y.append(pol_y_list)
            all_polarizations_z.append(pol_z_list)
            all_curls_x.append(curl_x_list)
            all_curls_y.append(curl_y_list)
            all_curls_z.append(curl_z_list)
            all_mode_shapes.append(modes_at_f)
            
            print(f"    Found {len(k_real_list)} modes for this frequency")
            f_elapsed = time.time() - f_start  
            print(f"    Time for this frequency: {f_elapsed:.2f} seconds") 
        
        self.results = {
            'frequencies': frequencies,
            'k_real': dispersion_k_real,
            'k_imag': dispersion_k_imag,
            'polarizations': [all_polarizations_x, all_polarizations_y, all_polarizations_z],
            'curls': [all_curls_x, all_curls_y, all_curls_z],
            'mode_shapes': all_mode_shapes,
            'L_periodic': L_f,
            'periodic_direction': self.config['periodic_direction']
        }
        
        print("\n" + "-"*70)
        print("INVERSE SOLVER COMPLETED")
        print("-"*70)
        total_modes = sum(len(k) for k in dispersion_k_real)
        print(f"  Total modes found: {total_modes}")
        print(f"  Results stored with keys: {list(self.results.keys())}")
        print("-"*70 + "\n")
        
        return self.results