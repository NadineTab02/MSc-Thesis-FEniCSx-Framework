"""
material_assigner.py
Assigns material properties to mesh cells
"""

import numpy as np
from dolfinx import fem
from dolfinx import default_scalar_type


class MaterialAssigner:
    """
    Assigns E, rho, nu, mu, lambda to cells based on tags.
    """
    
    def __init__(self, mesh, cell_tags):
        self.mesh = mesh
        self.cell_tags = cell_tags
        self.Q = fem.functionspace(mesh, ("DG", 0))
        print("[MaterialAssigner] Initialized")
    
    def assign(self, material_properties):
        """
        Assign material properties to cells.
        """
        print("\n" + "="*50)
        print("MATERIAL ASSIGNMENT")
        print("-"*50)
        
        # Get cell tag values
        if hasattr(self.cell_tags, 'values'):
            cell_values = self.cell_tags.values
        else:
            cell_values = self.cell_tags
        
        # Create functions
        print("[Step 1/3] Creating material function spaces...")
        E_func = fem.Function(self.Q)
        rho_func = fem.Function(self.Q)
        nu_func = fem.Function(self.Q)
        mu_func = fem.Function(self.Q)
        lambda_func = fem.Function(self.Q)
        
        total_cells = len(cell_values)
        print(f"  Total cells in mesh: {total_cells}")
        
        # Check if single material (dict without tags as keys)
        if 'E' in material_properties:
            # Single material
            print("\n[Step 2/3] Single material mode detected")
            E_val = material_properties['E']
            rho_val = material_properties['rho']
            nu_val = material_properties['nu']
            name = material_properties.get('name', 'unknown')
            
            mu_val = E_val / (2 * (1 + nu_val))
            lambda_val = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
            
            print(f"  Material: {name}")
            print(f"    E = {E_val:.2e} Pa")
            print(f"    ρ = {rho_val} kg/m³")
            print(f"    ν = {nu_val}")
            print(f"    μ = {mu_val:.2e} Pa")
            print(f"    λ = {lambda_val:.2e} Pa")
            
            print(f"\n[Step 3/3] Assigning to all {total_cells} cells...")
            E_func.x.array[:] = np.full(total_cells, E_val, dtype=default_scalar_type)
            rho_func.x.array[:] = np.full(total_cells, rho_val, dtype=default_scalar_type)
            nu_func.x.array[:] = np.full(total_cells, nu_val, dtype=default_scalar_type)
            mu_func.x.array[:] = np.full(total_cells, mu_val, dtype=default_scalar_type)
            lambda_func.x.array[:] = np.full(total_cells, lambda_val, dtype=default_scalar_type)
            
            print(f"  ✓ Assigned {name} to all {total_cells} cells")
        
        else:
            # Multi-material
            print("\n[Step 2/3] Multi-material mode detected")
            print(f"  Found {len(material_properties)} material(s) to assign")
            
            # Initialize with zeros
            E_func.x.array[:] = np.zeros(total_cells, dtype=default_scalar_type)
            rho_func.x.array[:] = np.zeros(total_cells, dtype=default_scalar_type)
            nu_func.x.array[:] = np.zeros(total_cells, dtype=default_scalar_type)
            mu_func.x.array[:] = np.zeros(total_cells, dtype=default_scalar_type)
            lambda_func.x.array[:] = np.zeros(total_cells, dtype=default_scalar_type)
            
            print("\n[Step 3/3] Assigning materials by tag:")
            for tag, props in material_properties.items():
                cells = np.where(cell_values == tag)[0]
                
                if len(cells) > 0:
                    E_val = props['E']
                    rho_val = props['rho']
                    nu_val = props['nu']
                    name = props.get('name', f'tag_{tag}')
                    
                    mu_val = E_val / (2 * (1 + nu_val))
                    lambda_val = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
                    
                    E_func.x.array[cells] = np.full_like(cells, E_val, dtype=default_scalar_type)
                    rho_func.x.array[cells] = np.full_like(cells, rho_val, dtype=default_scalar_type)
                    nu_func.x.array[cells] = np.full_like(cells, nu_val, dtype=default_scalar_type)
                    mu_func.x.array[cells] = np.full_like(cells, mu_val, dtype=default_scalar_type)
                    lambda_func.x.array[cells] = np.full_like(cells, lambda_val, dtype=default_scalar_type)
                    
                    print(f"    Tag {tag} ({name}): {len(cells)} cells")
                    print(f"      E={E_val:.2e}, ρ={rho_val}, ν={nu_val}")
                else:
                    print(f"     Tag {tag}: No cells found!")
            
            # Check for unassigned cells
            unassigned = np.sum(E_func.x.array == 0)
            if unassigned > 0:
                print(f"\n   Warning: {unassigned} cells have no material assigned!")
            else:
                print(f"\n  ✓ All {total_cells} cells assigned successfully")
        
        print("\n" + "-"*40)
        print("MATERIAL FUNCTION SUMMARY")
        print("-"*40)
        print(f"  E:   min={E_func.x.array.min():.2e}, max={E_func.x.array.max():.2e}")
        print(f"  ρ:   min={rho_func.x.array.min():.2e}, max={rho_func.x.array.max():.2e}")
        print(f"  ν:   min={nu_func.x.array.min():.4f}, max={nu_func.x.array.max():.4f}")
        print(f"  μ:   min={mu_func.x.array.min():.2e}, max={mu_func.x.array.max():.2e}")
        print(f"  λ:   min={lambda_func.x.array.min():.2e}, max={lambda_func.x.array.max():.2e}")
        print("-"*50 + "\n")
        
        return E_func, rho_func, nu_func, mu_func, lambda_func