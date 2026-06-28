"""
mesh_generator.py
Creates mesh from STEP file - Surfaces and volumes need to be modified manually for each geometry 
"""

import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as msh
from dolfinx.io import gmsh as gmshio
from dolfinx.fem import assemble_scalar, form
import ufl


class MeshGenerator:
    """Creates FEniCSx mesh from STEP CAD file"""
    
    def __init__(self, comm=None, model_rank=0):
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.model_rank = model_rank
        self.L_periodic = None
        self.periodic_direction = None
        print("[MeshGenerator] Initialized")
    
    def from_step_file(self, input_file, output_file, lc, periodic_direction='z'):
        """
        Creates mesh from STEP file.
        """
        print("\n" + "-"*70)
        print("MESH GENERATION STARTED")
        print("="*70)
        print(f"[MeshGenerator] Input file: {input_file}.step")
        print(f"[MeshGenerator] Output name: {output_file}")
        print(f"[MeshGenerator] Mesh size (lc): {lc}")
        print(f"[MeshGenerator] Periodic direction: {periodic_direction}")
        
        self.periodic_direction = periodic_direction
        
        print("\n[Step 1/6] Initializing Gmsh...")
        gmsh.initialize()
        
        print("[Step 2/6] Importing STEP file...")
        model = gmsh.model()
        gmsh.model.add(output_file)
        gmsh.model.occ.importShapes(f"{input_file}.step")
        gmsh.model.occ.synchronize()
        print("  ✓ STEP file imported")
        
        print("[Step 3/6] Processing volumes...")
        volumes = gmsh.model.getEntities(dim=3)
        volume_tags = [v[1] for v in volumes]
        print(f"  Found {len(volumes)} volume(s) with tags: {volume_tags}")
        
        if len(volume_tags) > 1:
            print("  Fusing volumes to create shared interfaces...")
            gmsh.model.occ.fragment([(3, tag) for tag in volume_tags], [])
            gmsh.model.occ.synchronize()
            volumes = gmsh.model.getEntities(dim=3)
            print(f"  After fusion: {len(volumes)} volume(s)")
        
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(dim=3)
        
        print("[Step 4/6] Adding physical groups to sub-volumes...")
        # Material physical groups (Modify according to your own geometry)
        if len(volumes) == 1:
            print("  One volume detected - creating material group with tag 1")
            gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1)
        else:
            print("  Multi-volumes detected - creating physical groups for sub-volumes:")
            listvols1 = [1, 2, 3, 4, 5, 6, 7, 9] # 
            listvols2 = [8, 10] 
            print(f"    Group 1 - volumes: {listvols1} → tag 101") 
            gmsh.model.addPhysicalGroup(3, listvols1, tag=101)
            print(f"    Group 2 - volumes: {listvols2} → tag 201")
            gmsh.model.addPhysicalGroup(3, listvols2, tag=201)
        
        # Surface physical groups
        surfaces = gmsh.model.getEntities(dim=2) 
        print(f"  Found {len(surfaces)} surfaces")
        # Modify according to your own geometry
        listsurf1 = [4]      # left/start face 
        listsurf2 = [5]      # right/end face 
        listsurf3 = [1, 2, 3, 6]  # other surfaces 

        # Iorio geometry
        # listsurf1 = [1, 24, 44, 39, 29, 34] 
        # listsurf2 = [3, 23, 43, 38, 28, 33]    
        # listsurf3 = [x for x in range(2, 71) if x not in listsurf1 + listsurf2]
        
        print(f"    Left boundary (tag 1001) - surfaces: {listsurf1}")
        gmsh.model.addPhysicalGroup(2, listsurf1, tag=1001)
        print(f"    Right boundary (tag 2001) - surfaces: {listsurf2}")
        gmsh.model.addPhysicalGroup(2, listsurf2, tag=2001)
        print(f"    Other boundaries (tag 3001) - {len(listsurf3)} surfaces")
        gmsh.model.addPhysicalGroup(2, listsurf3, tag=3001)
        
        # Volume connectivity check
        if len(volumes) > 1:
            print("\n  Volume connectivity check:")
            for i, vol1 in enumerate(volumes):
                for vol2 in volumes[i+1:]:
                    vol1_tag = vol1[1]
                    vol2_tag = vol2[1]
                    surf1 = gmsh.model.getBoundary([(3, vol1_tag)], oriented=False)
                    surf2 = gmsh.model.getBoundary([(3, vol2_tag)], oriented=False)
                    surf1_tags = [s[1] for s in surf1]
                    surf2_tags = [s[1] for s in surf2]
                    shared = set(surf1_tags) & set(surf2_tags)
                    if shared:
                        print(f"    ✓ Volume {vol1_tag} and {vol2_tag} share surfaces: {shared}")
                    else:
                        print(f"     Volume {vol1_tag} and {vol2_tag} are NOT connected!")
        
        print("[Step 5/6] Generating mesh...")
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        
        gmsh.model.mesh.generate(3)
        gmsh.write(f"{output_file}.msh")
        print("  ✓ Mesh generated and saved")
        
        print("[Step 6/6] Converting to FEniCSx format...")
        mesh_data = gmshio.model_to_mesh(model, self.comm, self.model_rank)
        gmsh.finalize()
        
        mesh = mesh_data.mesh
        mesh.geometry.x[:] *= 0.001  # scale to meters
        print("  ✓ Converted to FEniCSx mesh")
        
        # Calculate L_f based on periodic_direction
        coords = mesh.geometry.x
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        z_coords = coords[:, 2]
        
        dir_to_idx = {'x': 0, 'y': 1, 'z': 2}
        idx = dir_to_idx[periodic_direction]
        
        if periodic_direction == 'x':
            coord_array = x_coords
            coord_name = 'X'
        elif periodic_direction == 'y':
            coord_array = y_coords
            coord_name = 'Y'
        else:
            coord_array = z_coords
            coord_name = 'Z'
        
        self.L_periodic = coord_array.max() - coord_array.min()
        
        print("\n" + "-"*50)
        print("MESH SUMMARY")
        print("-"*50)
        print(f"  Number of nodes: {mesh.geometry.x.shape[0]}")
        print(f"  Number of cells: {mesh.topology.index_map(mesh.topology.dim).size_local}")
        print(f"  Periodic direction: {periodic_direction.upper()}")
        print(f"  Periodic length L_f = {self.L_periodic:.6f} m")
        print(f"  {coord_name} range: [{coord_array.min():.6f}, {coord_array.max():.6f}] m")
        print(f"  X range: [{x_coords.min():.6f}, {x_coords.max():.6f}] m")
        print(f"  Y range: [{y_coords.min():.6f}, {y_coords.max():.6f}] m")
        print(f"  Z range: [{z_coords.min():.6f}, {z_coords.max():.6f}] m")
        print("-"*70 + "\n")
        
        return mesh, mesh_data.cell_tags, mesh_data.facet_tags
    
    def get_geometry_properties(self, mesh, facet_tags, periodic_direction='z'):
        """
        Calculate geometry properties (L_f, A, I, radius).
        """
        print("\n[Geometry Properties] Calculating...")
        
        coords = mesh.geometry.x
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        z_coords = coords[:, 2]
        
        dir_to_idx = {'x': 0, 'y': 1, 'z': 2}
        idx = dir_to_idx[periodic_direction]
        
        if periodic_direction == 'x':
            L_f = x_coords.max() - x_coords.min()
            dim1 = y_coords.max() - y_coords.min()
            dim2 = z_coords.max() - z_coords.min()
            print(f"  Periodic direction: X")
        elif periodic_direction == 'y':
            L_f = y_coords.max() - y_coords.min()
            dim1 = x_coords.max() - x_coords.min()
            dim2 = z_coords.max() - z_coords.min()
            print(f"  Periodic direction: Y")
        else:
            L_f = z_coords.max() - z_coords.min()
            dim1 = x_coords.max() - x_coords.min()
            dim2 = y_coords.max() - y_coords.min()
            print(f"  Periodic direction: Z")
        
        print(f"  Computing cross-sectional area using ds(1001)...")
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
        A = assemble_scalar(form(1 * ds(1001))) 
        
        radius = 0.5 * min(dim1, dim2) # Remove if not needed for your geometry
        I = np.pi * radius**4 / 4 # Area moment of inertia
        
        print("\n" + "-"*40)
        print("GEOMETRY PROPERTIES")
        print("-"*40)
        print(f"  Periodic length L_f = {L_f:.6f} m")
        print(f"  Cross-sectional area A = {A:.6e} m²")
        print(f"  Radius = {radius:.6f} m")
        print(f"  Area Moment of inertia I = {I:.6e} m⁴")
        print("-"*40)
        
        return L_f, A, I, radius