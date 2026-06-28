# MSc-Thesis-FEniCSx-Framework

A modular FEniCSx-based open-source framework for obtaining dispersion curves of periodic elastic structures. 

## Overview

This framework provides tools for analyzing wave propagation in periodic elastic media:
- **Forward problem**: Compute ω(k) dispersion relations using SLEPc EPS
- **Inverse problem**: Compute complex k(ω) using SLEPc PEP (quadratic eigenvalue problem)
- **Periodic BCs**: Bloch-Floquet boundary conditions
- **Multi-material support**: Complex geometries with multiple materials
- **Mode analysis**: Polarization and curl decomposition
- **Interactive visualization**: Click on dispersion curves to view mode shapes

## Files

Class files:
- base_problem.py : Base class for all problems
- forward_solver.py : Forward solver (ω(k))
- inverse_solver.py : Inverse solver (k(ω))
- mesh_generator.py : Mesh creation from STEP files
- material_assigner.py : Material assignment
- periodic_bc.py : Periodic boundary conditions
- plotter.py : Plotting utilities

Runs:
- run_forward.py : Run forward problem
- run_inverse.py : Run inverse problem
- plot_forward.py : Plot forward results
- plot_inverse.py : Plot inverse results

## Usage

1. Place your geometry as `.step` file
2. Configure `run_forward.py` / `run_inverse.py` along with mesh_generator with your settings
3. Run: `run_forward.py` / `run_inverse.py`
4. Plot: `plot_forward.py` / `plot_inverse.py`

## Dependencies

- FEniCSx (complex mode)
- SLEPc
- dolfinx_mpc
- gmsh
- numpy, scipy
- matplotlib, pyvista
