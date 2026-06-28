"""
Run the forward plotter
Plots dispersion curves and visualizes mode shapes from saved results
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.plotter import ForwardPlotter

# ------------ PATHS ------------

SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')      # Set the name of the folder with saved results
PLOTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')       # Set the name of the folder where plots will be saved
STEP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Step_file')    # Set the name of the folder with STEP files

# Ensure plots directory exists
os.makedirs(PLOTS_PATH, exist_ok=True)


# ------------ CONFIGURATION ------------

config = {
    # Mesh configuration
    'step_filename': os.path.join(STEP_PATH, 'Test_periodic_rod_geo_r025'),                 # Name of your STEP-file
    'mesh_name': 'mesh_rod',                                                                # Name of the created mesh
    'lc': 10,                                                                               # Maximum characteristic mesh length in [mm]
    'periodic_direction': 'z',                                                              # Set periodicity direction: 'x', 'y', or 'z'
    
    # Result files to load
    'fenicsx_filename': 'fenicsx_dump_cylinder_v10.pkl',                                    # FEniCSx results file
    'comsol_filename': 'shortcut_dump_polxyz_curlxyz.pkl',                                  # COMSOL results file
    
    # Indicator selection
    # 0 = Polarization in x    3 = Curl around x
    # 1 = Polarization in y    4 = Curl around y
    # 2 = Polarization in z    5 = Curl around z
    # None = No colorbar (plain scatter plot)
    'indicator_index': 2,                                                                   # Choose which indicator to highlight
    
    # Plot style
    'same_figure': False,                                                                   # True: COMSOL and FEniCSx on same figure; False: separate figures
    
    # Axis limits
    'xlim': (0, 1),                                                                         # k range [π/a]
    'ylim': (0, 70000),                                                                     # f range [Hz]
    
    # Interactive mode shape visualization
    'enable_interactive_picker': True,                                                      # Set to False to skip interactive mode
    'visual_scale': 0.5,                                                                    # Scaling factor for displacement visualization, adjust for each geometry
}


# ------------ PLOT ------------

print("\n" + "-"*60)
print("FORWARD PLOTTER")
print("-"*60)

# Initialize the plotter
plotter = ForwardPlotter(
    save_path=SAVE_PATH,
    plots_path=PLOTS_PATH,
    indicator_index=config['indicator_index']
)

# Print configuration
print(f"\nConfiguration:")
print(f"  STEP file: {config['step_filename']}")
print(f"  Periodicity direction: {config['periodic_direction']}")
print(f"  Indicator: {plotter._get_selected_display_name()} (index: {config['indicator_index']})")
print(f"  Same figure: {config['same_figure']}")
print(f"  X limits: {config['xlim']}")
print(f"  Y limits: {config['ylim']}")
print(f"  Interactive picker: {config['enable_interactive_picker']}")

# Create mesh from STEP file
print("\n" + "-"*60)
print("CREATING MESH")
print("-"*60)

plotter.create_mesh_from_step(
    step_filename=config['step_filename'],
    mesh_name=config['mesh_name'],
    lc=config['lc'],
    periodic_direction=config['periodic_direction']
)

# Load FEniCSx results
print("\n" + "-"*60)
print("LOADING FEniCSx DATA")
print("-"*60)

plotter.load_fenicsx_data(config['fenicsx_filename'])

# Load COMSOL results
print("\n" + "-"*60)
print("LOADING COMSOL DATA")
print("-"*60)

plotter.load_comsol_data(config['comsol_filename'])

# Plot dispersion curves
print("\n" + "-"*60)
print("PLOTTING DISPERSION CURVES")
print("-"*60)

if config['same_figure']:
    plotter.plot_comparison(xlim=config['xlim'], ylim=config['ylim'])
else:
    plotter.plot_separate(xlim=config['xlim'], ylim=config['ylim'])

# Interactive mode shape picker
if config['enable_interactive_picker']:
    print("\n" + "-"*60)
    print("INTERACTIVE MODE SHAPE PICKER")
    print("-"*60)
    print("\nInstructions:")
    print("  1. Click on points in the dispersion curve to select modes")
    print("  2. Close the figure window when done selecting")
    print("  3. Mode shapes will be displayed for all selected points")
    
    clicked_points = plotter.start_interactive_picker(
        xlim=config['xlim'], 
        ylim=config['ylim']
    )
    
    if clicked_points:
        print(f"\n{len(clicked_points)} points selected. Plotting mode shapes...")
        plotter.plot_mode_shapes(visual_scale=config['visual_scale'])
    else:
        print("\nNo points were selected.")

print("\n" + "-"*60)
print("PLOTTING COMPLETE")
print("-"*60)