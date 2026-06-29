"""
Run the inverse plotter
Plots dispersion curves and visualizes mode shapes from saved inverse problem results
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.plotter import InversePlotter

# ------------ PATHS ------------

SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')      # Folder with saved results
PLOTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')       # Folder where plots will be saved
STEP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Step_file')    # Folder with STEP files

# Ensure plots directory exists
os.makedirs(PLOTS_PATH, exist_ok=True)


# ------------ CONFIGURATION ------------

config = {
    # Mesh configuration
    'step_filename': os.path.join(STEP_PATH, 'Paper_Iorio_rotonlike_dispersion_relation'),  # Name of your STEP-file
    'mesh_name': 'mesh_paper_Iori_1cell',                                                    # Name of the created mesh
    'lc': 1,                                                                                # Maximum characteristic mesh length in [mm]
    'periodic_direction': 'x',                                                              # Set periodicity direction: 'x', 'y', or 'z'
    
    # Result file to load
    'inverse_filename': 'fenicsx_dump_inverse_iori_1_cell_original_mesh1.0.pkl',            # Inverse results file
   
    # Indicator selection
    # 0 = Polarization in x    3 = Curl around x
    # 1 = Polarization in y    4 = Curl around y
    # 2 = Polarization in z    5 = Curl around z
    # None = No colorbar (plain scatter plot)
    'indicator_index': None,                                                                # Choose which indicator to highlight
    
    # Axis limits
    'xlim_real': (0.001, 1),                                                                # X-axis limits for Re(k) 
    'xlim_imag': (0, 1),                                                                    # X-axis limits for Im(k)
    'ylim': (0, 20000),                                                                     # Frequency range [Hz]
    
    # Interactive mode shape visualization
    'enable_interactive_picker': True,                                                      # Set to False to skip interactive mode
    'visual_scale': 0.05,                                                                   # Scaling factor for displacement visualization, adjust for each geometry
}


# ------------ PLOT ------------

print("\n" + "-"*60)
print("INVERSE PLOTTER")
print("-"*60)

# Start the plotter
plotter = InversePlotter(
    save_path=SAVE_PATH,
    plots_path=PLOTS_PATH,
    indicator_index=config['indicator_index']
)

# Print configuration
print(f"\nConfiguration:")
print(f"  STEP file: {config['step_filename']}")
print(f"  Periodicity direction: {config['periodic_direction']}")
print(f"  Indicator: {plotter._get_selected_display_name()} (index: {config['indicator_index']})")
print(f"  Re(k) limits: {config['xlim_real']}")
print(f"  Im(k) limits: {config['xlim_imag']}")
print(f"  f limits: {config['ylim']}")
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

# Load inverse problem results
print("\n" + "-"*60)
print("LOADING INVERSE DATA")
print("-"*60)

plotter.load_inverse_data(config['inverse_filename'])

# Plot Re(k) and Im(k) side by side
print("\n" + "-"*60)
print("PLOTTING DISPERSION CURVES")
print("-"*60)

plotter.plot_real_imag_side_by_side(
    xlim_real=config['xlim_real'],
    xlim_imag=config['xlim_imag'],
    ylim=config['ylim']
)

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
        xlim=config['xlim_real'], 
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