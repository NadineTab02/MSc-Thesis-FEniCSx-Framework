"""
plotter.py
Post-processing and visualization for forward and inverse dispersion problems
"""

import os
import pickle
import jpype
import numpy as np
import matplotlib.pyplot as plt

# Try relative import first (if in same package), then absolute import
try:
    from .mesh_generator import MeshGenerator
except ImportError:
    from mesh_generator import MeshGenerator

import pyvista as pv
from dolfinx import plot



class ForwardPlotter:
    """Class for plotting dispersion curves and mode shapes from forward problem results.
    
    Uses MeshGenerator class for mesh creation.
    """
    
    # Class constants for indicator mapping
    INDICATOR_NAMES = {
        0: 'pol_x',
        1: 'pol_y', 
        2: 'pol_z',
        3: 'curl_x',
        4: 'curl_y',
        5: 'curl_z'
    }
    
    INDICATOR_DISPLAY_NAMES = {
        0: 'Polarization X',
        1: 'Polarization Y',
        2: 'Polarization Z',
        3: 'Curl X',
        4: 'Curl Y',
        5: 'Curl Z'
    }
    
    def __init__(self, save_path, plots_path, indicator_index=1):
        """
        Initialize the ForwardPlotter.
        
        Parameters
        ----------
        save_path : str
            Path to the directory containing result files
        plots_path : str
            Path to the directory where plots will be saved
        indicator_index : int or None
            Index of indicator to highlight (0-5) or None for no colorbar
        """
        self.save_path = save_path
        self.plots_path = plots_path
        self.indicator_index = indicator_index
        
        # Initialize mesh generator (will be configured when creating mesh)
        self.mesh_generator = None
        self.mesh = None
        self.cell_tags = None
        self.facet_tags = None
        self.function_space = None
        self.L_f = None
        
        # FEniCSx data
        self.k_values_fenicsx = None
        self.dispersion_freqs_fenicsx = None
        self.eval_var_out_fenicsx = None
        self.all_modes_displacements = None
        self.fenicsx_basename = None
        self.fenicsx_indicators = None
        
        # COMSOL data
        self.k_values_comsol = None
        self.dispersion_freqs_comsol = None
        self.eval_var_out = None
        self.comsol_basename = None
        self.comsol_indicators = None
        
        # Interactive picker data
        self.clicked_points_forward = []
        self.points_data_forward = []
        
        # Setup
        self._setup_matplotlib()
        self._start_jvm()
        
    def _setup_matplotlib(self):
        """Configure matplotlib for LaTeX-style rendering."""
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "axes.labelsize": 14,
            "font.size": 13,
            "legend.fontsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        })
        
    def _start_jvm(self):
        """Start Java Virtual Machine if not already running."""
        if not jpype.isJVMStarted():
            jpype.startJVM()
    
    def create_mesh_from_step(self, step_filename, mesh_name='mesh_rod', lc=10, 
                             periodic_direction='x'):
        """
        Create mesh from STEP file using MeshGenerator class.
        
        Parameters
        ----------
        step_filename : str
            Path to the STEP file (without extension)
        mesh_name : str
            Name for the mesh
        lc : float
            Characteristic mesh element size
        periodic_direction : str
            Direction of periodicity: 'x', 'y', or 'z'
        """
        print("\n" + "-"*60)
        print("CREATING MESH USING MESHGENERATOR")
        print("-"*60)
        
        # Initialize and use MeshGenerator
        self.mesh_generator = MeshGenerator()
        self.mesh, self.cell_tags, self.facet_tags = self.mesh_generator.from_step_file(
            input_file=step_filename,
            output_file=mesh_name,
            lc=lc,
            periodic_direction=periodic_direction
        )
        
        # Get L_f from the mesh generator (already calculated with correct direction)
        self.L_f = self.mesh_generator.L_periodic
        
        print(f"\n[MeshGenerator] Periodic direction: {periodic_direction}")
        print(f"[MeshGenerator] Periodic length L_f = {self.L_f:.6f} m")
        
        return self.mesh, self.cell_tags, self.facet_tags
    
    def load_fenicsx_data(self, filename):
        """
        Load FEniCSx results from pickle file.
        
        Parameters
        ----------
        filename : str
            Name of the pickle file (without path)
        """
        full_path = os.path.join(self.save_path, filename)
        self.fenicsx_basename = os.path.splitext(filename)[0]
        
        print(f"\nLoading data from: {full_path}")
        
        with open(full_path, 'rb') as f:
            obj = pickle.load(f)
            k_name = obj[0]
            num_k = obj[1]
            self.k_values_fenicsx = obj[2]
            self.dispersion_freqs_fenicsx = obj[3]
            self.eval_var_out_fenicsx = obj[4]
            self.all_modes_displacements = obj[5]
        
        # Normalize k-values using the periodicity length from MeshGenerator
        self.k_values_fenicsx = self.k_values_fenicsx * self.L_f / np.pi
        
        print(f"Loaded data: {num_k} k-values")
        print(f"k-values normalized with L_f = {self.L_f:.4f} m")
        
        # Create indicators
        self._create_fenicsx_indicators()
        
    def _create_fenicsx_indicators(self):
        """Create indicator dictionaries for FEniCSx data."""
        self.fenicsx_indicators = [
            {'name': r'$p_x$', 'tag': 'pol_x', 'values': self.eval_var_out_fenicsx[0], 
             'unit': '', 'colormap': 'turbo', 'cmap_on': True, 'limits': (0, 1)},
            {'name': r'$p_y$', 'tag': 'pol_y', 'values': self.eval_var_out_fenicsx[1], 
             'unit': '', 'colormap': 'turbo', 'cmap_on': True, 'limits': (0, 1)},
            {'name': r'$p_z$', 'tag': 'pol_z', 'values': self.eval_var_out_fenicsx[2], 
             'unit': '', 'colormap': 'turbo', 'cmap_on': True, 'limits': (0, 1)},
            {'name': r'$p_x^\psi$', 'tag': 'curl_x', 'values': self.eval_var_out_fenicsx[3], 
             'unit': '', 'colormap': 'cool', 'cmap_on': True, 'limits': (0, 1)},
            {'name': r'$p_y^\psi$', 'tag': 'curl_y', 'values': self.eval_var_out_fenicsx[4], 
             'unit': '', 'colormap': 'cool', 'cmap_on': True, 'limits': (0, 1)},
            {'name': r'$p_z^\psi$', 'tag': 'curl_z', 'values': self.eval_var_out_fenicsx[5], 
             'unit': '', 'colormap': 'cool', 'cmap_on': True, 'limits': (0, 1)},
        ]
    
    
    def load_comsol_data(self, filename):
        """
        Load COMSOL results from pickle file.
        
        Parameters
        ----------
        filename : str
            Name of the pickle file (without path)
        """
        full_path = os.path.join(self.save_path, filename)
        self.comsol_basename = os.path.splitext(filename)[0]
        
        print('Loading data from COMSOL')
        
        with open(full_path, 'rb') as f:
            obj = pickle.load(f)
            k_name = obj[0]
            self.k_values_comsol = obj[1]
            k_unit = obj[2]
            self.dispersion_freqs_comsol = obj[3]
            self.eval_var_out = obj[4]
        
        # Create indicators
        self._create_comsol_indicators()
    
    def _create_comsol_indicators(self):
        """Create indicator dictionaries for COMSOL data."""
        self.comsol_indicators = [
            {'name': r'$p_x$', 'tag': 'pol_x', 'values': self.eval_var_out[0], 
             'unit': '', 'colormap': 'turbo', 'cmap_on': True, 'limits': (0, 1)},
            {'name': 'Polarization in y', 'tag': 'pol_y', 'values': self.eval_var_out[1], 
             'unit': '', 'colormap': 'turbo', 'cmap_on': True, 'limits': (0, 1)},
            {'name': r'$p_z$', 'tag': 'pol_z', 'values': self.eval_var_out[2], 
             'unit': '', 'colormap': 'turbo', 'cmap_on': True, 'limits': (0, 1)},
            {'name': r'$p_x ^{(\psi)}$', 'tag': 'curl_x', 'values': self.eval_var_out[3], 
             'unit': '', 'colormap': 'cool', 'cmap_on': True, 'limits': (0, 1)},
            {'name': 'Curl around y', 'tag': 'curl_y', 'values': self.eval_var_out[4], 
             'unit': '', 'colormap': 'cool', 'cmap_on': True, 'limits': (0, 1)},
            {'name': r'$p_z ^{(\psi)}$', 'tag': 'curl_z', 'values': self.eval_var_out[5], 
             'unit': '', 'colormap': 'cool', 'cmap_on': True, 'limits': (0, 1)},
        ]
    
    def _plot_dispersion_curve(self, wavenumber, frequency, indicators=None, 
                               indicator_index=None, xlim=None, ylim=None,
                               color='blue', marker='*', s=20, alpha=0.8, 
                               label='test'):
        """
        Plot a single dispersion curve with optional indicator coloring.
        
        Parameters
        ----------
        wavenumber : dict
            Dictionary with 'values' key containing k-values
        frequency : dict
            Dictionary with 'values' key containing frequency values
        indicators : list or None
            List of indicator dictionaries
        indicator_index : int or None
            Index of indicator to use for coloring
        xlim, ylim : tuple or None
            Axis limits
        color : str
            Color for markers (when no indicator)
        marker : str
            Marker style
        s : int
            Marker size
        alpha : float
            Transparency
        label : str
            Legend label
        """
        k_vals = wavenumber['values']
        f_vals = frequency['values']
        
        if indicators is not None and indicator_index is not None:
            indicator = indicators[indicator_index]
            
            # Collect all data
            all_k = []
            all_freqs = []
            all_vals = []
            
            for i, freqs in enumerate(f_vals):
                k = k_vals[i]
                indicator_vals = indicator['values'][i]
                
                all_k.extend([k] * len(freqs))
                all_freqs.extend(freqs)
                all_vals.extend(indicator_vals)
            
            # Force color mapping to use 0-1 range
            norm = plt.Normalize(vmin=0, vmax=1)
            
            # Single scatter plot with all data
            sc = plt.scatter(all_k, all_freqs, c=all_vals, 
                           cmap=indicator['colormap'], norm=norm,
                           s=s, alpha=alpha, label=label)
            
            cbar = plt.colorbar(sc, label=indicator['name'])
            cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            print('Scatterplot with indicators done!')
        else:
            # Plot without colormap
            for i, freqs in enumerate(f_vals):
                k = k_vals[i]
                current_label = label if i == 0 else None
                plt.scatter([k] * len(freqs), freqs, color=color, s=s, marker=marker, 
                          alpha=alpha, label=current_label)
            print('Scatterplot without indicators done!')
        
        # Set limits if provided
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        print('Limits set')
    
    def plot_comparison(self, xlim=(0, 1), ylim=(0, 70000)):
        """
        Plot FEniCSx and COMSOL results on the same figure for comparison.
        
        Parameters
        ----------
        xlim : tuple
            X-axis limits
        ylim : tuple
            Y-axis limits
        """
        print('Plotting data from FEniCSx and COMSOL in same figure')
        
        selected_indicator = self._get_selected_indicator_name()
        
        plt.figure(figsize=(12, 8))
        
        # Plot COMSOL data
        self._plot_dispersion_curve(
            {'values': self.k_values_comsol},
            {'values': self.dispersion_freqs_comsol},
            indicators=self.comsol_indicators,
            indicator_index=None,  # No indicator for comparison to avoid two colorbars
            ylim=ylim,
            xlim=xlim,
            color='blue',
            marker='o',
            s=40,
            alpha=0.7,
            label='COMSOL'
        )
        
        # Plot FEniCSx data
        self._plot_dispersion_curve(
            {'values': self.k_values_fenicsx},
            {'values': self.dispersion_freqs_fenicsx},
            indicators=self.fenicsx_indicators,
            indicator_index=None,  # No indicator for comparison
            ylim=ylim,
            xlim=xlim,
            color='red',
            marker='s',
            s=15,
            alpha=0.7,
            label='FEniCSx'
        )
        
        plt.xlabel(r'$k$ [$\pi$/a]')
        plt.ylabel(r'$f$ [Hz]')
        plt.grid(True)
        plt.legend()
        
        # Save comparison figure
        comparison_filename = f'{self.fenicsx_basename}_vs_{self.comsol_basename}_comparison_{selected_indicator}.png'
        plt.savefig(os.path.join(self.plots_path, comparison_filename), dpi=300, bbox_inches='tight')
        print(f'Comparison figure saved to: {os.path.join(self.plots_path, comparison_filename)}')
        plt.show()
    
    def plot_separate(self, xlim=(0, 1), ylim=(0, 70000)):
        """
        Plot FEniCSx and COMSOL results on separate figures.
        
        Parameters
        ----------
        xlim : tuple
            X-axis limits
        ylim : tuple
            Y-axis limits
        """
        print('Plotting data from FEniCSx and COMSOL in separate figures')
        selected_indicator = self._get_selected_indicator_name()
        
        # First figure - FEniCSx
        print('Plotting data from FEniCSx')
        plt.figure(2, figsize=(8, 6))
        
        self._plot_dispersion_curve(
            {'values': self.k_values_fenicsx},
            {'values': self.dispersion_freqs_fenicsx},
            indicators=self.fenicsx_indicators,
            indicator_index=self.indicator_index,
            ylim=ylim,
            xlim=xlim,
            color='red',
            marker='o',
            s=15,
            alpha=1,
            label='FEniCSx'
        )
        
        plt.xlabel(r'$k$ [$\pi$/a]')
        plt.ylabel(r'$f$ [Hz]')
        plt.grid(True)
        
        # Save FEniCSx figure
        fenicsx_save_path = os.path.join(self.plots_path, 
                                         f'{self.fenicsx_basename}_dispersion_{selected_indicator}.png')
        plt.savefig(fenicsx_save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved as {fenicsx_save_path}')
        
        # Second figure - COMSOL
        print('Plotting data from COMSOL')
        plt.figure(3, figsize=(8, 6))
        
        self._plot_dispersion_curve(
            {'values': self.k_values_comsol},
            {'values': self.dispersion_freqs_comsol},
            indicators=self.comsol_indicators,
            indicator_index=self.indicator_index,
            color='blue',
            marker='o',
            ylim=ylim,
            xlim=xlim,
            s=15,
            alpha=1,
            label='COMSOL'
        )
        
        plt.xlabel(r'$k$ [$\pi$/a]')
        plt.ylabel(r'$f$ [Hz]')
        plt.grid(True)
        
        # Save COMSOL figure
        comsol_save_path = os.path.join(self.plots_path, 
                                        f'{self.comsol_basename}_dispersion_{selected_indicator}.png')
        plt.savefig(comsol_save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved as {comsol_save_path}')
        plt.show()
    
    def _get_selected_indicator_name(self):
        """Get the name of the currently selected indicator."""
        if self.indicator_index is not None:
            return self.INDICATOR_NAMES[self.indicator_index]
        return 'none'
    
    def _get_selected_display_name(self):
        """Get the display name of the currently selected indicator."""
        if self.indicator_index is not None:
            return self.INDICATOR_DISPLAY_NAMES[self.indicator_index]
        return 'No Colorbar'
    
    def start_interactive_picker(self, xlim=(0, 1), ylim=(0, 70000)):
        """
        Start interactive picker for selecting points on dispersion curve.
        Click on points to store them for later mode shape visualization.
        
        Parameters
        ----------
        xlim : tuple
            X-axis limits
        ylim : tuple
            Y-axis limits
        """
        if self.mesh is None:
            raise ValueError("Mesh must be created before starting interactive picker. "
                           "Call create_mesh_from_step() first.")
        
        print("\n" + "-"*50)
        print("INTERACTIVE MODE SHAPE PICKER - FORWARD PROBLEM")
        print("-"*50)
        
        # Create interactive picker plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Collect data for forward problem
        all_k_forward = []
        all_f_forward = []
        all_vals_forward = []
        self.points_data_forward = []
        
        for i, k in enumerate(self.k_values_fenicsx):
            freqs = self.dispersion_freqs_fenicsx[i]
            indicator_vals = self.eval_var_out_fenicsx[self.indicator_index][i]
            for j, f in enumerate(freqs):
                all_k_forward.append(k)
                all_f_forward.append(f)
                self.points_data_forward.append({'k': k, 'f': f, 'k_idx': i, 'mode_idx': j})
                if j < len(indicator_vals):
                    all_vals_forward.append(indicator_vals[j])
                else:
                    all_vals_forward.append(0.5)
        
        # Plot dispersion curve
        norm = plt.Normalize(vmin=0, vmax=1)
        sc = ax.scatter(all_k_forward, all_f_forward, c=all_vals_forward, 
                       cmap='turbo', norm=norm, s=20, picker=True)
        
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(self.fenicsx_indicators[self.indicator_index]['name'])
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        ax.set_xlabel(r'$k$ [$\pi$/a]')
        ax.set_ylabel(r'$f$ [Hz]')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True)
        
        # Reset clicked points
        self.clicked_points_forward = []
        
        # Connect pick event
        fig.canvas.mpl_connect('pick_event', self._on_pick_forward)
        plt.show()
        
        return self.clicked_points_forward
    
    def _on_pick_forward(self, event):
        """Handle pick events for the interactive plot."""
        ind = event.ind[0]
        point = self.points_data_forward[ind]
        
        # Diagnostic print
        print(f"\n{'='*50}")
        print(f"CLICKED POINT {len(self.clicked_points_forward)+1}:")
        print(f"  f = {point['f']:.2f} Hz")
        print(f"  k = {point['k']:.6f} (π/a)")
        print(f"  k_idx = {point['k_idx']}")
        print(f"  mode_idx = {point['mode_idx']}")
        print(f"{'-'*50}")
        
        self.clicked_points_forward.append({
            'f': point['f'],
            'k': point['k'],
            'k_idx': point['k_idx'],
            'mode_idx': point['mode_idx']
        })
    
    def plot_mode_shapes(self, visual_scale=0.5):
        """
        Plot mode shapes for all clicked points using 3D visualization.
        
        Parameters
        ----------
        visual_scale : float
            Scaling factor for displacement visualization
        """
        if not self.clicked_points_forward:
            print("No points clicked! Run start_interactive_picker() first.")
            return
        
        print(f"\n{'-'*50}")
        print(f"Plotting mode shapes for {len(self.clicked_points_forward)} clicked points...")
        print(f"{'-'*50}")
        
        # Get base mesh once
        topology, cell_types, geometry = plot.vtk_mesh(self.mesh)
        grid_base = pv.UnstructuredGrid(topology, cell_types, geometry)
        
        for idx, point in enumerate(self.clicked_points_forward):
            print(f"\nPlotting {idx+1}/{len(self.clicked_points_forward)}: "
                  f"f = {point['f']:.2f} Hz, k = {point['k']:.6f}")
            
            k_idx = point['k_idx']
            mode_idx = point['mode_idx']
            
            modes_at_k = self.all_modes_displacements[k_idx]
            
            # Diagnostic print: available modes
            print(f"\n  Available modes at k_idx={k_idx} "
                  f"(k={self.k_values_fenicsx[k_idx]:.6f}):")
            for m_idx, mode in enumerate(modes_at_k):
                freq = mode.get('frequency', 'N/A')
                print(f"    mode_idx={m_idx}, frequency={freq:.2f}")
            
            if mode_idx < len(modes_at_k):
                mode_data = modes_at_k[mode_idx]
                u_plot = mode_data['displacement_real']
                freq_actual = mode_data['frequency']
                k_real = point['k']
                
                # Diagnostic print: mode found
                print(f"\n   Found mode with mode_idx={mode_idx}")
                print(f"     clicked f = {point['f']:.2f} Hz")
                
                self._plot_single_mode_shape(grid_base, u_plot, freq_actual, 
                                            k_real, idx, visual_scale)
            else:
                print(f"  Mode index {mode_idx} not found for k index {k_idx}")
        
        print("\nDone! All mode shapes plotted.")
    
    def _plot_single_mode_shape(self, grid_base, u_plot, freq_actual, k_real, 
                                point_idx, visual_scale):
        """
        Plot a single mode shape using PyVista.
        
        Parameters
        ----------
        grid_base : pv.UnstructuredGrid
            Base mesh grid
        u_plot : np.ndarray
            Displacement field
        freq_actual : float
            Actual frequency of the mode
        k_real : float
            Actual wavenumber
        point_idx : int
            Index of the clicked point
        visual_scale : float
            Scaling factor for visualization
        """
        # Create grid and attach displacement
        grid = grid_base.copy()
        grid["u"] = u_plot
        
        # Calculate displacement magnitude
        u_mag = np.linalg.norm(u_plot, axis=1)
        max_u = np.max(u_mag)
        
        # Normalize for colorbar (0 to 1)
        if max_u > 0:
            u_mag_normalized = u_mag / max_u
        else:
            u_mag_normalized = u_mag
        
        # Store normalized values for colorbar
        grid["Normalized |u|"] = u_mag_normalized
        
        # Warp the mesh with fixed visual scale
        warped = grid.warp_by_vector("u", factor=visual_scale)
        
        # Copy normalized values to warped grid
        warped["Normalized |u|"] = u_mag_normalized
        
        # Create plotter
        p = pv.Plotter(window_size=(1000, 800))
        p.set_background('white')
        
        # Show undeformed mesh as wireframe
        p.add_mesh(grid, style="wireframe", color="lightgray", line_width=1)
        
        # Show deformed mesh colored by normalized displacement
        p.add_mesh(warped, scalars="Normalized |u|", cmap="plasma", 
                  show_scalar_bar=True,
                  scalar_bar_args={
                      'title': 'Normalized |u|',
                      'n_labels': 5,
                      'fmt': '%.2f'
                  })
        
        # Add text info
        p.add_text(f"Point {point_idx+1}: f = {freq_actual:.1f} Hz, "
                  f"k = {k_real:.4f}\nScale: {visual_scale:.3f}x", 
                  font_size=12, position='upper_left', color='black')
        
        p.show_axes()
        p.view_isometric()
        p.show()







class InversePlotter:
    """Class for plotting dispersion curves and mode shapes from inverse problem results.
    
    Uses MeshGenerator class for mesh creation.
    """
    
    # Class constants for indicator mapping
    INDICATOR_NAMES = {
        0: 'pol_x',
        1: 'pol_y', 
        2: 'pol_z',
        3: 'curl_x',
        4: 'curl_y',
        5: 'curl_z'
    }
    
    INDICATOR_DISPLAY_NAMES = {
        0: 'Polarization X',
        1: 'Polarization Y',
        2: 'Polarization Z',
        3: 'Curl X',
        4: 'Curl Y',
        5: 'Curl Z'
    }
    
    def __init__(self, save_path, plots_path, indicator_index=0):
        """
        Initialize the InversePlotter.
        
        Parameters
        ----------
        save_path : str
            Path to the directory containing result files
        plots_path : str
            Path to the directory where plots will be saved
        indicator_index : int or None
            Index of indicator to highlight (0-5) or None for no colorbar
        """
        self.save_path = save_path
        self.plots_path = plots_path
        self.indicator_index = indicator_index
        
        # Initialize mesh generator (will be configured when creating mesh)
        self.mesh_generator = None
        self.mesh = None
        self.cell_tags = None
        self.facet_tags = None
        self.function_space = None
        self.L_f = None
        
        # Inverse problem data
        self.f_values = None
        self.dispersion_k_real = None
        self.dispersion_k_imag = None
        self.eval_var_out = None
        self.all_modes_displacements = None
        self.file_basename = None
        self.indicators = None
        
        # Interactive picker data
        self.clicked_points_inverse = []
        self.points_data_inverse = []
        
        # Setup
        self._setup_matplotlib()
        
    def _setup_matplotlib(self):
        """Configure matplotlib for LaTeX-style rendering."""
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "axes.labelsize": 14,
            "font.size": 13,
            "legend.fontsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        })
    
    def create_mesh_from_step(self, step_filename, mesh_name='mesh_rod', lc=10, 
                             periodic_direction='x'):
        """
        Create mesh from STEP-file using MeshGenerator class.
        
        Parameters
        ----------
        step_filename : str
            Path to the STEP file (without extension)
        mesh_name : str
            Name for the mesh
        lc : float
            Characteristic mesh element length
        periodic_direction : str
            Direction of periodicity: 'x', 'y', or 'z'
        """
        print("\n" + "-"*60)
        print("CREATING MESH USING MESHGENERATOR")
        print("-"*60)
        
        # Initialize and use MeshGenerator
        self.mesh_generator = MeshGenerator()
        self.mesh, self.cell_tags, self.facet_tags = self.mesh_generator.from_step_file(
            input_file=step_filename,
            output_file=mesh_name,
            lc=lc,
            periodic_direction=periodic_direction
        )
        
        # Get L_f from the mesh generator (already calculated with correct direction)
        self.L_f = self.mesh_generator.L_periodic
        
        print(f"\n[MeshGenerator] Periodic direction: {periodic_direction}")
        print(f"[MeshGenerator] Periodic length L_f = {self.L_f:.6f} m")
        
        return self.mesh, self.cell_tags, self.facet_tags
    
    def load_inverse_data(self, filename):
        """
        Load inverse problem results from pickle file.
        
        Parameters
        ----------
        filename : str
            Name of the pickle file (without path)
        """
        full_path = os.path.join(self.save_path, filename)
        self.file_basename = os.path.splitext(filename)[0]
        
        print(f"\nLoading data from: {full_path}")
        print(f"File size: {os.path.getsize(full_path) / 1e6:.2f} MB")
        
        with open(full_path, 'rb') as f:
            obj = pickle.load(f)
            f_name = obj[0]
            num_f = obj[1]
            self.f_values = obj[2]
            self.dispersion_k_real = obj[3]
            self.dispersion_k_imag = obj[4]
            self.eval_var_out = obj[5]          # Polarization and curl data
            self.all_modes_displacements = obj[6]  # Mode shapes
        
        print(f"Loaded data: {len(self.f_values)} frequencies")
        print(f"Frequency range: {self.f_values[0]:.1f} - {self.f_values[-1]:.1f} Hz")
        
        # Scale k-values
        self._scale_k_values()
        
        # Create indicators
        self._create_indicators()
        
    def _scale_k_values(self):
        """Scale k-values using the periodicity length from MeshGenerator."""
        print("\nScaling k values...")
        
        # Scale real k-values
        for i in range(len(self.dispersion_k_real)):
            for j in range(len(self.dispersion_k_real[i])):
                self.dispersion_k_real[i][j] = self.dispersion_k_real[i][j] * self.L_f / np.pi
        
        # Scale imaginary k-values
        for i in range(len(self.dispersion_k_imag)):
            for j in range(len(self.dispersion_k_imag[i])):
                self.dispersion_k_imag[i][j] = self.dispersion_k_imag[i][j] * self.L_f / np.pi
        
        print(f"k-values scaled with L_f = {self.L_f:.4f} m")
        
    def _create_indicators(self):
        """Create indicator dictionaries for inverse problem data."""
        self.indicators = [
            {'name': r'$p_x$', 'tag': 'pol_x', 'values': self.eval_var_out[0], 
             'unit': '', 'colormap': 'turbo', 'cmap_on': True, 'limits': (0, 1)},
            {'name': 'Polarization in y', 'tag': 'pol_y', 'values': self.eval_var_out[1], 
             'unit': '', 'colormap': 'turbo', 'cmap_on': True, 'limits': (0, 1)},
            {'name': r'$p_z$', 'tag': 'pol_z', 'values': self.eval_var_out[2], 
             'unit': '', 'colormap': 'turbo', 'cmap_on': True, 'limits': (0, 1)},
            {'name': r'$p_x^{(\psi)}$', 'tag': 'curl_x', 'values': self.eval_var_out[3], 
             'unit': '', 'colormap': 'cool', 'cmap_on': True, 'limits': (0, 1)},
            {'name': 'Curl around y', 'tag': 'curl_y', 'values': self.eval_var_out[4], 
             'unit': '', 'colormap': 'cool', 'cmap_on': True, 'limits': (0, 1)},
            {'name': r'$p_z^{(\psi)}$', 'tag': 'curl_z', 'values': self.eval_var_out[5], 
             'unit': '', 'colormap': 'cool', 'cmap_on': True, 'limits': (0, 1)},
        ]
    
    def _plot_dispersion_curve_inverse(self, frequency, wavenumber, indicators=None,
                                       indicator_index=None, xlim=None, ylim=None,
                                       color='blue', marker='o', s=20, alpha=1,
                                       label=None, xlabel='k [π/a]', ylabel='f [Hz]',
                                       ax=None, filter_positive=True, add_colorbar=True,
                                       interactive=False):
        """
        Plot dispersion curve for inverse problem (Frequency vs Wavenumber).
        
        Parameters
        ----------
        frequency : dict
            Dictionary with 'values' key containing frequency values
        wavenumber : dict
            Dictionary with 'values' key containing wavenumber lists
        indicators : list or None
            List of indicator dictionaries
        indicator_index : int or None
            Index of indicator to use for coloring
        xlim, ylim : tuple or None
            Axis limits
        color : str
            Color for markers (when no indicator)
        marker : str
            Marker style
        s : int
            Marker size
        alpha : float
            Transparency
        label : str
            Legend label
        xlabel, ylabel : str
            Axis labels
        ax : matplotlib.axes.Axes or None
            Axes to plot on (creates new if None)
        filter_positive : bool
            If True, only plot positive wavenumbers
        add_colorbar : bool
            Whether to add a colorbar
        interactive : bool
            Enable interactive picking
            
        Returns
        -------
        fig, ax : tuple
            Figure and axes objects
        points_data : list
            List of point data for interactive picking
        """
        f_vals = frequency['values']
        k_vals_list = wavenumber['values']
        
        # Collect all data points
        all_k = []
        all_f = []
        all_indicators = []
        points_data = []
        
        for i, f in enumerate(f_vals):
            k_list = k_vals_list[i]
            if not k_list:
                continue
            
            if indicators is not None and indicator_index is not None:
                indicator_vals = indicators[indicator_index]['values'][i]
                for j, k in enumerate(k_list):
                    # Apply filtering based on parameter
                    should_include = (k > 0) if filter_positive else True
                    
                    if should_include:
                        all_k.append(k)
                        all_f.append(f)
                        
                        indicator_value = indicator_vals[j] if j < len(indicator_vals) else 0.5
                        all_indicators.append(indicator_value)
                        
                        mode_index = None
                        if i < len(self.all_modes_displacements) and j < len(self.all_modes_displacements[i]):
                            mode_index = self.all_modes_displacements[i][j].get('mode_index', j)
                        
                        points_data.append({
                            'k': k,
                            'f': f,
                            'f_idx': i,
                            'mode_j': j,
                            'mode_index': mode_index,
                            'indicator_value': indicator_value
                        })
            else:
                for j, k in enumerate(k_list):
                    should_include = (k > 0) if filter_positive else True
                    
                    if should_include:
                        all_k.append(k)
                        all_f.append(f)
                        
                        mode_index = None
                        if i < len(self.all_modes_displacements) and j < len(self.all_modes_displacements[i]):
                            mode_index = self.all_modes_displacements[i][j].get('mode_index', j)
                        
                        points_data.append({
                            'k': k,
                            'f': f,
                            'f_idx': i,
                            'mode_j': j,
                            'mode_index': mode_index,
                            'indicator_value': 0.5
                        })
        
        # Use existing axes or create new figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
        
        # Plot with or without colormap
        if indicators is not None and indicator_index is not None and all_indicators:
            cmap = indicators[indicator_index].get('colormap', 'turbo')
            norm = plt.Normalize(vmin=0, vmax=1)
            sc = ax.scatter(all_k, all_f, c=all_indicators, 
                          cmap=cmap, norm=norm, s=s, alpha=alpha, marker=marker, 
                          label=label if label else None,
                          picker=interactive)
            
            if add_colorbar:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(indicators[indicator_index]['name'])
                cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            
            if 'limits' in indicators[indicator_index]:
                sc.set_clim(indicators[indicator_index]['limits'])
        else:
            ax.scatter(all_k, all_f, color=color, s=s, marker=marker, 
                     alpha=alpha, label=label if label else None,
                     picker=interactive)
        
        # Labels and limits
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Add vertical line at x=0 if showing negative values
        if not filter_positive:
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        
        ax.grid(True, alpha=0.3)
        
        if label:
            ax.legend()
        
        plt.tight_layout()
        
        return fig, ax, points_data
    
    def plot_real_imag_side_by_side(self, xlim_real=(0.001, 1), xlim_imag=(0, 1), 
                                    ylim=(0, 20000)):
        """
        Plot real and imaginary dispersion curves side by side.
        
        Parameters
        ----------
        xlim_real : tuple
            X-axis limits for Re(k) plot
        xlim_imag : tuple
            X-axis limits for Im(k) plot
        ylim : tuple
            Y-axis limits for f 
        """
        
        selected_indicator = self._get_selected_indicator_name()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot real wavenumber dispersion
        fig, ax1, points_data_real = self._plot_dispersion_curve_inverse(
            frequency={'values': self.f_values},
            wavenumber={'values': self.dispersion_k_real},
            indicators=self.indicators,
            indicator_index=self.indicator_index,
            xlim=xlim_real,
            ylim=ylim,
            xlabel=r'Re$(k)$ [$\pi$/a]',
            ylabel=r'$f$ [Hz]',
            color='blue',
            marker='o',
            s=15,
            alpha=1,
            ax=ax1,
            filter_positive=True,
            add_colorbar=False,
            interactive=False
        )
        
        # Plot imaginary wavenumber dispersion
        fig, ax2, points_data_imag = self._plot_dispersion_curve_inverse(
            frequency={'values': self.f_values},
            wavenumber={'values': self.dispersion_k_imag},
            indicators=self.indicators,
            indicator_index=self.indicator_index,
            xlim=xlim_imag,
            ylim=ylim,
            xlabel=r'Im$(k)$ [$\pi$/a]',
            ylabel=r'$f$ [Hz]',
            color='red',
            marker='o',
            s=15,
            alpha=1,
            ax=ax2,
            filter_positive=False,
            add_colorbar=True,
            interactive=False
        )
        
        # Save combined figure
        save_path = os.path.join(self.plots_path, 
                                f'{self.file_basename}_real_imag_side_by_side_{selected_indicator}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Side-by-side plot saved to: {save_path}")
        plt.show()
    
    def _get_selected_indicator_name(self):
        """Get the name of the currently selected indicator."""
        if self.indicator_index is not None:
            return self.INDICATOR_NAMES[self.indicator_index]
        return 'none'
    
    def _get_selected_display_name(self):
        """Get the display name of the currently selected indicator."""
        if self.indicator_index is not None:
            return self.INDICATOR_DISPLAY_NAMES[self.indicator_index]
        return 'No Colorbar'
    
    def start_interactive_picker(self, xlim=(0, 1), ylim=(0, 20000)):
        """
        Start interactive picker for selecting points on dispersion curve.
        Click on points to store them for later mode shape visualization.
        
        Parameters
        ----------
        xlim : tuple
            X-axis limits
        ylim : tuple
            Y-axis limits
            
        Returns
        -------
        list
            List of clicked points
        """
        if self.mesh is None:
            raise ValueError("Mesh must be created before starting interactive picker. "
                           "Call create_mesh_from_step() first.")
        
        print("\n" + "-"*50)
        print("INTERACTIVE MODE SHAPE PICKER - INVERSE PROBLEM Re(k)")
        print("-"*50)
        
        # Create interactive picker plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot dispersion curve with interactive picking enabled
        fig, ax, self.points_data_inverse = self._plot_dispersion_curve_inverse(
            frequency={'values': self.f_values},
            wavenumber={'values': self.dispersion_k_real},
            indicators=self.indicators,
            indicator_index=self.indicator_index,
            xlim=xlim,
            ylim=ylim,
            xlabel=r'Re$(k)$ [$\pi$/a]',
            ylabel=r'$f$ [Hz]',
            ax=ax,
            filter_positive=True,
            add_colorbar=True,
            interactive=True
        )
        
        # Reset clicked points
        self.clicked_points_inverse = []
        
        # Connect pick event
        fig.canvas.mpl_connect('pick_event', self._on_pick_inverse)
        plt.show()
        
        return self.clicked_points_inverse
    
    def _on_pick_inverse(self, event):
        """Handle pick events for the interactive plot."""
        ind = event.ind[0]
        point = self.points_data_inverse[ind]
        
        # Diagnostic print
        print(f"\n{'-'*50}")
        print(f"CLICKED POINT {len(self.clicked_points_inverse)+1}:")
        print(f"  f = {point['f']:.2f} Hz")
        print(f"  k = {point['k']:.6f} (π/a)")
        print(f"  f_idx = {point['f_idx']}")
        print(f"  mode_j = {point['mode_j']}")
        print(f"  mode_index = {point['mode_index']}")
        print(f"{'-'*50}")
        
        self.clicked_points_inverse.append({
            'f': point['f'],
            'k': point['k'],
            'f_idx': point['f_idx'],
            'mode_j': point['mode_j'],
            'mode_index': point['mode_index']
        })
    
    def plot_mode_shapes(self, visual_scale=0.05):
        """
        Plot mode shapes for all clicked points using 3D visualization.
        
        Parameters
        ----------
        visual_scale : float
            Scaling factor for displacement visualization
        """
        if not self.clicked_points_inverse:
            print("No points clicked! Run start_interactive_picker() first.")
            return
        
        print(f"\n{'-'*50}")
        print(f"Plotting mode shapes for {len(self.clicked_points_inverse)} clicked points...")
        print(f"{'-'*50}")
        
        # Get base mesh once
        topology, cell_types, geometry = plot.vtk_mesh(self.mesh)
        grid_base = pv.UnstructuredGrid(topology, cell_types, geometry)
        
        k_scaling = self.L_f / np.pi
        
        for idx, point in enumerate(self.clicked_points_inverse):
            print(f"\nPlotting {idx+1}/{len(self.clicked_points_inverse)}: "
                  f"f = {point['f']:.2f} Hz, k = {point['k']:.6f}")
            
            f_idx = point['f_idx']
            f_actual = self.f_values[f_idx]
            modes_at_f = self.all_modes_displacements[f_idx]
            
            # Diagnostic print: available modes
            print(f"\n  Available modes at f_idx={f_idx} (f={f_actual:.2f} Hz):")
            for m in modes_at_f:
                print(f"    mode_index={m.get('mode_index')}, "
                      f"k_real={m.get('k_real'):.6f}, k_imag={m.get('k_imag'):.6f}")
            
            # Find mode by mode_index
            mode_found = None
            for mode in modes_at_f:
                if mode.get('mode_index') == point['mode_index']:
                    mode_found = mode
                    break
            
            # Try using mode_j as fallback (might not even need mode_j...)
            if mode_found is None:
                print(f"\n No mode found with mode_index = {point['mode_index']}")
                print(f"  Trying mode_j = {point['mode_j']} instead...")
                for mode in modes_at_f:
                    if mode.get('mode_index') == point['mode_j']:
                        mode_found = mode
                        print(f"  Found mode with mode_index = {point['mode_j']}")
                        break
            
            if mode_found is None:
                print(f"  Still no mode found. Skipping...")
                continue
            
            print(f"\n   Found mode with mode_index={mode_found.get('mode_index')}")
            
            u_plot = mode_found['displacement_real']
            k_real_scaled = mode_found['k_real'] * k_scaling
            k_imag_scaled = mode_found['k_imag'] * k_scaling
            
            self._plot_single_mode_shape(grid_base, u_plot, f_actual, 
                                        k_real_scaled, k_imag_scaled, idx, visual_scale)
        
        print("\nDone! All mode shapes plotted.")
    
    def _plot_single_mode_shape(self, grid_base, u_plot, freq_actual, k_real_scaled, 
                                k_imag_scaled, point_idx, visual_scale):
        """
        Plot a single mode shape using PyVista.
        
        Parameters
        ----------
        grid_base : pv.UnstructuredGrid
            Base mesh grid
        u_plot : np.ndarray
            Displacement field
        freq_actual : float
            Actual frequency of the mode
        k_real_scaled : float
            Scaled real wavenumber
        k_imag_scaled : float
            Scaled imaginary wavenumber
        point_idx : int
            Index of the clicked point
        visual_scale : float
            Scaling factor for visualization
        """
        # Create grid and attach displacement
        grid = grid_base.copy()
        grid["u"] = u_plot
        
        # Calculate displacement magnitude
        u_mag = np.linalg.norm(u_plot, axis=1)
        max_u = np.max(u_mag)
        
        # Normalize for colorbar (0 to 1)
        if max_u > 0:
            u_mag_normalized = u_mag / max_u
        else:
            u_mag_normalized = u_mag
        
        # Store normalized values for colorbar
        grid["Normalized |u|"] = u_mag_normalized
        
        # Warp the mesh with fixed visual scale
        warped = grid.warp_by_vector("u", factor=visual_scale)
        
        # Copy normalized values to warped grid
        warped["Normalized |u|"] = u_mag_normalized
        
        # Create plotter
        p = pv.Plotter(window_size=(1000, 800))
        p.set_background('white')
        
        # Show undeformed mesh as wireframe
        p.add_mesh(grid, style="wireframe", color="lightgray", line_width=1)
        
        # Show deformed mesh colored by normalized displacement
        p.add_mesh(warped, scalars="Normalized |u|", cmap="plasma", 
                  show_scalar_bar=True,
                  scalar_bar_args={
                      'title': 'Normalized |u|',
                      'n_labels': 5,
                      'fmt': '%.2f'
                  })
        
        # Add text info
        p.add_text(f"Point {point_idx+1}: f = {freq_actual:.1f} Hz\n"
                  f"k = {k_real_scaled:.4f} + {k_imag_scaled:.4f}i\n"
                  f"Scale: {visual_scale:.3f}x", 
                  font_size=12, position='upper_left', color='black')
        
        p.show_axes()
        p.view_isometric()
        p.show()