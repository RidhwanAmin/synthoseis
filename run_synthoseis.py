#!/usr/bin/env python3
"""
Synthoseis Runner Script

Converts the Jupyter notebook workflow into a command-line tool for generating
seismic data with different configurations (smooth vs faulty).

Usage:
    python run_synthoseis.py --config ./config/smooth_example.json
    python run_synthoseis.py --config ./config/faulty_example.json
"""

import argparse
import json
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pprint

# Import the synthoseis main module
import main as mn


def determine_config_type(config_path):
    """
    Determine if config is for 'smooth' or 'faulty' seismic data
    based on the filename.
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        str: Either 'smooth' or 'faulty'
    """
    config_filename = os.path.basename(config_path).lower()
    
    if 'smooth' in config_filename or 'clean' in config_filename or 'flat' in config_filename:
        return 'smooth'
    elif 'fault' in config_filename or 'faulty' in config_filename or 'structured' in config_filename or 'realistic' in config_filename:
        return 'faulty'
    else:
        # Default to faulty if unclear
        print(f"Warning: Could not determine config type from '{config_filename}'. Defaulting to 'faulty'.")
        return 'faulty'


def get_project_display_name(config_data, config_path):
    """
    Get a display name for the project that can be different from the technical 'project' parameter.
    
    Args:
        config_data (dict): Configuration dictionary
        config_path (str): Path to config file
        
    Returns:
        str: Display name for the project
    """
    # Check if there's a custom display name in the config
    if 'project_display_name' in config_data:
        return config_data['project_display_name']
    
    # Otherwise, derive from filename
    config_filename = os.path.basename(config_path)
    if 'clean' in config_filename.lower() or 'flat' in config_filename.lower():
        return 'flat_clean_model'
    elif 'structured' in config_filename.lower() or 'realistic' in config_filename.lower():
        return 'structured_realistic_model'
    else:
        return config_data.get('project', 'synthoseis_model')


def resolve_random_values(config):
    """
    Resolve random value arrays in config to actual values.
    
    Converts arrays like ["randint", 2, 3] to appropriate format based on parameter type:
    - Some become {"min": 2, "max": 3} (for dict-style access)
    - Some become [2, 3] (for indexed access)
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Configuration with resolved random values
    """
    resolved_config = {}
    
    # Parameters that should NOT be resolved (must remain as arrays/lists or simple values)
    array_parameters = {
        'cube_shape', 'incident_angles', 'max_column_height', 'closure_types',
        'project_folder', 'work_folder', 'project'  # Also protect string parameters
    }
    
    # Parameters that should remain as simple numeric values (not converted to arrays or dicts)
    simple_numeric_parameters = {
        'infill_factor', 'digi', 'pad_samples', 'min_closure_voxels_simple', 
        'min_closure_voxels_faulted', 'min_closure_voxels_onlap', 'num_lyr_lut',
        'dip_factor_max', 'thickness_min', 'thickness_max', 'sand_layer_thickness',
        'min_number_faults', 'max_number_faults',  # These must be simple integers for Synthoseis
        'bandwidth_ord'  # Filter order must be a simple integer
    }
    
    # Parameters that expect [min, max] array format (indexed access)
    indexed_array_parameters = {
        'initial_layer_stdev',
        'bandwidth_low', 'bandwidth_high',
        'seabed_min_depth'
    }
    
    # Parameters that expect [left, mode, right] triangular distribution format
    triangular_parameters = {
        'signal_to_noise_ratio_db'
    }
    
    # Parameters that expect {"min": x, "max": y} dict format
    dict_parameters = {
        'sand_layer_fraction'
    }
    
    for key, value in config.items():
        # Skip resolution for parameters that must remain as arrays
        if key in array_parameters:
            resolved_config[key] = value
            continue
            
        if isinstance(value, list) and len(value) >= 2:
            # Check if first element indicates a random function
            if isinstance(value[0], str) and value[0] == "randint":
                if len(value) == 3:
                    if key in simple_numeric_parameters:
                        # For simple numeric parameters, resolve to a single random value
                        resolved_value = np.random.randint(int(value[1]), int(value[2]) + 1)
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (single randint value)")
                    elif key in indexed_array_parameters:
                        # ["randint", low, high] -> [low, high]
                        resolved_value = [int(value[1]), int(value[2])]
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (array format)")
                    elif key in dict_parameters:
                        # ["randint", low, high] -> {"min": low, "max": high}
                        resolved_value = {"min": int(value[1]), "max": int(value[2])}
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (dict format)")
                    else:
                        # Default to array format for unknown parameters
                        resolved_value = [int(value[1]), int(value[2])]
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (default array format)")
                else:
                    print(f"  Warning: Invalid randint format for {key}: {value}")
                    resolved_config[key] = value
            elif isinstance(value[0], str) and value[0] == "uniform":
                if len(value) == 3:
                    if key in simple_numeric_parameters:
                        # For simple numeric parameters, resolve to a single random value
                        resolved_value = np.random.uniform(float(value[1]), float(value[2]))
                        
                        # Special handling for parameters that need bounds checking
                        if key == 'dip_factor_max':
                            # Ensure dip_factor_max stays within reasonable bounds
                            resolved_value = min(resolved_value, 1.0)
                        elif key == 'sand_layer_thickness':
                            # Ensure sand_layer_thickness is at least 1
                            resolved_value = max(resolved_value, 1.0)
                            
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (single uniform value)")
                    elif key in indexed_array_parameters:
                        # ["uniform", low, high] -> [low, high]
                        resolved_value = [float(value[1]), float(value[2])]
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (array format)")
                    elif key in triangular_parameters:
                        # ["uniform", low, high] -> [low, mode, high] where mode = (low+high)/2
                        low, high = float(value[1]), float(value[2])
                        mode = (low + high) / 2
                        resolved_value = [low, mode, high]
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (triangular format)")
                    elif key in dict_parameters:
                        # ["uniform", low, high] -> {"min": low, "max": high}
                        resolved_value = {"min": float(value[1]), "max": float(value[2])}
                        
                        # Special validation for sand_layer_fraction to ensure Markov chain validity
                        if key == 'sand_layer_fraction':
                            # Ensure sand_layer_fraction doesn't create invalid Markov probabilities
                            # The constraint is: sand_fraction <= sand_thickness / (1 + sand_thickness)
                            # With sand_thickness = 4: max_fraction = 0.8
                            # With sand_thickness = 6: max_fraction = 0.86
                            # To be safe, we'll cap it at 0.75 to work with sand_thickness >= 4
                            max_safe_fraction = 0.75
                            if resolved_value["max"] > max_safe_fraction:
                                print(f"  Warning: Capping sand_layer_fraction max from {resolved_value['max']} to {max_safe_fraction} for Markov chain stability")
                                resolved_value["max"] = max_safe_fraction
                            if resolved_value["min"] > max_safe_fraction:
                                print(f"  Warning: Capping sand_layer_fraction min from {resolved_value['min']} to {max_safe_fraction} for Markov chain stability")
                                resolved_value["min"] = max_safe_fraction
                        
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (dict format)")
                    else:
                        # Default to array format for unknown parameters
                        resolved_value = [float(value[1]), float(value[2])]
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (default array format)")
                elif len(value) == 2:
                    # ["uniform", value] -> [value, value] or {"min": value, "max": value}
                    if key in simple_numeric_parameters:
                        # For simple numeric parameters, just use the single value
                        resolved_value = float(value[1])
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (single value)")
                    elif key in triangular_parameters:
                        # Single value triangular -> [value, value, value]
                        resolved_value = [float(value[1]), float(value[1]), float(value[1])]
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (single value triangular)")
                    elif key in dict_parameters:
                        resolved_value = {"min": float(value[1]), "max": float(value[1])}
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (single value dict)")
                    else:
                        resolved_value = [float(value[1]), float(value[1])]
                        resolved_config[key] = resolved_value
                        print(f"  Resolved {key}: {value} -> {resolved_value} (single value array)")
                else:
                    print(f"  Warning: Invalid uniform format for {key}: {value}")
                    resolved_config[key] = value
            else:
                # Not a random array, keep original value
                resolved_config[key] = value
        elif isinstance(value, dict):
            # Handle nested dictionaries recursively
            resolved_config[key] = resolve_random_values(value)
        else:
            # Keep original value for non-list, non-dict types
            # But convert single values to arrays if needed (except for simple numeric parameters)
            if key in indexed_array_parameters and isinstance(value, (int, float)) and key not in simple_numeric_parameters:
                # Convert single value to [value-small_range, value+small_range] array
                # This ensures low < high for random sampling
                if isinstance(value, int):
                    # For integers, create a small range (e.g., [value-1, value+1])
                    delta = max(1, int(value * 0.1))  # 10% or minimum 1
                    resolved_value = [max(0, value - delta), value + delta]
                else:
                    # For floats, create a small range (e.g., [value*0.9, value*1.1])
                    delta = max(0.1, value * 0.1)  # 10% or minimum 0.1
                    resolved_value = [max(0.0, value - delta), value + delta]
                resolved_config[key] = resolved_value
                print(f"  Converted {key}: {value} -> {resolved_value} (single value to range)")
            else:
                resolved_config[key] = value
    
    return resolved_config


def post_process_config(config):
    """
    Post-process the resolved config to handle special cases.
    
    Args:
        config (dict): Resolved configuration dictionary
        
    Returns:
        dict: Post-processed configuration
    """
    # Handle the "no faults" case for clean/smooth models
    if ('min_number_faults' in config and 'max_number_faults' in config):
        min_faults = config['min_number_faults']
        max_faults = config['max_number_faults']
        
        # Check for zero-fault configuration
        if min_faults == 0 and max_faults == 0:
            print("  Detected no-fault configuration - ensuring clean model generation")
            
            # For no-fault models, set to a small positive range but disable faulting elsewhere
            # This avoids the np.random.randint(0, 0) error
            config['min_number_faults'] = 0
            config['max_number_faults'] = 1  # Allow randint(0, 1) which can return 0
            
            # Set dip_factor_max to very small value for smooth models
            if 'dip_factor_max' in config and isinstance(config['dip_factor_max'], (int, float)):
                if config['dip_factor_max'] < 0.1:  # If it's already very small
                    config['dip_factor_max'] = max(0.001, config['dip_factor_max'])  # Ensure it's not exactly 0
                    print(f"  Adjusted dip_factor_max to {config['dip_factor_max']} for smooth model")
        
        # Handle fault parameters that are arrays but should be integers for special cases
        elif (isinstance(min_faults, list) and len(min_faults) == 2 and
              isinstance(max_faults, list) and len(max_faults) == 2):
            
            min_val = min_faults[0]
            max_val = max_faults[1]
            
            # If min and max are the same, use simple integers with small range
            if min_val == max_val:
                if min_val == 0:
                    # Handle zero case specially
                    config['min_number_faults'] = 0
                    config['max_number_faults'] = 1
                    print(f"  Adjusted zero fault parameters to allow randint(0, 1)")
                else:
                    config['min_number_faults'] = int(min_val)
                    config['max_number_faults'] = int(max_val + 1)  # Ensure max > min
                    print(f"  Simplified fault parameters to {min_val}-{max_val+1} faults range")
    
    return config


def load_config(config_path):
    """
    Load configuration from JSON file and resolve random values.
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        dict: Configuration with resolved random values
        
    Raises:
        FileNotFoundError: If config file cannot be loaded
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("Resolving random values in config...")
        # Resolve any random value arrays
        config = resolve_random_values(config)
        
        # Post-process config to handle special cases
        config = post_process_config(config)
        
        return config
        
    except json.JSONDecodeError as e:
        raise FileNotFoundError(f"Invalid JSON in config file {config_path}: {e}")
    except Exception as e:
        raise FileNotFoundError(f"Could not load config file {config_path}: {e}")


def create_output_directories():
    """
    Create the output directory structure:
    output/
    ├── smooth/
    └── faulty/
    """
    output_dirs = ['output/smooth', 'output/faulty']
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")


def find_generated_npy_file(run_id):
    """
    Find the generated .npy file after model creation.
    
    Args:
        run_id (int): The run ID used in model creation
        
    Returns:
        str: Path to the generated .npy file
        
    Raises:
        FileNotFoundError: If no .npy file is found
    """
    # Look for the seismic output directory pattern - updated patterns based on actual file structure
    search_patterns = [
        f'synthoseis_example/assets/seismic__*_{run_id}/seismicCubes_RFC__*_normalized_*.npy',
        f'synthoseis_example/assets/seismic__*_{run_id}/seismicCubes_RFC_fullstack*.npy',
        f'synthoseis_example/seismic__*_{run_id}/seismicCubes_RFC_fullstack*.npy',
        f'temp_folder__*_{run_id}/seismicCubes_RFC_fullstack*.npy',
        f'/tmp/synthoseis_example*/seismic__*_{run_id}/seismicCubes_RFC_fullstack*.npy'
    ]
    
    for pattern in search_patterns:
        files = glob.glob(pattern)
        if files:
            print(f"Found seismic data file: {files[0]}")
            return files[0]
    
    # If not found, raise an error
    raise FileNotFoundError(f"Could not find generated .npy file for run_id {run_id}")


def plot_seismic_data(seismic_data, config_type, run_id, output_dir='output'):
    """
    Create and save plots of the seismic data with robust error handling.
    
    Args:
        seismic_data (np.ndarray): 3D seismic cube data
        config_type (str): 'smooth' or 'faulty'
        run_id (int): Run ID for unique file naming
        output_dir (str): Base output directory
    """
    print(f"Plotting seismic data of shape: {seismic_data.shape}")
    
    try:
        # Set matplotlib backend to ensure compatibility
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Get middle slices
        inline_index = seismic_data.shape[0] // 2
        crossline_index = seismic_data.shape[1] // 2
        #depth_index = seismic_data.shape[2] // 2
        
        # Create inline and crossline slices
        inline_slice = seismic_data[inline_index, :, :]
        crossline_slice = seismic_data[:, crossline_index, :]
        #depth_slice = seismic_data[:, :, depth_index]
        
        # Ensure output directory exists
        plot_output_dir = os.path.join(output_dir, config_type)
        os.makedirs(plot_output_dir, exist_ok=True)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Seismic Data Visualization - {config_type.capitalize()} Configuration', fontsize=16)
        
        # Calculate dynamic color scale based on data percentiles for better visualization
        vmin = np.percentile(seismic_data, 5)
        vmax = np.percentile(seismic_data, 95)
        
        # Plot inline slice
        im1 = axes[0].imshow(
            inline_slice.T,
            cmap="seismic",
            aspect="auto",
            interpolation="bilinear",
            vmin=vmin, vmax=vmax
        )
        axes[0].set_title(f'Inline Slice (index {inline_index})')
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Crossline')
        axes[0].set_ylabel('Depth')
        plt.colorbar(im1, ax=axes[0], label="Amplitude")
        
        # Plot crossline slice
        im2 = axes[1].imshow(
            crossline_slice.T,
            cmap="seismic",
            aspect="auto",
            interpolation="bilinear",
            vmin=vmin, vmax=vmax
        )
        axes[1].set_title(f'Crossline Slice (index {crossline_index})')
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Inline')
        axes[1].set_ylabel('Depth')
        plt.colorbar(im2, ax=axes[1], label="Amplitude")
        
        # # Plot depth slice
        # im3 = axes[1, 0].imshow(
        #     depth_slice,
        #     cmap="seismic",
        #     aspect="auto",
        #     interpolation="bilinear",
        #     vmin=vmin, vmax=vmax
        # )
        # axes[1, 0].set_title(f'Depth Slice (index {depth_index})')
        # axes[1, 0].set_xlabel('Crossline')
        # axes[1, 0].set_ylabel('Inline')
        # plt.colorbar(im3, ax=axes[1, 0], label="Amplitude")
        
        # Add text summary in the fourth subplot
        # axes[1, 1].text(0.1, 0.8, f'Data Shape: {seismic_data.shape}', fontsize=12, transform=axes[1, 1].transAxes)
        # axes[1, 1].text(0.1, 0.7, f'Config Type: {config_type}', fontsize=12, transform=axes[1, 1].transAxes)
        # axes[1, 1].text(0.1, 0.6, f'Data Range: [{seismic_data.min():.2e}, {seismic_data.max():.2e}]', fontsize=12, transform=axes[1, 1].transAxes)
        # axes[1, 1].text(0.1, 0.5, f'Data Mean: {seismic_data.mean():.2e}', fontsize=12, transform=axes[1, 1].transAxes)
        # axes[1, 1].text(0.1, 0.4, f'Data Std: {seismic_data.std():.2e}', fontsize=12, transform=axes[1, 1].transAxes)
        # axes[1, 1].text(0.1, 0.3, f'Color Scale: [{vmin:.2e}, {vmax:.2e}]', fontsize=12, transform=axes[1, 1].transAxes)
        # axes[1, 1].set_title('Data Summary')
        # axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save the plot with robust error handling
        output_path = os.path.join(plot_output_dir, f'fullstack_plot_run_{run_id}.png')
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"✅ Fullstack plot saved successfully to: {output_path}")
        except Exception as e:
            # Try alternative save method
            try:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"✅ Fullstack plot saved (lower DPI) to: {output_path}")
            except Exception as e2:
                print(f"❌ Failed to save fullstack plot: {e2}")
        
        plt.close(fig)
        
        # # Create and save a simple inline view
        # try:
        #     fig_simple = plt.figure(figsize=(12, 6))
        #     plt.imshow(
        #         inline_slice.T,
        #         cmap="seismic",
        #         aspect="auto",
        #         interpolation="bilinear",
        #         vmin=vmin, vmax=vmax
        #     )
        #     plt.colorbar(label="Amplitude")
        #     plt.title(f"Inline Slice - {config_type.capitalize()} Configuration (Run {run_id})")
        #     plt.gca().invert_yaxis()
        #     plt.xlabel('Crossline')
        #     plt.ylabel('Depth')
            
        #     simple_output_path = os.path.join(plot_output_dir, f'inline_slice_run_{run_id}.png')
        #     try:
        #         plt.savefig(simple_output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        #         print(f"✅ Inline slice plot saved successfully to: {simple_output_path}")
        #     except Exception as e:
        #         # Try alternative save method
        #         try:
        #             plt.savefig(simple_output_path, dpi=150, bbox_inches='tight')
        #             print(f"✅ Inline slice plot saved (lower DPI) to: {simple_output_path}")
        #         except Exception as e2:
        #             print(f"❌ Failed to save inline slice plot: {e2}")
            
        #     plt.close(fig_simple)
            
        # except Exception as e:
        #     print(f"❌ Failed to create inline slice plot: {e}")
        
        # # Create a depth slice plot as well
        # try:
        #     fig_depth = plt.figure(figsize=(10, 8))
        #     plt.imshow(
        #         depth_slice,
        #         cmap="seismic",
        #         aspect="auto",
        #         interpolation="bilinear",
        #         vmin=vmin, vmax=vmax
        #     )
        #     plt.colorbar(label="Amplitude")
        #     plt.title(f"Depth Slice - {config_type.capitalize()} Configuration (Run {run_id})")
        #     plt.xlabel('Crossline')
        #     plt.ylabel('Inline')
            
        #     depth_output_path = os.path.join(plot_output_dir, f'depth_slice_run_{run_id}.png')
        #     try:
        #         plt.savefig(depth_output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        #         print(f"✅ Depth slice plot saved successfully to: {depth_output_path}")
        #     except Exception as e:
        #         # Try alternative save method
        #         try:
        #             plt.savefig(depth_output_path, dpi=150, bbox_inches='tight')
        #             print(f"✅ Depth slice plot saved (lower DPI) to: {depth_output_path}")
        #         except Exception as e2:
        #             print(f"❌ Failed to save depth slice plot: {e2}")
            
        #     plt.close(fig_depth)
            
        # except Exception as e:
        #     print(f"❌ Failed to create depth slice plot: {e}")
        
        # # Always close all figures to prevent memory issues
        # plt.close('all')
        
    except Exception as e:
        print(f"❌ Critical error in plotting function: {e}")
        print("Continuing with pipeline execution...")


def main():
    """
    Main function to run the synthoseis pipeline.
    """
    parser = argparse.ArgumentParser(description='Run Synthoseis seismic data generation')
    parser.add_argument('--config', required=True, help='Path to JSON config file')
    parser.add_argument('--run-id', type=int, default=None, help='Run ID for the model (auto-generated if not provided)')
    parser.add_argument('--output-dir', default='output', help='Base output directory')
    parser.add_argument('--verbose', action='store_true', help='Print detailed config information')
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)
    
    try:
        # Load and validate config with random value resolution
        config_data = load_config(args.config)
        print(f"Config loaded successfully from: {args.config}")
        
        if args.verbose:
            print("\nFinal resolved config:")
            pprint.pprint(config_data)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)
    
    # Determine config type
    config_type = determine_config_type(args.config)
    project_display_name = get_project_display_name(config_data, args.config)
    print(f"Detected config type: {config_type}")
    print(f"Project display name: {project_display_name}")
    
    # Create output directories
    create_output_directories()
    
    # Generate run_id if not provided
    if args.run_id is None:
        import time
        args.run_id = int(time.time() * 100) % 10000  # Simple run ID generation
    
    print(f"Using run_id: {args.run_id}")
    
    try:
        # Run the model
        print("Starting model generation...")
        print(f"Config contains {len(config_data)} parameters")
        
        # Save the resolved config temporarily for synthoseis to use
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_config:
            json.dump(config_data, temp_config, indent=2)
            temp_config_path = temp_config.name
        
        try:
            # Call the synthoseis model builder with resolved config
            print(f"Calling mn.build_model with temp config: {temp_config_path}")
            print(f"Config preview of resolved values:")
            for k, v in config_data.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v} (type: {type(v).__name__})")
                elif isinstance(v, list) and len(v) <= 5:
                    print(f"  {k}: {v} (type: list, len: {len(v)})")
                    
            mn.build_model(user_json=temp_config_path, run_id=args.run_id)
        except Exception as e:
            print(f"Error in mn.build_model: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Clean up temporary file
            os.unlink(temp_config_path)
        
        print("Model generation completed successfully!")
        
        # Find the generated .npy file
        npy_file_path = find_generated_npy_file(args.run_id)
        
        # Load the seismic data
        print(f"Loading seismic data from: {npy_file_path}")
        seismic_data = np.load(npy_file_path)
        print(f"Loaded seismic data with shape: {seismic_data.shape}")
        print(f"Original data range: [{seismic_data.min():.2e}, {seismic_data.max():.2e}]")
        
        # Normalize data to range [-1e-3, 1e-3]
        data_max = np.max(np.abs(seismic_data))
        if data_max > 0:
            seismic_data = seismic_data / data_max * 1e-3
            print(f"Normalized data range: [{seismic_data.min():.2e}, {seismic_data.max():.2e}]")
        
        # Create and save plots
        plot_seismic_data(seismic_data, config_type, args.run_id, args.output_dir)
        
        # Save the numpy data with unique naming
        import time
        timestamp = int(time.time())
        npy_output_path = os.path.join(args.output_dir, config_type, f'seismic_data_run_{args.run_id}_{timestamp}.npy')
        np.save(npy_output_path, seismic_data)
        print(f"Seismic data saved to: {npy_output_path}")
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Config type: {config_type}")
        print(f"Run ID: {args.run_id}")
        print(f"Data shape: {seismic_data.shape}")
        print(f"Output directory: {os.path.join(args.output_dir, config_type)}")
        print(f"Files generated:")
        print(f"  - fullstack_plot_run_{args.run_id}.png (inline & crossline views)")
        print(f"  - seismic_data_run_{args.run_id}_{timestamp}.npy")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during model generation or processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
