"""
Charge density difference visualization for C20 and MC19 (M = Be, Mg, Ca)
Methodology: DFT with PBE functional, DNP basis set (DMol3)

This script:
1. Reads optimized structures from DMol3 calculations
2. Performs single-point calculations for total and fragment densities
3. Calculates charge density difference
4. Generates visualization files for analysis
"""

from ase.calculators.dmol import DMol3
from ase.io import read, write
import numpy as np
import os
import json

# ============================================================================
# 1. DMOL3 CALCULATOR SETUP
# ============================================================================

def setup_dmol3_calculator(multiplicity=1):
    """
    Configure DMol3 calculator for charge density calculation.
    """
    calc = DMol3(
        functional='pbe',                    # PBE functional
        basis='dnp',                         # DNP basis set
        symmetry='auto',                     # Auto symmetry detection
        spin_polarization='unrestricted' if multiplicity > 1 else 'restricted',
        charge=0,                            # Neutral system
        multiplicity=multiplicity,           # Spin multiplicity
        scf_density_convergence=1.0e-6,      # SCF convergence
        smearing=0.005,                      # Thermal smearing
        occupation='thermal',                # Thermal occupation
        cutoff=4.5,                          # Real-space cutoff (Angstrom)
        # Density output settings
        grid='fine',                         # Fine grid for density
        density='yes',                       # Output density
        potential='yes',                     # Output potential
        wavefunction='yes',                  # Save wavefunction
    )
    return calc

# ============================================================================
# 2. CHARGE DENSITY CALCULATION FUNCTIONS
# ============================================================================

def calculate_total_density(atoms, system_name, multiplicity=1):
    """
    Calculate total charge density for a system.
    """
    print(f"  Calculating total density for {system_name}...")
    
    # Setup calculator
    calc = setup_dmol3_calculator(multiplicity)
    atoms.calc = calc
    
    try:
        energy = atoms.get_potential_energy()
        print(f"    Total energy: {energy:.6f} Ha")
        
        # Get density grid
        if hasattr(calc, 'get_density_grid'):
            density_grid = calc.get_density_grid()
            return density_grid
        else:
            print(f"    Warning: Could not retrieve density grid from calculator")
            return None
            
    except Exception as e:
        print(f"    Error in density calculation: {e}")
        return None

def calculate_fragment_densities(atoms, fragments):
    """
    Calculate charge density for fragments of a system.
    
    fragments: list of atom indices for each fragment
    """
    fragment_densities = []
    
    for i, fragment_indices in enumerate(fragments):
        print(f"  Calculating density for fragment {i+1}...")
        
        # Create fragment
        fragment_atoms = atoms[fragment_indices]
        
        # Setup calculator for fragment
        calc = setup_dmol3_calculator(multiplicity=1)
        fragment_atoms.calc = calc
        
        try:
            energy = fragment_atoms.get_potential_energy()
            print(f"    Fragment energy: {energy:.6f} Ha")
            
            # Get density grid
            if hasattr(calc, 'get_density_grid'):
                density_grid = calc.get_density_grid()
                fragment_densities.append(density_grid)
            else:
                fragment_densities.append(None)
                
        except Exception as e:
            print(f"    Error in fragment density calculation: {e}")
            fragment_densities.append(None)
    
    return fragment_densities

def calculate_density_difference(total_density, fragment_densities, operation='subtract'):
    """
    Calculate density difference between total and fragments.
    
    operation: 'subtract' (total - sum(fragments)) or 'add' (sum(fragments) - total)
    """
    if total_density is None:
        return None
    
    # Sum fragment densities
    sum_fragment_density = None
    for frag_density in fragment_densities:
        if frag_density is not None:
            if sum_fragment_density is None:
                sum_fragment_density = np.copy(frag_density)
            else:
                sum_fragment_density += frag_density
    
    if sum_fragment_density is None:
        return None
    
    # Calculate difference
    if operation == 'subtract':
        difference = total_density - sum_fragment_density
    else:
        difference = sum_fragment_density - total_density
    
    return difference

# ============================================================================
# 3. VISUALIZATION FUNCTIONS
# ============================================================================

def write_cube_file(density_grid, atoms, filename, comment="Charge density"):
    """
    Write density grid to Gaussian cube file format.
    """
    if density_grid is None:
        print(f"Warning: No density data to write to {filename}")
        return
    
    try:
        # Get grid dimensions
        grid_shape = density_grid.shape
        if len(grid_shape) != 3:
            print(f"Warning: Invalid grid shape for cube file: {grid_shape}")
            return
        
        # Get atom positions
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        # Write cube file
        with open(filename, 'w') as f:
            # Header
            f.write(f"Cube file generated for {comment}\n")
            f.write("Generated by DMol3 density analysis\n")
            
            # Number of atoms and origin
            f.write(f"{len(atoms):5d} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
            
            # Grid dimensions and spacing
            # This is a simplified approach; actual grid parameters from DMol3
            nx, ny, nz = grid_shape
            dx, dy, dz = 0.1, 0.1, 0.1  # Approximate spacing
            
            f.write(f"{nx:5d} {dx:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
            f.write(f"{ny:5d} {0.0:12.6f} {dy:12.6f} {0.0:12.6f}\n")
            f.write(f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {dz:12.6f}\n")
            
            # Atom coordinates
            for symbol, pos in zip(symbols, positions):
                atomic_number = 6 if symbol == 'C' else 4 if symbol == 'Be' else 12 if symbol == 'Mg' else 20
                f.write(f"{atomic_number:5d} {0.0:12.6f} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
            
            # Density values
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        value = density_grid[i, j, k]
                        f.write(f"{value:13.6e}")
                        if (i * ny * nz + j * nz + k) % 6 == 5:
                            f.write("\n")
                    f.write("\n")
        
        print(f"  Cube file saved: {filename}")
        
    except Exception as e:
        print(f"  Error writing cube file: {e}")

def write_xyz_density(atoms, density_grid, filename):
    """
    Write atomic positions with density values to XYZ format.
    """
    if density_grid is None:
        print(f"Warning: No density data to write to {filename}")
        return
    
    try:
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        with open(filename, 'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write("Atomic positions with density values\n")
            
            # This is simplified; actual density values at atom positions
            # would require interpolation from grid
            for symbol, pos in zip(symbols, positions):
                # Placeholder density value (actual would be interpolated)
                density_value = 0.0
                f.write(f"{symbol} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f} {density_value:12.6f}\n")
        
        print(f"  XYZ file saved: {filename}")
        
    except Exception as e:
        print(f"  Error writing XYZ file: {e}")

# ============================================================================
# 4. CHARGE DENSITY DIFFERENCE ANALYSIS
# ============================================================================

def analyze_density_difference(density_diff):
    """
    Analyze charge density difference.
    """
    if density_diff is None:
        return None
    
    # Calculate statistics
    min_val = np.min(density_diff)
    max_val = np.max(density_diff)
    mean_val = np.mean(density_diff)
    std_val = np.std(density_diff)
    
    # Count positive and negative regions
    positive_count = np.sum(density_diff > 0)
    negative_count = np.sum(density_diff < 0)
    total_points = density_diff.size
    
    return {
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'std': std_val,
        'positive_fraction': positive_count / total_points,
        'negative_fraction': negative_count / total_points,
    }

# ============================================================================
# 5. MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow for charge density difference visualization.
    """
    print("="*60)
    print("CHARGE DENSITY DIFFERENCE VISUALIZATION")
    print("Method: DMol3 with PBE functional, DNP basis set")
    print("="*60)
    
    # Create output directory
    os.makedirs('density_analysis', exist_ok=True)
    
    # List of structures to analyze
    structures = {
        'C20': 'C20_optimized.xyz',
        'BeC19': 'BeC19_optimized.xyz',
        'MgC19': 'MgC19_optimized.xyz',
        'CaC19': 'CaC19_optimized.xyz',
    }
    
    all_results = {}
    
    for system_name, xyz_file in structures.items():
        print("\n" + "="*60)
        print(f"PROCESSING {system_name}")
        print("="*60)
        
        # Check if structure file exists
        if not os.path.exists(xyz_file):
            print(f"Warning: {xyz_file} not found. Skipping {system_name}")
            continue
        
        # Read optimized structure
        atoms = read(xyz_file)
        print(f"  Structure loaded: {len(atoms)} atoms")
        
        # Determine multiplicity (singlet for pure, adjust for doped if needed)
        multiplicity = 1
        
        # Step 1: Calculate total density
        total_density = calculate_total_density(atoms, system_name, multiplicity)
        
        if total_density is None:
            print("  Warning: Could not calculate total density. Skipping system.")
            continue
        
        # Step 2: Define fragments based on system type
        if system_name == 'C20':
            # For C20, analyze C-C bonding
            # Fragment 1: Carbon atoms (all)
            fragments = [list(range(len(atoms)))]
            fragment_names = ['C_all']
        else:
            # For doped systems, separate metal and carbon
            metal_index = None
            metal_symbols = ['Be', 'Mg', 'Ca']
            for i, atom in enumerate(atoms):
                if atom.symbol in metal_symbols:
                    metal_index = i
                    break
            
            if metal_index is not None:
                # Fragment 1: Metal atom
                # Fragment 2: Carbon atoms
                carbon_indices = [i for i in range(len(atoms)) if i != metal_index]
                fragments = [[metal_index], carbon_indices]
                fragment_names = ['metal', 'carbon_cage']
            else:
                # Default: separate first atom from rest
                fragments = [[0], list(range(1, len(atoms)))]
                fragment_names = ['fragment_1', 'fragment_2']
        
        # Step 3: Calculate fragment densities
        fragment_densities = calculate_fragment_densities(atoms, fragments)
        
        # Step 4: Calculate density difference
        print("  Calculating density difference...")
        density_diff = calculate_density_difference(total_density, fragment_densities, 'subtract')
        
        if density_diff is None:
            print("  Warning: Could not calculate density difference.")
            continue
        
        # Step 5: Analyze density difference
        diff_stats = analyze_density_difference(density_diff)
        
        # Step 6: Save visualization files
        print("  Saving visualization files...")
        
        # Save cube files for visualization with VMD, GaussView, etc.
        write_cube_file(total_density, atoms, 
                       f'density_analysis/{system_name}_total_density.cube', 
                       f'{system_name} total density')
        
        write_cube_file(density_diff, atoms, 
                       f'density_analysis/{system_name}_density_diff.cube', 
                       f'{system_name} density difference (total - fragments)')
        
        # Save XYZ files for quick viewing
        write_xyz_density(atoms, density_diff, 
                         f'density_analysis/{system_name}_density_diff.xyz')
        
        # Save density data as numpy array for further analysis
        np.save(f'density_analysis/{system_name}_total_density.npy', total_density)
        np.save(f'density_analysis/{system_name}_density_diff.npy', density_diff)
        
        # Store results
        all_results[system_name] = {
            'system': system_name,
            'fragments': fragment_names,
            'diff_stats': diff_stats,
        }
        
        print(f"\n  Density Difference Statistics:")
        print(f"    Min: {diff_stats['min']:.6e}")
        print(f"    Max: {diff_stats['max']:.6e}")
        print(f"    Mean: {diff_stats['mean']:.6e}")
        print(f"    Std: {diff_stats['std']:.6e}")
        print(f"    Positive fraction: {diff_stats['positive_fraction']:.2%}")
        print(f"    Negative fraction: {diff_stats['negative_fraction']:.2%}")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save summary
    with open('density_analysis/density_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("CHARGE DENSITY DIFFERENCE ANALYSIS SUMMARY\n")
        f.write("Method: DMol3 | PBE | DNP basis set\n")
        f.write("="*70 + "\n\n")
        
        for system_name, results in all_results.items():
            f.write(f"\n{system_name}:\n")
            f.write("-"*50 + "\n")
            f.write(f"  Fragments: {', '.join(results['fragments'])}\n\n")
            
            stats = results['diff_stats']
            f.write("  Density Difference Statistics:\n")
            f.write(f"    Min: {stats['min']:.6e}\n")
            f.write(f"    Max: {stats['max']:.6e}\n")
            f.write(f"    Mean: {stats['mean']:.6e}\n")
            f.write(f"    Std: {stats['std']:.6e}\n")
            f.write(f"    Positive fraction: {stats['positive_fraction']:.2%}\n")
            f.write(f"    Negative fraction: {stats['negative_fraction']:.2%}\n")
            f.write("\n")
    
    # Save summary as JSON
    with open('density_analysis/density_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("Results saved to:")
    print("  - density_analysis/ directory")
    print("  - density_analysis/density_summary.txt")
    print("  - density_analysis/density_summary.json")
    
    # Print final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'System':<15} {'Positive Fraction':<20} {'Negative Fraction':<20}")
    print("-"*55)
    for system_name, results in all_results.items():
        stats = results['diff_stats']
        print(f"{system_name:<15} {stats['positive_fraction']:<20.2%} {stats['negative_fraction']:<20.2%}")
    
    print("\n" + "="*60)
    print("Visualization files generated:")
    print("  - .cube files for VMD, GaussView, etc.")
    print("  - .xyz files for quick viewing")
    print("  - .npy files for further analysis")
    print("="*60)
    print("Charge density difference analysis complete!")

if __name__ == "__main__":
    main()