"""
HOMO-LUMO energy gap calculation for C20 and MC19 (M = Be, Mg, Ca)
Methodology: DFT with PBE functional, DNP basis set (DMol3), spin-polarized calculations

This script:
1. Reads optimized structures from DMol3 calculations
2. Performs single-point energy calculation with DMol3
3. Extracts HOMO and LUMO energies
4. Calculates energy gaps
"""

from ase.calculators.dmol import DMol3
from ase.io import read
import numpy as np
import json
import os

# ============================================================================
# 1. DMOL3 CALCULATOR SETUP
# ============================================================================

def setup_dmol3_calculator(multiplicity=1):
    """
    Configure DMol3 calculator for single-point energy calculation.
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
        # Important: Request wavefunction output for orbital analysis
        wavefunction='yes',                  # Save wavefunction for analysis
        orbital='yes',                       # Output orbital information
    )
    return calc

# ============================================================================
# 2. HOMO-LUMO EXTRACTION
# ============================================================================

def extract_homo_lumo_from_dmol3(calc):
    """
    Extract HOMO and LUMO energies from DMol3 calculator.
    
    Note: DMol3 output parsing depends on the version and output format.
    This function attempts to extract orbital energies from the calculator object.
    """
    try:
        # Attempt to get orbital energies from calculator
        # The exact attributes may vary with ASE/DMol3 version
        if hasattr(calc, 'get_homo_lumo'):
            homo, lumo = calc.get_homo_lumo()
            return homo, lumo
        elif hasattr(calc, 'get_orbital_energies'):
            # Get all orbital energies
            energies = calc.get_orbital_energies()
            if energies is not None and len(energies) > 0:
                # Find HOMO (highest occupied) and LUMO (lowest unoccupied)
                # This is a simplified approach; actual occupation may vary
                n_electrons = calc.get_number_of_electrons() if hasattr(calc, 'get_number_of_electrons') else None
                if n_electrons is not None:
                    homo_index = int(n_electrons / 2) - 1
                    lumo_index = homo_index + 1
                    homo = energies[homo_index]
                    lumo = energies[lumo_index]
                    return homo, lumo
        elif hasattr(calc, 'get_eigenvalues'):
            eigenvalues = calc.get_eigenvalues()
            if eigenvalues is not None:
                # Assuming eigenvalues sorted, HOMO is highest occupied
                # This needs to be adjusted based on occupation
                homo = np.max(eigenvalues[eigenvalues < 0])  # Approximate
                lumo = np.min(eigenvalues[eigenvalues >= 0])  # Approximate
                return homo, lumo
    except Exception as e:
        print(f"Warning: Could not extract HOMO-LUMO from calculator: {e}")
    
    return None, None

def get_homo_lumo_from_file(filename):
    """
    Parse DMol3 output file to extract HOMO and LUMO energies.
    """
    homo = None
    lumo = None
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
        # Look for HOMO and LUMO energies in output
        # DMol3 typically reports orbital energies in the output
        import re
        
        # Pattern for HOMO energy
        homo_pattern = r'HOMO\s*=\s*([-\d.]+)\s*eV'
        homo_match = re.search(homo_pattern, content)
        if homo_match:
            homo = float(homo_match.group(1))
        
        # Pattern for LUMO energy
        lumo_pattern = r'LUMO\s*=\s*([-\d.]+)\s*eV'
        lumo_match = re.search(lumo_pattern, content)
        if lumo_match:
            lumo = float(lumo_match.group(1))
        
        # Alternative patterns
        if homo is None or lumo is None:
            # Look for "Highest occupied orbital" and "Lowest unoccupied orbital"
            homo_pattern = r'Highest occupied orbital\s*:\s*([-\d.]+)\s*eV'
            lumo_pattern = r'Lowest unoccupied orbital\s*:\s*([-\d.]+)\s*eV'
            
            homo_match = re.search(homo_pattern, content)
            if homo_match:
                homo = float(homo_match.group(1))
            
            lumo_match = re.search(lumo_pattern, content)
            if lumo_match:
                lumo = float(lumo_match.group(1))
                
    except FileNotFoundError:
        print(f"Warning: Output file {filename} not found")
    except Exception as e:
        print(f"Warning: Error parsing output file: {e}")
    
    return homo, lumo

# ============================================================================
# 3. ENERGY GAP CALCULATION
# ============================================================================

def calculate_energy_gap(atoms, system_name, multiplicity=1):
    """
    Calculate HOMO-LUMO energy gap for a given structure.
    """
    print(f"\nCalculating HOMO-LUMO for {system_name}...")
    
    # Setup calculator
    calc = setup_dmol3_calculator(multiplicity)
    atoms.calc = calc
    
    # Run single-point calculation
    try:
        energy = atoms.get_potential_energy()
        print(f"  Total energy: {energy:.6f} Ha")
    except Exception as e:
        print(f"  Error in SCF calculation: {e}")
        return None
    
    # Try to extract HOMO-LUMO from calculator
    homo, lumo = extract_homo_lumo_from_dmol3(calc)
    
    # If extraction from calculator failed, try reading output file
    if homo is None or lumo is None:
        # DMol3 output file name varies; check common names
        output_files = [
            f'{system_name}.out',
            f'{system_name}_DMol3.out', 
            'DMol3.out',
            'output.out'
        ]
        for outfile in output_files:
            if os.path.exists(outfile):
                homo, lumo = get_homo_lumo_from_file(outfile)
                if homo is not None and lumo is not None:
                    break
    
    if homo is None or lumo is None:
        print(f"  Warning: Could not extract HOMO/LUMO for {system_name}")
        return None
    
    # Calculate energy gap
    energy_gap = lumo - homo
    
    return {
        'system': system_name,
        'total_energy_ha': energy,
        'total_energy_ev': energy * 27.2114,  # Convert Ha to eV
        'homo_ev': homo,
        'lumo_ev': lumo,
        'energy_gap_ev': energy_gap,
        'multiplicity': multiplicity,
    }

# ============================================================================
# 4. SPIN MULTIPLICITY SCREENING
# ============================================================================

def screen_multiplicities(atoms, system_name):
    """
    Test different spin multiplicities as per methodology.
    """
    multiplicities = [1, 2, 3, 4, 5, 6, 7]
    results = {}
    
    print(f"\nScreening spin multiplicities for {system_name}...")
    
    for mult in multiplicities:
        print(f"  Testing multiplicity {mult}...")
        result = calculate_energy_gap(atoms, f"{system_name}_mult{mult}", mult)
        if result:
            results[mult] = result
            print(f"    Multiplicity {mult}: Gap = {result['energy_gap_ev']:.4f} eV")
        else:
            results[mult] = None
            print(f"    Multiplicity {mult}: Failed")
    
    return results

# ============================================================================
# 5. MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow for HOMO-LUMO energy gap calculation.
    """
    print("="*60)
    print("HOMO-LUMO ENERGY GAP CALCULATION")
    print("Method: DMol3 with PBE functional, DNP basis set")
    print("="*60)
    
    # Create output directory
    os.makedirs('homo_lumo_results', exist_ok=True)
    
    # List of structures to analyze
    # Options: 'C20_optimized.xyz', 'BeC19_optimized.xyz', etc.
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
        
        # For pure C20, use singlet
        if system_name == 'C20':
            result = calculate_energy_gap(atoms, system_name, multiplicity=1)
            if result:
                all_results[system_name] = result
                print(f"\n{system_name} Results:")
                print(f"  Total Energy: {result['total_energy_ev']:.6f} eV")
                print(f"  HOMO: {result['homo_ev']:.4f} eV")
                print(f"  LUMO: {result['lumo_ev']:.4f} eV")
                print(f"  Energy Gap: {result['energy_gap_ev']:.4f} eV")
        else:
            # For doped structures, screen different spin multiplicities
            spin_results = screen_multiplicities(atoms, system_name)
            all_results[system_name] = spin_results
            
            # Find best multiplicity (lowest energy)
            valid_results = {k: v for k, v in spin_results.items() 
                           if v is not None}
            if valid_results:
                best_mult = min(valid_results.keys(), 
                              key=lambda k: valid_results[k]['total_energy_ha'])
                print(f"\n{system_name} Best Multiplicity: {best_mult}")
                best_result = valid_results[best_mult]
                print(f"  HOMO: {best_result['homo_ev']:.4f} eV")
                print(f"  LUMO: {best_result['lumo_ev']:.4f} eV")
                print(f"  Energy Gap: {best_result['energy_gap_ev']:.4f} eV")
    
    # Step: Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Convert results to serializable format
    results_serializable = {}
    for system_name, result in all_results.items():
        if isinstance(result, dict):
            if 'homo_ev' in result:  # Single result
                results_serializable[system_name] = {
                    'homo_ev': float(result['homo_ev']),
                    'lumo_ev': float(result['lumo_ev']),
                    'energy_gap_ev': float(result['energy_gap_ev']),
                    'total_energy_ev': float(result['total_energy_ev']),
                    'multiplicity': int(result['multiplicity']),
                }
            else:  # Multiplicity results
                results_serializable[system_name] = {}
                for mult, res in result.items():
                    if res is not None:
                        results_serializable[system_name][str(mult)] = {
                            'homo_ev': float(res['homo_ev']),
                            'lumo_ev': float(res['lumo_ev']),
                            'energy_gap_ev': float(res['energy_gap_ev']),
                            'total_energy_ev': float(res['total_energy_ev']),
                        }
    
    # Save to JSON
    with open('homo_lumo_results/energy_gap_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    # Save summary table
    with open('homo_lumo_results/energy_gap_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("HOMO-LUMO ENERGY GAP SUMMARY\n")
        f.write("Method: DMol3 | PBE | DNP basis set\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'System':<15} {'Multiplicity':<12} {'HOMO (eV)':<12} {'LUMO (eV)':<12} {'Gap (eV)':<12}\n")
        f.write("-"*70 + "\n")
        
        for system_name, result in all_results.items():
            if isinstance(result, dict) and 'homo_ev' in result:
                f.write(f"{system_name:<15} {result['multiplicity']:<12} {result['homo_ev']:<12.4f} {result['lumo_ev']:<12.4f} {result['energy_gap_ev']:<12.4f}\n")
            elif isinstance(result, dict):
                for mult, res in result.items():
                    if res is not None:
                        f.write(f"{system_name:<15} {mult:<12} {res['homo_ev']:<12.4f} {res['lumo_ev']:<12.4f} {res['energy_gap_ev']:<12.4f}\n")
    
    print("Results saved to:")
    print("  - homo_lumo_results/energy_gap_results.json")
    print("  - homo_lumo_results/energy_gap_summary.txt")
    
    # Print final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'System':<15} {'Multiplicity':<12} {'HOMO (eV)':<12} {'LUMO (eV)':<12} {'Gap (eV)':<12}")
    print("-"*70)
    for system_name, result in all_results.items():
        if isinstance(result, dict) and 'homo_ev' in result:
            print(f"{system_name:<15} {result['multiplicity']:<12} {result['homo_ev']:<12.4f} {result['lumo_ev']:<12.4f} {result['energy_gap_ev']:<12.4f}")
        elif isinstance(result, dict):
            # Show only best multiplicity for doped structures
            valid_results = {k: v for k, v in result.items() if v is not None}
            if valid_results:
                best_mult = min(valid_results.keys(), 
                              key=lambda k: valid_results[k]['total_energy_ev'])
                res = valid_results[best_mult]
                print(f"{system_name:<15} {best_mult:<12} {res['homo_ev']:<12.4f} {res['lumo_ev']:<12.4f} {res['energy_gap_ev']:<12.4f}")
    
    print("="*60)
    print("HOMO-LUMO calculation complete!")

if __name__ == "__main__":
    main()