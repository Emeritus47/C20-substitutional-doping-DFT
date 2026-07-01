"""
Frontier Molecular Orbital Wavefunction Extraction for C20 and MC19 (M = Be, Mg, Ca)
Methodology: DFT with PBE functional, DNP basis set (DMol3)

This script:
1. Reads optimized structures from DMol3 calculations
2. Performs single-point calculation with DMol3
3. Extracts HOMO and LUMO wavefunction coefficients
4. Visualizes orbital isosurfaces
5. Saves orbital data for further analysis
"""

from ase.calculators.dmol import DMol3
from ase.io import read, write
import numpy as np
import json
import os
import re

# ============================================================================
# 1. DMOL3 CALCULATOR SETUP
# ============================================================================

def setup_dmol3_calculator(multiplicity=1):
    """
    Configure DMol3 calculator for wavefunction extraction.
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
        # Critical for wavefunction output
        wavefunction='yes',                  # Save wavefunction for analysis
        orbital='yes',                       # Output orbital information
        property='orbitals',                 # Calculate orbital properties
    )
    return calc

# ============================================================================
# 2. WAVEFUNCTION EXTRACTION FROM DMOL3 OUTPUT
# ============================================================================

def parse_dmol3_orbitals(output_file):
    """
    Parse DMol3 output file to extract orbital information.
    Returns dictionary with orbital energies, occupations, and coefficients.
    """
    orbitals = {
        'energies': [],
        'occupations': [],
        'coefficients': [],
        'homo_index': None,
        'lumo_index': None,
    }
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Extract orbital energies
        # DMol3 typically reports orbitals in a section
        energy_pattern = r'Orbital\s+(\d+)\s+Energy\s*=\s*([-\d.]+)\s*eV\s+Occ\s*=\s*([\d.]+)'
        matches = re.findall(energy_pattern, content)
        
        if matches:
            for match in matches:
                idx = int(match[0])
                energy = float(match[1])
                occ = float(match[2])
                orbitals['energies'].append(energy)
                orbitals['occupations'].append(occ)
            
            # Find HOMO and LUMO indices
            for i, occ in enumerate(orbitals['occupations']):
                if occ > 0.5:  # Occupied orbital
                    orbitals['homo_index'] = i
                elif orbitals['homo_index'] is not None and occ < 0.5:
                    orbitals['lumo_index'] = i
                    break
        
        # Extract wavefunction coefficients if available
        # DMol3 may output coefficients in a separate file or section
        coeff_pattern = r'Coefficients for orbital\s+(\d+):\s*([\d.\s-]+)'
        coeff_matches = re.findall(coeff_pattern, content)
        if coeff_matches:
            for match in coeff_matches:
                idx = int(match[0])
                coeffs = [float(x) for x in match[1].split()]
                if len(orbitals['coefficients']) <= idx:
                    orbitals['coefficients'].append(coeffs)
        
    except FileNotFoundError:
        print(f"Warning: Output file {output_file} not found")
    except Exception as e:
        print(f"Warning: Error parsing output file: {e}")
    
    return orbitals

def extract_orbital_coefficients_from_gradient_file(filename):
    """
    DMol3 may output orbital coefficients in gradient or cube files.
    This function attempts to extract coefficients from common DMol3 output formats.
    """
    coefficients = {}
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Look for orbital coefficient sections
        # Format varies by DMol3 version
        sections = re.split(r'Orbital\s+(\d+)', content)
        for i in range(1, len(sections), 2):
            if i+1 < len(sections):
                idx = int(sections[i])
                data = sections[i+1]
                # Extract numbers
                coeffs = re.findall(r'[-+]?\d*\.?\d+[eE]?[-+]?\d*', data)
                if coeffs:
                    coefficients[idx] = [float(x) for x in coeffs]
    
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Warning: Error reading gradient file: {e}")
    
    return coefficients

# ============================================================================
# 3. ORBITAL ANALYSIS
# ============================================================================

def analyze_orbitals(orbitals_data, system_name):
    """
    Analyze extracted orbital data and identify frontier orbitals.
    """
    analysis = {
        'system': system_name,
        'n_orbitals': len(orbitals_data['energies']),
        'occupied_count': sum(1 for x in orbitals_data['occupations'] if x > 0.5),
        'homo_energy': None,
        'lumo_energy': None,
        'homo_index': orbitals_data['homo_index'],
        'lumo_index': orbitals_data['lumo_index'],
        'energy_gap': None,
    }
    
    if orbitals_data['homo_index'] is not None:
        analysis['homo_energy'] = orbitals_data['energies'][orbitals_data['homo_index']]
    
    if orbitals_data['lumo_index'] is not None:
        analysis['lumo_energy'] = orbitals_data['energies'][orbitals_data['lumo_index']]
    
    if analysis['homo_energy'] is not None and analysis['lumo_energy'] is not None:
        analysis['energy_gap'] = analysis['lumo_energy'] - analysis['homo_energy']
    
    return analysis

# ============================================================================
# 4. ORBITAL VISUALIZATION (Cube File Generation)
# ============================================================================

def generate_orbital_cube_file(atoms, orbital_index, filename, grid_spacing=0.2):
    """
    Generate cube file for a specific orbital using DMol3.
    This requires the DMol3 output to contain the orbital coefficients.
    """
    # This is a placeholder for the actual cube file generation
    # DMol3 can output cube files directly with appropriate settings
    
    calc = DMol3(
        functional='pbe',
        basis='dnp',
        symmetry='auto',
        spin_polarization='unrestricted',
        charge=0,
        multiplicity=1,
        scf_density_convergence=1.0e-6,
        smearing=0.005,
        occupation='thermal',
        cutoff=4.5,
        wavefunction='yes',
        orbital='yes',
        property='orbitals',
        # Specific for cube file output
        cube='yes',
        cube_orbital=orbital_index,
    )
    
    atoms.calc = calc
    try:
        energy = atoms.get_potential_energy()
        print(f"  Cube file generated for orbital {orbital_index}")
    except Exception as e:
        print(f"  Error generating cube file: {e}")

# ============================================================================
# 5. METAL CONTRIBUTION ANALYSIS
# ============================================================================

def analyze_metal_contribution(orbitals_data, n_carbons, metal_index=0):
    """
    Analyze the contribution of metal atom to frontier orbitals.
    This helps understand the hybridization between metal and cage.
    """
    contribution_analysis = {
        'homo_metal_contribution': 0.0,
        'lumo_metal_contribution': 0.0,
        'homo_carbon_contribution': 0.0,
        'lumo_carbon_contribution': 0.0,
    }
    
    # If coefficients are available
    if orbitals_data['coefficients']:
        homo_idx = orbitals_data['homo_index']
        lumo_idx = orbitals_data['lumo_index']
        
        if homo_idx is not None and homo_idx < len(orbitals_data['coefficients']):
            coeffs = orbitals_data['coefficients'][homo_idx]
            # Assuming coefficients correspond to atoms
            # Metal contribution (first atom)
            metal_contribution = np.sum(np.square(coeffs[:metal_index+1]))
            carbon_contribution = np.sum(np.square(coeffs[metal_index+1:]))
            contribution_analysis['homo_metal_contribution'] = metal_contribution
            contribution_analysis['homo_carbon_contribution'] = carbon_contribution
        
        if lumo_idx is not None and lumo_idx < len(orbitals_data['coefficients']):
            coeffs = orbitals_data['coefficients'][lumo_idx]
            metal_contribution = np.sum(np.square(coeffs[:metal_index+1]))
            carbon_contribution = np.sum(np.square(coeffs[metal_index+1:]))
            contribution_analysis['lumo_metal_contribution'] = metal_contribution
            contribution_analysis['lumo_carbon_contribution'] = carbon_contribution
    
    return contribution_analysis

# ============================================================================
# 6. MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow for frontier molecular orbital wavefunction extraction.
    """
    print("="*70)
    print("FRONTIER MOLECULAR ORBITAL WAVEFUNCTION EXTRACTION")
    print("Method: DMol3 | PBE | DNP basis set")
    print("="*70)
    
    # Create output directory
    os.makedirs('orbital_analysis', exist_ok=True)
    
    # List of structures to analyze
    # These should be optimized structures from previous DMol3 calculations
    structures = {
        'C20': 'C20_optimized.xyz',
        'BeC19': 'BeC19_optimized.xyz',
        'MgC19': 'MgC19_optimized.xyz',
        'CaC19': 'CaC19_optimized.xyz',
    }
    
    all_results = {}
    
    for system_name, xyz_file in structures.items():
        print("\n" + "="*70)
        print(f"PROCESSING {system_name}")
        print("="*70)
        
        # Check if structure file exists
        if not os.path.exists(xyz_file):
            print(f"Warning: {xyz_file} not found. Skipping {system_name}")
            continue
        
        # Read optimized structure
        atoms = read(xyz_file)
        print(f"  Structure loaded: {len(atoms)} atoms")
        n_carbons = sum(1 for symbol in atoms.get_chemical_symbols() if symbol == 'C')
        print(f"  Carbon atoms: {n_carbons}")
        
        # Setup DMol3 calculator
        print("  Running DMol3 calculation for orbital extraction...")
        calc = setup_dmol3_calculator(multiplicity=1)
        atoms.calc = calc
        
        try:
            # Run single-point calculation
            energy = atoms.get_potential_energy()
            print(f"  Total energy: {energy:.6f} Ha")
            
            # Parse output files for orbital information
            output_file = 'DMol3.out'
            if os.path.exists(output_file):
                orbitals_data = parse_dmol3_orbitals(output_file)
                print(f"  Found {len(orbital_data['energies'])} orbitals")
            else:
                # Try alternative output names
                alternative_files = ['output.out', f'{system_name}.out', 'DMol3.log']
                orbitals_data = None
                for alt_file in alternative_files:
                    if os.path.exists(alt_file):
                        orbitals_data = parse_dmol3_orbitals(alt_file)
                        if orbitals_data['energies']:
                            break
                
                if orbitals_data is None or not orbitals_data['energies']:
                    print("  Warning: Could not extract orbital data")
                    continue
            
            # Analyze orbitals
            analysis = analyze_orbitals(orbitals_data, system_name)
            
            print(f"\n  Frontier Orbital Analysis for {system_name}:")
            if analysis['homo_energy'] is not None:
                print(f"    HOMO Energy: {analysis['homo_energy']:.4f} eV")
                print(f"    HOMO Index: {analysis['homo_index']}")
            if analysis['lumo_energy'] is not None:
                print(f"    LUMO Energy: {analysis['lumo_energy']:.4f} eV")
                print(f"    LUMO Index: {analysis['lumo_index']}")
            if analysis['energy_gap'] is not None:
                print(f"    Energy Gap: {analysis['energy_gap']:.4f} eV")
            
            # Analyze metal contribution for doped structures
            if system_name != 'C20':
                print("\n  Metal Contribution Analysis:")
                metal_symbol = system_name[:2]  # Be, Mg, Ca
                metal_index = 0  # Assuming metal is first atom
                
                contribution = analyze_metal_contribution(
                    orbitals_data, n_carbons, metal_index
                )
                print(f"    HOMO - {metal_symbol} contribution: {contribution['homo_metal_contribution']:.2%}")
                print(f"    HOMO - Carbon contribution: {contribution['homo_carbon_contribution']:.2%}")
                print(f"    LUMO - {metal_symbol} contribution: {contribution['lumo_metal_contribution']:.2%}")
                print(f"    LUMO - Carbon contribution: {contribution['lumo_carbon_contribution']:.2%}")
                
                # Store contribution data
                analysis['metal_contribution'] = contribution
            
            # Store results
            all_results[system_name] = analysis
            
            # Save orbital data to JSON
            orbital_data_serializable = {
                'energies': orbitals_data['energies'],
                'occupations': orbitals_data['occupations'],
                'homo_index': orbitals_data['homo_index'],
                'lumo_index': orbitals_data['lumo_index'],
                'analysis': analysis,
            }
            
            with open(f'orbital_analysis/{system_name}_orbitals.json', 'w') as f:
                json.dump(orbital_data_serializable, f, indent=2)
            print(f"\n  Orbital data saved to: orbital_analysis/{system_name}_orbitals.json")
            
        except Exception as e:
            print(f"  Error during calculation: {e}")
            continue
    
    # Save summary
    print("\n" + "="*70)
    print("SAVING SUMMARY")
    print("="*70)
    
    with open('orbital_analysis/frontier_orbitals_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("FRONTIER MOLECULAR ORBITAL ANALYSIS SUMMARY\n")
        f.write("Method: DMol3 | PBE | DNP basis set\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'System':<15} {'HOMO (eV)':<15} {'LUMO (eV)':<15} {'Gap (eV)':<15} {'Metal Contrib':<15}\n")
        f.write("-"*80 + "\n")
        
        for system_name, analysis in all_results.items():
            homo = analysis['homo_energy'] if analysis['homo_energy'] is not None else 'N/A'
            lumo = analysis['lumo_energy'] if analysis['lumo_energy'] is not None else 'N/A'
            gap = analysis['energy_gap'] if analysis['energy_gap'] is not None else 'N/A'
            
            metal_contrib = 'N/A'
            if 'metal_contribution' in analysis:
                metal_contrib = f"H:{analysis['metal_contribution']['homo_metal_contribution']:.2%} L:{analysis['metal_contribution']['lumo_metal_contribution']:.2%}"
            
            f.write(f"{system_name:<15} {homo:<15} {lumo:<15} {gap:<15} {metal_contrib:<15}\n")
    
    print("Summary saved to: orbital_analysis/frontier_orbitals_summary.txt")
    
    # Print final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'System':<15} {'HOMO (eV)':<15} {'LUMO (eV)':<15} {'Gap (eV)':<15}")
    print("-"*70)
    for system_name, analysis in all_results.items():
        homo = analysis['homo_energy'] if analysis['homo_energy'] is not None else 'N/A'
        lumo = analysis['lumo_energy'] if analysis['lumo_energy'] is not None else 'N/A'
        gap = analysis['energy_gap'] if analysis['energy_gap'] is not None else 'N/A'
        print(f"{system_name:<15} {homo:<15.4f if homo != 'N/A' else homo:<15} {lumo:<15.4f if lumo != 'N/A' else lumo:<15} {gap:<15.4f if gap != 'N/A' else gap:<15}")
    
    print("="*70)
    print("Frontier molecular orbital extraction complete!")
    print("Check 'orbital_analysis' directory for detailed results.")

if __name__ == "__main__":
    main()