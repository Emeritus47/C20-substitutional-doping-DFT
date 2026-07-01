"""
Magnetic moment calculation for C20 and MC19 (M = Be, Mg, Ca)
Methodology: DFT with PBE functional, DNP basis set (DMol3), spin-polarized calculations

This script:
1. Reads optimized structures from DMol3 calculations
2. Performs spin-polarized single-point calculation with DMol3
3. Extracts magnetic moments using Mulliken population analysis
4. Calculates total and atomic spin moments
"""

from ase.calculators.dmol import DMol3
from ase.io import read
import numpy as np
import json
import os
import re

# ============================================================================
# 1. DMOL3 CALCULATOR SETUP
# ============================================================================

def setup_dmol3_calculator(multiplicity=1):
    """
    Configure DMol3 calculator for magnetic moment calculation.
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
        # Mulliken population analysis for magnetic moments
        population='mulliken',               # Mulliken population analysis
        spin_density='yes',                  # Output spin density
    )
    return calc

# ============================================================================
# 2. MAGNETIC MOMENT EXTRACTION
# ============================================================================

def extract_magnetic_moment_from_calculator(calc):
    """
    Extract total magnetic moment from DMol3 calculator.
    """
    try:
        # Try to get magnetic moment from calculator
        if hasattr(calc, 'get_magnetic_moment'):
            moment = calc.get_magnetic_moment()
            return float(moment)
        elif hasattr(calc, 'get_spin_moment'):
            moment = calc.get_spin_moment()
            return float(moment)
    except Exception as e:
        print(f"Warning: Could not extract magnetic moment from calculator: {e}")
    
    return None

def extract_atomic_spins_from_file(filename):
    """
    Parse DMol3 output file to extract atomic spin moments.
    """
    atomic_spins = {}
    total_moment = None
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Look for Mulliken population analysis section
        # DMol3 typically reports atomic charges and spins
        
        # Pattern for total magnetic moment
        total_pattern = r'Total spin\s*=\s*([-\d.]+)'
        total_match = re.search(total_pattern, content)
        if total_match:
            total_moment = float(total_match.group(1))
        
        # Look for atomic spin moments
        # Pattern for "Atom X spin = Y"
        atom_spin_pattern = r'Atom\s+(\d+)\s+spin\s*=\s*([-\d.]+)'
        for match in re.finditer(atom_spin_pattern, content):
            atom_index = int(match.group(1)) - 1  # 0-based index
            spin = float(match.group(2))
            atomic_spins[atom_index] = spin
        
        # Alternative pattern for Mulliken spin populations
        mulliken_pattern = r'Atom\s+\d+\s+[A-Z][a-z]?\s+spin\s*=\s*([-\d.]+)'
        if not atomic_spins:
            atom_count = 0
            for match in re.finditer(mulliken_pattern, content):
                spin = float(match.group(1))
                atomic_spins[atom_count] = spin
                atom_count += 1
        
        # Another pattern for spin populations table
        table_pattern = r'Spin\s+Populations\s+\(electrons\)[\s\S]+?(\d+\s+[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+)'
        # This is complex; might need specific pattern based on output format
        
    except FileNotFoundError:
        print(f"Warning: Output file {filename} not found")
    except Exception as e:
        print(f"Warning: Error parsing output file: {e}")
    
    return atomic_spins, total_moment

def get_magnetic_moments(atoms, calc):
    """
    Extract both total and atomic magnetic moments.
    """
    results = {
        'total_moment': None,
        'atomic_moments': {},
        'spin_up_electrons': None,
        'spin_down_electrons': None,
    }
    
    # Try from calculator first
    total_moment = extract_magnetic_moment_from_calculator(calc)
    if total_moment is not None:
        results['total_moment'] = total_moment
    
    # Try to get atomic spins from output file
    output_files = [
        'DMol3.out',
        'output.out',
        'magnetic.out'
    ]
    
    for outfile in output_files:
        if os.path.exists(outfile):
            atomic_spins, total_from_file = extract_atomic_spins_from_file(outfile)
            if atomic_spins:
                results['atomic_moments'] = atomic_spins
            if total_from_file is not None:
                results['total_moment'] = total_from_file
            break
    
    # Try to get spin populations from calculator if available
    try:
        if hasattr(calc, 'get_spin_populations'):
            spin_pop = calc.get_spin_populations()
            if spin_pop is not None:
                results['spin_up_electrons'] = spin_pop.get('up', None)
                results['spin_down_electrons'] = spin_pop.get('down', None)
    except Exception:
        pass
    
    return results

# ============================================================================
# 3. MAGNETIC MOMENT ANALYSIS
# ============================================================================

def analyze_magnetic_moments(atoms, system_name, multiplicity=1):
    """
    Calculate magnetic moments for a given structure.
    """
    print(f"\nCalculating magnetic moments for {system_name}...")
    
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
    
    # Extract magnetic moments
    moment_results = get_magnetic_moments(atoms, calc)
    
    # Add system info
    moment_results['system'] = system_name
    moment_results['multiplicity'] = multiplicity
    moment_results['total_energy_ha'] = energy
    moment_results['total_energy_ev'] = energy * 27.2114
    
    # Print results
    print(f"\n{system_name} Magnetic Moments:")
    print(f"  Multiplicity: {multiplicity}")
    if moment_results['total_moment'] is not None:
        print(f"  Total Magnetic Moment: {moment_results['total_moment']:.6f} μB")
    else:
        print("  Total Magnetic Moment: Not available")
    
    if moment_results['atomic_moments']:
        print(f"  Atomic Spin Moments:")
        symbols = atoms.get_chemical_symbols()
        for atom_idx, spin in sorted(moment_results['atomic_moments'].items()):
            if atom_idx < len(symbols):
                print(f"    Atom {atom_idx+1} ({symbols[atom_idx]}): {spin:.6f} μB")
    else:
        print("  Atomic Spin Moments: Not available")
    
    if moment_results['spin_up_electrons'] is not None:
        print(f"  Spin-up electrons: {moment_results['spin_up_electrons']:.6f}")
        print(f"  Spin-down electrons: {moment_results['spin_down_electrons']:.6f}")
        if moment_results['spin_up_electrons'] is not None and moment_results['spin_down_electrons'] is not None:
            moment_diff = moment_results['spin_up_electrons'] - moment_results['spin_down_electrons']
            print(f"  Spin difference (up - down): {moment_diff:.6f}")
    
    return moment_results

# ============================================================================
# 4. MAGNETIC MOMENT ANALYSIS FUNCTIONS
# ============================================================================

def calculate_atomic_magnetic_moments(atoms, spin_results):
    """
    Calculate atomic magnetic moments from spin populations.
    """
    atomic_moments = {}
    
    if spin_results and 'atomic_moments' in spin_results:
        atomic_moments = spin_results['atomic_moments']
    
    # Additional analysis: identify magnetic atoms
    symbols = atoms.get_chemical_symbols()
    magnetic_atoms = []
    for idx, moment in atomic_moments.items():
        if abs(moment) > 0.001:  # Threshold for significant moment
            magnetic_atoms.append({
                'index': idx,
                'symbol': symbols[idx] if idx < len(symbols) else 'Unknown',
                'moment': moment
            })
    
    return {
        'atomic_moments': atomic_moments,
        'magnetic_atoms': magnetic_atoms,
        'total_moment': spin_results.get('total_moment', 0.0)
    }

def check_spin_state(spin_results):
    """
    Determine if system is spin-polarized.
    """
    if spin_results is None:
        return 'Unknown'
    
    total_moment = spin_results.get('total_moment', None)
    
    if total_moment is None:
        return 'Unknown'
    
    if abs(total_moment) < 0.001:
        return 'Non-magnetic (Singlet)'
    elif abs(total_moment) < 0.01:
        return 'Weakly magnetic'
    elif abs(total_moment) < 1.0:
        return 'Magnetic (Doublet/Triplet)'
    else:
        return 'Strongly magnetic'

# ============================================================================
# 5. SPIN MULTIPLICITY SCREENING
# ============================================================================

def screen_spin_multiplicities(atoms, system_name):
    """
    Test different spin multiplicities to find ground state.
    """
    multiplicities = [1, 2, 3, 4, 5, 6, 7]
    results = {}
    
    print(f"\n" + "="*60)
    print(f"SCREENING SPIN MULTIPLICITIES FOR {system_name}")
    print("="*60)
    
    for mult in multiplicities:
        print(f"\nTesting multiplicity {mult}...")
        result = analyze_magnetic_moments(atoms, f"{system_name}_mult{mult}", mult)
        if result:
            results[mult] = result
        else:
            results[mult] = None
            print(f"  Multiplicity {mult}: Calculation failed")
    
    return results

# ============================================================================
# 6. MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow for magnetic moment calculation.
    """
    print("="*60)
    print("MAGNETIC MOMENT CALCULATION")
    print("Method: DMol3 | PBE | DNP basis set | Mulliken population analysis")
    print("="*60)
    
    # Create output directory
    os.makedirs('magnetic_results', exist_ok=True)
    
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
        print(f"Structure loaded: {len(atoms)} atoms")
        symbols = atoms.get_chemical_symbols()
        print(f"Composition: {dict(zip(*np.unique(symbols, return_counts=True)))}")
        
        # Screen different spin multiplicities
        spin_results = screen_spin_multiplicities(atoms, system_name)
        all_results[system_name] = spin_results
        
        # Find best multiplicity (lowest energy)
        valid_results = {k: v for k, v in spin_results.items() 
                        if v is not None}
        
        if valid_results:
            # Find multiplicity with lowest energy
            best_mult = min(valid_results.keys(), 
                          key=lambda k: valid_results[k]['total_energy_ha'])
            
            print("\n" + "="*60)
            print(f"BEST MULTIPLICITY FOR {system_name}: {best_mult}")
            print("="*60)
            print(f"  Total Energy: {valid_results[best_mult]['total_energy_ev']:.6f} eV")
            print(f"  Total Magnetic Moment: {valid_results[best_mult]['total_moment']:.6f} μB")
            print(f"  Spin State: {check_spin_state(valid_results[best_mult])}")
            
            # Analyze atomic magnetic moments
            atomic_analysis = calculate_atomic_magnetic_moments(atoms, valid_results[best_mult])
            
            if atomic_analysis['magnetic_atoms']:
                print(f"\n  Magnetic Atoms:")
                for atom in atomic_analysis['magnetic_atoms']:
                    print(f"    Atom {atom['index']+1} ({atom['symbol']}): {atom['moment']:.6f} μB")
            else:
                print("\n  No significant atomic magnetic moments detected")
    
    # Step: Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Convert results to serializable format
    results_serializable = {}
    for system_name, spin_results in all_results.items():
        results_serializable[system_name] = {}
        for mult, result in spin_results.items():
            if result is not None:
                results_serializable[system_name][str(mult)] = {
                    'total_moment_muB': float(result['total_moment']) if result['total_moment'] is not None else None,
                    'total_energy_ev': float(result['total_energy_ev']),
                    'multiplicity': int(result['multiplicity']),
                    'spin_state': check_spin_state(result),
                }
                
                # Add atomic moments if available
                if result.get('atomic_moments'):
                    results_serializable[system_name][str(mult)]['atomic_moments'] = {
                        str(k): float(v) for k, v in result['atomic_moments'].items()
                    }
    
    # Save to JSON
    with open('magnetic_results/magnetic_moments.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    # Save summary table
    with open('magnetic_results/magnetic_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("MAGNETIC MOMENT SUMMARY\n")
        f.write("Method: DMol3 | PBE | DNP basis set | Mulliken population analysis\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"{'System':<15} {'Mult.':<8} {'Energy (eV)':<15} {'Total Moment (μB)':<18} {'Spin State':<20}\n")
        f.write("-"*70 + "\n")
        
        for system_name, spin_results in all_results.items():
            for mult, result in sorted(spin_results.items()):
                if result is not None:
                    moment_str = f"{result['total_moment']:.6f}" if result['total_moment'] is not None else "N/A"
                    state_str = check_spin_state(result)
                    f.write(f"{system_name:<15} {mult:<8} {result['total_energy_ev']:<15.6f} {moment_str:<18} {state_str:<20}\n")
        
        # Add best multiplicity summary
        f.write("\n" + "="*70 + "\n")
        f.write("BEST MULTIPLICITY SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"{'System':<15} {'Best Mult.':<12} {'Moment (μB)':<15} {'Spin State':<20}\n")
        f.write("-"*70 + "\n")
        
        for system_name, spin_results in all_results.items():
            valid_results = {k: v for k, v in spin_results.items() 
                           if v is not None}
            if valid_results:
                best_mult = min(valid_results.keys(), 
                              key=lambda k: valid_results[k]['total_energy_ha'])
                result = valid_results[best_mult]
                moment_str = f"{result['total_moment']:.6f}" if result['total_moment'] is not None else "N/A"
                state_str = check_spin_state(result)
                f.write(f"{system_name:<15} {best_mult:<12} {moment_str:<15} {state_str:<20}\n")
    
    print("Results saved to:")
    print("  - magnetic_results/magnetic_moments.json")
    print("  - magnetic_results/magnetic_summary.txt")
    
    # Print final summary
    print("\n" + "="*60)
    print("SUMMARY OF MAGNETIC MOMENTS")
    print("="*60)
    
    print(f"{'System':<15} {'Best Mult.':<12} {'Moment (μB)':<15} {'Spin State':<20}")
    print("-"*60)
    
    for system_name, spin_results in all_results.items():
        valid_results = {k: v for k, v in spin_results.items() 
                       if v is not None}
        if valid_results:
            best_mult = min(valid_results.keys(), 
                          key=lambda k: valid_results[k]['total_energy_ha'])
            result = valid_results[best_mult]
            moment_str = f"{result['total_moment']:.6f}" if result['total_moment'] is not None else "N/A"
            state_str = check_spin_state(result)
            print(f"{system_name:<15} {best_mult:<12} {moment_str:<15} {state_str:<20}")
    
    print("="*60)
    print("Magnetic moment calculation complete!")

if __name__ == "__main__":
    main()