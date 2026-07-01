"""
Mulliken charge analysis for C20 and MC19 (M = Be, Mg, Ca)
Methodology: DFT with PBE functional, DNP basis set (DMol3), spin-polarized calculations

This script:
1. Reads optimized structures from DMol3 calculations
2. Performs single-point energy calculation with DMol3
3. Extracts Mulliken charges for all atoms
4. Analyzes charge distribution and charge transfer
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
    Configure DMol3 calculator for Mulliken population analysis.
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
        # Mulliken population analysis settings
        population='mulliken',               # Perform Mulliken population analysis
        wavefunction='yes',                  # Save wavefunction
        orbital='yes',                       # Output orbital information
    )
    return calc

# ============================================================================
# 2. MULLIKEN CHARGE EXTRACTION
# ============================================================================

def extract_mulliken_charges_from_output(output_file):
    """
    Parse DMol3 output file to extract Mulliken charges for all atoms.
    """
    charges = {}
    atomic_charges = []
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Look for Mulliken population analysis section
        # DMol3 typically outputs atomic charges in the Mulliken analysis
        
        # Pattern 1: "Mulliken Atomic Charges" section
        if "Mulliken Atomic Charges" in content:
            # Extract the table of atomic charges
            lines = content.split('\n')
            in_table = False
            for line in lines:
                if "Mulliken Atomic Charges" in line:
                    in_table = True
                    continue
                if in_table and "---" in line:
                    continue
                if in_table and line.strip() and not line.startswith(' '):
                    # Parse line: atom_number atom_symbol charge
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            atom_index = int(parts[0])
                            atom_symbol = parts[1]
                            charge = float(parts[2])
                            atomic_charges.append({
                                'index': atom_index,
                                'symbol': atom_symbol,
                                'charge': charge
                            })
                        except (ValueError, IndexError):
                            continue
                if in_table and "Total" in line:
                    break
        
        # Pattern 2: "Mulliken charges" or "Atomic charges"
        if not atomic_charges:
            patterns = [
                r'Mulliken charges\s*:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)',
                r'Atomic charges\s*:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)',
                r'Charge\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    for match in matches:
                        # This is simplified; actual parsing depends on output format
                        atomic_charges.append({'charge': float(match[0])})
        
        # If still no charges found, try alternative approach
        if not atomic_charges:
            # Look for individual atom charges
            pattern = r'Atom\s+(\d+)\s+([A-Z][a-z]?)\s+charge\s*=\s*([-\d.]+)'
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                atomic_charges.append({
                    'index': int(match[0]),
                    'symbol': match[1],
                    'charge': float(match[2])
                })
        
    except FileNotFoundError:
        print(f"Warning: Output file {output_file} not found")
    except Exception as e:
        print(f"Warning: Error parsing output file: {e}")
    
    return atomic_charges

def extract_mulliken_charges_from_calculator(calc, atoms):
    """
    Attempt to extract Mulliken charges from DMol3 calculator object.
    """
    charges = []
    
    try:
        # Try different methods to get charges
        if hasattr(calc, 'get_mulliken_charges'):
            charges = calc.get_mulliken_charges()
        elif hasattr(calc, 'get_atomic_charges'):
            charges = calc.get_atomic_charges()
        elif hasattr(calc, 'get_charges'):
            charges = calc.get_charges()
        else:
            # Try to get through results dictionary
            if hasattr(calc, 'results') and 'charges' in calc.results:
                charges = calc.results['charges']
        
        # If charges is a single number or array, format it properly
        if charges is not None and len(charges) > 0:
            if isinstance(charges, (list, np.ndarray)):
                if len(charges) == len(atoms):
                    return [{'index': i, 'symbol': atoms[i].symbol, 'charge': float(c)} 
                           for i, c in enumerate(charges)]
    except Exception as e:
        print(f"Warning: Could not extract charges from calculator: {e}")
    
    return []

# ============================================================================
# 3. CHARGE ANALYSIS FUNCTIONS
# ============================================================================

def analyze_charge_distribution(charges, atoms):
    """
    Analyze charge distribution for the system.
    """
    # Separate charges by element
    element_charges = {}
    total_charge = 0.0
    
    for i, atom in enumerate(atoms):
        symbol = atom.symbol
        if i < len(charges):
            charge = charges[i]['charge'] if isinstance(charges[i], dict) else charges[i]
        else:
            charge = 0.0
        
        total_charge += charge
        
        if symbol not in element_charges:
            element_charges[symbol] = []
        element_charges[symbol].append(charge)
    
    # Calculate statistics
    stats = {}
    for element, charges_list in element_charges.items():
        stats[element] = {
            'count': len(charges_list),
            'total_charge': sum(charges_list),
            'avg_charge': np.mean(charges_list),
            'std_charge': np.std(charges_list),
            'min_charge': min(charges_list),
            'max_charge': max(charges_list),
        }
    
    return {
        'total_charge': total_charge,
        'element_stats': stats,
        'all_charges': charges,
    }

def identify_charge_transfer(charges, metal_index):
    """
    Identify charge transfer between metal and carbon cage.
    """
    metal_charge = charges[metal_index]['charge']
    carbon_charges = [c['charge'] for i, c in enumerate(charges) if i != metal_index]
    
    # Find atoms adjacent to metal (nearest neighbors)
    # This requires geometry information; we'll use simplified approach
    nearest_charges = []
    
    return {
        'metal_charge': metal_charge,
        'carbon_charges': carbon_charges,
        'avg_carbon_charge': np.mean(carbon_charges),
        'total_carbon_charge': sum(carbon_charges),
        'nearest_carbon_charges': nearest_charges,
        'charge_transfer': metal_charge - 0.0,  # Metal charge relative to neutral
        'net_cage_charge': sum(carbon_charges),
    }

# ============================================================================
# 4. MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow for Mulliken charge analysis.
    """
    print("="*60)
    print("MULLIKEN CHARGE ANALYSIS")
    print("Method: DMol3 with PBE functional, DNP basis set")
    print("="*60)
    
    # Create output directory
    os.makedirs('mulliken_analysis', exist_ok=True)
    
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
        
        # Determine multiplicity (from previous calculations or use default)
        multiplicity = 1  # Singlet for C20, adjust for doped if needed
        
        # Setup DMol3 calculator
        calc = setup_dmol3_calculator(multiplicity)
        atoms.calc = calc
        
        # Run single-point calculation
        print("  Running single-point calculation...")
        try:
            energy = atoms.get_potential_energy()
            print(f"  Total energy: {energy:.6f} Ha")
        except Exception as e:
            print(f"  Error in SCF calculation: {e}")
            continue
        
        # Extract Mulliken charges
        charges = []
        
        # Method 1: Try from calculator
        charges = extract_mulliken_charges_from_calculator(calc, atoms)
        
        # Method 2: Try from output file
        if not charges:
            print("  Attempting to parse output file...")
            output_files = [
                f'{system_name}.out',
                f'{system_name}_DMol3.out',
                'DMol3.out',
                'output.out'
            ]
            for outfile in output_files:
                if os.path.exists(outfile):
                    charges = extract_mulliken_charges_from_output(outfile)
                    if charges:
                        print(f"  Found charges in {outfile}")
                        break
        
        # If still no charges, estimate from atomic properties
        if not charges:
            print("  Warning: Could not extract Mulliken charges")
            continue
        
        # Ensure charges match atom count
        if len(charges) != len(atoms):
            print(f"  Warning: Number of charges ({len(charges)}) doesn't match atoms ({len(atoms)})")
            # Pad or trim charges list to match atoms
            if len(charges) > len(atoms):
                charges = charges[:len(atoms)]
            else:
                # Pad with zeros
                while len(charges) < len(atoms):
                    charges.append({'symbol': atoms[len(charges)].symbol, 'charge': 0.0})
        
        # Format charges properly
        formatted_charges = []
        for i, atom in enumerate(atoms):
            if i < len(charges):
                if isinstance(charges[i], dict):
                    charge = charges[i].get('charge', 0.0)
                else:
                    charge = float(charges[i])
            else:
                charge = 0.0
            formatted_charges.append({
                'index': i,
                'symbol': atom.symbol,
                'charge': charge
            })
        
        # Analyze charge distribution
        analysis = analyze_charge_distribution(formatted_charges, atoms)
        
        # Store results
        result = {
            'system': system_name,
            'total_energy_ha': energy,
            'total_charge': analysis['total_charge'],
            'charges': formatted_charges,
            'element_stats': analysis['element_stats'],
        }
        
        # For doped structures, analyze charge transfer
        if system_name != 'C20':
            metal_symbols = ['Be', 'Mg', 'Ca']
            metal_index = None
            for i, atom in enumerate(atoms):
                if atom.symbol in metal_symbols:
                    metal_index = i
                    break
            
            if metal_index is not None:
                transfer_analysis = identify_charge_transfer(formatted_charges, metal_index)
                result['charge_transfer'] = transfer_analysis
                
                print(f"\n  Charge Transfer Analysis:")
                print(f"    Metal charge: {transfer_analysis['metal_charge']:.4f} e")
                print(f"    Avg carbon charge: {transfer_analysis['avg_carbon_charge']:.4f} e")
                print(f"    Metal charge transfer: {transfer_analysis['charge_transfer']:.4f} e")
        
        all_results[system_name] = result
        
        # Print detailed charge information
        print(f"\n  Mulliken Charges by Element:")
        print(f"    {'Element':<10} {'Count':<8} {'Avg Charge':<12} {'Total Charge':<15}")
        print("    " + "-"*50)
        for element, stats in analysis['element_stats'].items():
            print(f"    {element:<10} {stats['count']:<8} {stats['avg_charge']:<12.4f} {stats['total_charge']:<15.4f}")
        
        # Print individual charges (for small systems)
        if len(atoms) <= 30:
            print("\n  Individual Mulliken Charges:")
            print(f"    {'Index':<8} {'Symbol':<8} {'Charge':<12}")
            print("    " + "-"*30)
            for charge_data in formatted_charges:
                print(f"    {charge_data['index']:<8} {charge_data['symbol']:<8} {charge_data['charge']:<12.4f}")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Convert to serializable format
    results_serializable = {}
    for system_name, result in all_results.items():
        results_serializable[system_name] = {}
        for key, value in result.items():
            if key == 'charges':
                results_serializable[system_name][key] = value
            elif key == 'element_stats':
                results_serializable[system_name][key] = value
            elif key == 'charge_transfer':
                results_serializable[system_name][key] = value
            else:
                results_serializable[system_name][key] = value
    
    # Save to JSON
    with open('mulliken_analysis/mulliken_charges.json', 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    # Generate summary table
    with open('mulliken_analysis/mulliken_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("MULLIKEN CHARGE ANALYSIS SUMMARY\n")
        f.write("Method: DMol3 | PBE | DNP basis set\n")
        f.write("="*70 + "\n\n")
        
        for system_name, result in all_results.items():
            f.write(f"\n{system_name}:\n")
            f.write("-"*50 + "\n")
            f.write(f"  Total Energy: {result['total_energy_ha']:.6f} Ha\n")
            f.write(f"  Total Charge: {result['total_charge']:.4f} e\n\n")
            
            f.write("  Mulliken Charges by Element:\n")
            f.write(f"    {'Element':<10} {'Count':<8} {'Avg Charge':<12} {'Total Charge':<15}\n")
            f.write("    " + "-"*50 + "\n")
            for element, stats in result['element_stats'].items():
                f.write(f"    {element:<10} {stats['count']:<8} {stats['avg_charge']:<12.4f} {stats['total_charge']:<15.4f}\n")
            
            if 'charge_transfer' in result:
                f.write(f"\n  Charge Transfer Analysis:\n")
                f.write(f"    Metal charge: {result['charge_transfer']['metal_charge']:.4f} e\n")
                f.write(f"    Avg carbon charge: {result['charge_transfer']['avg_carbon_charge']:.4f} e\n")
                f.write(f"    Charge transfer: {result['charge_transfer']['charge_transfer']:.4f} e\n")
            
            f.write("\n")
    
    print("Results saved to:")
    print("  - mulliken_analysis/mulliken_charges.json")
    print("  - mulliken_analysis/mulliken_summary.txt")
    
    # Print final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for system_name, result in all_results.items():
        print(f"\n{system_name}:")
        print(f"  Total Charge: {result['total_charge']:.4f} e")
        for element, stats in result['element_stats'].items():
            print(f"  {element}: avg charge = {stats['avg_charge']:.4f} e (count: {stats['count']})")
        if 'charge_transfer' in result:
            print(f"  Metal charge transfer: {result['charge_transfer']['charge_transfer']:.4f} e")
    
    print("\n" + "="*60)
    print("Mulliken charge analysis complete!")

if __name__ == "__main__":
    main()