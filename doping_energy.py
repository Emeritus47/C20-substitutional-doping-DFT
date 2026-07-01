"""
Doping energy calculation for MC19 (M = Be, Mg, Ca)
Methodology: DFT with PBE functional, DNP basis set (DMol3), spin-polarized calculations

This script:
1. Reads optimized structures from DMol3 calculations
2. Performs single-point energy calculation with DMol3
3. Calculates doping energy for Be, Mg, Ca substitutional doping
4. Analyzes thermodynamic stability of doped structures
"""

from ase.calculators.dmol import DMol3
from ase.io import read
from ase.build import molecule
import numpy as np
import json
import os

# ============================================================================
# 1. DMOL3 CALCULATOR SETUP
# ============================================================================

def setup_dmol3_calculator(multiplicity=1):
    """
    Configure DMol3 calculator for single-point energy calculations.
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
    )
    return calc

# ============================================================================
# 2. ENERGY CALCULATION FUNCTIONS
# ============================================================================

def calculate_energy(atoms, system_name, multiplicity=1):
    """
    Calculate total energy for a given structure using DMol3.
    """
    print(f"  Calculating energy for {system_name}...")
    
    # Setup calculator
    calc = setup_dmol3_calculator(multiplicity)
    atoms.calc = calc
    
    try:
        energy = atoms.get_potential_energy()
        print(f"    Energy: {energy:.6f} Ha")
        return energy
    except Exception as e:
        print(f"    Error in SCF calculation: {e}")
        return None

def calculate_atomic_energy(symbol, multiplicity=1):
    """
    Calculate energy of isolated atom using DMol3.
    """
    print(f"  Calculating energy for isolated {symbol} atom...")
    
    # Create single atom
    atoms = molecule(symbol, vacuum=10.0)
    
    # Setup calculator
    calc = setup_dmol3_calculator(multiplicity)
    atoms.calc = calc
    
    try:
        energy = atoms.get_potential_energy()
        print(f"    {symbol} atom energy: {energy:.6f} Ha")
        return energy
    except Exception as e:
        print(f"    Error in SCF calculation for {symbol} atom: {e}")
        return None

# ============================================================================
# 3. DOPING ENERGY CALCULATION
# ============================================================================

def calculate_doping_energy(ec20, emc19, e_metal, e_carbon):
    """
    Calculate doping energy using the formula:
    E_dop = E_MC19 - (E_C20 + E_M) + E_C
    
    Where:
    - E_MC19: Total energy of doped cluster
    - E_C20: Total energy of pure C20
    - E_M: Total energy of isolated metal atom
    - E_C: Total energy of isolated carbon atom
    
    Positive doping energy indicates thermodynamic stability.
    """
    if None in [ec20, emc19, e_metal, e_carbon]:
        return None
    
    doping_energy = emc19 - (ec20 + e_metal) + e_carbon
    return doping_energy

# ============================================================================
# 4. STRUCTURE GENERATION FUNCTIONS
# ============================================================================

def build_c20():
    """
    Generate C20 fullerene with Ih symmetry for isolated energy calculation.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 20 vertices of a dodecahedron
    vertices = []
    for signs in [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
                  (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]:
        for perm in [(0,1,2), (1,2,0), (2,0,1)]:
            vals = [0, 1/phi, phi]
            x = signs[0] * vals[perm[0]]
            y = signs[1] * vals[perm[1]]
            z = signs[2] * vals[perm[2]]
            vertices.append([x, y, z])
    
    # Scale to ~1.45 Å bond length
    scale = 1.45 / 1.5
    vertices = np.array(vertices) * scale
    
    from ase import Atoms
    atoms = Atoms('C' * 20, positions=vertices)
    return atoms

# ============================================================================
# 5. MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow for doping energy calculation.
    """
    print("="*60)
    print("DOPING ENERGY CALCULATION")
    print("Method: DMol3 with PBE functional, DNP basis set")
    print("Formula: E_dop = E_MC19 - (E_C20 + E_M) + E_C")
    print("="*60)
    
    # Create output directory
    os.makedirs('doping_energy_results', exist_ok=True)
    
    # Step 1: Calculate isolated atom energies
    print("\n" + "="*60)
    print("CALCULATING ISOLATED ATOM ENERGIES")
    print("="*60)
    
    # Carbon atom
    e_carbon = calculate_atomic_energy('C', multiplicity=1)
    
    # Metal atoms
    metal_elements = ['Be', 'Mg', 'Ca']
    metal_energies = {}
    
    for metal in metal_elements:
        # Determine multiplicity for isolated metal atom (usually singlet)
        # Be: 1s2 2s2 -> singlet
        # Mg: [Ne] 3s2 -> singlet
        # Ca: [Ar] 4s2 -> singlet
        multiplicity = 1
        energy = calculate_atomic_energy(metal, multiplicity)
        metal_energies[metal] = energy
    
    # Step 2: Calculate C20 energy
    print("\n" + "="*60)
    print("CALCULATING C20 ENERGY")
    print("="*60)
    
    c20 = build_c20()
    c20_energy = calculate_energy(c20, 'C20', multiplicity=1)
    
    # Step 3: Calculate MC19 energies
    print("\n" + "="*60)
    print("CALCULATING MC19 ENERGIES")
    print("="*60)
    
    mc19_structures = {
        'BeC19': 'BeC19_optimized.xyz',
        'MgC19': 'MgC19_optimized.xyz',
        'CaC19': 'CaC19_optimized.xyz',
    }
    
    mc19_energies = {}
    doping_energies = {}
    
    for system_name, xyz_file in mc19_structures.items():
        print(f"\nProcessing {system_name}...")
        
        if not os.path.exists(xyz_file):
            print(f"  Warning: {xyz_file} not found. Skipping {system_name}")
            continue
        
        # Read optimized structure
        atoms = read(xyz_file)
        
        # Determine metal symbol
        metal_symbol = system_name.replace('C19', '')
        
        # Calculate energy (multiplicity from optimization)
        energy = calculate_energy(atoms, system_name, multiplicity=1)
        mc19_energies[system_name] = energy
    
    # Step 4: Calculate doping energies
    print("\n" + "="*60)
    print("CALCULATING DOPING ENERGIES")
    print("="*60)
    
    for metal, e_metal in metal_energies.items():
        system_name = f"{metal}C19"
        if system_name in mc19_energies:
            e_mc19 = mc19_energies[system_name]
            e_dop = calculate_doping_energy(
                c20_energy, e_mc19, e_metal, e_carbon
            )
            if e_dop is not None:
                doping_energies[system_name] = {
                    'doping_energy_ha': e_dop,
                    'doping_energy_ev': e_dop * 27.2114,  # Convert to eV
                    'e_mc19_ha': e_mc19,
                    'e_c20_ha': c20_energy,
                    'e_metal_ha': e_metal,
                    'e_carbon_ha': e_carbon,
                }
                print(f"\n{system_name}:")
                print(f"  E_MC19 = {e_mc19:.6f} Ha")
                print(f"  E_C20 = {c20_energy:.6f} Ha")
                print(f"  E_M = {e_metal:.6f} Ha")
                print(f"  E_C = {e_carbon:.6f} Ha")
                print(f"  E_dop = {e_dop:.6f} Ha")
                print(f"  E_dop = {e_dop * 27.2114:.4f} eV")
    
    # Step 5: Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Prepare data for JSON
    results = {
        'isolated_atoms': {
            'C': {'energy_ha': e_carbon, 'energy_ev': e_carbon * 27.2114 if e_carbon else None},
        },
        'c20': {
            'energy_ha': c20_energy,
            'energy_ev': c20_energy * 27.2114 if c20_energy else None,
        },
        'doped_structures': {},
        'doping_energies': {},
    }
    
    # Add metal energies
    for metal, energy in metal_energies.items():
        results['isolated_atoms'][metal] = {
            'energy_ha': energy,
            'energy_ev': energy * 27.2114 if energy else None,
        }
    
    # Add MC19 energies
    for system_name, energy in mc19_energies.items():
        results['doped_structures'][system_name] = {
            'energy_ha': energy,
            'energy_ev': energy * 27.2114 if energy else None,
        }
    
    # Add doping energies
    for system_name, dop_data in doping_energies.items():
        results['doping_energies'][system_name] = {
            'doping_energy_ha': dop_data['doping_energy_ha'],
            'doping_energy_ev': dop_data['doping_energy_ev'],
            'e_mc19_ha': dop_data['e_mc19_ha'],
            'e_c20_ha': dop_data['e_c20_ha'],
            'e_metal_ha': dop_data['e_metal_ha'],
            'e_carbon_ha': dop_data['e_carbon_ha'],
        }
    
    # Save to JSON
    with open('doping_energy_results/doping_energies.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate summary report
    with open('doping_energy_results/doping_energy_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("DOPING ENERGY CALCULATION SUMMARY\n")
        f.write("Method: DMol3 | PBE | DNP basis set\n")
        f.write("Formula: E_dop = E_MC19 - (E_C20 + E_M) + E_C\n")
        f.write("="*70 + "\n\n")
        
        f.write("Isolated Atom Energies:\n")
        f.write("-"*50 + "\n")
        for element, data in results['isolated_atoms'].items():
            if data['energy_ha'] is not None:
                f.write(f"  {element}: {data['energy_ha']:.6f} Ha ({data['energy_ev']:.6f} eV)\n")
        
        f.write(f"\nC20 Energy:\n")
        f.write(f"  {results['c20']['energy_ha']:.6f} Ha ({results['c20']['energy_ev']:.6f} eV)\n")
        
        f.write("\nMC19 Energies:\n")
        f.write("-"*50 + "\n")
        for system_name, data in results['doped_structures'].items():
            if data['energy_ha'] is not None:
                f.write(f"  {system_name}: {data['energy_ha']:.6f} Ha ({data['energy_ev']:.6f} eV)\n")
        
        f.write("\nDoping Energies:\n")
        f.write("-"*50 + "\n")
        f.write(f"{'System':<12} {'E_dop (Ha)':<15} {'E_dop (eV)':<15} {'Stability'}\n")
        f.write("-"*50 + "\n")
        
        for system_name, dop_data in results['doping_energies'].items():
            stability = "Thermodynamically stable" if dop_data['doping_energy_ha'] > 0 else "Thermodynamically unstable"
            f.write(f"{system_name:<12} {dop_data['doping_energy_ha']:<15.6f} {dop_data['doping_energy_ev']:<15.4f} {stability}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Doping Energy Details:\n")
        f.write("="*70 + "\n\n")
        
        for system_name, dop_data in results['doping_energies'].items():
            f.write(f"{system_name}:\n")
            f.write(f"  E_MC19 = {dop_data['e_mc19_ha']:.6f} Ha\n")
            f.write(f"  E_C20 = {dop_data['e_c20_ha']:.6f} Ha\n")
            f.write(f"  E_M = {dop_data['e_metal_ha']:.6f} Ha\n")
            f.write(f"  E_C = {dop_data['e_carbon_ha']:.6f} Ha\n")
            f.write(f"  E_dop = {dop_data['doping_energy_ha']:.6f} Ha ({dop_data['doping_energy_ev']:.4f} eV)\n\n")
    
    print("Results saved to:")
    print("  - doping_energy_results/doping_energies.json")
    print("  - doping_energy_results/doping_energy_summary.txt")
    
    # Print final summary
    print("\n" + "="*60)
    print("DOPING ENERGY SUMMARY")
    print("="*60)
    print(f"{'System':<12} {'Doping Energy (Ha)':<20} {'Doping Energy (eV)':<20} {'Stability'}")
    print("-"*70)
    
    for system_name, dop_data in results['doping_energies'].items():
        stability = "Stable" if dop_data['doping_energy_ha'] > 0 else "Unstable"
        print(f"{system_name:<12} {dop_data['doping_energy_ha']:<20.6f} {dop_data['doping_energy_ev']:<20.4f} {stability}")
    
    print("\n" + "="*60)
    print("Doping energy calculation complete!")

if __name__ == "__main__":
    main()