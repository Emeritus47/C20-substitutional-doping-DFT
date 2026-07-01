"""
Global reactivity descriptors calculation for C20 and MC19 (M = Be, Mg, Ca)
Methodology: DFT with PBE functional, DNP basis set (DMol3)

This script calculates:
1. Chemical potential (μ)
2. Hardness (η)
3. Softness (S)
4. Electrophilicity (ω)

Using ionization potential (I) and electron affinity (A) from DMol3 calculations.
"""

from ase.calculators.dmol import DMol3
from ase.io import read
import numpy as np
import json
import os

# ============================================================================
# 1. DMOL3 CALCULATOR SETUP
# ============================================================================

def setup_dmol3_calculator(charge=0, multiplicity=1):
    """
    Configure DMol3 calculator for energy calculations.
    """
    calc = DMol3(
        functional='pbe',                    # PBE functional
        basis='dnp',                         # DNP basis set
        symmetry='auto',                     # Auto symmetry detection
        spin_polarization='unrestricted' if multiplicity > 1 else 'restricted',
        charge=charge,                       # System charge
        multiplicity=multiplicity,           # Spin multiplicity
        scf_density_convergence=1.0e-6,      # SCF convergence
        smearing=0.005,                      # Thermal smearing
        occupation='thermal',                # Thermal occupation
        cutoff=4.5,                          # Real-space cutoff (Angstrom)
    )
    return calc

# ============================================================================
# 2. ENERGY CALCULATIONS
# ============================================================================

def calculate_total_energy(atoms, charge=0, multiplicity=1):
    """
    Calculate total energy for a given charge state.
    """
    calc = setup_dmol3_calculator(charge, multiplicity)
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    
    try:
        energy = atoms_copy.get_potential_energy()
        return energy
    except Exception as e:
        print(f"  Error in SCF calculation for charge {charge}: {e}")
        return None

def get_ionization_potential(atoms, multiplicity=1):
    """
    Calculate ionization potential (I) = E(N-1) - E(N)
    Energy of neutral system minus energy of cationic system.
    """
    print("  Calculating ionization potential...")
    
    # Energy of neutral system (charge = 0)
    e_neutral = calculate_total_energy(atoms, charge=0, multiplicity=multiplicity)
    if e_neutral is None:
        return None
    
    # Energy of cationic system (charge = +1)
    e_cation = calculate_total_energy(atoms, charge=1, multiplicity=multiplicity)
    if e_cation is None:
        return None
    
    # I = E(N-1) - E(N)
    # Note: In Hartree, energy of cation is usually higher than neutral
    ip = e_cation - e_neutral
    ip_ev = ip * 27.2114  # Convert Ha to eV
    
    print(f"    E(neutral): {e_neutral:.6f} Ha")
    print(f"    E(cation):  {e_cation:.6f} Ha")
    print(f"    IP:         {ip:.6f} Ha ({ip_ev:.4f} eV)")
    
    return ip_ev

def get_electron_affinity(atoms, multiplicity=1):
    """
    Calculate electron affinity (A) = E(N) - E(N+1)
    Energy of neutral system minus energy of anionic system.
    """
    print("  Calculating electron affinity...")
    
    # Energy of neutral system (charge = 0)
    e_neutral = calculate_total_energy(atoms, charge=0, multiplicity=multiplicity)
    if e_neutral is None:
        return None
    
    # Energy of anionic system (charge = -1)
    e_anion = calculate_total_energy(atoms, charge=-1, multiplicity=multiplicity)
    if e_anion is None:
        return None
    
    # A = E(N) - E(N+1)
    ea = e_neutral - e_anion
    ea_ev = ea * 27.2114  # Convert Ha to eV
    
    print(f"    E(neutral): {e_neutral:.6f} Ha")
    print(f"    E(anion):   {e_anion:.6f} Ha")
    print(f"    EA:         {ea:.6f} Ha ({ea_ev:.4f} eV)")
    
    return ea_ev

# ============================================================================
# 3. GLOBAL REACTIVITY DESCRIPTORS
# ============================================================================

def calculate_reactivity_descriptors(ip, ea):
    """
    Calculate global reactivity descriptors using Koopmans' theorem approximation.
    
    Parameters:
    -----------
    ip : float
        Ionization potential (eV)
    ea : float
        Electron affinity (eV)
    
    Returns:
    --------
    dict : Contains chemical potential, hardness, softness, electrophilicity
    """
    if ip is None or ea is None:
        return None
    
    # Chemical potential (μ) = -(I + A) / 2
    mu = -(ip + ea) / 2
    
    # Hardness (η) = (I - A) / 2
    eta = (ip - ea) / 2
    
    # Softness (S) = 1 / (2 * η)
    if eta != 0:
        softness = 1 / (2 * eta)
    else:
        softness = float('inf')
    
    # Electrophilicity (ω) = μ² / (2 * η)
    if eta != 0:
        electrophilicity = (mu ** 2) / (2 * eta)
    else:
        electrophilicity = float('inf')
    
    return {
        'ionization_potential_ev': ip,
        'electron_affinity_ev': ea,
        'chemical_potential_ev': mu,
        'hardness_ev': eta,
        'softness_ev_inv': softness,
        'electrophilicity_ev': electrophilicity,
    }

def calculate_descriptors_from_energies(e_neutral, e_cation, e_anion):
    """
    Calculate descriptors directly from energy values.
    
    Parameters:
    -----------
    e_neutral : float
        Energy of neutral system (Ha)
    e_cation : float
        Energy of cationic system (Ha)
    e_anion : float
        Energy of anionic system (Ha)
    """
    # Convert to eV
    e_neutral_ev = e_neutral * 27.2114
    e_cation_ev = e_cation * 27.2114
    e_anion_ev = e_anion * 27.2114
    
    # IP = E(N-1) - E(N)
    ip = e_cation_ev - e_neutral_ev
    
    # EA = E(N) - E(N+1)
    ea = e_neutral_ev - e_anion_ev
    
    return calculate_reactivity_descriptors(ip, ea)

# ============================================================================
# 4. SPIN MULTIPLICITY SCREENING
# ============================================================================

def screen_multiplicities(atoms, system_name):
    """
    Test different spin multiplicities for reactivity calculations.
    """
    multiplicities = [1, 2, 3, 4, 5, 6, 7]
    results = {}
    
    print(f"\nScreening spin multiplicities for {system_name}...")
    
    for mult in multiplicities:
        print(f"\n  Testing multiplicity {mult}:")
        
        # Calculate IP and EA for this multiplicity
        ip = get_ionization_potential(atoms, mult)
        ea = get_electron_affinity(atoms, mult)
        
        if ip is not None and ea is not None:
            descriptors = calculate_reactivity_descriptors(ip, ea)
            if descriptors:
                results[mult] = descriptors
                print(f"\n  Multiplicity {mult} results:")
                print(f"    μ = {descriptors['chemical_potential_ev']:.4f} eV")
                print(f"    η = {descriptors['hardness_ev']:.4f} eV")
                print(f"    S = {descriptors['softness_ev_inv']:.4f} eV⁻¹")
                print(f"    ω = {descriptors['electrophilicity_ev']:.4f} eV")
        else:
            print(f"  Multiplicity {mult}: Failed to calculate IP/EA")
            results[mult] = None
    
    return results

# ============================================================================
# 5. MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow for global reactivity descriptor calculation.
    """
    print("="*70)
    print("GLOBAL REACTIVITY DESCRIPTORS CALCULATION")
    print("Method: DMol3 | PBE | DNP basis set")
    print("="*70)
    
    # Create output directory
    os.makedirs('reactivity_results', exist_ok=True)
    
    # List of structures to analyze
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
        print(f"Structure loaded: {len(atoms)} atoms")
        
        # For pure C20, use singlet
        if system_name == 'C20':
            print("\nCalculating for singlet multiplicity...")
            
            # Calculate energies
            e_neutral = calculate_total_energy(atoms, charge=0, multiplicity=1)
            e_cation = calculate_total_energy(atoms, charge=1, multiplicity=1)
            e_anion = calculate_total_energy(atoms, charge=-1, multiplicity=1)
            
            if all([e is not None for e in [e_neutral, e_cation, e_anion]]):
                descriptors = calculate_descriptors_from_energies(e_neutral, e_cation, e_anion)
                if descriptors:
                    all_results[system_name] = descriptors
                    
                    print(f"\n{system_name} Reactivity Descriptors:")
                    print(f"  Ionization Potential (I): {descriptors['ionization_potential_ev']:.4f} eV")
                    print(f"  Electron Affinity (A):    {descriptors['electron_affinity_ev']:.4f} eV")
                    print(f"  Chemical Potential (μ):   {descriptors['chemical_potential_ev']:.4f} eV")
                    print(f"  Hardness (η):             {descriptors['hardness_ev']:.4f} eV")
                    print(f"  Softness (S):             {descriptors['softness_ev_inv']:.4f} eV⁻¹")
                    print(f"  Electrophilicity (ω):     {descriptors['electrophilicity_ev']:.4f} eV")
        else:
            # For doped structures, screen different spin multiplicities
            spin_results = screen_multiplicities(atoms, system_name)
            
            # Find best multiplicity (lowest chemical potential or highest stability)
            valid_results = {k: v for k, v in spin_results.items() if v is not None}
            if valid_results:
                # Store all results
                all_results[system_name] = valid_results
                
                # Show summary for best multiplicity
                # Usually the lowest energy multiplicity is chosen
                # For now, we'll show all multiplicities
                print(f"\n{system_name} Summary by Multiplicity:")
                print(f"{'Mult':<6} {'μ (eV)':<12} {'η (eV)':<12} {'S (eV⁻¹)':<12} {'ω (eV)':<12}")
                print("-"*55)
                for mult, desc in sorted(valid_results.items()):
                    print(f"{mult:<6} {desc['chemical_potential_ev']:<12.4f} {desc['hardness_ev']:<12.4f} {desc['softness_ev_inv']:<12.4f} {desc['electrophilicity_ev']:<12.4f}")
    
    # Step: Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Convert results to serializable format
    results_serializable = {}
    for system_name, result in all_results.items():
        if isinstance(result, dict):
            if 'chemical_potential_ev' in result:  # Single result
                results_serializable[system_name] = {
                    'ionization_potential_ev': float(result['ionization_potential_ev']),
                    'electron_affinity_ev': float(result['electron_affinity_ev']),
                    'chemical_potential_ev': float(result['chemical_potential_ev']),
                    'hardness_ev': float(result['hardness_ev']),
                    'softness_ev_inv': float(result['softness_ev_inv']),
                    'electrophilicity_ev': float(result['electrophilicity_ev']),
                }
            else:  # Multiplicity results
                results_serializable[system_name] = {}
                for mult, desc in result.items():
                    if desc is not None:
                        results_serializable[system_name][str(mult)] = {
                            'ionization_potential_ev': float(desc['ionization_potential_ev']),
                            'electron_affinity_ev': float(desc['electron_affinity_ev']),
                            'chemical_potential_ev': float(desc['chemical_potential_ev']),
                            'hardness_ev': float(desc['hardness_ev']),
                            'softness_ev_inv': float(desc['softness_ev_inv']),
                            'electrophilicity_ev': float(desc['electrophilicity_ev']),
                        }
    
    # Save to JSON
    with open('reactivity_results/reactivity_descriptors.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    # Save summary table
    with open('reactivity_results/reactivity_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("GLOBAL REACTIVITY DESCRIPTORS SUMMARY\n")
        f.write("Method: DMol3 | PBE | DNP basis set\n")
        f.write("="*80 + "\n\n")
        
        for system_name, result in all_results.items():
            f.write(f"\n{system_name}\n")
            f.write("-"*40 + "\n")
            
            if isinstance(result, dict) and 'chemical_potential_ev' in result:
                f.write(f"  Ionization Potential (I): {result['ionization_potential_ev']:.4f} eV\n")
                f.write(f"  Electron Affinity (A):    {result['electron_affinity_ev']:.4f} eV\n")
                f.write(f"  Chemical Potential (μ):   {result['chemical_potential_ev']:.4f} eV\n")
                f.write(f"  Hardness (η):             {result['hardness_ev']:.4f} eV\n")
                f.write(f"  Softness (S):             {result['softness_ev_inv']:.4f} eV⁻¹\n")
                f.write(f"  Electrophilicity (ω):     {result['electrophilicity_ev']:.4f} eV\n")
            elif isinstance(result, dict):
                f.write(f"  {'Multiplicity':<12} {'μ (eV)':<12} {'η (eV)':<12} {'S (eV⁻¹)':<12} {'ω (eV)':<12}\n")
                f.write("  " + "-"*55 + "\n")
                for mult, desc in sorted(result.items()):
                    if desc is not None:
                        f.write(f"  {mult:<12} {desc['chemical_potential_ev']:<12.4f} {desc['hardness_ev']:<12.4f} {desc['softness_ev_inv']:<12.4f} {desc['electrophilicity_ev']:<12.4f}\n")
    
    print("Results saved to:")
    print("  - reactivity_results/reactivity_descriptors.json")
    print("  - reactivity_results/reactivity_summary.txt")
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'System':<15} {'μ (eV)':<12} {'η (eV)':<12} {'S (eV⁻¹)':<12} {'ω (eV)':<12}")
    print("-"*70)
    
    for system_name, result in all_results.items():
        if isinstance(result, dict) and 'chemical_potential_ev' in result:
            print(f"{system_name:<15} {result['chemical_potential_ev']:<12.4f} {result['hardness_ev']:<12.4f} {result['softness_ev_inv']:<12.4f} {result['electrophilicity_ev']:<12.4f}")
        elif isinstance(result, dict):
            # Show only the best multiplicity (lowest chemical potential)
            valid_results = {k: v for k, v in result.items() if v is not None}
            if valid_results:
                best_mult = min(valid_results.keys(), 
                              key=lambda k: valid_results[k]['chemical_potential_ev'])
                desc = valid_results[best_mult]
                print(f"{system_name:<15} {desc['chemical_potential_ev']:<12.4f} {desc['hardness_ev']:<12.4f} {desc['softness_ev_inv']:<12.4f} {desc['electrophilicity_ev']:<12.4f}")
    
    print("="*70)
    print("Reactivity descriptors calculation complete!")

if __name__ == "__main__":
    main()