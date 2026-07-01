"""
Structure optimization for C20 and alkaline earth metal doped C20 (MC19)
Methodology: DFT with PBE functional, DNP basis set (DMol3), spin-polarized calculations

This script:
1. Builds C20 (Ih symmetry) and MC19 (M = Be, Mg, Ca) structures
2. Performs DMol3 geometry optimization
3. Extracts bond lengths, angles, and energies
"""

from ase.build import molecule
from ase.calculators.dmol import DMol3
from ase.optimize import BFGS
from ase.io import read, write
import numpy as np
import json
import os

# ============================================================================
# 1. STRUCTURE GENERATION
# ============================================================================

def build_c20():
    """
    Generate C20 fullerene with Ih symmetry.
    Coordinates based on dodecahedron vertices.
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
    
    # Create ASE Atoms object
    from ase import Atoms
    atoms = Atoms('C' * 20, positions=vertices)
    return atoms

def substitute_dopant(atoms, dopant_symbol, index=0):
    """
    Substitute one C atom with alkaline earth metal (Be, Mg, Ca).
    """
    symbols = list(atoms.get_chemical_symbols())
    symbols[index] = dopant_symbol
    atoms.set_chemical_symbols(symbols)
    return atoms

# ============================================================================
# 2. DMOL3 CALCULATOR SETUP
# ============================================================================

def setup_dmol3_calculator():
    """
    Configure DMol3 calculator with parameters matching the paper.
    """
    calc = DMol3(
        functional='pbe',                    # PBE functional
        basis='dnp',                         # DNP basis set
        symmetry='auto',                     # Auto symmetry detection
        spin_polarization='unrestricted',    # Spin-polarized calculations
        charge=0,                            # Neutral system
        scf_density_convergence=1.0e-6,      # SCF convergence: 1e-6 Hartree
        # DMol3 specific settings
        smearing=0.005,                      # Thermal smearing for SCF stability
        occupation='thermal',                # Thermal occupation
        cutoff=4.5,                          # Real-space cutoff (Angstrom)
    )
    return calc

# ============================================================================
# 3. GEOMETRY OPTIMIZATION
# ============================================================================

def optimize_structure(atoms, dopant_symbol, max_steps=200):
    """
    Optimize structure using BFGS algorithm.
    Convergence: fmax = 0.01 eV/A (matching paper's 0.01 eV/A force criterion)
    """
    # Attach calculator
    calc = setup_dmol3_calculator()
    atoms.calc = calc
    
    print(f"Starting geometry optimization for {dopant_symbol}C19...")
    
    # Run optimization
    opt = BFGS(atoms, trajectory=f'{dopant_symbol}C19.traj', logfile=f'{dopant_symbol}C19.log')
    opt.run(fmax=0.01, steps=max_steps)  # 0.01 eV/A force convergence
    
    # Save final structure
    write(f'{dopant_symbol}C19_optimized.xyz', atoms)
    
    return atoms

# ============================================================================
# 4. PROPERTY EXTRACTION
# ============================================================================

def extract_bond_lengths(atoms, metal_index=0):
    """
    Extract M-C bond lengths from optimized structure.
    """
    positions = atoms.get_positions()
    metal_pos = positions[metal_index]
    
    bond_lengths = []
    for i, pos in enumerate(positions):
        if i != metal_index:
            dist = np.linalg.norm(pos - metal_pos)
            bond_lengths.append(dist)
    
    return bond_lengths

def extract_bond_angles(atoms, metal_index=0):
    """
    Extract C-M-C bond angles from optimized structure.
    """
    positions = atoms.get_positions()
    metal_pos = positions[metal_index]
    n_atoms = len(positions)
    
    angles = []
    for i in range(n_atoms):
        if i == metal_index:
            continue
        for j in range(i+1, n_atoms):
            if j == metal_index:
                continue
            vec1 = positions[i] - metal_pos
            vec2 = positions[j] - metal_pos
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            angles.append(angle)
    
    return angles

def extract_bond_lengths_cc(atoms):
    """
    Extract C-C bond lengths from optimized structure (for pure C20).
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    bond_lengths = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if symbols[i] == 'C' and symbols[j] == 'C':
                dist = np.linalg.norm(positions[i] - positions[j])
                bond_lengths.append(dist)
    
    return bond_lengths

def extract_properties(atoms, dopant_symbol, metal_index=0):
    """
    Extract all properties from optimized structure.
    """
    # Energy
    energy = atoms.get_potential_energy()
    
    # For pure C20
    if dopant_symbol == 'C':
        bond_lengths = extract_bond_lengths_cc(atoms)
        return {
            'symbol': 'C20',
            'energy': energy,
            'bond_lengths': bond_lengths,
            'avg_bond_length': np.mean(bond_lengths),
            'std_bond_length': np.std(bond_lengths),
        }
    
    # For doped structures
    bond_lengths = extract_bond_lengths(atoms, metal_index)
    bond_angles = extract_bond_angles(atoms, metal_index)
    
    return {
        'symbol': f'{dopant_symbol}C19',
        'energy': energy,
        'bond_lengths': bond_lengths,
        'bond_angles': bond_angles,
        'avg_bond_length': np.mean(bond_lengths),
        'std_bond_length': np.std(bond_lengths),
        'avg_bond_angle': np.mean(bond_angles),
        'std_bond_angle': np.std(bond_angles),
        'min_bond_length': np.min(bond_lengths),
        'max_bond_length': np.max(bond_lengths),
        'min_bond_angle': np.min(bond_angles),
        'max_bond_angle': np.max(bond_angles),
    }

# ============================================================================
# 5. SPIN MULTIPLICITY SCREENING
# ============================================================================

def screen_spin_multiplicity(atoms, dopant_symbol):
    """
    Screen different spin multiplicities as per paper Table 1.
    Returns energy for singlet, doublet, triplet, etc.
    """
    # Multiplicities to test: 1(singlet), 2(doublet), 3(triplet), etc.
    multiplicities = [1, 2, 3, 4, 5, 6, 7]
    results = {}
    
    for mult in multiplicities:
        print(f"Testing multiplicity {mult} for {dopant_symbol}C19...")
        
        # Create calculator with specific spin multiplicity
        calc = DMol3(
            functional='pbe',
            basis='dnp',
            symmetry='auto',
            spin_polarization='unrestricted' if mult > 1 else 'restricted',
            charge=0,
            multiplicity=mult,
            scf_density_convergence=1.0e-6,
            smearing=0.005,
            occupation='thermal',
            cutoff=4.5,
        )
        
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        
        try:
            energy = atoms_copy.get_potential_energy()
            results[mult] = energy
            print(f"  Multiplicity {mult}: Energy = {energy:.6f} Ha")
        except Exception as e:
            print(f"  Multiplicity {mult}: Failed - {str(e)}")
            results[mult] = None
    
    return results

# ============================================================================
# 6. MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow for DMol3 optimization of C20 and MC19 structures.
    """
    # Create output directory
    os.makedirs('dmol3_output', exist_ok=True)
    os.chdir('dmol3_output')
    
    # Step 1: Build C20
    print("="*60)
    print("GENERATING C20 FULLERENE STRUCTURE")
    print("="*60)
    c20 = build_c20()
    print(f"C20 created with {len(c20)} atoms")
    write('C20_initial.xyz', c20)
    
    # Step 2: Optimize pure C20
    print("\n" + "="*60)
    print("OPTIMIZING PURE C20")
    print("="*60)
    opt_c20 = optimize_structure(c20, 'C')
    
    # Extract C20 properties
    c20_props = extract_properties(opt_c20, 'C')
    print(f"\nC20 Results:")
    print(f"  Energy: {c20_props['energy']:.6f} Ha")
    print(f"  Avg C-C bond length: {c20_props['avg_bond_length']:.4f} Å")
    print(f"  Std C-C bond length: {c20_props['std_bond_length']:.4f} Å")
    
    # Step 3: Dopants
    dopants = ['Be', 'Mg', 'Ca']
    results = {'C20': c20_props}
    
    for dopant in dopants:
        print("\n" + "="*60)
        print(f"OPTIMIZING {dopant}C19")
        print("="*60)
        
        # Substitute dopant
        mc19 = substitute_dopant(c20.copy(), dopant)
        write(f'{dopant}C19_initial.xyz', mc19)
        
        # Screen spin multiplicities (Table 1 in paper)
        print(f"\nScreening spin multiplicities for {dopant}C19...")
        spin_results = screen_spin_multiplicity(mc19, dopant)
        
        # Find lowest energy multiplicity
        valid_results = {k: v for k, v in spin_results.items() if v is not None}
        if valid_results:
            lowest_mult = min(valid_results, key=valid_results.get)
            print(f"\nLowest energy multiplicity: {lowest_mult} (Singlet)" 
                  if lowest_mult == 1 else f"Lowest energy multiplicity: {lowest_mult}")
        
        # Optimize with singlet multiplicity (as per paper)
        print(f"\nPerforming full optimization for {dopant}C19 with singlet multiplicity...")
        opt_mc19 = optimize_structure(mc19, dopant)
        
        # Extract properties
        props = extract_properties(opt_mc19, dopant)
        results[f'{dopant}C19'] = props
        
        print(f"\n{dopant}C19 Results:")
        print(f"  Energy: {props['energy']:.6f} Ha")
        print(f"  Avg M-C bond length: {props['avg_bond_length']:.4f} Å")
        print(f"  Std M-C bond length: {props['std_bond_length']:.4f} Å")
        print(f"  Avg C-M-C angle: {props['avg_bond_angle']:.4f}°")
        print(f"  Std C-M-C angle: {props['std_bond_angle']:.4f}°")
        print(f"  Min/Max M-C bond: {props['min_bond_length']:.4f} / {props['max_bond_length']:.4f} Å")
        print(f"  Min/Max C-M-C angle: {props['min_bond_angle']:.4f} / {props['max_bond_angle']:.4f}°")
    
    # Step 4: Save all results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        results_serializable[key] = {}
        for k, v in value.items():
            if isinstance(v, np.ndarray):
                results_serializable[key][k] = v.tolist()
            elif isinstance(v, np.float64) or isinstance(v, np.float32):
                results_serializable[key][k] = float(v)
            else:
                results_serializable[key][k] = v
    
    with open('optimization_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print("Results saved to optimization_results.json")
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Structure          Energy (Ha)    Avg Bond (Å)    Avg Angle (°)")
    print("-"*60)
    for key, value in results.items():
        if key == 'C20':
            print(f"{key:15} {value['energy']:12.6f}   {value['avg_bond_length']:10.4f}       -")
        else:
            print(f"{key:15} {value['energy']:12.6f}   {value['avg_bond_length']:10.4f}   {value['avg_bond_angle']:10.4f}")
    
    print("="*60)
    print("Optimization complete!")

if __name__ == "__main__":
    main()
