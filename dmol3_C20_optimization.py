"""
DMol3 Structure Optimization for C20 and MC19 (M = Be, Mg, Ca)
Based on: "Alkaline earth metal substitutional doped C20: a density functional study"

This script uses ASE's DMol3 calculator to run DFT calculations with:
- Functional: PBE
- Basis set: DNP
- Spin-polarized: unrestricted
- Convergence: 1e-6 Ha for energy, 0.01 eV/A for forces
"""

from ase.build import molecule
from ase.calculators.dmol import DMol3
from ase.optimize import BFGS
from ase.io import write
import numpy as np

# ============================================================================
# 1. BUILD C20 FULLERENE STRUCTURE
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
        functional='pbe',                    # PBE functional [citation:6][citation:8]
        basis='dnp',                         # DNP basis set (equivalent to 6-31G**) [citation:6][citation:8]
        symmetry='auto',                     # Auto symmetry detection
        spin_polarization='unrestricted',    # Spin-polarized calculations [citation:6][citation:8]
        charge=0,                            # Neutral system
        scf_density_convergence=1.0e-6,      # SCF convergence: 1e-6 Hartree
        # Additional convergence settings for geometry optimization
        # Note: ASE's DMol3 calculator supports energy and forces [citation:6]
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
    
    # Run optimization
    opt = BFGS(atoms, trajectory=f'{dopant_symbol}C19.traj', logfile=f'{dopant_symbol}C19.log')
    opt.run(fmax=0.01, steps=max_steps)  # 0.01 eV/A force convergence [citation:6]
    
    return atoms

# ============================================================================
# 4. PROPERTY EXTRACTION
# ============================================================================

def extract_properties(atoms, dopant_symbol, metal_index=0):
    """
    Extract bond lengths, bond angles, and energy from optimized structure.
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # M-C bond lengths
    metal_pos = positions[metal_index]
    bond_lengths = []
    for i, pos in enumerate(positions):
        if i != metal_index:
            dist = np.linalg.norm(pos - metal_pos)
            bond_lengths.append(dist)
    
    # C-M-C bond angles
    angles = []
    for i in range(len(positions)):
        if i == metal_index:
            continue
        for j in range(i+1, len(positions)):
            if j == metal_index:
                continue
            vec1 = positions[i] - metal_pos
            vec2 = positions[j] - metal_pos
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            angles.append(angle)
    
    # Energy
    energy = atoms.get_potential_energy()
    
    return {
        'symbol': dopant_symbol,
        'energy': energy,
        'bond_lengths': bond_lengths,
        'bond_angles': angles,
        'avg_bond_length': np.mean(bond_lengths),
        'avg_bond_angle': np.mean(angles),
    }

# ============================================================================
# 5. MAIN WORKFLOW
# ============================================================================

def main():
    # Build C20
    print("Building C20 fullerene...")
    c20 = build_c20()
    print(f"C20 has {len(c20)} atoms")
    
    # Dopants from the paper
    dopants = ['Be', 'Mg', 'Ca']
    results = {}
    
    for dopant in dopants:
        print(f"\n{'='*50}")
        print(f"Optimizing {dopant}C19...")
        print(f"{'='*50}")
        
        # Substitute dopant
        mc19 = substitute_dopant(c20.copy(), dopant)
        
        # Optimize
        opt_atoms = optimize_structure(mc19, dopant)
        
        # Extract properties
        props = extract_properties(opt_atoms, dopant)
        results[dopant] = props
        
        # Save structure
        write(f'{dopant}C19.xyz', opt_atoms)
        
        print(f"\n{dopant}C19 Results:")
        print(f"  Energy: {props['energy']:.6f} Ha")
        print(f"  Avg M-C bond length: {props['avg_bond_length']:.3f} Å")
        print(f"  Avg C-M-C angle: {props['avg_bond_angle']:.3f}°")
    
    # Compare with paper Table 2
    print("\n" + "="*50)
    print("Comparison with Paper (Table 2):")
    print("="*50)
    for dopant, props in results.items():
        print(f"\n{dopant}C19:")
        print(f"  M-C bond length: {props['avg_bond_length']:.3f} Å")
        print(f"  C-M-C angle: {props['avg_bond_angle']:.3f}°")
        
        # Paper values for reference
        paper_values = {
            'Be': {'bond': 1.75, 'angle': 42.607},
            'Mg': {'bond': 2.12, 'angle': 49.657},
            'Ca': {'bond': 2.39, 'angle': 55.492},
        }
        if dopant in paper_values:
            print(f"  Paper values: bond={paper_values[dopant]['bond']} Å, angle={paper_values[dopant]['angle']}°")

if __name__ == "__main__":
    main()