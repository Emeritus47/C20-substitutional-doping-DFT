"""
Density of States (DOS) and Partial Density of States (PDOS) extraction and plotting
Methodology: DFT with PBE functional, DNP basis set (DMol3), spin-polarized calculations

This script:
1. Reads optimized structures from DMol3 calculations
2. Performs single-point energy calculation with DMol3
3. Extracts DOS and PDOS data from DMol3 output
4. Plots total DOS and PDOS for analysis
"""

from ase.calculators.dmol import DMol3
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import re

# ============================================================================
# 1. DMOL3 CALCULATOR SETUP
# ============================================================================

def setup_dmol3_calculator(multiplicity=1):
    """
    Configure DMol3 calculator for DOS/PDOS calculation.
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
        # DOS/PDOS settings
        dos='yes',                           # Calculate DOS
        pdos='yes',                          # Calculate PDOS
        wavefunction='yes',                  # Save wavefunction
        orbital='yes',                       # Output orbital information
    )
    return calc

# ============================================================================
# 2. DOS/PDOS EXTRACTION FUNCTIONS
# ============================================================================

def extract_dos_from_output(output_file):
    """
    Parse DMol3 output file to extract total DOS data.
    """
    dos_data = {
        'energies': [],
        'dos': [],
        'fermi_energy': None,
    }
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Extract Fermi energy
        fermi_patterns = [
            r'Fermi energy\s*=\s*([-\d.]+)\s*Ha',
            r'Fermi energy\s*=\s*([-\d.]+)\s*eV',
            r'Fermi level\s*=\s*([-\d.]+)\s*eV',
        ]
        for pattern in fermi_patterns:
            match = re.search(pattern, content)
            if match:
                dos_data['fermi_energy'] = float(match.group(1))
                break
        
        # Extract DOS data
        # DMol3 typically outputs DOS in a table format
        dos_section = False
        energies = []
        dos_values = []
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'Total DOS' in line or 'DOS' in line and 'Energy' in line:
                dos_section = True
                continue
            if dos_section and line.strip() and not line.startswith('-'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        energy = float(parts[0])
                        dos = float(parts[1])
                        energies.append(energy)
                        dos_values.append(dos)
                    except (ValueError, IndexError):
                        continue
            if dos_section and 'Total' in line and 'DOS' in line:
                break
        
        dos_data['energies'] = energies
        dos_data['dos'] = dos_values
        
    except FileNotFoundError:
        print(f"Warning: Output file {output_file} not found")
    except Exception as e:
        print(f"Warning: Error parsing output file: {e}")
    
    return dos_data

def extract_pdos_from_output(output_file):
    """
    Parse DMol3 output file to extract PDOS data for each atom.
    """
    pdos_data = {
        'energies': [],
        'pdos': {},
        'fermi_energy': None,
    }
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Extract Fermi energy (same as DOS)
        fermi_patterns = [
            r'Fermi energy\s*=\s*([-\d.]+)\s*Ha',
            r'Fermi energy\s*=\s*([-\d.]+)\s*eV',
        ]
        for pattern in fermi_patterns:
            match = re.search(pattern, content)
            if match:
                pdos_data['fermi_energy'] = float(match.group(1))
                break
        
        # Find PDOS sections for each atom
        # Format varies by DMol3 version
        atom_pattern = r'Atom\s+(\d+)\s+([A-Z][a-z]?)\s+PDOS'
        atom_matches = re.finditer(atom_pattern, content)
        
        atom_sections = {}
        for match in atom_matches:
            atom_idx = int(match.group(1))
            atom_symbol = match.group(2)
            atom_sections[atom_idx] = {
                'symbol': atom_symbol,
                'start': match.end(),
                'data': []
            }
        
        # Extract PDOS data for each atom
        # Find energy and PDOS values
        lines = content.split('\n')
        current_atom = None
        
        for i, line in enumerate(lines):
            # Check if line starts a new atom section
            for atom_idx, section in atom_sections.items():
                if f'Atom {atom_idx} {section["symbol"]} PDOS' in line:
                    current_atom = atom_idx
                    continue
            
            # Parse PDOS data for current atom
            if current_atom is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        energy = float(parts[0])
                        pdos = float(parts[1])
                        atom_sections[current_atom]['data'].append({
                            'energy': energy,
                            'pdos': pdos
                        })
                    except (ValueError, IndexError):
                        continue
        
        # Organize data
        pdos_data['pdos'] = {}
        for atom_idx, section in atom_sections.items():
            atom_key = f"{section['symbol']}_{atom_idx}"
            pdos_data['pdos'][atom_key] = {
                'symbol': section['symbol'],
                'index': atom_idx,
                'energies': [d['energy'] for d in section['data']],
                'pdos': [d['pdos'] for d in section['data']],
            }
        
        # If individual PDOS found, extract energies from first atom
        if atom_sections:
            first_atom = list(atom_sections.values())[0]
            pdos_data['energies'] = [d['energy'] for d in first_atom['data']]
        
        # If no individual atom PDOS found, try orbital-resolved PDOS
        if not pdos_data['pdos']:
            # Look for orbital-resolved PDOS (s, p, d orbitals)
            orbital_patterns = {
                's': r's-orbital\s+PDOS',
                'p': r'p-orbital\s+PDOS',
                'd': r'd-orbital\s+PDOS',
            }
            
            for orbital, pattern in orbital_patterns.items():
                if re.search(pattern, content):
                    # Extract orbital PDOS data
                    orbital_data = []
                    in_orbital_section = False
                    
                    for line in lines:
                        if pattern in line:
                            in_orbital_section = True
                            continue
                        if in_orbital_section and line.strip() and not line.startswith('-'):
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    energy = float(parts[0])
                                    pdos = float(parts[1])
                                    orbital_data.append({
                                        'energy': energy,
                                        'pdos': pdos
                                    })
                                except (ValueError, IndexError):
                                    continue
                        if in_orbital_section and 'Total' in line:
                            break
                    
                    if orbital_data:
                        pdos_data['pdos'][f'total_{orbital}'] = {
                            'symbol': 'total',
                            'orbital': orbital,
                            'energies': [d['energy'] for d in orbital_data],
                            'pdos': [d['pdos'] for d in orbital_data],
                        }
        
    except FileNotFoundError:
        print(f"Warning: Output file {output_file} not found")
    except Exception as e:
        print(f"Warning: Error parsing output file: {e}")
    
    return pdos_data

# ============================================================================
# 3. DOS/PDOS PLOTTING FUNCTIONS
# ============================================================================

def plot_total_dos(dos_data, system_name, output_dir='dos_plots'):
    """
    Plot total DOS for a system.
    """
    if not dos_data['energies'] or not dos_data['dos']:
        print(f"Warning: No DOS data available for {system_name}")
        return
    
    # Convert to numpy arrays
    energies = np.array(dos_data['energies'])
    dos = np.array(dos_data['dos'])
    
    # Shift energies relative to Fermi level if available
    if dos_data['fermi_energy'] is not None:
        shift = dos_data['fermi_energy']
        energies_shifted = energies - shift
        label_shift = ' (E - Ef)'
    else:
        energies_shifted = energies
        label_shift = ' (eV)'
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(energies_shifted, dos, 'b-', linewidth=2, label='Total DOS')
    
    # Add vertical line at Fermi level
    if dos_data['fermi_energy'] is not None:
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(0.02, 0.98, 'Ef', transform=ax.transAxes, 
                verticalalignment='top', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel(f'Energy{label_shift}', fontsize=12)
    ax.set_ylabel('DOS (states/eV)', fontsize=12)
    ax.set_title(f'Total Density of States - {system_name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Set x-axis limits (show relevant range)
    if len(energies_shifted) > 0:
        e_min = np.min(energies_shifted)
        e_max = np.max(energies_shifted)
        ax.set_xlim(e_min, e_max)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{system_name}_total_dos.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{system_name}_total_dos.pdf', bbox_inches='tight')
    
    print(f"  Total DOS plot saved to {output_dir}/{system_name}_total_dos.png")
    plt.close()

def plot_pdos(pdos_data, system_name, output_dir='dos_plots'):
    """
    Plot PDOS for a system, showing contributions from different atoms/orbitals.
    """
    if not pdos_data['pdos']:
        print(f"Warning: No PDOS data available for {system_name}")
        return
    
    # Get energies from first PDOS entry
    first_key = list(pdos_data['pdos'].keys())[0]
    energies = np.array(pdos_data['pdos'][first_key]['energies'])
    
    # Shift energies relative to Fermi level
    if pdos_data['fermi_energy'] is not None:
        shift = pdos_data['fermi_energy']
        energies_shifted = energies - shift
        label_shift = ' (E - Ef)'
    else:
        energies_shifted = energies
        label_shift = ' (eV)'
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each PDOS entry
    colors = plt.cm.Set3(np.linspace(0, 1, len(pdos_data['pdos'])))
    
    for idx, (key, data) in enumerate(pdos_data['pdos'].items()):
        pdos = np.array(data['pdos'])
        label = f"{data.get('symbol', 'Unknown')}"
        if 'orbital' in data:
            label += f" ({data['orbital']})"
        elif 'index' in data:
            label += f" (atom {data['index']})"
        
        ax.plot(energies_shifted, pdos, color=colors[idx], linewidth=1.5, 
               label=label, alpha=0.8)
    
    # Add vertical line at Fermi level
    if pdos_data['fermi_energy'] is not None:
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(0.02, 0.98, 'Ef', transform=ax.transAxes, 
                verticalalignment='top', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel(f'Energy{label_shift}', fontsize=12)
    ax.set_ylabel('PDOS (states/eV)', fontsize=12)
    ax.set_title(f'Partial Density of States - {system_name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, ncol=2)
    
    # Set x-axis limits
    if len(energies_shifted) > 0:
        e_min = np.min(energies_shifted)
        e_max = np.max(energies_shifted)
        ax.set_xlim(e_min, e_max)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{system_name}_pdos.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{system_name}_pdos.pdf', bbox_inches='tight')
    
    print(f"  PDOS plot saved to {output_dir}/{system_name}_pdos.png")
    plt.close()

def plot_dos_and_pdos_combined(dos_data, pdos_data, system_name, output_dir='dos_plots'):
    """
    Plot total DOS and PDOS in a combined figure.
    """
    if not dos_data['energies'] or not dos_data['dos']:
        print(f"Warning: No DOS data available for {system_name}")
        return
    
    # Get energies from DOS
    energies_dos = np.array(dos_data['energies'])
    dos = np.array(dos_data['dos'])
    
    # Shift energies
    if dos_data['fermi_energy'] is not None:
        shift = dos_data['fermi_energy']
        energies_shifted = energies_dos - shift
        label_shift = ' (E - Ef)'
    else:
        energies_shifted = energies_dos
        label_shift = ' (eV)'
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot total DOS
    ax1.plot(energies_shifted, dos, 'b-', linewidth=2, label='Total DOS')
    if dos_data['fermi_energy'] is not None:
        ax1.axvline(x=0, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.text(0.02, 0.98, 'Ef', transform=ax1.transAxes, 
                verticalalignment='top', fontsize=12)
    ax1.set_xlabel(f'Energy{label_shift}', fontsize=12)
    ax1.set_ylabel('DOS (states/eV)', fontsize=12)
    ax1.set_title('Total DOS', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot PDOS
    if pdos_data['pdos']:
        colors = plt.cm.Set3(np.linspace(0, 1, len(pdos_data['pdos'])))
        
        for idx, (key, data) in enumerate(pdos_data['pdos'].items()):
            # Get energies from PDOS data
            pdos_energies = np.array(data['energies'])
            if pdos_data['fermi_energy'] is not None:
                pdos_energies_shifted = pdos_energies - shift
            else:
                pdos_energies_shifted = pdos_energies
            
            pdos = np.array(data['pdos'])
            label = f"{data.get('symbol', 'Unknown')}"
            if 'orbital' in data:
                label += f" ({data['orbital']})"
            elif 'index' in data:
                label += f" (atom {data['index']})"
            
            ax2.plot(pdos_energies_shifted, pdos, color=colors[idx], 
                    linewidth=1.5, label=label, alpha=0.8)
        
        if dos_data['fermi_energy'] is not None:
            ax2.axvline(x=0, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
            ax2.text(0.02, 0.98, 'Ef', transform=ax2.transAxes, 
                    verticalalignment='top', fontsize=12)
        
        ax2.set_xlabel(f'Energy{label_shift}', fontsize=12)
        ax2.set_ylabel('PDOS (states/eV)', fontsize=12)
        ax2.set_title('Partial DOS', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=8, ncol=2)
    else:
        ax2.text(0.5, 0.5, 'No PDOS data available', 
                transform=ax2.transAxes, ha='center', va='center')
    
    # Set common x-axis limits
    if len(energies_shifted) > 0:
        e_min = np.min(energies_shifted)
        e_max = np.max(energies_shifted)
        ax1.set_xlim(e_min, e_max)
        ax2.set_xlim(e_min, e_max)
    
    # Overall title
    fig.suptitle(f'DOS and PDOS - {system_name}', fontsize=14)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{system_name}_dos_pdos_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{system_name}_dos_pdos_combined.pdf', bbox_inches='tight')
    
    print(f"  Combined DOS/PDOS plot saved to {output_dir}/{system_name}_dos_pdos_combined.png")
    plt.close()

# ============================================================================
# 4. MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow for DOS/PDOS calculation and plotting.
    """
    print("="*60)
    print("DOS AND PDOS CALCULATION AND PLOTTING")
    print("Method: DMol3 with PBE functional, DNP basis set")
    print("="*60)
    
    # Create output directory
    os.makedirs('dos_analysis', exist_ok=True)
    os.makedirs('dos_plots', exist_ok=True)
    
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
        
        # Run single-point calculation with DOS/PDOS
        print("  Running DOS/PDOS calculation...")
        try:
            energy = atoms.get_potential_energy()
            print(f"  Total energy: {energy:.6f} Ha")
        except Exception as e:
            print(f"  Error in SCF calculation: {e}")
            continue
        
        # Extract DOS data from output file
        print("  Extracting DOS data...")
        dos_data = {}
        output_files = [
            f'{system_name}.out',
            f'{system_name}_DMol3.out',
            'DMol3.out',
            'output.out'
        ]
        
        for outfile in output_files:
            if os.path.exists(outfile):
                dos_data = extract_dos_from_output(outfile)
                if dos_data['energies']:
                    print(f"  Found DOS data in {outfile}")
                    break
        
        # Extract PDOS data
        print("  Extracting PDOS data...")
        pdos_data = {}
        for outfile in output_files:
            if os.path.exists(outfile):
                pdos_data = extract_pdos_from_output(outfile)
                if pdos_data['pdos']:
                    print(f"  Found PDOS data in {outfile}")
                    break
        
        # Store results
        result = {
            'system': system_name,
            'total_energy_ha': energy,
            'dos_data': dos_data,
            'pdos_data': pdos_data,
        }
        all_results[system_name] = result
        
        # Plot DOS
        print("  Generating plots...")
        if dos_data['energies']:
            plot_total_dos(dos_data, system_name, output_dir='dos_plots')
            
            if pdos_data['pdos']:
                plot_pdos(pdos_data, system_name, output_dir='dos_plots')
                plot_dos_and_pdos_combined(dos_data, pdos_data, system_name, output_dir='dos_plots')
            else:
                print("  Warning: No PDOS data available for plotting")
        else:
            print("  Warning: No DOS data available for plotting")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Convert to serializable format
    results_serializable = {}
    for system_name, result in all_results.items():
        results_serializable[system_name] = {
            'system': result['system'],
            'total_energy_ha': result['total_energy_ha'],
            'dos_data': {
                'energies': result['dos_data'].get('energies', []),
                'dos': result['dos_data'].get('dos', []),
                'fermi_energy': result['dos_data'].get('fermi_energy'),
            },
            'pdos_data': {
                'energies': result['pdos_data'].get('energies', []),
                'fermi_energy': result['pdos_data'].get('fermi_energy'),
                'pdos': {}
            }
        }
        
        # Add PDOS data
        for key, data in result['pdos_data'].get('pdos', {}).items():
            results_serializable[system_name]['pdos_data']['pdos'][key] = {
                'symbol': data.get('symbol', ''),
                'index': data.get('index'),
                'energies': data.get('energies', []),
                'pdos': data.get('pdos', []),
            }
    
    # Save to JSON
    with open('dos_analysis/dos_pdos_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    print("Results saved to:")
    print("  - dos_analysis/dos_pdos_results.json")
    print("  - dos_plots/ (PNG and PDF files)")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for system_name, result in all_results.items():
        print(f"\n{system_name}:")
        print(f"  Energy: {result['total_energy_ha']:.6f} Ha")
        
        if result['dos_data'].get('energies'):
            print(f"  DOS points: {len(result['dos_data']['energies'])}")
            if result['dos_data'].get('fermi_energy') is not None:
                print(f"  Fermi energy: {result['dos_data']['fermi_energy']:.4f} eV")
        
        if result['pdos_data'].get('pdos'):
            print(f"  PDOS entries: {len(result['pdos_data']['pdos'])}")
            for key in result['pdos_data']['pdos'].keys():
                print(f"    - {key}")
    
    print("\n" + "="*60)
    print("DOS/PDOS analysis complete!")
    print("Plots saved in 'dos_plots' directory")

if __name__ == "__main__":
    main()