import numpy as np
import sys
import os
import glob
import datetime

def read_vasp(filename):
    """Read a VASP file and extract coordinates and element types."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Read the scaling factor and lattice vectors
        scaling = float(lines[1].strip())
        lattice = np.zeros((3, 3))
        for i in range(3):
            lattice[i] = np.array([float(x) for x in lines[i+2].split()])
        lattice *= scaling
        
        # Read element types and counts
        elements = lines[5].split()
        element_counts = [int(x) for x in lines[6].split()]
        
        # Create a list of element labels for each atom
        atom_types = []
        for elem, count in zip(elements, element_counts):
            atom_types.extend([elem] * count)
        
        total_atoms = sum(element_counts)
        
        # Read coordinates starting from line 8 (assuming Cartesian coordinates)
        coords = np.zeros((total_atoms, 3))
        for i in range(total_atoms):
            coords[i] = np.array([float(x) for x in lines[i+8].split()[:3]])
        
        return coords, atom_types, lattice, True
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        return None, None, None, False

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points (p2 is the vertex)."""
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize the vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate the angle using the dot product
    dot_product = np.dot(v1_norm, v2_norm)
    # Clip to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def process_vasp_file(filename, bond_cutoff=3.5, output_file=None):
    # Read the VASP file
    coords, atom_types, lattice, success = read_vasp(filename)
    
    if not success:
        return
    
    # Get indices of Pb and I atoms
    pb_indices = [i for i, atom in enumerate(atom_types) if atom == 'Pb']
    i_indices = [i for i, atom in enumerate(atom_types) if atom == 'I']
    
    # Print header information
    print(f"\nVASP file: {filename}")
    print(f"Number of Pb atoms: {len(pb_indices)}")
    print(f"Number of I atoms: {len(i_indices)}")
    
    if not pb_indices or not i_indices:
        print("No Pb or I atoms found in this file. Skipping.")
        return
    
    # Print angle information to console
    print("\nPb-I-Pb angles:")
    print("-" * 60)
    print(f"{'Pb 1':<10}{'I (center)':<15}{'Pb 2':<10}{'Angle (degrees)':<15}")
    print("-" * 60)
    
    # Calculate Pb-I-Pb angles
    pb_i_pb_angles = []
    
    # For each I atom
    for i_idx in i_indices:
        # Find Pb atoms bonded to this I atom
        bonded_pb = []
        for pb_idx in pb_indices:
            distance = np.linalg.norm(coords[pb_idx] - coords[i_idx])
            if distance < bond_cutoff:
                bonded_pb.append(pb_idx)
        
        # Calculate angles between all pairs of bonded Pb atoms
        for pb1_idx_idx in range(len(bonded_pb)):
            for pb2_idx_idx in range(pb1_idx_idx + 1, len(bonded_pb)):
                pb1_idx = bonded_pb[pb1_idx_idx]
                pb2_idx = bonded_pb[pb2_idx_idx]
                
                angle = calculate_angle(coords[pb1_idx], coords[i_idx], coords[pb2_idx])
                pb_i_pb_angles.append((pb1_idx, i_idx, pb2_idx, angle))
                
                # Print result to console
                print(f"Pb{pb1_idx+1:<7} I{i_idx+1:<13} Pb{pb2_idx+1:<7} {angle:<15.2f}")
    
    # Write data to output file if provided
    if output_file and pb_i_pb_angles:
        with open(output_file, 'a') as f:
            # Write file information header
            f.write(f"# Results for file: {os.path.basename(filename)}\n")
            f.write(f"# Analyzed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Bond cutoff distance: {bond_cutoff} Angstroms\n")
            f.write(f"# Number of Pb atoms: {len(pb_indices)}\n")
            f.write(f"# Number of I atoms: {len(i_indices)}\n")
            f.write(f"# Number of Pb-I-Pb angles: {len(pb_i_pb_angles)}\n\n")
            
            # Write column headers with fixed width for easy reading
            f.write("{:<10} {:<10} {:<10} {:<15}\n".format("Pb1_idx", "I_idx", "Pb2_idx", "Angle(deg)"))
            f.write("-" * 50 + "\n")
            
            # Write angle data
            for pb1_idx, i_idx, pb2_idx, angle in pb_i_pb_angles:
                f.write("{:<10} {:<10} {:<10} {:<15.2f}\n".format(
                    f"Pb{pb1_idx+1}", f"I{i_idx+1}", f"Pb{pb2_idx+1}", angle))
            
            # Write summary statistics
            if pb_i_pb_angles:
                angles = [angle[3] for angle in pb_i_pb_angles]
                f.write("\n# Summary Statistics:\n")
                f.write(f"# Average angle: {np.mean(angles):.2f}°\n")
                f.write(f"# Minimum angle: {np.min(angles):.2f}°\n")
                f.write(f"# Maximum angle: {np.max(angles):.2f}°\n")
                f.write(f"# Standard deviation: {np.std(angles):.2f}°\n")
                f.write("\n" + "="*50 + "\n\n")  # Section separator for multiple files
    
    # Print summary statistics to console
    if pb_i_pb_angles:
        angles = [angle[3] for angle in pb_i_pb_angles]
        print("\nSummary Statistics:")
        print(f"Number of Pb-I-Pb angles: {len(angles)}")
        print(f"Average Pb-I-Pb angle: {np.mean(angles):.2f}°")
        print(f"Minimum Pb-I-Pb angle: {np.min(angles):.2f}°")
        print(f"Maximum Pb-I-Pb angle: {np.max(angles):.2f}°")
        print(f"Standard deviation: {np.std(angles):.2f}°")
    else:
        print("\nNo Pb-I-Pb angles found. Try increasing the bond cutoff distance.")

def main():
    print("VASP Pb-I-Pb Angle Calculator")
    print("=" * 40)
    
    # Get input from user
    while True:
        filepath = input("Enter the path to your VASP file or directory: ").strip().strip('"\'')
        
        # Check if the path is valid
        if not os.path.exists(filepath):
            print(f"Error: The path '{filepath}' does not exist. Please enter a valid path.")
            continue
        
        # Check if bond cutoff should be customized
        try:
            bond_cutoff = float(input("Enter bond cutoff distance in Angstroms (default: 3.5): ") or 3.5)
        except ValueError:
            print("Invalid input. Using default bond cutoff of 3.5 Angstroms.")
            bond_cutoff = 3.5
        
        # Set up output file
        output_filename = input("Enter output file name (default: Pb_I_Pb_angles.txt): ") or "Pb_I_Pb_angles.txt"
        if not output_filename.endswith('.txt'):
            output_filename += '.txt'
        
        # Create a fresh output file with a header
        with open(output_filename, 'w') as f:
            f.write("# VASP Pb-I-Pb Angle Analysis Results\n")
            f.write(f"# Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Bond cutoff distance: {bond_cutoff} Angstroms\n\n")
        
        print(f"Results will be saved to: {output_filename}")
        
        # Process file or directory
        if os.path.isdir(filepath):
            vasp_files = []
            
            # Look for common VASP filenames
            for pattern in ['POSCAR*', 'CONTCAR*', '*.vasp']:
                vasp_files.extend(glob.glob(os.path.join(filepath, pattern)))
            
            if not vasp_files:
                print(f"No VASP files found in directory '{filepath}'")
                print("Please specify the exact filename:")
                filename = input("Enter the filename within this directory: ")
                full_path = os.path.join(filepath, filename)
                if os.path.isfile(full_path):
                    process_vasp_file(full_path, bond_cutoff, output_filename)
                else:
                    print(f"Error: File '{full_path}' not found.")
            else:
                print(f"Found {len(vasp_files)} VASP files in the directory.")
                for vasp_file in vasp_files:
                    process_vasp_file(vasp_file, bond_cutoff, output_filename)
        else:
            process_vasp_file(filepath, bond_cutoff, output_filename)
        
        print(f"\nResults have been saved to {output_filename}")
        
        # Ask if the user wants to analyze another file
        if input("\nDo you want to analyze another file? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print("\nThank you for using the VASP Pb-I-Pb Angle Calculator!")