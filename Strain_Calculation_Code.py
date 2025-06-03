import argparse
from pymatgen.core import Structure
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import os

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate lattice strain gradients between two structures (Pb and I atoms only).')
    parser.add_argument('--reference', type=str, required=True, help='Path to reference CIF file')
    parser.add_argument('--deformed', type=str, required=True, help='Path to deformed CIF file')
    parser.add_argument('--cutoff', type=float, default=5.0, help='Cutoff distance (Å) for neighbor search')
    parser.add_argument('--min_neighbors', type=int, default=4, help='Minimum number of neighbors for strain calculation')
    parser.add_argument('--output_dir', type=str, default='strain_analysis', help='Output directory for results')
    parser.add_argument('--smoothing', type=float, default=0.5, help='Gaussian smoothing sigma for gradient calculation')
    return parser.parse_args()

def filter_structure_by_elements(structure, elements=["Pb", "I"]):
    """
    Filter a structure to include only specific elements.
    
    Args:
        structure: pymatgen Structure object
        elements: List of element symbols to include
        
    Returns:
        filtered_structure: New structure with only specified elements
        indices_map: Dictionary mapping indices in filtered structure to original structure
    """
    indices_to_keep = []
    for i, site in enumerate(structure):
        if site.species_string in elements:
            indices_to_keep.append(i)
    
    # Create a new structure with only the specified elements
    filtered_structure = Structure(
        lattice=structure.lattice,
        species=[structure[i].species for i in indices_to_keep],
        coords=[structure[i].frac_coords for i in indices_to_keep],
        coords_are_cartesian=False
    )
    
    # Create a mapping from new indices to original indices
    indices_map = {new_idx: old_idx for new_idx, old_idx in enumerate(indices_to_keep)}
    
    return filtered_structure, indices_map

def compute_local_strain(ref_pos, def_pos, neighbors_ref, neighbors_def):
    """
    Compute the local strain tensor using the method of least squares.
    
    Args:
        ref_pos: Position of the central atom in the reference structure
        def_pos: Position of the central atom in the deformed structure
        neighbors_ref: Positions of neighboring atoms in the reference structure
        neighbors_def: Positions of neighboring atoms in the deformed structure
        
    Returns:
        E: The local strain tensor (3x3)
        F: The deformation gradient tensor (3x3)
    """
    A, B = [], []
    
    # Create displacement vectors
    for r_n, d_n in zip(neighbors_ref, neighbors_def):
        A.append(r_n - ref_pos)
        B.append(d_n - def_pos)
    
    A = np.array(A)
    B = np.array(B)
    
    try:
        # Solve for deformation gradient tensor F: B = F·A
        F = np.linalg.lstsq(A, B, rcond=None)[0].T
        
        # Calculate Green-Lagrangian strain tensor
        E = 0.5 * (F.T @ F - np.identity(3))
        return E, F
    except np.linalg.LinAlgError:
        return np.zeros((3, 3)), np.identity(3)

def calculate_strain_measures(E):
    """
    Calculate various strain measures from the strain tensor.
    
    Args:
        E: The strain tensor (3x3)
        
    Returns:
        dict: Dictionary containing different strain measures
    """
    # Eigenvalues (principal strains)
    eigenvals = np.linalg.eigvals(E)
    
    # Various strain metrics
    volumetric_strain = np.trace(E)
    von_mises_strain = np.sqrt(2/3 * np.sum((eigenvals - np.mean(eigenvals))**2))
    max_shear_strain = (max(eigenvals) - min(eigenvals)) / 2
    
    return {
        'volumetric': volumetric_strain,
        'von_mises': von_mises_strain,
        'max_shear': max_shear_strain,
        'principal_strains': eigenvals,
        'tensor': E
    }

def calculate_strain_gradient(coords, strains, smoothing_sigma=0.5):
    """
    Calculate the spatial gradient of strain values.
    
    Args:
        coords: Atomic coordinates
        strains: Strain values at each coordinate
        smoothing_sigma: Sigma for Gaussian smoothing
        
    Returns:
        magnitude: Magnitude of strain gradient at each point
        gradient_vectors: (x,y,z) components of gradient at each point
    """
    # Convert to 3D grid for gradient calculation
    x_min, y_min, z_min = np.min(coords, axis=0)
    x_max, y_max, z_max = np.max(coords, axis=0)
    
    # Set grid resolution
    resolution = 0.5  # Å
    
    # Create regular grid
    x = np.arange(x_min-1, x_max+1, resolution)
    y = np.arange(y_min-1, y_max+1, resolution)
    z = np.arange(z_min-1, z_max+1, resolution)
    
    # Initialize grid
    grid = np.zeros((len(x), len(y), len(z)))
    
    # Map strain values to grid using nearest neighbor interpolation
    for i, (coord, strain) in enumerate(zip(coords, strains)):
        # Find closest grid point
        idx_x = int((coord[0] - x_min + 1) / resolution)
        idx_y = int((coord[1] - y_min + 1) / resolution)
        idx_z = int((coord[2] - z_min + 1) / resolution)
        
        # Ensure indices are within bounds
        idx_x = max(0, min(idx_x, len(x)-1))
        idx_y = max(0, min(idx_y, len(y)-1))
        idx_z = max(0, min(idx_z, len(z)-1))
        
        grid[idx_x, idx_y, idx_z] = strain
    
    # Apply Gaussian smoothing to the strain field
    smoothed_grid = gaussian_filter(grid, sigma=smoothing_sigma)
    
    # Calculate gradients
    grad_x, grad_y, grad_z = np.gradient(smoothed_grid, resolution)
    
    # Map gradients back to original atomic positions
    gradient_vectors = []
    gradient_magnitudes = []
    
    for coord in coords:
        idx_x = int((coord[0] - x_min + 1) / resolution)
        idx_y = int((coord[1] - y_min + 1) / resolution)
        idx_z = int((coord[2] - z_min + 1) / resolution)
        
        # Ensure indices are within bounds
        idx_x = max(0, min(idx_x, len(x)-1))
        idx_y = max(0, min(idx_y, len(y)-1))
        idx_z = max(0, min(idx_z, len(z)-1))
        
        # Get gradient at this position
        grad_vector = np.array([grad_x[idx_x, idx_y, idx_z], 
                               grad_y[idx_x, idx_y, idx_z], 
                               grad_z[idx_x, idx_y, idx_z]])
        
        gradient_vectors.append(grad_vector)
        gradient_magnitudes.append(np.linalg.norm(grad_vector))
    
    return np.array(gradient_magnitudes), np.array(gradient_vectors)

def main():
    """Main function to calculate and visualize strain gradients."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Loading reference structure: {args.reference}")
    print(f"Loading deformed structure: {args.deformed}")
    
    # Load structures
    try:
        ref_full = Structure.from_file(args.reference)
        def_full = Structure.from_file(args.deformed)
    except Exception as e:
        print(f"Error loading structures: {str(e)}")
        return
    
    # Filter structures to include only Pb and I atoms
    print("Filtering structures to include only Pb and I atoms...")
    ref, ref_indices_map = filter_structure_by_elements(ref_full, elements=["Pb", "I"])
    def_struct, def_indices_map = filter_structure_by_elements(def_full, elements=["Pb", "I"])
    
    # Count atoms by element
    ref_elements = [site.species_string for site in ref]
    print(f"Reference structure filtered: {len(ref)} atoms")
    print(f"  Pb atoms: {ref_elements.count('Pb')}")
    print(f"  I atoms: {ref_elements.count('I')}")
    
    # Check consistency of filtered structures
    if len(ref) != len(def_struct):
        print(f"Warning: Number of filtered atoms do not match! Reference: {len(ref)}, Deformed: {len(def_struct)}")
        print("Continuing with calculation, but results may be unreliable.")
    
    # Build KD trees for neighbors (using filtered structures)
    ref_tree = KDTree(ref.cart_coords)
    def_tree = KDTree(def_struct.cart_coords)
    
    # Initialize arrays for results
    strain_tensors = []
    strain_measures = []
    
    print(f"Calculating local strain for {len(ref)} atoms (Pb and I only)...")
    
    # Loop over atoms in filtered structures
    for i, (r_site, d_site) in enumerate(zip(ref, def_struct)):
        # Find neighbors within cutoff
        idx_ref = ref_tree.query_ball_point(r_site.coords, args.cutoff)
        idx_def = def_tree.query_ball_point(d_site.coords, args.cutoff)
        
        # Find common neighbors
        common = list(set(idx_ref) & set(idx_def))
        
        # Skip if not enough neighbors
        if len(common) < args.min_neighbors:
            strain_tensors.append(np.zeros((3, 3)))
            strain_measures.append({
                'volumetric': 0.0,
                'von_mises': 0.0,
                'max_shear': 0.0,
                'principal_strains': np.zeros(3),
                'tensor': np.zeros((3, 3))
            })
            continue
        
        # Get coordinates of neighbors
        ref_neighbors = np.array([ref[j].coords for j in common])
        def_neighbors = np.array([def_struct[j].coords for j in common])
        
        # Calculate strain tensor
        E, F = compute_local_strain(r_site.coords, d_site.coords, ref_neighbors, def_neighbors)
        strain_tensors.append(E)
        
        # Calculate strain measures
        measures = calculate_strain_measures(E)
        strain_measures.append(measures)
    
    # Extract volumetric strain for each atom
    volumetric_strains = [m['volumetric'] for m in strain_measures]
    von_mises_strains = [m['von_mises'] for m in strain_measures]
    max_shear_strains = [m['max_shear'] for m in strain_measures]
    
    # Calculate strain gradients
    coords = np.array([site.coords for site in def_struct])
    
    # Create element type array for visualization
    element_types = np.array([1 if site.species_string == "Pb" else 0 for site in def_struct])
    
    print("Calculating strain gradients...")
    
    # Calculate gradients for different strain measures
    vol_grad_mag, vol_grad_vec = calculate_strain_gradient(
        coords, volumetric_strains, args.smoothing)
    
    vm_grad_mag, vm_grad_vec = calculate_strain_gradient(
        coords, von_mises_strains, args.smoothing)
    
    shear_grad_mag, shear_grad_vec = calculate_strain_gradient(
        coords, max_shear_strains, args.smoothing)
    
    # Create visualization
    print("Generating visualizations...")
    
    # Define a consistent colormap
    strain_cmap = plt.cm.coolwarm
    gradient_cmap = plt.cm.viridis
    
    # Plot volumetric strain
    plt.figure(figsize=(14, 10))
    
    # Create separate plots for Pb and I atoms
    pb_indices = np.where(element_types == 1)[0]
    i_indices = np.where(element_types == 0)[0]
    
    # Plot 2D projections - Volumetric Strain
    plt.subplot(2, 3, 1)
    sc_pb = plt.scatter(coords[pb_indices, 0], coords[pb_indices, 1], c=np.array(volumetric_strains)[pb_indices], 
                      cmap=strain_cmap, s=50, vmin=-0.05, vmax=0.05, marker='s', label='Pb')
    sc_i = plt.scatter(coords[i_indices, 0], coords[i_indices, 1], c=np.array(volumetric_strains)[i_indices], 
                     cmap=strain_cmap, s=35, vmin=-0.05, vmax=0.05, marker='o', label='I')
    plt.colorbar(sc_pb, label="Volumetric Strain (Tr(E))")
    plt.xlabel("x (Å)")
    plt.ylabel("y (Å)")
    plt.title("Volumetric Strain (xy projection)")
    plt.legend()
    
    plt.subplot(2, 3, 2)
    sc_pb = plt.scatter(coords[pb_indices, 0], coords[pb_indices, 2], c=np.array(volumetric_strains)[pb_indices], 
                      cmap=strain_cmap, s=50, vmin=-0.05, vmax=0.05, marker='s')
    sc_i = plt.scatter(coords[i_indices, 0], coords[i_indices, 2], c=np.array(volumetric_strains)[i_indices], 
                     cmap=strain_cmap, s=35, vmin=-0.05, vmax=0.05, marker='o')
    plt.colorbar(sc_pb, label="Volumetric Strain (Tr(E))")
    plt.xlabel("x (Å)")
    plt.ylabel("z (Å)")
    plt.title("Volumetric Strain (xz projection)")
    
    plt.subplot(2, 3, 3)
    sc_pb = plt.scatter(coords[pb_indices, 1], coords[pb_indices, 2], c=np.array(volumetric_strains)[pb_indices], 
                      cmap=strain_cmap, s=50, vmin=-0.05, vmax=0.05, marker='s')
    sc_i = plt.scatter(coords[i_indices, 1], coords[i_indices, 2], c=np.array(volumetric_strains)[i_indices], 
                     cmap=strain_cmap, s=35, vmin=-0.05, vmax=0.05, marker='o')
    plt.colorbar(sc_pb, label="Volumetric Strain (Tr(E))")
    plt.xlabel("y (Å)")
    plt.ylabel("z (Å)")
    plt.title("Volumetric Strain (yz projection)")
    
    # Plot strain gradients
    plt.subplot(2, 3, 4)
    sc_pb = plt.scatter(coords[pb_indices, 0], coords[pb_indices, 1], c=vol_grad_mag[pb_indices], 
                      cmap=gradient_cmap, s=50, marker='s')
    sc_i = plt.scatter(coords[i_indices, 0], coords[i_indices, 1], c=vol_grad_mag[i_indices], 
                     cmap=gradient_cmap, s=35, marker='o')
    plt.colorbar(sc_pb, label="Strain Gradient Magnitude (Å⁻¹)")
    plt.xlabel("x (Å)")
    plt.ylabel("y (Å)")
    plt.title("Volumetric Strain Gradient (xy projection)")
    
    plt.subplot(2, 3, 5)
    sc_pb = plt.scatter(coords[pb_indices, 0], coords[pb_indices, 2], c=vol_grad_mag[pb_indices], 
                      cmap=gradient_cmap, s=50, marker='s')
    sc_i = plt.scatter(coords[i_indices, 0], coords[i_indices, 2], c=vol_grad_mag[i_indices], 
                     cmap=gradient_cmap, s=35, marker='o')
    plt.colorbar(sc_pb, label="Strain Gradient Magnitude (Å⁻¹)")
    plt.xlabel("x (Å)")
    plt.ylabel("z (Å)")
    plt.title("Volumetric Strain Gradient (xz projection)")
    
    plt.subplot(2, 3, 6)
    sc_pb = plt.scatter(coords[pb_indices, 1], coords[pb_indices, 2], c=vol_grad_mag[pb_indices], 
                      cmap=gradient_cmap, s=50, marker='s')
    sc_i = plt.scatter(coords[i_indices, 1], coords[i_indices, 2], c=vol_grad_mag[i_indices], 
                     cmap=gradient_cmap, s=35, marker='o')
    plt.colorbar(sc_pb, label="Strain Gradient Magnitude (Å⁻¹)")
    plt.xlabel("y (Å)")
    plt.ylabel("z (Å)")
    plt.title("Volumetric Strain Gradient (yz projection)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "volumetric_strain_and_gradients_Pb_I.png"), dpi=300)
    
    # Plot von Mises strain
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 3, 1)
    sc_pb = plt.scatter(coords[pb_indices, 0], coords[pb_indices, 1], c=np.array(von_mises_strains)[pb_indices], 
                      cmap=strain_cmap, s=50, marker='s', label='Pb')
    sc_i = plt.scatter(coords[i_indices, 0], coords[i_indices, 1], c=np.array(von_mises_strains)[i_indices], 
                     cmap=strain_cmap, s=35, marker='o', label='I')
    plt.colorbar(sc_pb, label="von Mises Strain")
    plt.xlabel("x (Å)")
    plt.ylabel("y (Å)")
    plt.title("von Mises Strain (xy projection)")
    plt.legend()
    
    plt.subplot(2, 3, 2)
    sc_pb = plt.scatter(coords[pb_indices, 0], coords[pb_indices, 2], c=np.array(von_mises_strains)[pb_indices], 
                      cmap=strain_cmap, s=50, marker='s')
    sc_i = plt.scatter(coords[i_indices, 0], coords[i_indices, 2], c=np.array(von_mises_strains)[i_indices], 
                     cmap=strain_cmap, s=35, marker='o')
    plt.colorbar(sc_pb, label="von Mises Strain")
    plt.xlabel("x (Å)")
    plt.ylabel("z (Å)")
    plt.title("von Mises Strain (xz projection)")
    
    plt.subplot(2, 3, 3)
    sc_pb = plt.scatter(coords[pb_indices, 1], coords[pb_indices, 2], c=np.array(von_mises_strains)[pb_indices], 
                      cmap=strain_cmap, s=50, marker='s')
    sc_i = plt.scatter(coords[i_indices, 1], coords[i_indices, 2], c=np.array(von_mises_strains)[i_indices], 
                     cmap=strain_cmap, s=35, marker='o')
    plt.colorbar(sc_pb, label="von Mises Strain")
    plt.xlabel("y (Å)")
    plt.ylabel("z (Å)")
    plt.title("von Mises Strain (yz projection)")
    
    plt.subplot(2, 3, 4)
    sc_pb = plt.scatter(coords[pb_indices, 0], coords[pb_indices, 1], c=vm_grad_mag[pb_indices], 
                      cmap=gradient_cmap, s=50, marker='s')
    sc_i = plt.scatter(coords[i_indices, 0], coords[i_indices, 1], c=vm_grad_mag[i_indices], 
                     cmap=gradient_cmap, s=35, marker='o')
    plt.colorbar(sc_pb, label="Strain Gradient Magnitude (Å⁻¹)")
    plt.xlabel("x (Å)")
    plt.ylabel("y (Å)")
    plt.title("von Mises Strain Gradient (xy projection)")
    
    plt.subplot(2, 3, 5)
    sc_pb = plt.scatter(coords[pb_indices, 0], coords[pb_indices, 2], c=vm_grad_mag[pb_indices], 
                      cmap=gradient_cmap, s=50, marker='s')
    sc_i = plt.scatter(coords[i_indices, 0], coords[i_indices, 2], c=vm_grad_mag[i_indices], 
                     cmap=gradient_cmap, s=35, marker='o')
    plt.colorbar(sc_pb, label="Strain Gradient Magnitude (Å⁻¹)")
    plt.xlabel("x (Å)")
    plt.ylabel("z (Å)")
    plt.title("von Mises Strain Gradient (xz projection)")
    
    plt.subplot(2, 3, 6)
    sc_pb = plt.scatter(coords[pb_indices, 1], coords[pb_indices, 2], c=vm_grad_mag[pb_indices], 
                      cmap=gradient_cmap, s=50, marker='s')
    sc_i = plt.scatter(coords[i_indices, 1], coords[i_indices, 2], c=vm_grad_mag[i_indices], 
                     cmap=gradient_cmap, s=35, marker='o')
    plt.colorbar(sc_pb, label="Strain Gradient Magnitude (Å⁻¹)")
    plt.xlabel("y (Å)")
    plt.ylabel("z (Å)")
    plt.title("von Mises Strain Gradient (yz projection)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "von_mises_strain_and_gradients_Pb_I.png"), dpi=300)
    
    # Create separate analysis for Pb and I atoms
    print("\nAnalyzing strain by element type...")
    
    # Calculate average strain values for each element type
    pb_vol_strain = np.array(volumetric_strains)[pb_indices]
    i_vol_strain = np.array(volumetric_strains)[i_indices]
    
    pb_vm_strain = np.array(von_mises_strains)[pb_indices]
    i_vm_strain = np.array(von_mises_strains)[i_indices]
    
    # Plot histograms of strain distributions
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(pb_vol_strain, bins=20, alpha=0.7, label='Pb')
    plt.hist(i_vol_strain, bins=20, alpha=0.7, label='I')
    plt.xlabel("Volumetric Strain")
    plt.ylabel("Count")
    plt.title("Volumetric Strain Distribution by Element")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.hist(pb_vm_strain, bins=20, alpha=0.7, label='Pb')
    plt.hist(i_vm_strain, bins=20, alpha=0.7, label='I')
    plt.xlabel("von Mises Strain")
    plt.ylabel("Count")
    plt.title("von Mises Strain Distribution by Element")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.hist(vol_grad_mag[pb_indices], bins=20, alpha=0.7, label='Pb')
    plt.hist(vol_grad_mag[i_indices], bins=20, alpha=0.7, label='I')
    plt.xlabel("Volumetric Strain Gradient Magnitude (Å⁻¹)")
    plt.ylabel("Count")
    plt.title("Volumetric Strain Gradient Distribution by Element")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.hist(vm_grad_mag[pb_indices], bins=20, alpha=0.7, label='Pb')
    plt.hist(vm_grad_mag[i_indices], bins=20, alpha=0.7, label='I')
    plt.xlabel("von Mises Strain Gradient Magnitude (Å⁻¹)")
    plt.ylabel("Count")
    plt.title("von Mises Strain Gradient Distribution by Element")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "strain_distributions_by_element.png"), dpi=300)
    
    # Save numerical results to file (now with element information)
    element_data = np.array([site.species_string for site in def_struct])
    
    np.savez(os.path.join(args.output_dir, 'strain_results_Pb_I.npz'),
             coords=coords,
             elements=element_data,
             volumetric_strain=volumetric_strains,
             von_mises_strain=von_mises_strains,
             max_shear_strain=max_shear_strains,
             vol_gradient_magnitude=vol_grad_mag,
             vol_gradient_vectors=vol_grad_vec,
             vm_gradient_magnitude=vm_grad_mag,
             vm_gradient_vectors=vm_grad_vec,
             shear_gradient_magnitude=shear_grad_mag,
             shear_gradient_vectors=shear_grad_vec)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")
    
    # Print summary statistics by element type
    print("\nSummary Statistics by Element:")
    print("Pb atoms:")
    print(f"  Count: {len(pb_indices)}")
    print(f"  Average volumetric strain: {np.mean(pb_vol_strain):.6f}")
    print(f"  Max volumetric strain: {np.max(pb_vol_strain):.6f}")
    print(f"  Min volumetric strain: {np.min(pb_vol_strain):.6f}")
    print(f"  Average von Mises strain: {np.mean(pb_vm_strain):.6f}")
    print(f"  Max strain gradient magnitude: {np.max(vol_grad_mag[pb_indices]):.6f} Å⁻¹")
    
    print("\nI atoms:")
    print(f"  Count: {len(i_indices)}")
    print(f"  Average volumetric strain: {np.mean(i_vol_strain):.6f}")
    print(f"  Max volumetric strain: {np.max(i_vol_strain):.6f}")
    print(f"  Min volumetric strain: {np.min(i_vol_strain):.6f}")
    print(f"  Average von Mises strain: {np.mean(i_vm_strain):.6f}")
    print(f"  Max strain gradient magnitude: {np.max(vol_grad_mag[i_indices]):.6f} Å⁻¹")

if __name__ == "__main__":
    main()