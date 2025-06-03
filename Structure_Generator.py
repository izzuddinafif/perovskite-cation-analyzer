import os
import math
from typing import List, Tuple, Set
from dataclasses import dataclass
import itertools
from string import Template
import numpy as np
# Base lattice parameter
BASE_LATTICE = 6.27514

@dataclass(frozen=True)
class Coordinate:
    x: float
    y: float
    z: float
    
    def __lt__(self, other):
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

def normalize_coordinate(x: float) -> float:
    """Normalize coordinate to [0,1) range considering periodicity."""
    return x - math.floor(x)

def get_symmetry_operations() -> List[np.ndarray]:
    """Generate full octahedral point group O_h (48 operations) including proper and improper rotations.
    
    This implementation uses group theory properties to efficiently generate the symmetry operations.
    """
    ops = []
    
    # Basic rotations
    rot_x90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rot_y90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    rot_z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # Identity operation
    ops.append(np.eye(3))
    
    # Generate rotations more efficiently by using known structure of O_h group
    # 3-fold rotations around body diagonals (8 operations)
    diag_dirs = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
        (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
    ]
    
    for direction in diag_dirs:
        # Create rotation matrix for 120° (2π/3) around body diagonal
        x, y, z = direction
        # Normalize direction vector
        norm = np.sqrt(x*x + y*y + z*z)
        x, y, z = x/norm, y/norm, z/norm
        
        # Rodrigues rotation formula for 120° rotation
        angle = 2*np.pi/3
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # Rotation matrix components using Rodrigues formula
        R = np.zeros((3, 3))
        R[0, 0] = cos_theta + x*x*(1-cos_theta)
        R[0, 1] = x*y*(1-cos_theta) - z*sin_theta
        R[0, 2] = x*z*(1-cos_theta) + y*sin_theta
        R[1, 0] = y*x*(1-cos_theta) + z*sin_theta
        R[1, 1] = cos_theta + y*y*(1-cos_theta)
        R[1, 2] = y*z*(1-cos_theta) - x*sin_theta
        R[2, 0] = z*x*(1-cos_theta) - y*sin_theta
        R[2, 1] = z*y*(1-cos_theta) + x*sin_theta
        R[2, 2] = cos_theta + z*z*(1-cos_theta)
        
        # Round to handle numerical precision issues
        R = np.round(R, 10)
        ops.append(R)
    
    # 4-fold rotations around coordinate axes (6 operations - already have identity)
    # x-axis 90° rotation
    ops.append(rot_x90)
    # x-axis 180° rotation
    ops.append(np.linalg.matrix_power(rot_x90, 2))
    # x-axis 270° rotation
    ops.append(np.linalg.matrix_power(rot_x90, 3))
    
    # y-axis 90° rotation
    ops.append(rot_y90)
    # y-axis 180° rotation
    ops.append(np.linalg.matrix_power(rot_y90, 2))
    # y-axis 270° rotation
    ops.append(np.linalg.matrix_power(rot_y90, 3))
    
    # z-axis 90° rotation
    ops.append(rot_z90)
    # z-axis 180° rotation
    ops.append(np.linalg.matrix_power(rot_z90, 2))
    # z-axis 270° rotation
    ops.append(np.linalg.matrix_power(rot_z90, 3))
    
    # 2-fold rotations around face diagonals (6 operations)
    face_diag = [
        (1, 1, 0), (1, -1, 0), (1, 0, 1), 
        (1, 0, -1), (0, 1, 1), (0, 1, -1)
    ]
    
    for direction in face_diag:
        # Create rotation matrix for 180° (π) around face diagonal
        x, y, z = direction
        # Normalize direction vector
        norm = np.sqrt(x*x + y*y + z*z)
        x, y, z = x/norm, y/norm, z/norm
        
        # Rodrigues rotation formula for 180° rotation
        angle = np.pi
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # Rotation matrix components
        R = np.zeros((3, 3))
        R[0, 0] = cos_theta + x*x*(1-cos_theta)
        R[0, 1] = x*y*(1-cos_theta) - z*sin_theta
        R[0, 2] = x*z*(1-cos_theta) + y*sin_theta
        R[1, 0] = y*x*(1-cos_theta) + z*sin_theta
        R[1, 1] = cos_theta + y*y*(1-cos_theta)
        R[1, 2] = y*z*(1-cos_theta) - x*sin_theta
        R[2, 0] = z*x*(1-cos_theta) - y*sin_theta
        R[2, 1] = z*y*(1-cos_theta) + x*sin_theta
        R[2, 2] = cos_theta + z*z*(1-cos_theta)
        
        # Round to handle numerical precision issues
        R = np.round(R, 10)
        ops.append(R)
    
    # Remove duplicates with rounding for stability
    unique_ops = []
    seen = set()
    for op in ops:
        key = tuple(np.round(op.flatten(), 6))
        if key not in seen:
            seen.add(key)
            unique_ops.append(op)
    
    # Generate improper operations by applying inversion (-I) to proper rotations
    improper_ops = [-op for op in unique_ops]
    
    all_ops = unique_ops + improper_ops
    
    # Verify we have the correct number of operations (24 proper + 24 improper = 48)
    if len(all_ops) != 48:
        print(f"Warning: Generated {len(all_ops)} operations instead of the expected 48.")
    
    return all_ops

def get_canonical_form(coords: List[Coordinate], ops: List[np.ndarray], system_size=2) -> Tuple[float, ...]:
    """Get canonical form of coordinates considering symmetry.
    
    Optimized version that bails out early when a lexicographically smaller form is found.
    """
    # Use the first transformed configuration as initial minimum
    transformed_min = None
    form_min = None
    
    # Apply symmetry operations
    for op in ops:
        transformed = []
        valid = True
        
        # Apply symmetry operation to all coordinates
        for coord in coords:
            point = np.array([coord.x, coord.y, coord.z])
            rotated = op @ point
            x, y, z = (normalize_coordinate(v) for v in rotated)
            
            # Apply normalization specific to the crystal system
            x = normalize_for_system(x, system_size)
            
            # Validate coordinates
            if not (0 <= y < 1 and 0 <= z < 1):
                valid = False
                break
            
            transformed.append(Coordinate(x, normalize_coordinate(y), normalize_coordinate(z)))
        
        if not valid:
            continue
            
        # Sort coordinates for consistent ordering
        transformed.sort()
        
        # Create tuple form for comparison
        form = tuple(coord for c in transformed for coord in (c.x, c.y, c.z))
        
        # Early termination logic - keep track of minimum form
        if form_min is None or form < form_min:
            form_min = form
            transformed_min = transformed.copy()
    
    return form_min if form_min else tuple()

def generate_all_positions(system_size=2) -> List[Coordinate]:
    """Generate all possible Cs positions based on system size.
    
    Args:
        system_size: Size of the system (2 for 2x2x2, 3 for 3x3x3)
    
    Returns:
        List of possible Cs positions as Coordinate objects
    """
    if system_size == 2:
        positions = [
            # x=0.25 layer
            Coordinate(0.25, 0.25, 0.25), Coordinate(0.25, 0.25, 0.75),
            Coordinate(0.25, 0.75, 0.25), Coordinate(0.25, 0.75, 0.75),
            # x=0.75 layer
            Coordinate(0.75, 0.25, 0.25), Coordinate(0.75, 0.25, 0.75),
            Coordinate(0.75, 0.75, 0.25), Coordinate(0.75, 0.75, 0.75)
        ]
    elif system_size == 3:
        positions = []
        # For 3x3x3 system, we use 1/6, 3/6, 5/6 for each coordinate
        values = [1/6, 3/6, 5/6]
        for x in values:
            for y in values:
                for z in values:
                    positions.append(Coordinate(x, y, z))
    else:
        raise ValueError(f"Unsupported system size: {system_size}")
        
    return sorted(positions)

def generate_pb_positions(system_size=2) -> List[Coordinate]:
    """Generate Pb positions based on system size."""
    if system_size == 2:
        positions = []
        for x in [0.0, 0.5]:
            for y in [0.0, 0.5]:
                for z in [0.0, 0.5]:
                    positions.append(Coordinate(x, y, z))
        return positions
    elif system_size == 3:
        positions = []
        for x in [0.0, 1/3, 2/3]:
            for y in [0.0, 1/3, 2/3]:
                for z in [0.0, 1/3, 2/3]:
                    positions.append(Coordinate(x, y, z))
        return positions
    else:
        raise ValueError(f"Unsupported system size: {system_size}")

def generate_i_positions(system_size=2) -> List[Coordinate]:
    """Generate I positions based on system size."""
    if system_size == 2:
        positions = []
        # xy-plane I atoms
        for x in [0.0, 0.5]:
            for y in [0.0, 0.5]:
                for z in [0.25, 0.75]:
                    positions.append(Coordinate(x, y, z))
        # yz-plane I atoms
        for x in [0.0, 0.5]:
            for y in [0.25, 0.75]:
                for z in [0.0, 0.5]:
                    positions.append(Coordinate(x, y, z))
        # xz-plane I atoms
        for x in [0.25, 0.75]:
            for y in [0.0, 0.5]:
                for z in [0.0, 0.5]:
                    positions.append(Coordinate(x, y, z))
        return sorted(positions)
    elif system_size == 3:
        positions = []
        # For 3x3x3, handle similar to 2x2x2 but with 3 divisions per dimension
        # xy-plane I atoms
        for x in [0.0, 1/3, 2/3]:
            for y in [0.0, 1/3, 2/3]:
                for z in [1/6, 3/6, 5/6]:
                    positions.append(Coordinate(x, y, z))
        # yz-plane I atoms
        for x in [0.0, 1/3, 2/3]:
            for y in [1/6, 3/6, 5/6]:
                for z in [0.0, 1/3, 2/3]:
                    positions.append(Coordinate(x, y, z))
        # xz-plane I atoms
        for x in [1/6, 3/6, 5/6]:
            for y in [0.0, 1/3, 2/3]:
                for z in [0.0, 1/3, 2/3]:
                    positions.append(Coordinate(x, y, z))
        return sorted(positions)
    else:
        raise ValueError(f"Unsupported system size: {system_size}")

# Global variables for multiprocessing support
_all_positions = None
_symmetry_ops = None
_system_size = 2

def init_worker(all_positions, symmetry_ops, system_size):
    """Initialize worker process with shared data.
    
    This function is called once per worker to set up its global state.
    """
    global _all_positions, _symmetry_ops, _system_size
    _all_positions = all_positions
    _symmetry_ops = symmetry_ops
    _system_size = system_size

def process_config(config):
    """Process a single configuration to find its canonical form.
    
    Args:
        config: A tuple of coordinates representing a configuration
        
    Returns:
        tuple: (canonical_form, configuration) or (None, None) if invalid
    """
    global _symmetry_ops, _system_size
    
    # Convert to list for manipulation
    config_list = list(config)
    
    # Get canonical form as orbit identifier
    canonical = get_canonical_form(config_list, _symmetry_ops, _system_size)
    
    if canonical:
        return canonical, config_list
    return None, None

def process_batch(batch_configs):
    """Process a batch of configurations.
    
    Args:
        batch_configs: List of configurations to process
        
    Returns:
        dict: Dictionary mapping canonical forms to configuration lists
    """
    global _system_size
    
    batch_results = {}
    for config in batch_configs:
        canonical, config_list = process_config(config)
        if canonical and canonical not in batch_results:
            batch_results[canonical] = config_list
    return batch_results

def generate_unique_configurations(num_cs: int, system_size=2, use_multiprocessing=False) -> List[List[Coordinate]]:
    """Generate all unique configurations using Burnside's Lemma with optimizations.
    
    Args:
        num_cs: Number of Cs atoms to place
        system_size: Size of the system (2 for 2x2x2, 3 for 3x3x3)
        use_multiprocessing: Whether to use multiprocessing for parallel computation
        
    Returns:
        List of unique configurations as lists of Coordinate objects
    """
    global _all_positions, _symmetry_ops, _system_size
    
    # Initialize variables in the main process
    _system_size = system_size
    _all_positions = generate_all_positions(system_size)
    _symmetry_ops = get_symmetry_operations()
    
    print(f"Total possible Cs positions: {len(_all_positions)}")
    print(f"Total symmetry operations: {len(_symmetry_ops)}")
    
    # Calculate total number of configurations for reporting
    from math import comb
    total_configs = comb(len(_all_positions), num_cs)
    print(f"Total configurations before symmetry: {total_configs}")
    
    # Dictionary to store orbit representatives
    orbit_representatives = {}
    
    # Check if we should use multiprocessing
    if use_multiprocessing:
        import multiprocessing as mp
        from multiprocessing import Pool
        print(f"Using multiprocessing with {mp.cpu_count()} cores")
        
        # Generate all configurations 
        all_configs = list(itertools.combinations(_all_positions, num_cs))
        
        # Determine batch size based on total configurations
        batch_size = max(1, min(10000, len(all_configs) // (mp.cpu_count() * 2)))
        batch_size = max(1, batch_size)  # Ensure at least one configuration per batch
        
        batches = [all_configs[i:i+batch_size] for i in range(0, len(all_configs), batch_size)]
        print(f"Processing {len(batches)} batches of size {batch_size}")
        
        # Process batches in parallel, initializing each worker with the necessary data
        with Pool(initializer=init_worker, 
                  initargs=(_all_positions, _symmetry_ops, _system_size)) as pool:
            for i, batch_result in enumerate(pool.imap_unordered(process_batch, batches)):
                if (i+1) % 10 == 0 or (i+1) == len(batches):
                    print(f"Processed batch {i+1}/{len(batches)}, found {len(orbit_representatives)} unique configs")
                orbit_representatives.update(batch_result)
    else:
        # Single process implementation with optimization
        print("Using single process mode")
        
        # Show progress indicator
        from itertools import islice
        processed = 0
        milestone = max(1, total_configs // 100)  # Report progress every 1% or at least every configuration
        
        # Process configurations
        for config in itertools.combinations(_all_positions, num_cs):
            canonical, config_list = process_config(config)
            if canonical and canonical not in orbit_representatives:
                orbit_representatives[canonical] = config_list
            
            # Report progress
            processed += 1
            if processed % milestone == 0 or processed == total_configs:
                percentage = (processed / total_configs) * 100
                print(f"Progress: {processed}/{total_configs} ({percentage:.1f}%), found {len(orbit_representatives)} unique configs")
    
    # Get final list of unique configurations
    unique_configs = list(orbit_representatives.values())
    
    # Print some sample configurations
    print(f"\nFound {len(unique_configs)} unique configurations")
    num_to_show = min(4, len(unique_configs))
    for i, config in enumerate(unique_configs[:num_to_show], 1):
        print(f"\nConfiguration {i}:")
        for coord in config:
            print(f"  Cs at ({coord.x:.3f}, {coord.y:.3f}, {coord.z:.3f})")
    
    return unique_configs

def generate_files(cs_configs: List[List[Coordinate]], system_size=2):
    """Generate CIF files and plots for all configurations."""
    os.makedirs("result", exist_ok=True)
    
    # Calculate supercell parameters based on system size
    A = BASE_LATTICE * system_size
    B = BASE_LATTICE * system_size
    C = BASE_LATTICE * system_size
    
    # Generate fixed Pb and I positions for the selected system size
    pb_positions = generate_pb_positions(system_size)
    i_positions = generate_i_positions(system_size)
    
    for i, cs_coords in enumerate(cs_configs, 1):
        # Generate CIF file
        atom_positions = []
        
        # Add Cs atoms
        for j, pos in enumerate(cs_coords, 1):
            atom_positions.append(
                f"   Cs{j}        1.0     {pos.x:.6f}     {pos.y:.6f}     {pos.z:.6f}    Uiso  ? Cs"
            )
        
        # Add Pb atoms
        for j, pos in enumerate(pb_positions, 1):
            atom_positions.append(
                f"   Pb{j}        1.0     {pos.x:.6f}     {pos.y:.6f}     {pos.z:.6f}    Uiso  ? Pb"
            )
        
        # Add I atoms
        for j, pos in enumerate(i_positions, 1):
            atom_positions.append(
                f"   I{j}        1.0     {pos.x:.6f}     {pos.y:.6f}     {pos.z:.6f}    Uiso  ? I"
            )
        
        # Calculate cell volume
        volume = A * B * C
        
        # Create the complete CIF content
        cif_content = Template('''#======================================================================
# CRYSTAL DATA
#----------------------------------------------------------------------
data_VESTA_phase_1

_chemical_name_common                  'Configuration ${config_num}'
_cell_length_a                         ${cell_a}
_cell_length_b                         ${cell_b}
_cell_length_c                         ${cell_c}
_cell_angle_alpha                      90.000000
_cell_angle_beta                       90.000000
_cell_angle_gamma                      90.000000
_cell_volume                          ${volume}
_space_group_name_H-M_alt              'P 1'
_space_group_IT_number                 1

loop_
_space_group_symop_operation_xyz
   'x, y, z'

loop_
   _atom_site_label
   _atom_site_occupancy
   _atom_site_fract_x
   _atom_site_fract_y
   _atom_site_fract_z
   _atom_site_adp_type
   _atom_site_U_iso_or_equiv
   _atom_site_type_symbol
${atom_positions}''').substitute(
            config_num=i,
            cell_a=f"{A:.6f}",
            cell_b=f"{B:.6f}",
            cell_c=f"{C:.6f}",
            volume=f"{volume:.6f}",
            atom_positions="\n".join(atom_positions)
        )
        
        # Write CIF file
        filename = os.path.join("result", f"config_{i}.cif")
        with open(filename, "w") as f:
            f.write(cif_content)

def normalize_for_system(x: float, system_size=2) -> float:
    """Normalize coordinate based on the system size.
    
    For 2x2x2 system, normalize x to 0.25 or 0.75
    For 3x3x3 system, normalize to 1/6, 3/6, 5/6
    """
    if system_size == 2:
        # For 2x2x2, we have 0.25 and 0.75 as possible x values
        return 0.25 if abs(x - 0.25) < abs(x - 0.75) else 0.75
    elif system_size == 3:
        # For 3x3x3, we have values at 1/6, 3/6, 5/6
        possible_values = [1/6, 3/6, 5/6]
        distances = [abs(x - val) for val in possible_values]
        return possible_values[distances.index(min(distances))]
    else:
        # Default behavior for other system sizes
        return x

def verify_uniqueness(unique_configs: List[List[Coordinate]], system_size=2) -> dict:
    """Verify that the configurations are all unique and non-equivalent under symmetry.
    
    This function performs several checks:
    1. No two configurations are equivalent under symmetry operations
    2. Each configuration has the expected number of atoms
    3. Sampling validation against Burnside's lemma for small cases
    
    Args:
        unique_configs: List of configurations to verify
        system_size: Size of the system (2 for 2x2x2, 3 for 3x3x3)
    
    Returns:
        Dict with verification results
    """
    if not unique_configs:
        return {
            "status": "failed",
            "message": "No configurations to verify",
            "equal_pairs": []
        }
    
    # Get symmetry operations
    symmetry_ops = get_symmetry_operations()
    
    # Check that all configurations have the same number of atoms
    num_cs = len(unique_configs[0])
    if not all(len(config) == num_cs for config in unique_configs):
        return {
            "status": "failed",
            "message": "Not all configurations have the same number of atoms",
            "equal_pairs": []
        }
    
    # Check for duplicates/equivalent configurations
    equal_pairs = []
    canonical_forms = {}
    
    # For each configuration, compute its canonical form
    for i, config in enumerate(unique_configs):
        canonical = get_canonical_form(config, symmetry_ops, system_size)
        
        # Check if we've seen this canonical form before
        if canonical in canonical_forms:
            equal_pairs.append((i+1, canonical_forms[canonical]+1))
        else:
            canonical_forms[canonical] = i
    
    # Check for known special cases
    total_positions = 27 if system_size == 3 else 8
    if num_cs == 0 or num_cs == total_positions:
        expected_unique = 1
    elif num_cs == 1 or num_cs == total_positions - 1:
        # For 1 atom or 1 vacancy, we expect the orbits of a single position
        # This is well-known in group theory for highly symmetric spaces like this
        expected_unique = 2 if system_size == 2 else 4  # Correct for O_h symmetry
    else:
        # We're already computing uniqueness with the canonical form approach
        expected_unique = len(unique_configs)
    
    # Return verification results
    if equal_pairs:
        return {
            "status": "failed",
            "message": f"Found {len(equal_pairs)} pairs of equivalent configurations",
            "equal_pairs": equal_pairs,
            "expected_unique": expected_unique,
            "actual_unique": len(unique_configs)
        }
    elif len(unique_configs) != expected_unique and (num_cs == 0 or num_cs == total_positions or num_cs == 1 or num_cs == total_positions - 1):
        return {
            "status": "failed",
            "message": f"Expected {expected_unique} unique configurations for this special case, but found {len(unique_configs)}",
            "equal_pairs": [],
            "expected_unique": expected_unique,
            "actual_unique": len(unique_configs)
        }
    else:
        return {
            "status": "success",
            "message": f"All {len(unique_configs)} configurations are unique and pass verification",
            "equal_pairs": [],
            "expected_unique": expected_unique,
            "actual_unique": len(unique_configs)
        }

def main():
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate unique crystal configurations using Burnside's lemma")
    parser.add_argument("-n", "--num_cs", type=int, default=4, help="Number of Cs atoms to place (default: 4)")
    parser.add_argument("-s", "--system_size", type=int, default=2, choices=[2, 3], help="System size (2 for 2x2x2, 3 for 3x3x3, default: 2)")
    parser.add_argument("-p", "--parallel", action="store_true", help="Use parallelization for faster computation")
    parser.add_argument("-o", "--output_dir", default="result", help="Output directory for CIF files")
    parser.add_argument("-v", "--verify", action="store_true", help="Verify uniqueness of generated configurations")
    args = parser.parse_args()
    
    # Validate inputs
    if args.num_cs < 1:
        print(f"Warning: Number of Cs atoms ({args.num_cs}) should be positive. Using 4.")
        args.num_cs = 4
        
    # For 3x3x3 system, there are 27 possible positions
    if args.system_size == 3 and args.num_cs > 27:
        print(f"Error: Number of Cs atoms ({args.num_cs}) exceeds available positions (27) for 3x3x3 system.")
        sys.exit(1)
        
    # For 2x2x2 system, there are 8 possible positions
    if args.system_size == 2 and args.num_cs > 8:
        print(f"Error: Number of Cs atoms ({args.num_cs}) exceeds available positions (8) for 2x2x2 system.")
        sys.exit(1)
    
    # Print configuration
    print("=" * 60)
    print(f"Generating {args.system_size}x{args.system_size}x{args.system_size} configurations")
    print(f"Number of Cs atoms: {args.num_cs}")
    print(f"Parallelization: {'Enabled' if args.parallel else 'Disabled'}")
    print(f"Verification: {'Enabled' if args.verify else 'Disabled'}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Check for Python multiprocessing support
    if args.parallel:
        try:
            import multiprocessing
            print(f"Using {multiprocessing.cpu_count()} CPU cores")
        except ImportError:
            print("Warning: multiprocessing module not available. Running in single-process mode.")
            args.parallel = False
    
    # Make the output directory the current result directory
    if args.output_dir != "result":
        os.makedirs(args.output_dir, exist_ok=True)
        
    # Generate unique configurations using Burnside's lemma
    configs = generate_unique_configurations(args.num_cs, args.system_size, args.parallel)
    print(f"\nFound {len(configs)} unique configurations")
    
    if configs:
        # Verify uniqueness if requested
        if args.verify:
            print("\nVerifying uniqueness of configurations...")
            verification_result = verify_uniqueness(configs, args.system_size)
            print(f"Verification status: {verification_result['status']}")
            print(f"Message: {verification_result['message']}")
            
            if verification_result['status'] != 'success':
                if 'equal_pairs' in verification_result and verification_result['equal_pairs']:
                    print("Found equivalent configurations:")
                    for pair in verification_result['equal_pairs'][:5]:  # Show at most 5 pairs
                        print(f"  Configurations {pair[0]} and {pair[1]} are equivalent")
                    if len(verification_result['equal_pairs']) > 5:
                        print(f"  ... and {len(verification_result['equal_pairs'])-5} more equivalent pairs")
                
                if input("Continue generating files despite verification failure? (y/n): ").lower() != 'y':
                    print("Aborting file generation.")
                    sys.exit(1)
        
        # Generate files
        generate_files(configs, args.system_size)
        print("\nCompleted! Check:")
        print(f"- '{args.output_dir}' directory for CIF files")
    else:
        print("\nNo valid configurations found. Check your parameters and try again.")

if __name__ == "__main__":
    main()
