# pdb2vol.py - Modified to match the first program's density map generation approach 2025/5/28 xxy

from Bio.PDB import PDBParser, MMCIFParser
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
from scipy.ndimage import fourier_gaussian, gaussian_filter, zoom
from scipy.fftpack import fftn, ifftn
import math
from tqdm import tqdm
from numba import njit
from numba.typed import Dict, List

# Dictionary of atom types and their corresponding masses/weights
atom_mass_dict = Dict()
atom_mass_dict["H"] = 1.008
atom_mass_dict["C"] = 12.011
atom_mass_dict["CA"] = 12.011  # for PDB files without element notations
atom_mass_dict["N"] = 14.007
atom_mass_dict["O"] = 15.999
atom_mass_dict["P"] = 30.974
atom_mass_dict["S"] = 32.066

# For atomic number based weighting (similar to first program)
# Use float values to avoid numba typing issues
atomic_number_dict = Dict()
atomic_number_dict["H"] = 1.0
atomic_number_dict["C"] = 6.0
atomic_number_dict["CA"] = 6.0
atomic_number_dict["N"] = 7.0
atomic_number_dict["O"] = 8.0
atomic_number_dict["P"] = 15.0
atomic_number_dict["S"] = 16.0


def get_atom_list(pdb_file, backbone_only=False):
    """
    Retrieve the coordinates and atom types from a PDB or CIF file.
    For multi-model structures, only the first model is used.
    """
    if pdb_file.endswith(".pdb") or pdb_file.endswith(".ent"):  
        st_parser = PDBParser(QUIET=True)
    elif pdb_file.endswith(".cif"):
        st_parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file extension for {pdb_file}. Supported extensions: .pdb, .cif, .ent")
        
    structure = st_parser.get_structure("protein", pdb_file)

    models = list(structure.get_models())
    if not models:
        raise ValueError(f"No models found in {pdb_file}")

    first_model = models[0]
    atom_list = []
    atom_type_list = List()

    if backbone_only:
        for chain in first_model:
            for residue in chain:
                if "CA" in residue:
                    atom_list.append(residue["CA"].get_coord())
                    atom_type_list.append(residue["CA"].element)
                    if "C" in residue:
                        atom_list.append(residue["C"].get_coord())
                        atom_type_list.append(residue["C"].element)
                    if "N" in residue:
                        atom_list.append(residue["N"].get_coord())
                        atom_type_list.append(residue["N"].element)
                    if "O" in residue:
                        atom_list.append(residue["O"].get_coord())
                        atom_type_list.append(residue["O"].element)
    else:
        for atom in first_model.get_atoms():
            atom_list.append(atom.get_coord())
            atom_type_list.append(atom.element)

    return np.array(atom_list), atom_type_list


def calculate_centre_of_mass(atom_list, atom_type_list):
    """Calculate the centre of mass for a given list of atoms and their types."""
    atom_list = np.array(atom_list)
    atom_type_list = np.array(atom_type_list)
    x = atom_list[:, 0]
    y = atom_list[:, 1]
    z = atom_list[:, 2]
    m = np.array([atom_mass_dict.get(atom_type, 0.0) for atom_type in atom_type_list])
    mass_total = np.sum(m)
    x_co_m = np.sum(x * m) / mass_total
    y_co_m = np.sum(y * m) / mass_total
    z_co_m = np.sum(z * m) / mass_total
    return x_co_m, y_co_m, z_co_m


def prot2map_improved(atom_list, atom_type_list, voxel_size, resolution=None):
    """
    Calculate the size and origin of a protein map - improved version
    matching the ChimeraX approach more closely.
    """
    max_x, max_y, max_z = atom_list.max(axis=0)
    min_x, min_y, min_z = atom_list.min(axis=0)

    # Use padding similar to first program (3 times resolution)
    if resolution is not None:
        pad = 3.0 * resolution
    else:
        pad = np.array([10.0, 10.0, 10.0])

    # Calculate dimensions with padding
    x_size = math.ceil((max_x - min_x + 2 * pad) / voxel_size[0])
    y_size = math.ceil((max_y - min_y + 2 * pad) / voxel_size[1])
    z_size = math.ceil((max_z - min_z + 2 * pad) / voxel_size[2])
    
    # 计算分子中心
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    center_z = (max_z + min_z) / 2
    
    # 设置原点让分子居中（类似ChimeraX的做法）
    x_origin = center_x - (x_size * voxel_size[0]) / 2
    y_origin = center_y - (y_size * voxel_size[1]) / 2
    z_origin = center_z - (z_size * voxel_size[2]) / 2

    return (z_size, y_size, x_size), (x_origin, y_origin, z_origin)


@njit(fastmath=True, nogil=True)
def add_gaussian_to_grid(grid_data, atom_coord, weight, origin, voxel_size,
                         sigma_voxels, cutoff_range=5.0):
    """
    Add a 3D Gaussian distribution centered at atom_coord to the grid.
    This mimics the approach used in the first program.
    """
    # Convert atom coordinate to grid indices (float)
    i_center = (atom_coord[0] - origin[0]) / voxel_size[0]
    j_center = (atom_coord[1] - origin[1]) / voxel_size[1]
    k_center = (atom_coord[2] - origin[2]) / voxel_size[2]

    # Calculate the range of grid points to consider
    i_range = int(np.ceil(cutoff_range * sigma_voxels[0]))
    j_range = int(np.ceil(cutoff_range * sigma_voxels[1]))
    k_range = int(np.ceil(cutoff_range * sigma_voxels[2]))

    # Grid boundaries
    nz, ny, nx = grid_data.shape

    # Iterate over nearby grid points
    for k in range(max(0, int(k_center) - k_range),
                   min(nz, int(k_center) + k_range + 1)):
        for j in range(max(0, int(j_center) - j_range),
                       min(ny, int(j_center) + j_range + 1)):
            for i in range(max(0, int(i_center) - i_range),
                           min(nx, int(i_center) + i_range + 1)):

                # Calculate distance from atom center in voxel units
                di = (i - i_center) / sigma_voxels[0]
                dj = (j - j_center) / sigma_voxels[1]
                dk = (k - k_center) / sigma_voxels[2]

                # Distance squared
                r2 = di * di + dj * dj + dk * dk

                # Apply cutoff
                if r2 <= cutoff_range * cutoff_range:
                    # Gaussian value
                    gaussian_val = np.exp(-0.5 * r2)
                    grid_data[k, j, i] += weight * gaussian_val


def make_gaussian_density_map(origin, voxel_size, box_size, atom_list, atom_type_list,
                              resolution, sigma_factor=1 / (np.pi * np.sqrt(2)),
                              cutoff_range=5.0, use_atomic_number=True):
    """
    Create a density map using 3D Gaussians, similar to the first program.

    Parameters:
    - sigma_factor: default 1/(pi*sqrt(2)) ≈ 0.225, same as first program
    - use_atomic_number: if True, use atomic numbers as weights (like first program)
    """
    map_data = np.zeros(box_size, dtype=np.float32)

    # Calculate sigma in real space
    sigma_real = resolution * sigma_factor

    # Convert sigma to voxel units for each dimension
    sigma_voxels = np.array([
        sigma_real / voxel_size[0],
        sigma_real / voxel_size[1],
        sigma_real / voxel_size[2]
    ])

    print(f"Resolution: {resolution}, Sigma (real): {sigma_real}")
    print(f"Sigma (voxels): {sigma_voxels}")

    # Prepare weights array to avoid numba typing issues
    weights = np.zeros(len(atom_list), dtype=np.float32)
    for i, atom_type in enumerate(atom_type_list):
        if use_atomic_number:
            # Use get method with proper type matching
            if atom_type in atomic_number_dict:
                weights[i] = atomic_number_dict[atom_type]
            else:
                weights[i] = 1.0
        else:
            if atom_type in atom_mass_dict:
                weights[i] = atom_mass_dict[atom_type]
            else:
                weights[i] = 1.0

    # Add Gaussian for each atom
    for i, atom_coord in enumerate(atom_list):
        add_gaussian_to_grid(map_data, atom_coord, weights[i], origin,
                             voxel_size, sigma_voxels, cutoff_range)

    # Apply normalization factor like the first program
    # normalization = (2*pi)^(-1.5) * sigma^(-3)
    normalization = np.power(2 * np.pi, -1.5) * np.power(sigma_real, -3)
    map_data *= normalization

    return map_data


def write_mrc_file(data, origin, voxel_size, mrc_file):
    """Write a data array to an MRC file."""
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.update_header_from_data()
        mrc.voxel_size = tuple(voxel_size)
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]
        mrc.update_header_stats()
        mrc.flush()


def resample_by_box_size(data, box_size):
    """Resample data to match specified box size using cubic spline interpolation."""
    zoom_factor = np.array(box_size) / np.array(data.shape)
    return zoom(data, zoom_factor, order=3)


def pdb2vol_improved(
        input_pdb,
        resolution,
        output_mrc=None,
        ref_map=False,
        sigma_factor=1 / (np.pi * np.sqrt(2)),  # Default same as first program
        cutoff_range=5.0,
        use_atomic_number=True,  # Use atomic numbers like first program
        backbone_only=False,
        contour=False,
        bin_mask=False,
        return_data=False,
):
    """
    Convert a PDB or CIF file to a volumetric map using the same approach as the first program.

    Key changes:
    - Uses 3D Gaussian distributions instead of trilinear interpolation
    - Uses atomic numbers as weights (like first program)
    - Uses theoretical normalization instead of statistical normalization
    - Default sigma_factor matches first program: 1/(pi*sqrt(2)) ≈ 0.225
    """

    if input_pdb.split(".")[-1] not in ["pdb", "cif", "ent"]:
        raise ValueError("Input file must be a pdb, cif, or ent file")

    atoms, types = get_atom_list(input_pdb, backbone_only=backbone_only)

    if len(atoms) == 0:
        raise ValueError("No atoms found in input file")
    if len(atoms) != len(types):
        raise ValueError("Number of atoms and atom types does not match")

    # Determine grid parameters
    if not ref_map:
        # Default grid spacing: 1/3 of resolution (same as first program)
        grid_spacing = resolution / 3.0
        voxel_size = np.array([grid_spacing, grid_spacing, grid_spacing])
        dims, origin = prot2map_improved(atoms, types, voxel_size, resolution)
    else:
        with mrcfile.open(ref_map, permissive=True) as mrc:
            voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
            dims = mrc.data.shape
            origin = np.array([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z])

    print(f"Grid dimensions: {dims}")
    print(f"Voxel size: {voxel_size}")
    print(f"Origin: {origin}")

    # Generate density map using Gaussian approach (like first program)
    map_data = make_gaussian_density_map(
        origin, voxel_size, dims, atoms, types,
        resolution, sigma_factor, cutoff_range, use_atomic_number
    )

    # Resample if needed (for reference map matching)
    if ref_map:
        target_dims = dims
        if map_data.shape != target_dims:
            map_data = resample_by_box_size(map_data, target_dims)

    # Apply contour threshold if specified
    if contour:
        map_data = np.where(map_data > contour, map_data, 0)

    # Apply binary mask if specified
    if bin_mask:
        map_data = np.where(map_data > 0, 1, 0)

    print(f"Map statistics: min={map_data.min():.6f}, max={map_data.max():.6f}, mean={map_data.mean():.6f}")

    # Write output file
    if output_mrc is not None:
        write_mrc_file(map_data, origin, voxel_size, output_mrc)
        print(f"Density map saved to: {output_mrc}")

    if return_data:
        return map_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDB/CIF to volumetric map using Gaussian approach")
    parser.add_argument("input_pdb", help="Path to the input PDB or CIF file")
    parser.add_argument("resolution", type=float, help="Resolution of the output map")
    parser.add_argument("output_mrc", help="Path to save the output MRC file")
    parser.add_argument("-m", "--ref_map", help="Path to a reference map in MRC format", default=None)
    parser.add_argument("-s", "--sigma_factor", type=float, default=1 / (np.pi * np.sqrt(2)),
                        help="Sigma factor (default: 1/(pi*sqrt(2)) ≈ 0.225)")
    parser.add_argument("-c", "--cutoff_range", type=float, default=5.0,
                        help="Cutoff range in standard deviations")
    parser.add_argument("--use_mass", action="store_true", default=False,
                        help="Use atomic mass instead of atomic number as weight")
    parser.add_argument("-bb", "--backbone_only", action="store_true", default=False,
                        help="Only consider backbone atoms")
    parser.add_argument("-b", "--bin_mask", action="store_true", default=False,
                        help="Binarize the output map")
    parser.add_argument("--contour", type=float, default=0.0,
                        help="Contour level for thresholding")

    args = parser.parse_args()

    pdb2vol_improved(
        args.input_pdb,
        args.resolution,
        args.output_mrc,
        args.ref_map,
        args.sigma_factor,
        args.cutoff_range,
        not args.use_mass,  # use_atomic_number = not use_mass
        args.backbone_only,
        args.contour,
        args.bin_mask,
        False,
    )