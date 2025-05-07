"""
GUESTNEST
Copyright (C) 2025  Conor D. Rankine

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software 
Foundation, either Version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# =============================================================================
#                               LIBRARY IMPORTS
# =============================================================================

import psutil
import numpy as np
from rdkit import Chem
from copy import deepcopy

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

def centre(
    mol: Chem.Mol,
    conf_idx: int = -1,
    inplace: bool = False
) -> Chem.Mol:
    """
    Centres a molecule via translation in Cartesian coordinates such that the
    centre of mass is coincident with the origin ([0.0, 0.0, 0.0]).

    Args:
        mol (Chem.Mol): Molecule.
        conf_idx (int, optional): Conformer index. Defaults to -1.
        inplace (bool, optional): If True, the molecule is modified inplace
            and returned; if False, a copy is created, modified, and returned.
            Defaults to False.

    Returns:
        Chem.Mol: Centred molecule.
    """
    
    target = mol if inplace else deepcopy(mol)    
    _set_centre_of_mass(
        target, [0.0, 0.0, 0.0], conf_idx = conf_idx
    )

    return target

def translate_mol(
    mol: Chem.Mol,
    distances: np.ndarray,
    conf_idx: int = -1,
    inplace: bool = False
) -> Chem.Mol:
    """
    Translates a molecule in Cartesian coordinates.

    Args:
        mol (Chem.Mol): Molecule.
        distances (np.ndarray): Translation vector ([x, y, z]) in Angstroem.
        conf_idx (int, optional): Conformer index. Defaults to -1.
        inplace (bool, optional): If True, the molecule is modified inplace
            and returned; if False, a copy is created, modified, and returned.
            Defaults to False.

    Returns:
        Chem.Mol: Translated molecule.
    """
    
    target = mol if inplace else deepcopy(mol)    
    translated_coords = translate_coords(
        get_coords(target, conf_idx = conf_idx), distances
    )
    set_coords(target, translated_coords, conf_idx = conf_idx)

    return target

def rotate_mol(
    mol: Chem.Mol,
    angles: np.ndarray,
    conf_idx: int = -1,
    inplace: bool = False
) -> Chem.Mol:
    """
    Rotates a molecule around the origin ([0.0, 0.0, 0.0]).

    Args:
        mol (Chem.Mol): Molecule.
        angles (np.ndarray): Rotation angles ([a, b, c]) in radians.
        conf_idx (int, optional): Conformer index. Defaults to -1.
        inplace (bool, optional): If True, the molecule is modified inplace
            and returned; if False, a copy is created, modified, and returned.
            Defaults to False.

    Returns:
        Chem.Mol: Rotated molecule.
    """
    
    target = mol if inplace else deepcopy(mol)    
    rotated_coords = rotate_coords(
        get_coords(target, conf_idx = conf_idx), angles
    )
    set_coords(target, rotated_coords, conf_idx = conf_idx)

    return target

def rotate_and_translate_mol(
    mol: Chem.Mol,
    angles: np.ndarray,
    distances: np.ndarray,
    conf_idx: int = -1,
    inplace: bool = False
) -> Chem.Mol:
    """
    Rotates (around the origin ([0.0, 0.0, 0.0])) and translates (in Cartesian
    coordinates) a molecule sequentially.

    Args:
        mol (Chem.Mol): Molecule.
        angles (np.ndarray): Rotation angles ([a, b, c]) in radians.
        distances (np.ndarray): Translation vector ([x, y, z]) in Angstroem.
        conf_idx (int, optional): Conformer index. Defaults to -1.
        inplace (bool, optional): If True, the molecule is modified inplace
            and returned; if False, a copy is created, modified, and returned.
            Defaults to False.

    Returns:
        Chem.Mol: Rotated and translated molecule.
    """
    
    target = mol if inplace else deepcopy(mol)    
    target = rotate_mol(
        target, angles, conf_idx = conf_idx, inplace = True
    )
    target = translate_mol(
        target, distances, conf_idx = conf_idx, inplace = True
    )

    return target

def translate_coords(
    coords: np.ndarray,
    distances: np.ndarray
) -> np.ndarray:
    """
    Translates a set of Cartesian coordinates.

    Args:
        coords (np.ndarray): Cartesian coordinates as an array of shape (n, 3).
        distances (np.ndarray): Translation vector ([x, y, z]).

    Returns:
        np.ndarray: Translated Cartesian coordinates as an array of shape
            (n, 3).
    """
    
    return coords + distances

def rotate_coords(
    coords: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    """
    Rotates a set of Cartesian coordinates around the origin ([0.0, 0.0, 0.0]).

    Args:
        coords (np.ndarray): Cartesian coordinates as an array of shape (n, 3).
        angles (np.ndarray): Rotation angles ([a, b, c]) in radians.

    Returns:
        np.ndarray: Rotated Cartesian coordinates as an array of shape
            (n, 3).
    """
    
    return coords @ _angles_to_rotation_matrix(angles).T

def rotate_and_translate_coords(
    coords: np.ndarray,
    angles: np.ndarray,
    distances: np.ndarray
) -> np.ndarray:
    """
    Rotates (around the origin ([0.0, 0.0, 0.0])) and translates a set of
    Cartesian coordinates sequentially.

    Args:
        coords (np.ndarray): Cartesian coordinates as an array of shape (n, 3).
        angles (np.ndarray): Rotation angles ([a, b, c]) in radians.
        distances (np.ndarray): Translation vector ([x, y, z]).

    Returns:
        np.ndarray: Rotated and translated Cartesian coordinates as an array of
            shape (n, 3).
    """
    
    return translate_coords(rotate_coords(coords, angles), distances)

def get_coords(
    mol: Chem.Mol,
    conf_idx: int = -1
) -> np.ndarray:
    """
    Gets the Cartesian coordinates of the atoms in a molecule as an array of
    shape (n_atoms, 3).

    Args:
        mol (Chem.Mol): Molecule.
        conf_idx (int, optional): Conformer index. Defaults to -1.

    Returns:
        np.ndarray: Cartesian coordinates as an array of shape (n_atoms, 3).
    """
    
    conf = mol.GetConformer(conf_idx)
    return np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )

def set_coords(
    mol: Chem.Mol,
    coords: np.ndarray,
    conf_idx: int = -1
) -> None:
    """
    Sets the Cartesian coordinates of the atoms in a molecule from an array of
    shape (n_atoms, 3).

    Args:
        mol (Chem.Mol): Molecule.
        coords (np.ndarray): Cartesian coordinates as an array of shape
            (n_atoms, 3).
        conf_idx (int, optional): Conformer index. Defaults to -1.
    """
    
    conf = mol.GetConformer(conf_idx)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (x, y, z))

def _get_centre_of_mass(
    mol: Chem.Mol,
    conf_idx: int = -1
) -> np.ndarray:
    """
    Gets the center of mass of a molecule.

    Args:
        mol (Chem.Mol): Molecule.
        conf_idx (int, optional): Conformer index. Defaults to -1.

    Returns:
        np.ndarray: Center of mass ([x, y, z]).
    """
    
    masses = np.array([atom.GetMass() for atom in mol.GetAtoms()])
    coords = get_coords(mol, conf_idx = conf_idx)
    centre_of_mass = (
        np.sum(coords * masses[:, None], axis = 0) / np.sum(masses)
    )

    return centre_of_mass

def _set_centre_of_mass(
    mol: Chem.Mol,
    com: np.ndarray,
    conf_idx: int = -1,
) -> None:
    """
    Sets the center of mass of a molecule via translation in Cartesian
    coordinates such that the centre of mass is coincident with `com`.

    Args:
        mol (Chem.Mol): Molecule.
        com (np.ndarray): Center of mass ([x, y, z]).
        conf_idx (int, optional): Conformer index. Defaults to -1.
    """
    
    coords_init = get_coords(mol, conf_idx = conf_idx)
    com_init = _get_centre_of_mass(mol, conf_idx = conf_idx)

    set_coords(
        mol, translate_coords(coords_init, com - com_init), conf_idx = conf_idx
    )

def _angles_to_rotation_matrix(
    angles: np.ndarray
) -> np.ndarray:
    """
    Returns the 3D rotation matrix for a set of (Euler) rotation angles
    ([a, b, c]) in radians.

    Args:
        angles (np.ndarray, optional): Rotation angles ([a, b, c]) in radians.

    Returns:
        np.ndarray: 3D rotation matrix of shape (3, 3).
    """
    
    alpha, beta, gamma = angles

    rx = np.array([
        [1.0,            0.0,            0.0          ],
        [0.0,            np.cos(alpha), -np.sin(alpha)],
        [0.0,            np.sin(alpha),  np.cos(alpha)]
    ])
    ry = np.array([
        [ np.cos(beta),  0.0,             np.sin(beta)],
        [ 0.0,           1.0,             0.0         ],
        [-np.sin(beta),  0.0,             np.cos(beta)]
    ])
    rz = np.array([
        [np.cos(gamma), -np.sin(gamma),   0.0         ],
        [np.sin(gamma),  np.cos(gamma),   0.0         ],
        [0.0,            0.0,             1.0         ]
    ])

    return rz @ ry @ rx

def spherical_to_cartesian(
    r: float,
    theta: float,
    phi: float
) -> np.ndarray:
    """
    Converts spherical coordinates ([r, theta, phi]) to Cartesian coordinates
    ([x, y, z]).

    Args:
        r (float): Radial distance.
        theta (float): Zenith angle in radians.
        phi (float): Azimuthal angle in radians.

    Returns:
        np.ndarray: Cartesian coordinates ([x, y, z]).
    """
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([x, y, z])

def cartesian_to_spherical(
    x: float,
    y: float,
    z: float
) -> np.ndarray:
    """
    Converts Cartesian coordinates ([x, y, z]) to spherical coordinates
    ([r, theta, phi]).

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.
        z (float): X coordinate.

    Returns:
        np.ndarray: Spherical coordinates ([r, theta, phi]).
    """

    r = np.sqrt((x**2) + (y**2) + (z**2))
    if r == 0.0:
        return np.array([0.0, 0.0, 0.0])
    else:
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return np.array([r, theta, phi])

def get_vdw_distance_matrix(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    vdw_scaling: float = 1.0
) -> np.ndarray:
    """
    Returns a matrix of van der Waals distances between `mol1` and `mol2` as an
    array of shape (n_atoms$_{mol2}$, n_atoms$_{mol1}$) where each element, ij,
    corresponds to the sum of the van der Waals radii of the i$^{th}$ atom in
    `mol2` and the j$^{th}$ atom in `mol2`.

    Args:
        mol1 (Chem.Mol): Host molecule.
        mol2 (Chem.Mol): Guest molecule.
        vdw_scaling (float, optional): Scaling factor for the van der Waals
            radii. Defaults to 1.0.

    Returns:
        np.ndarray: Van der Waals distance matrix as an array of shape
            (n_atoms$_{mol2}$, n_atoms$_{mol1}$).
    """
    
    pt = Chem.GetPeriodicTable() 
    
    mol1_vdw = np.array(
        [pt.GetRvdw(atom.GetSymbol()) for atom in mol1.GetAtoms()]
    ) * vdw_scaling
    mol2_vdw = np.array(
        [pt.GetRvdw(atom.GetSymbol()) for atom in mol2.GetAtoms()]
    ) * vdw_scaling
    
    vdw_distance_matrix = mol2_vdw + mol1_vdw[:, None]

    return vdw_distance_matrix

def get_rmsd(
    mol1: Chem.Mol,
    mol2: Chem.Mol
) -> float:
    """
    Returns the root-mean-squared difference (RMSD) of the atomic positions in
    Cartesian coordinates between `mol1` and `mol2`; the RMSD is calculated
    directly *without* any alignment of the geometries of `mol1` and `mol2`.

    Args:
        mol1 (Chem.Mol): Molecule #1.
        mol2 (Chem.Mol): Molecule #2.

    Returns:
        float: RMSD of the atomic positions between `mol1` and `mol2`.
    """
    
    squared_diff = np.sum(
        (get_coords(mol1) - get_coords(mol2))**2, axis = 1
    )

    mean_squared_diff = np.mean(squared_diff)
      
    rmsd = np.sqrt(mean_squared_diff)
    
    return rmsd

def get_rmsd_matrix(
    mols: list[Chem.Mol],
    mem_safety_fraction: float = 0.8
) -> np.ndarray:
    """
    Calculates the pairwise root-mean-squared distance (RMSD) matrix for a
    list of molecules; the pairwise RMSD matrix is a square symmetric matrix
    where each element, ij, corresponds to the RMSD between the i$^{th}$ and
    the j$^{th}$ molecules in `mols`.

    Args:
        mols (list[Chem.Mol]): List of molecules.
        mem_safety_fraction (float, optional): Fraction of the available
            memory ring-fenced as 'safe for use'. Defaults to 0.8.

    Returns:
        np.ndarray: RMSD matrix as an array of shape (n_mols, n_mols).
    """

    print('calculating the pairwise RMSD matrix:')

    n_mols = len(mols)
    n_atoms = mols[0].GetNumAtoms()

    estimated_mem = _estimate_mem_for_rmsd_matrix_calc(
        n_mols, n_atoms
    )

    available_mem = (
        psutil.virtual_memory().available * mem_safety_fraction
    )

    print(f'- available mem: {(available_mem / (1024**3)):.2f} GB')
    print(f'- estimated mem: {(estimated_mem / (1024**3)):.2f} GB')

    if estimated_mem < available_mem:
        print('- calculation method: full vectorisation\n')
        return _get_rmsd_matrix_vectorised(mols)
    else:
        print('- calculation method: block vectorisation\n')
        block_size = _get_optimal_block_size(
            n_mols, n_atoms, available_mem
        )
        return _get_rmsd_matrix_block_vectorised(
            mols, block_size = block_size
        )

def _get_rmsd_matrix_vectorised(
    mols: list[Chem.Mol]
) -> np.ndarray:
    """
    Calculates the pairwise root-mean-squared distance (RMSD) matrix for a
    list of molecules in a single vectorised operation using broadcasting.

    This is a super-fast, but memory-intensive, approach.

    Args:
        mols (list[Chem.Mol]): List of molecules.

    Returns:
        np.ndarray: RMSD matrix as an array of shape (n_mols, n_mols).
    """
    
    coords = np.array([get_coords(mol) for mol in mols])

    coords1 = coords[:, np.newaxis, :, :]
    coords2 = coords[np.newaxis, :, :, :]

    squared_diff = np.sum((coords1 - coords2)**2, axis = -1)
    mean_squared_diff = np.mean(squared_diff, axis = -1)
    rmsd_matrix = np.sqrt(mean_squared_diff)

    return rmsd_matrix

def _get_rmsd_matrix_block_vectorised(
    mols: list[Chem.Mol],
    block_size: int = 100
) -> np.ndarray:
    """
    Calculates the pairwise root-mean-squared distance (RMSD) matrix for a
    list of molecules in a series of blocked ('chunk-by-chunk') vectorised
    operations using broadcasting.

    This is an approach that balances speed and memory usage.

    Args:
        mols (list[Chem.Mol]): List of molecules.

    Returns:
        np.ndarray: RMSD matrix as an array of shape (n_mols, n_mols).
    """
    
    rmsd_matrix = np.zeros((len(mols), len(mols)))

    coords = np.array([get_coords(mol) for mol in mols])

    for block1_start in range(0, len(mols), block_size):
        block1_end = min(block1_start + block_size, len(mols))
        block1_coords = coords[block1_start:block1_end]
        coords1 = block1_coords[:, np.newaxis, :, :]

        for block2_start in range(0, len(mols), block_size):
            block2_end = min(block2_start + block_size, len(mols))
            block2_coords = coords[block2_start:block2_end]
            coords2 = block2_coords[np.newaxis, :, :, :]

            squared_diff = np.sum((coords1 - coords2)**2, axis = -1)
            mean_squared_diff = np.mean(squared_diff, axis = -1)
            rmsd_matrix[block1_start:block1_end, block2_start:block2_end] = (
                np.sqrt(mean_squared_diff)
            )

    return rmsd_matrix

def _get_rmsd_matrix_iterative(
    mols: list[Chem.Mol]
) -> np.ndarray:
    """
    Calculates the pairwise root-mean-squared distance (RMSD) matrix for a
    list of molecules iteratively by looping over all $i$-$j$ pairs.

    This is a super-slow, but memory-efficient, approach.

    Args:
        mols (list[Chem.Mol]): List of molecules.

    Returns:
        np.ndarray: RMSD matrix as an array of shape (n_mols, n_mols).
    """
    
    rmsd_matrix = np.zeros([len(mols), len(mols)])
    for i in range(len(mols)):
        for j in range(i+1, len(mols)):
            rmsd_matrix[i,j] = rmsd_matrix[j,i] = get_rmsd(mols[i], mols[j])

    return rmsd_matrix

def _get_optimal_block_size(
    n_mols: int,
    n_atoms: int,
    available_mem: float
) -> int:
    """
    Calculates the optimal block size to use in the block-vectorised
    calculation of the RMSD matrix implemented in the \`_get_rmsd_matrix_
    block_vectorised()\` function.

    Args:
        n_mols (int): Number of molecules.
        n_atoms (int): Number of atoms per molecule.
        available_mem (float): Available memory (in bytes).

    Returns:
        int: Optimal block size.
    """
    
    estimated_mem = _estimate_mem_for_rmsd_matrix_calc(
        n_mols = n_mols, n_atoms = n_atoms
    )

    mem_ratio = (available_mem / estimated_mem)**0.5
    optimal_block_size = max(1, min(int(mem_ratio * n_mols), n_mols))

    return optimal_block_size

def _estimate_mem_for_rmsd_matrix_calc(
    n_mols: int,
    n_atoms: int
) -> float:
    """
    Estimates the memory usage (in bytes) of the intermediate arrays used in
    the vectorised calculation of the RMSD matrix implemented in the
    `_get_rmsd_matrix_vectorised()\` function.

    Args:
        n_mols (int): Number of molecules.
        n_atoms (int): Number of atoms per molecule.

    Returns:
        float: Estimated memory usage (in bytes).
    """
    
    estimated_mem = (
        (n_mols * n_atoms * 3 * 8)              # coords
        + (2 * (n_mols * n_atoms * 3 * 8))      # broadcasting intermediates
        + (n_mols * n_mols * n_atoms * 8)       # squared diff
        + (n_mols * n_mols * 8)                 # mean squared diff
        + (n_mols * n_mols * 8)                 # rmsd_matrix
    )
    
    return estimated_mem
