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
from .geometry import get_coords, set_coords

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

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

# =============================================================================
#                                     EOF
# =============================================================================