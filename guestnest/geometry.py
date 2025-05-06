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
    mols: list[Chem.Mol]
) -> np.ndarray:
    """
    Calculates the pairwise root-mean-squared distance (RMSD) matrix for a
    list of molecules; the pairwise RMSD matrix is a square symmetric matrix
    where each element, ij, corresponds to the RMSD between the i$^{th}$ and
    the j$^{th}$ molecules in `mols`.

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