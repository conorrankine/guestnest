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
    delta = com - com_init

    set_coords(mol, coords_init + delta, conf_idx = conf_idx)

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
        z (float): Z coordinate.

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

# =============================================================================
#                                     EOF
# =============================================================================