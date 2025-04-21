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

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np
from rdkit import Chem
from copy import deepcopy

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

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
    
    return coords + distances

def rotate_coords(
    coords: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    
    return coords @ _angles_to_rotation_matrix(angles).T

def rotate_and_translate_coords(
    coords: np.ndarray,
    angles: np.ndarray,
    distances: np.ndarray
) -> np.ndarray:
    
    return translate_coords(rotate_coords(coords, angles), distances)

def get_coords(
    mol: Chem.Mol,
    conf_idx: int = -1
) -> np.ndarray:
    
    conf = mol.GetConformer(conf_idx)
    return np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )

def set_coords(
    mol: Chem.Mol,
    coords: np.ndarray,
    conf_idx: int = -1
) -> None:
    
    conf = mol.GetConformer(conf_idx)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (x, y, z))

def _get_centre_of_mass(
    mol: Chem.Mol,
    conf_idx: int = -1
) -> np.ndarray:
    
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
    
    coords_init = get_coords(mol, conf_idx = conf_idx)
    com_init = _get_centre_of_mass(mol, conf_idx = conf_idx)

    set_coords(
        mol, translate_coords(coords_init, com - com_init), conf_idx = conf_idx
    )

def _angles_to_rotation_matrix(
    angles = np.ndarray
) -> np.ndarray:
    
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
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([x, y, z])

def cartesian_to_spherical(
    x: float,
    y: float,
    z: float
) -> np.ndarray:

    r = np.sqrt((x**2) + (y**2) + (z**2))
    if r == 0.0:
        return np.array([0.0, 0.0, 0.0])
    else:
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return np.array([r, theta, phi])

def get_vdw_distance_matrix(
    host: Chem.Mol,
    guest: Chem.Mol,
    vdw_scaling: float = 1.0
) -> np.ndarray:
    
    pt = Chem.GetPeriodicTable() 
    
    host_vdw = np.array(
        [pt.GetRvdw(atom.GetSymbol()) for atom in host.GetAtoms()]
    ) * vdw_scaling
    guest_vdw = np.array(
        [pt.GetRvdw(atom.GetSymbol()) for atom in guest.GetAtoms()]
    ) * vdw_scaling
    
    vdw_distance_matrix = guest_vdw + host_vdw[:, None]

    return vdw_distance_matrix
