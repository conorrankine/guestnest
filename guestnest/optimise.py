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
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFGetMoleculeForceField,
    MMFFGetMoleculeProperties
)
from scipy.optimize import basinhopping
from scipy.optimize._optimize import OptimizeResult

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def centre(
    mol: Chem.Mol,
    conf_idx: int = -1
) -> None:
    
    centre_of_mass = _get_centre_of_mass(mol, conf_idx = conf_idx)
    centred_coords = _translate(
        _get_coords(mol, conf_idx = conf_idx), -1.0 * centre_of_mass
    )
    _set_coords(mol, centred_coords, conf_idx = conf_idx)

def optimise_geom_mmff(
    mol: Chem.Mol,
    fixed_atoms: list[int] = None
) -> Chem.Mol:

    ff = MMFFGetMoleculeForceField(
        mol, MMFFGetMoleculeProperties(mol)
    )

    if fixed_atoms is not None:
        for fixed_atom in fixed_atoms:
            ff.AddFixedPoint(fixed_atom)

    ff.Minimize()

    return mol

def optimise_fit(
    host: Chem.Mol,
    guest: Chem.Mol,
    niter: int = 100,
    stepsize: float = 2.5,
    temperature: float = 5.0,
    distance_threshold: float = 2.0,
    alpha: float = 1.0,
    beta: float = 0.01,
    max_distances: tuple = (5.0, 5.0, 5.0),
    max_angles: tuple = (np.pi, np.pi, np.pi)
) -> tuple[Chem.Mol, OptimizeResult]:
    
    bounds = np.array(
        [[-1.0 * x, x] for x in max_distances] +
        [[-1.0 * x, x] for x in max_angles]
    )

    x0 = np.random.uniform(bounds[:,0], bounds[:,1])

    opt = basinhopping(
        _objective_function,
        x0,
        niter = niter,
        stepsize = stepsize,
        T = temperature,
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': bounds,
            'args': (host, guest, distance_threshold, alpha, beta)
        },
        disp = True
    )

    opt_distances, opt_angles = np.split(opt.x, 2)
    opt_guest_coords = _transform_coords(
        _get_coords(guest), opt_distances, opt_angles
    )
    _set_coords(guest, opt_guest_coords)

    host_guest_complex = Chem.CombineMols(host, guest)
    
    return host_guest_complex, opt

def _objective_function(
    x,
    host: Chem.Mol,
    guest: Chem.Mol,
    distance_threshold: float = 2.0,
    alpha: float = 1.0,
    beta: float = 0.01
) -> float:
    
    guest_copy = Chem.Mol(guest)
       
    distances, angles = np.split(x, 2)
    transformed_guest_coords = _transform_coords(
        _get_coords(guest_copy), distances, angles
    )
    _set_coords(guest_copy, transformed_guest_coords)
    
    return _penalty_function(
        host,
        guest_copy,
        distance_threshold = distance_threshold,
        alpha = alpha,
        beta = beta
    )

def _penalty_function(
    host: Chem.Mol,
    guest: Chem.Mol,
    distance_threshold: float = 2.0,
    alpha: float = 1.0,
    beta: float = 0.01
) -> float:
    
    host_coords = _get_coords(host)
    guest_coords = _get_coords(guest)
    
    distance_matrix = np.linalg.norm(
        host_coords[:, None, :] - guest_coords[None, :, :], axis = -1
    )

    distance_penalty = (
        np.sum(
            np.square(
                np.maximum(0, (distance_threshold - distance_matrix))
            )
        )
    )

    energy_penalty = _eval_energy(
        Chem.CombineMols(host, guest)
    )
        
    return (alpha * distance_penalty) + (beta * energy_penalty)

def _transform_coords(
    coords: np.ndarray,
    distances: np.ndarray,
    angles: np.ndarray
) -> np.ndarray:
    
    return _translate(_rotate(coords, angles,), distances)

def _get_centre_of_mass(
    mol: Chem.Mol,
    conf_idx: int = -1
) -> np.ndarray:
    
    masses = np.array([atom.GetMass() for atom in mol.GetAtoms()])
    coords = _get_coords(mol, conf_idx = conf_idx)
    centre_of_mass = (
        np.sum(coords * masses[:, None], axis = 0) / np.sum(masses)
    )

    return centre_of_mass

def _get_coords(
    mol: Chem.Mol,
    conf_idx: int = -1
) -> np.ndarray:
    
    conf = mol.GetConformer(conf_idx)
    return np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )

def _set_coords(
    mol: Chem.Mol,
    coords: np.ndarray,
    conf_idx: int = -1
) -> None:
    
    conf = mol.GetConformer(conf_idx)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (x, y, z))

def _translate(
    coords: np.ndarray,
    distances: np.ndarray
) -> np.ndarray:
    
    return coords + distances

def _rotate(
    coords: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    
    return coords @ _angles_to_rotation_matrix(angles).T

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

def _eval_energy(
    mol: Chem.Mol
) -> float:
    
    ff = MMFFGetMoleculeForceField(
        mol, MMFFGetMoleculeProperties(mol)
    )

    return ff.CalcEnergy()
