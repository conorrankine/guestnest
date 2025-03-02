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
from scipy.optimize import basinhopping

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def optimise_fit(
    host: Chem.Mol,
    guest: Chem.Mol,
    niter: int = 100,
    max_distances: tuple = (5.0, 5.0, 5.0),
    max_angles: tuple = (np.pi, np.pi, np.pi)
) -> Chem.Mol:
    
    host_coords = _get_coords(host)
    guest_coords = _get_coords(guest)

    x0 = np.zeros(6)

    bounds = (
        [[-1.0 * x, x] for x in max_distances] +
        [[-1.0 * x, x] for x in max_angles]
    )

    res = basinhopping(
        objective,
        x0,
        niter = niter,
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': bounds,
            'args': (host_coords, guest_coords)
        }
    )
    
    return None

def objective(
    x,
    host_coords: np.ndarray,
    guest_coords: np.ndarray
) -> float:
    
    return 0.0

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