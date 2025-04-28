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
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFGetMoleculeForceField,
    MMFFGetMoleculeProperties
)
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult
from .geometry import (
    centre,
    get_coords,
    set_coords,
    rotate_and_translate_coords,
    rotate_and_translate_mol,
    spherical_to_cartesian,
    cartesian_to_spherical,
    get_vdw_distance_matrix
)

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

def random_fit(
    host: Chem.Mol,
    guest: Chem.Mol,
    maxiter: int = 100,
    host_cavity_dims: list = [4.0, 4.0, 4.0],
    vdw_scaling: float = 1.0,
    rng: np.random.Generator = None
) -> tuple[Chem.Mol, OptimizeResult]:
    
    if rng is None:
        rng = np.random.default_rng()

    host_cavity_dims = np.array(host_cavity_dims)

    host_ = centre(host)
    guest_ = centre(guest)
    
    bounds = np.array([
        [0.0, 1.0],             # radial distance
        [0.0, np.pi],           # zenith angle
        [0.0, 2.0 * np.pi],     # azimuthal angle
        [0.0, np.pi],           # rotation angle (x)
        [0.0, np.pi],           # rotation angle (y)
        [0.0, np.pi]            # rotation angle (z)
    ])

    x0 = rng.uniform(bounds[:,0], bounds[:,1])

    vdw_distance_matrix = get_vdw_distance_matrix(
        host, guest, vdw_scaling = vdw_scaling
    )

    opt = minimize(
        _objective_function,
        x0 = x0,
        args = (
            get_coords(host_),
            get_coords(guest_),
            host_cavity_dims,
            vdw_distance_matrix
        ),
        options = {
            'maxiter': maxiter,
            'disp': False
        }
    )

    opt_spherical_coords, opt_rotation_angles = np.split(opt.x, 2)

    opt_cartesian_coords = (
        spherical_to_cartesian(*opt_spherical_coords) * host_cavity_dims
    )

    fitted_guest_ = rotate_and_translate_mol(
        guest_, opt_rotation_angles, opt_cartesian_coords
    )

    host_guest_complex = Chem.CombineMols(host_, fitted_guest_)
    
    return host_guest_complex, opt

def _objective_function(
    x,
    host_coords: np.ndarray,
    guest_coords: np.ndarray,
    host_cavity_dims: np.ndarray,
    vdw_distance_matrix: np.ndarray
) -> float:

    spherical_coords, rotation_angles = np.split(x, 2)

    cartesian_coords = (
        spherical_to_cartesian(*spherical_coords) * host_cavity_dims
    )

    transformed_guest_coords = rotate_and_translate_coords(
        guest_coords, rotation_angles, cartesian_coords
    )

    return _penalty_function(
        host_coords,
        transformed_guest_coords,
        host_cavity_dims,
        vdw_distance_matrix
    )

def _penalty_function(
    host_coords: np.ndarray,
    guest_coords: np.ndarray,
    host_cavity_dims: np.ndarray,
    vdw_distance_matrix: np.ndarray
) -> float:
    
    overlap_penalty = _get_overlap_penalty(
        host_coords, guest_coords, vdw_distance_matrix
    )

    cavity_penalty = _get_cavity_penalty(
        guest_coords, host_cavity_dims
    )

    return overlap_penalty + cavity_penalty

def _get_overlap_penalty(
    host_coords: np.ndarray,
    guest_coords: np.ndarray,
    vdw_distance_matrix: np.ndarray
) -> float:

    distance_matrix = np.linalg.norm(
        host_coords[:, None, :] - guest_coords[None, :, :], axis = -1
    )

    overlap_penalty_matrix = vdw_distance_matrix - distance_matrix
    overlap_penalty_matrix[overlap_penalty_matrix < 0] = 0

    return np.sum(np.square(overlap_penalty_matrix))

def _get_cavity_penalty(
    guest_coords: np.ndarray,
    host_cavity_dims: np.ndarray
) -> float:
    
    cavity_pos = np.sum(
        ((guest_coords**2) / (host_cavity_dims**2)), axis = 1
    )

    cavity_boundary_violations = cavity_pos - 1
    cavity_boundary_violations[cavity_boundary_violations < 0] = 0

    return np.sum(np.square(cavity_boundary_violations))    

def optimise_geom_mmff(
    mol: Chem.Mol,
    fixed_atoms: list[int] = None
) -> Chem.Mol:

    ff = MMFFGetMoleculeForceField(
        mol, MMFFGetMoleculeProperties(mol)
    )

    if fixed_atoms is not None:
        for fixed_atom in fixed_atoms:
            ff.MMFFAddPositionConstraint(fixed_atom, 0.0, 1.0E5)
            
    ff.Minimize()

    return mol

def _eval_energy_mmff(
    mol: Chem.Mol
) -> float:
    
    ff = MMFFGetMoleculeForceField(
        mol, MMFFGetMoleculeProperties(mol)
    )

    return ff.CalcEnergy()