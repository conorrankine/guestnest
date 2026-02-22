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
from dataclasses import dataclass
from rdkit import Chem
from scipy.stats import qmc
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from typing import Generator
from .xtb_wrapper import XTBCalculator
from .geometry import (
    centre,
    get_coords,
    set_coords,
    spherical_to_cartesian,
    cartesian_to_spherical,
    get_vdw_distance_matrix
)

# =============================================================================
#                                   CLASSES
# =============================================================================

@dataclass(frozen = True, slots = True)
class FitResult:
    """
    Container for host-guest fitting outputs.

    Note:
        `frozen = True` enforces shallow immutability only; mutable objects in
        the container, e.g., `pose` (`Chem.Mol`), `opt_x` (`np.ndarray`), and
        `valid_metrics` (`dict`), are still able to be modified in place.
    """

    pose: Chem.Mol
    opt_success: bool
    opt_fun: float
    opt_x: np.ndarray
    valid: bool
    valid_metrics: dict[str, float]

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

def generate_initial_poses(
    n_samples: int = 1,
    host_cavity_dims: np.ndarray | None = None,
    theta_range: tuple[float, float] = (0.0, np.pi),
    phi_range: tuple[float, float] = (0.0, 2.0 * np.pi),
    rng: np.random.Generator = None
) -> Generator[np.ndarray, None, None]:
    """
    Yields quasi-random poses inside a symmetric ellipsoidal cavity.
    
    Uses a Sobol sequence sampler for low-discrepancy sampling of:
    - Cartesian positions ([x, y, z]) within the symmetric ellipsoidal cavity,
      optionally constrained by zenith (θ) and azimuthal (φ) angular ranges;
    - rotations ([rx, ry, rz]) derived from quaternions.

    Args:
        n_samples (int): Number of initial poses to generate.
        host_cavity_dims (np.ndarray | None, optional): 3-element array of
            per-axis scale factors (semi-axes) for the symmetric ellipsoidal
            cavity. Defaults to the unit cube ([1.0, 1.0, 1.0]).
        theta_range (tuple[float, float], optional): Zenith (θ) angle limits
            (radians; 0 = +Z). Defaults to (0.0, π).
        phi_range (tuple[float, float], optional): Azimuthal (φ) angle limits
            (radians). Defaults to (0.0, 2π).
        rng (np.random.Generator, optional): Random number generator used to
            initialise the Sobol sequence sampler. If None, a default random
            number generator is used.

    Yields:
        np.ndarray: 6-element array ([x, y, z, rx, ry, rz]) comprising the
            Cartesian position vector ([x, y, z]) and rotation vector
            ([rx, ry, rz]) that define the quasi-random pose.
    """
    
    if host_cavity_dims is None:
        host_cavity_dims = np.array([1.0, 1.0, 1.0])

    sampler = qmc.Sobol(d = 6, scramble = True, rng = rng)

    for sample in sampler.random(n_samples):
        positions = _sample_position(
            sample[:3], theta_range, phi_range, host_cavity_dims
        )
        rotations = _sample_rotation(
            sample[3:]
        )
        yield np.hstack((positions, rotations))

def fit(
    host: Chem.Mol,
    guest: Chem.Mol,
    x0: np.ndarray,
    host_cavity_dims: np.ndarray,
    vdw_scaling: float = 1.0,
    maxiter: int = 100
) -> FitResult:
    """
    Optimises a guest molecule pose inside a host molecule by minimising a two-
    part penalty function.

    Args:
        host (Chem.Mol): Host molecule.
        guest (Chem.Mol): Guest molecule.
        x0 (np.ndarray): 6-element array ([x, y, z, rx, ry, rz]) comprising the
            Cartesian position vector ([x, y, z]) and rotation vector
            ([rx, ry, rz]) that define the (initial) guest molecule pose.
        host_cavity_dims (np.ndarray): 3-element array of per-axis scale
            factors (semi-axes) for the symmetric ellipsoidal cavity.
        vdw_scaling (float, optional): Scaling factor applied to vdW radii when
            computing the vdW overlap penalty. Defaults to 1.0.
        maxiter (int, optional): Maximum number of optimisation iterations.
            Defaults to 100.

    Returns:
        FitResult: Dataclass containing the fitted host-guest complex (`pose`),
            optimiser outputs (`opt_success`; `opt_fun`; `opt_x`), and pose
            validity diagnostics (`valid`; `valid_metrics`).
    """

    host_cavity_dims = np.array(host_cavity_dims)

    host_centred = centre(host)
    guest_centred = centre(guest)

    vdw_distance_matrix = get_vdw_distance_matrix(
        host, guest, vdw_scaling = vdw_scaling
    )

    opt = minimize(
        _objective_function,
        x0 = x0,
        args = (
            get_coords(host_centred),
            get_coords(guest_centred),
            host_cavity_dims,
            vdw_distance_matrix
        ),
        method = 'Powell',
        options = {
            'maxiter': maxiter,
            'disp': False
        }
    )

    position_vector, rotation_vector = np.split(opt.x, 2)
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()

    guest_coords = get_coords(guest_centred)
    guest_coords_transformed = (
        (rotation_matrix @ guest_coords.T).T + position_vector
    )
    set_coords(guest_centred, guest_coords_transformed)

    valid, valid_metrics = _is_valid(
        get_coords(host_centred),
        get_coords(guest_centred),
        host_cavity_dims,
        vdw_distance_matrix
    )

    host_guest_complex = Chem.CombineMols(host_centred, guest_centred)
    
    return FitResult(
        pose = host_guest_complex,
        opt_success = opt.success,
        opt_fun = opt.fun,
        opt_x = opt.x,
        valid = valid,
        valid_metrics = valid_metrics
    )

def _sample_position(
    sample: np.ndarray,
    theta_range: tuple[float, float],
    phi_range: tuple[float, float],
    host_cavity_dims: np.ndarray
) -> np.ndarray:
    """
    Returns a Cartesian position vector ([x, y, z]) derived from a 3-element
    array of Sobol components ([u1, u2, u3]).

    Args:
        sample (np.ndarray): 3-element array of Sobol components ([u1, u2, u3])
            used to sample a Cartesian position.
        theta_range (tuple[float, float]): Zenith (θ) angle limits (radians;
            0 = +Z).
        phi_range (tuple[float, float]): Azimuthal (φ) angle limits (radians).
        host_cavity_dims (np.ndarray): 3-element array of per-axis scale
            factors (semi-axes) for the symmetric ellipsoidal cavity.

    Returns:
        np.ndarray: Cartesian position vector ([x, y, z]).
    """

    theta_min, theta_max = theta_range
    phi_min, phi_max = phi_range
    
    u1, u2, u3 = sample
    r = u1 ** (1.0 / 3.0)
    theta = np.arccos(
        np.cos(theta_min) + (np.cos(theta_max) - np.cos(theta_min)) * u2
    )
    phi = phi_min + (phi_max - phi_min) * u3
    position = spherical_to_cartesian(r, theta, phi) * host_cavity_dims

    return position

def _sample_rotation(
    sample: np.ndarray
) -> np.ndarray:
    """
    Returns a rotation vector ([rx, ry, rz]) derived from a 3-element array of
    Sobol components ([u1, u2, u3]).

    Args:
        sample (np.ndarray): 3-element array of Sobol components ([u1, u2, u3])
            used to sample a rotation.

    Returns:
        np.ndarray: Rotation vector ([rx, ry, rz]).
    """
    
    u1, u2, u3 = sample
    quat = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    quat /= np.linalg.norm(quat)
    rotation = R.from_quat(quat).as_rotvec()

    return rotation

def _is_valid(
    host_coords: np.ndarray,
    guest_coords: np.ndarray,
    host_cavity_dims: np.ndarray,
    vdw_distance_matrix: np.ndarray,
    max_cavity_pos_tol: float = 1.15,
    min_ratio_tol: float = 0.65,
) -> tuple[bool, dict[str, float]]:
    """
    Evaluates whether a pose is valid based on cavity encapsulation and vdW
    separation criteria.

    Checks:
    - cavity encapsulation: the guest atoms are located inside a symmetric
        ellipsoidal cavity (max normalised position <= `max_cavity_pos_tol`);
    - vdW separation: the minimum host-guest distance (scaled by the vdW
      distance matrix) is >= `min_ratio_tol`.

    Args:
        host_coords (np.ndarray): Host molecule Cartesian coordinates.
        guest_coords (np.ndarray): Guest molecule Cartesian coordinates.
        host_cavity_dims (np.ndarray): 3-element array of per-axis scale
            factors (semi-axes) for the symmetric ellipsoidal cavity.
        vdw_distance_matrix (np.ndarray): Pairwise vdW distance matrix between
            host molecule and guest molecule atoms.
        max_cavity_pos_tol (float, optional): Tolerance for cavity
            encapsulation check. Defaults to 1.15.
        min_ratio_tol (float, optional): Tolerance for vdW separation check;
            minimum allowed host-guest distance (scaled by the vdW distance
            matrix). Defaults to 0.65.

    Returns:
        tuple[bool, dict]: Validity flag (True/False) and a metrics dictionary
            containing the keys `max_cavity_pos` and `min_ratio`.
    """

    cavity_pos = np.sum(
        ((guest_coords**2) / (host_cavity_dims**2)), axis = 1
    )
    max_cavity_pos = float(np.max(cavity_pos))

    distance_matrix = np.linalg.norm(
        host_coords[:, None, :] - guest_coords[None, :, :], axis = -1
    )
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        ratio = distance_matrix / vdw_distance_matrix
        ratio = ratio[np.isfinite(ratio)]
    min_ratio = float(np.min(ratio)) if ratio.size else 0.0

    metrics = {
        'max_cavity_pos': max_cavity_pos,
        'min_ratio': min_ratio,
    }

    valid = (
        (max_cavity_pos <= max_cavity_pos_tol)
        and (min_ratio >= min_ratio_tol)
    )
    
    return valid, metrics

def _objective_function(
    x,
    host_coords: np.ndarray,
    guest_coords: np.ndarray,
    host_cavity_dims: np.ndarray,
    vdw_distance_matrix: np.ndarray
) -> float:

    position_vector, rotation_vector = np.split(x, 2)
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()

    guest_coords_transformed = (
        (rotation_matrix @ guest_coords.T).T + position_vector
    )

    return _penalty_function(
        host_coords,
        guest_coords_transformed,
        host_cavity_dims,
        vdw_distance_matrix
    )

def _penalty_function(
    host_coords: np.ndarray,
    guest_coords: np.ndarray,
    host_cavity_dims: np.ndarray,
    vdw_distance_matrix: np.ndarray,
    return_components: bool = False
) -> float | tuple[float]:
    
    overlap_penalty = _get_overlap_penalty(
        host_coords, guest_coords, vdw_distance_matrix
    )

    cavity_penalty = _get_cavity_penalty(
        guest_coords, host_cavity_dims
    )

    penalty = overlap_penalty + cavity_penalty
    if return_components:
        return penalty, overlap_penalty, cavity_penalty
    else:
        return penalty

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

def optimise_geom_xtb(
    mol: Chem.Mol,
    fixed_atoms: list[int] = None
) -> Chem.Mol:

    calculator = XTBCalculator(
        mol,
        engine = 'ancopt' if not fixed_atoms else 'lbfgs'
    )

    if fixed_atoms is not None:
        for fixed_atom in fixed_atoms:
            calculator.AddFixedPoint(fixed_atom)
            
    calculator.Minimize()

    return mol

def eval_energy_xtb(
    mol: Chem.Mol
) -> float:
    
    calculator = XTBCalculator(mol)

    return calculator.CalcEnergy()

# =============================================================================
#                                     EOF
# =============================================================================
