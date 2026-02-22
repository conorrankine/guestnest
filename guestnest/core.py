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
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm
from .optimise import (
    generate_initial_poses,
    fit,
    optimise_geom_xtb,
    eval_energy_xtb
)
from . import deduplicate
from .io import (
    read,
    MultiSDFWriter,
    MultiXYZWriter
)

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

def run(
    host_f: str | Path,
    guest_f: str | Path,
    output_f: str | Path = './host_guest_complex.sdf',
    n_complexes: int = 1,
    host_cavity_dims: tuple[float, float, float] = (1.0, 1.0, 1.0),
    theta_range: tuple[float, float] = (0.0, np.pi),
    phi_range: tuple[float, float] = (0.0, 2.0 * np.pi),
    vdw_scaling: float = 1.0,
    rmsd_threshold: float = 0.1,
    energy_threshold: float = 5E-3,
    maxiter: int = 250,
    random_seed: int | None = None
) -> list[Chem.Mol]:
    """
    Runs the host-guest complex generation workflow.

    Args:
        host_f (str | Path): Path to an input structure file for the host
            molecule.
        guest_f (str | Path): Path to an input structure file for the guest
            molecule.
        output_f (str | Path, optional): Path to the output structure file for
            generated host-guest complex(es). Defaults to
            './host_guest_complex.sdf'.
        n_complexes (int, optional): Maximum number of host-guest geometries to
            generate. Defaults to 1.
        host_cavity_dims (tuple[float, float, float], optional): 3-element array
            of per-axis scale factors (semi-axes; Angstroem) for the symmetric
            ellipsoidal cavity. Defaults to the unit cube ([1.0, 1.0, 1.0]).
        theta_range (tuple[float, float], optional): Zenith (θ) angle limits
            (radians; 0 = +Z). Defaults to (0.0, π).
        phi_range (tuple[float, float], optional): Azimuthal (φ) angle limits
            (radians). Defaults to (0.0, 2π).
        vdw_scaling (float, optional): Scaling factor for van der Waals radii.
            Defaults to 1.0.
        rmsd_threshold (float, optional): RMSD threshold (Angstroem) for RMSD-
            based deduplication. Defaults to 0.1.
        energy_threshold (float, optional): Energy threshold (kcal/mol) for
            energy-based deduplication. Defaults to 5E-3.
        maxiter (int, optional): Maximum number of optimisation iterations for
            host-guest geometry generation. Defaults to 250.
        random_seed (int | None, optional): Random seed for host-guest geometry
            generation. Defaults to None.

    Returns:
        list[Chem.Mol]: Generated host-guest complexes after filtering and
            deduplication.
    """

    output_f = Path(output_f)
    output_suffix = output_f.suffix.lower()
    if output_suffix == '.sdf':
        writer_cls = MultiSDFWriter
    elif output_suffix == '.xyz':
        writer_cls = MultiXYZWriter
    else:
        raise ValueError(
            f'unsupported output file extension: {output_f.suffix}; '
            f'expected one of {{\'.sdf\', \'.xyz\'}}'
        )

    host, guest = read(host_f), read(guest_f)
    for mol, label in zip((host, guest), ('host', 'guest')):
        print(
            f'{label:<5} | '
            f'n. atoms: {mol.GetNumAtoms():>3} | '
            f'charge: {Chem.GetFormalCharge(mol):>3} |'
        )
    print()

    rng = np.random.default_rng(random_seed)

    host_guest_complexes: list[Chem.Mol] = []

    samples = generate_initial_poses(
        n_samples = n_complexes,
        host_cavity_dims = host_cavity_dims,
        theta_range = theta_range,
        phi_range = phi_range,
        rng = rng
    )

    for sample in tqdm(
        samples,
        desc = 'creating complexes',
        total = n_complexes,
        bar_format = (
            '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ),
        ncols = 60
    ):

        fit_result = fit(
            host,
            guest,
            sample,
            host_cavity_dims,
            vdw_scaling = vdw_scaling,
            maxiter = maxiter
        )

        if fit_result.opt_success and fit_result.valid:
            host_guest_complex = fit_result.pose
            host_guest_complex = optimise_geom_xtb(
                host_guest_complex,
                fixed_atoms = [i for i in range(host.GetNumAtoms())]
            )
            energy = eval_energy_xtb(host_guest_complex)
            host_guest_complex.SetDoubleProp('E(XTB)', energy)
            host_guest_complex.GetConformer().SetDoubleProp('E(XTB)', energy)
            host_guest_complexes.append(host_guest_complex)
        elif fit_result.opt_success and not fit_result.valid:
            valid_metrics = fit_result.valid_metrics
            tqdm.write(
                f'pose failed validation: '
                f'max. cavity pos. = {valid_metrics["max_cavity_pos"]:.3f} | '
                f'min. vdW ratio = {valid_metrics["min_ratio"]:.3f}'
            )
        else:
            tqdm.write(
                f'pose fitting failed'
            )

    if host_guest_complexes:
        host_guest_complexes = deduplicate.by_rmsd(
            host_guest_complexes,
            rmsd_threshold = rmsd_threshold,
            heavy_atoms_only = True
        )
        host_guest_complexes = deduplicate.by_energy(
            host_guest_complexes,
            energy_threshold = energy_threshold
        )
        print('-' * 24)
        print(
            f'{"complex":<6}'
            f'{"E(XTB) (kcal/mol)":>18}'
        )
        print('-' * 24)
        for i, host_guest_complex in enumerate(host_guest_complexes, start = 1):
            energy = host_guest_complex.GetDoubleProp('E(XTB)')
            print(f'{i:06d}{energy:>18.6f}')
        print('-' * 24 + '\n')
        with writer_cls(output_f, energy_prop = 'E(XTB)') as writer:
            for host_guest_complex in host_guest_complexes:
                writer.write(host_guest_complex)

    return host_guest_complexes

# =============================================================================
#                                     EOF
# =============================================================================