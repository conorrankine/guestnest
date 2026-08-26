"""
GUESTNEST
Copyright (C) 2026  Conor D. Rankine

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
"""

# =============================================================================
#                               LIBRARY IMPORTS
# =============================================================================

import numpy as np
from rdkit import Chem

from .clustering import deduplicate_by_rmsd
from .config import GuestnestConfig
from .io import (
    read,
    MultiSDFWriter,
    MultiXYZWriter
)
from .optimise import (
    generate_initial_poses,
    fit
)
from .xtb_wrapper import XTBCalculator

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

def run(config: GuestnestConfig) -> list[Chem.Mol]:
    """
    Runs the host-guest complex generation workflow.

    Args:
        config (GuestnestConfig): Validated run configuration.

    Returns:
        list[Chem.Mol]: Generated host-guest complexes after filtering and
            deduplication.
    """

    pose_config = config.pose
    xtb_config = config.xtb
    output_f = config.output_file
    writer_cls = {
        '.sdf': MultiSDFWriter,
        '.xyz': MultiXYZWriter
    }[output_f.suffix.lower()]

    host, guest = read(config.host_file), read(config.guest_file)
    for mol, label in zip((host, guest), ('host', 'guest')):
        print(
            f'{label:<5} | '
            f'n. atoms: {mol.GetNumAtoms():>3} | '
            f'charge: {Chem.GetFormalCharge(mol):>3} |'
        )
    print()

    rng = np.random.default_rng(pose_config.random_seed)

    host_guest_complexes: list[Chem.Mol] = []
    n_fit_failures = 0
    n_validation_failures = 0
    n_xtb_optimisation_failures = 0
    n_xtb_energy_failures = 0

    samples = generate_initial_poses(
        n_samples = pose_config.n_complexes,
        host_cavity_dims = pose_config.cavity_dimensions,
        theta_range = pose_config.theta_range,
        phi_range = pose_config.phi_range,
        rng = rng
    )

    for pose_idx, sample in enumerate(samples, start = 1):
        pose_label = f'pose {pose_idx:06d}/{pose_config.n_complexes:06d}'
        print(f'{pose_label} | started', flush = True)

        fit_result = fit(
            host,
            guest,
            sample,
            pose_config.cavity_dimensions,
            vdw_scaling = pose_config.vdw_scaling
        )

        if fit_result.opt_success and fit_result.valid:
            host_guest_complex = fit_result.pose

            calculator = XTBCalculator(
                host_guest_complex,
                engine = 'lbfgs',
                charge = xtb_config.charge,
                uhf = xtb_config.uhf
            )
            for atom_idx in range(host.GetNumAtoms()):
                calculator.AddFixedPoint(atom_idx)

            optimisation_status = calculator.Minimize()
            if optimisation_status != 0:
                n_xtb_optimisation_failures += 1
                print(
                    f'{pose_label} | XTB geometry optimisation failed',
                    flush = True
                )
                continue

            energy = XTBCalculator(
                host_guest_complex,
                charge = xtb_config.charge,
                uhf = xtb_config.uhf
            ).CalcEnergy()
            if not np.isfinite(energy):
                n_xtb_energy_failures += 1
                print(
                    f'{pose_label} | XTB energy calculation failed',
                    flush = True
                )
                continue

            host_guest_complex.SetDoubleProp('E(XTB)', energy)
            host_guest_complex.GetConformer().SetDoubleProp('E(XTB)', energy)
            host_guest_complexes.append(host_guest_complex)
            print(
                f'{pose_label} | accepted | E(XTB) = {energy:.6f} kcal/mol',
                flush = True
            )
        elif fit_result.opt_success and not fit_result.valid:
            n_validation_failures += 1
            valid_metrics = fit_result.valid_metrics
            print(
                f'{pose_label} | failed validation | '
                f'max. cavity pos. = {valid_metrics["max_cavity_pos"]:.3f} | '
                f'min. vdW ratio = {valid_metrics["min_ratio"]:.3f}',
                flush = True
            )
        else:
            n_fit_failures += 1
            print(
                f'{pose_label} | fitting failed | '
                f'objective fun. = {fit_result.opt_fun:.3f} | '
                f'n. iter. = {fit_result.opt_nit}',
                flush = True
            )

    print(
        'workflow summary:\n'
        f'- requested poses: {pose_config.n_complexes}\n'
        f'- pose fitting failures: {n_fit_failures}\n'
        f'- pose validation failures: {n_validation_failures}\n'
        f'- XTB optimisation failures: {n_xtb_optimisation_failures}\n'
        f'- XTB energy failures: {n_xtb_energy_failures}\n'
        f'- accepted before RMSD deduplication: {len(host_guest_complexes)}\n'
    )

    if host_guest_complexes:
        guest_heavy_atom_indices = [
            atom.GetIdx()
            for atom in host_guest_complexes[0].GetAtoms()
            if (
                atom.GetIdx() >= host.GetNumAtoms()
                and atom.GetAtomicNum() != 1
            )
        ]
        host_guest_complexes = deduplicate_by_rmsd(
            host_guest_complexes,
            atom_indices = guest_heavy_atom_indices,
            rmsd_threshold = pose_config.rmsd_threshold
        )
        host_guest_complexes = sorted(
            host_guest_complexes,
            key = lambda mol: mol.GetDoubleProp('E(XTB)')
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
