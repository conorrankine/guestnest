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

import datetime
import tqdm
from . import optimise
from . import deduplicate
from .io import read, MultiSDFWriter, MultiXYZWriter
from numpy.random import default_rng
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rdkit import Chem

# =============================================================================
#                               ARGUMENT PARSING
# =============================================================================

def parse_args() -> Namespace:
    """
    Parses command line arguments for `guestnest:cli.py`.

    Returns:
        argparse.Namespace: Parsed command line arguments as an
        argparse.Namespace object that holds the arguments as attributes.
    """

    p = ArgumentParser()

    p.add_argument(
        'host_f',
        type = Path,
        help = 'path to an input structure file containing the host molecule'
    )
    p.add_argument(
        'guest_f',
        type = Path,
        help = 'path to an input structure file containing the guest molecule'
    )
    p.add_argument(
        '-o', '--output_f',
        type = Path, default = './host_guest_complex.sdf',
        help = 'path to an output structure file for host-guest complex(es)'
    )
    p.add_argument(
        '-n', '--n_complexes',
        type = int, default = 1,
        help = 'maximum number of host-guest geometries to generate'
    )
    p.add_argument(
        '-d', '--host_cavity_dims',
        type = float, nargs = 3, default = [4.0, 4.0, 4.0],
        help = ('dimensions ([x, y, z]) of the spherical (if x = y = z) or '
            'elliptical (x = y != z) host molecule cavity')
    )
    p.add_argument(
        '-s', '--vdw_scaling',
        type = float, default = 1.0,
        help = 'scaling factor for van der Waals radii'
    )
    p.add_argument(
        '-t', '--rmsd_threshold',
        type = float, default = 0.1,
        help = 'RMSD threshold (Angstroem) RMSD-based duplication'
    )
    p.add_argument(
        '-e', '--energy_threshold',
        type = float, default = 5E-3,
        help = 'energy threshold (kcal/mol) for energy-based deduplication'
    )
    p.add_argument(
        '-i', '--maxiter',
        type = int, default = 250,
        help = 'max number of iterations for host-guest geometry generation'
    )
    p.add_argument(
        '-r', '--random_seed',
        type = int, default = None,
        help = 'random seed for host-guest geometry generation'
    )

    args = p.parse_args()

    return args

# =============================================================================
#                                MAIN FUNCTION
# =============================================================================

def main():

    datetime_ = datetime.datetime.now()
    print(f'launched @ {datetime_.strftime("%H:%M:%S (%Y-%m-%d)")}\n')

    args = parse_args()

    output_suffix = args.output_f.suffix.lower()
    if output_suffix == '.sdf':
        writer_cls = MultiSDFWriter
    elif output_suffix == '.xyz':
        writer_cls = MultiXYZWriter
    else:
        raise ValueError(
            f'unsupported output file extension: {args.output_f.suffix}; '
            f'expected one of {{\'.sdf\', \'.xyz\'}}'
        )

    host, guest = read(args.host_f), read(args.guest_f)
    for mol, label in zip((host, guest), ('host', 'guest')):
        print(
            f'{label:<5} | '
            f'n. atoms: {mol.GetNumAtoms():>3} | '
            f'charge: {Chem.GetFormalCharge(mol):>3} |'
        )
    print()

    rng = default_rng(args.random_seed)

    host_guest_complexes = []

    samples = optimise.generate_initial_poses(
        n_samples = args.n_complexes,
        host_cavity_dims = args.host_cavity_dims,
        rng = rng
    )

    for sample in tqdm.tqdm(
        samples,
        desc = 'creating complexes',
        total = args.n_complexes,
        bar_format = (
            '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ),
        ncols = 60
    ):

        fit_result = optimise.fit(
            host,
            guest,
            sample,
            args.host_cavity_dims,
            vdw_scaling = args.vdw_scaling,
            maxiter = args.maxiter
        )

        if fit_result.opt_success and fit_result.valid:
            host_guest_complex = fit_result.pose
            host_guest_complex = optimise.optimise_geom_xtb(
                host_guest_complex,
                fixed_atoms = [i for i in range(host.GetNumAtoms())]
            )
            energy = optimise.eval_energy_xtb(host_guest_complex)
            host_guest_complex.SetDoubleProp('E(XTB)', energy)
            host_guest_complex.GetConformer().SetDoubleProp('E(XTB)', energy)
            host_guest_complexes.append(host_guest_complex)
        elif fit_result.opt_success and not fit_result.valid:
            valid_metrics = fit_result.valid_metrics
            tqdm.tqdm.write(
                f'pose failed validation: '
                f'max. cavity pos. = {valid_metrics["max_cavity_pos"]:.3f} | '
                f'min. vdW ratio = {valid_metrics["min_ratio"]:.3f}'
            )
        else:
            tqdm.tqdm.write(
                f'pose fitting failed'
            )

    if host_guest_complexes:
        host_guest_complexes = deduplicate.by_rmsd(
            host_guest_complexes,
            rmsd_threshold = args.rmsd_threshold,
            heavy_atoms_only = True
        )
        host_guest_complexes = deduplicate.by_energy(
            host_guest_complexes,
            energy_threshold = args.energy_threshold
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
        with writer_cls(args.output_f, energy_prop = 'E(XTB)') as writer:
            for host_guest_complex in host_guest_complexes:
                writer.write(host_guest_complex)

    datetime_ = datetime.datetime.now()
    print(f'finished @ {datetime_.strftime("%H:%M:%S (%Y-%m-%d)")}')

# =============================================================================
#                             PROGRAM STARTS HERE
# =============================================================================

if __name__ == '__main__':
    main()

# =============================================================================
#                              PROGRAM ENDS HERE
# =============================================================================