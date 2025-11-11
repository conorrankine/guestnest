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
        'host_sdf',
        type = Path,
        help = 'path to an .sdf/mol file for the host molecule'
    )
    p.add_argument(
        'guest_sdf',
        type = Path,
        help = 'path to an .sdf/mol file for the guest molecule'
    )
    p.add_argument(
        '-o', '--output_f',
        type = Path, default = './host_guest_complex.sdf',
        help = 'path to an output .sdf/mol file for the host-guest complex'
    )
    p.add_argument(
        '-n', '--n_complexes',
        type = int, default = 1,
        help = 'number of host-guest geometries to generate'
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

    host = Chem.MolFromMolFile(args.host_sdf, removeHs = False)
    guest = Chem.MolFromMolFile(args.guest_sdf, removeHs = False)
    for mol, label in zip((host, guest), ('host', 'guest')):
        print(
            f'{label:<5} | '
            f'n. atoms: {mol.GetNumAtoms():>3} | '
            f'charge: {Chem.GetFormalCharge(mol):>3} |'
        )
    print()

    rng = default_rng(args.random_seed)

    host_guest_complexes = []

    for _ in tqdm.tqdm(
        range(args.n_complexes),
        desc = 'creating complexes',
        bar_format = (
            '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ),
        ncols = 60
    ):

        host_guest_complex, opt = optimise.random_fit(
            host,
            guest,
            maxiter = args.maxiter,
            host_cavity_dims = args.host_cavity_dims,
            vdw_scaling = args.vdw_scaling,
            rng = rng
        )

        host_guest_complex = optimise.optimise_geom_xtb(
            host_guest_complex,
            fixed_atoms = [i for i in range(host.GetNumAtoms())]
        )

        host_guest_complex.SetDoubleProp(
            'E(XTB)', optimise.eval_energy_xtb(host_guest_complex)
        )

        host_guest_complexes.append(host_guest_complex)

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
        f'{"E(XTB) / a.u.":>18}'
    )
    print('-' * 24)
    for i, host_guest_complex in enumerate(host_guest_complexes, start = 1):
        print(
            f'{i:06d}'
            f'{host_guest_complex.GetDoubleProp("E(XTB)"):>18.6f}'
        )
    print('-' * 24 + '\n')

    with Chem.SDWriter(args.output_f) as writer:
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