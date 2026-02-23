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
import numpy as np
from .core import run
from argparse import ArgumentParser, Namespace
from pathlib import Path

# =============================================================================
#                                LOGGING SETUP
# =============================================================================

import logging

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
        '--theta_range',
        type = float, nargs = 2, default = [0.0, np.pi],
        help = 'zenith (θ) angle limits (radians; 0 = +Z)'
    )
    p.add_argument(
        '--phi_range',
        type = float, nargs = 2, default = [0.0, 2.0 * np.pi],
        help = 'azimuthal (φ) angle limits (radians)'
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

    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt = '%H:%M:%S',
        force = True
    )

    logger = logging.getLogger(__name__)

    datetime_ = datetime.datetime.now()
    logger.info(f'launched @ {datetime_.strftime("%H:%M:%S (%Y-%m-%d)")}')

    args = parse_args()

    run(
        host_f = args.host_f,
        guest_f = args.guest_f,
        output_f = args.output_f,
        n_complexes = args.n_complexes,
        host_cavity_dims = args.host_cavity_dims,
        theta_range = args.theta_range,
        phi_range = args.phi_range,
        vdw_scaling = args.vdw_scaling,
        rmsd_threshold = args.rmsd_threshold,
        energy_threshold = args.energy_threshold,
        random_seed = args.random_seed
    )

    datetime_ = datetime.datetime.now()
    logger.info(f'finished @ {datetime_.strftime("%H:%M:%S (%Y-%m-%d)")}')

# =============================================================================
#                             PROGRAM STARTS HERE
# =============================================================================

if __name__ == '__main__':
    main()

# =============================================================================
#                              PROGRAM ENDS HERE
# =============================================================================