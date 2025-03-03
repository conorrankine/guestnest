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

import datetime
from . import optimise
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rdkit import Chem

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################

def parse_args() -> Namespace:
    """
    Parses command line arguments for `guestnest:cli.py`.

    Returns:
        argparse.Namespace: Parsed command line arguments as an
        argparse.Namespace object that holds the arguments as attributes.
    """

    p = ArgumentParser()

    p.add_argument(
        'host_sdf', type = Path,
        help = 'path to an .sdf/mol file for the host molecule'
    )
    p.add_argument(
        'guest_sdf', type = Path,
        help = 'path to an .sdf/mol file for the guest molecule'
    )
    p.add_argument(
        '-n', '--niter', type = int, default = 25,
        help = 'number of iterations for the basin-hopping algorithm'
    )
    p.add_argument(
        '-t', '--temperature', type = float, default = 5.0,
        help = '"temperature" parameter for the basin-hopping algorithm'
    )

    args = p.parse_args()

    return args

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################

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

    optimise.centre(host)
    optimise.centre(guest)

    print('starting basin-hopping...')
    host_guest_complex, opt = optimise.optimise_fit(
        host,
        guest,
        niter = args.niter,
        temperature = args.temperature
    )
    print(f'...finished basin-hopping after {opt.nit} iterations!\n')

    print('optimising the host-guest complex using MMFF94...')
    host_guest_complex = optimise.optimise_geom_mmff(
        host_guest_complex,
        fixed_atoms = [i for i in range(host.GetNumAtoms())]
    )
    print('...optimised the host-guest complex!\n')

    writer = Chem.SDWriter('./host_guest_complex.sdf')
    writer.write(host_guest_complex)
    writer.close()

    datetime_ = datetime.datetime.now()
    print(f'finished @ {datetime_.strftime("%H:%M:%S (%Y-%m-%d)")}')

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################

if __name__ == '__main__':
    main()

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################
