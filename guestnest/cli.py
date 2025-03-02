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

from argparse import ArgumentParser, Namespace
from pathlib import Path

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

    args = p.parse_args()

    return args

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################

def main():

    pass

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################

if __name__ == '__main__':
    main()

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################
