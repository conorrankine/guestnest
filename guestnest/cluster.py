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
from rdkit.Chem.rdMolAlign import GetBestRMS

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

def unique_mols(
    mols: list[Chem.Mol]
) -> list[Chem.Mol]:
    
    rmsd_matrix = _get_rmsd_matrix(mols)

def _get_rmsd_matrix(
    mols: list[Chem.Mol]
) -> np.ndarray:
    
    rmsd_matrix = np.zeros([len(mols), len(mols)])
    for i in range(len(mols)):
        for j in range(i+1, len(mols)):
            rmsd_matrix[i,j] = rmsd_matrix[j,i] = GetBestRMS(mols[i], mols[j])

    return rmsd_matrix            
