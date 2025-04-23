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
from scipy.cluster.hierarchy import linkage, fcluster

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

def unique_mols(
    mols: list[Chem.Mol],
    rmsd_threshold: float = 0.1
) -> list[Chem.Mol]:
    
    rmsd_matrix = _get_rmsd_matrix(mols)

    linkage_matrix = _get_linkage_matrix(rmsd_matrix)

    clusters = fcluster(
        linkage_matrix,
        t = rmsd_threshold,
        criterion = 'distance'
    )

    unique_mol_idx = _get_cluster_representative_idx(clusters)

def _get_rmsd_matrix(
    mols: list[Chem.Mol]
) -> np.ndarray:
    
    rmsd_matrix = np.zeros([len(mols), len(mols)])
    for i in range(len(mols)):
        for j in range(i+1, len(mols)):
            rmsd_matrix[i,j] = rmsd_matrix[j,i] = GetBestRMS(mols[i], mols[j])

    return rmsd_matrix

def _get_linkage_matrix(
    matrix: np.ndarray,
    method: str = 'average',
    metric: str = 'euclidean'
) -> np.ndarray:
    
    dim1, dim2 = matrix.shape
    if dim1 != dim2:
        raise ValueError(
            '`matrix` should be an array of shape (N,N), i.e. a 2D array with '
            f'equal dimensions; got an array of shape {matrix.shape}'
        )

    return linkage(
        matrix[np.triu_indices(dim1, k = 1)],
        method = method,
        metric = metric
    )

def _get_cluster_representative_idx(
    clusters: np.ndarray
) -> list[int]:
    
    cluster_map = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = idx

    return sorted(cluster_map.values())