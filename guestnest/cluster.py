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
from scipy.cluster.hierarchy import linkage, fcluster
from .geometry import get_rmsd

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

def unique_mols(
    mols: list[Chem.Mol],
    rmsd_threshold: float = 0.1
) -> list[Chem.Mol]:
    """
    Clusters molecules hierarchically using the root-mean-squared distance
    (RMSD) as the similarity metric and returns a list of representative
    molecules for each cluster.

    Args:
        mols (list[Chem.Mol]): List of molecules to cluster hierarchically.
        rmsd_threshold (float, optional): RMSD threshold for hierarchical
            clustering in Angstroems (molecules with RMSD below this threshold
            will be considered to belong to the same cluster). Defaults to 0.1.

    Returns:
        list[Chem.Mol]: List of representative molecules for each cluster.
    """
    
    rmsd_matrix = _get_rmsd_matrix(mols)

    linkage_matrix = _get_linkage_matrix(rmsd_matrix)

    clusters = fcluster(
        linkage_matrix,
        t = rmsd_threshold,
        criterion = 'distance'
    )

    unique_mols = [
        mols[idx] for idx in _get_cluster_representative_idx(clusters)
    ]

    return unique_mols

def _get_rmsd_matrix(
    mols: list[Chem.Mol]
) -> np.ndarray:
    """
    Calculates the pairwise root-mean-squared distance (RMSD) matrix for a
    list of molecules; the pairwise RMSD matrix is a square symmetric matrix
    where each element, ij, corresponds to the RMSD between the i$^{th}$ and
    the j$^{th}$ molecules in `mols`.

    Args:
        mols (list[Chem.Mol]): List of molecules.

    Returns:
        np.ndarray: RMSD matrix as an array of shape (n_mols, n_mols).
    """
    
    rmsd_matrix = np.zeros([len(mols), len(mols)])
    for i in range(len(mols)):
        for j in range(i+1, len(mols)):
            rmsd_matrix[i,j] = rmsd_matrix[j,i] = get_rmsd(mols[i], mols[j])

    return rmsd_matrix

def _get_linkage_matrix(
    matrix: np.ndarray,
    method: str = 'average',
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Calculates the hierarchical clustering linkage matrix from a square
    symmetric distance matrix.

    Args:
        matrix (np.ndarray): Distance matrix as an array of shape (n, n).
        method (str, optional): Linkage method; see the documentation for
            scipy.cluster.hierarchy.linkage() for a full list of options.
            Defaults to 'average'.
        metric (str, optional): Linkage metric; see the documentation for
            scipy.cluster.hierarchy.linkage() for a full list of options.
            Defaults to 'euclidean'.

    Raises:
        ValueError: If `matrix` is not a square matrix.

    Returns:
        np.ndarray: Hierarchical clustering linkage matrix.
    """
    
    dim1, dim2 = matrix.shape
    if dim1 != dim2:
        raise ValueError(
            '`matrix` should be an array of shape (n, n), i.e. a 2D array '
            f'with equal dimensions; got an array of shape {matrix.shape}'
        )

    return linkage(
        matrix[np.triu_indices(dim1, k = 1)],
        method = method,
        metric = metric
    )

def _get_cluster_representative_idx(
    clusters: np.ndarray
) -> list[int]:
    """
    Returns a sorted list of indices defining representatives for each cluster
    in `clusters`.

    Args:
        clusters (np.ndarray): Cluster assignments as an array of shape (n, )
            where each element is an integer that indicates cluster membership.

    Returns:
        list[int]: Sorted list of indices defining representatives for each
            cluster in `clusters`.
    """
    
    cluster_map = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = idx

    return sorted(cluster_map.values())

def _get_cluster_centroid(
    cluster_indices: np.ndarray,
    distance_matrix: np.ndarray
) -> int:
    """
    Returns the index corresponding to the centroid of the cluster.
    
    Args:
        cluster_indices (np.ndarray): Indices defining cluster members.
        distance_matrix (np.ndarray): Distance matrix; square symmetric matrix
            of shape (n, n) where distance_matrix[i,j] stores the distance
            between the i$^{th}$ and j$^{th}$ cluster members.
        
    Returns:
        int: Index corresponding to the centroid of the cluster.
    """
    
    avg_distances = []
    for i in cluster_indices:
        avg_distance = np.mean(
            distance_matrix[i, j] for j in cluster_indices if j != i
        )
        avg_distances.append(avg_distance)
    
    return cluster_indices[np.argmin(avg_distances)]

def _get_cluster_medoid(
    cluster_indices: np.ndarray,
    distance_matrix: np.ndarray
) -> int:
    """
    Returns the index corresponding to the medoid of the cluster.
    
    Args:
        cluster_indices (np.ndarray): Indices defining cluster members.
        distance_matrix (np.ndarray): Distance matrix; square symmetric matrix
            of shape (n, n) where distance_matrix[i,j] stores the distance
            between the i$^{th}$ and j$^{th}$ cluster members.
        
    Returns:
        int: Index corresponding to the medoid of the cluster.
    """
    
    sum_distances = []
    for i in cluster_indices:
        sum_distance = np.sum(
            distance_matrix[i, j] for j in cluster_indices if j != i
        )
        sum_distances.append(sum_distance)
    
    return cluster_indices[np.argmin(sum_distances)]