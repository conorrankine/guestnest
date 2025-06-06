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
from scipy.spatial.distance import squareform
from .geometry import get_rmsd_matrix

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

def unique_mols(
    mols: list[Chem.Mol],
    rmsd_threshold: float = 0.1,
    method: str = 'centroid'
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
        method (str, optional): Method for picking representative molecules:
            - 'centroid': select the molecule with the minimum average RMSD
                to all other molecules in a given cluster;
            - 'medoid': select the molecule with the minimum sum RMSD to all
                other molecules in a given cluster.
            Defaults to 'centroid'.

    Returns:
        list[Chem.Mol]: List of representative molecules for each cluster.
    """

    if len(mols) <= 1:
        return mols
    
    print('deduplicating complexes via heirarchical clustering on RMSD:\n')
    
    rmsd_matrix = get_rmsd_matrix(mols)

    cluster_assignments = fcluster(
        linkage(squareform(rmsd_matrix), method = 'average'),
        t = rmsd_threshold,
        criterion = 'distance'
    )

    cluster_representatives = _pick_cluster_representatives(
        cluster_assignments,
        rmsd_matrix,
        method = method
    )

    unique_mols = [mols[idx] for idx in cluster_representatives]

    print(
        'heirarchical clustering on RMSD complete:\n' + 
        f'- n. complexes (pre-deduplication): {len(mols)}\n' +
        f'- n. complexes (post-deduplication): {len(unique_mols)}\n'
    )

    return unique_mols

def _pick_cluster_representatives(
    cluster_assignments: np.ndarray,
    distance_matrix: np.ndarray,
    method: str = 'centroid'
) -> list[int]:
    """
    Picks a single representative datapoint from each cluster according to a
    protocol (e.g., centroids, medoids) and returns their indices.
    
    Args:
        cluster_assignments (np.ndarray): 1D array of integer labels where
            the value of the i$^{th}$ element indicates the cluster to
            which the i$^{th}$ datapoint is assigned membership.
        distance_matrix (np.ndarray): Distance matrix; square symmetric matrix
            of shape (n, n) where distance_matrix[i,j] stores the distance
            between the i$^{th}$ and j$^{th}$ cluster members.
        method (str, optional): Method for picking representative datapoints:
            - 'centroid': select the datapoint with the minimum average
                distance to all other datapoints in a given cluster;
            - 'medoid': select the datapoint with the minimum sum distance to
                all other datapoints in a given cluster.
            Defaults to 'centroid'.
            
    Returns:
        list[int]: Sorted list of indices corresponding to the representative
            datapoint for each cluster.
            
    Raises:
        ValueError: If `method` is not one of 'centroid' or 'medoid'.
    """
    
    representatives = []

    method_map = {
        'centroid': _get_cluster_centroid,
        'medoid': _get_cluster_medoid
    }

    if method not in method_map:
        raise ValueError(
            f'`method` should be one of {[key for key in method_map.keys()]}; '
            f'got {method}'
        )

    cluster_group_map = _group_indices_by_cluster(cluster_assignments)

    for cluster_id in sorted(cluster_group_map.keys()):
        cluster_indices = cluster_group_map[cluster_id]
        if len(cluster_indices) == 1:
            representatives.append(
                cluster_indices[0]
            )
        else:
            representatives.append(
                method_map[method](cluster_indices, distance_matrix)
            )

    return sorted(representatives)    

def _group_indices_by_cluster(
    cluster_assignments: np.ndarray
) -> dict[int, np.ndarray]:
    """
    Returns a dictionary mapping cluster assigments to the indices of the
    cluster members.

    Args:
        cluster_assignments (np.ndarray): 1D array of integer labels where
            the value of the i$^{th}$ element indicates the cluster to
            which the i$^{th}$ datapoint is assigned membership.

    Returns:
        dict[int, np.ndarray]: Dictionary mapping cluster assigments to the
            indices of the cluster members where:
                - keys are integer labels for the clusters;
                - values are 1D arrays containing the indices of datapoints
                  that are assigned membership to the cluster.
    """

    return {
        cluster_id: np.where(cluster_assignments == cluster_id)[0]
            for cluster_id in np.unique(cluster_assignments)
    }

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
            [distance_matrix[i, j] for j in cluster_indices if j != i]
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
            [distance_matrix[i, j] for j in cluster_indices if j != i]
        )
        sum_distances.append(sum_distance)
    
    return cluster_indices[np.argmin(sum_distances)]