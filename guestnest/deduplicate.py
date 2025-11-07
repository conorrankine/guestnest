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

from rdkit import Chem
from rdkit.ML.Cluster import Butina
from .rmsd import get_rmsd, get_rmsd_matrix

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

def by_energy(
    mols: list[Chem.Mol],
    energy_threshold: float = 1E-3,
    energy_property: str = 'E(XTB)'
) -> list[Chem.Mol]:
    """
    Deduplicates a list of molecules by energy threshold; where two molecules
    have an energy difference (ΔE) below the energy threshold, the higher-
    energy molecule of the two is dropped.

    As a side-effect, the list of molecules returned is sorted in ascending
    order by energy.

    Args:
        mols (list[Chem.Mol]): List of molecules.
        energy_threshold (float, optional): Minimum absolute ΔE threshold;
            where two molecules have a ΔE below this value, the higher-energy
            molecule of the two is dropped. Defaults to 1E-3.
        energy_property (str, optional): RDKit double property key containing
            the energy value. Defaults to 'E(XTB)'.

    Returns:
        list[Chem.Mol]: List of molecules deduplicated by energy threshold.
    """

    sorted_mols = sorted(
        mols, key = lambda mol: mol.GetDoubleProp(energy_property)
    )

    deduplicated_mols: list[Chem.Mol] = []
    last_energy: float = None
    for mol in sorted_mols:
        energy = mol.GetDoubleProp(energy_property)
        if (
            (last_energy is not None)
            and (abs(energy - last_energy) < energy_threshold)
        ):
            continue
        deduplicated_mols.append(mol)
        last_energy = energy
    
    return deduplicated_mols

def by_rmsd(
    mols: list[Chem.Mol],
    rmsd_threshold: float = 0.5,
    energy_property: str = 'E(XTB)'
) -> list[Chem.Mol]:
    """
    Deduplicates a list of molecules by RMSD threshold; where two (or more)
    molecules are grouped into the same cluster by Butina clustering on intra-
    cluster RMSD, the higher-energy molecule(s) is(/are) dropped.

    Args:
        mols (list[Chem.Mol]): List of molecules.
        rmsd_threshold (float, optional): Maximum intra-cluster RMSD for the
            Butina clustering algorithm in Angstroem. Defaults to 0.5.
        energy_property (str, optional): RDKit double property key containing
            the energy value. Defaults to 'E(XTB)'.

    Returns:
        list[Chem.Mol]: List of molecules deduplicated by Butina clustering on
            intra-cluster RMSD.
    """
    
    rmsd_matrix = get_rmsd_matrix(mols)

    clusters = Butina.ClusterData(
        rmsd_matrix,
        len(mols),
        rmsd_threshold,
        isDistData = True
    )

    keep_mols_idx: list[int] = []
    for cluster in clusters:
        keep_mols_idx.append(
            min(
                cluster,
                key = lambda i: mols[i].GetDoubleProp(energy_property)
            )
        )

    deduplicated_mols = [mols[i] for i in keep_mols_idx]

    return deduplicated_mols
