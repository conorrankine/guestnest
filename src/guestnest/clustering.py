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
this program. If not, see <https://www.gnu.org/licenses/>.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


def deduplicate_by_rmsd(
    mols: list[Chem.Mol],
    atom_indices: list[int],
    rmsd_threshold: float = 0.5,
    energy_property: str = 'E(XTB)'
) -> list[Chem.Mol]:
    """Retain the lowest-energy molecule from each RMSD cluster.

    RMSDs are calculated without alignment between the selected atoms in the
    coordinate frame shared by all conformers.

    Args:
        mols (list[Chem.Mol]): Molecules with identical atom ordering.
        atom_indices (list[int]): Atom indices to include in RMSD calculations.
        rmsd_threshold (float, optional): Butina clustering threshold in
            Angstroem. Defaults to 0.5.
        energy_property (str, optional): RDKit double property used to select
            the cluster representative. Defaults to 'E(XTB)'.

    Returns:
        list[Chem.Mol]: Lowest-energy molecule from each RMSD cluster.
    """

    if not mols:
        return []
    if not atom_indices:
        raise ValueError('at least one atom index is required for RMSD')

    atom_indices = sorted(set(atom_indices))
    conformer_mol = _select_atoms(mols[0], atom_indices)
    conformer_mol.RemoveAllConformers()
    for mol in mols:
        source_conformer = mol.GetConformer()
        conformer = Chem.Conformer(len(atom_indices))
        for new_idx, source_idx in enumerate(atom_indices):
            conformer.SetAtomPosition(
                new_idx,
                source_conformer.GetAtomPosition(source_idx)
            )
        conformer_mol.AddConformer(
            conformer,
            assignId = True
        )

    rmsd_distances = AllChem.GetConformerRMSMatrix(
        conformer_mol,
        prealigned = True
    )

    clusters = Butina.ClusterData(
        rmsd_distances,
        len(mols),
        rmsd_threshold,
        isDistData = True
    )

    keep_mols_idx = [
        min(
            cluster,
            key = lambda i: mols[i].GetDoubleProp(energy_property)
        )
        for cluster in clusters
    ]

    return [mols[i] for i in keep_mols_idx]


def _select_atoms(
    mol: Chem.Mol,
    atom_indices: list[int]
) -> Chem.Mol:
    """Return a copy of a molecule containing only selected atoms."""

    n_atoms = mol.GetNumAtoms()
    if atom_indices[-1] >= n_atoms or atom_indices[0] < 0:
        raise IndexError('RMSD atom index out of range')

    selected_indices = set(atom_indices)
    editable_mol = Chem.RWMol(mol)
    for atom_idx in reversed(range(n_atoms)):
        if atom_idx not in selected_indices:
            editable_mol.RemoveAtom(atom_idx)

    return editable_mol.GetMol()
