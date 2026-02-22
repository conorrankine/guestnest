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

from pathlib import Path
from rdkit import Chem

# =============================================================================
#                                   CLASSES
# =============================================================================

class MultiXYZWriter:
    """
    Writes multiple molecules into a single .xyz file.

    The output file (`output_f`) is opened in exclusive creation mode and must
    not already exist.
    """

    def __init__(
        self,
        output_f: str | Path,
        energy_prop: str = 'energy'
    ):

        self.output_f = Path(output_f)
        self.energy_prop = energy_prop
        self._file = self.output_f.open('x')

    def write(
        self,
        mol: Chem.Mol,
        conf_id: int = -1
    ) -> None:

        write_xyz(
            self._file,
            mol,
            conf_id = conf_id,
            energy_prop = self.energy_prop
        )

    def close(self) -> None:

        if not self._file.closed:
            self._file.close()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc, tb):

        self.close()

class MultiSDFWriter:
    """
    Writes multiple molecules into a single .sdf file.

    The output file (`output_f`) is opened in exclusive creation mode and must
    not already exist.
    """

    def __init__(
        self,
        output_f: str | Path,
        energy_prop: str = 'energy'
    ):

        self.output_f = Path(output_f)
        self.energy_prop = energy_prop
        self._file = self.output_f.open('x')

    def write(
        self,
        mol: Chem.Mol,
        conf_id: int = -1
    ) -> None:

        write_sdf(
            self._file,
            mol,
            conf_id = conf_id,
            energy_prop = self.energy_prop
        )

    def close(self) -> None:

        if not self._file.closed:
            self._file.close()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc, tb):

        self.close()

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

def write_xyz(
    file,
    mol: Chem.Mol,
    conf_id: int = -1,
    energy_prop: str = 'energy'
) -> None:
    """
    Writes a single molecule in .xyz format.

    The .xyz comment line is formatted as `<energy_prop> = <VALUE>` using
    the conformer property identified by `energy_prop`; `<VALUE>` is accessed
    via `conf.GetDoubleProp(<energy_prop>)`.

    Args:
        file: Writable text stream.
        mol (Chem.Mol): Molecule.
        conf_id (int, optional): Conformer ID. Defaults to -1.
        energy_prop (str, optional): Conformer property key containing the
            energy value. Defaults to 'energy'.
    """

    conf = mol.GetConformer(conf_id)
    energy = conf.GetDoubleProp(energy_prop)

    file.write(f'{mol.GetNumAtoms()}\n')
    file.write(f'{energy_prop} = {energy:.6f}\n')

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        position = conf.GetAtomPosition(atom_idx)
        file.write(
            f'{atom.GetSymbol():<2} '
            f'{position.x:>12.6f} '
            f'{position.y:>12.6f} '
            f'{position.z:>12.6f}\n'
        )

def write_sdf(
    file,
    mol: Chem.Mol,
    conf_id: int = -1,
    energy_prop: str = 'energy'
) -> None:
    """
    Writes a single molecule in .sdf format.

    The .sdf data block is populated with the field `<energy_prop>` and value
    `<VALUE>` using the conformer property identified by `energy_prop`;
    `<VALUE>` is accessed via `conf.GetDoubleProp(<energy_prop>)`.

    Args:
        file: Writable text stream.
        mol (Chem.Mol): Molecule.
        conf_id (int, optional): Conformer ID. Defaults to -1.
        energy_prop (str, optional): Conformer property key containing the
            energy value. Defaults to 'energy'.
    """

    conf = mol.GetConformer(conf_id)
    energy = conf.GetDoubleProp(energy_prop)

    file.write(Chem.MolToMolBlock(mol, confId = conf_id))
    file.write(f'>  <{energy_prop}>\n')
    file.write(f'{energy:.6f}\n\n')
    file.write(f'$$$$\n')

# =============================================================================
#                                     EOF
# =============================================================================
