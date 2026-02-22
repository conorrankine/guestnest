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

from abc import ABC, abstractmethod
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

# =============================================================================
#                                   CLASSES
# =============================================================================

class BaseMultiWriter(ABC):
    """
    Base class for writing multiple molecules into a single file.

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

    @abstractmethod
    def write(
        self,
        mol: Chem.Mol,
        conf_id: int = -1
    ) -> None:
        """
        Writes a single molecule to the open output file.

        Args:
            mol (Chem.Mol): Molecule.
            conf_id (int, optional): Conformer ID. Defaults to -1.
        """
        
        pass

    def close(self) -> None:

        if not self._file.closed:
            self._file.close()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc, tb):

        self.close()

class MultiXYZWriter(BaseMultiWriter):
    """
    Writes multiple molecules into a single .xyz file.
    """

    def write(
        self,
        mol: Chem.Mol,
        conf_id: int = -1
    ) -> None:
        """
        Writes a single molecule in .xyz format to the open output file.

        Args:
            mol (Chem.Mol): Molecule.
            conf_id (int, optional): Conformer ID. Defaults to -1.
        """

        write_xyz(
            self._file,
            mol,
            conf_id = conf_id,
            energy_prop = self.energy_prop
        )

class MultiSDFWriter(BaseMultiWriter):
    """
    Writes multiple molecules into a single .sdf file.
    """

    def write(
        self,
        mol: Chem.Mol,
        conf_id: int = -1
    ) -> None:
        """
        Writes a single molecule in .sdf format to the open output file.

        Args:
            mol (Chem.Mol): Molecule.
            conf_id (int, optional): Conformer ID. Defaults to -1.
        """

        write_sdf(
            self._file,
            mol,
            conf_id = conf_id,
            energy_prop = self.energy_prop
        )

# =============================================================================
#                                  FUNCTIONS
# =============================================================================

def read(
    input_f: str | Path,
    remove_hs: bool = False,
    strict: bool = True
) -> Chem.Mol | None:
    """
    Reads a molecule from an input structure file (.xyz/.sdf) by extension.

    Args:
        input_f (str | Path): Path to the input structure file.
        remove_hs (bool, optional): If `True`, explicit hydrogens are stripped.
            Defaults to `False`.
        strict (bool, optional): If `True`, raises a RuntimeError when a
            molecule cannot be read. Defaults to `True`.

    Raises:
        ValueError: If the input file extension is unsupported, i.e., if the
            file suffix is not one of {'.xyz', '.sdf'}.

    Returns:
        Chem.Mol | None: Molecule, else `None` if reading fails and `strict` is
            set to `False`.
    """

    input_f = Path(input_f)
    input_suffix = input_f.suffix.lower()

    if input_suffix == '.sdf':
        return read_sdf(input_f, remove_hs = remove_hs, strict = strict)
    elif input_suffix == '.xyz':
        return read_xyz(input_f, remove_hs = remove_hs, strict = strict)
    else:
        raise ValueError(
            f'unsupported input file extension (\'{input_f.suffix}\'); '
            f'expected one of {{\'.xyz\', \'.sdf\'}}'
        )

def read_sdf(
    input_f: str | Path,
    remove_hs: bool = False,
    strict: bool = True
) -> Chem.Mol | None:
    """
    Reads a molecule from an .sdf file.

    Args:
        input_f (str | Path): Path to the input .sdf file.
        remove_hs (bool, optional): If `True`, explicit hydrogens are stripped.
            Defaults to `False`.
        strict (bool, optional): If `True`, raises a RuntimeError when a
            molecule cannot be read. Defaults to `True`.

    Raises:
        RuntimeError: If `strict` is `True` and a molecule cannot be read from
            `input_f`.

    Returns:
        Chem.Mol | None: Molecule, else `None` if reading fails and `strict` is
            set to `False`.
    """

    mol = Chem.MolFromMolFile(str(input_f), removeHs = remove_hs)
    if mol is None and strict:
        raise RuntimeError(
            f'could not read molecule from {input_f} '
            f'[Chem.MolFromMolFile({input_f}) returned `None`]'
        )

    return mol

def read_xyz(
    input_f: str | Path,
    remove_hs: bool = False,
    strict: bool = True
) -> Chem.Mol | None:
    """
    Reads a molecule from an .xyz file.

    Args:
        input_f (str | Path): Path to the input .xyz file.
        remove_hs (bool, optional): If `True`, explicit hydrogens are stripped.
            Defaults to `False`.
        strict (bool, optional): If `True`, raises a RuntimeError when a
            molecule cannot be read. Defaults to `True`.

    Raises:
        RuntimeError: If `strict` is `True` and a molecule cannot be read from
            `input_f`;
        RuntimeError: If `strict` is `True` and molecular connectivity cannot
            be determined from the .xyz coordinates in `input_f`.

    Returns:
        Chem.Mol | None: Molecule, else `None` if reading fails and `strict` is
            set to `False`.
    """

    mol = Chem.MolFromXYZFile(str(input_f))
    if mol is None and strict:
        raise RuntimeError(
            f'could not read molecule from {input_f} '
            f'[Chem.MolFromXYZFile({input_f}) returned `None`]'
        )
    if mol is not None:
        try:
            rdDetermineBonds.DetermineBonds(mol)
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f'could not determine connectivity for {input_f}: {e}'
                ) from e
    if mol is not None and remove_hs:
        mol = Chem.RemoveHs(mol)

    return mol

def write(
    output_f: str | Path,
    mol: Chem.Mol,
    conf_id: int = -1,
    energy_prop: str = 'energy'
) -> None:
    """
    Writes a molecule to an output structure file (.xyz/.sdf) by extension.

    The output file is opened in exclusive creation mode and cannot already
    exist.

    Args:
        output_f (str | Path): Path to the output structure file.
        mol (Chem.Mol): Molecule.
        conf_id (int, optional): Conformer ID. Defaults to -1.
        energy_prop (str, optional): Conformer property key containing the
            energy value. Defaults to 'energy'.

    Raises:
        ValueError: If the output file extension is unsupported, i.e., if the
            file suffix is not one of {'.xyz', '.sdf'}.
    """

    output_f = Path(output_f)
    output_suffix = output_f.suffix.lower()

    with output_f.open('x') as file:
        if output_suffix == '.sdf':
            write_sdf(
                file,
                mol,
                conf_id = conf_id,
                energy_prop = energy_prop
            )
        elif output_suffix == '.xyz':
            write_xyz(
                file,
                mol,
                conf_id = conf_id,
                energy_prop = energy_prop
            )
        else:
            raise ValueError(
                f'unsupported output file extension (\'{output_f.suffix}\'); '
                f'expected one of {{\'.xyz\', \'.sdf\'}}'
            )

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
