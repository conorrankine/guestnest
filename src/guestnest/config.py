"""
GUESTNEST
Copyright (C) 2026  Conor D. Rankine

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

# =============================================================================
#                               LIBRARY IMPORTS
# =============================================================================

from dataclasses import dataclass, field
from math import isfinite, pi
from pathlib import Path

# =============================================================================
#                                  CONSTANTS
# =============================================================================

SUPPORTED_STRUCTURE_SUFFIXES = frozenset({'.sdf', '.xyz'})

# =============================================================================
#                                   CLASSES
# =============================================================================

@dataclass(frozen = True, slots = True)
class PoseConfig:

    n_complexes: int = 1
    cavity_dimensions: tuple[float, float, float] = (4.0, 4.0, 4.0)
    theta_range: tuple[float, float] = (0.0, pi)
    phi_range: tuple[float, float] = (0.0, 2.0 * pi)
    vdw_scaling: float = 1.0
    rmsd_threshold: float = 0.1
    random_seed: int | None = None

    def __post_init__(self) -> None:

        if self.n_complexes < 1:
            raise ValueError('n_complexes must be at least one')
        if len(self.cavity_dimensions) != 3 or (
            not all(isfinite(value) for value in self.cavity_dimensions)
            or not all(value > 0.0 for value in self.cavity_dimensions)
        ):
            raise ValueError(
                'cavity_dimensions must contain three finite, positive values'
            )

        if len(self.theta_range) != 2:
            raise ValueError('theta_range must contain two values')
        theta_min, theta_max = self.theta_range
        if (
            not all(isfinite(value) for value in self.theta_range)
            or not 0.0 <= theta_min <= theta_max <= pi
        ):
            raise ValueError(
                'theta limits must satisfy 0 <= MIN <= MAX <= pi'
            )

        if len(self.phi_range) != 2:
            raise ValueError('phi_range must contain two values')
        phi_min, phi_max = self.phi_range
        if (
            not all(isfinite(value) for value in self.phi_range)
            or phi_min > phi_max
        ):
            raise ValueError('phi limits must be finite and satisfy MIN <= MAX')
        if not isfinite(self.vdw_scaling) or self.vdw_scaling <= 0.0:
            raise ValueError('vdw_scaling must be finite and positive')
        if not isfinite(self.rmsd_threshold) or self.rmsd_threshold < 0.0:
            raise ValueError('rmsd_threshold must be finite and non-negative')
        if self.random_seed is not None and self.random_seed < 0:
            raise ValueError('random_seed must be non-negative')


@dataclass(frozen = True, slots = True)
class XTBConfig:

    charge: int | None = None
    uhf: int | None = None

    def __post_init__(self) -> None:

        if self.uhf is not None and self.uhf < 0:
            raise ValueError('uhf must be non-negative')


@dataclass(frozen = True, slots = True)
class GuestnestConfig:

    host_file: Path
    guest_file: Path
    output_file: Path = Path('host_guest_complex.sdf')
    pose: PoseConfig = field(default_factory = PoseConfig)
    xtb: XTBConfig = field(default_factory = XTBConfig)

    def __post_init__(self) -> None:

        object.__setattr__(self, 'host_file', Path(self.host_file))
        object.__setattr__(self, 'guest_file', Path(self.guest_file))
        object.__setattr__(self, 'output_file', Path(self.output_file))

        for path, label in (
            (self.host_file, 'host input'),
            (self.guest_file, 'guest input'),
            (self.output_file, 'output')
        ):
            if path.suffix.lower() not in SUPPORTED_STRUCTURE_SUFFIXES:
                supported_suffixes = ', '.join(
                    sorted(SUPPORTED_STRUCTURE_SUFFIXES)
                )
                raise ValueError(
                    f'unsupported {label} file extension: {path.suffix}; '
                    f'expected one of {{{supported_suffixes}}}'
                )

# =============================================================================
#                                     EOF
# =============================================================================
