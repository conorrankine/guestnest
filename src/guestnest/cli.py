"""Command-line interface for Guestnest."""

from pathlib import Path

import numpy as np
import typer

from guestnest.core import run

app = typer.Typer()


@app.command()
def main(
    host: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        resolve_path=True, help="Structure file containing the host molecule.",
    ),
    guest: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        resolve_path=True, help="Structure file containing the guest molecule.",
    ),
    output: Path = typer.Option(
        Path("host_guest_complex.sdf"), "--output", "-o", file_okay=True,
        dir_okay=False, writable=True, resolve_path=True,
        help="SDF or XYZ file to write.",
    ),
    n_complexes: int = typer.Option(
        1, "--n-complexes", "-n", min=1,
        help="Number of initial host–guest poses to generate.",
    ),
    cavity_dimensions: tuple[float, float, float] = typer.Option(
        (4.0, 4.0, 4.0), "--cavity-dimensions", "-d",
        help="Ellipsoidal cavity semi-axes in angstroms: X Y Z.",
    ),
    theta_range: tuple[float, float] = typer.Option(
        (0.0, np.pi), "--theta-range",
        help="Zenith-angle limits in radians.",
    ),
    phi_range: tuple[float, float] = typer.Option(
        (0.0, 2.0 * np.pi), "--phi-range",
        help="Azimuthal-angle limits in radians.",
    ),
    vdw_scaling: float = typer.Option(
        1.0, "--vdw-scaling", "-s", min=0.0,
        help="Scaling factor for van der Waals radii.",
    ),
    rmsd_threshold: float = typer.Option(
        0.1, "--rmsd-threshold", "-t", min=0.0,
        help="RMSD deduplication threshold in angstroms.",
    ),
    random_seed: int | None = typer.Option(
        None, "--random-seed", "-r",
        help="Random seed used for pose generation.",
    ),
) -> None:
    """Generate and optimize candidate host–guest complexes."""

    run(
        host_f=host,
        guest_f=guest,
        output_f=output,
        n_complexes=n_complexes,
        host_cavity_dims=cavity_dimensions,
        theta_range=theta_range,
        phi_range=phi_range,
        vdw_scaling=vdw_scaling,
        rmsd_threshold=rmsd_threshold,
        random_seed=random_seed,
    )
