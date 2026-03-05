#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from pathlib import Path

import click

from revng.pypeline.cli.utils import EagerParsedPath, PypeGroup, StorageProviderUrl
from revng.pypeline.pipeline_parser import load_pipeline_yaml_file
from revng.pypeline.utils import cache_directory

from .analyze import analyze
from .artifact import artifact
from .daemon import run_daemon


@click.group(
    cls=PypeGroup,
    help="Project commands (porcelain)",
)
@click.option(
    "--pipeline",
    "pipeline",
    type=EagerParsedPath(
        name="pipeline",
        parser=lambda path, _ctx: load_pipeline_yaml_file(path),
    ),
    help='Path to the pipeline file. Defaults to the "PYPELINE_PIPELINE" environment if set',
    default="pipeline.yml",
    envvar="PYPELINE_PIPELINE",
    show_default=True,
)
@click.option(
    "--storage-provider",
    "storage_provider",
    type=StorageProviderUrl(),
    help=("The URL of the storage provider to use."),
    default="local://",
    envvar="PYPELINE_STORAGE_PROVIDER",
    show_default=True,
)
@click.option(
    "--cache-dir",
    "cache_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True),
    help=("The directory to use for caching."),
    default=str(cache_directory()),
    show_default=True,
)
@click.pass_context
def project(
    ctx: click.Context,
    pipeline: Path,
    storage_provider: StorageProviderUrl,
    cache_dir: Path,
) -> None:
    os.makedirs(ctx.params["cache_dir"], exist_ok=True)
    if ctx.obj is None:
        ctx.obj = {}
    # Store the params so the subcommands can access them
    ctx.obj.update(
        {
            "cache_dir": cache_dir,
            "storage_provider": storage_provider,
            "pipeline": pipeline,
        }
    )


project.add_command(analyze)
project.add_command(artifact)
project.add_command(run_daemon, name="daemon")
