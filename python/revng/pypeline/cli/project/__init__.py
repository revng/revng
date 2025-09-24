#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from pathlib import Path

import click
from xdg import xdg_cache_home

from revng.pypeline.cli.utils import EagerParsedPath, StorageProviderUrl
from revng.pypeline.pipeline_parser import load_pipeline_yaml_file

from .analyze import analyze
from .artifact import artifact
from .daemon import run_daemon


@click.group(
    help="Project commands (porcelain)",
)
@click.option(
    "--pipeline",
    "pipeline",
    type=EagerParsedPath(
        name="pipeline",
        parser=lambda path, _ctx: load_pipeline_yaml_file(path),
    ),
    help='Path to the pipeline file. Defaults to the "PIPELINE" environment if set',
    default=os.environ.get("PIPELINE", "pipeline.yml"),
    show_default=True,
)
@click.option(
    "--storage-provider",
    "storage_provider",
    type=StorageProviderUrl(),
    help=("The URL of the storage provider to use."),
    # TODO: check env var name
    default=os.environ.get("STORAGE_PROVIDER", "local://"),
    show_default=True,
)
@click.option(
    "--cache-dir",
    "cache_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True),
    help=("The directory to use for caching."),
    default=str(Path(os.environ.get("CACHE_DIR", xdg_cache_home() / "pipeline"))),
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
