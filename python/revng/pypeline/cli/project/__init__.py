#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os

import click

from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.cli.utils import EagerParsedPath, StorageProviderUrl
from revng.pypeline.pipeline import Pipeline
from revng.pypeline.pipeline_parser import load_pipeline_yaml_file
from revng.pypeline.utils import cache_directory

from .analyze import analyze
from .artifact import artifact
from .daemon import run_daemon
from .dump_pipeline import dump_pipeline


@click.group(help="Project commands (porcelain)")
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
    help="The URL of the storage provider to use.",
    default="local://",
    envvar="PYPELINE_STORAGE_PROVIDER",
    show_default=True,
)
@click.option(
    "--cache-dir",
    "cache_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True),
    help="The directory to use for caching.",
    default=str(cache_directory()),
    show_default=True,
)
@pass_context
def project(
    ctx: ClickContext,
    pipeline: Pipeline,
    storage_provider: str,
    cache_dir: str,
) -> None:
    os.makedirs(ctx.params["cache_dir"], exist_ok=True)
    # Store the params so the subcommands can access them
    ctx.obj.cache_dir = cache_dir
    ctx.obj.storage_provider_url = storage_provider
    ctx.obj.pipeline = pipeline


project.add_command(analyze)
project.add_command(artifact)
project.add_command(run_daemon, name="daemon")
project.add_command(dump_pipeline, name="dump-pipeline")
