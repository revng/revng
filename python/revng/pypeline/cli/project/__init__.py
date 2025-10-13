#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os

import click

from revng.pypeline.cli.utils import EagerParsedPath, LazyGroup
from revng.pypeline.pipeline_parser import load_pipeline_yaml_file


@click.group(
    cls=LazyGroup,
    lazy_subcommands={
        "analyze": "revng.pypeline.cli.project.analyze:analyze",
        "artifact": "revng.pypeline.cli.project.artifact:get_artifact",
    },
    help="Project commands (porcelain)",
)
@click.option(
    "--pipeline",
    type=EagerParsedPath(
        name="pipeline",
        parser=load_pipeline_yaml_file,
    ),
    help=(
        'Path to the pipeline file. Defaults to the "PIPELINE" environment '
        'variable, then "pipeline.yml".'
    ),
    default=os.environ.get("PIPELINE", "pipeline.yml"),
    expose_value=False,
)
def project() -> None:
    pass
