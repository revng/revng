#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

"""
This is just a wrapper over `pype` that sets pipebox to the revng pipebox path.
The path is computed relatively to this file, so this should work regardless of
where revng is installed.
"""

import asyncio
import os
from pathlib import Path
from typing import AsyncContextManager

import click
import yaml

from revng.internal.support import cache_directory
from revng.pypeline.cli.project import project
from revng.pypeline.cli.utils import PypeGroup, project_id_option, token_option
from revng.pypeline.main import pype, run
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.storage.storage_provider import StorageProvider
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.pypeline.storage.util import compute_hash
from revng.pypeline.utils.registry import get_singleton


def generate_model_with_binaries(binaries: list[Path]):
    result = []
    for index, binary in enumerate(binaries):
        with open(binary, "rb") as f:
            hash_ = compute_hash(f)
            size = f.seek(0, os.SEEK_END)
        result.append({"Index": index, "Hash": hash_, "Size": size, "Name": binary.name})

    return {"Binaries": result}


@click.group(cls=PypeGroup)
def quick():
    """Quick commands (japanese toilet)"""
    # TODO


@click.command()
@click.argument(
    "binary",
    required=False,
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--no-initial-auto-analysis", is_flag=True)
@project_id_option
@token_option
@click.pass_context
def init(ctx, binary: Path | None, no_initial_auto_analysis: bool, project_id: str, token: str):
    """Initialize a new project."""
    model_type = get_singleton(Model)  # type: ignore[type-abstract]
    model_name = model_type.model_name()
    model_file = Path.cwd() / model_name
    if model_file.exists():
        raise click.UsageError(
            f"File {model_name} is already present in the current directory. "
            "Refusing to overwrite it."
        )
    model_file.touch()

    if binary is not None:
        model_raw = yaml.safe_dump(generate_model_with_binaries([binary])).encode()
        with open(model_file, "wb") as f:
            f.write(model_raw)
    else:
        model_raw = b""

    if no_initial_auto_analysis:
        return

    async def async_part_of_command(
        storage_provider_context: AsyncContextManager[StorageProvider],
    ):
        pipeline = ctx.obj["pipeline"]
        async with storage_provider_context as storage_provider:
            model = model_type.deserialize(model_raw)
            analysis_list = pipeline.analysis_lists["initial-auto-analysis"]
            analysis_configuration = ["" for _ in analysis_list.analyses]
            pipeline.run_analysis_list(
                model=ReadOnlyModel(model),
                analysis_list=analysis_list,
                analysis_configuration=analysis_configuration,
                pipeline_configuration={},
                storage_provider=storage_provider,
            )

    storage_provider_factory = storage_provider_factory_factory(ctx.obj["storage_provider"])
    storage_provider_context = storage_provider_factory.get(
        project_id=project_id,
        token=token,
        cache_dir=ctx.obj["cache_dir"],
    )
    asyncio.run(async_part_of_command(storage_provider_context))


def patch_pype():
    """
    revng2 is based on `pype`, but we want to change some defaults to be revng specific,
    and we want to add some commands.
    """
    # Replace the name (needed for autocompletion and usage)
    pype.name = "revng2"
    pype.add_command(quick)
    # Replace the default for pipebox
    for param in pype.params:
        if param.name == "pipebox":
            param.default = os.environ.get("PIPEBOX", Path(__file__).parent.parent / "pipebox.py")

    # Add `init` to project subcommand
    project.add_command(init)
    # Change the default for pipeline
    for param in project.params:
        if param.name == "pipeline":
            param.default = os.environ.get(
                "PIPELINE", Path(__file__).parent.parent / "pipeline.yml"
            )
        elif param.name == "cache_dir":
            param.default = cache_directory()


def main():
    """Entry point for revng2."""
    patch_pype()
    run()


if __name__ == "__main__":
    main()
