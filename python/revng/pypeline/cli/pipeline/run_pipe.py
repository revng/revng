#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
from pathlib import Path

import click

from revng.pypeline.cli.utils import build_arg_objects, build_help_text, compute_objects
from revng.pypeline.cli.utils import normalize_kwarg_name, normalize_whitespace
from revng.pypeline.container import dump_container, load_container
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import ObjectSet
from revng.pypeline.storage.file_provider import FileProvider, FileRequest
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.task.task import TaskArgumentAccess
from revng.pypeline.utils.registry import get_registry, get_singleton

logger = logging.getLogger(__name__)


# A file storage implementation that works with a provided directory. Will look
# in the directory for a file named as the requested hash.
class SimpleFileProvider(FileProvider):
    def __init__(self, directory: Path):
        self._directory = directory

    def get_files(self, requests: list[FileRequest]) -> dict[str, bytes]:
        if len(requests) == 0:
            return {}

        for request in requests:
            if not (self._directory / request.hash).is_file():
                raise ValueError(
                    f"File {request.hash} not found, please specify a directory"
                    " with all input files via the --file-storage option"
                )

        return {r.hash: (self._directory / r.hash).read_bytes() for r in requests}


class RunPipeGroup(click.Group):
    """We need to create a custom command for each pipe we loaded from the registry.
    Since we already have to generate the code dynamically, we do it lazily so
    we generate only the commands that are requested.
    This is based on the LazyGroup from revng.pypeline.cli.utils."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry: dict[str, type[Pipe]] = get_registry(Pipe)
        self.model_ty: type[Model] = get_singleton(Model)  # type: ignore[type-abstract]

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        return base + sorted(self.registry.keys())

    def get_command(self, ctx, cmd_name):
        if cmd_name in self.registry:
            return self._build_pipe_command(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _build_pipe_command(self, pipe_name: str):
        """Dynamically create a command for running a pipe."""
        pipe_ty: type[Pipe] = self.registry[pipe_name]

        if pipe_ty.__doc__:
            help_text = click.wrap_text(f"\n{normalize_whitespace(pipe_ty.__doc__)}")
        else:
            help_text = f"Run the pipe: {pipe_name}"

        help_text = build_help_text(
            prologue=help_text,
            args=pipe_ty.signature(),
        )

        # Add options for static configuration and configuration, only if the
        # pipe doesn't disable them by defining them as None
        static_config = (
            pipe_ty.static_configuration_help()
            or f"Static configuration for the pipe '{pipe_name}'."
        )

        # Build the actual function that will be the command
        run_pipe_command = build_pipe_command(
            pipe_name=pipe_name,
            help_text=help_text,
            pipe_ty=pipe_ty,
            model_ty=self.model_ty,
        )

        # Decorate it to add the arguments it needs
        if static_config is not None:
            run_pipe_command = click.option(
                "-s",
                "--static-configuration",
                type=str,
                default="",
                help=normalize_whitespace(static_config),
            )(run_pipe_command)
        config = getattr(
            pipe_ty, "configuration_help", f"Configuration for the pipe '{pipe_name}'."
        )
        if config is not None:
            run_pipe_command = click.option(
                "-c",
                "--configuration",
                type=str,
                default="",
                help=normalize_whitespace(config),
            )(run_pipe_command)

        # For each argument, call the `click.argument` decorator to dynamically add
        # them to the command
        for arg in pipe_ty.signature():
            if arg.access == TaskArgumentAccess.READ_WRITE:
                run_pipe_command = click.argument(
                    f"{arg.name}-input",
                    type=click.Path(exists=True, dir_okay=False, readable=True),
                )(run_pipe_command)
                run_pipe_command = click.argument(
                    f"{arg.name}-output",
                    type=click.Path(dir_okay=False, writable=True),
                )(run_pipe_command)
            elif arg.access == TaskArgumentAccess.READ:
                run_pipe_command = click.argument(
                    arg.name,
                    type=click.Path(exists=True, dir_okay=False, readable=True),
                )(run_pipe_command)
            elif arg.access == TaskArgumentAccess.WRITE:
                run_pipe_command = click.argument(
                    arg.name,
                    type=click.Path(dir_okay=False, writable=True),
                )(run_pipe_command)
            else:
                raise ValueError("Unreachable code: unknown access type %s")
            # For output arguments, we can specify which objects we want
            # to request
            if arg.access & TaskArgumentAccess.WRITE:
                run_pipe_command = build_arg_objects(arg)(run_pipe_command)

        return run_pipe_command


def build_pipe_command(
    pipe_name: str,
    help_text: str,
    pipe_ty: type[Pipe],
    model_ty: type[Model],
):
    @click.command(name=pipe_name, help=help_text)
    @click.argument(
        "model",
        type=click.Path(exists=True, dir_okay=False, readable=True),
        required=True,
    )
    @click.option(
        "--list",
        type=bool,
        is_flag=True,
        default=False,
        help="List the available objects for each argument.",
    )
    @click.option(
        "--file-storage",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=Path.cwd(),
    )
    def run_pipe_command(
        model: str,
        static_configuration: str,
        configuration: str,
        file_storage: Path,
        **kwargs,
    ) -> None:
        logger.debug("Running pipe: %s", pipe_name)
        logger.debug("with static configuration: %s", static_configuration)
        logger.debug("configuration: %s", configuration)
        logger.debug("model: %s", model)
        logger.debug("and kwargs: %s", kwargs)

        # Create the pipe
        pipe = pipe_ty(
            name=pipe_name,
            static_configuration=static_configuration,
        )
        # Load the model
        with open(model, "rb") as model_file:
            loaded_model = model_ty.deserialize(model_file.read())
        # Load the containers with args form the command line
        containers = []
        for arg in pipe.arguments:
            arg_name = normalize_kwarg_name(arg.name)
            # Write-only containers can be empty
            if arg.access == TaskArgumentAccess.WRITE:
                containers.append(arg.container_type())
                continue

            # Otherwise we need to load the container from the filesystem
            if arg.access == TaskArgumentAccess.READ_WRITE:
                path = kwargs[f"{arg_name}_input"]
            else:
                path = kwargs[arg_name]
            containers.append(
                load_container(
                    arg.container_type,
                    path,
                )
            )

        # From the command line figure out the requests for each argument
        outgoing: list[ObjectSet] = []
        for arg in pipe.signature():
            if arg.access == TaskArgumentAccess.READ:
                # If the argument is read-only, we don't need to request anything
                continue
            # If the argument is writable, we need to request the objects
            outgoing.append(
                compute_objects(
                    model=ReadOnlyModel(loaded_model),
                    arg_name=arg.name,
                    kind=arg.container_type.kind,
                    kwargs=kwargs,
                )
            )

        # Ask the pipe for the requests it needs
        incoming = pipe.prerequisites_for(
            model=ReadOnlyModel(loaded_model),
            requests=outgoing,
        )
        # Ensure that the user provided all the required arguments
        for container, request in zip(containers, incoming):
            if not request:
                # If the request is empty, we don't need to load anything
                continue
            if container.kind not in request:
                raise click.UsageError(
                    f"Container {container} does not have the required objects: {request}"
                )

        # Finally, run the pipe
        object_deps = pipe.run(
            file_provider=SimpleFileProvider(file_storage),
            model=ReadOnlyModel(loaded_model),
            containers=containers,
            incoming=incoming,
            outgoing=outgoing,
            configuration=configuration,
        )
        logger.debug("Pipe run completed, object dependencies: `%s`", object_deps)

        # Dump back the modified containers to the filesystem
        for arg, container in zip(pipe.signature(), containers):
            if arg.access == TaskArgumentAccess.READ:
                continue

            arg_name = normalize_kwarg_name(arg.name)
            # If the argument is writable, we dump the container
            # to the filesystem
            if arg.access == TaskArgumentAccess.READ_WRITE:
                path = kwargs[f"{arg_name}_output"]
            else:
                path = kwargs[arg_name]
            logger.info("Dumping container %s to %s", arg.name, path)
            dump_container(
                container,
                path,
            )

    return run_pipe_command


@click.group(
    cls=RunPipeGroup,
    help="Run a pipe",
)
def run_pipe() -> None:
    pass
