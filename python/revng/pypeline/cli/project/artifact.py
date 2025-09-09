#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging

import click

from revng.pypeline.cli.utils import build_arg_objects, build_help_text, compute_objects
from revng.pypeline.cli.utils import normalize_whitespace, storage_provider_factory
from revng.pypeline.container import dump_container
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import ObjectSet
from revng.pypeline.pipeline import Artifact, Pipeline
from revng.pypeline.task.task import TaskArgument, TaskArgumentAccess
from revng.pypeline.utils.registry import get_singleton

logger = logging.getLogger(__name__)


class ArtifactGroup(click.Group):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_ty: type[Model] = get_singleton(Model)  # type: ignore[type-abstract]

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        pipeline = ctx.obj.get("pipeline")
        if pipeline is None:
            return base
        return base + sorted(pipeline.artifacts.keys())

    def get_command(self, ctx, cmd_name):
        pipeline = ctx.obj.get("pipeline")
        if pipeline is None:
            return super().get_command(ctx, cmd_name)
        if cmd_name not in pipeline.artifacts:
            return super().get_command(ctx, cmd_name)
        return self._build_artifact_command(
            artifact_name=cmd_name,
            pipeline=pipeline,
        )

    def _build_artifact_command(self, artifact_name: str, pipeline: Pipeline):
        """Dynamically create a command for getting an artifact."""
        artifact: Artifact = pipeline.artifacts[artifact_name]

        if artifact.__doc__:
            help_text = click.wrap_text(f"\n{normalize_whitespace(artifact.__doc__)}")
        else:
            help_text = f"Get the artifact: {artifact_name}"

        help_text = build_help_text(
            prologue=help_text,
            args=[
                TaskArgument(
                    name=artifact.container.name,
                    container_type=artifact.container.container_type,
                    access=TaskArgumentAccess.WRITE,
                    help_text=artifact.container.container_type.__doc__ or "",
                )
            ],
        )

        # Build the actual function that will be the command
        run_artifact_command = build_artifact_command(
            artifact=artifact,
            help_text=help_text,
            model_ty=self.model_ty,
            pipeline=pipeline,
        )

        config = getattr(
            artifact,
            "configuration_help",
            f"Configuration for the artifact '{artifact_name}'.",
        )
        if config is not None:
            run_artifact_command = click.option(
                "-c",
                "--configuration",
                type=str,
                default="",
                help=normalize_whitespace(config),
            )(run_artifact_command)

        # For the only container, call the `click.argument` decorator to
        # dynamically add its `objects` argument to the command
        run_artifact_command = click.argument(
            artifact.container.name,
            type=click.Path(dir_okay=False, writable=True),
        )(run_artifact_command)
        run_artifact_command = build_arg_objects(artifact.container)(run_artifact_command)

        return run_artifact_command


def build_artifact_command(
    artifact: Artifact,
    help_text: str,
    model_ty: type[Model],
    pipeline: Pipeline,
):
    artifact_name: str = artifact.name

    @click.command(name=artifact_name, help=help_text)
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
    def run_analysis_command(
        model: str,
        configuration: str,
        **kwargs,
    ) -> None:
        logger.debug("Running artifact: `%s`", artifact_name)
        logger.debug("configuration: `%s`", configuration)
        logger.debug("model: `%s`", model)
        logger.debug("and kwargs: `%s`", kwargs)

        # Load the model
        storage_provider = storage_provider_factory(model_path=model)
        loaded_model: Model = model_ty()
        loaded_model.deserialize(storage_provider.get_model())

        logger.debug("Model loaded: `%s`", loaded_model)

        arg_name = artifact.container.name
        artifact_kind = artifact.container.container_type.kind
        if kwargs["list"]:
            # If the user requested to list the available objects, we print them
            # and exit
            print(f"Available objects for `{arg_name}` kind: {artifact_kind.__name__}")
            for obj in loaded_model.all_objects(artifact_kind):
                print(f" - {obj}")
            return

        # Compute the requests for the incoming containers of the
        # analysis
        incoming: ObjectSet = loaded_model.all_objects(
            artifact_kind,
        )
        # If the argument is writable, we need to request the objects
        incoming = compute_objects(
            model=ReadOnlyModel(loaded_model),
            arg_name=arg_name,
            kind=artifact_kind,
            kwargs=kwargs,
        )

        # Finally, run the analysis
        res_container = pipeline.get_artifact(
            model=ReadOnlyModel(loaded_model),
            artifact=artifact,
            requests=incoming,
            pipeline_configuration={},
            storage_provider=storage_provider,
        )
        logger.debug("Artifact computed")

        res_path = kwargs[arg_name]
        logger.debug("Writing result to: `%s`", res_path)
        dump_container(
            res_container,
            res_path,
        )

    return run_analysis_command


@click.group(
    cls=ArtifactGroup,
    help="Compute an Artifact",
)
def get_artifact() -> None:
    pass
