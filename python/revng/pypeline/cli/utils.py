#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import importlib
import importlib.util
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import click

from revng.pypeline.container import ContainerDeclaration
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.object import Kind, ObjectID, ObjectSet
from revng.pypeline.storage.storage_provider import StorageProviderFactory
from revng.pypeline.task.task import TaskArgument, TaskArgumentAccess
from revng.pypeline.utils.registry import get_registry, get_singleton

logger = logging.getLogger(__name__)


class RegistryChoice(click.Choice):
    """A click.Choice that uses the registry of a given type, and
    returns the actual object from the registry instead of just the string."""

    def __init__(
        self, ty: type, case_sensitive: bool = False, subclass_filter: Optional[type] = None
    ) -> None:
        self.ty = ty
        self.registry: dict[str, type] = {  # type: ignore[var-annotated]
            k: v
            for k, v in get_registry(ty).items()
            if subclass_filter is None or issubclass(v, subclass_filter)
        }
        super().__init__(
            choices=sorted(self.registry.keys()),
            case_sensitive=case_sensitive,
        )

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Any:
        res = super().convert(value, param, ctx)
        # Compared to a normal click.Choice, we need to return the actual object
        # from the registry, not just the string
        if isinstance(res, str):
            if res not in self.registry:
                raise ValueError("This should already be checked by click. ")
            return self.registry[res]
        return super().convert(value, param, ctx)


class EagerParsedPath(click.Path):
    """
    A click.Path that does eager parsing, meaning that it will call your function during parsing.
    This is useful for arguments that need to be parsed in order to provide
    useful auto-completion or validation.
    The cli and other arguments can retrieve the parsed value from the context object
    using the name of the argument, like `ctx.obj['pipebox']`.
    Additionally, it stores the path used to parse the value as `ctx.obj[self.name + "_path"]`.
    """

    def __init__(
        self,
        name: str,
        parser: Callable[[str, click.Context], Any],
        *args,
        **kwargs,
    ):
        # Sensible defaults for our use case
        kwargs.setdefault("exists", True)
        kwargs.setdefault("dir_okay", False)
        kwargs.setdefault("resolve_path", True)
        super().__init__(*args, **kwargs)
        self.parser = parser
        self.name = name

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Any:
        path = super().convert(value, param, ctx)
        if not isinstance(path, str):
            raise ValueError(f"Invalid path: {path!r}")
        # If the value is a string, we parse it using the provided parser
        if ctx is None:
            raise ValueError("Context is required for parsing")
        res = self.parser(path, ctx)
        if ctx is not None:
            if ctx.obj is None:
                ctx.obj = {}
            if self.name in ctx.obj:
                raise ValueError(
                    f"Argument `{self.name}` already set in context, "
                    "this is likely a bug in the code."
                )
            # Store the parsed value in the context object
            ctx.obj[self.name] = res
            # And also store the path to the parsed value
            ctx.obj[self.name + "_path"] = Path(path).resolve()
        return res


class LazyGroup(click.Group):
    """A click.Group that lazily loads subcommands from modules.
    This allows for a more modular command line interface where subcommands
    can be defined in separate modules and loaded only when needed.
    This is good for performance, but it's especially needed for us to load the
    pipebox before loading the subcommands, so the subcommands can assume that the
    registries are already populated with the objects defined in the pipebox.

    This implementation is a slightly modified version of the one from
    the click documentation: https://click.palletsprojects.com/en/stable/complex/
    """

    # lazy_subcommands is a map of the form:
    #
    #   {command-name} -> {module-name}:{command-object-name}
    #
    def __init__(self, *args, lazy_subcommands=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy_subcommands = lazy_subcommands or {}

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        lazy = sorted(self.lazy_subcommands.keys())
        return base + lazy

    def get_command(self, ctx, cmd_name):
        if cmd_name in self.lazy_subcommands:
            return self._lazy_load(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _lazy_load(self, cmd_name):
        # Lazily loading a command, first get the module name and attribute name
        import_path = self.lazy_subcommands[cmd_name]
        modname, cmd_object_name = import_path.rsplit(":", 1)
        # Do the import
        mod = importlib.import_module(modname)
        # Get the Command object from that module
        cmd_object = getattr(mod, cmd_object_name)
        # Check the result to make debugging easier
        if not isinstance(cmd_object, click.Command):
            raise ValueError(
                f"Lazy loading of {import_path} failed by returning " "a non-command object"
            )
        return cmd_object


class StorageProviderUrl(click.ParamType):
    """A custom type for the URL of a storage provider that is validated."""

    name = "url"

    def convert(self, value, param, ctx):
        # Ensure we are dealing with a string
        if not isinstance(value, str):
            self.fail(
                f"Expected a string, but got value of type {type(value).__name__}.", param, ctx
            )

        try:
            parsed_url = urlparse(value)
        except ValueError:
            # urlparse can raise ValueError on rare malformed inputs
            self.fail(f"'{value}' could not be parsed as a URL.", param, ctx)

        # get all the registered providers
        allowed_schemes = {
            factory.protocol().lower() for factory in get_registry(StorageProviderFactory).values()
        }
        # check that the scheme is supported
        if parsed_url.scheme.lower() not in allowed_schemes:
            allowed_str = ", ".join(sorted(allowed_schemes))
            self.fail(
                f"URL scheme '{parsed_url.scheme}' is not supported. "
                f"Allowed schemes are: {allowed_str}.",
                param,
                ctx,
            )
        # If all checks pass, return the original, validated string
        return value


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in a string by removing leading and trailing
    whitespace and replacing multiple spaces with a single space.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_flag(name: str) -> str:
    """
    Normalize a flag name by replacing spaces and underscores with
    hyphens and converting it to lowercase.
    """
    return normalize_whitespace(name).replace(" ", "-").replace("_", "-").lower()


def normalize_pos_arg_name(name: str) -> str:
    """
    Normalize a positional argument name by replacing spaces and underscores
    with hyphens and converting it to lowercase.
    This is used for positional arguments that are not flags.
    """
    return normalize_whitespace(name).replace(" ", "_").replace("-", "_").upper()


def build_arg_objects(
    container_decl: ContainerDeclaration,
) -> Callable:
    """
    A decorator that adds an argument to a click command for
    the objects that the user wants in a specific container.
    """
    arg_name = normalize_flag(container_decl.name)
    kind = container_decl.container_type.kind
    return click.option(
        f"--{arg_name}-objects",
        metavar=f"/{kind.serialize()}1,/{kind.serialize()}2,...",
        type=str,
        help=(
            f"The objects to require from container {arg_name.upper()}"
            " as a comma-separated list of IDs. If not passed, all "
            "objects will be requested."
        ),
    )


def build_help_text(
    args: Sequence[TaskArgument],
    prologue: str = "",
    epilogue: str = "",
) -> str:
    """
    Build a standardized help text for a command.
    """
    help_text = prologue
    help_text += "\n\n\b\nArguments:"
    help_text += "\n - [R] MODEL : Path - The path to the model file to use."

    for arg in args:
        help_text += _build_help_line(arg)

    help_text += epilogue
    return help_text


def _build_help_line(arg: TaskArgument) -> str:
    if arg.access != TaskArgumentAccess.READ_WRITE:
        access = "R" if arg.access == TaskArgumentAccess.READ else "W"
        arg_name = normalize_pos_arg_name(arg.name)
        line = f"\n - [{access}] {arg_name} : "
        line += f"{arg.container_type.__name__} - "
        line += normalize_whitespace(arg.help_text)
        return line.rstrip()
    # If the argument is read-write, we need to add both input and output
    return _build_help_line(
        TaskArgument(
            name=f"{arg.name}-input",
            container_type=arg.container_type,
            help_text=arg.help_text,
            access=TaskArgumentAccess.READ,
        )
    ) + _build_help_line(
        TaskArgument(
            name=f"{arg.name}-output",
            container_type=arg.container_type,
            help_text="Like above, but for output.",
            access=TaskArgumentAccess.WRITE,
        )
    )


def list_objects_for_container(
    model: ReadOnlyModel,
    arg_name: str,
    kind: Kind,
):
    """
    Print all available objects for a given container kind in the model.
    """
    print(f"Available objects for `{arg_name}` kind: {kind.serialize()}")
    for obj in model.all_objects(kind):
        print(f" - {obj}")


def compute_objects(
    model: ReadOnlyModel,
    arg_name: str,
    kind: Kind,
    kwargs: dict[str, str],
) -> ObjectSet:
    """
    Check if the user provided a list of objects for the given
    argument name, and if so, return an ObjectSet with those objects
    deserialized.
    Otherwise, return all objects of the given kind from the model.
    """
    arg_name = normalize_flag(arg_name)
    obj_id_ty = get_singleton(ObjectID)  # type: ignore[type-abstract]
    if f"{arg_name}_objects" in kwargs:
        objects = kwargs.get(f"{arg_name}_objects", "")
        if objects:
            return ObjectSet(
                kind=kind,
                objects={obj_id_ty.deserialize(obj) for obj in objects.split(",") if obj.strip()},
            )
    return model.all_objects(kind)


# Options that are common to multiple commands
project_id_option = click.option(
    "--project-id",
    type=str,
    help=("Project id to use for the storage provider."),
    # WIP: discuss env var names
    default=os.environ.get("PROJECT_ID"),
    show_default=True,
)

list_objects_option = click.option(
    "--list",
    type=bool,
    is_flag=True,
    default=False,
    help="List the available objects for each argument.",
)

token_option = click.option(
    "--token",
    type=str,
    required=False,
    help="The token to pass to the storage provider.",
)
