#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Optional, cast
from urllib.parse import urlparse

import click
from click_option_group import GroupedOption, OptionGroup
from click_option_group._core import _GroupTitleFakeOption
from click_option_group._helpers import get_fake_option_name, resolve_wrappers

from revng.pypeline.cli.context import ClickContext
from revng.pypeline.container import ContainerDeclaration
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.object import Kind, ObjectID, ObjectSet
from revng.pypeline.storage.storage_provider import StorageProviderFactory
from revng.pypeline.task.task import TaskArgument
from revng.pypeline.utils.registry import get_registry, get_singleton


class PypeCommand(click.Command):
    """
    An extension of click.Command that modifies the usage line to document that the arguments after
    "--" are passed to the pipebox initialize function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.help is None:
            self.help = ""
        self.help += (
            "\n\nNote: All arguments after '--' are passed to the pipebox initialize function."
        )

    def collect_usage_pieces(self, ctx) -> list[str]:
        return super().collect_usage_pieces(ctx) + ["--", "[PIPEBOX ARGS...]"]


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
    using the name of the argument, like `ctx.obj.pipebox`.
    Additionally, it stores the path used to parse the value as `ctx.obj.<self.name + "_path">`.
    """

    # Due to the way click works, the `convert` function below is called with
    # `value` set to the default parameter if the command-line option is not
    # used. This is problematic because since this type is generic it cannot
    # differentiate between a value passed from the command-line and a value
    # set as a default. To avoid this problem, set the click default to this
    # singleton value and pass the actual default value to the type constructor.
    DEFAULT = object()

    def __init__(
        self,
        name: str,
        parser: Callable[[str, "ClickContext"], Any],
        default: Any = None,
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
        self.default = default

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Any:
        if value is self.__class__.DEFAULT:
            return self.default

        path = super().convert(value, param, ctx)
        if not isinstance(path, str):
            raise ValueError(f"Invalid path: {path!r}")
        if ctx is None:
            raise ValueError("Context is required for parsing")
        # If the value is a string, we parse it using the provided parser
        res = self.parser(path, cast("ClickContext", ctx))
        # Store the parsed value in the context object
        setattr(ctx.obj, self.name, res)
        # And also store the path to the parsed value
        setattr(ctx.obj, self.name + "_path", Path(path).resolve())
        return res


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
            self.fail(f'"{value}" could not be parsed as a URL.', param, ctx)

        # Get all the registered providers
        allowed_schemes = {
            factory.scheme() for factory in get_registry(StorageProviderFactory).values()
        }
        # Check that the scheme is supported
        if parsed_url.scheme not in allowed_schemes:
            allowed_str = ", ".join(sorted(allowed_schemes))
            self.fail(
                f'URL scheme "{parsed_url.scheme}" is not supported. '
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


def normalize_kwarg_name(name: str) -> str:
    """
    Normalize the provided name to the convention used by click on naming
    command handler variable arguments.
    """
    return name.replace("-", "_").lower()


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
    extra_args: list[str] | None = None,
    model_help: bool = True,
) -> str:
    """
    Build a standardized help text for a command.
    """
    help_text: str = prologue
    if args or model_help or extra_args is not None:
        help_text += "\n\n\b\nArguments:"
    if model_help:
        help_text += "\n - MODEL : Path - The path to the model file to use."

    if extra_args is not None:
        for extra_arg in extra_args:
            help_text += f"\n - {extra_arg}"

    for arg in args:
        help_text += _build_help_line(arg)

    help_text += epilogue
    return help_text


def _build_help_line(arg: TaskArgument) -> str:
    arg_name = normalize_pos_arg_name(arg.name)
    line = f"\n - {arg_name} : "
    line += f"{arg.container_type.__name__} - "
    line += normalize_whitespace(arg.help_text)
    return line.rstrip()


def list_objects_for_container(
    model: ReadOnlyModel,
    arg_name: str,
    kind: Kind,
):
    """
    Print all available objects for a given container kind in the model.
    """
    print(f'Available objects for "{arg_name}" kind: "{kind.serialize()}"')
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
    obj_id_type = get_singleton(ObjectID)  # type: ignore[type-abstract]
    if f"{arg_name}_objects" in kwargs:
        objects = kwargs.get(f"{arg_name}_objects", "")
        if objects:
            return ObjectSet(
                kind=kind,
                objects={obj_id_type.deserialize(obj) for obj in objects.split(",") if obj.strip()},
            )
    return model.all_objects(kind)


def get_root_command_name(ctx: click.Context) -> str:
    root = ctx.find_root()
    command = root.command
    assert command.name is not None, "Command name should not be None"
    return command.name


def detect_autocomplete(ctx: click.Context | None) -> bool:
    """Detect if we are in auto-complete mode."""
    if ctx is not None:
        command_name = get_root_command_name(ctx)
    else:
        command_name = os.path.basename(sys.argv[0])

    return f"_{command_name.upper()}_COMPLETE" in os.environ or "autocomplete" in sys.argv


def add_group_option_fake_title(command: click.Command, group: OptionGroup):
    """
    This functions adds the fake title command-line "option" to allow printing
    the option group title when outputting the help page. This is equivalent of
    what the `@optgroup.group` decorator does but can be applied to an
    already-created command.
    """
    command.params.append(_GroupTitleFakeOption((get_fake_option_name(),), group=group))


def create_option(group: OptionGroup, command: click.Command, *args, **kwargs):
    """Create a GroupedOption and add it to the specified option group"""
    option = GroupedOption(*args, **kwargs, group=group)
    assert command.callback is not None
    group._options[resolve_wrappers(command.callback)][option.name] = option
    return option


def sort_option_groups(command: click.Command):
    """
    When using the click-option-group decorators, the code assumes the decorator
    ordering (bottom-to-top). This is reversed once click constructs the
    `Command` object. To make the output nice we read `command.params` and
    re-order them so that they are in this order:
    1. Positional arguments
    2. Option group tile
    3. Option group options
    4. Repeat 2-3 for each option group
    5. Ungrouped options
    """

    # Positional arguments
    arguments: list[click.Argument] = []
    # Grouped options
    option_groups: dict[OptionGroup, list[click.Option]] = defaultdict(list)
    # Ungrouped options
    options: list[click.Option] = []

    for param in command.params:
        if isinstance(param, click.Argument):
            arguments.append(param)
        elif isinstance(param, _GroupTitleFakeOption):
            # click-option-group does not expose the `__group` attribute, so
            # we need to retrieve it manually
            group = getattr(param, f"{param.__class__.__name__}__group")
            option_groups[group].insert(0, param)
        elif isinstance(param, GroupedOption):
            option_groups[param.group].append(param)
        else:
            assert isinstance(param, click.Option)
            assert not isinstance(param, GroupedOption)
            options.append(param)

    # Construct the new `params` list
    new_params: list[click.Parameter] = cast(list[click.Parameter], arguments.copy())
    for group, group_options in option_groups.items():
        new_params.extend(group_options)
    new_params.extend(options)

    # Assign the re-ordered list to the command
    command.params = new_params
