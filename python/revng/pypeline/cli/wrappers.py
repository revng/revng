#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import functools
import inspect
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import click
from click_option_group import OptionGroup

from revng.pypeline.cli.context import ClickContext

from .utils import PypeCommand, add_group_option_fake_title, create_option, sort_option_groups


@dataclass
class Wrapper:
    param: click.Parameter
    prefix: list[str]


class WrapperOption:
    class WrapperType(click.ParamType):
        def __init__(
            self, *args, name: str, prefix_generator: Callable[[Any], list[str]], **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.name = name
            self.prefix_generator = prefix_generator

        def convert(self, value, param, ctx: ClickContext):  # type: ignore
            if not value:
                return

            if ctx.obj.wrapper is not None:
                wrapper: Wrapper = ctx.obj.wrapper
                raise click.UsageError(
                    f"option {param.get_error_hint(ctx)} is incompatible with "
                    f"{wrapper.param.get_error_hint(ctx)}, use one or the other",
                    ctx,
                )

            ctx.obj.wrapper = Wrapper(param, self.prefix_generator(value))

    def __init__(
        self,
        name: str,
        help: str,  # noqa: A002
        prefix: list[str] | None = None,
        type_: type = bool,
    ):
        self.name = name
        self.help = help
        self.prefix = prefix
        self.type_ = type_

    def generate_prefix(self, value: Any) -> list[str]:
        assert self.prefix is not None
        return self.prefix

    def make_option(self, group: OptionGroup, command: click.Command):
        assert self.type_ in (bool, str)
        return create_option(
            group,
            command,
            (f"--{self.name}",),
            type=self.__class__.WrapperType(name=self.name, prefix_generator=self.generate_prefix),
            help=self.help,
            expose_value=False,
            is_flag=self.type_ is bool,
        )


class WrapperRegistry:
    def __init__(self):
        self.wrappers: list[WrapperOption] = []
        self.commands: list[click.Command] = []
        self.group = OptionGroup(
            "Wrappers", help="Run program(s) wrapped inside one of the specified wrappers"
        )

    def register_command(self, command: click.Command):
        add_group_option_fake_title(command, self.group)
        self.commands.append(command)
        for wrapper in self.wrappers:
            command.params.append(wrapper.make_option(self.group, command))
        sort_option_groups(command)

    def register_wrapper(self, wrapper: WrapperOption):
        self.wrappers.append(wrapper)
        for command in self.commands:
            command.params.append(wrapper.make_option(self.group, command))
        sort_option_groups(command)

    def register_wrappers(self, *wrappers: WrapperOption):
        for wrapper in wrappers:
            self.register_wrapper(wrapper)


WRAPPER_REGISTRY = WrapperRegistry()


class WrappableCommand(click.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        WRAPPER_REGISTRY.register_command(self)


class WrappablePypeCommand(WrappableCommand, PypeCommand):
    pass


def exec_wrapper_if_needed(obj):
    if isinstance(obj, click.Command):
        target_function = obj.callback
        is_command = True
    else:
        assert inspect.isfunction(obj)
        target_function = obj
        is_command = False

    @functools.wraps(target_function)
    def wrapper(*args, **kwargs):
        ctx = click.get_current_context()
        wrapper: Wrapper | None = ctx.obj.wrapper
        # If there is no wrapper or we're already wrapped call the function normally
        if wrapper is None or os.environ.get("_PYPE_WRAPPER") == "1":
            return target_function(*args, **kwargs)

        env = {**os.environ, "_PYPE_WRAPPER": "1"}
        os.execvpe(wrapper.prefix[0], [*wrapper.prefix, sys.executable, *sys.argv], env)
        return None

    if is_command:
        obj.callback = wrapper
        return obj
    else:
        return wrapper


def exec_with_wrapper(args: list[str], env: Mapping[str, str] | None = None):
    if env is None:
        env = os.environ
    ctx = click.get_current_context()
    wrapper: Wrapper | None = ctx.obj.wrapper
    if wrapper is not None:
        args = [*wrapper.prefix, *args]
    os.execvpe(args[0], args, env)
