#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import click

from revng.internal.cli.common import ClickContext, CommandRegistry, WrappableCommand, pass_context
from revng.internal.cli.support import build_command_with_loads


@click.command(
    cls=WrappableCommand,
    name="opt",
    help="LLVM's opt with rev.ng passes",
    add_help_option=False,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("arguments", metavar="[ARGS]...", nargs=-1, type=click.UNPROCESSED)
@pass_context
def opt(ctx: ClickContext, arguments: tuple[str, ...]) -> int:
    args = list(arguments)
    if not any(arg.startswith("-enable-new-pm") for arg in args):
        args = ["-enable-new-pm=0", *args]
    args.append("--emit-hex-constant-literals-from=4096")
    return ctx.obj.run(build_command_with_loads("opt", args))


def setup(registry: CommandRegistry):
    registry.register((), opt)
