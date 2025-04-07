#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import sys
from importlib import import_module
from inspect import isfunction
from pathlib import Path

from revng.internal.support.collect import collect_files_recursive
from revng.support import get_root, read_lines

from .commands_registry import CommandsRegistry, Options
from .support import executable_name, is_file_executable

_commands_registry = CommandsRegistry()
_commands_registry.define_namespace(
    ("model",), f"Model manipulation helpers, see {executable_name()} model --help"
)
_commands_registry.define_namespace(("tar",), "Manipulate tar archives")
_commands_registry.define_namespace(
    ("model", "import"), f"Model import helpers, see {executable_name()} model import --help"
)
_commands_registry.define_namespace(
    ("trace",), f"Trace-related tools, see {executable_name()} trace --help"
)
_commands_registry.define_namespace(
    ("mass-testing",), f"Mass-testing CLI tools, see {executable_name()} mass-testing --help"
)


def extend_list(paths, new_items):
    return new_items + [path for path in paths if path not in new_items]


def revng_driver_init(arguments) -> Options:
    options = Options(None, [], [], False, False, False, [])
    modules = []
    with os.scandir(Path(__file__).parent / "_commands") as scan:
        for entry in scan:
            entry_path = Path(entry.path)
            if entry_path.name.startswith("__") or entry_path.name.startswith("."):
                continue
            if entry.is_file():
                modules.append(import_module(f"._commands.{entry_path.stem}", __package__))
            elif entry.is_dir():
                modules.append(import_module(f"._commands.{entry_path.name}", __package__))

    for module in modules:
        setup = getattr(module, "setup", None)
        if setup is not None and isfunction(setup):
            setup(_commands_registry)

    # Add forced search prefixes
    options.search_prefixes = extend_list(
        options.search_prefixes, read_lines(get_root() / "additional-search-prefixes")
    )

    # Add root as search prefix
    options.search_prefixes = extend_list(options.search_prefixes, [str(get_root())])

    # Collect search prefixes
    prefixes = []
    for arg, next_arg in zip(arguments, arguments[1:] + [""]):
        if arg.startswith("--prefix="):
            prefixes += [arg[len("--prefix=") :]]
        elif arg == "--prefix" and next_arg != "":
            prefixes += [next_arg]
    options.search_prefixes = extend_list(options.search_prefixes, prefixes)

    # Register external commands
    for executable, path in collect_files_recursive(
        options.search_prefixes, ["libexec", "revng"], "*"
    ):
        if not (is_file_executable(path) and os.path.splitext(path)[1] == ""):
            continue

        if not _commands_registry.has_command(executable):
            _commands_registry.register_external_command(executable, path)

    _commands_registry.post_init()
    return options


def run_revng_command(arguments, options: Options):
    if options.verbose:
        sys.stderr.write("{}\n\n".format(" \\\n  ".join([sys.argv[0]] + arguments)))

    if options.dry_run:
        return 0

    return _commands_registry.run(arguments, options)


def main():
    arguments = sys.argv[1:]
    options = revng_driver_init(arguments)
    return run_revng_command(arguments, options)
