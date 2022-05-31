#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import sys

from .commands_registry import Options, commands_registry
from .support import collect_files


def extend_list(paths, new_items):
    return new_items + [path for path in paths if path not in new_items]


# flake8: noqa: F401
def run_revng_command(arguments, options: Options):
    # Import built-in commands
    from .daemon import DaemonCommand
    from .lift import LiftCommand
    from .llvm_pipeline import IRPipelineCommand
    from .opt import IROptCommand
    from .override_by_name import ModelOverrideByName
    from .pipeline import PipelineCommand
    from .translate import TranslateCommand

    if options.verbose:
        sys.stderr.write("{}\n\n".format(" \\\n  ".join([sys.argv[0]] + arguments)))

    if options.dry_run:
        return 0

    # Collect search prefixes
    prefixes = []
    for arg, next_arg in zip(arguments, arguments[1:] + [""]):
        if arg.startswith("--prefix="):
            prefixes += [arg[len("--prefix=") :]]
        elif arg == "--prefix" and next_arg != "":
            prefixes += [next_arg]
    options.search_prefixes = extend_list(options.search_prefixes, prefixes)

    # Register external commands
    prefix = "revng-"
    for executable in collect_files(options.search_prefixes, ["libexec", "revng"], f"{prefix}*"):
        name = os.path.basename(executable)[len(prefix) :]
        if not commands_registry.has_command(name):
            commands_registry.register_external_command(name, executable)

    return commands_registry.run(arguments, options)


def main():
    return run_revng_command(sys.argv[1:], Options(None, [], [], False, False, False))
