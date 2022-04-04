#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys
import os

from .commands_registry import commands_registry, Options
from .support import collect_files


def extend_list(list, new_items):
    new_items_set = set(new_items)
    return new_items + [path for path in list if path not in new_items]


def run_revng_command(arguments, options: Options):
    # Import built-in commands
    from .lift import LiftCommand
    from .translate import TranslateCommand
    from .opt import IROptCommand
    from .pipeline import PipelineCommand
    from .llvm_pipeline import IRPipelineCommand

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
    script_path = os.path.dirname(os.path.realpath(__file__))
    options.search_prefixes = extend_list(
        options.search_prefixes, prefixes + [os.path.join(script_path, "..", "..", "..", "..")]
    )

    # Register external commands
    prefix = "revng-"
    for executable in collect_files(options.search_prefixes, ["libexec", "revng"], f"{prefix}*"):
        name = os.path.basename(executable)[len(prefix) :]
        if not commands_registry.has_command(name):
            commands_registry.register_external_command(name, executable)

    return commands_registry.run(arguments, options)


def main():
    return run_revng_command(sys.argv[1:], Options(None, [], [], [], False, False, False))
