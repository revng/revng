#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

from tempfile import NamedTemporaryFile

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.revng import run_revng_command
from revng.internal.cli.support import log_error


class LiftCommand(Command):
    def __init__(self):
        super().__init__(("lift",), "revng lift wrapper")

    def register_arguments(self, parser):
        parser.add_argument("input", type=str, nargs=1, help="Input binary to be lifted")
        parser.add_argument("output", type=str, nargs=1, help="Output module path")
        parser.add_argument("--record-asm", action="store_true")
        parser.add_argument("--record-ptc", action="store_true")
        parser.add_argument("--external", action="store_true")
        parser.add_argument("--base", type=str)
        parser.add_argument("--entry", type=str)
        parser.add_argument("--debug-info", type=str)
        parser.add_argument("--import-debug-info", type=str, action="append", default=[])

    def run(self, options: Options):
        if options.remaining_args:
            log_error("Unknown arguments passed in")
            return 1

        args = options.parsed_args

        arg_or_empty = (
            lambda args, name: [f"--{name}={args.__dict__[name]}"] if args.__dict__[name] else []
        )

        with NamedTemporaryFile(
            prefix="revng-lift-", suffix=".yml", delete=not options.keep_temporaries
        ) as model, NamedTemporaryFile(
            prefix="revng-lift-", suffix=".ll", delete=not options.keep_temporaries
        ) as model_in_module:
            # Run revng model import args.input
            run_revng_command(
                [
                    "model",
                    "import",
                    "binary",
                    args.input[0],
                    "-o",
                    model.name,
                ]
                + [f"--import-debug-info={value}" for value in args.import_debug_info]
                + arg_or_empty(args, "base")
                + arg_or_empty(args, "entry"),
                options,
            )

            run_revng_command(
                [
                    "model",
                    "opt",
                    "--add-primitive-types",
                    model.name,
                    "-o",
                    model.name,
                ],
                options,
            )

            run_revng_command(
                [
                    "model",
                    "inject",
                    model.name,
                    "/dev/null",
                    "-o",
                    model_in_module.name,
                ],
                options,
            )

            run_revng_command(
                [
                    "opt",
                    model_in_module.name,
                    "-o",
                    args.output[0],
                    "-load-binary",
                    f"-binary-path={args.input[0]}",
                    "-lift",
                ]
                + arg_or_empty(args, "external")
                + arg_or_empty(args, "record_asm")
                + arg_or_empty(args, "record_ptc"),
                options,
            )
        return 0
        # TODO: annotate IR


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(LiftCommand())
